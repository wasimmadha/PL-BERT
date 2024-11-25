import os
import shutil
import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.utils import LoggerType

from transformers import AdamW
from transformers import AlbertConfig, AlbertModel
from accelerate import DistributedDataParallelKwargs

from model import MultiTaskModel
from mydataloader import build_dataloader
from utils import length_to_mask, scan_checkpoint

from datasets import load_from_disk

from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset, concatenate_datasets
import wandb

import yaml
import pickle

config_path = "/home/PL-BERT/multilingual-pl-bert/config.yml" # you can change it to anything else
config = yaml.safe_load(open(config_path))

with open(config['dataset_params']['token_maps'], 'rb') as handle:
    token_maps = pickle.load(handle)

print(len(token_maps))

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained(config['dataset_params']['tokenizer'])

criterion = nn.CrossEntropyLoss() # F0 loss (regression)

best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
loss_train_record = list([])
loss_test_record = list([])

num_steps = config['num_steps']
log_interval = config['log_interval']
save_interval = config['save_interval']


def build_dataloaders(dataset, batch_size, num_workers, dataset_config):
    # Split dataset into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(len(train_dataset), len(val_dataset))
    train_loader = build_dataloader(train_dataset, 
                                  batch_size=batch_size, 
                                  num_workers=num_workers, 
                                  dataset_config=dataset_config)
    
    val_loader = build_dataloader(val_dataset, 
                                batch_size=batch_size, 
                                num_workers=num_workers, 
                                dataset_config=dataset_config)
    
    return train_loader, val_loader

def validate(model, val_loader, criterion, device, accelerator):
    model.eval()
    total_val_loss = 0
    total_val_vocab_loss = 0
    total_val_token_loss = 0
    val_steps = 0
    
    with torch.no_grad():
        for batch in val_loader:
            words, labels, phonemes, input_lengths, masked_indices = [x.to(device) if torch.is_tensor(x) else x for x in batch]
            text_mask = length_to_mask(torch.Tensor(input_lengths)).to(device)
            
            tokens_pred, words_pred = model(phonemes, attention_mask=(~text_mask).int())
            
            # Calculate vocab loss
            loss_vocab = 0
            for _s2s_pred, _text_input, _text_length, _masked_indices in zip(words_pred, words, input_lengths, masked_indices):
                loss_vocab += criterion(_s2s_pred[:_text_length], 
                                     _text_input[:_text_length])
            loss_vocab /= words.size(0)
            
            # Calculate token loss
            loss_token = 0
            sizes = 1
            for _s2s_pred, _text_input, _text_length, _masked_indices in zip(tokens_pred, labels, input_lengths, masked_indices):
                if len(_masked_indices) > 0:
                    _text_input = _text_input[:_text_length][_masked_indices]
                    loss_tmp = criterion(_s2s_pred[:_text_length][_masked_indices], 
                                      _text_input[:_text_length])
                    loss_token += loss_tmp
                    sizes += 1
            loss_token /= sizes
            
            loss = loss_vocab + loss_token
            
            total_val_loss += loss
            total_val_vocab_loss += loss_vocab
            total_val_token_loss += loss_token
            val_steps += 1
    
    avg_val_loss = total_val_loss / val_steps
    avg_val_vocab_loss = total_val_vocab_loss / val_steps
    avg_val_token_loss = total_val_token_loss / val_steps
    
    model.train()
    return avg_val_loss, avg_val_vocab_loss, avg_val_token_loss



# for _, batch in enumerate(train_loader):   
#     words, labels, phonemes, input_lengths, masked_indices = batch

import time

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    curr_steps = 0
    # dataset_hi = load_from_disk("/home/PL-BERT/data/hi")
    # dataset_en = load_from_disk("/home/PL-BERT/data/en_ds")

    dataset = load_from_disk("/home/PL-BERT/data/filtered_en_hi")
    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    
    batch_size = config["batch_size"]
    train_loader, val_loader = build_dataloaders(dataset, 
                                               batch_size=batch_size, 
                                               num_workers=0, 
                                               dataset_config=config['dataset_params'])

    albert_base_configuration = AlbertConfig(**config['model_params'])
    
    bert = AlbertModel(albert_base_configuration)
    bert = MultiTaskModel(bert, 
                          num_vocab=1 + max([m['token'] for m in token_maps.values()]), 
                          num_tokens=config['model_params']['vocab_size'],
                          hidden_size=config['model_params']['hidden_size'])
    bert = bert.to(device)  # Move model to GPU
    
    load = False
    try:
        files = os.listdir(log_dir)
        ckpts = []
        for f in os.listdir(log_dir):
            if f.startswith("step_"): ckpts.append(f)

        iters = [int(f.split('_')[-1].split('.')[0]) for f in ckpts if os.path.isfile(os.path.join(log_dir, f))]
        iters = sorted(iters)[-1]
    except:
        iters = 0
        load = False
    
    optimizer = AdamW(bert.parameters(), lr=1e-4)
    
    accelerator = Accelerator(mixed_precision=config['mixed_precision'], 
                              split_batches=True, 
                              kwargs_handlers=[ddp_kwargs],
                              device_placement=True)  # Enable device placement
    load = True
    if load:
        checkpoint = torch.load("/home/PL-BERT/top_loss_plbert_en_hi/step_21801.t7", map_location='cpu')
        state_dict = checkpoint['net']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v

        bert.load_state_dict(new_state_dict, strict=False)
        
        accelerator.print('Checkpoint loaded.')
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    bert, optimizer, train_loader, val_loader = accelerator.prepare(
        bert, optimizer, train_loader, val_loader
    )
    
    val_loss, val_vocab_loss, val_token_loss = validate(bert, val_loader, criterion, device, accelerator)

    if accelerator.is_main_process:
        print("val_total_loss:", val_loss)
        print("val_vocab_loss:", val_vocab_loss)
        print("val_token_loss:", val_token_loss)
        print('Start training...')
    num_epochs = config["epochs"]
    for epoch in range(num_epochs):
        running_loss = 0
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for step, batch in enumerate(train_loader):
            step_start_time = time.time()  # Start timing for the step
            curr_steps += 1

            words, labels, phonemes, input_lengths, masked_indices = [x.to(device) if torch.is_tensor(x) else x for x in batch]
            text_mask = length_to_mask(torch.Tensor(input_lengths)).to(device)
            
            tokens_pred, words_pred = bert(phonemes, attention_mask=(~text_mask).int())
            
            loss_vocab = 0
            for _s2s_pred, _text_input, _text_length, _masked_indices in zip(words_pred, words, input_lengths, masked_indices):
                loss_vocab += criterion(_s2s_pred[:_text_length], 
                                        _text_input[:_text_length])
            loss_vocab /= words.size(0)
            
            loss_token = 0
            sizes = 1
            for _s2s_pred, _text_input, _text_length, _masked_indices in zip(tokens_pred, labels, input_lengths, masked_indices):
                if len(_masked_indices) > 0:
                    _text_input = _text_input[:_text_length][_masked_indices]
                    loss_tmp = criterion(_s2s_pred[:_text_length][_masked_indices], 
                                         _text_input[:_text_length]) 
                    loss_token += loss_tmp
                    sizes += 1
            loss_token /= sizes

            loss = loss_vocab + loss_token

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            running_loss += loss.item()

            step_time = time.time() - step_start_time  # Calculate step duration

            if (curr_steps + 1) % log_interval == 0 and accelerator.is_main_process:  
                print(f'Step [{curr_steps + 1}/{num_steps}], '
                      f'Loss: {running_loss / log_interval:.5f}, '
                      f'Vocab Loss: {loss_vocab:.5f}, '
                      f'Token Loss: {loss_token:.5f}, '
                      f'Time per step: {step_time:.4f} seconds')
                running_loss = 0

        val_loss, val_vocab_loss, val_token_loss = validate(bert, val_loader, criterion, device, accelerator)

        if accelerator.is_main_process:
            print("val_total_loss:", val_loss)
            print("val_vocab_loss:", val_vocab_loss)
            print("val_token_loss:", val_token_loss)
            print('Start training...')
            print(f"Epoch {epoch + 1}/{num_epochs}")


            print(f'Average Train Loss: {loss:.5f}')
            print(f'Validation Loss: {val_loss:.5f}')

            accelerator.print('Saving..')

            state = {
                'net':  bert.state_dict(),
                'step': curr_steps,
                'optimizer': optimizer.state_dict(),
            }

            accelerator.save(state, log_dir + '/step_' + str(curr_steps + 1) + '.t7')




if __name__ == "__main__":
    train()
