import yaml
import os
import logging
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm
from phonemize import phonemize
import phonemizer
from transformers import AutoTokenizer
from datasets import load_dataset, disable_progress_bar, concatenate_datasets
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('datasets').setLevel(logging.ERROR)

# Global variables that will be shared across processes
global_phonemizer = None
tokenizer = None
dataset = None
num_shards = None
root_directory = None

def safe_phonemize(text, phonemizer, tokenizer):
    try:
        return phonemize(text, phonemizer, tokenizer)
    except Exception as e:
        # Return empty result if phonemization fails
        print(e)
        print(f"Phonemization error: {str(e)[:100]}...")  # Print first 100 chars of error
        return {"phonemes": "", "tokens": []}
        # raise`` e


def init_worker():  
    """Initialize global variables for each worker process"""
    global global_phonemizer, tokenizer, dataset
    
    # Initialize phonemizer for this process
    global_phonemizer = phonemizer.backend.EspeakBackend(
        language='en-us', 
        preserve_punctuation=True,
        with_stress=True,
        language_switch="remove-flags"
    )
    
    # Load tokenizer for this process
    config = yaml.safe_load(open("Configs/config.yml"))
    tokenizer = AutoTokenizer.from_pretrained(config['dataset_params']['tokenizer'])
    
    # Load and concatenate datasets
    dataset = load_dataset("wikimedia/wikipedia", "20231101.hi")['train']
    # dataset_en = load_dataset("wikimedia/wikipedia", "20231101.en")['train']
    
    # # Concatenate datasets
    # dataset = concatenate_datasets([dataset_hi, dataset_en])
    
    # Disable progress bar for dataset operations
    disable_progress_bar()

def process_shard(shard_idx):
    directory = os.path.join(root_directory, f"shard_{shard_idx}")
    
    if os.path.exists(directory):
        print(f"Shard {shard_idx} already exists!")
        return
    
    print(f'Processing shard {shard_idx}...')
    
    try:
        # Shard the dataset
        shard = dataset.shard(num_shards=num_shards, index=shard_idx)
        
        # Process the shard without progress bar, using safe_phonemize
        processed_dataset = shard.map(
            lambda t: safe_phonemize(t['text'], global_phonemizer, tokenizer),
            remove_columns=['text'],
            load_from_cache_file=False
        )
        
        # Check if processed dataset is empty
        if len(processed_dataset) == 0:
            print(f"Warning: Shard {shard_idx} produced empty dataset!")
            return
        
        # Save the processed shard
        os.makedirs(directory, exist_ok=True)
        processed_dataset.save_to_disk(directory)
        print(f'Completed shard {shard_idx}')
        
    except Exception as e:
        print(f"Error processing shard {shard_idx}: {str(e)[:100]}...")
        # Continue with next shard instead of crashing

def main():
    global root_directory, num_shards
    
    # Initialize global variables
    root_directory = "/mnt/data/wiki_phoneme_hindi"
    num_shards = 50000
    
    # Create root directory
    os.makedirs(root_directory, exist_ok=True)
    
    # Number of processes
    num_processes = 96
    print(f"Starting processing with {num_processes} processes")
    
    # Create process pool with initialization
    with Pool(processes=num_processes, initializer=init_worker) as pool:
        # Process shards with progress bar
        list(tqdm(
            pool.imap_unordered(process_shard, range(num_shards)),
            total=num_shards,
            desc="Processing shards"
        ))

if __name__ == "__main__":
    main()