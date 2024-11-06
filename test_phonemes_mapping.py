def align_subtokens_to_phonemes(subtokens, ipa_phonemes, token_ids):
    """
    Align subtokens to IPA phonemes along with token IDs based on length and sound structure.
    
    Parameters:
    subtokens (list of str): List of subtokens for a word.
    ipa_phonemes (str): IPA phoneme string for the word.
    token_ids (list of int): List of token IDs for each subtoken.
    
    Returns:
    list of tuple: Each tuple contains a subtoken, its aligned IPA phoneme, and its token ID.
    """
    # Check if subtokens and token IDs are of equal length
    if len(subtokens) != len(token_ids):
        raise ValueError("Subtokens and token IDs must be of the same length.")
    
    # Step 1: Calculate approximate length of IPA phonemes for each subtoken
    ipa_length = len(ipa_phonemes)
    subtoken_lengths = [len(st.replace('##', '')) for st in subtokens]
    total_length = sum(subtoken_lengths)
    
    # Determine split points in the IPA phonemes based on subtoken lengths
    split_points = []
    accumulated = 0
    for length in subtoken_lengths:
        portion = (length / total_length) * ipa_length
        accumulated += portion
        split_points.append(int(round(accumulated)))  # Round to nearest index
    
    # Step 2: Split the IPA phoneme string at the determined points
    aligned_phonemes = []
    start = 0
    for end in split_points:
        aligned_phonemes.append(ipa_phonemes[start:end])
        start = end

    # Step 3: Create a mapping of each subtoken to its aligned IPA phoneme and token ID
    mapping = list(zip(subtokens, aligned_phonemes, token_ids))
    
    return mapping

# Updated data including token IDs
data = [
    (['र', '##ी', '##वा'], 'ɾˈiːʋaː', [891, 10914, 28960]),
    (['के'], 'keː', [10412]),
    (['ज', '##ंग', '##लों'], 'ɟˈʌŋɡəlˌõ', [872, 31222, 51665]),
    (['में'], 'mẽː', [10532]),
    (['ही'], 'hˈi', [14080]),
    (['स', '##फ', '##ेद'], 'səpʰˈeːd', [898, 28863, 82813]),
    (['ब', '##ा', '##घ'], 'bˈaːɡʰ', [887, 11208, 55759]),
    (['की'], 'ki', [10826]),
    (['न', '##स', '##्ल'], 'nˈʌslə', [884, 13432, 50101]),
    (['प', '##ाई'], 'pˈaːi', [885, 30472]),
    
    # New test data
    (['हैं'], 'hɛ̃', [11716]),
    (['।'], 'pˈuːrnwɪɾˈaːm', [920]),
    (['जिले'], 'ɟˈɪleː', [32291]),
    (['की'], 'ki', [10826]),
    (['प्रमुख'], 'pɾˈʌmʊkʰ', [29218]),
    (['उ', '##प', '##ज'], 'ˈʊpəɟ', [855, 18187, 17413]),
    (['ध', '##ान'], 'dʰˈaːn', [883, 21202]),
    (['है'], 'hɛː', [10569]),
    (['।'], 'pˈuːrnwɪɾˈaːm', [920]),
    (['जिले'], 'ɟˈɪleː', [32291]),
    (['के'], 'keː', [10412]),
    (['त', '##ाला'], 'tˈaːlaː', [880, 65986]),
    (['नामक'], 'nˈaːmək', [56734]),
    (['ज', '##ंग', '##ल'], 'ɟˈʌŋɡəl', [872, 31222, 11714])
]

# Run alignment for each word in the new test data
for subtokens, ipa_phonemes, token_ids in data:
    mapping = align_subtokens_to_phonemes(subtokens, ipa_phonemes, token_ids)
    print(f"IPA: {ipa_phonemes}, Subtokens: {subtokens}, Token IDs: {token_ids}")
    print("Mapping:", mapping)
    print()
