import torch
import os
from collections import Counter

def preprocess_data():
    # Verify data file exists
    if not os.path.exists("data/formatted_movie_lines.txt"):
        raise FileNotFoundError("Please download and place formatted_movie_lines.txt in data/ directory")
    
    # Load and parse data
    with open("data/formatted_movie_lines.txt", "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip().split(" +++$+++ ")[-1] for line in f if line.strip()]
    
    # Create context-response pairs with sliding window
    pairs = []
    for i in range(len(lines)-1):
        context = lines[i]
        response = lines[i+1]
        pairs.append((context, response))
    
    # Build vocabulary with special tokens
    word_counts = Counter()
    special_tokens = ["<pad>", "<unk>", "<start>", "<end>"]
    
    for context, response in pairs:
        for sentence in [context, response]:
            word_counts.update(sentence.lower().split())
    
    # Filter rare words and create mappings
    vocab = special_tokens + [word for word, count in word_counts.items() if count >= 3]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    
    # Create output directory if needed
    os.makedirs("data", exist_ok=True)
    
    # Save processed data
    torch.save(pairs, "data/processed_pairs.pt")
    torch.save(word2idx, "data/word2idx.pt")
    print(f"Preprocessed {len(pairs)} dialogue pairs. Vocabulary size: {len(vocab)}")

if __name__ == "__main__":
    preprocess_data()
