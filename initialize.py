import os
import json
import torch
from collections import Counter

def preprocess_data():
    # Validate input data
    data_path = "data/formatted_movie_lines.txt"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing data file: {data_path}")

    # Load and clean data
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip().split(" +++$+++ ")[-1] for line in f if line.strip()]

    # Create dialogue pairs with quality checks
    pairs = []
    for i in range(len(lines)-1):
        context = lines[i].strip()
        response = lines[i+1].strip()
        if context and response:  # Skip empty strings
            pairs.append((context, response))

    # Build vocabulary with frequency filtering
    word_counts = Counter()
    for context, response in pairs:
        word_counts.update(context.lower().split())
        word_counts.update(response.lower().split())

    vocab = ["<pad>", "<unk>", "<start>", "<end>"] + \
            [word for word, count in word_counts.items() if count >= 3]
    word2idx = {word: idx for idx, word in enumerate(vocab)}

    # Save processed data
    os.makedirs("data", exist_ok=True)
    
    with open("data/processed_pairs.json", "w") as f:
        json.dump(pairs, f)
    
    with open("data/word2idx.json", "w") as f:
        json.dump(word2idx, f)
    
    torch.save(word2idx, "data/word2idx.pt")  # PyTorch format
    
    print(f"✅ Preprocessed {len(pairs)} pairs")
    print(f"📖 Vocabulary size: {len(vocab)}")

if __name__ == "__main__":
    preprocess_data()
