import os
import json
from collections import Counter

def preprocess_data():
    # Load and validate data
    data_path = "data/formatted_movie_lines.txt"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing data file: {data_path}")

    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip().split(" +++$+++ ")[-1] for line in f if line.strip()]

    # Create context-response pairs with validation
    pairs = []
    for i in range(len(lines)-1):
        if len(lines[i]) > 1 and len(lines[i+1]) > 1:  # Skip empty lines
            pairs.append((lines[i], lines[i+1]))

    # Build vocabulary with frequency threshold
    word_counts = Counter()
    for context, response in pairs:
        for sentence in [context, response]:
            word_counts.update(sentence.lower().split())

    vocab = ["<pad>", "<unk>", "<start>", "<end>"] + \
            [word for word, count in word_counts.items() if count >= 3]
    word2idx = {word: idx for idx, word in enumerate(vocab)}

    # Save with proper serialization
    os.makedirs("data", exist_ok=True)
    
    # Save pairs as JSON
    with open("data/processed_pairs.json", "w") as f:
        json.dump(pairs, f)
    
    # Save vocabulary as JSON
    with open("data/word2idx.json", "w") as f:
        json.dump(word2idx, f)
    
    print(f"Preprocessed {len(pairs)} pairs. Vocabulary size: {len(vocab)}")

if __name__ == "__main__":
    preprocess_data()
