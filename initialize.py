import os
import json
import torch
from collections import Counter

def preprocess_data():
    # Validate input data
    data_path = "data/movie_lines.txt"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing data file: {data_path}")

    # Load and parse raw movie lines
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = []
        for line in f:
            # Parse the structured format
            parts = line.strip().split(" +++$+++ ")
            if len(parts) >= 5:  # Ensure we have all components
                dialogue = parts[4].strip()  # 5th element is the actual dialogue
                if dialogue:  # Skip empty dialogue
                    lines.append(dialogue)

    # Create context-response pairs from consecutive lines
    pairs = []
    for i in range(len(lines)-1):
        context = lines[i]
        response = lines[i+1]
        if context and response:  # Skip any empty strings
            pairs.append((context, response))

    # Build vocabulary with frequency filtering
    word_counts = Counter()
    for context, response in pairs:
        word_counts.update(context.lower().split())
        word_counts.update(response.lower().split())

    # Create vocabulary with special tokens
    vocab = ["<pad>", "<unk>", "<start>", "<end>"] + \
            [word for word, count in word_counts.items() if count >= 3]
    word2idx = {word: idx for idx, word in enumerate(vocab)}

    # Save processed data
    os.makedirs("data", exist_ok=True)
    
    with open("data/processed_pairs.json", "w") as f:
        json.dump(pairs, f, ensure_ascii=False)
    
    with open("data/word2idx.json", "w") as f:
        json.dump(word2idx, f, ensure_ascii=False)
    
    torch.save(word2idx, "data/word2idx.pt")
    
    print(f"âœ… Successfully processed {len(pairs)} dialogue pairs")
    print(f"ðŸ“– Final vocabulary size: {len(vocab)}")

if __name__ == "__main__":
    preprocess_data()