#!/usr/bin/env python3
"""
Cornell Movie Dialogues Preprocessor with Verbose Logging
"""

import os
import json
import torch
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# === Logging Utilities ========================================================
def log(message: str, level: str = "INFO"):
    """Custom logging function with level prefixes"""
    print(f"[{level}] {message}")

# === Core Processing Functions ================================================
def load_raw_data(data_path: Path) -> list:
    """Load and validate raw movie lines file"""
    log(f"Loading raw data from: {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {data_path}")
    
    dialogues = []
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        total_lines = sum(1 for _ in f)
        f.seek(0)
        
        log(f"Processing {total_lines:,} raw lines")
        for line in tqdm(f, total=total_lines, desc="Parsing lines", unit=" lines"):
            parts = line.strip().split(" +++$+++ ")
            if len(parts) >= 5:
                dialogue = parts[4].strip()
                if dialogue:
                    dialogues.append(dialogue)
    
    log(f"Found {len(dialogues):,} valid dialogue entries")
    return dialogues

def create_context_pairs(dialogues: list) -> list:
    """Generate context-response pairs from sequential dialogues"""
    log("Creating conversation pairs")
    
    pairs = []
    for i in tqdm(range(len(dialogues)-1), 
                desc="Generating pairs", 
                unit=" pairs"):
        if dialogues[i] and dialogues[i+1]:
            pairs.append((dialogues[i], dialogues[i+1]))
    
    log(f"Created {len(pairs):,} valid context-response pairs")
    return pairs

def build_vocabulary(pairs: list, min_freq: int = 5) -> tuple:
    """Build vocabulary with frequency filtering"""
    log(f"Building vocabulary (min frequency={min_freq})")
    
    word_counts = Counter()
    for context, response in tqdm(pairs, 
                                desc="Counting words", 
                                unit=" pairs"):
        word_counts.update(context.lower().split())
        word_counts.update(response.lower().split())
    
    special_tokens = ["<pad>", "<unk>", "<start>", "<end>"]
    vocab = special_tokens + [
        word for word, count in word_counts.items() 
        if count >= min_freq
    ]
    
    log(f"Vocabulary contains {len(vocab):,} unique tokens")
    return vocab, {word: idx for idx, word in enumerate(vocab)}

# === Main Processing Pipeline =================================================
def preprocess_corpus(
    input_path: Path = Path("data/movie_lines.txt"),
    output_dir: Path = Path("processed"),
    min_freq: int = 5
):
    """Full preprocessing pipeline"""
    try:
        # Setup directories
        output_dir.mkdir(parents=True, exist_ok=True)
        log(f"Output directory: {output_dir}")

        # Load and parse raw data
        raw_dialogues = load_raw_data(input_path)

        # Create conversation pairs
        pairs = create_context_pairs(raw_dialogues)

        # Build vocabulary
        vocab, word2idx = build_vocabulary(pairs, min_freq)

        # Save processed data
        log("Saving processed artifacts")
        
        # Save vocabulary
        vocab_path = output_dir / "word2idx.pt"
        torch.save(word2idx, vocab_path)
        log(f"Saved vocabulary to {vocab_path}")

        # Save pairs
        pairs_path = output_dir / "processed_pairs.json"
        with open(pairs_path, "w") as f:
            json.dump(pairs, f, ensure_ascii=False)
        log(f"Saved processed pairs to {pairs_path}")

        # Final report
        log("\nPreprocessing complete!")
        log(f"┌{'─' * 40}┐")
        log(f"│ {'Total pairs:':<20}{len(pairs):>18,} │")
        log(f"│ {'Vocabulary size:':<20}{len(vocab):>18,} │")
        log(f"└{'─' * 40}┘")

    except Exception as e:
        log(f"Preprocessing failed: {str(e)}", level="ERROR")
        raise

if __name__ == "__main__":
    # Configure paths
    input_data = Path("data/movie_lines.txt")
    output_dir = Path("processed")
    
    # Run preprocessing
    preprocess_corpus(input_data, output_dir)