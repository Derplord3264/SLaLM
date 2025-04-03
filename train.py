import torch
import snntorch as snn
from snntorch import surrogate
from torch import nn, optim
import os
import shutil
import json
import time
from tqdm import tqdm

class DialogueSNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super().__init__()
        self.time_steps = 20
        self.hidden_size = hidden_size
        
        self.embed = nn.Embedding(vocab_size, 64)
        self.lif1 = snn.Leaky(beta=0.95, spike_grad=surrogate.atan())
        self.lif2 = snn.Leaky(beta=0.95, spike_grad=surrogate.atan())
        self.fc1 = nn.Linear(64, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        self.mem1 = None
        self.mem2 = None

    def forward(self, x):
        batch_size, seq_len = x.shape
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t]
            x_embedded = self.embed(x_t)
            
            for _ in range(self.time_steps):
                cur1 = self.fc1(x_embedded)
                spk1, self.mem1 = self.lif1(cur1, self.mem1)
                cur2 = self.fc2(spk1)
                spk2, self.mem2 = self.lif2(cur2, self.mem2)
            
            outputs.append(spk2)
        
        return torch.stack(outputs).permute(1, 0, 2)

def safe_save(state, filename):
    temp_file = f"{filename}.tmp"
    torch.save(state, temp_file)
    shutil.move(temp_file, filename)

def process_seq(text, word2idx, max_len=15):
    tokens = [word2idx["<start>"]]
    tokens += [word2idx.get(w.lower(), word2idx["<unk>"]) 
              for w in text.split()][:max_len-1]
    orig_length = len(tokens)
    tokens += [word2idx["<pad>"]] * (max_len - len(tokens))
    return torch.tensor(tokens[:max_len]), orig_length

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data with verbose logging
    print("⏳ Loading dataset...")
    with open("data/word2idx.json", "r") as f:
        word2idx = json.load(f)
    with open("data/processed_pairs.json", "r") as f:
        pairs = json.load(f)
    
    print(f"📦 Dataset stats:\n"
          f" - Total pairs: {len(pairs):,}\n"
          f" - Vocabulary size: {len(word2idx):,}\n"
          f" - Using device: {device}\n"
          f" - CUDA available: {torch.cuda.is_available()}\n")

    model = DialogueSNN(len(word2idx)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<pad>"])
    
    # Training state initialization
    start_epoch = 0
    if os.path.exists("checkpoint.pt"):
        print("🔍 Found existing checkpoint:")
        checkpoint = torch.load("checkpoint.pt", map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        print(f" - Resuming from epoch {start_epoch}")
        print(f" - Previous loss: {checkpoint['loss']:.4f}\n")
    else:
        print("🚀 Starting fresh training session\n")

    # Training loop with progress tracking
    for epoch in range(start_epoch, 200):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        processed_pairs = 0
        
        # Progress bar with detailed stats
        progress = tqdm(
            pairs[:500],  # Use subset for demo
            desc=f"🏃 Epoch {epoch+1:03d}",
            bar_format="{l_bar}{bar:20}{r_bar}",
            postfix=dict(loss="N/A")
        )

        for context, response in progress:
            # Process sequences
            ctx_tensor, ctx_len = process_seq(context, word2idx)
            resp_tensor, resp_len = process_seq(response, word2idx)
            
            ctx_tensor = ctx_tensor.to(device).unsqueeze(0)
            resp_tensor = resp_tensor.to(device)
            
            # Training step
            optimizer.zero_grad()
            outputs = model(ctx_tensor)
            
            loss = 0
            for t in range(resp_len):
                loss += criterion(outputs[:, t, :], resp_tensor[t].unsqueeze(0))
            
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            processed_pairs += 1
            
            # Update progress bar every 10 steps
            if processed_pairs % 10 == 0:
                progress.set_postfix({
                    'loss': f"{total_loss/processed_pairs:.4f}",
                    'pairs': processed_pairs
                })

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(pairs)
        
        # Save checkpoint with verbose output
        safe_save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": avg_loss,
        }, "checkpoint.pt")
        
        print(f"\n✅ Epoch {epoch+1:03d} complete\n"
              f" - Avg loss: {avg_loss:.4f}\n"
              f" - Duration: {epoch_time:.1f}s\n"
              f" - Checkpoint saved: checkpoint.pt\n"
              f"{'='*40}\n")

if __name__ == "__main__":
    train()
