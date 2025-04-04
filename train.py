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

    def forward(self, x, mem1=None, mem2=None):
        batch_size, seq_len = x.shape
        mem1 = self.lif1.init_leaky() if mem1 is None else mem1
        mem2 = self.lif2.init_leaky() if mem2 is None else mem2
        
        outputs = []
        for t in range(seq_len):
            x_embedded = self.embed(x[:, t])
            
            # Temporal processing loop
            for _ in range(self.time_steps):
                cur1 = self.fc1(x_embedded)
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
            
            outputs.append(spk2)
        
        return torch.stack(outputs).permute(1, 0, 2), mem1, mem2

def safe_save(state, filename):
    """Atomic save to prevent corruption"""
    temp_file = f"{filename}.tmp"
    torch.save(state, temp_file)
    shutil.move(temp_file, filename)

def process_seq(text, word2idx, max_len=15, add_end=False):
    """Process text sequence with start/end tokens"""
    tokens = [word2idx["<start>"]]
    tokens += [word2idx.get(w.lower(), word2idx["<unk>"]) 
              for w in text.split()][:max_len - (2 if add_end else 1)]
    
    if add_end:
        tokens.append(word2idx["<end>"])
    
    orig_length = len(tokens)
    tokens += [word2idx["<pad>"]] * (max_len - len(tokens))
    return torch.tensor(tokens[:max_len]), orig_length

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load processed data
    print("⏳ Loading training data...")
    with open("data/word2idx.json", "r") as f:
        word2idx = json.load(f)
    with open("data/processed_pairs.json", "r") as f:
        pairs = json.load(f)
    
    # Training configuration
    model = DialogueSNN(len(word2idx)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<pad>"])
    
    # Resume training if checkpoint exists
    start_epoch = 0
    if os.path.exists("checkpoint.pt"):
        print("🔍 Loading existing checkpoint...")
        checkpoint = torch.load("checkpoint.pt", map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"↳ Resuming from epoch {start_epoch} (loss: {checkpoint['loss']:.4f})")
    else:
        print("🚀 Starting new training session")

    # Training loop
    print(f"\n🏁 Starting training on {device}")
    for epoch in range(start_epoch, 200):
        epoch_loss = 0.0
        model.train()
        start_time = time.time()
        
        progress = tqdm(
            pairs[:5000],  # Use subset for demonstration
            desc=f"Epoch {epoch+1:03d}",
            bar_format="{l_bar}{bar:20}{r_bar}",
            postfix={"loss": "N/A"}
        )

        for context, response in progress:
            # Prepare sequences
            ctx_tensor, _ = process_seq(context, word2idx, add_end=False)
            resp_tensor, resp_len = process_seq(response, word2idx, add_end=True)
            
            ctx_tensor = ctx_tensor.to(device).unsqueeze(0)
            resp_tensor = resp_tensor.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs, _, _ = model(ctx_tensor)
            
            # Calculate loss
            loss = criterion(
                outputs.view(-1, outputs.size(-1)),
                resp_tensor.view(-1)
            )
            
            # Backpropagation
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            progress.set_postfix({
                "loss": f"{epoch_loss/(progress.n+1):.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.4f}"
            })

        # Save checkpoint
        avg_loss = epoch_loss / len(pairs)
        safe_save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": avg_loss,
        }, "checkpoint.pt")
        
        # Epoch summary
        epoch_time = time.time() - start_time
        print(f"\n⏱  Epoch {epoch+1} completed in {epoch_time:.1f}s")
        print(f"📉 Average loss: {avg_loss:.4f}")
        print(f"💾 Checkpoint saved\n{'='*40}")

if __name__ == "__main__":
    train()
