import torch
import snntorch as snn
from snntorch import surrogate
from torch import nn, optim
import os
import shutil

class DialogueSNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super().__init__()
        self.time_steps = 20  # Temporal processing window
        self.max_seq_len = 15  # Maximum sequence length to process
        self.hidden_size = hidden_size
        
        self.embed = nn.Embedding(vocab_size, 64)
        
        # Spiking layers with recurrence
        self.lif1 = snn.Leaky(beta=0.95, spike_grad=surrogate.atan())
        self.lif2 = snn.Leaky(beta=0.95, spike_grad=surrogate.atan())
        
        self.fc1 = nn.Linear(64, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        
        # Membrane potential states
        self.mem1 = None
        self.mem2 = None

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        batch_size, seq_len = x.shape
        
        # Initialize membrane potentials
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()
        
        # Process entire sequence through temporal window
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t]  # Current timestep input
            x_embedded = self.embed(x_t)  # [batch_size, embedding_dim]
            
            # Spiking neural processing
            for _ in range(self.time_steps):
                cur1 = self.fc1(x_embedded)
                spk1, self.mem1 = self.lif1(cur1, self.mem1)
                cur2 = self.fc2(spk1)
                spk2, self.mem2 = self.lif2(cur2, self.mem2)
            
            outputs.append(spk2)
        
        # [seq_len, batch_size, vocab_size] -> [batch_size, seq_len, vocab_size]
        return torch.stack(outputs).permute(1, 0, 2)

def safe_save(state, filename):
    """Atomic checkpoint save to prevent corruption"""
    temp_file = f"{filename}.tmp"
    torch.save(state, temp_file)
    shutil.move(temp_file, filename)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load processed data
    word2idx = torch.load("data/word2idx.pt")
    pairs = torch.load("data/processed_pairs.pt")
    
    # Model setup
    model = DialogueSNN(len(word2idx)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<pad>"])
    
    # Checkpoint recovery
    start_epoch = 0
    if os.path.exists("checkpoint.pt"):
        print("Resuming from checkpoint...")
        checkpoint = torch.load("checkpoint.pt", map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
    
    # Training loop
    for epoch in range(start_epoch, 200):
        model.train()
        total_loss = 0.0
        
        for context, response in pairs[:500]:  # Use first 500 pairs for stability
            # Encode sequences with padding/trimming
            def process_seq(text, max_len=15):
                tokens = [word2idx["<start>"]] + \
                        [word2idx.get(w.lower(), word2idx["<unk>"]) 
                        for w in text.split()][:max_len-1]
                tokens += [word2idx["<pad>"]] * (max_len - len(tokens))
                return torch.tensor(tokens[:max_len], len(tokens)
            
            ctx_tensor, ctx_len = process_seq(context)
            resp_tensor, resp_len = process_seq(response)
            
            # Move to device
            ctx_tensor = ctx_tensor.to(device)
            resp_tensor = resp_tensor.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(ctx_tensor.unsqueeze(0))  # [1, seq_len, vocab_size]
            
            # Calculate loss for each sequence position
            loss = 0
            for t in range(resp_len):
                loss += criterion(outputs[:, t, :], resp_tensor[t].unsqueeze(0))
            
            # Backpropagate and update
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Save checkpoint
        safe_save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": total_loss / len(pairs),
        }, "checkpoint.pt")
        
        print(f"Epoch {epoch+1:03d} | Loss: {total_loss/len(pairs):.4f}")

if __name__ == "__main__":
    train()
