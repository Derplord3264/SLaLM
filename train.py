import torch
import snntorch as snn
from snntorch import surrogate
from torch import nn, optim
import os
import shutil

class DialogueSNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super().__init__()
        self.time_steps = 20  # Temporal resolution
        self.hidden_size = hidden_size
        
        # Embedding layer converts words to spike inputs
        self.embed = nn.Embedding(vocab_size, 64)
        
        # Spiking neural network components
        self.lif1 = snn.Leaky(beta=0.95, spike_grad=surrogate.atan())
        self.lif2 = snn.Leaky(beta=0.95, spike_grad=surrogate.atan())
        
        # Recurrent connections
        self.fc1 = nn.Linear(64, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        
        # Membrane potential memory
        self.mem1 = None
        self.mem2 = None
    
    def forward(self, x):
        # Initialize membrane potentials
        if self.mem1 is None:
            self.mem1 = self.lif1.init_leaky()
        if self.mem2 is None:
            self.mem2 = self.lif2.init_leaky()
        
        # Temporal processing loop
        for _ in range(self.time_steps):
            x_embedded = self.embed(x)
            cur1 = self.fc1(x_embedded.mean(dim=1))
            spk1, self.mem1 = self.lif1(cur1, self.mem1)
            cur2 = self.fc2(spk1)
            spk2, self.mem2 = self.lif2(cur2, self.mem2)
        
        return torch.softmax(self.mem2, dim=-1)

def safe_save(state, filename):
    # Atomic save to prevent corruption
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
        
        for i, (context, response) in enumerate(pairs[:1000]):  # Use subset for demo
            # Encode sentences
            context_encoded = [word2idx["<start>"]] + \
                [word2idx.get(word.lower(), word2idx["<unk>"]) for word in context.split()] + \
                [word2idx["<end>"]]
            
            response_encoded = [word2idx["<start>"]] + \
                [word2idx.get(word.lower(), word2idx["<unk>"]) for word in response.split()] + \
                [word2idx["<end>"]]
            
            # Convert to tensors
            ctx_tensor = torch.tensor(context_encoded, dtype=torch.long, device=device)
            resp_tensor = torch.tensor(response_encoded, dtype=torch.long, device=device)
            
            # Training step
            optimizer.zero_grad()
            output = model(ctx_tensor.unsqueeze(0))
            loss = criterion(output, resp_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Save checkpoint safely
        safe_save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": total_loss / len(pairs),
        }, "checkpoint.pt")
        
        print(f"Epoch {epoch+1:03d} | Loss: {total_loss/len(pairs):.4f}")

if __name__ == "__main__":
    train()
