import torch
import snntorch as snn
from snntorch import surrogate
from torch import nn, optim
import os
import shutil
import json
import time
import numpy as np
from tqdm import tqdm
from visualization import plot_spike_raster

class DialogueSNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, embed_size=128):
        super().__init__()
        self.time_steps = 20
        self.hidden_size = hidden_size
        
        # Enhanced network architecture
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.lif1 = snn.Leaky(beta=0.85, spike_grad=surrogate.atan())
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=0.85, spike_grad=surrogate.atan())
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.lif3 = snn.Leaky(beta=0.85, spike_grad=surrogate.atan())
        self.fc_out = nn.Linear(hidden_size, vocab_size)

        # STDP parameters
        self.std_learning = True
        self.a_pre = 0.005
        self.a_post = -0.003
        self.stdp_interval = 3
        self.stdp_decay = 0.95

        # Hybrid learning balance
        self.bp_scale = 0.7
        self.stdp_scale = 0.3

        # Spike recording buffers
        self.spike_records = {
            'fc1': [],
            'fc2': [],
            'fc3': []
        }

    def forward(self, x, mem1=None, mem2=None, mem3=None):
        batch_size, seq_len = x.shape
        mem1 = self.lif1.init_leaky() if mem1 is None else mem1
        mem2 = self.lif2.init_leaky() if mem2 is None else mem2
        mem3 = self.lif3.init_leaky() if mem3 is None else mem3
        
        # Reset spike records
        self.spike_records = {k: [] for k in self.spike_records}
        stdp_updates = {}

        outputs = []
        for t in range(seq_len):
            x_embedded = self.embed(x[:, t])
            temporal_outputs = []

            for step in range(self.time_steps):
                # Layer 1
                cur1 = self.fc1(x_embedded)
                spk1, mem1 = self.lif1(cur1, mem1)
                self.spike_records['fc1'].append(spk1.detach())

                # Layer 2
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                self.spike_records['fc2'].append(spk2.detach())

                # Layer 3
                cur3 = self.fc3(spk2)
                spk3, mem3 = self.lif3(cur3, mem3)
                self.spike_records['fc3'].append(spk3.detach())
                temporal_outputs.append(spk3)

                # STDP calculations
                if step % self.stdp_interval == 0:
                    with torch.no_grad():
                        # Calculate STDP updates
                        pre_act = x_embedded.mean(0)
                        post_act = spk1.mean(0)
                        delta_fc1 = torch.outer(self.a_post * post_act, self.a_pre * pre_act)

                        pre_act = spk1.mean(0)
                        post_act = spk2.mean(0)
                        delta_fc2 = torch.outer(self.a_post * post_act, self.a_pre * pre_act)

                        pre_act = spk2.mean(0)
                        post_act = spk3.mean(0)
                        delta_fc3 = torch.outer(self.a_post * post_act, self.a_pre * pre_act)

                        # Accumulate updates
                        if 'fc1' not in stdp_updates:
                            stdp_updates['fc1'] = delta_fc1
                            stdp_updates['fc2'] = delta_fc2
                            stdp_updates['fc3'] = delta_fc3
                        else:
                            stdp_updates['fc1'] += delta_fc1
                            stdp_updates['fc2'] += delta_fc2
                            stdp_updates['fc3'] += delta_fc3

            # Temporal pooling
            pooled_output = torch.stack(temporal_outputs).mean(dim=0)
            outputs.append(self.fc_out(pooled_output))

        # Convert spike records
        spike_data = {
            layer: torch.stack(self.spike_records[layer]).squeeze().cpu()
            for layer in self.spike_records
        }

        return torch.stack(outputs).permute(1, 0, 2), mem1, mem2, mem3, stdp_updates, spike_data

def safe_save(state, filename):
    """Atomic model saving"""
    temp_file = f"{filename}.tmp"
    torch.save(state, temp_file)
    shutil.move(temp_file, filename)

def process_seq(text, word2idx, max_len=15, add_end=False):
    """Process text sequence with safety checks"""
    tokens = [word2idx["<start>"]]
    valid_tokens = [word2idx.get(w.lower(), word2idx["<unk>"]) 
                   for w in text.split()[:max_len - (2 if add_end else 1)]]
    tokens += [t for t in valid_tokens if t != word2idx["<unk>"]]
    
    if add_end:
        tokens.append(word2idx["<end>"])
    
    padding = [word2idx["<pad>"]] * (max_len - len(tokens))
    return torch.tensor(tokens + padding[:max_len]), len(tokens)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    with open("data/word2idx.json", "r") as f:
        word2idx = json.load(f)
    with open("data/processed_pairs.json", "r") as f:
        pairs = json.load(f)

    # Initialize model
    model = DialogueSNN(len(word2idx)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<pad>"])

    # Training configuration
    vis_frequency = 50  # Visualize every 50 batches
    best_loss = float('inf')
    start_epoch = 0

    # Resume training
    if os.path.exists("checkpoint.pt"):
        checkpoint = torch.load("checkpoint.pt", map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print(f"Resumed training from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, 200):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        progress = tqdm(
            pairs[:5000],
            desc=f"Epoch {epoch+1:03d}",
            bar_format="{l_bar}{bar:20}{r_bar}",
            postfix={"loss": "N/A"}
        )

        for batch_idx, (context, response) in enumerate(progress):
            # Process sequences
            ctx_tensor, _ = process_seq(context, word2idx)
            resp_tensor, _ = process_seq(response, word2idx, add_end=True)
            ctx_tensor = ctx_tensor.to(device).unsqueeze(0)
            resp_tensor = resp_tensor.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs, mem1, mem2, mem3, stdp_updates, spike_data = model(ctx_tensor)
            
            # Calculate loss
            loss = criterion(
                outputs.view(-1, outputs.size(-1)),
                resp_tensor.view(-1)
            )
            
            # Backpropagation
            loss.backward()
            
            # Apply STDP updates
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'fc' in name and 'weight' in name:
                        layer = name.split('.')[0]
                        stdp_grad = stdp_updates[layer].to(device)
                        param.grad = (model.bp_scale * param.grad +
                                      model.stdp_scale * stdp_grad)
            
            # Optimization step
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            progress.set_postfix({
                "loss": f"{epoch_loss/(batch_idx+1):.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.4f}"
            })

            # Generate visualizations
            if batch_idx % vis_frequency == 0 and batch_idx < 5:
                plot_path = plot_spike_raster(
                    spike_data,
                    epoch=epoch+1,
                    batch_idx=batch_idx,
                    prefix=f"epoch{epoch+1}_batch{batch_idx}"
                )
                progress.write(f"Saved spike plot: {plot_path}")

        # Epoch post-processing
        avg_loss = epoch_loss / len(progress)
        scheduler.step(avg_loss)
        
        # Save checkpoint
        safe_save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss,
        }, "checkpoint.pt")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            shutil.copyfile("checkpoint.pt", "best_model.pt")

        # Final visualization
        plot_path = plot_spike_raster(
            spike_data,
            epoch=epoch+1,
            batch_idx='final',
            prefix=f"epoch{epoch+1}_final"
        )
        print(f"\nSaved final spike plot: {plot_path}")
        print(f"Epoch {epoch+1} completed in {time.time()-start_time:.1f}s")
        print(f"Average loss: {avg_loss:.4f}")
        print("="*50)

if __name__ == "__main__":
    train()