#!/usr/bin/env python3
"""
SNN Dialogue Trainer with Verbose Logging and Progress Tracking
"""

import time
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from torch import nn, optim
import snntorch as snn
from snntorch import surrogate

# === Logging Configuration ====================================================
def log(message: str, level: str = "INFO"):
    print(f"[{level}] {message}")

# === Model Architecture =======================================================
class SpikeDialogueModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 512, 
                time_steps: int = 50, use_stdp: bool = False):
        super().__init__()
        log(f"Initializing SNN with: vocab_size={vocab_size}, "
            f"hidden_size={hidden_size}, time_steps={time_steps}, STDP={use_stdp}")
        
        # Architecture parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.time_steps = time_steps
        self.use_stdp = use_stdp
        
        # Embedding layer
        self.embed = nn.Embedding(vocab_size, 256)
        log(f"Created embedding layer: {self.embed.weight.shape}")
        
        # Spiking neurons
        self.lif1 = snn.Leaky(
            beta=0.85, threshold=0.8, learn_beta=True,
            spike_grad=surrogate.fast_sigmoid()
        )
        self.lif2 = snn.Synaptic(
            alpha=0.9, beta=0.8, learn_alpha=use_stdp,
            learn_beta=use_stdp, reset_mechanism="none"
        )
        log("Initialized spiking layers")
        
        # Dense projections
        self.fc1 = nn.Linear(256, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        log(f"Created dense layers: {self.fc1}, {self.fc2}")
        
        # STDP parameters
        if use_stdp:
            self.stdp_amp = 0.01
            self.stdp_window = 20
            log(f"STDP enabled: amplitude={self.stdp_amp}, window={self.stdp_window}")

    def forward(self, x: torch.Tensor, state: tuple = None) -> tuple:
        # State initialization
        batch_size, seq_len = x.shape
        mem1 = self.lif1.init_leaky() if state is None else state[0]
        syn2, mem2 = self.lif2.init_synaptic() if state is None else state[1]
        
        outputs = []
        stdp_traces = []
        
        # Temporal processing
        for t in range(seq_len):
            x_embedded = self.embed(x[:, t])
            
            for _ in range(self.time_steps):
                cur1 = self.fc1(x_embedded)
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spk1)
                spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
                
                if self.use_stdp:
                    stdp_traces.append((spk1.detach(), spk2.detach()))
            
            outputs.append(spk2)
        
        return (torch.stack(outputs).permute(1, 0, 2),
                stdp_traces,
                (mem1, (syn2, mem2)))

# === Training Utilities =======================================================
def stdp_update(model: SpikeDialogueModel, traces: list):
    if not model.use_stdp or not traces:
        return
    
    with torch.no_grad():
        log(f"Applying STDP updates ({len(traces)} traces)")
        for pre, post in traces[-model.stdp_window:]:
            dw = torch.outer(post.mean(0), pre.mean(0)) * model.stdp_amp
            model.fc1.weight.data += dw
        
        # Weight normalization
        model.fc1.weight.data = nn.functional.normalize(
            model.fc1.weight.data, p=2, dim=1)
        log("STDP updates applied")

def process_batch(pair: tuple, word2idx: dict, max_len: int = 50) -> tuple:
    context, response = pair
    
    # Process context
    ctx_tokens = [word2idx["<start>"]] + [
        word2idx.get(w.lower(), word2idx["<unk>"]) 
        for w in context.split()][:max_len-1]
    ctx_tensor = torch.tensor(ctx_tokens + [word2idx["<pad>"]] * (max_len - len(ctx_tokens)))
    
    # Process response
    resp_tokens = [word2idx["<start>"]] + [
        word2idx.get(w.lower(), word2idx["<unk>"]) 
        for w in response.split()][:max_len-2]
    resp_tokens += [word2idx["<end>"]]
    resp_tensor = torch.tensor(resp_tokens + [word2idx["<pad>"]] * (max_len - len(resp_tokens)))
    
    return ctx_tensor, resp_tensor

# === Training Loop ============================================================
def main_training_loop(args):
    # === Initialization ===
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Training initialized on {device}")
    log(f"Configuration: {vars(args)}")
    
    # === Data Loading ===
    log("Loading training data")
    data_dir = Path(args.data_dir)
    with open(data_dir/"processed_pairs.json", "r") as f:
        pairs = json.load(f)
    word2idx = torch.load(data_dir/"word2idx.pt")
    log(f"Loaded {len(pairs)} pairs, vocab_size={len(word2idx)}")
    
    # === Model Setup ===
    model = SpikeDialogueModel(
        vocab_size=len(word2idx),
        hidden_size=args.hidden_size,
        time_steps=args.time_steps,
        use_stdp=args.use_stdp
    ).to(device)
    log(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # === Optimization Setup ===
    optimizer = optim.AdamW([
        {'params': model.embed.parameters(), 'lr': 3e-4},
        {'params': model.lif1.parameters(), 'lr': 1e-4},
        {'params': model.lif2.parameters(), 'lr': 5e-5},
        {'params': model.fc1.parameters(), 'lr': 3e-4},
        {'params': model.fc2.parameters(), 'lr': 3e-4},
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<pad>"])
    log("Optimization setup complete")
    
    # === Training State ===
    best_loss = float('inf')
    training_stats = []
    
    # === Epoch Loop ===
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        spike_count = 0
        
        log(f"\n{'=' * 40}")
        log(f"Epoch {epoch+1}/{args.epochs}")
        log(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # === Batch Processing ===
        progress = tqdm(
            pairs[:args.train_size],
            desc=f"Training",
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
            postfix={"loss": "N/A", "spikes": "N/A"}
        )
        
        for pair in progress:
            # Data preparation
            ctx, resp = process_batch(pair, word2idx)
            ctx = ctx.unsqueeze(0).to(device)
            resp = resp.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs, traces, _ = model(ctx)
            
            # Loss calculation
            loss = criterion(outputs.view(-1, outputs.size(-1)), resp.view(-1))
            
            # STDP updates
            if args.use_stdp:
                stdp_update(model, traces)
            
            # Backpropagation
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Metrics tracking
            total_loss += loss.item()
            spike_count += (outputs > 0).sum().item()
            avg_loss = total_loss / (progress.n + 1)
            spike_rate = spike_count / ((progress.n + 1) * outputs.numel())
            
            progress.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "spikes": f"{spike_rate:.2%}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # === Epoch Finalization ===
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(pairs)
        scheduler.step(avg_loss)
        
        # === Validation ===
        if (epoch + 1) % args.val_interval == 0:
            val_loss = validate(model, pairs[-args.val_size:], word2idx, device, criterion)
            log(f"Validation loss: {val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(model, optimizer, epoch, args, best_loss)
                log(f"New best model saved (loss: {best_loss:.4f})")
        
        # === Checkpointing ===
        if (epoch + 1) % args.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch, args, avg_loss)
            log(f"Checkpoint saved at epoch {epoch+1}")
        
        # === Statistics ===
        training_stats.append({
            "epoch": epoch+1,
            "train_loss": avg_loss,
            "val_loss": val_loss if (epoch + 1) % args.val_interval == 0 else None,
            "spike_rate": spike_rate,
            "lr": optimizer.param_groups[0]['lr'],
            "time": epoch_time
        })
        
        log(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")
        log(f"Average training loss: {avg_loss:.4f}")
        log(f"Spike rate: {spike_rate:.2%}")
    
    # === Final Report ===
    total_time = time.time() - start_time
    log(f"\n{'=' * 40}")
    log(f"Training complete in {total_time/3600:.2f} hours")
    log(f"Best validation loss: {best_loss:.4f}")
    log(f"Final learning rate: {optimizer.param_groups[0]['lr']:.2e}")

def validate(model, pairs, word2idx, device, criterion):
    model.eval()
    total_loss = 0
    
    with tqdm(pairs, desc="Validating", bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}") as val_bar:
        for pair in val_bar:
            ctx, resp = process_batch(pair, word2idx)
            ctx = ctx.unsqueeze(0).to(device)
            resp = resp.to(device)
            
            with torch.no_grad():
                outputs, _, _ = model(ctx)
                loss = criterion(outputs.view(-1, outputs.size(-1)), resp.view(-1))
                total_loss += loss.item()
                
            val_bar.set_postfix({"loss": f"{total_loss/(val_bar.n+1):.4f}"})
    
    return total_loss / len(pairs)

def save_checkpoint(model, optimizer, epoch, args, loss):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "loss": loss,
        "config": vars(args)
    }
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, Path(args.save_dir)/f"checkpoint_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SNN Dialogue Model")
    parser.add_argument("--data_dir", type=str, default="processed")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--time_steps", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_size", type=int, default=5000)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--use_stdp", action="store_true")
    args = parser.parse_args()
    
    main_training_loop(args)