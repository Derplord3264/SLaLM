#!/usr/bin/env python3
"""
SNN Dialogue Trainer - Complete Production Version
"""

import os
import json
import time
import torch
import argparse
import snntorch as snn
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from snntorch import surrogate, utils
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

# === Constants ================================================================
DEFAULT_HIDDEN_SIZE = 512
DEFAULT_TIME_STEPS = 50
STDP_UPDATE_INTERVAL = 100

# === Logging Setup ============================================================
class TrainingLogger:
    def __init__(self, log_dir: str = "logs"):
        self.writer = SummaryWriter(log_dir)
        self.console = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}")
        
    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)
        self.console(f"{tag}: {value:.4f}")

    def log_histogram(self, tag: str, values, step: int):
        self.writer.add_histogram(tag, values, step)

# === Model Architecture =======================================================
class SpikeDialogueModel(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 hidden_size: int = DEFAULT_HIDDEN_SIZE,
                 time_steps: int = DEFAULT_TIME_STEPS,
                 use_stdp: bool = False):
        super().__init__()
        
        # Architectural parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.time_steps = time_steps
        self.use_stdp = use_stdp
        
        # Embedding layer
        self.embed = nn.Embedding(vocab_size, 256)
        
        # Spiking neural layers
        self.lif1 = snn.Leaky(
            beta=0.85, threshold=0.8, learn_beta=True,
            spike_grad=surrogate.fast_sigmoid(), init_hidden=True
        )
        self.lif2 = snn.Synaptic(
            alpha=0.9, beta=0.8, learn_alpha=True, learn_beta=True,
            spike_grad=surrogate.atan(), init_hidden=True
        )
        
        # Dense projections
        self.fc1 = nn.Linear(256, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        
        # STDP parameters
        if use_stdp:
            self.stdp_amp = 0.01
            self.stdp_window = 20
            self.spike_buffer = defaultdict(list)

    def forward(self, x: torch.Tensor, state: tuple = None) -> tuple:
        batch_size, seq_len = x.size()
        
        # Initialize hidden states
        mem1 = self.lif1.init_leaky() if state is None else state[0]
        syn2, mem2 = self.lif2.init_synaptic() if state is None else state[1]
        
        outputs = []
        stdp_traces = []

        # Temporal processing loop
        for t in range(seq_len):
            x_embedded = self.embed(x[:, t])
            
            # Spike processing over multiple time steps
            for step in range(self.time_steps):
                cur1 = self.fc1(x_embedded)
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spk1)
                spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
                
                # Store spikes for STDP
                if self.use_stdp and self.training:
                    stdp_traces.append((spk1.detach(), spk2.detach()))
            
            outputs.append(spk2)

        return (
            torch.stack(outputs).permute(1, 0, 2),  # [batch, seq, vocab]
            stdp_traces,
            (mem1, (syn2, mem2))
        )

# === STDP Learning ============================================================
class STDPLearner:
    def __init__(self, model: SpikeDialogueModel):
        self.model = model
        self.amp = model.stdp_amp
        self.window = model.stdp_window
        
    def update(self, traces: list):
        if not traces or not self.model.use_stdp:
            return
        
        with torch.no_grad():
            # Apply STDP to fc2 weights
            for pre, post in traces[-self.window:]:
                dw = torch.outer(post.mean(0), pre.mean(0)) * self.amp
                self.model.fc2.weight.data += dw.T  # Fix dimension mismatch
                
            # Normalize weights
            self.model.fc2.weight.data = nn.functional.normalize(
                self.model.fc2.weight.data, p=2, dim=1
            )

# === Training Core ============================================================
class SNNTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = TrainingLogger(config["log_dir"])
        
        # Load data
        with open(Path(config["data_dir"])/"processed_pairs.json", "r") as f:
            self.pairs = json.load(f)
        self.word2idx = torch.load(Path(config["data_dir"])/"word2idx.pt")
        
        # Initialize model
        self.model = SpikeDialogueModel(
            len(self.word2idx),
            hidden_size=config["hidden_size"],
            time_steps=config["time_steps"],
            use_stdp=config["use_stdp"]
        ).to(self.device)
        
        # Optimization setup
        self.optimizer = optim.AdamW([
            {'params': self.model.embed.parameters(), 'lr': 3e-4},
            {'params': self.model.lif1.parameters(), 'lr': 1e-4},
            {'params': self.model.lif2.parameters(), 'lr': 5e-5},
            {'params': self.model.fc1.parameters(), 'lr': 3e-4},
            {'params': self.model.fc2.parameters(), 'lr': 3e-4},
        ])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.word2idx["<pad>"]
        )
        self.std_learner = STDPLearner(self.model) if config["use_stdp"] else None
        
    def process_batch(self, pair: tuple) -> tuple:
        context, response = pair
        max_len = self.config["max_seq_len"]
        
        # Process context (input)
        ctx_tokens = [self.word2idx["<start>"]] + [
            self.word2idx.get(w.lower(), self.word2idx["<unk>"]) 
            for w in context.split()][:max_len-1]
        ctx_tensor = torch.tensor(
            ctx_tokens + [self.word2idx["<pad>"]]*(max_len - len(ctx_tokens)),
            device=self.device
        )
        
        # Process response (target)
        resp_tokens = [self.word2idx["<start>"]] + [
            self.word2idx.get(w.lower(), self.word2idx["<unk>"]) 
            for w in response.split()][:max_len-2] + [self.word2idx["<end>"]]
        resp_tensor = torch.tensor(
            resp_tokens + [self.word2idx["<pad>"]]*(max_len - len(resp_tokens)),
            device=self.device
        )
        
        return ctx_tensor.unsqueeze(0), resp_tensor  # Add batch dim

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        spike_count = 0
        start_time = time.time()
        
        progress = tqdm(
            self.pairs[:self.config["train_size"]],
            desc=f"Epoch {epoch+1}/{self.config['epochs']}",
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
            postfix={"loss": "N/A", "spikes": "N/A", "lr": "N/A"}
        )
        
        for step, pair in enumerate(progress):
            # Prepare batch
            ctx, resp = self.process_batch(pair)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs, traces, _ = self.model(ctx)
            
            # STDP updates
            if self.std_learner and (step % STDP_UPDATE_INTERVAL == 0):
                self.std_learner.update(traces)
            
            # Loss calculation
            loss = self.criterion(
                outputs.view(-1, outputs.size(-1)),
                resp.view(-1)
            )
            
            # Backpropagation
            loss.backward()
            clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            spike_count += (outputs > 0).sum().item()
            
            # Update progress
            avg_loss = total_loss / (step + 1)
            spike_rate = spike_count / ((step + 1) * outputs.numel())
            progress.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "spikes": f"{spike_rate:.2%}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Logging
            if step % 100 == 0:
                self.logger.log_scalar("Train/Loss", avg_loss, epoch*len(progress)+step)
                self.logger.log_scalar("Train/SpikeRate", spike_rate, epoch*len(progress)+step)
                self.logger.log_histogram("Weights/fc1", self.model.fc1.weight, epoch)
                self.logger.log_histogram("Weights/fc2", self.model.fc2.weight, epoch)
        
        # Epoch summary
        epoch_time = time.time() - start_time
        self.logger.log_scalar("LR", self.optimizer.param_groups[0]['lr'], epoch)
        self.scheduler.step(avg_loss)
        
        return avg_loss, spike_rate, epoch_time

    def validate(self, epoch: int):
        self.model.eval()
        total_loss = 0.0
        start_time = time.time()
        
        with torch.no_grad():
            val_pairs = self.pairs[-self.config["val_size"]:]
            progress = tqdm(
                val_pairs,
                desc=f"Validating Epoch {epoch+1}",
                bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"
            )
            
            for pair in progress:
                ctx, resp = self.process_batch(pair)
                outputs, _, _ = self.model(ctx)
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)),
                    resp.view(-1)
                )
                total_loss += loss.item()
                progress.set_postfix({"loss": f"{total_loss/(progress.n+1):.4f}"})
        
        avg_loss = total_loss / len(val_pairs)
        self.logger.log_scalar("Val/Loss", avg_loss, epoch)
        return avg_loss

    def save_checkpoint(self, epoch: int, loss: float):
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "loss": loss,
            "config": self.config
        }
        save_path = Path(self.config["save_dir"])/f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, save_path)
        self.logger.console(f"Saved checkpoint: {save_path}")

    def run(self):
        best_loss = float('inf')
        start_time = time.time()
        
        try:
            for epoch in range(self.config["epochs"]):
                # Training phase
                train_loss, spike_rate, epoch_time = self.train_epoch(epoch)
                
                # Validation phase
                if (epoch + 1) % self.config["val_interval"] == 0:
                    val_loss = self.validate(epoch)
                    if val_loss < best_loss:
                        best_loss = val_loss
                        self.save_checkpoint(epoch, val_loss)
                
                # Checkpointing
                if (epoch + 1) % self.config["checkpoint_interval"] == 0:
                    self.save_checkpoint(epoch, train_loss)
                
                # Epoch report
                self.logger.console(
                    f"Epoch {epoch+1} | "
                    f"Loss: {train_loss:.4f} | "
                    f"Spikes: {spike_rate:.2%} | "
                    f"Time: {epoch_time:.1f}s"
                )
        
        except KeyboardInterrupt:
            self.logger.console("Training interrupted! Saving final state...")
            self.save_checkpoint(epoch, train_loss)
        
        finally:
            total_time = time.time() - start_time
            self.logger.console(
                f"\nTraining completed in {total_time/3600:.2f} hours\n"
                f"Best validation loss: {best_loss:.4f}"
            )

# === Main Execution ===========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SNN Dialogue Model")
    
    # Data configuration
    parser.add_argument("--data_dir", type=str, default="processed",
                       help="Directory containing processed data")
    parser.add_argument("--max_seq_len", type=int, default=50,
                       help="Maximum sequence length")
    
    # Model configuration
    parser.add_argument("--hidden_size", type=int, default=512,
                       help="Size of hidden layer")
    parser.add_argument("--time_steps", type=int, default=50,
                       help="Number of temporal steps per sequence element")
    parser.add_argument("--use_stdp", action="store_true",
                       help="Enable STDP learning")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--train_size", type=int, default=5000,
                       help="Number of training pairs per epoch")
    parser.add_argument("--val_size", type=int, default=1000,
                       help="Number of validation pairs")
    parser.add_argument("--val_interval", type=int, default=5,
                       help="Validate every N epochs")
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                       help="Save checkpoint every N epochs")
    
    # System configuration
    parser.add_argument("--log_dir", type=str, default="logs",
                       help="TensorBoard log directory")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                       help="Model checkpoint directory")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize and run training
    trainer = SNNTrainer(vars(args))
    trainer.run()
