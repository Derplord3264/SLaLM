#!/usr/bin/env python3
"""
SNN Chat Interface with Verbose Logging
"""

import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional
from train import SpikeDialogueModel  # Reuse model definition

# === Logging Setup ============================================================

def log(message: str, level: str = "INFO"):
    """Custom logging function with level prefixes"""
    print(f"[{level}] {message}")

# === Chat Agent Core ==========================================================

class VerboseChatAgent:
    """Production chat agent with detailed logging and state tracking"""
    
    def __init__(self, 
                 model_path: str,
                 data_dir: str = "processed",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        # === Initialization Logging ===
        log("Initializing chat agent")
        log(f"Model path: {model_path}")
        log(f"Data directory: {data_dir}")
        log(f"Using device: {device}")
        
        # === Vocabulary Loading ===
        log("Loading vocabulary...")
        vocab_path = Path(data_dir)/"word2idx.pt"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
        
        self.word2idx = torch.load(vocab_path)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        log(f"Loaded vocabulary ({len(self.word2idx)} words)")
        
        # === Model Loading ===
        log("Initializing model architecture...")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device)
        self.config = checkpoint.get('config', {})
        log(f"Model config: {self.config}")
        
        self.model = SpikeDialogueModel(
            vocab_size=len(self.word2idx),
            hidden_size=self.config.get('hidden_size', 512),
            time_steps=self.config.get('time_steps', 50),
            use_stdp=self.config.get('use_stdp', False)
        ).to(device)
        
        log("Loading model weights...")
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        log("Model initialized successfully")
        
        # === State Management ===
        self.device = torch.device(device)
        self.reset_state()
        log("Chat agent ready")

    def reset_state(self):
        """Reset neural conversation state with logging"""
        log("Resetting neural state")
        self.mem1 = self.model.lif1.init_leaky()
        self.syn2, self.mem2 = self.model.lif2.init_synaptic()
        log("State reset complete")

    # === Encoding/Decoding ====================================================

    def encode(self, text: str, max_len: int = 50) -> torch.Tensor:
        """Convert text to tensor with detailed logging"""
        log(f"Encoding input: {text[:60]}{'...' if len(text) >60 else ''}")
        
        tokens = [self.word2idx["<start>"]] + [
            self.word2idx.get(w.lower(), self.word2idx["<unk>"]) 
            for w in text.split()][:max_len-1]
        
        log(f"Encoded {len(tokens)} tokens")
        return torch.tensor(tokens, device=self.device).unsqueeze(0)

    def decode(self, indices: List[int]) -> str:
        """Convert indices to text with filtering"""
        log(f"Decoding {len(indices)} tokens")
        filtered = [
            self.idx2word[idx] for idx in indices 
            if idx not in {self.word2idx["<start>"], self.word2idx["<end>"]}
        ]
        return ' '.join(filtered)

    # === Generation Core ======================================================

    def generate_response(self,
                         input_text: str,
                         max_length: int = 100,
                         temperature: float = 0.7,
                         top_k: Optional[int] = 40) -> str:
        """Generate response with full progress tracking"""
        
        # === Initialization ===
        log("Starting generation process")
        log(f"Max length: {max_length} | Temperature: {temperature} | Top-K: {top_k}")
        
        input_tensor = self.encode(input_text)
        output_ids = []
        
        # === Generation Loop ===
        progress = tqdm(
            range(max_length),
            desc="Generating tokens",
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
            disable=False  # Always show progress
        )
        
        for step in progress:
            with torch.no_grad():
                # === Neural Forward Pass ===
                log(f"Processing step {step+1}", level="DEBUG")
                logits, _, state = self.model(
                    input_tensor[:, -1:],
                    state=(self.mem1, (self.syn2, self.mem2))
                )
                self.mem1, (self.syn2, self.mem2) = state
                
                # === Sampling Preparation ===
                logits = logits[0, -1] / temperature
                if top_k is not None:
                    top_values = torch.topk(logits, top_k).values
                    logits[logits < top_values[:, -1]] = -float('inf')
                    log(f"Top-k filtering kept {top_k} candidates", level="DEBUG")
                
                # === Token Selection ===
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
                log(f"Selected token ID: {next_id}", level="DEBUG")
                
                # === Stopping Condition ===
                if next_id == self.word2idx["<end>"]:
                    log("End token detected, stopping generation")
                    break
                
                output_ids.append(next_id)
                input_tensor = torch.cat([
                    input_tensor,
                    torch.tensor([[next_id]], device=self.device)
                ], dim=1)
                
                # Update progress bar
                progress.set_postfix({
                    "current_token": self.idx2word.get(next_id, "<unk>"),
                    "length": len(output_ids)
                })
        
        # === Finalization ===
        log(f"Generated {len(output_ids)} tokens")
        return self.decode(output_ids)

# === Main Interface ===========================================================

def main():
    print("\n" + "="*60)
    print("SNN CHAT INTERFACE - VERBOSE MODE")
    print("="*60)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--temp", type=float, default=0.7,
                       help="Sampling temperature (0.1-2.0)")
    parser.add_argument("--top_k", type=int, default=40,
                       help="Top-k filtering (0 to disable)")
    args = parser.parse_args()
    
    try:
        # === Agent Initialization ===
        log("Starting chat session")
        agent = VerboseChatAgent(args.model)
        
        # === Session Loop ===
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue
                
                # === Special Commands ===
                if user_input.lower() == 'exit':
                    log("Exit command received")
                    break
                if user_input.lower() == 'reset':
                    log("Reset command received")
                    agent.reset_state()
                    print("[SYSTEM] Conversation history cleared")
                    continue
                
                # === Generation ===
                print("")  # Add spacing before bot response
                response = agent.generate_response(
                    user_input,
                    temperature=args.temp,
                    top_k=args.top_k if args.top_k > 0 else None
                )
                
                # === Display Response ===
                print(f"\nBot: {response}")
                
            except KeyboardInterrupt:
                log("Keyboard interrupt received")
                break
            except Exception as e:
                log(f"Error: {str(e)}", level="ERROR")
                agent.reset_state()
                
    finally:
        log("Closing chat session")

if __name__ == "__main__":
    main()