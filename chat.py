import torch
import os
import random
from train import DialogueSNN

class ChatBot:
    def __init__(self, max_length=25, temp=0.7):
        # Load vocabulary
        self.word2idx = torch.load("data/word2idx.pt")
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.max_length = max_length
        self.temperature = temp
        
        # Initialize model
        self.model = DialogueSNN(len(self.word2idx))
        
        # Load trained weights
        if os.path.exists("checkpoint.pt"):
            checkpoint = torch.load("checkpoint.pt", map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
            print(f"🤖 Loaded model (trained for {checkpoint['epoch']+1} epochs)")
        else:
            raise FileNotFoundError("Model checkpoint not found")
        
        self.model.eval()
    
    def _encode(self, text):
        return [self.word2idx["<start>"]] + \
               [self.word2idx.get(word.lower(), self.word2idx["<unk>"]) 
                for word in text.split()] + \
               [self.word2idx["<end>"]]
    
    def _decode(self, indices):
        return " ".join(
            self.idx2word.get(idx, "<unk>") 
            for idx in indices 
            if idx not in {self.word2idx["<start>"], self.word2idx["<end>"]}
        )
    
    def generate_response(self, input_text):
        with torch.no_grad():
            # Encode input
            input_seq = self._encode(input_text)
            input_tensor = torch.tensor(input_seq).unsqueeze(0)
            
            # Initialize state
            mem1 = self.model.lif1.init_leaky()
            mem2 = self.model.lif2.init_leaky()
            output = []
            
            for _ in range(self.max_length):
                # Forward pass with current state
                logits, mem1, mem2 = self.model(
                    input_tensor[:, -1:],  # Only last token
                    mem1,
                    mem2
                )
                
                # Sample from probabilities
                probs = torch.softmax(logits[0, -1] / self.temperature, -1)
                next_idx = torch.multinomial(probs, 1).item()
                
                # Stop condition
                if next_idx == self.word2idx["<end>"]:
                    break
                
                output.append(next_idx)
                input_tensor = torch.cat([
                    input_tensor,
                    torch.tensor([[next_idx]], dtype=torch.long)
                ], dim=1)
            
            return self._decode(output)

if __name__ == "__main__":
    try:
        bot = ChatBot()
        print("🎬 Movie Dialogue SNN Chatbot")
        print("Type 'exit' to quit\n" + "-"*40)
        
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break
            
            if not user_input:
                print("Bot: Please say something!")
                continue
            
            response = bot.generate_response(user_input)
            print(f"Bot: {response}\n")
            
    except Exception as e:
        print(f"\n⚠️  Error: {str(e)}")
