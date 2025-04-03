import torch
import random
from train import DialogueSNN

class ChatBot:
    def __init__(self):
        # Load vocabulary and model
        self.word2idx = torch.load("data/word2idx.pt")
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        
        # Initialize model
        self.model = DialogueSNN(len(self.word2idx))
        
        # Load checkpoint if available
        if os.path.exists("checkpoint.pt"):
            checkpoint = torch.load("checkpoint.pt", map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
            print(f"Loaded model trained for {checkpoint['epoch']+1} epochs")
        else:
            raise FileNotFoundError("No checkpoint found - train the model first")
        
        self.model.eval()
    
    def _encode(self, text):
        return [self.word2idx["<start>"]] + \
               [self.word2idx.get(word.lower(), self.word2idx["<unk>"]) 
                for word in text.split()] + \
               [self.word2idx["<end>"]]
    
    def _decode(self, indices):
        return " ".join([self.idx2word.get(idx, "<unk>") for idx in indices])
    
    def generate_response(self, input_text, max_length=20, temperature=0.7):
        with torch.no_grad():
            # Encode input
            input_seq = self._encode(input_text)
            input_tensor = torch.tensor(input_seq, dtype=torch.long)
            
            # Generate response
            output = []
            for _ in range(max_length):
                logits = self.model(input_tensor.unsqueeze(0))[0]
                
                # Apply temperature scaling
                probs = torch.softmax(logits / temperature, dim=-1)
                next_idx = torch.multinomial(probs, 1).item()
                
                if next_idx == self.word2idx["<end>"]:
                    break
                
                output.append(next_idx)
                input_tensor = torch.cat([input_tensor, torch.tensor([next_idx])])
            
            return self._decode(output)

if __name__ == "__main__":
    try:
        bot = ChatBot()
        print("Movie Dialogue SNN Chatbot (type 'quit' to exit)")
        print("-----------------------------------------------")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            
            response = bot.generate_response(user_input)
            print(f"Bot: {response}\n")
    
    except Exception as e:
        print(f"Error: {str(e)}")
