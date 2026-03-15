import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=35, hidden_dim=27, num_layers=2):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            bidirectional=True)
                            
class MLPHead(nn.Module):
    def __init__(self):
        super(MLPHead, self).__init__()
        self.fc1 = nn.Linear(54, 442)  
        self.fc2 = nn.Linear(442, 21)
        self.fc3 = nn.Linear(21, 2)
        
encoder = LSTMEncoder()
head = MLPHead()
enc_p = sum(p.numel() for p in encoder.parameters())
head_p = sum(p.numel() for p in head.parameters())
print(f"LSTM: {enc_p}")
print(f"MLP: {head_p}")
print(f"Total: {enc_p + head_p}")
