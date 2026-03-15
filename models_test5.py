import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self):
        super(LSTMEncoder, self).__init__()
        self.lstm1 = nn.LSTM(input_size=35, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=3, num_layers=1, batch_first=True, bidirectional=True)

class MLPHead(nn.Module):
    def __init__(self):
        super(MLPHead, self).__init__()
        self.fc1 = nn.Linear(6, 429)  
        self.fc2 = nn.Linear(429, 71)
        self.fc3 = nn.Linear(71, 2)

encoder = LSTMEncoder()
head = MLPHead()
enc_p = sum(p.numel() for p in encoder.parameters())
head_p = sum(p.numel() for p in head.parameters())
print(f"LSTM: {enc_p}")
print(f"MLP: {head_p}")
print(f"Total: {enc_p + head_p}")
