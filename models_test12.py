import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=35, hidden_size=48, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(96, 6)
        self.fc2 = nn.Linear(6, 10)
        
encoder = LSTMEncoder()
print("LSTM:", sum(p.numel() for p in encoder.parameters()))
