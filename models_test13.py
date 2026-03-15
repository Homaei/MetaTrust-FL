import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=35, hidden_size=48, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(96, 5) # 485
        self.fc2 = nn.Linear(5, 25) # 150 (Total 32640 + 485 + 150 = 33275)
        self.fc3 = nn.Linear(2, 1) # 3
        self.fc4 = nn.Linear(1, 1) # 2 (Total = 33280 exactly)
        
encoder = LSTMEncoder()
print("LSTM:", sum(p.numel() for p in encoder.parameters()))
