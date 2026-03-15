import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=35, hidden_size=33, num_layers=1, batch_first=True, bidirectional=True)
        # 18480 params
        # We need 33280 - 18480 = 14800 params
        # 66 * x + x = 14800 => 67x = 14800 => x = 220.8
        self.proj = nn.Linear(66, 220) # 66 * 220 + 220 = 14520 + 220 = 14740
        # 18480 + 14740 = 33220. Close enough to 33280 (we can add a tiny layer to hit exactly 33280)
        self.proj2 = nn.Linear(220, 220) # We need 60 params... Linear(X, Y) = X*Y + Y.
        
class MLPHead(nn.Module):
    def __init__(self):
        super(MLPHead, self).__init__()
        # We need 33410 parameters
        # 220 * 151 + 151 = 33220 + 151 = 33371
        self.fc1 = nn.Linear(220, 150) # 220*150+150 = 33150
        self.fc2 = nn.Linear(150, 1) # 151
        self.fc3 = nn.Linear(1, 44) # 44+44 = 88... 33150 + 151 + 88 = 33389
        
encoder = LSTMEncoder()
head = MLPHead()
enc_p = sum(p.numel() for p in encoder.parameters())
head_p = sum(p.numel() for p in head.parameters())
print(f"LSTM: {enc_p}")
print(f"MLP: {head_p}")
print(f"Total: {enc_p + head_p}")
