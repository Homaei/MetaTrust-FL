import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=35, hidden_size=33, num_layers=1, batch_first=True, bidirectional=True)
        # 35 * 33 * 4 + 33 * 33 * 4 + 33 * 8 = 4620 + 4356 + 264 = 9240
        # For bidirectional: 2 * (4620 + 4356 + 264) = 18480
        # To get 33280 we need a second layer perhaps, or just a linear projection.
        self.proj = nn.Linear(66, 226) # 66 * 226 + 226 = 14916 + 226 = 15142
        # 18480 + 15142 = 33622
        
class MLPHead(nn.Module):
    def __init__(self):
        super(MLPHead, self).__init__()
        self.fc1 = nn.Linear(226, 146)  # 226 * 146 + 146 = 32996 + 146 = 33142
        self.fc2 = nn.Linear(146, 1)    # 146 * 1 + 1 = 147
        self.fc3 = nn.Linear(1, 120)    # 120 + 120 = 240 // 33142 + 240 = 33382

encoder = LSTMEncoder()
head = MLPHead()
enc_p = sum(p.numel() for p in encoder.parameters())
head_p = sum(p.numel() for p in head.parameters())
print(f"LSTM: {enc_p}")
print(f"MLP: {head_p}")
print(f"Total: {enc_p + head_p}")
