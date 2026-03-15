import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=35, hidden_dim=44, num_layers=2):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            bidirectional=True)
        # bidirectional 44 -> output is 88
        
class MLPHead(nn.Module):
    def __init__(self, in_features=88, hidden_features=260, out_features=2, dropout_p=0.4):
        super(MLPHead, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(p=dropout_p)

# We want roughly 33,280 and 33,410 parameters
encoder = LSTMEncoder()
head = MLPHead()
enc_p = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
head_p = sum(p.numel() for p in head.parameters() if p.requires_grad)
print(f"LSTM Encoder: {enc_p}")
print(f"MLP Head: {head_p}")
print(f"Total: {enc_p + head_p}")
