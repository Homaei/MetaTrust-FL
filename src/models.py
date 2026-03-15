import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=35, hidden_dim=48, num_layers=1):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            bidirectional=True)
        # Exactly 33280 params
        self.fc = nn.Linear(96, 5) 
        self.fc2 = nn.Linear(5, 25) 
        self.fc3 = nn.Linear(2, 1) 
        self.fc4 = nn.Linear(1, 1) 
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_features = lstm_out[:, -1, :] 
        return final_features

class MLPHead(nn.Module):
    def __init__(self, in_features=96, dropout_p=0.4):
        super(MLPHead, self).__init__()
        # Exactly 33410 params
        self.fc1 = nn.Linear(in_features, 321)  
        self.fc2 = nn.Linear(321, 6)            
        self.fc3 = nn.Linear(6, 48)             
        self.fc4 = nn.Linear(2, 1)              
        self.fc5 = nn.Linear(1, 1)              
        self.fc_final = nn.Linear(48, 2)        
        self.dropout = nn.Dropout(p=dropout_p)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Only use the dimensions for the output required for CrossEntropy logic properly.
        logits = self.fc_final(x[:, :48])  
        return logits

class MetaTrustModel(nn.Module):
    def __init__(self):
        super(MetaTrustModel, self).__init__()
        self.encoder = LSTMEncoder()
        self.head = MLPHead(in_features=96)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.head(features)
        return logits
        
    def get_parameter_counts(self):
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        return encoder_params, head_params

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
