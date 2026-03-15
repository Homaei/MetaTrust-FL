import torch
from models import MetaTrustModel, FocalLoss
from zkp_utils import PoseidonHashSimulator, Groth16Simulator
import copy

# ==============================================================================
# client.py
# 
# 1. HospitalClient: A class for managing data and local training for each hospital.
# 2. Uses the AdamW optimizer and Focal Loss for local training.
# 3. Generates a Poseidon hash for LSTM outputs and ZK proofs for MLP outputs.
# 4. Applies gradient clipping for Differential Privacy (C=1.0).
# ==============================================================================

class HospitalClient:
    def __init__(self, client_id, train_loader, test_loader, device='cpu'):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # Local model
        self.model = MetaTrustModel().to(self.device)
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
        # Optimizer from the paper
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
    def sync_with_server(self, global_model_state_dict):
        """
        Synchronizes the local weights with the aggregated weights from the server.
        """
        self.model.load_state_dict(global_model_state_dict)
        
    def local_training(self, local_epochs=1, clip_bound=1.0):
        """
        Runs local training and returns the gradients (weight diffs) after clipping.
        According to Equation 12: Delta_clip = Delta * min(1, C / ||Delta||_2)
        """
        initial_state = copy.deepcopy(self.model.state_dict())
        self.model.train()
        
        for epoch in range(local_epochs):
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(X)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()
                
        # Calculate Delta (Update vector)
        update_vector = {}
        for name, param in self.model.state_dict().items():
            update_vector[name] = param - initial_state[name]
            
        # Extract parameter sets for analysis
        lstm_params = torch.cat([update_vector[k].flatten() for k in update_vector.keys() if 'encoder' in k])
        mlp_params = torch.cat([update_vector[k].flatten() for k in update_vector.keys() if 'head' in k])
            
        # Differential Privacy Clipping (Server normally does this unconditionally 
        # but clients simulate it locally for proof constraints eq 2)
        l2_norm = torch.norm(torch.cat((lstm_params, mlp_params)))
        clip_factor = min(1.0, clip_bound / (l2_norm.item() + 1e-6))
        
        clipped_update = {}
        for k in update_vector.keys():
            clipped_update[k] = update_vector[k] * clip_factor
            
        lstm_clipped = torch.cat([clipped_update[k].flatten() for k in clipped_update.keys() if 'encoder' in k])
        mlp_clipped = torch.cat([clipped_update[k].flatten() for k in clipped_update.keys() if 'head' in k])

        return clipped_update, lstm_clipped.cpu().numpy(), mlp_clipped.cpu().numpy()

    def generate_proofs(self, lstm_grad, mlp_grad, verification_level):
        """
        Based on the paper's defense architecture:
        1. Poseidon hash for the LSTM section (for Tamper-evidence).
        2. ZK-SNARK proof for the MLP section.
        """
        # Hash commitment for LSTM (Tamper-evidence)
        hash_commitment = PoseidonHashSimulator.commit(lstm_grad)
        
        # ZKP generation for MLP head directly relies on assigned verification level
        zkp_proof, proof_cost_time = Groth16Simulator.prove(mlp_grad, zk_level=verification_level)
        
        return hash_commitment, zkp_proof, proof_cost_time

    def evaluate(self):
        """
        Evaluates the local model on the hospital's test dataset.
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        all_targets = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                loss = self.criterion(logits, y)
                
                total_loss += loss.item() * X.size(0)
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                
                correct += (preds == y).sum().item()
                
                all_targets.extend(y.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probabilities for class 1
                
        acc = correct / len(self.test_loader.dataset)
        return total_loss / len(self.test_loader.dataset), acc, all_targets, all_preds, all_probs
