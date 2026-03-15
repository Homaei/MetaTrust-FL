import numpy as np
import torch

# ==============================================================================
# attacks.py
# 
# 1. Random Gradient Attack
# 2. Sign-Flipping Attack
# 3. Scaling Attack (kappa = 10)
# 4. Null-Space Attack (Adaptive)
# ==============================================================================

class ByzantineAttacker:
    @staticmethod
    def random_gradient(update_dict, magnitude=1.0):
        """
        Replaces gradients with Gaussian noise. (Random Gradient Attack)
        """
        poisoned_update = {}
        for k, v in update_dict.items():
            poisoned_update[k] = torch.randn_like(v) * magnitude
        return poisoned_update

    @staticmethod
    def sign_flipping(update_dict):
        """
        Inverts the gradient direction to sabotage the model.
        """
        poisoned_update = {}
        for k, v in update_dict.items():
            poisoned_update[k] = -v
        return poisoned_update

    @staticmethod
    def scaling_attack(update_dict, kappa=10.0):
        """
        Magnifies the gradient size to dominate the aggregation step (Scaling Attack).
        """
        poisoned_update = {}
        for k, v in update_dict.items():
            poisoned_update[k] = v * kappa
        return poisoned_update

    @staticmethod
    def null_space_attack(update_dict, pca_basis, alpha=0.5):
        """
        A sophisticated attack targeting the PCA null-space.
        This attack actively attempts to slip past the PCA-based Anomaly Detector.
        pca_basis: The U matrix corresponding to the PCA.
        """
        # A true null-space attack computes a vector orthogonal to all columns of U.
        # Here we approximate a simple orthogonal projection manipulation.
        # delta_adv = delta_honest + alpha * v_perp
        
        # Flatten the honest update
        flat_honest = torch.cat([v.flatten() for v in update_dict.values()]).cpu().numpy()
        
        # Approximate a random vector
        random_v = np.random.randn(*flat_honest.shape)
        
        # Project random vector onto U
        proj = pca_basis @ (pca_basis.T @ random_v.reshape(-1, 1))
        
        # Compute orthogonal component: v - proj
        v_perp = random_v - proj.flatten()
        
        # Normalize v_perp to scale with alpha
        if np.linalg.norm(v_perp) > 1e-6:
            v_perp = (v_perp / np.linalg.norm(v_perp)) * alpha * np.linalg.norm(flat_honest)
        
        flat_poisoned = flat_honest + v_perp
        
        # Reconstruct state dict
        poisoned_update = {}
        idx = 0
        for k, v in update_dict.items():
            numel = v.numel()
            shape = v.shape
            chunk = flat_poisoned[idx:idx+numel]
            poisoned_update[k] = torch.tensor(chunk, dtype=torch.float32).reshape(shape).to(v.device)
            idx += numel
            
        return poisoned_update

def extract_flat_arrays(update_dict):
    """
    Splits the flattened gradients (numpy arrays) back into LSTM and MLP sections for the simulators.
    """
    lstm_clipped = torch.cat([update_dict[k].flatten() for k in update_dict.keys() if 'encoder' in k])
    mlp_clipped = torch.cat([update_dict[k].flatten() for k in update_dict.keys() if 'head' in k])
    
    return lstm_clipped.cpu().numpy(), mlp_clipped.cpu().numpy()
