import numpy as np
import torch
import copy
from typing import List, Dict, Any, Tuple

class MAR_ALS:
    """
    Implements the Matrix Autoregressive Model MAR(1) solved via Alternating Least Squares (ALS)
    as presented in FLANDERS: "Securing Federated Learning Against Extreme Model Poisoning Attacks 
    via Multidimensional Time Series Anomaly Detection on Local Updates" (Gabrielli et al.).
    
    Formula: Theta_t = A * Theta_{t-1} * B + E_t
    - Theta is a matrix of size d x m (d = parameters, m = clients).
    - A is a d x d coefficient matrix capturing row-wise dependencies.
    - B is an m x m coefficient matrix capturing column-wise dependencies.
    """
    def __init__(self, d: int, m: int, max_iter: int = 100, tol: float = 1e-4):
        """
        Args:
            d: Reduced parameter dimension (e.g., 500 for tractability)
            m: Number of clients
            max_iter: Maximum ALS iterations
            tol: Convergence tolerance
        """
        self.d = d
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        
        # Initialize coefficients A and B randomly
        self.A = np.random.randn(d, d) * 0.01
        self.B = np.random.randn(m, m) * 0.01

    def fit(self, history: List[np.ndarray]):
        """
        Fits the MAR(1) model to the sliding window of historical observations.
        history: A list of length `l` containing Theta matrices of shape (d, m).
                 Ordered from oldest to newest. Theta_{t} is predicted by Theta_{t-1}.
        """
        if len(history) < 2:
            return # Need at least two matrices to form a transition t-1 -> t
            
        l = len(history)
        
        for iteration in range(self.max_iter):
            # 1. Update A
            numerator_A = np.zeros((self.d, self.d))
            denominator_A = np.zeros((self.d, self.d))
            for j in range(l - 1):
                Theta_curr = history[j + 1] # Theta_{t-j} 
                Theta_prev = history[j]     # Theta_{t-j-1}
                
                term1 = Theta_curr @ self.B.T @ Theta_prev.T
                term2 = Theta_prev @ self.B @ self.B.T @ Theta_prev.T
                
                numerator_A += term1
                denominator_A += term2
            
            # Add small identity to denominator for numerical stability (Ridge-like)
            denominator_A += np.eye(self.d) * 1e-6
            A_new = numerator_A @ np.linalg.inv(denominator_A)
            
            # 2. Update B
            numerator_B = np.zeros((self.m, self.m))
            denominator_B = np.zeros((self.m, self.m))
            for j in range(l - 1):
                Theta_curr = history[j + 1]
                Theta_prev = history[j]
                
                term1 = Theta_prev.T @ A_new.T @ Theta_curr
                term2 = Theta_prev.T @ A_new.T @ A_new @ Theta_prev
                
                numerator_B += term1
                denominator_B += term2
                
            denominator_B += np.eye(self.m) * 1e-6
            B_new = np.linalg.inv(denominator_B) @ numerator_B
            
            # Check convergence
            diff_A = np.linalg.norm(A_new - self.A, ord='fro')
            diff_B = np.linalg.norm(B_new - self.B, ord='fro')
            
            self.A = A_new
            self.B = B_new
            
            if diff_A < self.tol and diff_B < self.tol:
                break
                
    def predict(self, Theta_prev: np.ndarray) -> np.ndarray:
        """
        Predicts Theta_t given Theta_{t-1} using learned A and B.
        Formula: \hat{Theta}_t = A * Theta_{t-1} * B
        """
        return self.A @ Theta_prev @ self.B


class FlandersFilter:
    """
    Main FLANDERS server-side pre-aggregation filter.
    Maintains a sliding window of recent local models and dynamically identifies outliers.
    """
    def __init__(self, num_clients: int, window_size: int = 5, subset_size: int = 500, k_trusted: int = None):
        """
        Args:
            num_clients: Total participating clients in the FL round (m).
            window_size: Size of sliding window `l` (e.g., last 5 rounds).
            subset_size: The number of dimension `d` to sample for computational feasibility. Default 500.
            k_trusted: Number of clients to accept. If None, uses a dynamic distance-based threshold.
        """
        self.num_clients = num_clients
        self.window_size = window_size
        self.subset_size = subset_size
        self.k_trusted = k_trusted
        
        self.history_matrices = [] # Sliding window of Theta matrices
        self.sampled_indices = None # The specific d dimensions we sample across all rounds
        
        self.mar_model = MAR_ALS(d=subset_size, m=num_clients)

    def extract_subset_matrix(self, client_updates: List[Tuple]) -> np.ndarray:
        """
        Extracts a d x m matrix from the raw client state dicts.
        If self.sampled_indices is None, it initializes the random subset.
        client_updates format: List of (client_id, update_dict, lstm_grad, mlp_grad, hash_c, zkp_proof, p_cost)
        """
        # Flatten the first client's update to get total parameters
        first_client_dict = client_updates[0][1]
        flat_sizes = [v.numel() for v in first_client_dict.values()]
        total_params = sum(flat_sizes)
        
        if self.sampled_indices is None:
            # Sample `subset_size` parameters uniformly without replacement
            self.sampled_indices = np.random.choice(total_params, min(self.subset_size, total_params), replace=False)
            self.subset_size = len(self.sampled_indices) # Adjust if model is smaller than 500
            self.mar_model = MAR_ALS(d=self.subset_size, m=self.num_clients)
            
        m = self.num_clients
        d = self.subset_size
        Theta = np.zeros((d, m))
        
        # Sort updates by client ID implicitly or explicitly
        client_updates_sorted = sorted(client_updates, key=lambda x: x[0])
        client_ids_sorted = [x[0] for x in client_updates_sorted]
        
        for col_idx, update_tuple in enumerate(client_updates_sorted):
            update_dict = update_tuple[1]
            # Flatten dict to 1D tensor
            flat_update = torch.cat([v.flatten() for v in update_dict.values()]).cpu().numpy()
            Theta[:, col_idx] = flat_update[self.sampled_indices]
            
        return Theta, client_ids_sorted

    def filter_updates(self, client_updates: List[Tuple], global_model_state: Dict[str, torch.Tensor]) -> Tuple[List[int], Dict[int, float]]:
        """
        Executes the FLANDERS algorithm for the current round.
        Returns:
            trusted_client_ids: List of client IDs whose updates are accepted.
            anomaly_scores: Dictionary mapping client ID to its calculated distance score.
        """
        Theta_t, client_ids = self.extract_subset_matrix(client_updates)
        
        if len(self.history_matrices) < 1:
            # Round 1 (t=1): No history to build MAR. Pass through all clients (or use robust aggregation backup)
            self.history_matrices.append(Theta_t)
            return client_ids, {c: 0.0 for c in client_ids}
            
        # Predict current matrix using the MAR(1) model based on previous round's Theta
        # At round t=2, the MAR won't be fitted yet, but we use the initialized small A,B
        # or we just rely on the first fit that happens next.
        Theta_pred = self.mar_model.predict(self.history_matrices[-1])
        
        # Calculate Anomaly Scores (L2 distance squared)
        anomaly_scores = {}
        for col_idx, client_id in enumerate(client_ids):
            actual_col = Theta_t[:, col_idx]
            pred_col = Theta_pred[:, col_idx]
            dist = np.linalg.norm(actual_col - pred_col, ord=2) ** 2
            anomaly_scores[client_id] = float(dist)
            
        # Determine Trusted Clients
        trusted_client_ids = []
        if self.k_trusted is not None:
            # Sort by ascending anomaly score and pick top K
            sorted_clients = sorted(anomaly_scores.keys(), key=lambda c: anomaly_scores[c])
            trusted_client_ids = sorted_clients[:self.k_trusted]
        else:
            # If no K provided, pick dynamic threshold (e.g. 1.5 * IQR above median)
            # Or use a strict median filter for simplicity
            scores_array = np.array(list(anomaly_scores.values()))
            median_score = np.median(scores_array)
            mad = np.median(np.abs(scores_array - median_score))
            threshold = median_score + 3.0 * mad # 3 MADs threshold
            
            for cid, score in anomaly_scores.items():
                if score <= threshold:
                    trusted_client_ids.append(cid)
                    
        # --- Update MAR Model with sanitized history ---
        # The paper specifies replacing malicious local models with the global model
        # so MAR doesn't get corrupted by malicious trajectories.
        Theta_sanitized = Theta_t.copy()
        flat_global = torch.cat([v.flatten() for v in global_model_state.values()]).cpu().numpy()
        sampled_global = flat_global[self.sampled_indices]
        
        for col_idx, client_id in enumerate(client_ids):
            if client_id not in trusted_client_ids:
                Theta_sanitized[:, col_idx] = sampled_global
                
        self.history_matrices.append(Theta_sanitized)
        if len(self.history_matrices) > self.window_size:
            self.history_matrices.pop(0)
            
        # Retrain MAR model on the newly appended window history
        if len(self.history_matrices) >= 2:
            self.mar_model.fit(self.history_matrices)
            
        return trusted_client_ids, anomaly_scores
