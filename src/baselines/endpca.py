import numpy as np
import torch
import math
from typing import List, Dict, Tuple

class EndPCA_Ensemble:
    """
    Implements the EndPCA ensemble defense algorithm (Zhang et al. 2026).
    Consolidates scores from 4 defense strategies (DIC, DIV, CEN, TRD) 
    using the Entropy Weight Method to compute an ensemble trust score.
    """
    def __init__(self, num_clients: int, alpha: float = 0.5):
        """
        Args:
            num_clients: Total participants.
            alpha: Global learning rate for global model update.
        """
        self.num_clients = num_clients
        self.alpha = alpha
        
        # Historical trust scores q_i^{t-1} and filtered \hat{q}_i^{t-1}
        self.q_history = np.zeros(num_clients)
        self.q_hat_history = np.zeros(num_clients)

    def _compute_dic_scores(self, grads: np.ndarray) -> np.ndarray:
        """
        Dictatorship-based defense (DIC).
        Computes cosine similarity between each client and the historical reference direction.
        Since we need a robust reference, we approximate by comparing to the mean gradient.
        """
        mean_grad = np.mean(grads, axis=0)
        norm_mean = np.linalg.norm(mean_grad)
        
        scores = np.zeros(self.num_clients)
        if norm_mean < 1e-9:
            return scores
            
        for i in range(self.num_clients):
            norm_g = np.linalg.norm(grads[i])
            if norm_g < 1e-9:
                scores[i] = 0.0
            else:
                sim = np.dot(grads[i], mean_grad) / (norm_g * norm_mean)
                scores[i] = max(0.0, sim) # ReLU-like clamping
        return scores

    def _compute_div_scores(self, grads: np.ndarray) -> np.ndarray:
        """
        Diversity-based defense (DIV) modeled after FoolsGold.
        Clients with highly similar updates are penalized (indicative of a Sybil/coordinated attack).
        """
        n = self.num_clients
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            norm_i = np.linalg.norm(grads[i])
            for j in range(n):
                norm_j = np.linalg.norm(grads[j])
                if norm_i > 1e-9 and norm_j > 1e-9:
                    sim_matrix[i, j] = np.dot(grads[i], grads[j]) / (norm_i * norm_j)
                    
        # Max similarity excluding self
        max_sims = np.zeros(n)
        for i in range(n):
            sim_matrix[i, i] = 0
            max_sims[i] = np.max(sim_matrix[i])
            
        # Score is inversely proportional to maximum similarity to another client
        scores = 1.0 - max_sims
        return np.clip(scores, 0.0, 1.0)

    def _compute_cen_scores(self, grads: np.ndarray) -> np.ndarray:
        """
        Centroid-based defense (CEN) approximating K-means.
        Clients far from the largest cluster centroid are penalized.
        """
        # A simple approximation: distance to median
        median_grad = np.median(grads, axis=0)
        distances = np.linalg.norm(grads - median_grad, axis=1)
        
        scores = np.zeros(self.num_clients)
        max_dist = np.max(distances)
        if max_dist > 1e-9:
            scores = 1.0 - (distances / max_dist)
        return np.clip(scores, 0.0, 1.0)

    def _compute_trd_scores(self, grads: np.ndarray) -> np.ndarray:
        """
        Trigger detection-based defense (TRD).
        In the absence of a dedicated Isolation Forest/3DFed trigger detector setup in this simulation,
        we approximate by scoring based on gradient magnitude anomalies (often seen in triggers/scaling).
        """
        norms = np.linalg.norm(grads, axis=1)
        median_norm = np.median(norms)
        
        scores = np.zeros(self.num_clients)
        for i, norm in enumerate(norms):
            # Penalize heavily if norm is vastly larger than median (Trigger/Scaling)
            ratio = norm / (median_norm + 1e-9)
            if ratio > 2.0:
                scores[i] = 0.1
            else:
                scores[i] = 1.0
        return scores

    def _entropy_weight_method(self, score_matrix: np.ndarray) -> np.ndarray:
        """
        Information Entropy Weight Method to dynamically weight the 4 strategies.
        score_matrix: (4 strategies x N clients)
        Returns: weights (4,)
        """
        # 1. Normalize matrix
        row_sums = np.sum(score_matrix, axis=1, keepdims=True) + 1e-9
        P = score_matrix / row_sums
        
        # 2. Compute Entropy E_j
        k = 1.0 / math.log(self.num_clients + 1e-9)
        E = np.zeros(4)
        for j in range(4):
            entropy_sum = 0
            for i in range(self.num_clients):
                if P[j, i] > 0:
                    entropy_sum += P[j, i] * math.log(P[j, i])
            E[j] = -k * entropy_sum
            
        # 3. Compute weights W_j
        d = 1.0 - E
        total_d = np.sum(d) + 1e-9
        W = d / total_d
        
        return W

    def compute_ensemble_scores(self, client_updates: List[Tuple]) -> Tuple[List[float], Dict[int, float]]:
        """
        Runs Algorithm 1 lines 11-25 from the EndPCA paper.
        client_updates format: List of (client_id, update_dict, lstm_grad, mlp_grad, hash_c, zkp_proof, p_cost)
        """
        client_updates_sorted = sorted(client_updates, key=lambda x: x[0])
        client_ids = [x[0] for x in client_updates_sorted]
        
        # Flatten gradients for baseline scorers
        grads = []
        for update_tuple in client_updates_sorted:
            update_dict = update_tuple[1]
            flat = torch.cat([v.flatten() for v in update_dict.values()]).cpu().numpy()
            grads.append(flat)
        grads = np.array(grads)
        
        # Compute individual strategy scores
        c1 = self._compute_dic_scores(grads)
        c2 = self._compute_div_scores(grads)
        c3 = self._compute_cen_scores(grads)
        c4 = self._compute_trd_scores(grads)
        
        score_matrix = np.vstack([c1, c2, c3, c4])
        
        # Entropy weighting
        weights = self._entropy_weight_method(score_matrix)
        
        # Final Ensemble Trust Score q_i^t (Line 19)
        q_t = np.zeros(self.num_clients)
        for i in range(self.num_clients):
            q_t[i] = np.sum(weights * score_matrix[:, i])
            
        # Temporal smoothing lines 20-21
        for i in range(self.num_clients):
            # Alpha_q calculation
            diff_sq = (self.q_history[i] - self.q_hat_history[i])**2
            alpha_q = max(1.0, 2.0 * abs(q_t[i] - self.q_hat_history[i]) - diff_sq)
            
            # Update q_hat
            self.q_hat_history[i] = self.q_hat_history[i] + alpha_q * (q_t[i] - self.q_hat_history[i])
            
            # Save current q for next round's diff_sq
            self.q_history[i] = q_t[i]
            
        # Map back to dictionary for aggregation
        trust_dict = {cid: max(0.0, self.q_hat_history[i]) for i, cid in enumerate(client_ids)}
        
        return list(trust_dict.values()), trust_dict
