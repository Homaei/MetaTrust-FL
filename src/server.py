import torch
import numpy as np
from models import MetaTrustModel
from anomaly_detector import PCAMahalanobisDetector
from policy import TrustDynamics, REINFORCEAgent
from zkp_utils import Groth16Simulator

# New imports from the instruction
from client import HospitalClient
from zkp_utils import PoseidonHashSimulator
from baselines.flanders import FlandersFilter
from baselines.endpca import EndPCA_Ensemble
import copy
import time

# ==============================================================================
# server.py
# 
# 1. MetaTrustServer: Manages the server, aggregates weights, and tracks the trust system.
# 2. Implements central Differential Privacy with Gaussian noise (sigma = 0.8).
# 3. Computes the state vector for the REINFORCE agent using anomalies and trust scores.
# ==============================================================================

class MetaTrustServer:
    def __init__(self, num_clients=5, global_lr=1.0, defense_type="metatrust", dp_sigma=0.8, clip_bound=1.0, device='cpu'):
        """
        Initializes the server with the specified defense mechanism:
        - 'metatrust': The paper's core adaptive ZKP method
        - 'flanders': The FLANDERS MAR(1) baseline
        - 'endpca': The EndPCA Ensemble baseline
        """
        self.num_clients = num_clients
        self.global_lr = global_lr
        self.defense_type = defense_type.lower()
        self.dp_sigma = dp_sigma
        self.clip_bound = clip_bound
        self.device = device
        
        self.global_model = MetaTrustModel().to(self.device)
        self.global_model_state = self.global_model.state_dict() # Initialize global_model_state
        
        # --- MetaTrust Core Components ---
        self.policy_agent = REINFORCEAgent(state_dim=5, lr=0.005)
        self.trust_system = TrustDynamics(num_clients=num_clients, t_min=0.1, gamma=0.95)
        
        # Calculate exactly how many parameters are trainable from the flattening logic
        dummy_counts = self.global_model.get_parameter_counts()
        exact_params = dummy_counts[0] + dummy_counts[1]
        
        self.anomaly_detector = PCAMahalanobisDetector(n_params=exact_params, d=7, window_size=10)
        self.accumulated_trajectories = [] # For Meta-training EQ 5
        
        # --- Baseline Components ---
        self.flanders_filter = FlandersFilter(num_clients=num_clients, window_size=5, k_trusted=int(0.6 * num_clients))
        self.endpca_ensemble = EndPCA_Ensemble(num_clients=num_clients)
        
    def get_global_weights(self):
        return self.global_model.state_dict()
        
    def add_dp_noise(self, aggregated_update):
        """
        Adds Gaussian differential noise to preserve privacy (Differential Privacy).
        The paper uses the Central DP mechanism (per Section 4.8).
        Noise is scaled by sensitivity (C) and sigma.
        """
        noisy_update = {}
        # Since gradients are already clipped (L2 norm <= 1.0)
        # the sensitivity Delta_f = C = 1.0
        scale = self.dp_sigma * self.clip_bound
        
        for k, v in aggregated_update.items():
            noise = torch.normal(mean=0.0, std=scale, size=v.size()).to(self.device)
            noisy_update[k] = v + noise
            
        return noisy_update

    def process_and_aggregate(self, client_updates, round_num, total_rounds, evaluate_byzantine_flags=None):
        """
        The core aggregation algorithm routed by the chosen defense strategy.
        If 'metatrust': Runs Algorithm 2 (ZKP Adaptive proofs, RL policy).
        If 'flanders': Runs MAR(1) anomaly scorer.
        If 'endpca': Runs EndPCA entropy-based ensemble.
        """
        if self.defense_type == "flanders":
            return self._aggregate_flanders(client_updates)
        elif self.defense_type == "endpca":
            return self._aggregate_endpca(client_updates)
        else: # Default to metatrust
            return self._aggregate_metatrust(client_updates, round_num, total_rounds, evaluate_byzantine_flags)
            
    def _aggregate_flanders(self, client_updates):
        """
        Baseline 1: FLANDERS
        """
        print("[FLANDERS] Running MAR(1) Pre-Aggregation Filter...")
        trusted_client_ids, anomaly_scores = self.flanders_filter.filter_updates(client_updates, self.global_model_state)
        
        print(f"[FLANDERS] Accepted Clients: {trusted_client_ids}")
        accepted_updates = []
        for cid in trusted_client_ids:
            accepted_updates.append(client_updates[cid])
            
        if not accepted_updates:
            print("[FLANDERS] All clients rejected! Skipping round.")
            return self.global_model_state
            
        return self._simple_fedavg(accepted_updates)

    def _aggregate_endpca(self, client_updates):
        """
        Baseline 2: EndPCA
        """
        print("[EndPCA] Computing Entropy-Weighted Trust Scores...")
        trust_scores_list, trust_scores_dict = self.endpca_ensemble.compute_ensemble_scores(client_updates)
        print(f"[EndPCA] Distributed Trust Scores: {trust_scores_dict}")
        
        # Normalize sum to 1.0 (or sum(trust_scores))
        total_trust = sum(trust_scores_list) + 1e-9
        normalized_trust = [t / total_trust for t in trust_scores_list]
        
        # client_updates is already a List[Tuple] sorted by client_id in compute_ensemble_scores implicit logic
        # but to be safe we sort here or pass as is if compute_ensemble_scores handled it.
        # It's better to extract just the dicts from the tuples.
        sorted_updates = sorted(client_updates, key=lambda x: x[0])
        return self._weighted_fedavg(sorted_updates, normalized_trust)

    def _simple_fedavg(self, accepted_updates):
        """Standard uniform FedAvg"""
        aggregated_update = {}
        # Extract just the PyTorch state dicts from the tuples
        update_dicts = [u[1] for u in accepted_updates]
        
        for k in update_dicts[0].keys():
            aggregated_update[k] = torch.stack([u[k] for u in update_dicts]).mean(dim=0)
        noisy_update = self.add_dp_noise(aggregated_update)
        for k in self.global_model_state.keys():
            self.global_model_state[k] -= self.global_lr * noisy_update[k]
        return self.global_model_state

    def _weighted_fedavg(self, accepted_updates, accepted_trust_scores):
        """Trust-weighted FedAvg"""
        total_trust = sum(accepted_trust_scores) + 1e-9
        aggregated_update = {}
        # Extract just the PyTorch state dicts from the tuples
        update_dicts = [u[1] for u in accepted_updates]
        
        for k in update_dicts[0].keys():
            weighted_sum = sum(u[k] * t for u, t in zip(update_dicts, accepted_trust_scores))
            aggregated_update[k] = weighted_sum / total_trust
            
        noisy_update = self.add_dp_noise(aggregated_update)
        for k in self.global_model_state.keys():
            self.global_model_state[k] -= self.global_lr * noisy_update[k]
        return self.global_model_state

    def _aggregate_metatrust(self, client_updates, round_num, total_rounds, evaluate_byzantine_flags=None):
        """
        The core aggregation algorithm with trust evaluation and ZKP proofs based on Algorithm 2.
        evaluate_byzantine_flags: A boolean array indicating if each client is malicious (used only for Reward calculation).
        """
        accepted_updates = []
        accepted_trust_scores = []
        
        for i, (client_id, update_dict, lstm_grad, mlp_grad, hash_comm, zkp_proof, proof_cost) in enumerate(client_updates):
            flat_grad = np.concatenate((lstm_grad, mlp_grad))
            
            # Step 1: Calculate anomaly score
            A_i, projected_y = self.anomaly_detector.calculate_anomaly_score(flat_grad)
            
            # Normalize anomaly logic (simplified for agent state)
            a_norm = min(1.0, A_i / 20.0)
            
            # Build state vector for policy eq 4
            t_prev = self.trust_system.get_score(i)
            # Use empirical mean/var of projected_y for state abstraction or history var
            mean_h = np.mean(projected_y)
            var_h = np.var(projected_y)
            
            state_vector = torch.tensor([a_norm, t_prev, mean_h, var_h, round_num/total_rounds], dtype=torch.float32)
            
            # Step 2: Policy assigns ZKP level
            action, log_prob = self.policy_agent.get_action(state_vector)
            assigned_level = "FULL_ZKP" if action == 0 else "SAMPLE_ZKP"
            
            # (In simulation: the client ALREADY generated the proof based on what the server requested.
            # Here we just verify the exact payload submitted matching the requested level).
            
            # Step 3: Verify Cryptographic Proof limit Groth16 Simulator
            is_valid = Groth16Simulator.verify(zkp_proof, mlp_grad)
            
            # Step 4: Update Trust Dynamics
            new_trust = self.trust_system.update(client_idx=i, verification_failed=not is_valid, action_taken=action)
            
            # Save Trajectory for REINFORCE
            if evaluate_byzantine_flags is not None:
                is_byzantine = evaluate_byzantine_flags[i]
                reward = self.policy_agent.compute_reward(is_byzantine, passed_check=is_valid, verification_cost=proof_cost)
                self.accumulated_trajectories.append((log_prob, reward))
            
            if is_valid:
                accepted_updates.append(update_dict)
                accepted_trust_scores.append(new_trust)
            # Step anomaly detector with verified state
            self.anomaly_detector.step(flat_grad, passed_verification=True)
                
        # Aggregate accepted (Trust-Weighted FedAvg Eq 14 implicit in algorithm 2)
        if not accepted_updates:
            print("WARNING: All updates rejected! Skpping aggregation.")
            return

        aggregated_update = {}
        total_trust = sum(accepted_trust_scores)
        
        for k in accepted_updates[0].keys():
            aggregated_update[k] = torch.zeros_like(accepted_updates[0][k])
            for idx, update in enumerate(accepted_updates):
                weight = accepted_trust_scores[idx] / total_trust
                aggregated_update[k] += update[k] * weight
                
        # Appy DP
        aggregated_update = self.add_dp_noise(aggregated_update)
        
        # Apply to global model
        current_state = self.global_model.state_dict()
        for k in current_state.keys():
            current_state[k] += aggregated_update[k]
            
        self.global_model.load_state_dict(current_state)
        
    def optimize_policy(self):
        """
        Executes the agent update step (Eq. 5) after collecting a batch of episodes during Meta-training.
        """
        avg_reward = self.policy_agent.update_policy(self.accumulated_trajectories)
        self.accumulated_trajectories = [] # clear buffer
        return avg_reward
