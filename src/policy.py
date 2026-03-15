import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

# ==============================================================================
# policy.py
# 
# 1. TrustPolicyNetwork: A 3-layer MLP network activated by a Softmax function.
# 2. REINFORCE Algorithm: Parameter updates using policy gradients and a moving baseline.
# 3. Computes trust dynamics: forgetting factor, recovery, and the lower bound.
# ==============================================================================

class TrustPolicyNetwork(nn.Module):
    """
    The trust evaluation network that assigns verification levels based on the system state.
    The 10-dimensional input includes: normalized anomaly score, previous trust, mean/var of trust history, and time step.
    Outputs a probability distribution over FULL_ZKP and SAMPLE_ZKP. (Hash checks bypass this logic).
    """
    def __init__(self, state_dim=5, hidden_1=32, hidden_2=16, out_dim=2):
        super(TrustPolicyNetwork, self).__init__()
        # State: [anomaly_norm, trust_prev, mean_h, var_h, t/T] -> length=5
        # The paper specifies length=10, meaning perhaps history buffers or extended stats.
        # We'll use the core 5 components explicitly mentioned in Eq 4 definition text.
        self.fc1 = nn.Linear(state_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, out_dim)

    def forward(self, x):
        # x shape (batch, state_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Use Softmax
        return F.softmax(x, dim=-1)

class REINFORCEAgent:
    """
    REINFORCE-based policy training (Algorithm 1).
    Action states: 0 -> FULL_ZKP, 1 -> SAMPLE_ZKP
    """
    def __init__(self, state_dim=5, lr=0.001):
        self.policy = TrustPolicyNetwork(state_dim=state_dim, out_dim=2)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.baseline = 0.0
        
        # Hyperparameters for reward mapping Eq 6
        self.lambda_1 = 1.0  # Reward for detecting Byzantine
        self.lambda_2 = 0.3  # Penalty for verification cost
        self.lambda_3 = 0.5  # Penalty for rejecting honest
        self.lambda_4 = 2.0  # Penalty for missing Byzantine

    def get_action(self, state_tensor):
        """
        Samples an action from the policy (Policy inference).
        """
        probs = self.policy(state_tensor)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def compute_reward(self, is_byzantine, passed_check, verification_cost):
        """
        Calculates the reward at each step based on the action outcome during the federated round.
        """
        reward = 0.0
        # Eq 6 implementation
        if is_byzantine and not passed_check:
            reward += self.lambda_1
        if is_byzantine and passed_check:
            # Byzantine missed
            reward -= self.lambda_4
        if not is_byzantine and not passed_check:
            # Honest rejected (false positive)
            reward -= self.lambda_3
            
        reward -= self.lambda_2 * verification_cost
        
        return reward

    def update_policy(self, trajectories):
        """
        Updates the Policy parameters using state vectors, log probabilities, and rewards.
        Eq 5: \nabla J = E[ \sum \nabla \log \pi * (R - b) ]
        """
        if not trajectories:
            return 0.0
            
        policy_loss = []
        total_rewards = []
        
        for (log_prob, reward) in trajectories:
            # Subtract baseline for variance reduction
            advantage = reward - self.baseline
            policy_loss.append(-log_prob * advantage)
            total_rewards.append(reward)
            
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        # Update baseline (EMA)
        mean_reward = np.mean(total_rewards)
        self.baseline = 0.9 * self.baseline + 0.1 * mean_reward
        
        return mean_reward

class TrustDynamics:
    """
    Evaluates and updates trust scores over time using Equations 7 and 8.
    """
    def __init__(self, num_clients, t_min=0.1, gamma=0.95, k_rec=5, lam_rec=0.05):
        self.trust_scores = np.full(num_clients, 0.5)
        self.t_min = t_min
        self.gamma = gamma
        self.k_rec = k_rec
        self.lam_rec = lam_rec
        
        self.consecutive_passes = np.zeros(num_clients)
        
    def update(self, client_idx, verification_failed, action_taken):
        """
        Eq. 8.
        action_taken: 0 = FULL_ZKP (v=0), 1 = SAMPLE_ZKP (v=0.5)
        """
        v_cost = 0.0 if action_taken == 0 else 0.5
        
        if verification_failed:
            self.trust_scores[client_idx] = self.t_min
            self.consecutive_passes[client_idx] = 0
            
        else:
            prev_t = self.trust_scores[client_idx]
            self.trust_scores[client_idx] = max(self.t_min, self.gamma * prev_t + (1 - self.gamma) * v_cost)
            self.consecutive_passes[client_idx] += 1
            
            # Application of recovery if enough history is clean
            if self.consecutive_passes[client_idx] >= self.k_rec:
                self.trust_scores[client_idx] = min(self.trust_scores[client_idx] + self.lam_rec, 1.0)
                
        return self.trust_scores[client_idx]
        
    def get_score(self, client_idx):
        return self.trust_scores[client_idx]
