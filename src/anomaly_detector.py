import hashlib
import time
import numpy as np

# ==============================================================================
# zkp_utils.py & anomaly_detector.py functionality
# 
# 1. Groth16 Simulator: Simulates proving time (4.70s) for Full-ZKP
#    and verification time (0.11s) as measured in the paper.
# 2. Poseidon Hash Simulator: Tamper-evidence bounds for LSTM gradient.
# ==============================================================================

class PoseidonHashSimulator:
    """
    The paper uses the Poseidon hash as a Commitment to prove the integrity 
    (Tamper-evidence) of the 33,280 parameters in the LSTM section.
    This class simulates that behavior focusing strictly on security semantics and hash generation.
    """
    @staticmethod
    def commit(gradient_vector):
        # We simulate the arithmetic Poseidon hash over the BN254 curve 
        # using Python's SHA-256 for rapid deterministic commitment mapping.
        grad_bytes = gradient_vector.tobytes()
        return hashlib.sha256(grad_bytes).hexdigest()

class Groth16Simulator:
    """
    Unlike standard neural networks, the Groth16 zk-SNARK protocol requires Polynomial 
    IOP cryptography over the BN254 scalar field. The paper clearly states that for 33,410 
    parameters, this translates to 1.67 million R1CS constraints.
    Proof generation time: 4.70 seconds
    Proof verification time: 0.11 seconds
    
    This class perfectly mocks these delays and logic outputs for the ATBV module.
    """
    @staticmethod
    def prove(gradient_vector, zk_level="FULL_ZKP", simulate_delay=False):
        """
        Generates an artificial ZK proof. 
        If zk_level is SAMPLE_ZKP, only 10% of the gradient is proven.
        """
        # In actual deployment, simulate_delay=True enforces real delays.
        # For rapid meta-training convergence in tests, we skip time.sleep 
        # but account for the 'cost' in the environment loop.
        
        proof_size_bytes = 128  # Constant size for Groth16
        
        if zk_level == "FULL_ZKP":
            cost = 4.70
            fraction = 1.0
        elif zk_level == "SAMPLE_ZKP":
            cost = 0.47  # Only 10% is proved (approx scaling)
            fraction = 0.10
        else:
            raise ValueError(f"Unknown ZKP level: {zk_level}")
            
        if simulate_delay:
            time.sleep(cost)
            
        proof_payload = f"proof_{zk_level}_{hashlib.sha256(gradient_vector[:int(len(gradient_vector)*fraction)].tobytes()).hexdigest()[:16]}"
        return proof_payload, cost

    @staticmethod
    def verify(proof_payload, gradient_vector, simulate_delay=False):
        """
        Proof verification, which takes constant time regardless of the circuit size (O(1)).
        """
        if simulate_delay:
            time.sleep(0.11)
            
        # Simplistic verification check: assuming honest proof generation for simulation
        if proof_payload.startswith("proof_"):
            return True
        return False
        
class PCAMahalanobisDetector:
    """
    Section 3.8 of the paper: Anomaly detection based on online PCA and Mahalanobis distance.
    Stage 1: Oja's Rule PCA (d=7 components)
    Stage 2: Diagonal Mahalanobis Distance over a window (W=10)
    """
    def __init__(self, n_params, d=7, window_size=10, epsilon=1e-6):
        self.n_params = n_params
        self.d = d
        self.W = window_size
        self.epsilon = epsilon
        
        # Oja's PCA initialized basis
        self.U = np.random.randn(n_params, d)
        self.U, _ = np.linalg.qr(self.U) # Orthonormal basis
        
        # Window memory for projections
        self.projection_history = []
        self.learning_rate_pca = 0.01

    def update_basis_oja(self, delta):
        """
        Update the top-d principal components U using Oja's rule.
        $\Delta U = \eta (\Delta (\Delta^T U) - U (\Delta^T U)^T (\Delta^T U))$
        """
        delta = delta.reshape(-1, 1) # (P, 1)
        y = self.U.T @ delta         # (d, 1) projected
        
        # Oja's update
        dU = self.learning_rate_pca * (delta @ y.T - self.U @ (y @ y.T))
        self.U = self.U + dU
        
        # Re-orthonormalize for stability
        self.U, _ = np.linalg.qr(self.U)

    def calculate_anomaly_score(self, delta):
        """
        Project and compute diagonal Mahalanobis distance.
        Eq. 11
        """
        # 1. Project gradient onto top-d components
        y_t = (self.U.T @ delta.reshape(-1, 1)).flatten() # (d,)
        
        if len(self.projection_history) < 2:
            return 0.0, y_t # Not enough history for variance

        historical_y = np.array(self.projection_history[-self.W:]) # (W, d)
        
        # 2. Diagonal Mahalanobis formulation
        mu = np.mean(historical_y, axis=0)
        sigma2 = np.var(historical_y, axis=0) + self.epsilon
        
        # A_i^t = \sum_j (y_t,j - mu_j)^2 / sigma^2_j
        A = np.sum(((y_t - mu) ** 2) / sigma2)
        
        return A, y_t
        
    def step(self, delta, passed_verification):
        """
        Called after evaluating the gradient. If passed, updates history.
        """
        A, y_t = self.calculate_anomaly_score(delta)
        
        if passed_verification:
            self.update_basis_oja(delta)
            self.projection_history.append(y_t)
            
        return A
