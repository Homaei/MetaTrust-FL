import hashlib
import time

class PoseidonHashSimulator:
    @staticmethod
    def commit(gradient_vector):
        grad_bytes = gradient_vector.tobytes()
        return hashlib.sha256(grad_bytes).hexdigest()

class Groth16Simulator:
    @staticmethod
    def prove(gradient_vector, zk_level="FULL_ZKP", simulate_delay=False):
        proof_size_bytes = 128
        if zk_level == "FULL_ZKP":
            cost = 4.70
            fraction = 1.0
        elif zk_level == "SAMPLE_ZKP":
            cost = 0.47
            fraction = 0.10
        else:
            raise ValueError(f"Unknown ZKP level: {zk_level}")
            
        if simulate_delay:
            time.sleep(cost)
            
        proof_payload = f"proof_{zk_level}_{hashlib.sha256(gradient_vector[:int(len(gradient_vector)*fraction)].tobytes()).hexdigest()[:16]}"
        return proof_payload, cost

    @staticmethod
    def verify(proof_payload, gradient_vector, simulate_delay=False):
        if simulate_delay:
            time.sleep(0.11)
        if proof_payload.startswith("proof_"):
            return True
        return False
