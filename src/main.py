import torch
import numpy as np
import time

from dataset import simulate_eICU_federated_data, get_dataloaders
from client import HospitalClient
from server import MetaTrustServer
from attacks import ByzantineAttacker, extract_flat_arrays
from zkp_utils import Groth16Simulator, PoseidonHashSimulator

# ==============================================================================
# main.py
# 
# In this file, all components—models, datasets, clients, server, cryptography, 
# and anomaly detection—are wired up together.
# 
# Workflow:
# 1. Simulate data for the 5 hospitals.
# 2. Initialize and assign models.
# 3. Train the Trust Policy agent (Meta-training) (only if MetaTrust)
# 4. Run FL (Federated Learning) rounds while simulating Byzantine attacks.
# ==============================================================================

def main():
    print("====================================")
    print("MetaTrust-FL: Initialization started")
    print("====================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Data Simulation
    total_samples = 15400  # Number of simulated eICU patients
    hospital_data = simulate_eICU_federated_data(total_samples)
    train_loaders, test_loaders = get_dataloaders(hospital_data, batch_size=128)
    
    N_servers = 5
    clients = []
    for i in range(N_servers):
        client = HospitalClient(client_id=i, 
                                train_loader=train_loaders[i], 
                                test_loader=test_loaders[i], 
                                device=device)
        clients.append(client)
        
    # Select the defense strategy here: 'metatrust', 'flanders', or 'endpca'
    defense_strategy = "metatrust"
    print(f"\n--- Initializing Server with Defense: {defense_strategy.upper()} ---")
    server = MetaTrustServer(num_clients=len(clients), global_lr=1.0, defense_type=defense_strategy, device=device)

    print("\n[Phase 2] Executing FL Evaluation Rounds...")
    # --- PHASE 2: FL Evaluation Rounds (Algorithm 2) ---
    
    total_fl_rounds = 20  # Abridged from 100 to execute quickly in proof of concept
    start_time = time.time()
    
    # Define attack params
    byzantine_client_idx = 4 
    attack_type = "SignFlipping" # One of: Random, SignFlipping, Scaling, NullSpace
    
    print(f"Executing with Attack: {attack_type} on Client {byzantine_client_idx}")

    for r in range(1, total_fl_rounds + 1):
        
        # 1. Sync clients
        for c in clients:
            c.sync_with_server(server.get_global_weights())
            
        server_updates = []
        is_byz_flags = [False]*N_servers
        is_byz_flags[byzantine_client_idx] = True
        
        for i, c in enumerate(clients):
            # 2. Local Train
            update_dict, lstm_grad, mlp_grad = c.local_training(local_epochs=1)
            
            # 3. Attacker logic
            if i == byzantine_client_idx:
                if attack_type == "Random":
                    update_dict = ByzantineAttacker.random_gradient(update_dict, magnitude=2.0)
                elif attack_type == "SignFlipping":
                    update_dict = ByzantineAttacker.sign_flipping(update_dict)
                elif attack_type == "Scaling":
                    update_dict = ByzantineAttacker.scaling_attack(update_dict, kappa=10.0)
                elif attack_type == "NullSpace":
                    # Reconstruct orthogonal matrix approximately using MetaTrust detector
                    if hasattr(server, 'anomaly_detector') and hasattr(server.anomaly_detector, 'U'):
                        update_dict = ByzantineAttacker.null_space_attack(update_dict, server.anomaly_detector.U, alpha=0.5)
                
                lstm_grad, mlp_grad = extract_flat_arrays(update_dict)
            
            # 4. Mock ZKP Generation
            hash_c, zkp_proof, p_cost = c.generate_proofs(lstm_grad, mlp_grad, verification_level="FULL_ZKP")
            
            # 5. Send to Server
            server_updates.append((c.client_id, update_dict, lstm_grad, mlp_grad, hash_c, zkp_proof, p_cost))
            
        # 6. Aggregate at server
        server.process_and_aggregate(server_updates, r, total_fl_rounds, evaluate_byzantine_flags=is_byz_flags)
        
        # Provide diagnostic trace
        if r % 5 == 0 or r == total_fl_rounds:
            print(f"--- Round {r}/{total_fl_rounds} ---")
            if defense_strategy == "metatrust":
                for i in range(N_servers):
                    trust = server.trust_system.get_score(i)
                    print(f"   Hospital {i+1}: Trust={trust:.2f}")

    # Evaluate Global Model
    print("\nGlobal Evaluation on test sets (AUC approx):")
    total_acc = 0.0
    for i, c in enumerate(clients):
        c.sync_with_server(server.get_global_weights())
        loss, acc, _, _, _ = c.evaluate()
        total_acc += acc
        print(f"Hospital {i+1} Acc: {acc*100:.1f}%, Loss: {loss:.4f}")
        
    print(f"\nAverage Accuracy: {(total_acc / N_servers)*100:.1f}%")
    print(f"Execution finished in {time.time()-start_time:.1f}s.")

if __name__ == "__main__":
    main()
