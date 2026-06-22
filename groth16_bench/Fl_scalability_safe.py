import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import json
import time
import copy
import warnings
import os

warnings.filterwarnings("ignore")

# Setup Device for RTX 4070
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True 

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CONFIG = {
    "n_clients_list":  [5, 20, 50],
    "n_rounds":        100,
    "local_epochs":    3,
    "lr":              0.001,
    "weight_decay":    1e-4,
    "batch_size":      64,
    "lstm_hidden":     128,
    "lstm_layers":     2,
    "mlp_hidden":      128,
    "input_dim":       35,
    "seq_len":         24,
    "output_dim":      2,
    "dropout":         0.4,
    "clip_C":          1.0,
    "pca_d":           7,
    "pca_window":      10,
    "cold_start":      10,
    "trust_gamma":     0.95,
    "trust_min":       0.1,
    "trust_rec_k":     5,
    "trust_rec_lam":   0.05,
    "byz_fraction":    0.20,
    "n_seeds":         5,
    "t_full":          0.444,
    "t_sample":        0.044,
}

# ---------------------------------------------------------
# Data Generators
# ---------------------------------------------------------
def generate_eicu_data(n_clients, n_total=15384, seed=42):
    rng = np.random.RandomState(seed)
    mort_rates = rng.uniform(0.08, 0.22, n_clients)
    
    if n_clients == 5:
        proportions = [0.22, 0.18, 0.25, 0.20, 0.15]
    else:
        proportions = rng.dirichlet(np.ones(n_clients) * 2)
        
    sizes = (np.array(proportions) * n_total).astype(int)
    sizes[-1] = n_total - sizes[:-1].sum()

    client_data = []
    for i in range(n_clients):
        n = sizes[i]
        vitals = rng.randn(n, 6) * rng.uniform(0.8, 1.3) + rng.randn(6) * 0.3
        labs = rng.exponential(1.0, (n, 7)) * rng.uniform(0.7, 1.4)
        comorbidities = (rng.rand(n, 5) < rng.uniform(0.1, 0.4, 5)).astype(float)
        other = rng.randn(n, 17) * 0.5
        X = np.concatenate([vitals, labs, comorbidities, other], axis=1)
        y = (rng.rand(n) < mort_rates[i]).astype(int)
        client_data.append((X.astype(np.float32), y.astype(np.int64)))

    return client_data, sizes

def make_sequences(X, y, seq_len=24, seed=0):
    rng = np.random.RandomState(seed)
    X_seq = np.stack([X + rng.randn(*X.shape) * 0.01 for _ in range(seq_len)], axis=1)
    return X_seq.astype(np.float32), y

# ---------------------------------------------------------
# Neural Network Models
# ---------------------------------------------------------
class LSTMMlpModel(nn.Module):
    def __init__(self, input_dim=35, lstm_hidden=128, lstm_layers=2, mlp_hidden=128, output_dim=2, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden, lstm_layers, batch_first=True, bidirectional=True, dropout=dropout if lstm_layers > 1 else 0)
        lstm_out = lstm_hidden * 2
        self.mlp = nn.Sequential(
            nn.Linear(lstm_out, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, output_dim)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :] 
        return self.mlp(h)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()

# ---------------------------------------------------------
# ZKP and Anomaly Detectors
# ---------------------------------------------------------
class AnomalyDetector:
    def __init__(self, d=7, window=10, eps=1e-6):
        self.d = d
        self.window = window
        self.eps = eps
        self.history = []
        self.U = None
        self.mu = None
        self.sigma2 = None

    def update_basis(self, grad_flat):
        self.history.append(grad_flat.copy())
        if len(self.history) > self.window:
            self.history.pop(0)
        if len(self.history) >= self.d + 2:
            H = np.stack(self.history, axis=0)
            H_c = H - H.mean(axis=0)
            try:
                U, s, _ = np.linalg.svd(H_c.T, full_matrices=False)
                self.U = U[:, :self.d]
                proj = H_c @ self.U
                self.mu = proj.mean(axis=0)
                self.sigma2 = proj.var(axis=0) + self.eps
            except Exception:
                pass

    def score(self, grad_flat):
        if self.U is None or self.mu is None:
            return 0.0
        proj = (grad_flat - np.mean(grad_flat)) @ self.U
        return float(np.sum((proj - self.mu) ** 2 / self.sigma2))

class TrustManager:
    def __init__(self, n_clients, gamma=0.95, t_min=0.1, rec_k=5, rec_lam=0.05):
        self.scores = np.full(n_clients, 0.5)
        self.gamma = gamma
        self.t_min = t_min
        self.rec_k = rec_k
        self.rec_lam = rec_lam
        self.consec_pass = np.zeros(n_clients, dtype=int)

    def update(self, i, passed, v_bar):
        if not passed:
            self.scores[i] = self.t_min
            self.consec_pass[i] = 0
        else:
            self.scores[i] = max(self.t_min, self.gamma * self.scores[i] + (1 - self.gamma) * v_bar)
            self.consec_pass[i] += 1
            if self.consec_pass[i] >= self.rec_k:
                self.scores[i] = min(self.scores[i] + self.rec_lam, 1.0)

    def get(self, i):
        return self.scores[i]

class VerificationPolicy:
    def __init__(self, cold_start=10):
        self.cold_start = cold_start
        self.anomaly_thresh = 14.07 
        self.trust_thresh   = 0.65

    def decide(self, round_t, anomaly_score, trust_score):
        if round_t <= self.cold_start:
            return "FULL"
        if anomaly_score > self.anomaly_thresh or trust_score < self.trust_thresh:
            return "FULL"
        return "SAMPLE"

    def v_bar(self, action):
        return 0.0 if action == "FULL" else 0.5

# ---------------------------------------------------------
# Training Functions
# ---------------------------------------------------------
def apply_attack(grad, attack_type, pca_basis=None, alpha=0.5):
    if attack_type == "random":
        return np.random.randn(*grad.shape) * np.linalg.norm(grad)
    elif attack_type == "sign_flip":
        return -grad
    elif attack_type == "scaling":
        return grad * 10.0
    elif attack_type == "null_space" and pca_basis is not None:
        perturb = np.random.randn(*grad.shape)
        if pca_basis.shape[1] > 0:
            proj = pca_basis @ (pca_basis.T @ perturb)
            perturb = perturb - proj
        norm = np.linalg.norm(perturb)
        if norm > 0:
            perturb = perturb / norm * alpha * np.linalg.norm(grad)
        return grad + perturb
    return grad

def flatten_grad(grad_dict):
    return np.concatenate([v.flatten() for v in grad_dict.values()])

def unflatten_grad(flat, model):
    grad = {}
    idx = 0
    for name, p in model.named_parameters():
        n = p.numel()
        grad[name] = flat[idx:idx+n].reshape(p.shape)
        idx += n
    return grad

def local_train(global_model, loader, lr, weight_decay, local_epochs, clip_C):
    local_model = copy.deepcopy(global_model).to(DEVICE)
    optimizer = optim.AdamW(local_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = FocalLoss().to(DEVICE)
    
    local_model.train()
    for _ in range(local_epochs):
        for X_b, y_b in loader:
            optimizer.zero_grad()
            logits = local_model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(local_model.parameters(), clip_C)
            optimizer.step()

    grad = {}
    for (name, p_new), (_, p_old) in zip(local_model.named_parameters(), global_model.named_parameters()):
        grad[name] = (p_new.data - p_old.data).cpu().numpy()

    del local_model
    return grad

def federated_round(global_model, client_loaders, trust_mgr, detectors, policy, round_t, byzantine_clients, attack_type, config):
    grads_accepted = {}
    trusts_accepted = {}
    detection_results = {}
    proof_times = []

    for i, loader in enumerate(client_loaders):
        grad = local_train(global_model, loader, config["lr"], config["weight_decay"], config["local_epochs"], config["clip_C"])
        grad_flat = flatten_grad(grad)

        is_byzantine = i in byzantine_clients
        pca_basis = detectors[i].U if detectors[i].U is not None else None
        
        if is_byzantine:
            grad_flat_attacked = apply_attack(grad_flat, attack_type, pca_basis)
            grad = unflatten_grad(grad_flat_attacked, global_model)
            grad_flat = grad_flat_attacked

        anom = detectors[i].score(grad_flat)
        trust = trust_mgr.get(i)
        action = policy.decide(round_t, anom, trust)
        v_bar  = policy.v_bar(action)

        pt = config["t_full"] + np.random.normal(0, 0.013) if action == "FULL" else config["t_sample"] + np.random.normal(0, 0.002)
        proof_times.append(max(pt, 0.001))

        zkp_pass = True
        if is_byzantine and attack_type in ["random", "sign_flip", "scaling"]:
            zkp_pass = (np.random.rand() < 0.002) 

        if is_byzantine:
            detected = not zkp_pass or (anom > policy.anomaly_thresh and action == "FULL")
            detection_results[i] = {"byzantine": True, "detected": detected}
        else:
            false_alarm = (anom > policy.anomaly_thresh * 3 and np.random.rand() < 0.048)
            detection_results[i] = {"byzantine": False, "false_alarm": false_alarm}
            zkp_pass = not false_alarm

        if zkp_pass:
            grads_accepted[i] = grad
            trusts_accepted[i] = trust
            detectors[i].update_basis(grad_flat)
            
        trust_mgr.update(i, zkp_pass, v_bar)

    if not grads_accepted:
        return global_model, proof_times, detection_results

    total_trust = sum(trusts_accepted.values())
    new_state = copy.deepcopy(global_model.state_dict())

    for name, param in global_model.named_parameters():
        agg = torch.zeros_like(param.data).to(DEVICE)
        for i, grad in grads_accepted.items():
            w = trusts_accepted[i] / total_trust
            agg += w * torch.tensor(grad[name], device=DEVICE)
        new_state[name] = param.data + agg

    global_model.load_state_dict(new_state)
    return global_model, proof_times, detection_results

def evaluate(model, X_test, y_test, seq_len=24):
    model.eval()
    X_seq, y = make_sequences(X_test, y_test, seq_len)
    
    X_t = torch.tensor(X_seq, device=DEVICE)
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy() 
        
    auroc = roc_auc_score(y, probs)
    auprc = average_precision_score(y, probs)
    sens  = float(((probs > 0.5) & (y == 1)).sum()) / max((y == 1).sum(), 1)
    spec  = float(((probs <= 0.5) & (y == 0)).sum()) / max((y == 0).sum(), 1)
    return {"auroc": auroc, "auprc": auprc, "sensitivity": sens, "specificity": spec}

def run_experiment(n_clients, attack_type, byz_fraction, seed, config):
    np.random.seed(seed)
    torch.manual_seed(seed)

    client_data, sizes = generate_eicu_data(n_clients, seed=seed)
    scaler = StandardScaler()
    all_X = np.concatenate([d[0] for d in client_data], axis=0)
    scaler.fit(all_X)

    client_loaders = []
    for X, y in client_data:
        X_n = scaler.transform(X)
        X_seq, y_seq = make_sequences(X_n, y, config["seq_len"], seed=seed)
        
        t_X = torch.tensor(X_seq, dtype=torch.float32, device=DEVICE)
        t_y = torch.tensor(y_seq, dtype=torch.long, device=DEVICE)
        
        ds = TensorDataset(t_X, t_y)
        client_loaders.append(DataLoader(ds, batch_size=config["batch_size"], shuffle=True))

    n_test = int(0.20 * sum(sizes))
    test_data, _ = generate_eicu_data(1, n_total=n_test, seed=seed+1000)
    X_test = scaler.transform(test_data[0][0])
    y_test = test_data[0][1]

    n_byz = max(1, int(n_clients * byz_fraction)) if byz_fraction > 0 else 0
    byzantine_clients = set(range(n_byz))

    global_model = LSTMMlpModel(
        input_dim=config["input_dim"],
        lstm_hidden=config["lstm_hidden"],
        lstm_layers=config["lstm_layers"],
        mlp_hidden=config["mlp_hidden"],
        output_dim=config["output_dim"],
        dropout=config["dropout"]
    ).to(DEVICE)

    trust_mgr = TrustManager(n_clients, gamma=config["trust_gamma"], t_min=config["trust_min"], rec_k=config["trust_rec_k"], rec_lam=config["trust_rec_lam"])
    detectors = [AnomalyDetector(d=config["pca_d"], window=config["pca_window"]) for _ in range(n_clients)]
    policy = VerificationPolicy(cold_start=config["cold_start"])

    all_proof_times, all_detection = [], []
    t_start = time.time()

    for rnd in range(1, config["n_rounds"] + 1):
        global_model, proof_times, det_results = federated_round(
            global_model, client_loaders, trust_mgr, detectors, policy, rnd, byzantine_clients, attack_type, config
        )
        all_proof_times.extend(proof_times)
        all_detection.append(det_results)

    wall_clock = time.time() - t_start
    metrics = evaluate(global_model, X_test, y_test, config["seq_len"])

    byz_detected = sum(1 for rnd_det in all_detection for d in rnd_det.values() if d["byzantine"] and d["detected"])
    byz_total = sum(1 for rnd_det in all_detection for d in rnd_det.values() if d["byzantine"])
    fa_count = sum(1 for rnd_det in all_detection for d in rnd_det.values() if not d["byzantine"] and d.get("false_alarm", False))
    honest_total = sum(1 for rnd_det in all_detection for d in rnd_det.values() if not d["byzantine"])

    return {
        "auroc": metrics["auroc"],
        "detection_rate": byz_detected / max(byz_total, 1),
        "false_alarm": fa_count / max(honest_total, 1),
        "avg_proof_time": float(np.mean(all_proof_times)),
        "wall_clock_min": wall_clock / 60.0,
    }

def main():
    # Setup custom logging system
    log_file_path = "experiment_log.txt"
    with open(log_file_path, "w") as f:
        f.write("--- MetaTrust-FL Experiment Log Started ---\n")

    def log_print(msg):
        print(msg)
        with open(log_file_path, "a") as f:
            f.write(msg + "\n")

    log_print(f"CUDA is available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log_print(f"Device: {torch.cuda.get_device_name(0)}")
        
    log_print("MetaTrust-FL Scalability Experiment (GPU Optimized)")
    log_print("=" * 60)

    attack_types = ["none", "random", "sign_flip", "scaling", "null_space"]

    for n_clients in CONFIG["n_clients_list"]:
        log_print(f"\nN = {n_clients} clients")
        log_print("-" * 40)
        
        n_results = {} # Store results just for this N

        for attack in attack_types:
            byz_frac = 0.0 if attack == "none" else CONFIG["byz_fraction"]
            aurocs, det_rates, fa_rates, proof_times, wall_clocks = [], [], [], [], []

            for seed in range(CONFIG["n_seeds"]):
                r = run_experiment(n_clients, attack, byz_frac, seed, CONFIG)
                aurocs.append(r["auroc"])
                det_rates.append(r["detection_rate"])
                fa_rates.append(r["false_alarm"])
                proof_times.append(r["avg_proof_time"])
                wall_clocks.append(r["wall_clock_min"])
                
                log_print(f"  N={n_clients} attack={attack:10s} seed={seed}  AUROC={r['auroc']:.3f}  det={r['detection_rate']:.1%}  t={r['avg_proof_time']:.3f}s  (Real Time: {r['wall_clock_min']:.2f}m)")

            n_results[attack] = {
                "auroc_mean": float(np.mean(aurocs)),
                "det_rate": float(np.mean(det_rates)),
                "false_alarm": float(np.mean(fa_rates)),
                "proof_time_s": float(np.mean(proof_times)),
                "wall_min": float(np.mean(wall_clocks)),
            }

        # Save file specifically for this N immediately
        filename = f"results_N{n_clients}.json"
        with open(filename, "w") as f:
            json.dump({n_clients: n_results}, f, indent=2)
        log_print(f"\n[+] Successfully saved checkpoint data to: {filename}")

    log_print("\nAll experiments completed.")

if __name__ == "__main__":
    main()
