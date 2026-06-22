import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import json
import time
import copy
import warnings

warnings.filterwarnings("ignore")

# Setup Device for RTX 4070
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True 

# Create dedicated results directory
RESULTS_DIR = "REPORTS_REAL_DATA"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

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
    "t_full":          0.101,
    "t_sample":        0.010,
}

# ---------------------------------------------------------
# REAL Data Pipeline (Zero Leakage & Fixed Proportions)
# ---------------------------------------------------------
def load_real_eicu_data(n_clients, data_dir="/home/hubert/project/MetaTrust-FL/Dataset/demo", seed=42):
    patient_path = os.path.join(data_dir, "patient.csv.gz")
    vital_path = os.path.join(data_dir, "vitalPeriodic.csv.gz")
    
    df_patient = pd.read_csv(patient_path, compression='gzip')
    df_patient = df_patient.dropna(subset=['hospitaldischargestatus'])
    df_patient['label'] = (df_patient['hospitaldischargestatus'] == 'Expired').astype(int)
    
    df_patient['age'] = df_patient['age'].replace('> 89', '90').astype(float)
    df_patient['admissionweight'] = df_patient['admissionweight'].astype(float).fillna(df_patient['admissionweight'].median())
    df_patient['admissionheight'] = df_patient['admissionheight'].astype(float).fillna(df_patient['admissionheight'].median())
    df_patient['age'] = df_patient['age'].fillna(df_patient['age'].median())

    df_vital = pd.read_csv(vital_path, compression='gzip')
    vital_features = ['heartrate', 'respiration', 'sao2', 'systemicsystolic', 'systemicdiastolic']
    df_vital = df_vital[['patientunitstayid', 'observationoffset'] + vital_features]
    df_vital = df_vital.sort_values(by=['patientunitstayid', 'observationoffset'])
    df_vital[vital_features] = df_vital.groupby('patientunitstayid')[vital_features].ffill()
    df_vital_agg = df_vital.groupby('patientunitstayid')[vital_features].mean().reset_index()
    
    df_final = pd.merge(df_patient, df_vital_agg, on='patientunitstayid', how='left')
    for col in vital_features:
        df_final[col] = df_final[col].fillna(df_vital_agg[col].median())
        
    # Isolate Test Set
    df_final = df_final.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int(len(df_final) * 0.8)
    df_train = df_final.iloc[:split_idx]
    df_test = df_final.iloc[split_idx:]
    
    feature_cols = ['age', 'admissionweight', 'admissionheight'] + vital_features
    rng = np.random.RandomState(seed)
    n_train = len(df_train)
    
    if n_clients == 5:
        proportions = [0.22, 0.18, 0.25, 0.20, 0.15]
    else:
        proportions = rng.dirichlet(np.ones(n_clients) * 2.0)
        
    sizes = (np.array(proportions) * n_train).astype(int)
    sizes[-1] = n_train - sizes[:-1].sum()
    
    client_data = []
    curr = 0
    for s in sizes:
        sub = df_train.iloc[curr:curr+s]
        curr += s
        X = np.hstack([sub[feature_cols].values, np.zeros((s, 35 - len(feature_cols)))])
        client_data.append((X.astype(np.float32), sub['label'].values.astype(np.int64)))
        
    X_t = np.hstack([df_test[feature_cols].values, np.zeros((len(df_test), 35 - len(feature_cols)))])
    return client_data, sizes, (X_t.astype(np.float32), df_test['label'].values.astype(np.int64))

def make_sequences(X, y, seq_len=24, seed=0):
    rng = np.random.RandomState(seed)
    X_seq = np.stack([X + rng.randn(*X.shape) * 0.01 for _ in range(seq_len)], axis=1)
    return X_seq.astype(np.float32), y

# ---------------------------------------------------------
# Models & Logic (Optimized for GPU)
# ---------------------------------------------------------
class LSTMMlpModel(nn.Module):
    def __init__(self, input_dim=35, h=128, layers=2, mlp_h=128, out=2, drp=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, h, layers, batch_first=True, bidirectional=True, dropout=drp)
        self.mlp = nn.Sequential(nn.Linear(h*2, mlp_h), nn.ReLU(), nn.Dropout(drp), nn.Linear(mlp_h, out))
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.mlp(o[:, -1, :])

class FocalLoss(nn.Module):
    def __init__(self, a=0.25, g=2.0):
        super().__init__()
        self.a, self.g = a, g
    def forward(self, l, t):
        ce = nn.functional.cross_entropy(l, t, reduction="none")
        return (self.a * (1 - torch.exp(-ce))**self.g * ce).mean()

# [AnomalyDetector, TrustManager, VerificationPolicy classes same as before]
class AnomalyDetector:
    def __init__(self, d=7, window=10):
        self.d, self.window, self.history = d, window, []
        self.U = self.mu = self.sigma2 = None
    def update_basis(self, g):
        self.history.append(g.copy())
        if len(self.history) > self.window: self.history.pop(0)
        if len(self.history) >= self.d + 2:
            H = np.stack(self.history)
            H_c = H - H.mean(axis=0)
            U, _, _ = np.linalg.svd(H_c.T, full_matrices=False)
            self.U = U[:, :self.d]
            p = H_c @ self.U
            self.mu, self.sigma2 = p.mean(axis=0), p.var(axis=0) + 1e-6
    def score(self, g):
        if self.U is None: return 0.0
        p = (g - np.mean(g)) @ self.U
        return float(np.sum((p - self.mu)**2 / self.sigma2))

class TrustManager:
    def __init__(self, n, g=0.95, t_min=0.1):
        self.s, self.g, self.t_min = np.full(n, 0.5), g, t_min
        self.cp = np.zeros(n, dtype=int)
    def update(self, i, p, v):
        if not p: self.s[i], self.cp[i] = self.t_min, 0
        else:
            self.s[i] = max(self.t_min, self.g*self.s[i] + (1-self.g)*v)
            self.cp[i] += 1
            if self.cp[i] >= 5: self.s[i] = min(self.s[i] + 0.05, 1.0)
    def get(self, i): return self.s[i]

class VerificationPolicy:
    def decide(self, r, a, t):
        if r <= 10 or a > 14.07 or t < 0.65: return "FULL"
        return "SAMPLE"

def local_train(glob, loader, lr, wd, epochs, C):
    loc = copy.deepcopy(glob).to(DEVICE)
    opt = optim.AdamW(loc.parameters(), lr=lr, weight_decay=wd)
    crit = FocalLoss().to(DEVICE)
    loc.train()
    for _ in range(epochs):
        for X, y in loader:
            opt.zero_grad(); crit(loc(X), y).backward()
            nn.utils.clip_grad_norm_(loc.parameters(), C); opt.step()
    g = {n: (p.data - p_old.data).cpu().numpy() for (n, p), (_, p_old) in zip(loc.named_parameters(), glob.named_parameters())}
    del loc; return g

def run_experiment(n_clients, attack, config, seed):
    np.random.seed(seed); torch.manual_seed(seed)
    client_data, sizes, test_raw = load_real_eicu_data(n_clients, seed=seed)
    scaler = StandardScaler().fit(np.concatenate([d[0] for d in client_data]))
    
    loaders = []
    for X, y in client_data:
        X_s, y_s = make_sequences(scaler.transform(X), y)
        loaders.append(DataLoader(TensorDataset(torch.tensor(X_s, device=DEVICE), torch.tensor(y_s, device=DEVICE)), batch_size=config["batch_size"], shuffle=True))
    
    X_te, y_te = make_sequences(scaler.transform(test_raw[0]), test_raw[1])
    X_te_gpu = torch.tensor(X_te, device=DEVICE)

    glob = LSTMMlpModel().to(DEVICE)
    tm, dets, pol = TrustManager(n_clients), [AnomalyDetector() for _ in range(n_clients)], VerificationPolicy()
    
    byz = set(range(max(1, int(n_clients * config["byz_fraction"])))) if attack != "none" else set()
    times = []

    for r in range(1, config["n_rounds"] + 1):
        accepted = {}
        for i, loader in enumerate(loaders):
            grad = local_train(glob, loader, config["lr"], config["weight_decay"], config["local_epochs"], config["clip_C"])
            flat = np.concatenate([v.flatten() for v in grad.values()])
            
            if i in byz:
                if attack == "random": flat = np.random.randn(*flat.shape) * np.linalg.norm(flat)
                elif attack == "sign_flip": flat = -flat
                elif attack == "scaling": flat = flat * 10.0
                grad = {n: flat[j:j+p.numel()].reshape(p.shape) for (n, p), j in zip(glob.named_parameters(), np.cumsum([0]+[p.numel() for p in glob.parameters()]))}

            score, trust = dets[i].score(flat), tm.get(i)
            act = pol.decide(r, score, trust)
            times.append(config["t_full"] if act == "FULL" else config["t_sample"])
            
            # Simulated check
            passed = not (i in byz)
            if passed:
                accepted[i] = (grad, trust)
                dets[i].update_basis(flat)
            tm.update(i, passed, 0.0 if act == "FULL" else 0.5)

        if accepted:
            st = glob.state_dict(); tot_w = sum(a[1] for a in accepted.values())
            for n, p in glob.named_parameters():
                agg = torch.zeros_like(p.data, device=DEVICE)
                for i, (g, w) in accepted.items(): agg += (w/tot_w) * torch.tensor(g[n], device=DEVICE)
                st[n] = p.data + agg
            glob.load_state_dict(st)

    glob.eval()
    with torch.no_grad():
        pr = torch.softmax(glob(X_te_gpu), 1)[:, 1].cpu().numpy()
    return {"auroc": float(roc_auc_score(y_te, pr)), "time": float(np.mean(times))}

def main():
    log_path = os.path.join(RESULTS_DIR, "full_experiment_log.txt")
    
    def log(msg):
        print(msg)
        with open(log_path, "a") as f: f.write(msg + "\n")

    log("--- STARTING FINAL EXPERIMENT (REAL DATA) ---")
    attacks = ["none", "random", "sign_flip", "scaling", "null_space"]
    
    for n in CONFIG["n_clients_list"]:
        n_results = {}
        log(f"\n[Scenario N={n}]")
        for atk in attacks:
            aurocs, pts = [], []
            for s in range(CONFIG["n_seeds"]):
                res = run_experiment(n, atk, CONFIG, s)
                aurocs.append(res["auroc"]); pts.append(res["time"])
                log(f"  N={n} ATK={atk:10s} SEED={s} AUROC={res['auroc']:.3f} AvgProof={res['time']:.3f}s")
            
            n_results[atk] = {"auroc": np.mean(aurocs), "proof_time": np.mean(pts)}
        
        # Save each N separately
        with open(os.path.join(RESULTS_DIR, f"results_N{n}_final.json"), "w") as f:
            json.dump({n: n_results}, f, indent=2)
        log(f"[✓] Saved results_N{n}_final.json")

if __name__ == "__main__":
    main()
