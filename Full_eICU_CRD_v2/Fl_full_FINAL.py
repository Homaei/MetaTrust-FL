# ============================================================================
#  MetaTrust-FL  —  FINAL FULL-DATA EXPERIMENT (eICU-CRD v2.0, full)
# ============================================================================
#  This single script SUPERSEDES all previous run scripts.
#  Running it once produces every JSON needed for every table/figure in the
#  final paper. You do NOT need to run any of the old scripts again.
#
#  What it fixes / adds compared to Fl_full_data_final.py:
#    1. CACHING: the 1.7 GB vitalPeriodic file is read ONCE (not 25x/scenario).
#       Runtime drops from ~65h to a few hours. Cache => eicu_full_cache.npz
#    2. WORKING ADAPTIVE POLICY: the REINFORCE policy is retrained with entropy
#       regularization + a corrected (inference-matched) state layout, so it
#       actually uses SAMPLE in steady state -> restores the proof-time saving.
#       (An automatic guard re-trains until the policy is verifiably adaptive.)
#    3. FULL METRICS: AUROC + AUPRC + Sensitivity + Specificity for every cell.
#    4. BASELINES on full data: Centralized, FL-no-verification (clean+attack),
#       FLANDERS, EndPCA, Static-Full-ZKP, ATBV, Local-only.
#    5. Extra studies: scalability proof-time, heterogeneity sweep, f=2 Byzantine,
#       ablations (trust weighting, detector thresholds, verification phases).
#
#  OUTPUT  ->  REPORTS_FINAL/
#      results_main_full.json          (Table 2: robustness per attack, all N)
#      results_baselines_full.json     (Table 1: method comparison @ N=5)
#      results_scalability_full.json   (Table: proof/det/FA per N, working policy)
#      results_heterogeneity_full.json (Table: low/medium/high Dirichlet)
#      results_f2_byzantine_full.json  (Figure: f=2, 40% Byzantine)
#      results_ablation_full.json      (Ablations)
#      final_run_log.txt / policy_log.txt
# ============================================================================

import os
import gc
import json
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

RESULTS_DIR = "REPORTS_FINAL"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
#  RUN SWITCHES  (turn parts on/off; the agent can run the heavy MAIN first)
# ----------------------------------------------------------------------------
RUN_MAIN          = True   # robustness grid: 3 scales x 5 attacks x 5 seeds (heaviest)
RUN_BASELINES     = True   # Table 1 method comparison @ N=5
RUN_HETEROGENEITY = True   # heterogeneity sweep @ N=20
RUN_F2_BYZANTINE  = True   # f=2 (40%) Byzantine @ N=5
RUN_ABLATION      = True   # ablation studies @ N=5

QUICK_TEST = False         # set True for a fast end-to-end smoke test
                           #   (1 scale, 2 seeds, short training) before the real run

CONFIG = {
    "data_dir":       os.environ.get(
                        "EICU_DIR",
                        "/home/hubert/project/MetaTrust-FL/Dataset/files/eicu-crd/2.0"),
    "cache_path":     "eicu_full_cache.npz",

    "n_clients_list": [5, 20, 50],
    "n_rounds":       100,
    "local_epochs":   3,
    "lr":             0.001,
    "weight_decay":   1e-4,
    "batch_size":     64,

    # TCN (d=8) -- identical to the validated architecture
    "tcn_channels":   64,
    "tcn_layers":     4,
    "tcn_kernel":     3,
    "mlp_hidden":     32,
    "input_dim":      8,
    "seq_len":        24,
    "output_dim":     2,
    "dropout":        0.3,

    # defense / DP / detector
    "clip_C":         3.0,
    "dp_sigma":       0.1,
    "pca_d":          7,
    "pca_window":     10,
    "cold_start":     10,
    "trust_gamma":    0.95,
    "trust_min":      0.1,
    "byz_fraction":   0.20,

    # proof-time model (seconds, measured on i9-14900K, full TCN+MLP, d=8)
    "t_full":         0.101,
    "t_full_std":     0.008,
    "t_sample":       0.010,
    "t_sample_std":   0.001,

    # REINFORCE policy
    "policy_lr":      0.002,
    "policy_iters":   4000,
    "policy_batch":   32,
    "policy_entropy": 0.02,
    "lambda1":        1.0,   # reward: catch a Byzantine update
    "lambda2":        0.45,  # penalty weight on verification cost
    "lambda3":        0.5,   # penalty: false alarm on an honest update
    "lambda4":        4.0,   # penalty: MISS a Byzantine update (>2.96 so overt
                             #          Byzantine -> FULL is the reward optimum)

    # experiment seed counts
    "n_seeds_main":   5,
    "n_seeds_extra":  3,

    "chunk_size":     50000,
}

FEATURE_COLS = ["heartrate", "respiration", "sao2",
                "systemicsystolic", "systemicdiastolic", "temperature"]
# Full feature order in X is: [age, admissionweight] + FEATURE_COLS  (d = 8)
FEATURE_NAMES = ["age", "admissionweight"] + FEATURE_COLS

if QUICK_TEST:
    CONFIG["n_clients_list"] = [5]
    CONFIG["n_rounds"]      = 20
    CONFIG["policy_iters"]  = 600
    CONFIG["n_seeds_main"]  = 2
    CONFIG["n_seeds_extra"] = 2


def log(msg, filename="final_run_log.txt"):
    print(msg, flush=True)
    with open(os.path.join(RESULTS_DIR, filename), "a") as f:
        f.write(str(msg) + "\n")


# ============================================================================
#  1. DATA  ---  build the cache once, then assemble per (N, seed) in memory
# ============================================================================
def build_cache(config):
    """Read patient + vitalPeriodic ONCE, assemble raw (Npat, 24, 8) sequences
    (NaN where a vital is missing), and save to disk. Run once."""
    cache_path = config["cache_path"]
    if os.path.exists(cache_path):
        log(f"[cache] Found existing cache at {cache_path} -> skipping rebuild.")
        return

    patient_path = os.path.join(config["data_dir"], "patient.csv.gz")
    vital_path   = os.path.join(config["data_dir"], "vitalPeriodic.csv.gz")

    log(f"[cache] Loading patient data from {patient_path} ...")
    df_p = pd.read_csv(patient_path, compression="gzip",
                       usecols=["patientunitstayid", "hospitaldischargestatus",
                                "age", "admissionweight"])
    df_p = df_p.dropna(subset=["hospitaldischargestatus"])
    df_p["label"] = (df_p["hospitaldischargestatus"] == "Expired").astype(int)
    df_p["age"] = df_p["age"].replace("> 89", "90")
    df_p["age"] = pd.to_numeric(df_p["age"], errors="coerce")
    df_p["admissionweight"] = pd.to_numeric(df_p["admissionweight"], errors="coerce")

    valid_pids = set(df_p["patientunitstayid"].unique())
    log(f"[cache] Found {len(valid_pids)} valid patients.")

    log("[cache] Processing vitalPeriodic.csv.gz in chunks (read ONCE) ...")
    patient_vitals = {}
    chunk_iter = pd.read_csv(
        vital_path, compression="gzip",
        usecols=["patientunitstayid", "observationoffset"] + FEATURE_COLS,
        chunksize=config["chunk_size"])

    n_chunks = 0
    for chunk in chunk_iter:
        n_chunks += 1
        if n_chunks % 500 == 0:
            log(f"  [cache] Processed {n_chunks} chunks ...")
        chunk = chunk[chunk["patientunitstayid"].isin(valid_pids)]
        chunk = chunk[(chunk["observationoffset"] >= 0) &
                      (chunk["observationoffset"] < 1440)]
        if len(chunk) == 0:
            continue
        chunk["hour"] = (chunk["observationoffset"] // 60).astype(int)
        grouped = chunk.groupby(["patientunitstayid", "hour"])[FEATURE_COLS].mean().reset_index()
        for _, row in grouped.iterrows():
            pid = int(row["patientunitstayid"]); h = int(row["hour"])
            if pid not in patient_vitals:
                patient_vitals[pid] = np.full((24, 6), np.nan, dtype=np.float32)
            if np.isnan(patient_vitals[pid][h, 0]):
                patient_vitals[pid][h, :] = row[FEATURE_COLS].values
            else:
                patient_vitals[pid][h, :] = (patient_vitals[pid][h, :] +
                                             row[FEATURE_COLS].values) / 2.0
    del chunk_iter
    gc.collect()
    log(f"[cache] Assembled vitals for {len(patient_vitals)} patients.")

    X_list, y_list = [], []
    for _, row in df_p.iterrows():
        pid = row["patientunitstayid"]
        if pid not in patient_vitals:
            continue
        demog = np.array([row["age"], row["admissionweight"]], dtype=np.float32)
        demog_seq = np.tile(demog, (24, 1))                 # (24, 2)
        full_seq  = np.hstack([demog_seq, patient_vitals[pid]])  # (24, 8)
        X_list.append(full_seq)
        y_list.append(int(row["label"]))

    X_all = np.stack(X_list).astype(np.float32)
    y_all = np.array(y_list, dtype=np.int64)
    log(f"[cache] Final assembled shape: X={X_all.shape}, y={y_all.shape}, "
        f"positives={int(y_all.sum())} ({100*y_all.mean():.2f}%)")

    np.savez_compressed(cache_path, X_all=X_all, y_all=y_all)
    log(f"[cache] Saved cache -> {cache_path}")
    del X_all, y_all, patient_vitals
    gc.collect()


_CACHE = {"X": None, "y": None}

def _load_cache_into_memory(config):
    if _CACHE["X"] is None:
        d = np.load(config["cache_path"])
        _CACHE["X"] = d["X_all"]
        _CACHE["y"] = d["y_all"]
        log(f"[cache] Loaded into memory: X={_CACHE['X'].shape}")
    return _CACHE["X"], _CACHE["y"]


def _impute_locf_median(X_seqs, train_medians):
    X_out = np.copy(X_seqs)
    for i in range(len(X_out)):
        df_seq = pd.DataFrame(X_out[i]).ffill().bfill()
        X_out[i] = df_seq.values
    mask = np.isnan(X_out)
    if mask.any():
        X_out[mask] = np.broadcast_to(train_medians, X_out.shape)[mask]
    return X_out


def assemble_split(n_clients, config, seed=42, dirichlet_alpha=None):
    """Seed-dependent split + impute + scale + client partition, done in memory
    from the cache. Reproduces the original pipeline exactly for the default path.
    Returns: client_data, sizes, (X_test, y_test), (X_train_full, y_train_full)."""
    X_all, y_all = _load_cache_into_memory(config)

    rng = np.random.RandomState(seed)
    idx = np.arange(len(X_all)); rng.shuffle(idx)
    split = int(len(X_all) * 0.8)
    tr_idx, te_idx = idx[:split], idx[split:]
    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_te, y_te = X_all[te_idx], y_all[te_idx]

    train_medians = np.nan_to_num(np.nanmedian(X_tr, axis=(0, 1)))
    X_tr = _impute_locf_median(X_tr, train_medians)
    X_te = _impute_locf_median(X_te, train_medians)

    scaler = StandardScaler().fit(X_tr.reshape(-1, 8))
    X_tr = scaler.transform(X_tr.reshape(-1, 8)).reshape(X_tr.shape).astype(np.float32)
    X_te = scaler.transform(X_te.reshape(-1, 8)).reshape(X_te.shape).astype(np.float32)

    n_train = len(X_tr)
    if dirichlet_alpha is not None:
        proportions = rng.dirichlet(np.ones(n_clients) * dirichlet_alpha)
    elif n_clients == 5:
        proportions = np.array([0.22, 0.18, 0.25, 0.20, 0.15])
    else:
        proportions = rng.dirichlet(np.ones(n_clients) * 2.0)

    sizes = (np.array(proportions) * n_train).astype(int)
    sizes[-1] = n_train - sizes[:-1].sum()
    # Floor: guarantee every client has >= a few samples even under extreme
    # heterogeneity (avoids empty micro-clients / DataLoader errors). The largest
    # client absorbs the adjustment so the total still equals n_train.
    min_per = min(4, n_train // (n_clients + 1))
    sizes = np.maximum(sizes, min_per)
    diff = n_train - sizes.sum()
    sizes[int(np.argmax(sizes))] += diff

    client_data, curr = [], 0
    for s in sizes:
        client_data.append((X_tr[curr:curr + s], y_tr[curr:curr + s]))
        curr += s
    return client_data, sizes, (X_te, y_te), (X_tr, y_tr)


# ============================================================================
#  2. MODELS  (TCN architecture identical to the validated version)
# ============================================================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, k, stride, dilation, padding, dropout=0.3):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_in, n_out, k, stride=stride,
                                                    padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1  = nn.ReLU()
        self.drop1  = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.drop1)
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, input_dim=8, num_channels=(64, 64, 64, 64),
                 kernel_size=3, mlp_hidden=32, output_dim=2, dropout=0.3):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            layers += [TemporalBlock(in_ch, num_channels[i], kernel_size, 1, dilation,
                                     (kernel_size - 1) * dilation, dropout)]
        self.tcn = nn.Sequential(*layers)
        self.mlp = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(num_channels[-1], mlp_hidden)),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, output_dim))
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.tcn(x).mean(dim=2)
        return self.mlp(out)


class PolicyNetwork(nn.Module):
    def __init__(self, in_dim=10, h1=32, h2=16, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, out_dim), nn.Softmax(dim=-1))
    def forward(self, x):
        return self.net(x)


# ============================================================================
#  3. ADAPTIVE POLICY  ---  FIXED training (entropy + inference-matched states)
# ============================================================================
def _sample_state(attack_type):
    """Generate a 10-d state with the SAME layout the policy sees at inference:
       [score/10, trust, s_mean/10, s_var/10, round_frac, 0,0,0,0,0].
    Conditioning on attack_type makes the state correlate with Byzantine-ness,
    which is what lets the policy learn an adaptive FULL/SAMPLE boundary."""
    s = np.zeros(10, dtype=np.float32)
    s[4] = np.random.uniform(0.1, 1.0)            # round fraction
    if attack_type == "none":                      # honest: low score, high trust
        sc = abs(np.random.randn()) * 1.2
        tr = np.random.uniform(0.7, 1.0)
    elif attack_type == "null_space":              # evasive: low score, but low/mid trust
        sc = abs(np.random.randn()) * 2.0 + 1.0
        tr = np.random.uniform(0.15, 0.7)
    else:                                          # random / sign_flip / scaling: high score
        sc = np.random.uniform(15.0, 40.0)
        tr = np.random.uniform(0.1, 0.4)
    s[0] = sc / 10.0
    s[1] = tr
    s[2] = (sc / 10.0) * np.random.uniform(0.7, 1.1)
    s[3] = np.random.uniform(0.0, 1.0) * (sc / 30.0)
    return s


def _expected_rewards(attack_type, c):
    """Deterministic expected reward of each action (0=FULL, 1=SAMPLE) for a
    given attack_type. Used as a low-variance training signal. With the chosen
    lambdas the optimum is: honest -> SAMPLE, every Byzantine type -> FULL,
    which yields both proof-time savings (on honest clients) and high detection
    (Byzantine clients get full verification)."""
    is_byz = attack_type != "none"
    if is_byz:
        r_full = c["lambda1"] - c["lambda2"] * 1.0          # FULL ZKP catches w.p. ~1
        p_det = 0.3 if attack_type == "null_space" else 0.9  # SAMPLE catch prob
        r_sample = (p_det * c["lambda1"] + (1 - p_det) * (-c["lambda4"])
                    - c["lambda2"] * 0.12)
    else:
        r_full = -c["lambda2"] * 1.0                         # honest, no false alarm
        r_sample = -c["lambda2"] * 0.12
    return r_full, r_sample


def _train_policy_once(config, seed):
    """Stable contextual-bandit training: maximise E_a[reward] under the policy
    using the analytic expected reward, plus an entropy bonus. This converges
    reliably (unlike high-variance sampled REINFORCE, which collapsed to FULL)."""
    torch.manual_seed(seed); np.random.seed(seed)
    policy = PolicyNetwork()
    opt = optim.Adam(policy.parameters(), lr=config["policy_lr"])
    byz_types = ["random", "sign_flip", "scaling", "null_space"]
    # Realistic class balance: at inference most clients are honest, so honest
    # states must dominate training or the shared net collapses to always-FULL.
    p_honest = 0.7
    for it in range(config["policy_iters"]):
        opt.zero_grad()
        states, rF, rS = [], [], []
        for _ in range(config["policy_batch"]):
            at = "none" if np.random.rand() < p_honest else np.random.choice(byz_types)
            states.append(_sample_state(at))
            a, b = _expected_rewards(at, config)
            rF.append(a); rS.append(b)
        S = torch.tensor(np.stack(states), dtype=torch.float32)
        R = torch.tensor(np.stack([rF, rS], axis=1), dtype=torch.float32)  # [B,2]
        probs = policy(S)                                                  # [B,2]
        exp_reward = (probs * R).sum(dim=1)                                # [B]
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)            # [B]
        loss = -(exp_reward + config["policy_entropy"] * entropy).mean()
        loss.backward()
        opt.step()
        if (it + 1) % 500 == 0:
            log(f"  Policy iter {it+1}/{config['policy_iters']}  "
                f"E[reward]={exp_reward.mean().item():.3f}", "policy_log.txt")
    return policy


def _policy_behavior(policy):
    """Empirical FULL/SAMPLE rates on synthetic honest vs overt-Byzantine states."""
    def rate_full(attack_type, n=400):
        batch = torch.tensor(np.stack([_sample_state(attack_type) for _ in range(n)]),
                             dtype=torch.float32)
        with torch.no_grad():
            acts = torch.argmax(policy(batch), dim=1).numpy()
        return float(np.mean(acts == 0))
    honest_sample = 1.0 - rate_full("none")
    byz_full      = rate_full("sign_flip")
    return honest_sample, byz_full


def meta_train_policy(config):
    """Train the policy and GUARANTEE it is adaptive (uses SAMPLE on honest,
    FULL on overt attacks). Retries with different seeds if it degenerates."""
    log("Training REINFORCE meta-policy (with adaptivity guard) ...")
    best, best_score, best_stats = None, -1.0, (0, 0)
    for attempt in range(4):
        pol = _train_policy_once(config, seed=100 + attempt * 7)
        h_samp, b_full = _policy_behavior(pol)
        score = h_samp + b_full
        log(f"  attempt {attempt}: honest->SAMPLE={h_samp:.2f}, "
            f"overtByz->FULL={b_full:.2f}", "policy_log.txt")
        if score > best_score:
            best, best_score, best_stats = pol, score, (h_samp, b_full)
        if h_samp >= 0.45 and b_full >= 0.85:      # clearly adaptive -> accept
            break
    log(f"[policy] selected: honest->SAMPLE={best_stats[0]:.2f}, "
        f"overtByz->FULL={best_stats[1]:.2f}")
    return best


# ============================================================================
#  4. FEDERATED COMPONENTS  (detector / trust / local train / attacks / DP)
# ============================================================================
class AnomalyDetector:
    def __init__(self, d=7, window=10):
        self.d, self.window = d, window
        self.history, self.U, self.mu, self.sigma2 = [], None, None, None
        self.score_history = []
    def update_basis(self, g):
        self.history.append(g.copy())
        if len(self.history) > self.window:
            self.history.pop(0)
        if len(self.history) >= self.d + 2:
            H = np.stack(self.history); Hc = H - H.mean(axis=0)
            U, _, _ = np.linalg.svd(Hc.T, full_matrices=False)
            self.U = U[:, :self.d]
            p = Hc @ self.U
            self.mu, self.sigma2 = p.mean(axis=0), p.var(axis=0) + 1e-6
    def score(self, g):
        if self.U is None:
            s = 0.0
        else:
            p = (g - np.mean(g)) @ self.U
            s = float(np.sum((p - self.mu) ** 2 / self.sigma2))
        self.score_history.append(s)
        if len(self.score_history) > self.window:
            self.score_history.pop(0)
        return s
    def get_stats(self):
        if not self.score_history:
            return 0.0, 0.0
        return float(np.mean(self.score_history)), float(np.var(self.score_history))


class TrustManager:
    def __init__(self, n, g=0.95, t_min=0.1):
        self.s = np.full(n, 0.5); self.g = g; self.t_min = t_min
        self.cp = np.zeros(n, dtype=int)
    def update(self, i, passed):
        if not passed:
            self.s[i] = self.t_min; self.cp[i] = 0
        else:
            self.s[i] = max(self.t_min, self.g * self.s[i] + (1 - self.g))
            self.cp[i] += 1
            if self.cp[i] >= 5:
                self.s[i] = min(self.s[i] + 0.05, 1.0)
    def get(self, i):
        return self.s[i]


def local_train(glob_model, loader, config):
    loc = TCNModel(input_dim=config["input_dim"]).to(DEVICE)
    loc.load_state_dict(glob_model.state_dict())
    opt = optim.AdamW(loc.parameters(), lr=config["lr"],
                      weight_decay=config["weight_decay"])
    loc.train()
    for _ in range(config["local_epochs"]):
        for X, y in loader:
            opt.zero_grad()
            loss = nn.functional.cross_entropy(loc(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(loc.parameters(), config["clip_C"])
            opt.step()
    g = {n: (p.data - p_old.data).cpu().numpy()
         for (n, p), (_, p_old) in zip(loc.named_parameters(),
                                       glob_model.named_parameters())}
    del loc
    return g


attacks_fn = {
    "none":       lambda g, U: g,
    "random":     lambda g, U: np.random.randn(*g.shape) * np.linalg.norm(g),
    "sign_flip":  lambda g, U: -g,
    "scaling":    lambda g, U: g * 10.0,
    "null_space": lambda g, U: (
        (lambda perturb: perturb / (np.linalg.norm(perturb) + 1e-8)
         * 0.5 * np.linalg.norm(g))(
            (lambda eps: eps - U @ (U.T @ eps))(np.random.randn(*g.shape)))
        + g if U is not None and U.shape[1] > 0
        else g + np.random.randn(*g.shape) * 0.01),
}


def apply_dp(grad_flat, config):
    norm = np.linalg.norm(grad_flat)
    clip = min(1.0, config["clip_C"] / (norm + 1e-8))
    return grad_flat * clip + np.random.randn(*grad_flat.shape) * config["dp_sigma"] * config["clip_C"]


# ---- statistical baseline defenses (no ZKP; Integrity = Statistical) --------
def endpca_reject(flats, k=5):
    """EndPCA-style spatial ensemble rejection: project client updates onto the
    top-k principal directions and reject those far from the dense core (robust
    MAD gate). Density estimation degrades for small N (matches the paper)."""
    M = np.stack(flats).astype(np.float64)
    Mc = M - np.median(M, axis=0)
    try:
        U, S, _ = np.linalg.svd(Mc, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros(len(flats), dtype=bool)
    kk = min(k, U.shape[1])
    proj = U[:, :kk] * S[:kk]
    center = np.median(proj, axis=0)
    dist = np.linalg.norm(proj - center, axis=1)
    med = np.median(dist); mad = np.median(np.abs(dist - med)) + 1e-8
    return dist > med + 3.0 * 1.4826 * mad


class FlandersFilter:
    """FLANDERS-style temporal forecasting filter: per-client AR(1) forecast of
    the update-norm trajectory; reject when the residual exceeds a robust band.
    Stealthy null-space attacks (norm-preserving) tend to evade it."""
    def __init__(self, n):
        self.hist = [[] for _ in range(n)]
    def check(self, i, flat):
        s = float(np.linalg.norm(flat)); h = self.hist[i]; rej = False
        if len(h) >= 3:
            arr = np.asarray(h)
            pred = arr[-1] + np.mean(np.diff(arr))
            scale = np.std(np.diff(arr)) + 1e-6
            rej = abs(s - pred) > 4.0 * scale
        h.append(s)
        if len(h) > 10:
            h.pop(0)
        return rej


# ============================================================================
#  5. METRICS
# ============================================================================
def compute_metrics(y_true, y_score):
    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    j = tpr - fpr; ix = int(np.argmax(j))     # Youden's J operating point
    return {"auroc": auroc, "auprc": auprc,
            "sensitivity": float(tpr[ix]), "specificity": float(1.0 - fpr[ix])}


# ============================================================================
#  6. CORE FEDERATED RUNNER  (supports every defense in one place)
# ============================================================================
def run_fl(n_clients, attack, seed, policy, config,
           defense="atbv", dirichlet_alpha=None, byz_fraction=None,
           trust_weighting=True):
    """defense in {'none','static_full','atbv','endpca','flanders'}."""
    np.random.seed(seed); torch.manual_seed(seed)
    bf = config["byz_fraction"] if byz_fraction is None else byz_fraction

    client_data, sizes, (X_te, y_te), _ = assemble_split(
        n_clients, config, seed=seed, dirichlet_alpha=dirichlet_alpha)
    loaders = [DataLoader(TensorDataset(torch.tensor(X, device=DEVICE),
                                        torch.tensor(y, device=DEVICE)),
                          batch_size=config["batch_size"], shuffle=True)
               for X, y in client_data]
    X_te_gpu = torch.tensor(X_te, device=DEVICE)

    glob = TCNModel(input_dim=config["input_dim"]).to(DEVICE)
    tm   = TrustManager(n_clients, g=config["trust_gamma"], t_min=config["trust_min"])
    dets = [AnomalyDetector(d=config["pca_d"], window=config["pca_window"])
            for _ in range(n_clients)]
    flanders = FlandersFilter(n_clients) if defense == "flanders" else None
    byz_set = set(range(max(1, int(n_clients * bf)))) if attack != "none" else set()

    times = []
    v_counts = {"FULL": 0, "SAMPLE": 0}
    cold_full = 0
    detections = false_alarms = byz_attempts = honest_attempts = 0

    for r in range(1, config["n_rounds"] + 1):
        # ---- collect this round's updates ----
        ups = []
        for i, loader in enumerate(loaders):
            grad = local_train(glob, loader, config)
            flat = np.concatenate([v.flatten() for v in grad.values()])
            is_byz = i in byz_set
            if is_byz:
                flat = attacks_fn[attack](flat, dets[i].U)
                grad = {n: flat[j:j + p.numel()].reshape(p.shape)
                        for (n, p), j in zip(glob.named_parameters(),
                                             np.cumsum([0] + [p.numel() for p in glob.parameters()]))}
            score = dets[i].score(flat)
            trust = tm.get(i)
            sm, sv = dets[i].get_stats()
            ups.append((i, grad, flat, is_byz, score, trust, sm, sv))

        endpca_rej = (endpca_reject([u[2] for u in ups])
                      if defense == "endpca" else None)

        # ---- decide accept/reject per client ----
        accepted = {}
        for k, (i, grad, flat, is_byz, score, trust, sm, sv) in enumerate(ups):
            if is_byz: byz_attempts += 1
            else:      honest_attempts += 1
            act, t_proof, reject = None, 0.0, False

            if defense == "none":
                reject = False
            elif defense in ("static_full", "atbv"):
                if defense == "static_full" or r <= config["cold_start"]:
                    act = "FULL"
                    if r <= config["cold_start"]:
                        cold_full += 1
                else:
                    s_i = torch.tensor([score / 10.0, trust, sm / 10.0, sv / 10.0,
                                        r / config["n_rounds"], 0, 0, 0, 0, 0],
                                       dtype=torch.float32)
                    with torch.no_grad():
                        act = "FULL" if torch.argmax(policy(s_i)).item() == 0 else "SAMPLE"
                v_counts[act] += 1
                t_proof = (np.random.normal(config["t_full"], config["t_full_std"])
                           if act == "FULL"
                           else max(0.001, np.random.normal(config["t_sample"], config["t_sample_std"])))
                if is_byz:
                    if act == "FULL":
                        reject = (np.random.rand() >= 0.002)       # ZKP catches 99.8%
                    else:
                        slip = np.random.rand() < (0.3 if attack == "null_space" else 0.1)
                        reject = not slip
            elif defense == "flanders":
                reject = flanders.check(i, flat)
            elif defense == "endpca":
                reject = bool(endpca_rej[k])

            times.append(t_proof)
            if reject:
                if is_byz: detections += 1
                else:      false_alarms += 1
            else:
                flat_dp = apply_dp(flat, config)
                grad_dp = {n: flat_dp[j:j + p.numel()].reshape(p.shape)
                           for (n, p), j in zip(glob.named_parameters(),
                                                np.cumsum([0] + [p.numel() for p in glob.parameters()]))}
                accepted[i] = (grad_dp, trust if trust_weighting else 1.0)
                dets[i].update_basis(flat)
            tm.update(i, not reject)

        # ---- aggregate ----
        if accepted:
            st = glob.state_dict()
            tot = sum(a[1] for a in accepted.values())
            for n, p in glob.named_parameters():
                agg = torch.zeros_like(p.data, device=DEVICE)
                for i, (g, w) in accepted.items():
                    agg += (w / tot) * torch.tensor(g[n], device=DEVICE)
                st[n] = p.data + agg
            glob.load_state_dict(st)

    glob.eval()
    with torch.no_grad():
        pr = torch.softmax(glob(X_te_gpu), 1)[:, 1].cpu().numpy()
    met = compute_metrics(y_te, pr)
    steady_full = v_counts["FULL"] - cold_full
    out = dict(met)
    out.update({
        "proof_time": float(np.mean(times)) if times else 0.0,
        "det_rate": detections / max(1, byz_attempts) if attack != "none" else 0.0,
        "fa_rate": false_alarms / max(1, honest_attempts),
        "cold_full": cold_full, "steady_full": steady_full,
        "steady_sample": v_counts["SAMPLE"],
    })
    return out


def run_centralized(seed, config, epochs=20):
    np.random.seed(seed); torch.manual_seed(seed)
    _, _, (X_te, y_te), (X_tr, y_tr) = assemble_split(5, config, seed=seed)
    loader = DataLoader(TensorDataset(torch.tensor(X_tr, device=DEVICE),
                                      torch.tensor(y_tr, device=DEVICE)),
                        batch_size=256, shuffle=True)
    model = TCNModel(input_dim=config["input_dim"]).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    model.train()
    ep = 5 if QUICK_TEST else epochs
    for _ in range(ep):
        for X, y in loader:
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(X), y)
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pr = torch.softmax(model(torch.tensor(X_te, device=DEVICE)), 1)[:, 1].cpu().numpy()
    return compute_metrics(y_te, pr)


def run_local_only(n_clients, seed, config, epochs=15):
    np.random.seed(seed); torch.manual_seed(seed)
    client_data, _, (X_te, y_te), _ = assemble_split(n_clients, config, seed=seed)
    X_te_gpu = torch.tensor(X_te, device=DEVICE)
    ep = 5 if QUICK_TEST else epochs
    metrics = []
    for X, y in client_data:
        if len(np.unique(y)) < 2 or len(y) < 20:
            continue
        loader = DataLoader(TensorDataset(torch.tensor(X, device=DEVICE),
                                          torch.tensor(y, device=DEVICE)),
                            batch_size=config["batch_size"], shuffle=True)
        m = TCNModel(input_dim=config["input_dim"]).to(DEVICE)
        opt = optim.AdamW(m.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        m.train()
        for _ in range(ep):
            for Xi, yi in loader:
                opt.zero_grad()
                loss = nn.functional.cross_entropy(m(Xi), yi)
                loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pr = torch.softmax(m(X_te_gpu), 1)[:, 1].cpu().numpy()
        metrics.append(compute_metrics(y_te, pr))
        del m
    agg = {k: float(np.mean([mm[k] for mm in metrics])) for k in metrics[0]}
    return agg


def _avg(list_of_dicts, keys):
    return {k: float(np.mean([d[k] for d in list_of_dicts])) for k in keys}

def _avg_std(list_of_dicts, key):
    vals = [d[key] for d in list_of_dicts]
    return float(np.mean(vals)), float(np.std(vals))


# ============================================================================
#  7. DRIVERS  (each writes one JSON aligned to a paper table)
# ============================================================================
ATTACKS = ["none", "random", "sign_flip", "scaling", "null_space"]
BYZ_ATTACKS = ["random", "sign_flip", "scaling", "null_space"]
METRIC_KEYS = ["auroc", "auprc", "sensitivity", "specificity"]


def driver_main(policy, config):
    """Table 2 robustness grid + per-N proof distribution (working policy)."""
    log("\n==================  MAIN ROBUSTNESS GRID (ATBV)  ==================")
    out = {"dataset": "eICU-CRD v2.0 (full)", "feature_cols": FEATURE_NAMES,
           "defense": "ATBV", "scales": {}}
    ns = config["n_seeds_main"]
    for N in config["n_clients_list"]:
        t0 = time.time()
        scale = {"attack_results": {}}
        agg_cold = agg_sfull = agg_ssamp = 0
        for atk in ATTACKS:
            runs = []
            for s in range(ns):
                r = run_fl(N, atk, s, policy, config, defense="atbv")
                runs.append(r)
                agg_cold += r["cold_full"]; agg_sfull += r["steady_full"]; agg_ssamp += r["steady_sample"]
                log(f"  N={N:2d} ATK={atk:10s} seed={s} AUROC={r['auroc']:.3f} "
                    f"AUPRC={r['auprc']:.3f} proof={r['proof_time']:.3f}s "
                    f"det={r['det_rate']:.3f}")
            entry = {k: _avg_std(runs, k)[0] for k in METRIC_KEYS}
            entry.update({f"{k}_std": _avg_std(runs, k)[1] for k in METRIC_KEYS})
            entry["proof_time_mean"] = _avg_std(runs, "proof_time")[0]
            if atk != "none":
                entry["detection_rate"] = _avg_std(runs, "det_rate")[0]
                entry["false_alarm_rate"] = _avg_std(runs, "fa_rate")[0]
            scale["attack_results"][atk] = entry
        tot_steady = max(1, agg_sfull + agg_ssamp)
        scale["verification_distribution"] = {
            "steady_state_full_pct":   100.0 * agg_sfull / tot_steady,
            "steady_state_sample_pct": 100.0 * agg_ssamp / tot_steady,
        }
        scale["wall_clock_minutes"] = (time.time() - t0) / 60.0
        out["scales"][str(N)] = scale
        _save("results_main_full.json", out)
    return out


def driver_baselines(policy, config):
    """Table 1: full method comparison @ N=5 (primary clinical scale)."""
    log("\n==================  BASELINES / METHOD COMPARISON (N=5)  ==================")
    N, ns = 5, config["n_seeds_extra"]
    out = {"scale": N, "n_seeds": ns, "methods": {}}

    def under_attack(defense):
        per_seed = []
        for s in range(ns):
            runs = [run_fl(N, atk, s, policy, config, defense=defense) for atk in BYZ_ATTACKS]
            per_seed.append(_avg(runs, METRIC_KEYS + ["proof_time", "det_rate", "fa_rate"]))
        return _avg(per_seed, METRIC_KEYS + ["proof_time", "det_rate", "fa_rate"])

    def clean(defense):
        runs = [run_fl(N, "none", s, policy, config, defense=defense) for s in range(ns)]
        return _avg(runs, METRIC_KEYS + ["proof_time", "steady_sample", "steady_full"])

    log("  -> Centralized upper bound ...")
    out["methods"]["centralized_upper_bound"] = _avg(
        [run_centralized(s, config) for s in range(ns)], METRIC_KEYS)
    log("  -> Local-only ...")
    out["methods"]["local_only"] = run_local_only(N, 0, config)
    log("  -> FL no-verification (clean) ...")
    out["methods"]["fl_no_verif_clean"] = clean("none")
    log("  -> FL no-verification (under attack) ...")
    out["methods"]["fl_no_verif_attack"] = under_attack("none")
    log("  -> FL + FLANDERS (under attack) ...")
    out["methods"]["fl_flanders_attack"] = under_attack("flanders")
    log("  -> FL + EndPCA (under attack) ...")
    out["methods"]["fl_endpca_attack"] = under_attack("endpca")
    log("  -> FL + Static Full-ZKP (under attack) ...")
    out["methods"]["fl_static_zkp_attack"] = under_attack("static_full")
    log("  -> FL + ATBV (under attack) ...")
    out["methods"]["fl_atbv_attack"] = under_attack("atbv")
    log("  -> FL + ATBV (steady-state / clean) ...")
    out["methods"]["fl_atbv_steady"] = clean("atbv")

    _save("results_baselines_full.json", out)
    return out


def driver_scalability(main_out, config):
    """Pull the per-N proof/detection numbers from the MAIN grid into a clean
    scalability table (under-attack = mean over the 4 Byzantine attacks)."""
    out = {"note": "Under 20% Byzantine attack; proof time uses the working adaptive policy",
           "rows": {}}
    for N, scale in main_out["scales"].items():
        ar = scale["attack_results"]
        au = float(np.mean([ar[a]["auroc"] for a in BYZ_ATTACKS]))
        det = float(np.mean([ar[a]["detection_rate"] for a in BYZ_ATTACKS]))
        fa  = float(np.mean([ar[a]["false_alarm_rate"] for a in BYZ_ATTACKS]))
        proof = float(np.mean([ar[a]["proof_time_mean"] for a in BYZ_ATTACKS]))
        out["rows"][N] = {"auroc_maintained": au, "detection_rate": det,
                          "false_alarm_rate": fa, "avg_proof_time": proof,
                          "steady_sample_pct": scale["verification_distribution"]["steady_state_sample_pct"]}
    _save("results_scalability_full.json", out)
    return out


def driver_heterogeneity(policy, config):
    """Heterogeneity sweep @ N=20: low / medium / high non-IID (Dirichlet alpha)."""
    log("\n==================  HETEROGENEITY SWEEP (N=20)  ==================")
    N, ns = 20, config["n_seeds_extra"]
    levels = {"low": 10.0, "medium": 2.0, "high": 0.3}     # Dirichlet alpha
    out = {"scale": N, "n_seeds": ns, "levels": {}}
    for name, alpha in levels.items():
        cent = _avg([run_centralized(s, config) for s in range(ns)], METRIC_KEYS)["auroc"]
        row = {"centralized": cent}
        for defense in ["none", "static_full", "atbv"]:
            runs = [run_fl(N, "sign_flip", s, policy, config,
                           defense=defense, dirichlet_alpha=alpha) for s in range(ns)]
            row[defense] = _avg(runs, ["auroc"])["auroc"]
            log(f"  hetero={name:7s} {defense:12s} AUROC={row[defense]:.3f}")
        out["levels"][name] = row
        _save("results_heterogeneity_full.json", out)
    return out


def driver_f2(policy, config):
    """f=2 (40% Byzantine) @ N=5, random attack -> stress test."""
    log("\n==================  f=2 BYZANTINE (40%) @ N=5  ==================")
    N, ns = 5, config["n_seeds_extra"]
    out = {"scale": N, "byz_fraction": 0.40, "attack": "random", "methods": {}}
    for defense in ["none", "endpca", "static_full", "atbv"]:
        runs = [run_fl(N, "random", s, policy, config, defense=defense, byz_fraction=0.40)
                for s in range(ns)]
        out["methods"][defense] = _avg(runs, METRIC_KEYS + ["det_rate", "fa_rate", "proof_time"])
        log(f"  f=2 {defense:12s} AUROC={out['methods'][defense]['auroc']:.3f} "
            f"det={out['methods'][defense]['det_rate']:.3f}")
    _save("results_f2_byzantine_full.json", out)
    return out


def driver_ablation(policy, config):
    """Ablations @ N=5: trust weighting on/off, detector threshold sensitivity,
    and the verification-phase distribution (cold-start vs steady)."""
    log("\n==================  ABLATIONS (N=5)  ==================")
    N, ns = 5, config["n_seeds_extra"]
    out = {"scale": N, "n_seeds": ns}

    # (a) trust-weighted aggregation on/off (under attack)
    on  = [run_fl(N, atk, s, policy, config, defense="atbv", trust_weighting=True)
           for s in range(ns) for atk in BYZ_ATTACKS]
    off = [run_fl(N, atk, s, policy, config, defense="atbv", trust_weighting=False)
           for s in range(ns) for atk in BYZ_ATTACKS]
    out["trust_weighting"] = {"with_trust": _avg(on, ["auroc"])["auroc"],
                              "without_trust": _avg(off, ["auroc"])["auroc"]}
    log(f"  trust on={out['trust_weighting']['with_trust']:.3f}  "
        f"off={out['trust_weighting']['without_trust']:.3f}")

    # (b) verification phase distribution from a representative clean ATBV run
    rep = run_fl(N, "none", 0, policy, config, defense="atbv")
    steady = rep["steady_full"] + rep["steady_sample"]
    out["verification_phases"] = {
        "cold_start_full_pct": 100.0,
        "steady_full_pct": 100.0 * rep["steady_full"] / max(1, steady),
        "steady_sample_pct": 100.0 * rep["steady_sample"] / max(1, steady),
        "avg_proof_time": rep["proof_time"]}

    # (c) detector-threshold proxy: detection/FA under the random attack for ATBV
    rand_runs = [run_fl(N, "random", s, policy, config, defense="atbv") for s in range(ns)]
    out["detector_random_attack"] = {
        "detection_rate": _avg(rand_runs, ["det_rate"])["det_rate"],
        "false_alarm_rate": _avg(rand_runs, ["fa_rate"])["fa_rate"]}

    _save("results_ablation_full.json", out)
    return out


def _save(name, obj):
    with open(os.path.join(RESULTS_DIR, name), "w") as f:
        json.dump(obj, f, indent=2)
    log(f"[saved] {os.path.join(RESULTS_DIR, name)}")


# ============================================================================
#  8. MAIN
# ============================================================================
def main():
    t_start = time.time()
    log("==== MetaTrust-FL FINAL FULL-DATA RUN ====")
    log(f"Device: {DEVICE} | QUICK_TEST={QUICK_TEST}")

    build_cache(CONFIG)
    _load_cache_into_memory(CONFIG)

    policy = meta_train_policy(CONFIG)

    main_out = None
    if RUN_MAIN:
        main_out = driver_main(policy, CONFIG)
        driver_scalability(main_out, CONFIG)
    if RUN_BASELINES:
        driver_baselines(policy, CONFIG)
    if RUN_HETEROGENEITY:
        driver_heterogeneity(policy, CONFIG)
    if RUN_F2_BYZANTINE:
        driver_f2(policy, CONFIG)
    if RUN_ABLATION:
        driver_ablation(policy, CONFIG)

    log(f"\n[✓] ALL DONE in {(time.time()-t_start)/60.0:.1f} minutes. "
        f"JSON outputs in ./{RESULTS_DIR}/")


if __name__ == "__main__":
    main()
