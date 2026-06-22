# MetaTrust-FL

**Adaptive Zero-Knowledge Verification for Federated Learning: A Meta-Learned Trust Policy with Byzantine Resilience**

> IEEE Transactions on Information Forensics and Security, 2026

---

## Authors

- **Mohammadhossein Homaei** — Universidad de Extremadura, Spain
- **Iman Khazrak** — Bowling Green State University, USA
- **Naghmeh Moradpoor** — Edinburgh Napier University, UK
- **Andrés Caro** — Universidad de Extremadura, Spain
- **Mar Ávila** — Universidad de Extremadura, Spain

---

## Overview

MetaTrust-FL is a federated learning framework that integrates **Groth16 zk-SNARKs** with a **meta-learned adaptive verification policy** to enforce end-to-end cryptographic guarantees on submitted model updates while controlling proof overhead.

### Key Results (eICU-CRD v2.0, 190,535 patients)

| Metric | Value |
|--------|-------|
| AUROC under 20% Byzantine attack | **0.725** |
| AUROC at steady-state (no attack) | **0.792** |
| Avg. proof time (ATBV) | **0.035 s** |
| Proof time reduction vs. static full-ZKP | **65%** |
| Byzantine detection rate | **99.7%** |
| False alarm rate | **0.0%** |
| DP privacy budget (ε) | **0.026** |

---

## Architecture

MetaTrust-FL combines three defense layers:

1. **ZKP Layer** — Unified Groth16 zk-SNARK circuit covering the complete TCN+MLP model (144,212 R1CS constraints, d=8)
2. **Statistical Layer** — PCA-Mahalanobis anomaly detection with server-side root of trust
3. **DP Layer** — Rényi Differential Privacy (σ=0.8, ε=0.026), orthogonal to ZKP

### Model

- **TCN Encoder**: 4 dilated causal convolutional blocks (k=3, dilation={1,2,4,8}, 64 channels) — 39,488 parameters
- **MLP Head**: 64→32→2 — 2,178 parameters
- **Total**: 41,666 parameters, 100% cryptographically covered

### Verification Policy

REINFORCE-based policy gradient (866 parameters) learns to assign:
- `FULL_ZKP` — 100% of model parameters verified (0.101 s)
- `SAMPLE_ZKP` — 10% random sample verified (0.010 s)

`HASH_CHECK` is excluded by design to guarantee cryptographic coverage at every round.

---

## Repository Structure

```
MetaTrust-FL/
│
├── Demo_eICU/                    ← eICU Demo dataset version
│   ├── Fl_real_data_final.py     (open-access, 2,526 patients)
│   └── README.md
│
├── Full_eICU_CRD_v2/             ← eICU-CRD v2.0 full dataset version
│   ├── Fl_full_FINAL.py          (credentialed access, 190,535 patients)
│   └── README.md
│
└── README.md                     ← this file
```

---

## Requirements

```bash
pip install torch numpy scikit-learn scipy matplotlib
```

---

## Dataset Access

### Demo Version (open-access)
```
https://physionet.org/content/eicu-crd-demo/2.0/
```
No credentials required.

### Full Version (credentialed access)
```
https://physionet.org/content/eicu-crd/2.0/
```
Requires PhysioNet credentialed access. Apply at the link above.

---

## Usage

### Demo Version
```bash
cd Demo_eICU
python3 Fl_real_data_final.py
```
Expected runtime: ~15–30 minutes on CPU.

### Full Version
```bash
cd Full_eICU_CRD_v2
python3 Fl_full_FINAL.py
```
Expected runtime: ~7–10 hours on i9-14900K + RTX 4070.
A cache file (`eicu_full_cache.npz`) is generated on first run (~2 hours) and reused automatically.

---

## Citation

```bibtex
@article{homaei2026metatrust,
  title   = {Adaptive Zero-Knowledge Verification for Federated Learning:
             A Meta-Learned Trust Policy with Byzantine Resilience},
  author  = {Homaei, Mohammadhossein and Khazrak, Iman and
             Moradpoor, Naghmeh and Caro, Andr{\'e}s and {\'A}vila, Mar},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2026}
}
```

---

## Acknowledgment

This activity has been co-financed 85% by the European Union, the European Regional Development Fund and the Regional Government of Extremadura. Grant File Number: GR24170.

---

## License

This code is released for research reproducibility purposes.
Clinical data (eICU-CRD) is subject to PhysioNet's data use agreement.
