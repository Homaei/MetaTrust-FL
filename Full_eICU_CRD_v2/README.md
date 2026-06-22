# Full_eICU_CRD_v2 — eICU-CRD v2.0 Full Dataset Version

Dataset: eICU Collaborative Research Database v2.0 (credentialed access via PhysioNet)
Patients: 190,535  (training: 152,428 | test: 38,107)
Main script: Fl_full_FINAL.py
Results: results/REPORTS_FINAL/
Cache: eicu_full_cache.npz  (1.7 GB — do not delete, saves ~2h preprocessing)

Key results (5 seeds, N=5, 20% Byzantine attack):
  Centralized upper bound AUROC : 0.826
  FL + ATBV (under attack) AUROC: 0.725
  FL + ATBV (steady-state) AUROC: 0.792
  Static Full-ZKP proof time   : 0.101 s
  ATBV avg proof time          : 0.035 s  (65% reduction)
  Byzantine detection rate     : 99.7%
  False alarm rate             : 0.0%
