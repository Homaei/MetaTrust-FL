# Demo_eICU — eICU Demo Dataset Version

Dataset: eICU Collaborative Research Database Demo v2.0 (open-access)
Patients: 2,526  (training: ~1,993 | test: ~499)
Main script: Fl_real_data_final.py
Results: results/REPORTS_REAL_DATA/

Bug fixes applied to Fl_real_data_final.py:
  A. scaling attack now correctly applies 10× multiplication
  B. proof times updated from LSTM values (0.444s) to TCN values (0.101s)
  C. null_space Byzantine clients are now correctly rejected
