# Bayesian-DGCRAN — Master's Thesis Codebase (Phase 1: Core Infrastructure)

**Thesis Title:** *Evaluating and Enhancing Reliability in Road Traffic Forecasting by Integrating Spatio-Temporal Graph Neural Networks with Bayesian Approaches*  
**Author:** Hamidreza Farhadipour — Allameh Tabataba'i University (M.Sc. in Data Science)  
**Interactive Companion Site:** `https://thesis.farhadipour.com`

## Repository Layout (Phase 1 Complete)
```
bayesian-dgcran-thesis/
├── configs/
│   └── datasets.yaml              # Exact specs of all 5 datasets (matches Thesis Table 3.4)
├── scripts/
│   └── prepare_datasets.py        # Verifies & standardizes raw PeMS03/04/08 & METR-LA/PEMS-BAY files
├── src/
│   ├── data/
│   │   └── preprocessing.py       # Leak-free preprocessing, LOCF test imputation, calendar & sliding windows
│   ├── metrics/
│   │   ├── evaluation.py          # All 11 metrics (MAE, RMSE, MAPE, PICP, MPIW, MHPICE, MIS, MNLL, CRPS, ECE, Sparsification)
│   │   └── diebold_mariano.py     # Diebold-Mariano + Newey-West Bartlett + HLN finite-sample test
│   ├── calibration/
│   │   └── nemhcc.py              # Algorithm 3-2: Non-Exchangeable Multi-Horizon Conformal Calibration
│   └── utils/
│       └── io_schema.py           # JSON/CSV exporter for Chapter 4 LaTeX tables & thesis.farhadipour.com
├── tests/
│   └── test_phase1_core.py        # Full automated verification suite
└── notebooks/
    └── 01_kaggle_data_setup.ipynb # Ready-to-run Kaggle notebook to verify all 5 datasets
```

## Quick Verification
```bash
python3 -m unittest discover -s tests -v
```
