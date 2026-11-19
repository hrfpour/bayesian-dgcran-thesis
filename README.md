# Bayesian-DGCRAN — Master's Thesis Codebase (Phase 2: Full Models & Kaggle GPU Pipeline)

**Thesis Title:** *Evaluating and Enhancing Reliability in Road Traffic Forecasting by Integrating Spatio-Temporal Graph Neural Networks with Bayesian Approaches*  
**Author:** Hamidreza Farhadipour — Allameh Tabataba'i University (M.Sc. in Data Science)  
**Interactive Companion Site:** `https://thesis.farhadipour.com`

## Benchmark Datasets (All 6 Datasets — Matches Thesis Table 3.4)
| Dataset | Nodes ($N$) | Edges ($|\mathcal{E}|$) | Timesteps ($\mathcal{T}$) | Target | Split (Train/Val/Test) |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **PeMS03** | 358 | 547 | 26,208 | Traffic Flow (`veh/5min`) | 60 / 20 / 20 |
| **PeMS04** | 307 | 340 | 16,992 | Traffic Flow (`veh/5min`) | 60 / 20 / 20 |
| **PeMS07** | 883 | 866 | 28,224 | Traffic Flow (`veh/5min`) | 60 / 20 / 20 (`batch_size=16`, `AMP fp16`) |
| **PeMS08** | 170 | 295 | 17,856 | Traffic Flow (`veh/5min`) | 60 / 20 / 20 |
| **METR-LA** | 207 | 1,515 | 34,272 | Space-Mean Speed (`mph`) | 70 / 10 / 20 |
| **PEMS-BAY** | 325 | 2,369 | 52,116 | Space-Mean Speed (`mph`) | 70 / 10 / 20 |

## Repository Layout (Phase 2 Complete)
```
bayesian-dgcran-thesis/
├── configs/
│   └── datasets.yaml                  # Exact specs of all 6 datasets (PeMS03/04/07/08, METR-LA, PEMS-BAY)
├── notebooks/
│   └── kaggle_run_dataset.ipynb       # Ready-to-run Kaggle GPU notebook (Commit / 12-hr background run)
├── scripts/
│   ├── prepare_datasets.py            # Verifies raw files & shapes for all 6 datasets against Table 3.4
│   ├── smoke_test_gpu.py              # 10-second PyTorch GPU forward/backward/MC test for all models
│   └── run_experiment.py              # Main CLI runner (15 models/ablations + robustness missing-data study)
├── src/
│   ├── data/
│   │   ├── preprocessing.py           # Leak-free preprocessing, causal LOCF, calendar & sliding windows, PeMS03 ID map
│   │   └── loader.py                  # Unified NPZ/H5/CSV/PKL loader & PyTorch DataLoaders
│   ├── models/
│   │   ├── bayesian_layers.py         # V-NAPL (Local Reparameterization), B-DAGM (Concrete Mask), BDGCRU & B-AGCRU
│   │   ├── bayesian_dgcran.py         # Full Bayesian-DGCRAN with H-STDecoder & heteroscedastic log-var clamping
│   │   └── baselines.py               # HA, VAR, DCRNN, AGCRN, DGCRN, DGCRAN, MCDropout, DeepSTUQ, DeepEnsembles & 5 Ablations
│   ├── engine/
│   │   └── trainer.py                 # Algorithms 3-1 & 3-2 (Balanced ELBO, Cosine KL Annealing, AWA, AMP fp16)
│   ├── metrics/
│   │   ├── evaluation.py              # All 11 metrics (MAE, RMSE, MAPE, PICP, MPIW, MHPICE, MIS, MNLL, CRPS, ECE, Sparsification)
│   │   └── diebold_mariano.py         # Diebold-Mariano + Newey-West Bartlett + HLN finite-sample test
│   ├── calibration/
│   │   └── nemhcc.py                  # Algorithm 3-2: Non-Exchangeable Multi-Horizon Conformal Calibration
│   └── utils/
│       └── io_schema.py               # JSON/CSV exporter for Chapter 4 LaTeX tables & thesis.farhadipour.com
└── tests/
    └── test_phase1_core.py            # Automated unit test suite
```

## Quick Verification & Kaggle Execution
```bash
# 1. Verify raw datasets (all 6 benchmarks)
python3 scripts/prepare_datasets.py /kaggle/input/pems-metr-traffic-benchmarks

# 2. Run automated unit tests
python3 -m unittest discover -s tests -v

# 3. Run 10-second PyTorch GPU smoke test on all models
python3 scripts/smoke_test_gpu.py

# 4. Train & evaluate a model (example: Bayesian-DGCRAN on PeMS08)
python3 scripts/run_experiment.py --dataset PeMS08 --model Bayesian-DGCRAN --data_root /kaggle/input/pems-metr-traffic-benchmarks --output_dir /kaggle/working/results_PeMS08
```
