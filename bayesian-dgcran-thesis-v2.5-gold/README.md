# Bayesian-DGCRAN — Master's Thesis Complete Experimental Suite (v2.0)

**Thesis Title:** *Evaluating and Enhancing Reliability in Road Traffic Forecasting by Integrating Spatio-Temporal Graph Neural Networks with Bayesian Approaches*  
**Author:** Hamidreza Farhadipour — Allameh Tabataba'i University (M.Sc. in Data Science)  
**Interactive Companion Site:** `https://thesis.farhadipour.com`

## 1. Complete 21-Model Experimental Suite (Matches Chapters 2 & 3)
1. **Classical & Deep Sequence Time-Series (No Graph):** `HA`, `VAR`, `SARIMA`, `FC-LSTM`
2. **STGNN Generation 1 (Static Graph):** `STGCN`, `DCRNN`, `ASTGCN`
3. **STGNN Generation 2 (Adaptive Graph):** `GraphWaveNet`, `AGCRN`
4. **STGNN Generation 3 (Dynamic Graph & Transformer):** `DGCRN`, `STAEformer`, `DGCRAN`
5. **Probabilistic & Bayesian UQ Baselines:** `MCDropout-DGCRAN`, `DeepSTUQ`, `DeepEnsembles`
6. **Proposed Model & 5 Component Ablations:** `Bayesian-DGCRAN`, `Ablation-NoVNAPL`, `Ablation-NoBDAGM`, `Ablation-NoVarMasks`, `Ablation-NoKLAWA`, `Ablation-NoNEMHCC` (+ 10%–50% missing-data robustness study)

## 2. Benchmark Datasets (All 6 Datasets — Matches Thesis Table 3.4)
| Dataset | Nodes ($N$) | Edges ($|\mathcal{E}|$) | Timesteps ($\mathcal{T}$) | Target | Split |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **PeMS03** | 358 | 547 | 26,208 | Traffic Flow (`veh/5min`) | 60 / 20 / 20 |
| **PeMS04** | 307 | 340 | 16,992 | Traffic Flow (`veh/5min`) | 60 / 20 / 20 |
| **PeMS07** | 883 | 866 | 28,224 | Traffic Flow (`veh/5min`) | 60 / 20 / 20 (`batch=16`, `AMP`) |
| **PeMS08** | 170 | 295 | 17,856 | Traffic Flow (`veh/5min`) | 60 / 20 / 20 |
| **METR-LA** | 207 | 1,515 | 34,272 | Space-Mean Speed (`mph`) | 70 / 10 / 20 |
| **PEMS-BAY** | 325 | 2,369 | 52,116 | Space-Mean Speed (`mph`) | 70 / 10 / 20 |

## 3. Repository Layout
```
bayesian-dgcran-thesis/
├── configs/
│   ├── datasets.yaml                        # Exact specs of all 6 datasets
│   └── published_benchmarks.yaml            # Verified published paper metrics (Li 2025, Liu 2023)
├── notebooks/
│   ├── 01_kaggle_baselines_verification.ipynb  # Notebook A: 12 prior models + literature verification
│   └── 02_kaggle_bayesian_proposed.ipynb       # Notebook B: Bayesian baselines, Proposed, 5 Ablations & Robustness
├── scripts/
│   ├── prepare_datasets.py                  # Verifies all 6 raw datasets
│   ├── smoke_test_gpu.py                    # 15-sec PyTorch GPU test across all 18 neural models
│   ├── run_experiment.py                    # Unified CLI runner + JSON/CSV/trajectory/DM-error export
│   └── compare_with_literature.py           # Side-by-side table: Ours vs. Published Papers
├── src/
│   ├── data/                                # Leak-free preprocessing, PeMS03 ID map, DataLoaders
│   ├── models/                              # All 21 models (Classical, Gen 1-3 STGNN, Bayesian, Ablations)
│   ├── engine/                              # Algorithms 3-1 & 3-2 (KL Annealing, AWA, AMP fp16)
│   ├── metrics/                             # 11 metrics + Diebold-Mariano / Newey-West / HLN
│   ├── calibration/                         # Algorithm 3-2: NE-MHCC
│   └── utils/                               # Web & Chapter 4 JSON/CSV exporter
└── tests/
    └── test_phase1_core.py                  # Automated unit test suite
```
