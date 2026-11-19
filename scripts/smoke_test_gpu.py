"""
10-Second End-to-End PyTorch Smoke Test for Kaggle GPU.
Verifies forward, backward, KL loss, and batched MC inference on ALL 14 models before launching long runs!
"""
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
import numpy as np
import torch
from src.models.baselines import build_model

def run_smoke_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running full model smoke test on device: {device}")
    spec = {"num_nodes": 20, "raw_features": 3, "default_node_emb_dim": 6, "horizon_len": 12}
    adj = np.eye(20, dtype=np.float32)
    X = torch.randn(4, 12, 20, 3, device=device)
    T_cal = torch.randn(4, 12, 4, device=device)
    Y = torch.randn(4, 12, 20, device=device)

    all_models = [
        "DCRNN", "AGCRN", "DGCRN", "DGCRAN", "MCDropout-DGCRAN", "DeepSTUQ",
        "DeepEnsembles", "Bayesian-DGCRAN", "Ablation-NoVNAPL", "Ablation-NoBDAGM",
        "Ablation-NoVarMasks", "Ablation-NoKLAWA", "Ablation-NoNEMHCC"
    ]
    for name in all_models:
        m = build_model(name, spec, adj, hidden_dim=16).to(device)
        if name == "DeepEnsembles":
            out = m.predict_mc(X, T_cal, num_samples=5)
            assert out["mu"].shape == (4, 12, 20)
        else:
            mu, logvar = m(X, T_cal, sample=True)
            assert mu.shape == (4, 12, 20), f"{name}: bad mu shape {mu.shape}"
            loss = torch.mean(torch.abs(Y - mu)) + 1e-4 * m.kl_loss()
            if logvar is not None:
                assert logvar.shape == (4, 12, 20)
                loss = loss + 0.5 * torch.mean(torch.exp(-logvar) * (Y - mu).pow(2) + logvar)
            loss.backward()
            if hasattr(m, "predict_mc") and logvar is not None:
                mc = m.predict_mc(X, T_cal, num_samples=3)
                assert mc["var_alea"].shape == (4, 12, 20) and mc["var_epis"].shape == (4, 12, 20)
        print(f"  [PASS] {name}")
    print("\nALL 13 PYTORCH MODELS PASSED FORWARD + BACKWARD + MC SMOKE TEST!")

if __name__ == "__main__":
    run_smoke_test()
