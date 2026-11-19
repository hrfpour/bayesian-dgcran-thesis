"""
End-to-End Training & Evaluation Engine Strictly Implementing Algorithms 3-1 and 3-2 of Chapter 3:
  - Phase 1 Pre-training with cosine KL Annealing (Eq. 3.44), GNLL + lambda_reg * MAE + KL / (M_train * N * T')
  - Gradient norm clipping (||grad||_2 <= 5.0) & optional AMP fp16 (for PeMS07 N=883)
  - Phase 2 Adaptive Weight Averaging (AWA, Eq. 3.52): odd epochs cosine escape, even epochs flat convergence & averaging
  - Monte Carlo Inference (M=10 parallel passes) + NE-MHCC Online Rolling Calibration + All 11 Metrics
"""
from __future__ import annotations
import copy, math, time
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

from src.calibration.nemhcc import NEMHCCCalibrator
from src.metrics.evaluation import evaluate_all_metrics, compute_point_metrics


def kl_annealing_beta(epoch: int, warmup_epochs: int, beta_max: float = 1.0) -> float:
    """Cosine KL Annealing Schedule (Chapter 3 Eq. 3.44)."""
    if warmup_epochs <= 0:
        return beta_max
    progress = min(epoch, warmup_epochs) / float(warmup_epochs)
    return float(beta_max * min(1.0, 0.5 * (1.0 - math.cos(progress * math.pi))))


def compute_loss(
    mu: torch.Tensor,
    logvar: Optional[torch.Tensor],
    y: torch.Tensor,
    kl: torch.Tensor,
    beta_kl: float,
    num_train_obs: int,
    lambda_reg: float = 0.2,
    deterministic_mae_only: bool = False,
) -> torch.Tensor:
    """Eq. (3.43): Heteroscedastic GNLL + lambda_reg * MAE + (beta_kl / (M_train * N * T')) * KL."""
    mae = torch.mean(torch.abs(y - mu))
    if deterministic_mae_only or (logvar is None):
        return mae
    gnll = 0.5 * torch.mean(torch.exp(-logvar) * (y - mu).pow(2) + logvar)
    kl_term = (beta_kl / max(1, num_train_obs)) * kl
    return gnll + lambda_reg * mae + kl_term


def _average_state_dicts(awa_state: Dict[str, torch.Tensor], cur_state: Dict[str, torch.Tensor], n_models: int) -> Dict[str, torch.Tensor]:
    """Algorithm 3-1 Step 3-c: phi_AWA <- (n * phi_AWA + phi) / (n + 1)."""
    out = {}
    for k in awa_state:
        if awa_state[k].is_floating_point():
            out[k] = (n_models * awa_state[k] + cur_state[k].detach()) / (n_models + 1)
        else:
            out[k] = cur_state[k].detach().clone()
    return out


def train_single_model(
    model: nn.Module,
    loaders: Dict[str, Any],
    device: torch.device,
    epochs_pre: int = 80,
    epochs_warmup: int = 20,
    epochs_awa: int = 20,
    lr_max: float = 3e-3,
    lr_min: float = 3e-4,
    beta_max: float = 1.0,
    lambda_reg: float = 0.2,
    use_amp: bool = False,
    deterministic_mae_only: bool = False,
    use_kl_awa: bool = True,
) -> nn.Module:
    """Executes Algorithm 3-1 (Phase 1 Variational Pre-Training + Phase 2 AWA)."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_max, weight_decay=1e-4)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    sample_X, _, sample_Y = next(iter(train_loader))
    num_train_obs = len(train_loader.dataset) * sample_Y.shape[1] * sample_Y.shape[2]

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    # Phase 1: Variational Pre-Training with Cosine KL Annealing
    for epoch in range(1, epochs_pre + 1):
        model.train()
        beta = kl_annealing_beta(epoch, epochs_warmup, beta_max) if use_kl_awa else beta_max
        for X_b, T_b, Y_b in train_loader:
            X_b, T_b, Y_b = X_b.to(device), T_b.to(device), Y_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                mu, logvar = model(X_b, T_b, sample=True)
                loss = compute_loss(mu, logvar, Y_b, model.kl_loss(), beta, num_train_obs, lambda_reg, deterministic_mae_only)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()

        # Validation checkpointing
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_b, T_b, Y_b in val_loader:
                X_b, T_b, Y_b = X_b.to(device), T_b.to(device), Y_b.to(device)
                mu, logvar = model(X_b, T_b, sample=False)
                val_losses.append(float(compute_loss(mu, logvar, Y_b, model.kl_loss(), beta, num_train_obs, lambda_reg, deterministic_mae_only)))
        mean_val = float(np.mean(val_losses))
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    # Phase 2: Adaptive Weight Averaging (AWA) — Algorithm 3-1 Step 3
    if use_kl_awa and epochs_awa > 0 and (not deterministic_mae_only):
        awa_state = copy.deepcopy(model.state_dict())
        n_models = 1
        n_iter = max(1, len(train_loader))
        for e_awa in range(1, epochs_awa + 1):
            model.train()
            is_odd = (e_awa % 2 == 1)
            for step_i, (X_b, T_b, Y_b) in enumerate(train_loader):
                lr_now = (
                    lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos((step_i / n_iter) * math.pi))
                    if is_odd else lr_min
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now
                X_b, T_b, Y_b = X_b.to(device), T_b.to(device), Y_b.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                    mu, logvar = model(X_b, T_b, sample=True)
                    loss = compute_loss(mu, logvar, Y_b, model.kl_loss(), beta_max, num_train_obs, lambda_reg, False)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()

            if not is_odd:
                awa_state = _average_state_dicts(awa_state, model.state_dict(), n_models)
                n_models += 1
        model.load_state_dict(awa_state)

    return model


@torch.no_grad()
def run_inference_split(
    model: nn.Module,
    loader: Any,
    scaler: Any,
    device: torch.device,
    target_channel: int = 0,
    num_mc: int = 10,
    is_probabilistic: bool = True,
) -> Dict[str, np.ndarray]:
    """Runs batched inference and recovers exact physical-scale moments (Chapter 3 Eq. 3.9)."""
    model.eval()
    ys, mus, v_aleas, v_episs, mc_list = [], [], [], [], []
    for X_b, T_b, Y_b in loader:
        X_b, T_b = X_b.to(device), T_b.to(device)
        ys.append(scaler.inverse_mean(Y_b.numpy(), target_channel))
        if is_probabilistic and hasattr(model, "predict_mc"):
            out = model.predict_mc(X_b, T_b, num_samples=num_mc)
            mus.append(scaler.inverse_mean(out["mu"].cpu().numpy(), target_channel))
            v_aleas.append(scaler.inverse_variance(out["var_alea"].cpu().numpy(), target_channel))
            v_episs.append(scaler.inverse_variance(out["var_epis"].cpu().numpy(), target_channel))
            mc_list.append(scaler.inverse_mean(out["mc_mu_samples"].cpu().numpy(), target_channel))
        else:
            mu, _ = model(X_b, T_b, sample=False)
            mu_np = scaler.inverse_mean(mu.cpu().numpy(), target_channel)
            mus.append(mu_np)
            v_aleas.append(np.zeros_like(mu_np))
            v_episs.append(np.zeros_like(mu_np))

    res = {
        "y_true": np.concatenate(ys, axis=0),
        "mu": np.concatenate(mus, axis=0),
        "var_alea": np.concatenate(v_aleas, axis=0),
        "var_epis": np.concatenate(v_episs, axis=0),
    }
    if mc_list:
        res["mc_samples"] = np.concatenate(mc_list, axis=1)  # (M_mc, M_total, T', N)
    return res


def evaluate_probabilistic_with_calibration(
    val_out: Dict[str, np.ndarray],
    test_out: Dict[str, np.ndarray],
    alpha: float = 0.05,
    use_nemhcc: bool = True,
    rho_cal: float = 0.995,
    n_update: int = 288,
) -> Dict[str, Any]:
    """Applies NE-MHCC (or raw Gaussian z_{1-alpha/2} for ablation) and computes all 11 metrics."""
    sig_val = np.sqrt(np.maximum(val_out["var_alea"] + val_out["var_epis"], 1e-8))
    sig_test = np.sqrt(np.maximum(test_out["var_alea"] + test_out["var_epis"], 1e-8))

    if use_nemhcc:
        cal = NEMHCCCalibrator(alpha=alpha, rho_cal=rho_cal)
        lo, hi = cal.online_rolling_predict(
            val_out["y_true"], val_out["mu"], sig_val,
            test_out["y_true"], test_out["mu"], sig_test,
            n_update=n_update,
        )
    else:
        z = float(norm.ppf(1.0 - alpha / 2.0))
        lo = test_out["mu"] - z * sig_test
        hi = test_out["mu"] + z * sig_test

    return evaluate_all_metrics(
        y_true=test_out["y_true"],
        mu=test_out["mu"],
        var_alea=test_out["var_alea"],
        var_epis=test_out["var_epis"],
        lower=lo,
        upper=hi,
        alpha=alpha,
        mc_samples=test_out.get("mc_samples"),
    )
