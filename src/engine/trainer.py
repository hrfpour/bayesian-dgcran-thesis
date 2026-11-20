"""
End-to-End Training & Evaluation Engine (v2.2 Gold):
  - Phase 1 Pre-training with cosine KL Annealing (Eq. 3.44), CosineAnnealingLR, Early Stopping (patience=15)
  - Gradient norm clipping (||grad||_2 <= 5.0) & Tesla T4 Tensor Core AMP fp16
  - Phase 2 Adaptive Weight Averaging (AWA, Eq. 3.52)
  - Full per-epoch learning-curve logging & inference latency benchmarking for thesis.farhadipour.com
"""
from __future__ import annotations
import copy, math, time
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

from src.calibration.nemhcc import NEMHCCCalibrator
from src.metrics.evaluation import evaluate_all_metrics


def kl_annealing_beta(epoch: int, warmup_epochs: int, beta_max: float = 1.0) -> float:
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
    mae = torch.mean(torch.abs(y - mu))
    if deterministic_mae_only or (logvar is None):
        return mae
    gnll = 0.5 * torch.mean(torch.exp(-logvar) * (y - mu).pow(2) + logvar)
    kl_term = (beta_kl / max(1, num_train_obs)) * kl
    return gnll + lambda_reg * mae + kl_term


def _average_state_dicts(awa_state: Dict[str, torch.Tensor], cur_state: Dict[str, torch.Tensor], n_models: int) -> Dict[str, torch.Tensor]:
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
    lr_min: float = 1e-4,
    beta_max: float = 1.0,
    lambda_reg: float = 0.2,
    use_amp: bool = True,
    deterministic_mae_only: bool = False,
    use_kl_awa: bool = True,
    patience: int = 15,
) -> Tuple[nn.Module, Dict[str, Any]]:
    model = model.to(device)
    amp_on = bool(use_amp and device.type == "cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_max, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs_pre), eta_min=lr_min)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_on)

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    _, _, sample_Y = next(iter(train_loader))
    num_train_obs = len(train_loader.dataset) * sample_Y.shape[1] * sample_Y.shape[2]

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    best_state = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "val_loss": [], "epoch_sec": [], "best_epoch": 0, "epochs_run": 0}

    for epoch in range(1, epochs_pre + 1):
        t_ep = time.time()
        model.train()
        beta = kl_annealing_beta(epoch, epochs_warmup, beta_max) if use_kl_awa else beta_max
        tr_losses = []
        for X_b, T_b, Y_b in train_loader:
            X_b, T_b, Y_b = X_b.to(device, non_blocking=True), T_b.to(device, non_blocking=True), Y_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_on):
                mu, logvar = model(X_b, T_b, sample=True)
                loss = compute_loss(mu, logvar, Y_b, model.kl_loss(), beta, num_train_obs, lambda_reg, deterministic_mae_only)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            tr_losses.append(float(loss.detach()))
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_b, T_b, Y_b in val_loader:
                X_b, T_b, Y_b = X_b.to(device, non_blocking=True), T_b.to(device, non_blocking=True), Y_b.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=amp_on):
                    mu, logvar = model(X_b, T_b, sample=False)
                    vl = compute_loss(mu, logvar, Y_b, model.kl_loss(), beta, num_train_obs, lambda_reg, deterministic_mae_only)
                val_losses.append(float(vl))

        mean_tr = float(np.mean(tr_losses))
        mean_val = float(np.mean(val_losses))
        history["train_loss"].append(round(mean_tr, 5))
        history["val_loss"].append(round(mean_val, 5))
        history["epoch_sec"].append(round(time.time() - t_ep, 2))

        if mean_val < best_val_loss - 1e-5:
            best_val_loss = mean_val
            best_epoch = epoch
            no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1
            if no_improve >= patience and epoch >= max(20, epochs_warmup if (not deterministic_mae_only and use_kl_awa) else 20):
                break

    history["best_epoch"] = best_epoch
    history["epochs_run"] = len(history["train_loss"])
    model.load_state_dict(best_state)

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
                X_b, T_b, Y_b = X_b.to(device, non_blocking=True), T_b.to(device, non_blocking=True), Y_b.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp_on):
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

    return model, history


@torch.no_grad()
def run_inference_split(
    model: nn.Module,
    loader: Any,
    scaler: Any,
    device: torch.device,
    target_channel: int = 0,
    num_mc: int = 10,
    is_probabilistic: bool = True,
    use_amp: bool = True,
) -> Dict[str, np.ndarray]:
    model.eval()
    amp_on = bool(use_amp and device.type == "cuda")
    ys, mus, v_aleas, v_episs, mc_list = [], [], [], [], []
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    n_batches = 0
    for X_b, T_b, Y_b in loader:
        n_batches += 1
        X_b, T_b = X_b.to(device, non_blocking=True), T_b.to(device, non_blocking=True)
        ys.append(scaler.inverse_mean(Y_b.numpy(), target_channel))
        with torch.amp.autocast("cuda", enabled=amp_on):
            if is_probabilistic and hasattr(model, "predict_mc"):
                out = model.predict_mc(X_b, T_b, num_samples=num_mc)
                mus.append(scaler.inverse_mean(out["mu"].float().cpu().numpy(), target_channel))
                v_aleas.append(scaler.inverse_variance(out["var_alea"].float().cpu().numpy(), target_channel))
                v_episs.append(scaler.inverse_variance(out["var_epis"].float().cpu().numpy(), target_channel))
                mc_list.append(scaler.inverse_mean(out["mc_mu_samples"].float().cpu().numpy(), target_channel))
            else:
                mu, _ = model(X_b, T_b, sample=False)
                mu_np = scaler.inverse_mean(mu.float().cpu().numpy(), target_channel)
                mus.append(mu_np)
                v_aleas.append(np.zeros_like(mu_np))
                v_episs.append(np.zeros_like(mu_np))
    if device.type == "cuda":
        torch.cuda.synchronize()
    inf_ms = round((time.time() - t0) * 1000.0 / max(1, n_batches), 3)

    res = {
        "y_true": np.concatenate(ys, axis=0),
        "mu": np.concatenate(mus, axis=0),
        "var_alea": np.concatenate(v_aleas, axis=0),
        "var_epis": np.concatenate(v_episs, axis=0),
        "inference_ms_per_batch": inf_ms,
    }
    if mc_list:
        res["mc_samples"] = np.concatenate(mc_list, axis=1)
    return res


def evaluate_probabilistic_with_calibration(
    val_out: Dict[str, np.ndarray],
    test_out: Dict[str, np.ndarray],
    alpha: float = 0.05,
    use_nemhcc: bool = True,
    rho_cal: float = 0.995,
    n_update: int = 288,
) -> Dict[str, Any]:
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
