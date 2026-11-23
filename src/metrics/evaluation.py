"""
All 11 Evaluation Metrics Defined in Chapter 3, Section 3.6.2:
  Group 1 (Point): MAE, RMSE, MAPE
  Group 2 (Interval): PICP, MPIW, MHPICE, MIS_alpha (Winkler Score)
  Group 3 (Distributional & Calibration): MNLL, CRPS (Gaussian & Fair Empirical O(M log M)),
                                          Regression ECE, Sparsification Curve
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
from scipy.stats import norm


def compute_point_metrics(y_true: np.ndarray, y_pred: np.ndarray, mape_eps: float = 1.0, mask_zero_target: bool = False) -> Dict[str, float]:
    """
    y_true, y_pred: shape (M_test, T', N) in physical scale.
    Uses standard PeMS masking (|y_true| >= mape_eps) for MAPE to avoid division by zero.
    """
    if mask_zero_target:
        valid = np.abs(y_true) > 1e-4
        yt = y_true[valid]
        yp = y_pred[valid]
    else:
        yt = y_true
        yp = y_pred
    err = yt - yp
    mae = float(np.mean(np.abs(err))) if yt.size > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(err ** 2))) if yt.size > 0 else float("nan")
    mask = np.abs(yt) >= mape_eps
    mape = float(np.mean(np.abs(err[mask] / yt[mask])) * 100.0) if np.any(mask) else float("nan")
    denom_s = np.abs(yt) + np.abs(yp)
    smask = denom_s >= mape_eps
    smape = float(np.mean(2.0 * np.abs(err[smask]) / denom_s[smask]) * 100.0) if np.any(smask) else float("nan")
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2)) if yt.size > 0 else 1.0
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-8))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "SMAPE": smape, "R2": r2}


def compute_interval_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    y_true, lower, upper: shape (M_test, T', N).
    Returns PICP, MPIW, MHPICE (Eq. 3.58), and MIS_alpha / Winkler Score (Eq. 3.59).
    """
    inside = (y_true >= lower) & (y_true <= upper)
    picp = float(np.mean(inside))
    width = upper - lower
    mpiw = float(np.mean(width))

    # Horizon-wise PICP & MHPICE (Qian et al., 2024 Eq. 36-37)
    picp_per_horizon = np.mean(inside, axis=(0, 2))  # shape (T',)
    e_cov = np.maximum(0.0, (1.0 - alpha) - picp_per_horizon)
    mhpice = float(np.mean(e_cov))

    # Mean Interval Score (Winkler Score)
    below = np.maximum(0.0, lower - y_true)
    above = np.maximum(0.0, y_true - upper)
    mis = float(np.mean(width + (2.0 / alpha) * below + (2.0 / alpha) * above))
    return {"PICP": picp, "MPIW": mpiw, "MHPICE": mhpice, "MIS": mis}


def compute_mnll(y_true: np.ndarray, mu: np.ndarray, var_total: np.ndarray, eps: float = 1e-6) -> float:
    """Mean Gaussian Negative Log-Likelihood (Eq. 3.60)."""
    v = np.maximum(var_total, eps)
    nll = 0.5 * (np.log(2.0 * np.pi * v) + ((y_true - mu) ** 2) / v)
    return float(np.mean(nll))


def compute_crps_gaussian(y_true: np.ndarray, mu: np.ndarray, sigma_total: np.ndarray, eps: float = 1e-6) -> float:
    """Closed-form Gaussian CRPS (Gneiting & Raftery, 2007)."""
    s = np.maximum(sigma_total, eps)
    z = (y_true - mu) / s
    crps = s * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def compute_crps_empirical_fair(y_true: np.ndarray, samples: np.ndarray) -> float:
    """
    Exact O(M log M) Fair Empirical CRPS (Chapter 3 Eq. 3.61; Waghmare & Ziegel, 2026):
      samples: shape (M_mc, M_test, T', N)
      y_true:  shape (M_test, T', N)
    """
    M = samples.shape[0]
    assert M >= 2, "Fair empirical CRPS requires M >= 2 Monte Carlo samples."
    y_sorted = np.sort(samples, axis=0)  # (M, M_test, T', N)
    m_idx = np.arange(1, M + 1, dtype=np.float64).reshape((M,) + (1,) * (y_sorted.ndim - 1))
    y_exp = y_true[None, ...]
    ind = (y_exp < y_sorted).astype(np.float64)
    crps = (2.0 / (M * (M - 1))) * np.sum((y_sorted - y_exp) * ((M - 1) * ind - m_idx + 1.0), axis=0)
    return float(np.mean(crps))


def compute_regression_ece(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma_total: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Expected Calibration Error for Regression (Eq. 3.62; Kuleshov et al., 2018)."""
    s = np.maximum(sigma_total, eps)
    p_levels = np.linspace(1.0 / (n_bins + 1), n_bins / (n_bins + 1), n_bins)
    errs = []
    for p in p_levels:
        q_p = mu + s * norm.ppf(p)
        p_hat = float(np.mean(y_true <= q_p))
        errs.append(abs(p_hat - p))
    return float(np.mean(errs))


def compute_sparsification_curve(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma_total: np.ndarray,
    fractions: Optional[List[float]] = None,
) -> Dict[str, List[float]]:
    """
    Selective Regression / Sparsification Curve (Section 3.6.2; Yu et al., 2025).
    Drops top-xi most uncertain predictions and records monotonic MAE(xi) and RMSE(xi).
    """
    if fractions is None:
        fractions = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    abs_err = np.abs(y_true - mu).ravel()
    sq_err = ((y_true - mu) ** 2).ravel()
    unc = sigma_total.ravel()
    order = np.argsort(unc)  # ascending uncertainty -> keep lowest (1 - xi)
    total = len(order)
    mae_curve, rmse_curve = [], []
    for xi in fractions:
        keep = max(1, int(round((1.0 - xi) * total)))
        idx = order[:keep]
        mae_curve.append(float(np.mean(abs_err[idx])))
        rmse_curve.append(float(np.sqrt(np.mean(sq_err[idx]))))
    return {"fractions": fractions, "MAE": mae_curve, "RMSE": rmse_curve}


def evaluate_all_metrics(
    y_true: np.ndarray,
    mu: np.ndarray,
    var_alea: np.ndarray,
    var_epis: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.05,
    mc_samples: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Computes complete 11-metric suite overall and per key horizons (15, 30, 60 min)."""
    var_total = np.maximum(var_alea + var_epis, 1e-8)
    sigma_total = np.sqrt(var_total)

    overall = {}
    overall.update(compute_point_metrics(y_true, mu))
    overall.update(compute_interval_metrics(y_true, lower, upper, alpha=alpha))
    overall["MNLL"] = compute_mnll(y_true, mu, var_total)
    overall["CRPS_Gaussian"] = compute_crps_gaussian(y_true, mu, sigma_total)
    if mc_samples is not None and mc_samples.shape[0] >= 2:
        overall["CRPS_Empirical"] = compute_crps_empirical_fair(y_true, mc_samples)
    overall["ECE"] = compute_regression_ece(y_true, mu, sigma_total)

    per_horizon = {}
    T_prime = y_true.shape[1]
    for tau in [3, 6, 12]:
        if tau <= T_prime:
            sl = slice(tau - 1, tau)
            h_res = {}
            h_res.update(compute_point_metrics(y_true[:, sl, :], mu[:, sl, :]))
            h_res.update(compute_interval_metrics(y_true[:, sl, :], lower[:, sl, :], upper[:, sl, :], alpha=alpha))
            h_res["MNLL"] = compute_mnll(y_true[:, sl, :], mu[:, sl, :], var_total[:, sl, :])
            h_res["CRPS_Gaussian"] = compute_crps_gaussian(y_true[:, sl, :], mu[:, sl, :], sigma_total[:, sl, :])
            h_res["ECE"] = compute_regression_ece(y_true[:, sl, :], mu[:, sl, :], sigma_total[:, sl, :])
            per_horizon[f"horizon_{tau * 5}min"] = h_res

    sparsification = compute_sparsification_curve(y_true, mu, sigma_total)
    return {
        "overall": overall,
        "per_horizon": per_horizon,
        "sparsification": sparsification,
    }
