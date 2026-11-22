"""
Diebold-Mariano Asymptotic Test with Newey-West Bartlett Growing-Bandwidth Spectral Variance
and Harvey-Leybourne-Newbold (HLN) Finite-Sample Correction.
Strictly implements Theorem 3.6.1 & Eq. (3.65) of Chapter 3.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
from scipy.stats import norm, t as student_t


def diebold_mariano_test(
    loss_model_1: np.ndarray,
    loss_model_2: np.ndarray,
    horizon: int = 1,
) -> Dict[str, float]:
    """
    Tests H0: E[d_s] = 0 vs one-sided H1: E[d_s] < 0 (Model 1 strictly better than Model 2).
    loss_model_1, loss_model_2: 1D arrays of length M_test (spatially averaged loss per test step).
    """
    d = np.asarray(loss_model_1, dtype=np.float64) - np.asarray(loss_model_2, dtype=np.float64)
    M = len(d)
    assert M > max(horizon, 2), "Insufficient test length for DM test."
    d_bar = float(np.mean(d))
    d_centered = d - d_bar

    # Newey-West growing bandwidth L_lag = max(tau - 1, floor(M^(1/3)))
    L_lag = max(horizon - 1, int(np.floor(M ** (1.0 / 3.0))))
    gamma_0 = float(np.dot(d_centered, d_centered) / M)
    long_run_var = gamma_0
    for k in range(1, L_lag + 1):
        w_bart = 1.0 - k / (L_lag + 1.0)
        gamma_k = float(np.dot(d_centered[k:], d_centered[:-k]) / M)
        long_run_var += 2.0 * w_bart * gamma_k

    long_run_var = max(long_run_var, 1e-12)
    dm_stat = d_bar / np.sqrt(long_run_var / M)
    p_val_dm_one_sided = float(norm.cdf(dm_stat))

    # Harvey-Leybourne-Newbold (HLN) finite-sample correction (Eq. 3.65)
    hln_factor_sq = (M + 1.0 - 2.0 * horizon + (horizon * (horizon - 1.0)) / M) / M
    hln_factor = np.sqrt(max(hln_factor_sq, 1e-8))
    hln_stat = float(hln_factor * dm_stat)
    p_val_hln_one_sided = float(student_t.cdf(hln_stat, df=M - 1))

    return {
        "mean_loss_diff": d_bar,
        "L_lag": int(L_lag),
        "DM_stat": float(dm_stat),
        "DM_pvalue_one_sided": p_val_dm_one_sided,
        "HLN_stat": hln_stat,
        "HLN_pvalue_one_sided": p_val_hln_one_sided,
    }
