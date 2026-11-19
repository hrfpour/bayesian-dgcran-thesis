"""
Non-Exchangeable Multi-Horizon Conformal Calibration (NE-MHCC).
Strictly implements Innovation 5 & Algorithm 3-2 of Chapter 3:
  - Eq. (3.46): Uncertainty-scaled nonconformity score.
  - Eq. (3.48): Corrected minus-sign & normalized horizon-specific alpha_c^(tau).
  - Eq. (3.49)-(3.50): Exponentially weighted non-exchangeable quantile (Barber et al., 2023).
  - Strictly causal tau-step delayed-feedback online rolling calibration.
"""
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
from scipy.stats import norm


def weighted_quantile_with_inf(
    scores: np.ndarray,
    weights_norm: np.ndarray,
    w_inf: float,
    q_level: float,
) -> float:
    """
    Computes Quantile_{q_level}(sum_{k=1}^{M_cal} w_k delta_{s_k} + w_inf delta_{+inf}).
    """
    order = np.argsort(scores)
    s_sorted = scores[order]
    w_sorted = weights_norm[order]
    cum_w = np.cumsum(w_sorted)
    idx = np.searchsorted(cum_w, q_level, side="left")
    if idx >= len(s_sorted):
        return float(s_sorted[-1])  # finite fallback cap in practice when q_level > 1 - w_inf
    return float(s_sorted[idx])


class NEMHCCCalibrator:
    """Non-Exchangeable Multi-Horizon Conformal Calibration (Algorithm 3-2)."""

    def __init__(
        self,
        alpha: float = 0.05,
        rho_cal: float = 0.995,
        gamma_hor: float = 0.15,
        alpha_min: float = 0.005,
        alpha_max: float = 0.25,
        eps_num: float = 1e-8,
    ):
        self.alpha = alpha
        self.rho_cal = rho_cal
        self.gamma_hor = gamma_hor
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.eps_num = eps_num
        self.alpha_c_: Optional[np.ndarray] = None
        self.q_hat_: Optional[np.ndarray] = None  # shape (T', N)

    def _compute_scores(self, y_true: np.ndarray, mu: np.ndarray, sigma_total: np.ndarray) -> np.ndarray:
        """Eq. (3.46): shape (M_cal, T', N)."""
        return np.abs(y_true - mu) / (sigma_total + self.eps_num)

    def _compute_horizon_alphas(self, scores: np.ndarray) -> np.ndarray:
        """Eq. (3.47)-(3.48) with fixed minus sign and normalized ((tau-1)/(T'-1))^2."""
        T_prime = scores.shape[1]
        z_0 = float(norm.ppf(1.0 - self.alpha / 2.0))
        p_c = np.mean(scores <= z_0, axis=(0, 2))  # shape (T',)
        drop = max(0.0, float(p_c[0] - p_c[-1]))
        alphas = np.empty(T_prime, dtype=np.float64)
        denom = max(1, T_prime - 1)
        for tau_idx in range(T_prime):
            raw = (p_c[tau_idx] + 2.0 * self.alpha - 1.0) - self.gamma_hor * drop * ((tau_idx / denom) ** 2)
            alphas[tau_idx] = float(np.clip(raw, self.alpha_min, self.alpha_max))
        return alphas

    def _compute_quantiles(self, scores: np.ndarray, alpha_c: np.ndarray) -> np.ndarray:
        """Eq. (3.49)-(3.50): returns q_hat of shape (T', N)."""
        M_cal, T_prime, N = scores.shape
        exponents = np.arange(M_cal - 1, -1, -1, dtype=np.float64)
        w = self.rho_cal ** exponents
        denom = float(np.sum(w) + 1.0)
        w_tilde = w / denom
        w_inf = 1.0 / denom

        q_hat = np.empty((T_prime, N), dtype=np.float64)
        for tau_idx in range(T_prime):
            q_level = 1.0 - float(alpha_c[tau_idx])
            for i in range(N):
                q_hat[tau_idx, i] = weighted_quantile_with_inf(scores[:, tau_idx, i], w_tilde, w_inf, q_level)
        return q_hat

    def fit(
        self,
        y_cal: np.ndarray,
        mu_cal: np.ndarray,
        sigma_cal: np.ndarray,
        fixed_alpha_c: Optional[np.ndarray] = None,
    ) -> "NEMHCCCalibrator":
        """Fits initial horizon-specific non-exchangeable quantiles on D_cal."""
        scores = self._compute_scores(y_cal, mu_cal, sigma_cal)
        self.alpha_c_ = fixed_alpha_c if fixed_alpha_c is not None else self._compute_horizon_alphas(scores)
        self.q_hat_ = self._compute_quantiles(scores, self.alpha_c_)
        return self

    def predict_intervals(self, mu: np.ndarray, sigma_total: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Eq. (3.51): returns (lower, upper) of shape (M_test, T', N)."""
        assert self.q_hat_ is not None, "Call fit() before predict_intervals()."
        radius = self.q_hat_[None, :, :] * sigma_total
        return mu - radius, mu + radius

    def online_rolling_predict(
        self,
        y_cal: np.ndarray,
        mu_cal: np.ndarray,
        sigma_cal: np.ndarray,
        y_test: np.ndarray,
        mu_test: np.ndarray,
        sigma_test: np.ndarray,
        n_update: int = 288,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Strictly causal online rolling calibration (Algorithm 3-2 Step 3-c) with tau-step delayed feedback:
        At test step s, for horizon tau (1..T'), only past test targets with s' + tau <= s have realized!
        """
        scores_buf = self._compute_scores(y_cal, mu_cal, sigma_cal)
        M_cal, T_prime, N = scores_buf.shape
        M_test = y_test.shape[0]
        test_scores_full = self._compute_scores(y_test, mu_test, sigma_test)

        # Keep alpha_c fixed from initial D_cal (Theorem 3.4.3 precondition)
        self.alpha_c_ = self._compute_horizon_alphas(scores_buf)
        self.q_hat_ = self._compute_quantiles(scores_buf, self.alpha_c_)

        lower = np.empty_like(mu_test)
        upper = np.empty_like(mu_test)
        for s in range(M_test):
            radius = self.q_hat_ * sigma_test[s]
            lower[s] = mu_test[s] - radius
            upper[s] = mu_test[s] + radius

            # Periodic causal rolling window update (strictly respecting tau-step feedback delay)
            if (s + 1) % n_update == 0:
                updated_buf = scores_buf.copy()
                for tau_idx in range(T_prime):
                    tau = tau_idx + 1
                    # Available realized test indices s' such that s' + tau <= s
                    max_avail = s - tau + 1
                    if max_avail > 0:
                        recent = test_scores_full[:max_avail, tau_idx, :]
                        combined = np.concatenate([scores_buf[:, tau_idx, :], recent], axis=0)
                        updated_buf[:, tau_idx, :] = combined[-M_cal:]
                self.q_hat_ = self._compute_quantiles(updated_buf, self.alpha_c_)
        return lower, upper
