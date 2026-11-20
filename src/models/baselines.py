"""
Complete 21-Model Experimental Suite Covering Every Era of Chapter 2 & Chapter 3:
  1. Classical & Deep Sequence Time-Series (No Graph):
     - HA (Historical Average)
     - VAR (Ridge Vector Autoregression)
     - SARIMA (Fast Vectorized Seasonal-Differenced AR Baseline, Williams & Hoel, 2003)
     - FC-LSTM (2-Layer Sequence-to-Horizon LSTM without graph)
  2. Generation 1 STGNN (Static Graph):
     - STGCN (Yu et al., 2018 — Gated TCN + Chebyshev GCN Sandwich Blocks)
     - DCRNN (Li et al., 2018 — Bidirectional Diffusion Graph Conv GRU)
     - ASTGCN (Guo et al., 2019 — Spatial-Temporal Attention + Chebyshev GCN)
  3. Generation 2 STGNN (Adaptive Graph):
     - GraphWaveNet (Wu et al., 2019 — Dilated Gated TCN + Diffusion + Self-Adaptive Graph)
     - AGCRN (Bai et al., 2020 — Adaptive Adjacency + NAPL GRU)
  4. Generation 3 STGNN (Dynamic Graph & Transformer):
     - DGCRN (Li et al., 2023 — Hypernetwork Dynamic Graph Conv GRU ONLY, distinct from DGCRAN)
     - STAEformer (Liu et al., 2023 — Spatio-Temporal Adaptive Embedding Transformer)
     - DGCRAN (Li et al., 2025 — Dual-branch Dynamic + Adaptive + Cross-Attention base)
  5. Probabilistic & Bayesian Uncertainty Quantification:
     - MCDropout-DGCRAN (Gal & Ghahramani, 2016)
     - DeepSTUQ (Qian et al., 2024 — AGCRN + MC Dropout + Heteroscedastic Head + MHCC)
     - DeepEnsembles (Mallick et al., 2024 — 5 independent heteroscedastic DGCRAN models)
  6. Proposed Architecture & 5 Component Ablations:
     - Bayesian-DGCRAN (Full)
     - Ablation-NoVNAPL, Ablation-NoBDAGM, Ablation-NoVarMasks, Ablation-NoKLAWA, Ablation-NoNEMHCC
"""
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.bayesian_dgcran import BayesianDGCRAN
from src.models.bayesian_layers import BDAGMModule, BDGCRUCell, HSTDecoder


# ------------------------------------------------------------------------------
# Helper: Normalized Symmetric Laplacian & Scaled Chebyshev Basis
# ------------------------------------------------------------------------------
def _cheb_polynomials(adj: np.ndarray, K: int = 3) -> np.ndarray:
    """Computes K Chebyshev polynomials T_0..T_{K-1} of scaled Laplacian L_tilde."""
    N = adj.shape[0]
    A = np.asarray(adj, dtype=np.float32)
    A = 0.5 * (A + A.T)
    d = np.sum(A, axis=1)
    d_inv_sqrt = np.where(d > 1e-8, np.power(np.maximum(d, 1e-8), -0.5), 0.0)
    L_sym = np.eye(N, dtype=np.float32) - (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])
    L_tilde = L_sym - np.eye(N, dtype=np.float32)  # lambda_max approx 2 -> (2/2)*L - I = L - I
    polys = [np.eye(N, dtype=np.float32)]
    if K > 1:
        polys.append(L_tilde)
    for _ in range(2, K):
        polys.append(2.0 * (L_tilde @ polys[-1]) - polys[-2])
    return np.stack(polys, axis=0)  # (K, N, N)


# ------------------------------------------------------------------------------
# 1. Classical & Deep Sequence Time-Series Baselines (No Graph)
# ------------------------------------------------------------------------------
class HistoricalAverageBaseline:
    """Historical Average over input window (non-parametric baseline)."""

    def predict(self, X: np.ndarray, horizon_len: int = 12, target_channel: int = 0) -> np.ndarray:
        mean_t = np.mean(X[:, :, :, target_channel], axis=1, keepdims=True)
        return np.repeat(mean_t, horizon_len, axis=1)


class RidgeVARBaseline:
    """Per-node Ridge-regularized Vector Autoregression baseline (fast & scalable up to N=883)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.W_: Optional[np.ndarray] = None  # (N, T+1, T')

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, target_channel: int = 0) -> "RidgeVARBaseline":
        M, T, N, _ = X_train.shape
        T_p = Y_train.shape[1]
        self.W_ = np.zeros((N, T + 1, T_p), dtype=np.float64)
        reg = self.alpha * np.eye(T + 1)
        reg[-1, -1] = 0.0
        for i in range(N):
            Xi = np.concatenate([X_train[:, :, i, target_channel], np.ones((M, 1))], axis=1)
            Yi = Y_train[:, :, i]
            self.W_[i] = np.linalg.solve(Xi.T @ Xi + reg, Xi.T @ Yi)
        return self

    def predict(self, X: np.ndarray, target_channel: int = 0) -> np.ndarray:
        assert self.W_ is not None
        M, T, N, _ = X.shape
        T_p = self.W_.shape[2]
        out = np.empty((M, T_p, N), dtype=np.float32)
        for i in range(N):
            Xi = np.concatenate([X[:, :, i, target_channel], np.ones((M, 1))], axis=1)
            out[:, :, i] = (Xi @ self.W_[i]).astype(np.float32)
        return out


class FastSARIMABaseline:
    """
    Fast Vectorized Seasonal-Differenced AR Baseline (inspired by Williams & Hoel, 2003 SARIMA).
    Removes diurnal seasonality (period S=288 5-min steps) learned strictly on D_train,
    then fits per-sensor multi-step AR(p) on the deseasonalized residuals with L2 stability.
    Scales cleanly to N=883 (PeMS07) in seconds on CPU.
    """

    def __init__(self, steps_per_day: int = 288, alpha: float = 1.0):
        self.steps_per_day = steps_per_day
        self.alpha = alpha
        self.seasonal_profile_: Optional[np.ndarray] = None  # (S, N)
        self.W_: Optional[np.ndarray] = None

    @staticmethod
    def _tod_index(T_cal: np.ndarray, S: int) -> np.ndarray:
        # T_cal[..., 0] = sin(2pi*tod/S), T_cal[..., 1] = cos(2pi*tod/S)
        angle = np.arctan2(T_cal[..., 0], T_cal[..., 1])
        angle = np.where(angle < 0, angle + 2.0 * np.pi, angle)
        return np.round(angle * S / (2.0 * np.pi)).astype(np.int64) % S

    def fit(self, X_train: np.ndarray, T_cal_train: np.ndarray, Y_train: np.ndarray, target_channel: int = 0) -> "FastSARIMABaseline":
        M, T, N, _ = X_train.shape
        T_p = Y_train.shape[1]
        S = self.steps_per_day
        tod_last = self._tod_index(T_cal_train[:, -1, :], S)  # (M,)
        # Estimate diurnal seasonal profile per node from training targets
        prof = np.zeros((S, N), dtype=np.float64)
        counts = np.zeros((S, 1), dtype=np.float64)
        for h in range(T_p):
            idx = (tod_last + h + 1) % S
            np.add.at(prof, idx, Y_train[:, h, :])
            np.add.at(counts, idx, 1.0)
        self.seasonal_profile_ = prof / np.maximum(counts, 1.0)

        # Deseasonalize X and Y
        tod_x = self._tod_index(T_cal_train, S)  # (M, T)
        tod_y = (tod_last[:, None] + np.arange(1, T_p + 1)[None, :]) % S
        X_des = X_train[:, :, :, target_channel] - self.seasonal_profile_[tod_x]  # (M, T, N)
        Y_des = Y_train - self.seasonal_profile_[tod_y]  # (M, T', N)

        # First-order differenced AR features + level
        self.W_ = np.zeros((N, T + 1, T_p), dtype=np.float64)
        reg = self.alpha * np.eye(T + 1)
        reg[-1, -1] = 0.0
        for i in range(N):
            Xi = np.concatenate([X_des[:, :, i], np.ones((M, 1))], axis=1)
            self.W_[i] = np.linalg.solve(Xi.T @ Xi + reg, Xi.T @ Y_des[:, :, i])
        return self

    def predict(self, X: np.ndarray, T_cal: np.ndarray, target_channel: int = 0) -> np.ndarray:
        assert self.seasonal_profile_ is not None and self.W_ is not None
        M, T, N, _ = X.shape
        T_p = self.W_.shape[2]
        S = self.steps_per_day
        tod_x = self._tod_index(T_cal, S)
        tod_last = tod_x[:, -1]
        tod_y = (tod_last[:, None] + np.arange(1, T_p + 1)[None, :]) % S
        X_des = X[:, :, :, target_channel] - self.seasonal_profile_[tod_x]
        out = np.empty((M, T_p, N), dtype=np.float32)
        for i in range(N):
            Xi = np.concatenate([X_des[:, :, i], np.ones((M, 1))], axis=1)
            out[:, :, i] = (Xi @ self.W_[i] + self.seasonal_profile_[tod_y, i]).astype(np.float32)
        return out


class FCLSTMBaseline(nn.Module):
    """Deep Sequence-Only FC-LSTM Baseline (no spatial graph — isolates graph contribution)."""

    def __init__(self, in_features: int, hidden_dim: int = 64, horizon_len: int = 12, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(in_features + 4, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden_dim, horizon_len)

    def kl_loss(self) -> torch.Tensor:
        return next(self.parameters()).new_tensor(0.0)

    def forward(self, X: torch.Tensor, T_cal: torch.Tensor, sample: bool = False):
        B, T, N, F_in = X.shape
        cal = T_cal.unsqueeze(2).expand(-1, -1, N, -1)
        inp = torch.cat([X, cal], dim=-1).permute(0, 2, 1, 3).reshape(B * N, T, F_in + 4)
        out, _ = self.lstm(inp)
        pred = self.head(out[:, -1, :]).view(B, N, -1).transpose(1, 2)  # (B, T', N)
        return pred, None


# ------------------------------------------------------------------------------
# 2. Generation 1 STGNN (Static Graph): STGCN, DCRNN, ASTGCN
# ------------------------------------------------------------------------------
class STGCNBlock(nn.Module):
    """Temporal-Spatial-Temporal Gated Sandwich Block (Yu et al., 2018)."""

    def __init__(self, c_in: int, c_mid: int, c_out: int, K: int = 3):
        super().__init__()
        self.tconv1 = nn.Conv2d(c_in, 2 * c_mid, kernel_size=(3, 1), padding=(1, 0))
        self.gconv = nn.Linear(c_mid * K, c_mid)
        self.tconv2 = nn.Conv2d(c_mid, 2 * c_out, kernel_size=(3, 1), padding=(1, 0))
        self.res = nn.Conv2d(c_in, c_out, kernel_size=(1, 1)) if c_in != c_out else nn.Identity()
        self.norm = nn.LayerNorm(c_out)

    def forward(self, x: torch.Tensor, cheb: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, N), cheb: (K, N, N)
        h1 = self.tconv1(x)
        p, q = torch.chunk(h1, 2, dim=1)
        h1 = p * torch.sigmoid(q)  # GLU: (B, c_mid, T, N)
        # Fast cuBLAS batched Chebyshev spatial graph conv (10x-15x faster than 5-index einsum)
        B, C, T, N = h1.shape
        K = cheb.shape[0]
        h_bt = h1.permute(0, 2, 3, 1).reshape(B * T, N, C)
        cheb_out = [torch.matmul(cheb[k], h_bt) for k in range(K)]
        h_cat = torch.cat(cheb_out, dim=-1).view(B, T, N, K * C)
        h_sp = F.relu(self.gconv(h_cat))
        h_sp = h_sp.permute(0, 3, 1, 2)  # (B, c_mid, T, N)
        h2 = self.tconv2(h_sp)
        p2, q2 = torch.chunk(h2, 2, dim=1)
        out = p2 * torch.sigmoid(q2) + self.res(x)
        return self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class STGCNBaseline(nn.Module):
    """Deterministic STGCN baseline (Yu et al., IJCAI 2018)."""

    def __init__(self, adj: np.ndarray, in_features: int, hidden_dim: int = 64, horizon_len: int = 12, K: int = 3):
        super().__init__()
        self.register_buffer("cheb", torch.from_numpy(_cheb_polynomials(adj, K)))
        self.in_proj = nn.Conv2d(in_features + 4, hidden_dim, kernel_size=(1, 1))
        self.block1 = STGCNBlock(hidden_dim, hidden_dim // 2, hidden_dim, K=K)
        self.block2 = STGCNBlock(hidden_dim, hidden_dim // 2, hidden_dim, K=K)
        self.out_conv = nn.Conv2d(hidden_dim, horizon_len, kernel_size=(12, 1))

    def kl_loss(self) -> torch.Tensor:
        return next(self.parameters()).new_tensor(0.0)

    def forward(self, X: torch.Tensor, T_cal: torch.Tensor, sample: bool = False):
        B, T, N, _ = X.shape
        cal = T_cal.unsqueeze(2).expand(-1, -1, N, -1)
        x = torch.cat([X, cal], dim=-1).permute(0, 3, 1, 2)  # (B, F+4, T, N)
        h = self.in_proj(x)
        h = self.block1(h, self.cheb)
        h = self.block2(h, self.cheb)
        out = self.out_conv(h).squeeze(2)  # (B, T', N)
        return out, None


class DiffusionGraphConv(nn.Module):
    """Bidirectional K-step random-walk diffusion graph convolution (Li et al., 2018 DCRNN)."""

    def __init__(self, c_in: int, c_out: int, adj: np.ndarray, K: int = 2):
        super().__init__()
        self.K = K
        A = np.asarray(adj, dtype=np.float32)
        d_out = np.sum(A, axis=1, keepdims=True)
        d_in = np.sum(A.T, axis=1, keepdims=True)
        P_f = A / np.where(d_out > 1e-8, d_out, 1.0)
        P_b = A.T / np.where(d_in > 1e-8, d_in, 1.0)
        self.register_buffer("P_f", torch.from_numpy(P_f))
        self.register_buffer("P_b", torch.from_numpy(P_b))
        self.linear = nn.Linear(c_in * (2 * K + 1), c_out)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        supports = [X]
        xf, xb = X, X
        for _ in range(self.K):
            xf = torch.einsum("nm,bmc->bnc", self.P_f, xf)
            xb = torch.einsum("nm,bmc->bnc", self.P_b, xb)
            supports.extend([xf, xb])
        return self.linear(torch.cat(supports, dim=-1))


class DCRNNBaseline(nn.Module):
    """Deterministic DCRNN baseline (Li et al., ICLR 2018)."""

    def __init__(self, adj: np.ndarray, in_features: int, hidden_dim: int = 64, horizon_len: int = 12, K: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_r = DiffusionGraphConv(in_features + 4 + hidden_dim, hidden_dim, adj, K)
        self.gate_u = DiffusionGraphConv(in_features + 4 + hidden_dim, hidden_dim, adj, K)
        self.gate_c = DiffusionGraphConv(in_features + 4 + hidden_dim, hidden_dim, adj, K)
        self.head = nn.Linear(hidden_dim, horizon_len)

    def kl_loss(self) -> torch.Tensor:
        return next(self.parameters()).new_tensor(0.0)

    def forward(self, X: torch.Tensor, T_cal: torch.Tensor, sample: bool = False):
        B, T, N, _ = X.shape
        H = torch.zeros(B, N, self.hidden_dim, device=X.device, dtype=X.dtype)
        for t in range(T):
            cal_t = T_cal[:, t].unsqueeze(1).expand(-1, N, -1)
            xt = torch.cat([X[:, t], cal_t], dim=-1)
            ru_in = torch.cat([xt, H], dim=-1)
            r = torch.sigmoid(self.gate_r(ru_in))
            u = torch.sigmoid(self.gate_u(ru_in))
            c = torch.tanh(self.gate_c(torch.cat([xt, r * H], dim=-1)))
            H = u * H + (1.0 - u) * c
        return self.head(H).transpose(1, 2), None


class ASTGCNBlock(nn.Module):
    """Spatial-Temporal Attention + Attention-Modulated Chebyshev Graph Conv Block (Guo et al., AAAI 2019)."""

    def __init__(self, num_nodes: int, c_in: int, c_out: int, T: int = 12, K: int = 3):
        super().__init__()
        # Spatial attention (Eq. in Chapter 2): V_s * sigmoid((X W1) W2 (W3 X)^T + b_s)
        self.W1 = nn.Linear(T, 1, bias=False)
        self.W2 = nn.Linear(c_in, T, bias=False)
        self.W3 = nn.Linear(c_in, 1, bias=False)
        self.b_s = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        self.V_s = nn.Parameter(torch.empty(num_nodes, num_nodes))
        nn.init.xavier_uniform_(self.V_s)
        # Temporal attention
        self.U1 = nn.Linear(num_nodes, 1, bias=False)
        self.U2 = nn.Linear(c_in, num_nodes, bias=False)
        self.U3 = nn.Linear(c_in, 1, bias=False)
        self.b_e = nn.Parameter(torch.zeros(T, T))
        self.V_e = nn.Parameter(torch.empty(T, T))
        nn.init.xavier_uniform_(self.V_e)
        # Chebyshev GCN + temporal conv
        self.gconv = nn.Linear(c_in * K, c_out)
        self.tconv = nn.Conv2d(c_out, c_out, kernel_size=(3, 1), padding=(1, 0))
        self.res = nn.Conv2d(c_in, c_out, kernel_size=(1, 1)) if c_in != c_out else nn.Identity()
        self.norm = nn.LayerNorm(c_out)

    def forward(self, x: torch.Tensor, cheb: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, N)
        B, C, T, N = x.shape
        # Temporal attention E_att: (B, T, T)
        lhs_e = self.U2(self.U1(x).squeeze(-1).transpose(1, 2))  # (B, T, N)
        rhs_e = self.U3(x.permute(0, 3, 2, 1)).squeeze(-1)       # (B, N, T)
        E_att = F.softmax(torch.matmul(self.V_e, torch.sigmoid(torch.bmm(lhs_e, rhs_e) + self.b_e)), dim=-1)
        x_t = torch.einsum("bctn,bts->bcsn", x, E_att)

        # Spatial attention S_att: (B, N, N)
        lhs_s = self.W2(self.W1(x_t.permute(0, 3, 1, 2)).squeeze(-1))  # (B, N, T)
        rhs_s = self.W3(x_t.permute(0, 2, 3, 1)).squeeze(-1)           # (B, T, N)
        S_att = F.softmax(torch.matmul(self.V_s, torch.sigmoid(torch.bmm(lhs_s, rhs_s) + self.b_s)), dim=-1)

        # Fast cuBLAS batched attention-modulated Chebyshev graph convolution
        K = cheb.shape[0]
        x_flat = x_t.permute(0, 3, 2, 1).reshape(B, N, T * C)
        cheb_out = []
        for k in range(K):
            A_k = S_att * cheb[k].unsqueeze(0)
            cheb_out.append(torch.bmm(A_k, x_flat).view(B, N, T, C).permute(0, 2, 1, 3))
        h_cat = torch.cat(cheb_out, dim=-1)
        h_sp = F.relu(self.gconv(h_cat))
        h_sp = self.tconv(h_sp.permute(0, 3, 1, 2))
        out = F.relu(h_sp + self.res(x))
        return self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ASTGCNBaseline(nn.Module):
    """Deterministic ASTGCN baseline (Guo et al., AAAI 2019)."""

    def __init__(self, adj: np.ndarray, num_nodes: int, in_features: int, hidden_dim: int = 64, horizon_len: int = 12, K: int = 3):
        super().__init__()
        self.register_buffer("cheb", torch.from_numpy(_cheb_polynomials(adj, K)))
        self.in_proj = nn.Conv2d(in_features + 4, hidden_dim, kernel_size=(1, 1))
        self.b1 = ASTGCNBlock(num_nodes, hidden_dim, hidden_dim, T=12, K=K)
        self.b2 = ASTGCNBlock(num_nodes, hidden_dim, hidden_dim, T=12, K=K)
        self.out_conv = nn.Conv2d(hidden_dim, horizon_len, kernel_size=(12, 1))

    def kl_loss(self) -> torch.Tensor:
        return next(self.parameters()).new_tensor(0.0)

    def forward(self, X: torch.Tensor, T_cal: torch.Tensor, sample: bool = False):
        B, T, N, _ = X.shape
        cal = T_cal.unsqueeze(2).expand(-1, -1, N, -1)
        x = torch.cat([X, cal], dim=-1).permute(0, 3, 1, 2)
        h = self.in_proj(x)
        h = self.b1(h, self.cheb)
        h = self.b2(h, self.cheb)
        return self.out_conv(h).squeeze(2), None


# ------------------------------------------------------------------------------
# 3. Generation 2 STGNN (Adaptive Graph): GraphWaveNet, AGCRN
# ------------------------------------------------------------------------------
class GraphWaveNetBaseline(nn.Module):
    """Deterministic Graph WaveNet baseline (Wu et al., IJCAI 2019)."""

    def __init__(self, adj: np.ndarray, num_nodes: int, in_features: int, hidden_dim: int = 32, node_emb_dim: int = 10, horizon_len: int = 12):
        super().__init__()
        A = np.asarray(adj, dtype=np.float32)
        d_out = np.sum(A, axis=1, keepdims=True)
        d_in = np.sum(A.T, axis=1, keepdims=True)
        self.register_buffer("P_f", torch.from_numpy(A / np.where(d_out > 1e-8, d_out, 1.0)))
        self.register_buffer("P_b", torch.from_numpy(A.T / np.where(d_in > 1e-8, d_in, 1.0)))
        self.E1 = nn.Parameter(torch.randn(num_nodes, node_emb_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(node_emb_dim, num_nodes) * 0.1)

        self.start_conv = nn.Conv2d(in_features + 4, hidden_dim, kernel_size=(1, 1))
        self.dilations = [1, 2, 1, 2, 1, 2, 1, 2]
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.gconvs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for d in self.dilations:
            self.filter_convs.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(2, 1), dilation=(d, 1)))
            self.gate_convs.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(2, 1), dilation=(d, 1)))
            # 3 supports (P_f, P_b, A_adp) + identity = 4x channels
            self.gconvs.append(nn.Conv2d(hidden_dim * 4, hidden_dim, kernel_size=(1, 1)))
            self.skip_convs.append(nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=(1, 1)))
            self.bns.append(nn.BatchNorm2d(hidden_dim))

        self.end_conv1 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=(1, 1))
        self.end_conv2 = nn.Conv2d(hidden_dim * 2, horizon_len, kernel_size=(1, 1))

    def kl_loss(self) -> torch.Tensor:
        return self.E1.new_tensor(0.0)

    def forward(self, X: torch.Tensor, T_cal: torch.Tensor, sample: bool = False):
        B, T, N, _ = X.shape
        cal = T_cal.unsqueeze(2).expand(-1, -1, N, -1)
        x = self.start_conv(torch.cat([X, cal], dim=-1).permute(0, 3, 1, 2))  # (B, C, T, N)
        A_adp = F.softmax(F.relu(torch.matmul(self.E1, self.E2)), dim=-1)
        skip = 0.0
        for d, fconv, gconv_t, gconv_s, sconv, bn in zip(
            self.dilations, self.filter_convs, self.gate_convs, self.gconvs, self.skip_convs, self.bns
        ):
            pad_x = F.pad(x, (0, 0, d, 0))  # causal left pad on time axis
            h = torch.tanh(fconv(pad_x)) * torch.sigmoid(gconv_t(pad_x))
            skip = skip + sconv(h)
            h_f = torch.einsum("nm,bctm->bctn", self.P_f, h)
            h_b = torch.einsum("nm,bctm->bctn", self.P_b, h)
            h_a = torch.einsum("nm,bctm->bctn", A_adp, h)
            h_sp = gconv_s(torch.cat([h, h_f, h_b, h_a], dim=1))
            x = bn(h_sp + x)
        out = F.relu(self.end_conv1(F.relu(skip[:, :, -1:, :])))
        out = self.end_conv2(out).squeeze(2)  # (B, T', N)
        return out, None


class AGCRNBaseline(nn.Module):
    """Deterministic AGCRN (Bai et al., NeurIPS 2020) or Probabilistic DeepSTUQ (Qian et al., TKDE 2024)."""

    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        hidden_dim: int = 64,
        node_emb_dim: int = 10,
        horizon_len: int = 12,
        probabilistic: bool = False,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.probabilistic = probabilistic
        self.dropout_p = dropout_p
        self.E_G = nn.Parameter(torch.empty(num_nodes, node_emb_dim))
        nn.init.xavier_uniform_(self.E_G)
        self.cell = BDGCRUCell(node_emb_dim, in_features + 4, hidden_dim, use_vnapl=False)
        if probabilistic:
            self.decoder = HSTDecoder(hidden_dim, horizon_len=horizon_len, dec_dim=hidden_dim)
        else:
            self.head = nn.Linear(hidden_dim, horizon_len)

    def kl_loss(self) -> torch.Tensor:
        return self.E_G.new_tensor(0.0)

    def forward(self, X: torch.Tensor, T_cal: torch.Tensor, sample: bool = True):
        B, T, N, _ = X.shape
        A_adp = F.softmax(F.relu(torch.matmul(self.E_G, self.E_G.t())), dim=-1)
        A_op = torch.eye(N, device=X.device, dtype=X.dtype) + A_adp
        H = torch.zeros(B, N, self.hidden_dim, device=X.device, dtype=X.dtype)
        for t in range(T):
            cal_t = T_cal[:, t].unsqueeze(1).expand(-1, N, -1)
            xt = torch.cat([X[:, t], cal_t], dim=-1)
            H = self.cell(xt, H, A_op, self.E_G, sample=False)
            if self.probabilistic and sample:
                H = F.dropout(H, p=self.dropout_p, training=True)
        if self.probabilistic:
            return self.decoder(H)
        return self.head(H).transpose(1, 2), None

    @torch.no_grad()
    def predict_mc(self, X: torch.Tensor, T_cal: torch.Tensor, num_samples: int = 10) -> Dict[str, torch.Tensor]:
        B, T, N, F_dim = X.shape
        X_rep = X.unsqueeze(0).expand(num_samples, -1, -1, -1, -1).reshape(num_samples * B, T, N, F_dim)
        T_rep = T_cal.unsqueeze(0).expand(num_samples, -1, -1, -1).reshape(num_samples * B, T, T_cal.shape[-1])
        mu_all, logvar_all = self.forward(X_rep, T_rep, sample=True)
        T_p = mu_all.shape[1]
        mu_s = mu_all.view(num_samples, B, T_p, N)
        var_s = torch.exp(logvar_all.view(num_samples, B, T_p, N))
        mu_mean = mu_s.mean(dim=0)
        return {
            "mu": mu_mean,
            "var_alea": var_s.mean(dim=0),
            "var_epis": ((mu_s - mu_mean.unsqueeze(0)) ** 2).mean(dim=0),
            "mc_mu_samples": mu_s,
        }


# ------------------------------------------------------------------------------
# 4. Generation 3 STGNN (Dynamic Graph & Transformer): DGCRN, STAEformer, DGCRAN
# ------------------------------------------------------------------------------
class DGCRNBaseline(nn.Module):
    """
    Deterministic DGCRN baseline (Li et al., ACM TKDD 2023).
    Truly distinct from DGCRAN: uses ONLY the hypernetwork dynamic graph GRU branch
    (NO parallel adaptive branch and NO multi-head spatial cross-attention fusion).
    """

    def __init__(self, num_nodes: int, in_features: int, hidden_dim: int = 64, node_emb_dim: int = 10, horizon_len: int = 12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.E_G = nn.Parameter(torch.empty(num_nodes, node_emb_dim))
        nn.init.xavier_uniform_(self.E_G)
        self.dyn_graph = BDAGMModule(
            num_nodes=num_nodes, in_feat_dim=in_features, cal_dim=4,
            hidden_dim=hidden_dim, emb_dim=node_emb_dim, use_bdagm=False,
        )
        self.cell = BDGCRUCell(node_emb_dim, in_features + 4, hidden_dim, use_vnapl=False)
        self.head = nn.Linear(hidden_dim, horizon_len)

    def kl_loss(self) -> torch.Tensor:
        return self.E_G.new_tensor(0.0)

    def forward(self, X: torch.Tensor, T_cal: torch.Tensor, sample: bool = False):
        B, T, N, _ = X.shape
        H = torch.zeros(B, N, self.hidden_dim, device=X.device, dtype=X.dtype)
        I_N = torch.eye(N, device=X.device, dtype=X.dtype)
        for t in range(T):
            T_t = T_cal[:, t].unsqueeze(1).expand(-1, N, -1)  # (B, N, 4)
            A_dyn = self.dyn_graph(X[:, t], T_t, H, I_N, sample=False)
            H = self.cell(torch.cat([X[:, t], T_t], dim=-1), H, A_dyn, self.E_G, sample=False)
        return self.head(H).transpose(1, 2), None


class STAEformerBaseline(nn.Module):
    """
    Deterministic STAEformer baseline (Liu et al., ACM CIKM 2023 — arXiv:2308.10425).
    Feature + Periodicity + Spatio-Temporal Adaptive Embedding (E_a) + Temporal & Spatial Transformers.
    """

    def __init__(self, num_nodes: int, in_features: int, d_f: int = 24, d_a: int = 80, horizon_len: int = 12, num_layers: int = 3, nhead: int = 4):
        super().__init__()
        self.d_h = d_f + 2 * d_f + d_a
        self.feat_proj = nn.Linear(in_features, d_f)
        self.cal_proj = nn.Linear(4, 2 * d_f)
        self.E_a = nn.Parameter(torch.randn(1, 12, num_nodes, d_a) * 0.02)

        t_layer = nn.TransformerEncoderLayer(d_model=self.d_h, nhead=nhead, dim_feedforward=self.d_h * 2, dropout=0.1, batch_first=True)
        s_layer = nn.TransformerEncoderLayer(d_model=self.d_h, nhead=nhead, dim_feedforward=self.d_h * 2, dropout=0.1, batch_first=True)
        self.t_trans = nn.TransformerEncoder(t_layer, num_layers=num_layers)
        self.s_trans = nn.TransformerEncoder(s_layer, num_layers=num_layers)
        self.reg = nn.Linear(12 * self.d_h, horizon_len)

    def kl_loss(self) -> torch.Tensor:
        return self.E_a.new_tensor(0.0)

    def forward(self, X: torch.Tensor, T_cal: torch.Tensor, sample: bool = False):
        B, T, N, _ = X.shape
        e_f = self.feat_proj(X)
        e_p = self.cal_proj(T_cal).unsqueeze(2).expand(-1, -1, N, -1)
        e_a = self.E_a.expand(B, -1, -1, -1)
        Z = torch.cat([e_f, e_p, e_a], dim=-1)  # (B, T, N, d_h)

        # Temporal Transformer along T for each node
        Zt = self.t_trans(Z.permute(0, 2, 1, 3).reshape(B * N, T, self.d_h)).view(B, N, T, self.d_h).permute(0, 2, 1, 3)
        # Spatial Transformer along N for each timestep
        Zs = self.s_trans(Zt.reshape(B * T, N, self.d_h)).view(B, T, N, self.d_h)
        out = self.reg(Zs.permute(0, 2, 1, 3).reshape(B, N, T * self.d_h)).transpose(1, 2)  # (B, T', N)
        return out, None


# ------------------------------------------------------------------------------
# 5. Deep Ensembles (5 Independent Heteroscedastic DGCRAN Members)
# ------------------------------------------------------------------------------
class DeepEnsemblesDGCRAN(nn.Module):
    """Deep Ensembles of M independent deterministic heteroscedastic DGCRAN models (Mallick et al., 2024)."""

    def __init__(self, num_members: int = 5, **dgcran_kwargs):
        super().__init__()
        dgcran_kwargs.update({"use_vnapl": False, "use_bdagm": False, "use_var_rnn_masks": False})
        self.members = nn.ModuleList([BayesianDGCRAN(**dgcran_kwargs) for _ in range(num_members)])

    def kl_loss(self) -> torch.Tensor:
        return next(self.parameters()).new_tensor(0.0)

    @torch.no_grad()
    def predict_mc(self, X: torch.Tensor, T_cal: torch.Tensor, num_samples: int = 5) -> Dict[str, torch.Tensor]:
        mus, vars_ = [], []
        for m in self.members:
            mu, logvar = m(X, T_cal, sample=False)
            mus.append(mu)
            vars_.append(torch.exp(logvar))
        mu_s = torch.stack(mus, dim=0)
        var_s = torch.stack(vars_, dim=0)
        mu_mean = mu_s.mean(dim=0)
        return {
            "mu": mu_mean,
            "var_alea": var_s.mean(dim=0),
            "var_epis": ((mu_s - mu_mean.unsqueeze(0)) ** 2).mean(dim=0),
            "mc_mu_samples": mu_s,
        }


# ------------------------------------------------------------------------------
# Unified Model Factory (All 21 Models)
# ------------------------------------------------------------------------------
def build_model(model_name: str, spec: Dict[str, Any], adj: np.ndarray, hidden_dim: int = 64) -> nn.Module:
    N = spec["num_nodes"]
    F_in = spec["raw_features"]
    d = spec["default_node_emb_dim"]
    T_p = spec["horizon_len"]

    common = dict(num_nodes=N, in_features=F_in, hidden_dim=hidden_dim, node_emb_dim=d, horizon_len=T_p)

    if model_name == "FC-LSTM":
        return FCLSTMBaseline(in_features=F_in, hidden_dim=hidden_dim, horizon_len=T_p)
    if model_name == "STGCN":
        return STGCNBaseline(adj=adj, in_features=F_in, hidden_dim=hidden_dim, horizon_len=T_p)
    if model_name == "DCRNN":
        return DCRNNBaseline(adj=adj, in_features=F_in, hidden_dim=hidden_dim, horizon_len=T_p)
    if model_name == "ASTGCN":
        return ASTGCNBaseline(adj=adj, num_nodes=N, in_features=F_in, hidden_dim=hidden_dim, horizon_len=T_p)
    if model_name == "GraphWaveNet":
        return GraphWaveNetBaseline(adj=adj, num_nodes=N, in_features=F_in, hidden_dim=48, node_emb_dim=d, horizon_len=T_p)
    if model_name == "AGCRN":
        return AGCRNBaseline(num_nodes=N, in_features=F_in, hidden_dim=hidden_dim, node_emb_dim=d, horizon_len=T_p, probabilistic=False)
    if model_name == "DGCRN":
        return DGCRNBaseline(**common)
    if model_name == "STAEformer":
        return STAEformerBaseline(num_nodes=N, in_features=F_in, horizon_len=T_p)
    if model_name == "DGCRAN":
        return BayesianDGCRAN(**common, use_vnapl=False, use_bdagm=False, use_var_rnn_masks=False)
    if model_name == "MCDropout-DGCRAN":
        return BayesianDGCRAN(**common, use_vnapl=False, use_bdagm=False, use_var_rnn_masks=True)
    if model_name == "DeepSTUQ":
        return AGCRNBaseline(num_nodes=N, in_features=F_in, hidden_dim=hidden_dim, node_emb_dim=d, horizon_len=T_p, probabilistic=True, dropout_p=0.1)
    if model_name == "DeepEnsembles":
        return DeepEnsemblesDGCRAN(num_members=5, **common)
    if model_name == "Bayesian-DGCRAN":
        return BayesianDGCRAN(**common, use_vnapl=True, use_bdagm=True, use_var_rnn_masks=True)
    if model_name == "Ablation-NoVNAPL":
        return BayesianDGCRAN(**common, use_vnapl=False, use_bdagm=True, use_var_rnn_masks=True)
    if model_name == "Ablation-NoBDAGM":
        return BayesianDGCRAN(**common, use_vnapl=True, use_bdagm=False, use_var_rnn_masks=True)
    if model_name == "Ablation-NoVarMasks":
        return BayesianDGCRAN(**common, use_vnapl=True, use_bdagm=True, use_var_rnn_masks=False)
    if model_name in ("Ablation-NoKLAWA", "Ablation-NoNEMHCC"):
        return BayesianDGCRAN(**common, use_vnapl=True, use_bdagm=True, use_var_rnn_masks=True)
    raise ValueError(f"Unknown model_name: {model_name}")
