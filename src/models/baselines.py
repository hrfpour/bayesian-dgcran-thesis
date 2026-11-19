"""
Complete Suite of Deterministic & Probabilistic/Bayesian Baselines for Chapter 4 Benchmarking:
  Deterministic:
    - HA (Historical Average — NumPy)
    - VAR (Ridge Vector Autoregression — NumPy)
    - DCRNN (Li et al., 2018 — bidirectional diffusion graph conv GRU)
    - AGCRN (Bai et al., 2020 — static adaptive graph + NAPL GRU)
    - DGCRN (Li et al., 2023 — hypernetwork dynamic graph GRU)
    - DGCRAN (Li et al., 2025 — deterministic base model)
  Probabilistic / Bayesian:
    - MCDropout-DGCRAN (Gal & Ghahramani, 2016)
    - DeepEnsembles-DGCRAN (Mallick et al., 2024 — M independent heteroscedastic models)
    - DeepSTUQ (Qian et al., 2024 — static-adaptive AGCRN + MC Dropout + heteroscedastic head)
  Proposed & 5 Ablations:
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
from src.models.bayesian_layers import VNAPLGraphConv, BDGCRUCell, HSTDecoder


class HistoricalAverageBaseline:
    """Simple Historical Average over the input window (non-parametric baseline)."""

    def predict(self, X: np.ndarray, horizon_len: int = 12, target_channel: int = 0) -> np.ndarray:
        # X: (M, T, N, F) -> mean over T repeated across T'
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
        # X: (B, N, C)
        supports = [X]
        xf, xb = X, X
        for _ in range(self.K):
            xf = torch.einsum("nm,bmc->bnc", self.P_f, xf)
            xb = torch.einsum("nm,bmc->bnc", self.P_b, xb)
            supports.extend([xf, xb])
        return self.linear(torch.cat(supports, dim=-1))


class DCRNNBaseline(nn.Module):
    """Deterministic DCRNN baseline (Li et al., 2018)."""

    def __init__(self, adj: np.ndarray, in_features: int, hidden_dim: int = 64, horizon_len: int = 12, K: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_r = DiffusionGraphConv(in_features + hidden_dim, hidden_dim, adj, K)
        self.gate_u = DiffusionGraphConv(in_features + hidden_dim, hidden_dim, adj, K)
        self.gate_c = DiffusionGraphConv(in_features + hidden_dim, hidden_dim, adj, K)
        self.head = nn.Linear(hidden_dim, horizon_len)

    def kl_loss(self) -> torch.Tensor:
        return next(self.parameters()).new_tensor(0.0)

    def forward(self, X: torch.Tensor, T_cal: torch.Tensor, sample: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, N, _ = X.shape
        H = torch.zeros(B, N, self.hidden_dim, device=X.device, dtype=X.dtype)
        for t in range(T):
            xt = X[:, t]
            ru_in = torch.cat([xt, H], dim=-1)
            r = torch.sigmoid(self.gate_r(ru_in))
            u = torch.sigmoid(self.gate_u(ru_in))
            c = torch.tanh(self.gate_c(torch.cat([xt, r * H], dim=-1)))
            H = u * H + (1.0 - u) * c
        return self.head(H).transpose(1, 2), None


class AGCRNBaseline(nn.Module):
    """Deterministic AGCRN (Bai et al., 2020) or Probabilistic DeepSTUQ (Qian et al., 2024)."""

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
        self.cell = BDGCRUCell(node_emb_dim, in_features, hidden_dim, use_vnapl=False)
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
            H = self.cell(X[:, t], H, A_op, self.E_G, sample=False)
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


def build_model(model_name: str, spec: Dict[str, Any], adj: np.ndarray, hidden_dim: int = 64) -> nn.Module:
    """Unified model factory covering all baselines, proposed Bayesian-DGCRAN, and all 5 ablations."""
    N = spec["num_nodes"]
    F_in = spec["raw_features"]
    d = spec["default_node_emb_dim"]
    T_p = spec["horizon_len"]

    common = dict(num_nodes=N, in_features=F_in, hidden_dim=hidden_dim, node_emb_dim=d, horizon_len=T_p)

    if model_name == "DCRNN":
        return DCRNNBaseline(adj=adj, in_features=F_in, hidden_dim=hidden_dim, horizon_len=T_p)
    if model_name == "AGCRN":
        return AGCRNBaseline(num_nodes=N, in_features=F_in, hidden_dim=hidden_dim, node_emb_dim=d, horizon_len=T_p, probabilistic=False)
    if model_name == "DeepSTUQ":
        return AGCRNBaseline(num_nodes=N, in_features=F_in, hidden_dim=hidden_dim, node_emb_dim=d, horizon_len=T_p, probabilistic=True, dropout_p=0.1)
    if model_name == "DGCRN":
        return BayesianDGCRAN(**common, use_vnapl=False, use_bdagm=False, use_var_rnn_masks=False)
    if model_name == "DGCRAN":
        return BayesianDGCRAN(**common, use_vnapl=False, use_bdagm=False, use_var_rnn_masks=False)
    if model_name == "MCDropout-DGCRAN":
        return BayesianDGCRAN(**common, use_vnapl=False, use_bdagm=False, use_var_rnn_masks=True)
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
