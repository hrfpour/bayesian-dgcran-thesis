"""
Complete Bayesian-DGCRAN Architecture (Chapter 3, Section 3.4).
Supports all 5 ablation switches:
  - use_vnapl (Innovation 1)
  - use_bdagm (Innovation 2)
  - use_var_rnn_masks (Innovation 3)
  - (Innovation 4 KL-Annealing & AWA controlled in trainer.py)
  - (Innovation 5 NE-MHCC controlled in nemhcc.py)
"""
from __future__ import annotations
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.bayesian_layers import (
    BDAGMModule,
    BDGCRUCell,
    MultiHeadSpatialCrossAttention,
    HSTDecoder,
)


class BayesianDGCRAN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        cal_features: int = 4,
        hidden_dim: int = 64,
        node_emb_dim: int = 12,
        graph_emb_dim: int = 10,
        horizon_len: int = 12,
        num_heads: int = 4,
        p_x: float = 0.05,
        p_h: float = 0.10,
        prior_sigma: float = 1.0,
        use_vnapl: bool = True,
        use_bdagm: bool = True,
        use_var_rnn_masks: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.p_x = p_x
        self.p_h = p_h
        self.use_var_rnn_masks = use_var_rnn_masks

        # Shared node embeddings E_A (adaptive graph) and E_G (node parameter pool)
        self.E_A = nn.Parameter(torch.empty(num_nodes, graph_emb_dim))
        self.E_G = nn.Parameter(torch.empty(num_nodes, node_emb_dim))
        nn.init.xavier_uniform_(self.E_A)
        nn.init.xavier_uniform_(self.E_G)

        # Innovation 2: Bayesian Dynamic-Adaptive Graph Module
        self.bdagm = BDAGMModule(
            num_nodes=num_nodes,
            in_feat_dim=in_features,
            cal_dim=cal_features,
            hidden_dim=hidden_dim,
            emb_dim=graph_emb_dim,
            use_bdagm=use_bdagm,
        )

        # Parallel Branch 1 (Dynamic): BDGCRU
        self.bdgcru = BDGCRUCell(node_emb_dim, in_features + cal_features, hidden_dim, use_vnapl=use_vnapl, prior_sigma=prior_sigma)
        # Parallel Branch 2 (Static-Adaptive): B-AGCRU (Section 3.4.4)
        self.bagcru = BDGCRUCell(node_emb_dim, in_features + cal_features, hidden_dim, use_vnapl=use_vnapl, prior_sigma=prior_sigma)

        # Innovation 4: Cross-Attention Fusion & Dual-Head Heteroscedastic Decoder
        self.cross_attn = MultiHeadSpatialCrossAttention(hidden_dim, num_heads=num_heads)
        self.decoder = HSTDecoder(2 * hidden_dim, horizon_len=horizon_len, dec_dim=hidden_dim)

    def _sample_time_invariant_masks(self, B: int, device: torch.device, dtype: torch.dtype):
        if not self.use_var_rnn_masks:
            return None, None
        keep_x = 1.0 - self.p_x
        keep_h = 1.0 - self.p_h
        z_x = (torch.rand(B, self.num_nodes, self.in_features, device=device, dtype=dtype) < keep_x).to(dtype) / keep_x
        z_h = (torch.rand(B, self.num_nodes, self.hidden_dim, device=device, dtype=dtype) < keep_h).to(dtype) / keep_h
        return z_x, z_h

    def kl_loss(self) -> torch.Tensor:
        return self.bdgcru.kl_loss() + self.bagcru.kl_loss() + self.bdagm.edge_entropy_reg()

    def forward(self, X: torch.Tensor, T_cal: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        X: (B, T, N, F)
        T_cal: (B, T, 4)
        Returns (mu, logvar) each of shape (B, T', N)
        """
        B, T, N, _ = X.shape
        device, dtype = X.device, X.dtype

        # Algorithm 3-1 Step 2-b-1: compute A_adp INSIDE mini-batch forward for live gradients to E_A
        A_adp = F.softmax(F.relu(torch.matmul(self.E_A, self.E_A.t())), dim=-1)  # (N, N)
        A_static_adp = torch.eye(N, device=device, dtype=dtype) + A_adp

        z_x, z_h = self._sample_time_invariant_masks(B, device, dtype) if sample else (None, None)
        H_dyn = torch.zeros(B, N, self.hidden_dim, device=device, dtype=dtype)
        H_adp = torch.zeros(B, N, self.hidden_dim, device=device, dtype=dtype)

        for t in range(T):
            X_t = X[:, t]  # (B, N, F)
            T_t = T_cal[:, t].unsqueeze(1).expand(B, N, -1)  # (B, N, 4)
            A_t = self.bdagm(X_t, T_t, H_dyn, A_adp, sample=sample)
            X_in = (X_t * z_x) if (z_x is not None and sample) else X_t
            X_cat = torch.cat([X_in, T_t], dim=-1)
            H_dyn = self.bdgcru(X_cat, H_dyn, A_t, self.E_G, z_x=None, z_h=z_h, sample=sample)
            H_adp = self.bagcru(X_cat, H_adp, A_static_adp, self.E_G, z_x=None, z_h=z_h, sample=sample)

        H_cross = self.cross_attn(H_dyn, H_adp)
        H_fus = torch.cat([H_cross, H_adp], dim=-1)
        return self.decoder(H_fus)

    @torch.no_grad()
    def predict_mc(self, X: torch.Tensor, T_cal: torch.Tensor, num_samples: int = 10) -> Dict[str, torch.Tensor]:
        """
        Parallel M-pass Monte Carlo inference (Algorithm 3-2) with Law of Total Variance decomposition:
          var_alea = mean_m exp(s^(m))
          var_epis = mean_m (mu^(m) - mu_bar)^2
        """
        B, T, N, F_dim = X.shape
        X_rep = X.unsqueeze(0).expand(num_samples, -1, -1, -1, -1).reshape(num_samples * B, T, N, F_dim)
        T_rep = T_cal.unsqueeze(0).expand(num_samples, -1, -1, -1).reshape(num_samples * B, T, T_cal.shape[-1])
        mu_all, logvar_all = self.forward(X_rep, T_rep, sample=True)
        T_prime = mu_all.shape[1]
        mu_samples = mu_all.view(num_samples, B, T_prime, N)
        var_samples = torch.exp(logvar_all.view(num_samples, B, T_prime, N))

        mu_mean = mu_samples.mean(dim=0)
        var_alea = var_samples.mean(dim=0)
        var_epis = ((mu_samples - mu_mean.unsqueeze(0)) ** 2).mean(dim=0)
        return {
            "mu": mu_mean,
            "var_alea": var_alea,
            "var_epis": var_epis,
            "mc_mu_samples": mu_samples,
        }
