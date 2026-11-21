"""
Complete Bayesian-DGCRAN Architecture (v2.4 Gold — Strictly Aligned with Chapter 3, Section 3.4).
Upgrades:
  - Learnable discrete Time-of-Day (288) & Day-of-Week (7) periodic embeddings + Node Identity embedding
  - 2-layer stacked BDGCRU & B-AGCRU branches with residual connections (prevents over-smoothing)
  - Convex-normalized B-DAGM adjacency (spectral row-sum = 2.0)
  - Pre-LN gated MultiHeadSpatialCrossAttention + concatenated [H_cross || H_adp || E_id] into HSTDecoder
Supports all 5 ablation switches:
  - use_vnapl (Innovation 1), use_bdagm (Innovation 2), use_var_rnn_masks (Innovation 3)
  - (Innovation 4 KL-Annealing & AWA in trainer.py; Innovation 5 NE-MHCC in nemhcc.py)
"""
from __future__ import annotations
from typing import Tuple, Dict
import math
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
        num_layers: int = 2,
        tod_emb_dim: int = 16,
        dow_emb_dim: int = 8,
        node_id_dim: int = 16,
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
        self.num_layers = num_layers
        self.p_x = p_x
        self.p_h = p_h
        self.use_var_rnn_masks = use_var_rnn_masks

        # Learnable discrete periodic & node-identity embeddings
        self.tod_emb = nn.Embedding(288, tod_emb_dim)
        self.dow_emb = nn.Embedding(7, dow_emb_dim)
        self.node_id_emb = nn.Parameter(torch.randn(num_nodes, node_id_dim) * 0.05)

        # Shared node embeddings E_A (adaptive graph) and E_G (node parameter pool)
        self.E_A = nn.Parameter(torch.empty(num_nodes, graph_emb_dim))
        self.E_G = nn.Parameter(torch.empty(num_nodes, node_emb_dim))
        nn.init.xavier_uniform_(self.E_A)
        nn.init.xavier_uniform_(self.E_G)

        # Innovation 2: Bayesian Dynamic-Adaptive Graph Module (convex spectral bound)
        self.bdagm = BDAGMModule(
            num_nodes=num_nodes,
            in_feat_dim=in_features,
            cal_dim=cal_features,
            hidden_dim=hidden_dim,
            emb_dim=graph_emb_dim,
            use_bdagm=use_bdagm,
        )

        # Rich per-step input dimension: raw features + sin/cos + tod_emb + dow_emb + node_id
        step_in_dim = in_features + cal_features + tod_emb_dim + dow_emb_dim + node_id_dim

        # Parallel Branch 1 (Dynamic): Stacked BDGCRU layers
        self.bdgcru_layers = nn.ModuleList([
            BDGCRUCell(
                node_emb_dim,
                step_in_dim if l == 0 else hidden_dim,
                hidden_dim,
                use_vnapl=use_vnapl,
                prior_sigma=prior_sigma,
            )
            for l in range(num_layers)
        ])
        # Parallel Branch 2 (Static-Adaptive): Stacked B-AGCRU layers
        self.bagcru_layers = nn.ModuleList([
            BDGCRUCell(
                node_emb_dim,
                step_in_dim if l == 0 else hidden_dim,
                hidden_dim,
                use_vnapl=use_vnapl,
                prior_sigma=prior_sigma,
            )
            for l in range(num_layers)
        ])

        # Innovation 4: Gated Pre-LN Spatial Cross-Attention & Dual-Head Heteroscedastic Decoder
        self.cross_attn = MultiHeadSpatialCrossAttention(hidden_dim, num_heads=num_heads)
        self.decoder = HSTDecoder(
            2 * hidden_dim + node_id_dim + tod_emb_dim + dow_emb_dim,
            horizon_len=horizon_len,
            dec_dim=hidden_dim,
        )

    @staticmethod
    def _extract_cal_indices(T_cal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # T_cal: (B, T, 4) = [sin_tod, cos_tod, sin_dow, cos_dow]
        ang_tod = torch.atan2(T_cal[..., 0], T_cal[..., 1])
        ang_tod = torch.where(ang_tod < 0, ang_tod + 2.0 * math.pi, ang_tod)
        tod_idx = torch.round(ang_tod * 288.0 / (2.0 * math.pi)).long().remainder(288)

        ang_dow = torch.atan2(T_cal[..., 2], T_cal[..., 3])
        ang_dow = torch.where(ang_dow < 0, ang_dow + 2.0 * math.pi, ang_dow)
        dow_idx = torch.round(ang_dow * 7.0 / (2.0 * math.pi)).long().remainder(7)
        return tod_idx, dow_idx

    def _sample_time_invariant_masks(self, B: int, device: torch.device, dtype: torch.dtype):
        if not self.use_var_rnn_masks:
            return None, None
        keep_x = 1.0 - self.p_x
        keep_h = 1.0 - self.p_h
        z_x = (torch.rand(B, self.num_nodes, self.in_features, device=device, dtype=dtype) < keep_x).to(dtype) / keep_x
        z_h = (torch.rand(B, self.num_nodes, self.hidden_dim, device=device, dtype=dtype) < keep_h).to(dtype) / keep_h
        return z_x, z_h

    def kl_loss(self) -> torch.Tensor:
        kl = self.bdagm.edge_entropy_reg()
        for cell in self.bdgcru_layers:
            kl = kl + cell.kl_loss()
        for cell in self.bagcru_layers:
            kl = kl + cell.kl_loss()
        return kl

    def forward(self, X: torch.Tensor, T_cal: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, N, _ = X.shape
        device, dtype = X.device, X.dtype

        tod_idx, dow_idx = self._extract_cal_indices(T_cal)
        e_tod_seq = self.tod_emb(tod_idx).to(dtype)  # (B, T, d_tod)
        e_dow_seq = self.dow_emb(dow_idx).to(dtype)  # (B, T, d_dow)
        e_id = self.node_id_emb.to(dtype).unsqueeze(0).expand(B, N, -1)  # (B, N, d_id)

        # Algorithm 3-1 Step 2-b-1: compute A_adp INSIDE mini-batch forward for live gradients to E_A
        A_adp = F.softmax(F.relu(torch.matmul(self.E_A, self.E_A.t())), dim=-1)
        A_static_adp = torch.eye(N, device=device, dtype=dtype) + A_adp

        z_x, z_h = self._sample_time_invariant_masks(B, device, dtype) if sample else (None, None)
        H_dyn_list = [torch.zeros(B, N, self.hidden_dim, device=device, dtype=dtype) for _ in range(self.num_layers)]
        H_adp_list = [torch.zeros(B, N, self.hidden_dim, device=device, dtype=dtype) for _ in range(self.num_layers)]

        for t in range(T):
            X_t = X[:, t]  # (B, N, F)
            T_t = T_cal[:, t].unsqueeze(1).expand(B, N, -1)
            tod_t = e_tod_seq[:, t].unsqueeze(1).expand(B, N, -1)
            dow_t = e_dow_seq[:, t].unsqueeze(1).expand(B, N, -1)

            A_t = self.bdagm(X_t, T_t, H_dyn_list[0], A_adp, sample=sample)
            X_in = (X_t * z_x) if (z_x is not None and sample) else X_t
            inp_dyn = torch.cat([X_in, T_t, tod_t, dow_t, e_id], dim=-1)
            inp_adp = inp_dyn

            for l in range(self.num_layers):
                h_d_new = self.bdgcru_layers[l](inp_dyn, H_dyn_list[l], A_t, self.E_G, z_x=None, z_h=z_h, sample=sample)
                h_a_new = self.bagcru_layers[l](inp_adp, H_adp_list[l], A_static_adp, self.E_G, z_x=None, z_h=z_h, sample=sample)
                if l > 0:
                    h_d_new = h_d_new + inp_dyn
                    h_a_new = h_a_new + inp_adp
                H_dyn_list[l] = h_d_new
                H_adp_list[l] = h_a_new
                inp_dyn = h_d_new
                inp_adp = h_a_new

        H_dyn_final = H_dyn_list[-1]
        H_adp_final = H_adp_list[-1]
        H_cross = self.cross_attn(H_dyn_final, H_adp_final)
        tod_last = e_tod_seq[:, -1].unsqueeze(1).expand(B, N, -1)
        dow_last = e_dow_seq[:, -1].unsqueeze(1).expand(B, N, -1)
        H_fus = torch.cat([H_cross, H_adp_final, e_id, tod_last, dow_last], dim=-1)
        return self.decoder(H_fus)

    @torch.no_grad()
    def predict_mc(self, X: torch.Tensor, T_cal: torch.Tensor, num_samples: int = 10) -> Dict[str, torch.Tensor]:
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
