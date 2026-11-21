"""
Core Bayesian Spatio-Temporal Layers of Bayesian-DGCRAN (Chapter 3, Section 3.4):
  1. VNAPLGraphConv: Variational Node-Adaptive Parameter Learning with Local Reparameterization (Theorem 3.4.1)
  2. BDAGMModule: Bayesian Dynamic-Adaptive Graph Module with Concrete Topological Mask (Eqs. 3.29-3.32)
  3. BDGCRUCell: Bayesian Dynamic Graph Convolutional Recurrent Unit with Time-Invariant Masks (Eqs. 3.33-3.38)
  4. MultiHeadSpatialCrossAttention & HSTDecoder: Dual-head Heteroscedastic Decoder with clamping (Eqs. 3.39-3.42)
"""
from __future__ import annotations
import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kl_divergence(mu: torch.Tensor, sigma: torch.Tensor, prior_sigma: float = 1.0) -> torch.Tensor:
    """Exact closed-form KL(N(mu, sigma^2) || N(0, prior_sigma^2)) — Lemma 3.2.1 Eq. (3.13)."""
    var = sigma.pow(2)
    prior_var = prior_sigma ** 2
    return 0.5 * torch.sum((var + mu.pow(2)) / prior_var - 1.0 - torch.log(var / prior_var + 1e-12))


class VNAPLGraphConv(nn.Module):
    """
    Innovation 1: Variational Node-Adaptive Parameter Learning (V-NAPL).
    Maintains a low-rank variational weight pool W_G ~ N(M_W, softplus^2(P_W)) of shape (d, C_in, C_out)
    and samples pre-activations directly via Local Reparameterization (Kingma et al., 2015; Theorem 3.4.1).
    """

    def __init__(
        self,
        node_emb_dim: int,
        c_in: int,
        c_out: int,
        use_vnapl: bool = True,
        prior_sigma: float = 1.0,
        init_rho: float = -3.0,
        eps_num: float = 1e-8,
    ):
        super().__init__()
        self.d = node_emb_dim
        self.c_in = c_in
        self.c_out = c_out
        self.use_vnapl = use_vnapl
        self.prior_sigma = prior_sigma
        self.eps_num = eps_num

        # Variational mean pools M_W, M_b
        self.M_W = nn.Parameter(torch.empty(node_emb_dim, c_in, c_out))
        self.M_b = nn.Parameter(torch.zeros(node_emb_dim, c_out))
        nn.init.xavier_normal_(self.M_W)

        if use_vnapl:
            # Variational log-scale parameters P_W, P_b -> Sigma = softplus(P)^2
            self.P_W = nn.Parameter(torch.full((node_emb_dim, c_in, c_out), init_rho))
            self.P_b = nn.Parameter(torch.full((node_emb_dim, c_out), init_rho))
        else:
            self.register_parameter("P_W", None)
            self.register_parameter("P_b", None)

    def kl_loss(self) -> torch.Tensor:
        if not self.use_vnapl:
            return self.M_W.new_tensor(0.0)
        sig_w = F.softplus(self.P_W)
        sig_b = F.softplus(self.P_b)
        return gaussian_kl_divergence(self.M_W, sig_w, self.prior_sigma) + gaussian_kl_divergence(self.M_b, sig_b, self.prior_sigma)

    def forward(self, A_t: torch.Tensor, X_t: torch.Tensor, E_G: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        A_t: (B, N, N) or (N, N) transition operator
        X_t: (B, N, C_in)
        E_G: (N, d)
        Returns Z_t: (B, N, C_out)
        """
        # Graph propagation: X_tilde = A_t @ X_t -> (B, N, C_in)
        if A_t.dim() == 2:
            X_tilde = torch.einsum("nm,bmc->bnc", A_t, X_t)
        else:
            X_tilde = torch.bmm(A_t, X_t)

        # Two-stage tensor contraction for local mean Gamma_t (Eq. 3.27)
        # Step 1: node weights mean Theta_mu = E_G @ M_W -> (N, C_in, C_out)
        theta_mu = torch.einsum("nd,dco->nco", E_G, self.M_W)
        bias_mu = torch.einsum("nd,do->no", E_G, self.M_b)
        gamma = torch.einsum("bnc,nco->bno", X_tilde, theta_mu) + bias_mu.unsqueeze(0)

        if (not self.use_vnapl) or (not sample):
            return gamma

        # Two-stage tensor contraction for local variance Delta_t (Eq. 3.28)
        sig_w_sq = F.softplus(self.P_W).pow(2)
        sig_b_sq = F.softplus(self.P_b).pow(2)
        E_G_sq = E_G.pow(2)
        theta_var = torch.einsum("nd,dco->nco", E_G_sq, sig_w_sq)
        bias_var = torch.einsum("nd,do->no", E_G_sq, sig_b_sq)
        delta = torch.einsum("bnc,nco->bno", X_tilde.pow(2), theta_var) + bias_var.unsqueeze(0)

        # Local Reparameterization Trick (Eq. 3.26)
        zeta = torch.randn_like(gamma)
        return gamma + torch.sqrt(delta + self.eps_num) * zeta


class BDAGMModule(nn.Module):
    """
    Innovation 2: Bayesian Dynamic-Adaptive Graph Module (B-DAGM).
    Generates directed antisymmetric dynamic adjacency via source/target hypernetworks
    and applies a learnable Concrete topological edge mask (Gal et al., 2017; Hasanzadeh et al., 2020).
    """

    def __init__(
        self,
        num_nodes: int,
        in_feat_dim: int,
        cal_dim: int,
        hidden_dim: int,
        emb_dim: int = 10,
        use_bdagm: bool = True,
        alpha_hyp: float = 3.0,
        concrete_temp: float = 0.2,
        init_keep_prob: float = 0.9,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.use_bdagm = use_bdagm
        self.alpha_hyp = alpha_hyp
        self.tau_c = concrete_temp

        hyp_in = in_feat_dim + cal_dim + hidden_dim
        self.hyper_src = nn.Sequential(nn.Linear(hyp_in, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))
        self.hyper_dst = nn.Sequential(nn.Linear(hyp_in, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))

        # Learnable source/target static embeddings E_1, E_2
        self.E_1 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)
        self.E_2 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)

        # Learnable dynamic branch weight gamma_dyn in (0, 1]
        self.raw_gamma_dyn = nn.Parameter(torch.tensor(0.0))

        # Learnable Concrete edge-keep logit eta_edge (pi_edge = sigmoid(eta_edge))
        init_logit = math.log(init_keep_prob / (1.0 - init_keep_prob))
        self.eta_edge = nn.Parameter(torch.tensor(init_logit))

    def edge_entropy_reg(self, k_edge: float = 1e-4) -> torch.Tensor:
        if not self.use_bdagm:
            return self.eta_edge.new_tensor(0.0)
        pi_edge = torch.sigmoid(self.eta_edge).clamp(1e-4, 1.0 - 1e-4)
        pi_drop = 1.0 - pi_edge
        entropy = -(pi_drop * torch.log(pi_drop) + pi_edge * torch.log(pi_edge))
        return -k_edge * entropy

    def forward(
        self,
        X_t: torch.Tensor,
        T_t: torch.Tensor,
        H_prev: torch.Tensor,
        A_adp: torch.Tensor,
        sample: bool = True,
    ) -> torch.Tensor:
        """
        X_t: (B, N, F), T_t: (B, N, d_t), H_prev: (B, N, d_h), A_adp: (N, N)
        Returns A_t: (B, N, N)
        """
        B, N, _ = X_t.shape
        I_t = torch.cat([X_t, T_t, H_prev], dim=-1)  # (B, N, F + d_t + d_h)
        DF1 = self.hyper_src(I_t)  # (B, N, d_e)
        DF2 = self.hyper_dst(I_t)  # (B, N, d_e)

        DE1 = torch.tanh(self.alpha_hyp * (DF1 * self.E_1.unsqueeze(0)))
        DE2 = torch.tanh(self.alpha_hyp * (DF2 * self.E_2.unsqueeze(0)))

        # Directed antisymmetric dynamic similarity (Eq. 3.30)
        sim = torch.bmm(DE1, DE2.transpose(1, 2)) - torch.bmm(DE2, DE1.transpose(1, 2))
        DA_norm = F.softmax(F.relu(torch.tanh(self.alpha_hyp * sim)), dim=-1)  # (B, N, N)

        if self.use_bdagm and sample:
            pi_edge = torch.sigmoid(self.eta_edge).clamp(0.05, 0.995)
            pi_drop = 1.0 - pi_edge
            u = torch.rand_like(DA_norm).clamp(1e-6, 1.0 - 1e-6)
            logits = (torch.log(pi_edge / pi_drop) + torch.log(u / (1.0 - u))) / self.tau_c
            omega = torch.sigmoid(logits) / pi_edge  # inverse-scaled Concrete mask (Eq. 3.31)
            DA_eff = omega * DA_norm
        else:
            DA_eff = DA_norm

        gamma_dyn = torch.sigmoid(self.raw_gamma_dyn)
        eye = torch.eye(N, device=X_t.device, dtype=X_t.dtype).unsqueeze(0)
        return eye + A_adp.unsqueeze(0) + gamma_dyn * DA_eff


class BDGCRUCell(nn.Module):
    """
    Innovation 3: Bayesian Dynamic Graph Convolutional Recurrent Unit (BDGCRU).
    Also serves as B-AGCRU when given static-adaptive operator A_adp (Section 3.4.4).
    """

    def __init__(
        self,
        node_emb_dim: int,
        in_dim: int,
        hidden_dim: int,
        use_vnapl: bool = True,
        prior_sigma: float = 1.0,
    ):
        super().__init__()
        c_in = in_dim + hidden_dim
        self.gate_r = VNAPLGraphConv(node_emb_dim, c_in, hidden_dim, use_vnapl=use_vnapl, prior_sigma=prior_sigma)
        self.gate_u = VNAPLGraphConv(node_emb_dim, c_in, hidden_dim, use_vnapl=use_vnapl, prior_sigma=prior_sigma)
        self.gate_c = VNAPLGraphConv(node_emb_dim, c_in, hidden_dim, use_vnapl=use_vnapl, prior_sigma=prior_sigma)

    def kl_loss(self) -> torch.Tensor:
        return self.gate_r.kl_loss() + self.gate_u.kl_loss() + self.gate_c.kl_loss()

    def forward(
        self,
        X_t: torch.Tensor,
        H_prev: torch.Tensor,
        A_t: torch.Tensor,
        E_G: torch.Tensor,
        z_x: Optional[torch.Tensor] = None,
        z_h: Optional[torch.Tensor] = None,
        sample: bool = True,
    ) -> torch.Tensor:
        X_masked = X_t * z_x if z_x is not None else X_t
        H_masked = H_prev * z_h if z_h is not None else H_prev
        concat_ru = torch.cat([X_masked, H_masked], dim=-1)
        r_t = torch.sigmoid(self.gate_r(A_t, concat_ru, E_G, sample=sample))
        u_t = torch.sigmoid(self.gate_u(A_t, concat_ru, E_G, sample=sample))
        concat_c = torch.cat([X_masked, r_t * H_masked], dim=-1)
        c_tilde = torch.tanh(self.gate_c(A_t, concat_c, E_G, sample=sample))
        # Direct unmasked highway for H_prev preserves Constant Error Carousel (Gal & Ghahramani, 2016)
        return u_t * H_prev + (1.0 - u_t) * c_tilde


class MultiHeadSpatialCrossAttention(nn.Module):
    """
    Multi-Head Spatial Cross-Attention fusing H_dyn (Query) with H_adp (Key, Value) — Eqs. (3.39)-(3.40).
    Uses Pre-LayerNorm + learnable residual gate (gamma_attn init=0.1) so per-sensor flow magnitude
    and local node identity are strictly preserved while adding global cross-branch context.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_O = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gamma_attn = nn.Parameter(torch.tensor(0.1))

    def forward(self, H_dyn: torch.Tensor, H_adp: torch.Tensor) -> torch.Tensor:
        B, N, D = H_dyn.shape
        q_in = self.norm_q(H_dyn)
        kv_in = self.norm_kv(H_adp)
        Q = self.W_Q(q_in).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(kv_in).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(kv_in).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, N, D)
        return H_dyn + self.gamma_attn * self.W_O(out)


class HSTDecoder(nn.Module):
    """
    Innovation 4: Heteroscedastic Spatio-Temporal Probabilistic Decoder (Eqs. 3.41-3.42).
    Uses direct linear residual skip + SiLU nonlinear refinement (no ReLU clipping of negative Z-score
    nighttime traffic states) and smooth clamping [s_min, s_max] on log-aleatoric variance.
    """

    def __init__(
        self,
        hidden_dim: int,
        horizon_len: int = 12,
        dec_dim: int = 64,
        s_min: float = -10.0,
        s_max: float = 5.0,
    ):
        super().__init__()
        self.s_min = s_min
        self.s_max = s_max
        self.mu_skip = nn.Linear(hidden_dim, horizon_len)
        self.mu_mlp = nn.Sequential(nn.Linear(hidden_dim, dec_dim), nn.SiLU(), nn.Linear(dec_dim, horizon_len))
        self.logvar_head = nn.Sequential(nn.Linear(hidden_dim, dec_dim), nn.ELU(), nn.Linear(dec_dim, horizon_len))

    def forward(self, H_fus: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = (self.mu_skip(H_fus) + self.mu_mlp(H_fus)).transpose(1, 2)
        logvar = torch.clamp(self.logvar_head(H_fus).transpose(1, 2), min=self.s_min, max=self.s_max)
        return mu, logvar
