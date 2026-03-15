"""
Modality-Aware Mixture-of-Experts (MA-MoE) Fusion Module v2
============================================================
v2 changes (2026-03-14):
  - Replaced CrossAttentionExpert with FeatureGatingExpert to fix ACC↑ F1↓:
    CrossAttentionExpert used (N,1,D) tensors causing node-level global pooling:
      o[i] = Σ_j softmax(z1[i]·z2[j]^T/√D) · z2[j]   <- majority-cluster bias!
    FeatureGatingExpert uses dimension-wise sigmoid gate instead:
      g = sigmoid(W·[z1, z2]) ∈ (0,1)^D  (independent per node, per dim)
      out = g ⊙ z1 + (1-g) ⊙ z2         (no cross-node aggregation)

Unchanged from v1:
  - HadamardExpert, VarianceExpert, DifferenceExpert
  - Rich Gate Input, Noisy Gating, Load-Balancing Loss, Residual Gating
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Specialist Experts v2 ---------------------------------------------------

class FeatureGatingExpert(nn.Module):
    """Dimension-wise cross-modal gating — no cross-node aggregation.

    For each node independently:
      g = sigmoid(MLP([z1, z2]))   shape: (N, D)
      out = g ⊙ z1 + (1-g) ⊙ z2 + refine(out)

    This differs critically from CrossAttentionExpert which attended from
    every node to every other node (global pooling → majority-cluster bias).
    Here each node's output depends ONLY on that node's own z1 and z2.
    """
    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.refine = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate_net(torch.cat([z1, z2], dim=-1)))  # (N, D)
        out = g * z1 + (1.0 - g) * z2                                  # (N, D)
        return out + self.refine(out)                                   # residual


class HadamardExpert(nn.Module):
    """Element-wise product + MLP (effective when modalities share feature axes)."""
    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return self.net(z1 * z2)


class VarianceExpert(nn.Module):
    """Adaptive variance-weighted fusion with learnable temperature + refinement net."""
    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        temp = torch.exp(self.log_temp).clamp(0.1, 10.0)
        v1 = torch.var(z1) * temp
        v2 = torch.var(z2) * temp
        a1 = v1 / (v1 + v2 + 1e-8)
        z_var = a1 * z1 + (1.0 - a1) * z2
        return z_var + self.refine(z_var)


class DifferenceExpert(nn.Module):
    """Difference-aware fusion: explicitly models cross-modal disagreement."""
    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.diff_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.blend = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        consensus = (z1 + z2) * 0.5
        diff_feat = self.diff_net(torch.abs(z1 - z2))
        return self.blend(torch.cat([consensus, diff_feat], dim=-1))


# ---- Gate Network ------------------------------------------------------------

class MoEGate(nn.Module):
    """Routing gate with 4x rich features and optional noisy gating."""
    def __init__(self, latent_dim: int, num_experts: int, hidden_dim: int = 128):
        super().__init__()
        gate_input_dim = latent_dim * 4  # [z1, z2, z1*z2, |z1-z2|]
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_experts),
        )
        self.noise_scale = nn.Linear(gate_input_dim, num_experts)
        nn.init.zeros_(self.noise_scale.weight)
        nn.init.zeros_(self.noise_scale.bias)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, training: bool = True):
        feat = torch.cat([z1, z2, z1 * z2, torch.abs(z1 - z2)], dim=-1)
        logits = self.gate(feat)
        if training:
            noise_std = F.softplus(self.noise_scale(feat)) + 1e-2
            logits = logits + torch.randn_like(logits) * noise_std
        return F.softmax(logits, dim=-1)


# ---- Main MoE Module ---------------------------------------------------------

class MoEFusion(nn.Module):
    """
    Modality-Aware MoE Fusion v2.

    forward() returns: (fused, gate_weights, balance_loss)
      fused        : (N, expert_output_dim) merged representation
      gate_weights : (N, num_experts) routing probabilities
      balance_loss : scalar -- add to total loss * lambda_balance
    """

    def __init__(self, z1_dim: int, z2_dim: int, expert_output_dim: int,
                 num_experts: int = 4, hidden_dim: int = 128):
        super().__init__()
        assert z1_dim == z2_dim, "MoEFusion expects equal modality dims"
        latent_dim = z1_dim
        self.num_experts = num_experts

        # 4 specialist experts (v2: CrossAttention → FeatureGating)
        self.expert0 = FeatureGatingExpert(latent_dim, hidden_dim=hidden_dim // 2)
        self.expert1 = HadamardExpert(latent_dim, hidden_dim=hidden_dim)
        self.expert2 = VarianceExpert(latent_dim, hidden_dim=hidden_dim // 2)
        self.expert3 = DifferenceExpert(latent_dim, hidden_dim=hidden_dim)

        self.gate = MoEGate(latent_dim, num_experts, hidden_dim=hidden_dim)

        self.out_proj = (
            nn.Linear(latent_dim, expert_output_dim)
            if latent_dim != expert_output_dim else nn.Identity()
        )

        # Residual gate beta: Z_final = beta*Z_moe + (1-beta)*Z_variance
        self._res_gate_raw = nn.Parameter(torch.zeros(1))  # sigmoid(0) = 0.5

    @property
    def res_gate(self) -> torch.Tensor:
        return torch.sigmoid(self._res_gate_raw)

    @staticmethod
    def _variance_fusion(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        v1 = torch.var(z1)
        v2 = torch.var(z2)
        a1 = v1 / (v1 + v2 + 1e-8)
        return a1 * z1 + (1.0 - a1) * z2

    @staticmethod
    def _balance_loss(gate_weights: torch.Tensor) -> torch.Tensor:
        """Switch-Transformer load-balancing loss."""
        n = gate_weights.shape[1]
        f = gate_weights.mean(dim=0)
        p = F.one_hot(gate_weights.argmax(dim=-1), n).float().mean(0)
        return n * (f * p).sum()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        training = self.training

        gate_weights = self.gate(z1, z2, training=training)  # (N, 4)

        expert_outs = torch.stack([
            self.expert0(z1, z2),
            self.expert1(z1, z2),
            self.expert2(z1, z2),
            self.expert3(z1, z2),
        ], dim=1)                                              # (N, 4, D)

        z_moe = self.out_proj(torch.einsum('bk,bkd->bd', gate_weights, expert_outs))

        z_var = self._variance_fusion(z1, z2)
        if z_var.shape[-1] != z_moe.shape[-1]:
            z_var = self.out_proj(z_var)
        beta = self.res_gate
        fused = beta * z_moe + (1.0 - beta) * z_var

        balance_loss = self._balance_loss(gate_weights)

        return fused, gate_weights, balance_loss