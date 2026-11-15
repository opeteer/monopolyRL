"""Distributional Quantile Regression Centralized Critic (QR-Critic) for Model-3.

Estimates N=32 return quantiles Z_phi(S_global) for all 4 agents to capture bimodal
Monopoly bankruptcy vs survival distributions, supporting CVaR risk-sensitive GAE.
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_3.engine.monopoly_core import GLOBAL_STATE_DIM
from model_3.models.actor_network import init_weights


class QuantileCritic(nn.Module):
    """Model-3 Residual Centralized Quantile Critic (B, state_dim) -> (B, 4, num_quantiles)."""

    def __init__(
        self,
        state_dim: int = GLOBAL_STATE_DIM,
        hidden_dim: int = 512,
        num_agents: int = 4,
        num_quantiles: int = 32,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents
        self.num_quantiles = num_quantiles

        self.in_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
        )
        self.quantile_head = nn.Linear(hidden_dim // 2, num_agents * num_quantiles)

        # Register quantile midpoints tau_hat = (i + 0.5) / N
        taus = (torch.arange(num_quantiles, dtype=torch.float32) + 0.5) / float(num_quantiles)
        self.register_buffer("taus", taus)

        self._apply_orthogonal_init()

    def _apply_orthogonal_init(self):
        for layer in self.in_proj:
            init_weights(layer, gain=np.sqrt(2))
        for layer in self.res_block:
            init_weights(layer, gain=1.0)
        for layer in self.trunk:
            init_weights(layer, gain=np.sqrt(2))
        init_weights(self.quantile_head, gain=1.0)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """Returns estimated quantiles shape (..., num_agents, num_quantiles)."""
        h = self.in_proj(global_state)
        h = F.silu(h + self.res_block(h))
        feat = self.trunk(h)
        q = self.quantile_head(feat)
        lead_shape = global_state.shape[:-1]
        return q.view(*lead_shape, self.num_agents, self.num_quantiles)

    @staticmethod
    def risk_value_from_quantiles(
        quantiles: torch.Tensor, cvar_alpha: float = 0.25, cvar_weight: float = 0.30
    ) -> torch.Tensor:
        """Computes Mean + CVaR risk-adjusted scalar state value from quantile distribution."""
        mean_v = quantiles.mean(dim=-1)
        if cvar_weight <= 0.0:
            return mean_v
        n_q = quantiles.shape[-1]
        k = max(1, int(round(cvar_alpha * n_q)))
        sorted_q, _ = torch.sort(quantiles, dim=-1)
        cvar_v = sorted_q[..., :k].mean(dim=-1)
        return (1.0 - cvar_weight) * mean_v + cvar_weight * cvar_v

    def get_risk_value(
        self, global_state: torch.Tensor, cvar_alpha: float = 0.25, cvar_weight: float = 0.30
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (risk_adjusted_value [..., 4], full_quantiles [..., 4, N])."""
        quantiles = self.forward(global_state)
        risk_v = self.risk_value_from_quantiles(quantiles, cvar_alpha=cvar_alpha, cvar_weight=cvar_weight)
        return risk_v, quantiles
