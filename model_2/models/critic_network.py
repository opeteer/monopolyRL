r"""Centralized Critic Network for MAPPO (CTDE).

Evaluates Centralized Global State S_t^{global} containing full multi-agent
observations and hidden card deck queue states to estimate value functions
V_\phi^i(S_t^{global}) for all 4 agents.
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_2.models.actor_network import init_weights


class CentralizedCritic(nn.Module):
    """Model-2 Residual Centralized Critic evaluating global Markov Game state."""

    def __init__(self, state_dim: int = 2032, hidden_dim: int = 640, num_agents: int = 4):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents

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

        # Output head predicts value for all 4 agents simultaneously
        self.value_head = nn.Linear(hidden_dim // 2, num_agents)

        self.apply_orthogonal_init()

    def apply_orthogonal_init(self):
        """Orthogonal initialization with gain sqrt(2) on trunk and 1.0 on value head."""
        for layer in self.in_proj:
            init_weights(layer, gain=np.sqrt(2))
        for layer in self.res_block:
            init_weights(layer, gain=1.0)
        for layer in self.trunk:
            init_weights(layer, gain=np.sqrt(2))
        init_weights(self.value_head, gain=1.0)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Input: global_state shape (batch_size, state_dim) or (state_dim,)
        Output: values shape (batch_size, 4) or (4,)
        """
        h = self.in_proj(global_state)
        h = F.silu(h + self.res_block(h))
        features = self.trunk(h)
        values = self.value_head(features)
        return values
