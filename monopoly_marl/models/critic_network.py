r"""Centralized Critic Network for MAPPO (CTDE).

Evaluates Centralized Global State S_t^{global} containing full multi-agent
observations and hidden card deck queue states to estimate value functions
V_\phi^i(S_t^{global}) for all 4 agents.
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

from monopoly_marl.models.actor_network import init_weights


class CentralizedCritic(nn.Module):
    """Centralized Critic Network evaluating global Markov Game state."""

    def __init__(self, state_dim: int = 1340, hidden_dim: int = 512, num_agents: int = 4):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )

        # Output head predicts value for all 4 agents simultaneously
        self.value_head = nn.Linear(hidden_dim // 2, num_agents)

        self.apply_orthogonal_init()

    def apply_orthogonal_init(self):
        """Orthogonal initialization with gain sqrt(2) on trunk and 1.0 on value head."""
        for layer in self.trunk:
            init_weights(layer, gain=np.sqrt(2))
        init_weights(self.value_head, gain=1.0)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Input: global_state shape (batch_size, 1340) or (1340,)
        Output: values shape (batch_size, 4) or (4,)
        """
        features = self.trunk(global_state)
        values = self.value_head(features)
        return values
