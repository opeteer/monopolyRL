"""Multi-Agent CTDE Generalized Advantage Estimation (GAE) Rollout Buffer.

Stores vectorized transitions, computes multi-agent GAE advantages and target values,
and samples mini-batches for centralized MAPPO updates.
"""

from typing import Dict, Generator, Tuple, Any, Optional
import numpy as np
import torch


class RolloutBuffer:
    """Multi-Agent GAE Rollout Buffer supporting CTDE."""

    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        obs_dim: int = 327,
        state_dim: int = 1340,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device = torch.device("cpu"),
    ):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.num_agents = 4

        # Action head dimensions
        self.action_keys = [
            "core", "auction", "property", "trade_target",
            "trade_offer_prop", "trade_offer_cash", "trade_req_prop",
            "trade_req_cash", "trade_response"
        ]
        self.mask_dims = {
            "core": 6,
            "auction": 7,
            "property": 161,
            "trade_target": 3,
            "trade_offer_prop": 41,
            "trade_offer_cash": 10,
            "trade_req_prop": 41,
            "trade_req_cash": 10,
            "trade_response": 2,
        }

        # Allocate buffer tensors
        self.obs = torch.zeros((buffer_size, num_envs, self.num_agents, obs_dim), dtype=torch.float32, device=device)
        self.global_states = torch.zeros((buffer_size, num_envs, state_dim), dtype=torch.float32, device=device)

        # Actions and masks
        self.actions = {
            k: torch.zeros((buffer_size, num_envs, self.num_agents), dtype=torch.long, device=device)
            for k in self.action_keys
        }
        self.action_masks = {
            k: torch.zeros((buffer_size, num_envs, self.num_agents, self.mask_dims[k]), dtype=torch.int8, device=device)
            for k in self.action_keys
        }

        self.rewards = torch.zeros((buffer_size, num_envs, self.num_agents), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size, num_envs, self.num_agents), dtype=torch.float32, device=device)
        self.values = torch.zeros((buffer_size, num_envs, self.num_agents), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((buffer_size, num_envs, self.num_agents), dtype=torch.float32, device=device)
        self.active_masks = torch.ones((buffer_size, num_envs, self.num_agents), dtype=torch.float32, device=device)

        # Computed GAE targets
        self.advantages = torch.zeros((buffer_size, num_envs, self.num_agents), dtype=torch.float32, device=device)
        self.returns = torch.zeros((buffer_size, num_envs, self.num_agents), dtype=torch.float32, device=device)

        self.step = 0
        self.full = False

    def insert(
        self,
        obs: torch.Tensor,
        global_state: torch.Tensor,
        actions: Dict[str, torch.Tensor],
        action_masks: Dict[str, torch.Tensor],
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        active_masks: Optional[torch.Tensor] = None,
    ):
        """Inserts a single environment transition into buffer."""
        self.obs[self.step] = obs
        self.global_states[self.step] = global_state
        for k in self.action_keys:
            self.actions[k][self.step] = actions[k]
            self.action_masks[k][self.step] = action_masks[k]

        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.values[self.step] = values
        self.log_probs[self.step] = log_probs
        if active_masks is not None:
            self.active_masks[self.step] = active_masks
        else:
            self.active_masks[self.step] = 1.0

        self.step = (self.step + 1) % self.buffer_size
        if self.step == 0:
            self.full = True

    def compute_gae(self, last_values: torch.Tensor, last_dones: torch.Tensor):
        """Computes Generalized Advantage Estimation (GAE) across all agents."""
        gae = torch.zeros((self.num_envs, self.num_agents), dtype=torch.float32, device=self.device)
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_dones
                next_val = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_val = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_val * next_non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def get_generator(self, num_mini_batches: int) -> Generator[Dict[str, Any], None, None]:
        """Yields randomized mini-batches for PPO updates."""
        # Flatten (buffer_size, num_envs) -> batch_size
        batch_size = self.buffer_size * self.num_envs
        mini_batch_size = batch_size // num_mini_batches

        # Flatten tensors
        flat_obs = self.obs.view(batch_size, self.num_agents, self.obs_dim)
        flat_global_states = self.global_states.view(batch_size, self.state_dim)
        flat_values = self.values.view(batch_size, self.num_agents)
        flat_log_probs = self.log_probs.view(batch_size, self.num_agents)
        flat_advantages = self.advantages.view(batch_size, self.num_agents)
        flat_returns = self.returns.view(batch_size, self.num_agents)
        flat_active_masks = self.active_masks.view(batch_size, self.num_agents)

        flat_actions = {k: self.actions[k].view(batch_size, self.num_agents) for k in self.action_keys}
        flat_masks = {k: self.action_masks[k].view(batch_size, self.num_agents, self.mask_dims[k]) for k in self.action_keys}

        # Normalize advantages per agent
        adv_mean = flat_advantages.mean(dim=0, keepdim=True)
        adv_std = flat_advantages.std(dim=0, keepdim=True) + 1e-8
        norm_advantages = (flat_advantages - adv_mean) / adv_std

        indices = torch.randperm(batch_size, device=self.device)

        for i in range(num_mini_batches):
            mb_idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]

            yield {
                "obs": flat_obs[mb_idx],
                "global_state": flat_global_states[mb_idx],
                "actions": {k: flat_actions[k][mb_idx] for k in self.action_keys},
                "action_masks": {k: flat_masks[k][mb_idx] for k in self.action_keys},
                "values": flat_values[mb_idx],
                "log_probs": flat_log_probs[mb_idx],
                "advantages": norm_advantages[mb_idx],
                "returns": flat_returns[mb_idx],
                "active_masks": flat_active_masks[mb_idx],
            }
