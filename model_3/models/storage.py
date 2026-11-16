"""Multi-Agent Risk-Sensitive CTDE Rollout Buffer for Model-3 (CPU-RAM Optimized).

Stores:
- tile_state as float16 in CPU RAM (cast to float32 on mini-batch yield)
- scalar_obs (float32) & event_ring (int16)
- Scalar actions (int64) & 42-bit trade bundle vectors (int8)
- QR-Critic quantiles (N=32), 1-step distributional Bellman quantile targets, CVaR GAE
- Optional Search Distillation sparse targets (head index + target probability distribution)
"""

from typing import Dict, Generator, Any, Optional
import torch

from model_3.engine.monopoly_core import (
    TILE_FEAT_DIM,
    SCALAR_OBS_DIM,
    EVENT_RING_LEN,
    EVENT_FEAT_DIM,
    GLOBAL_STATE_DIM,
    ACTION_KEYS,
    MASK_SHAPES,
    ACT_SHAPES,
)
from model_3.models.critic_network import QuantileCritic


SEARCH_MAX_HEAD_DIM = 162  # Max discrete action space size among search-distilled heads (property=162)


class RolloutBufferV3:
    """CPU-RAM-friendly Rollout Buffer supporting 3-channel obs, QR targets, and Search Distillation."""

    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        num_quantiles: int = 32,
        gamma: float = 0.997,
        gae_lambda: float = 0.955,
        cvar_alpha: float = 0.25,
        cvar_weight: float = 0.30,
        device: torch.device = torch.device("cpu"),
    ):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.num_agents = 4
        self.num_quantiles = num_quantiles
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.cvar_alpha = cvar_alpha
        self.cvar_weight = cvar_weight
        self.device = device

        self.action_keys = list(ACTION_KEYS)
        self.mask_shapes = dict(MASK_SHAPES)
        self.act_shapes = dict(ACT_SHAPES)

        # Structured observation tensors (float16 tile_state cuts RAM ~2x with zero loss on normalized [0..2] values)
        self.tile_states = torch.zeros(
            (buffer_size, num_envs, 4, 40, TILE_FEAT_DIM), dtype=torch.float16, device=device
        )
        self.scalar_obs = torch.zeros(
            (buffer_size, num_envs, 4, SCALAR_OBS_DIM), dtype=torch.float32, device=device
        )
        self.event_rings = torch.zeros(
            (buffer_size, num_envs, 4, EVENT_RING_LEN, EVENT_FEAT_DIM), dtype=torch.int16, device=device
        )
        self.global_states = torch.zeros(
            (buffer_size, num_envs, GLOBAL_STATE_DIM), dtype=torch.float32, device=device
        )

        # Actions and masks
        self.actions: Dict[str, torch.Tensor] = {}
        self.action_masks: Dict[str, torch.Tensor] = {}
        for k in self.action_keys:
            if self.act_shapes[k] == ():
                self.actions[k] = torch.zeros((buffer_size, num_envs, 4), dtype=torch.long, device=device)
            else:
                self.actions[k] = torch.zeros(
                    (buffer_size, num_envs, 4) + self.act_shapes[k], dtype=torch.int8, device=device
                )
            self.action_masks[k] = torch.zeros(
                (buffer_size, num_envs, 4) + self.mask_shapes[k], dtype=torch.int8, device=device
            )

        self.rewards = torch.zeros((buffer_size, num_envs, 4), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size, num_envs, 4), dtype=torch.float32, device=device)
        self.values = torch.zeros((buffer_size, num_envs, 4), dtype=torch.float32, device=device)
        self.quantiles = torch.zeros((buffer_size, num_envs, 4, num_quantiles), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((buffer_size, num_envs, 4), dtype=torch.float32, device=device)
        self.active_masks = torch.ones((buffer_size, num_envs, 4), dtype=torch.float32, device=device)
        self.exploiter_masks = torch.zeros((buffer_size, num_envs, 4), dtype=torch.float32, device=device)

        # Risk-sensitive GAE & Quantile Bellman targets
        self.advantages = torch.zeros((buffer_size, num_envs, 4), dtype=torch.float32, device=device)
        self.returns = torch.zeros((buffer_size, num_envs, 4), dtype=torch.float32, device=device)
        self.quantile_targets = torch.zeros((buffer_size, num_envs, 4, num_quantiles), dtype=torch.float32, device=device)

        # Bounded Chance Lookahead Search Distillation targets (active actor only)
        self.search_valid = torch.zeros((buffer_size, num_envs, 4), dtype=torch.float32, device=device)
        self.search_head = torch.zeros((buffer_size, num_envs, 4), dtype=torch.long, device=device)
        self.search_probs = torch.zeros(
            (buffer_size, num_envs, 4, SEARCH_MAX_HEAD_DIM), dtype=torch.float32, device=device
        )

        self.step = 0
        self.full = False

    def insert_obs(
        self,
        tile_state: Any,
        scalar_obs: Any,
        event_ring: Any,
        global_state: Any,
        action_masks: Dict[str, Any],
    ) -> int:
        """Writes pre-step observations directly into buffer tensors without redundant intermediate copies."""
        idx = self.step
        self.tile_states[idx] = torch.as_tensor(tile_state, dtype=torch.float16, device=self.device)
        self.scalar_obs[idx] = torch.as_tensor(scalar_obs, dtype=torch.float32, device=self.device)
        self.event_rings[idx] = torch.as_tensor(event_ring, dtype=torch.int16, device=self.device)
        self.global_states[idx] = torch.as_tensor(global_state, dtype=torch.float32, device=self.device)
        for k in self.action_keys:
            self.action_masks[k][idx] = torch.as_tensor(action_masks[k], dtype=torch.float32, device=self.device)
        return idx

    def insert_step(
        self,
        idx: int,
        actions: Dict[str, Any],
        rewards: Any,
        dones: Any,
        values: Any,
        quantiles: Any,
        log_probs: Any,
        active_masks: Optional[Any] = None,
        exploiter_masks: Optional[Any] = None,
        search_valid: Optional[Any] = None,
        search_head: Optional[Any] = None,
        search_probs: Optional[Any] = None,
    ):
        """Writes transition outcomes and advantages for slot idx without copying observations."""
        for k in self.action_keys:
            a_dtype = torch.int8 if len(self.act_shapes[k]) > 0 else torch.int64
            self.actions[k][idx] = torch.as_tensor(actions[k], dtype=a_dtype, device=self.device)

        self.rewards[idx] = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        self.dones[idx] = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        self.values[idx] = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        self.quantiles[idx] = torch.as_tensor(quantiles, dtype=torch.float32, device=self.device)
        self.log_probs[idx] = torch.as_tensor(log_probs, dtype=torch.float32, device=self.device)
        if active_masks is not None:
            self.active_masks[idx] = torch.as_tensor(active_masks, dtype=torch.float32, device=self.device)
        else:
            self.active_masks[idx] = 1.0
        if exploiter_masks is not None:
            self.exploiter_masks[idx] = torch.as_tensor(exploiter_masks, dtype=torch.float32, device=self.device)
        else:
            self.exploiter_masks[idx].zero_()

        if search_valid is not None and search_head is not None and search_probs is not None:
            self.search_valid[idx] = torch.as_tensor(search_valid, dtype=torch.float32, device=self.device)
            self.search_head[idx] = torch.as_tensor(search_head, dtype=torch.int64, device=self.device)
            self.search_probs[idx] = torch.as_tensor(search_probs, dtype=torch.float32, device=self.device)
        else:
            self.search_valid[idx].zero_()

        self.step = (self.step + 1) % self.buffer_size
        if self.step == 0:
            self.full = True

    def insert(
        self,
        tile_state: Any,
        scalar_obs: Any,
        event_ring: Any,
        global_state: Any,
        actions: Dict[str, Any],
        action_masks: Dict[str, Any],
        rewards: Any,
        dones: Any,
        values: Any,
        quantiles: Any,
        log_probs: Any,
        active_masks: Optional[Any] = None,
        exploiter_masks: Optional[Any] = None,
        search_valid: Optional[Any] = None,
        search_head: Optional[Any] = None,
        search_probs: Optional[Any] = None,
    ):
        idx = self.step
        self.tile_states[idx] = torch.as_tensor(tile_state, dtype=torch.float16, device=self.device)
        self.scalar_obs[idx] = torch.as_tensor(scalar_obs, dtype=torch.float32, device=self.device)
        self.event_rings[idx] = torch.as_tensor(event_ring, dtype=torch.int16, device=self.device)
        self.global_states[idx] = torch.as_tensor(global_state, dtype=torch.float32, device=self.device)
        for k in self.action_keys:
            a_dtype = torch.int8 if len(self.act_shapes[k]) > 0 else torch.int64
            self.actions[k][idx] = torch.as_tensor(actions[k], dtype=a_dtype, device=self.device)
            self.action_masks[k][idx] = torch.as_tensor(action_masks[k], dtype=torch.float32, device=self.device)

        self.rewards[idx] = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        self.dones[idx] = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        self.values[idx] = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        self.quantiles[idx] = torch.as_tensor(quantiles, dtype=torch.float32, device=self.device)
        self.log_probs[idx] = torch.as_tensor(log_probs, dtype=torch.float32, device=self.device)
        if active_masks is not None:
            self.active_masks[idx] = torch.as_tensor(active_masks, dtype=torch.float32, device=self.device)
        else:
            self.active_masks[idx] = 1.0
        if exploiter_masks is not None:
            self.exploiter_masks[idx] = torch.as_tensor(exploiter_masks, dtype=torch.float32, device=self.device)
        else:
            self.exploiter_masks[idx].zero_()

        if search_valid is not None and search_head is not None and search_probs is not None:
            self.search_valid[idx] = torch.as_tensor(search_valid, dtype=torch.float32, device=self.device)
            self.search_head[idx] = torch.as_tensor(search_head, dtype=torch.int64, device=self.device)
            self.search_probs[idx] = torch.as_tensor(search_probs, dtype=torch.float32, device=self.device)
        else:
            self.search_valid[idx].zero_()

        self.step = (self.step + 1) % self.buffer_size
        if self.step == 0:
            self.full = True

    def compute_gae_and_quantile_targets(self, last_quantiles: Any, last_dones: Any):
        """Computes CVaR risk-sensitive GAE and multi-step Distributional TD(lambda) quantile targets."""
        last_q_t = torch.as_tensor(last_quantiles, dtype=torch.float32, device=self.device)
        last_d_t = torch.as_tensor(last_dones, dtype=torch.float32, device=self.device)
        last_values = QuantileCritic.risk_value_from_quantiles(
            last_q_t, cvar_alpha=self.cvar_alpha, cvar_weight=self.cvar_weight
        )
        gae = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        q_lambda = last_q_t
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_val = last_values
                next_non_terminal = 1.0 - last_d_t
                q_next_blend = last_q_t
            else:
                next_val = self.values[t + 1]
                next_q = self.quantiles[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
                q_next_blend = (1.0 - self.gae_lambda) * next_q + self.gae_lambda * q_lambda

            delta = self.rewards[t] + self.gamma * next_val * next_non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

            # Exact multi-step Distributional TD(lambda) target: propagates terminal outcomes rapidly
            q_lambda = self.rewards[t].unsqueeze(-1) + self.gamma * next_non_terminal.unsqueeze(-1) * q_next_blend
            self.quantile_targets[t] = q_lambda

    @staticmethod
    def _normalize_masked_advantages(flat_advantages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_bool = mask > 0.5
        if torch.any(mask_bool):
            vals = flat_advantages[mask_bool]
            if vals.numel() <= 1:
                return (
                    torch.zeros_like(flat_advantages)
                    if vals.numel() == 0
                    else (flat_advantages - vals.mean()).clamp(-10.0, 10.0)
                )
            return (flat_advantages - vals.mean()) / (vals.std(unbiased=False).clamp(min=1e-6) + 1e-8)
        if flat_advantages.numel() <= 1:
            return torch.zeros_like(flat_advantages)
        return (flat_advantages - flat_advantages.mean()) / (flat_advantages.std(unbiased=False).clamp(min=1e-6) + 1e-8)

    def get_generator(self, num_mini_batches: int) -> Generator[Dict[str, Any], None, None]:
        batch_size = self.buffer_size * self.num_envs
        mini_batch_size = max(1, batch_size // max(1, num_mini_batches))

        flat_tiles = self.tile_states.view(batch_size, 4, 40, TILE_FEAT_DIM)
        flat_scalars = self.scalar_obs.view(batch_size, 4, SCALAR_OBS_DIM)
        flat_events = self.event_rings.view(batch_size, 4, EVENT_RING_LEN, EVENT_FEAT_DIM)
        flat_states = self.global_states.view(batch_size, GLOBAL_STATE_DIM)
        flat_values = self.values.view(batch_size, 4)
        flat_q_targets = self.quantile_targets.view(batch_size, 4, self.num_quantiles)
        flat_log_probs = self.log_probs.view(batch_size, 4)
        flat_advantages = self.advantages.view(batch_size, 4)
        flat_returns = self.returns.view(batch_size, 4)
        flat_active = self.active_masks.view(batch_size, 4)
        flat_exploiter = self.exploiter_masks.view(batch_size, 4)

        flat_search_valid = self.search_valid.view(batch_size, 4)
        flat_search_head = self.search_head.view(batch_size, 4)
        flat_search_probs = self.search_probs.view(batch_size, 4, SEARCH_MAX_HEAD_DIM)

        flat_actions = {
            k: self.actions[k].view((batch_size, 4) + self.act_shapes[k])
            for k in self.action_keys
        }
        flat_masks = {
            k: self.action_masks[k].view((batch_size, 4) + self.mask_shapes[k])
            for k in self.action_keys
        }

        norm_adv = self._normalize_masked_advantages(flat_advantages, flat_active)
        norm_exp_adv = (
            self._normalize_masked_advantages(flat_advantages, flat_exploiter)
            if torch.any(flat_exploiter > 0.5)
            else flat_advantages
        )

        indices = torch.randperm(batch_size, device=self.device)
        for i in range(num_mini_batches):
            mb_idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
            if mb_idx.numel() == 0:
                continue
            yield {
                "tile_state": flat_tiles[mb_idx].to(torch.float32),
                "scalar_obs": flat_scalars[mb_idx],
                "event_ring": flat_events[mb_idx],
                "global_state": flat_states[mb_idx],
                "actions": {k: flat_actions[k][mb_idx] for k in self.action_keys},
                "action_masks": {k: flat_masks[k][mb_idx] for k in self.action_keys},
                "values": flat_values[mb_idx],
                "quantile_targets": flat_q_targets[mb_idx],
                "log_probs": flat_log_probs[mb_idx],
                "advantages": norm_adv[mb_idx],
                "exploiter_advantages": norm_exp_adv[mb_idx],
                "returns": flat_returns[mb_idx],
                "active_masks": flat_active[mb_idx],
                "exploiter_masks": flat_exploiter[mb_idx],
                "search_valid": flat_search_valid[mb_idx],
                "search_head": flat_search_head[mb_idx],
                "search_probs": flat_search_probs[mb_idx],
            }
