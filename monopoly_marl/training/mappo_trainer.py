"""Centralized Training with Decentralized Execution (CTDE) MAPPO Trainer.

Implements clipped PPO policy objective, centralized value loss on S_t^{global},
multi-head entropy regularization, and orthogonal gradient optimization.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from monopoly_marl.models.actor_network import ActorNetwork
from monopoly_marl.models.critic_network import CentralizedCritic
from monopoly_marl.models.storage import RolloutBuffer


class MAPPOTrainer:
    """MAPPO CTDE Trainer for 4-Agent Monopoly."""

    def __init__(
        self,
        actor: ActorNetwork,
        critic: CentralizedCritic,
        lr_actor: float = 3e-4,
        lr_critic: float = 5e-4,
        clip_param: float = 0.2,
        ppo_epochs: int = 5,
        num_mini_batches: int = 4,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.001,
        max_grad_norm: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ):
        self.actor = actor
        self.critic = critic
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-5)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)

    def set_entropy_coef(self, new_coef: float):
        """Dynamically adjusts the entropy regularization coefficient."""
        self.entropy_coef = float(new_coef)

    def train_step(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Performs PPO CTDE update epoch across mini-batches."""
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        num_updates = 0

        for epoch in range(self.ppo_epochs):
            data_generator = buffer.get_generator(self.num_mini_batches)

            for sample in data_generator:
                obs = sample["obs"]                          # (batch_size, 4, obs_dim)
                global_state = sample["global_state"]        # (batch_size, state_dim)
                actions = sample["actions"]                  # dict of (batch_size, 4)
                action_masks = sample["action_masks"]        # dict of (batch_size, 4, head_dim)
                old_values = sample["values"]                # (batch_size, 4)
                old_log_probs = sample["log_probs"]          # (batch_size, 4)
                advantages = sample["advantages"]            # (batch_size, 4)
                returns = sample["returns"]                  # (batch_size, 4)
                active_mask = sample.get("active_masks", torch.ones_like(advantages))
                flat_active_mask = active_mask.view(-1)
                active_count = flat_active_mask.sum().clamp(min=1.0)

                batch_size = obs.shape[0]

                # ----------------- 1. ACTOR UPDATE -----------------
                # Reshape for multi-agent evaluation: (batch_size * 4, ...)
                flat_obs = obs.view(-1, self.actor.obs_dim)
                flat_masks = {k: action_masks[k].view(-1, action_masks[k].shape[-1]) for k in action_masks}
                flat_actions = {k: actions[k].view(-1) for k in actions}
                flat_old_log_probs = old_log_probs.view(-1)
                flat_advantages = advantages.view(-1)

                new_log_probs, entropy = self.actor.evaluate_actions(flat_obs, flat_masks, flat_actions)

                # Importance sampling ratio r_t(theta)
                ratio = torch.exp(new_log_probs - flat_old_log_probs)

                # Clipped surrogate objective filtered by active_mask
                surr1 = ratio * flat_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * flat_advantages
                surr = torch.min(surr1, surr2) * flat_active_mask
                policy_loss = -(surr.sum() / active_count)

                entropy_loss = -((entropy * flat_active_mask).sum() / active_count)

                actor_loss = policy_loss + self.entropy_coef * entropy_loss

                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer_actor.step()

                # ----------------- 2. CENTRALIZED CRITIC UPDATE -----------------
                # Critic evaluates global_state: (batch_size, state_dim) -> (batch_size, 4)
                predicted_values = self.critic(global_state)

                # Active-agent weighted MSE value loss
                value_loss = 0.5 * (((predicted_values - returns) ** 2) * active_mask).sum() / active_mask.sum().clamp(min=1.0)

                self.optimizer_critic.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer_critic.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += (entropy * flat_active_mask).sum().item() / active_count.item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / max(1, num_updates),
            "value_loss": total_value_loss / max(1, num_updates),
            "entropy": total_entropy_loss / max(1, num_updates),
        }
