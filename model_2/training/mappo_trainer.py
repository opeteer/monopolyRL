"""Centralized Training with Decentralized Execution (CTDE) MAPPO Trainer.

Implements clipped PPO policy objective, centralized value loss on S_t^{global},
multi-head entropy regularization, and orthogonal gradient optimization.
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model_2.models.actor_network import ActorNetwork
from model_2.models.critic_network import CentralizedCritic
from model_2.models.storage import RolloutBuffer


class MAPPOTrainer:
    """Model-2 MAPPO CTDE Trainer with Cosine LR support & KL Trust-Region Guard."""

    def __init__(
        self,
        actor: ActorNetwork,
        critic: CentralizedCritic,
        lr_actor: float = 2.0e-4,
        lr_critic: float = 4.0e-4,
        clip_param: float = 0.12,
        ppo_epochs: int = 4,
        num_mini_batches: int = 8,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.003,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = 0.020,
        weight_decay: float = 1e-5,
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
        self.target_kl = target_kl
        self.device = device

        self.optimizer_actor = optim.AdamW(
            self.actor.parameters(), lr=lr_actor, eps=1e-5, weight_decay=weight_decay
        )
        self.optimizer_critic = optim.AdamW(
            self.critic.parameters(), lr=lr_critic, eps=1e-5, weight_decay=weight_decay
        )

    def set_entropy_coef(self, new_coef: float):
        """Dynamically adjusts the entropy regularization coefficient."""
        self.entropy_coef = float(new_coef)

    def set_lr(self, lr_actor: float, lr_critic: float):
        """Dynamically adjusts Actor and Critic learning rates for cosine annealing."""
        for pg in self.optimizer_actor.param_groups:
            pg["lr"] = float(lr_actor)
        for pg in self.optimizer_critic.param_groups:
            pg["lr"] = float(lr_critic)

    def train_step(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Performs PPO CTDE update epoch across mini-batches with KL early stopping."""
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_approx_kl = 0.0
        num_updates = 0
        num_actor_updates = 0
        early_stopped = False

        for epoch in range(self.ppo_epochs):
            if early_stopped:
                break
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

                # ----------------- 1. ACTOR UPDATE (Active Decision Players Only) -----------------
                flat_active_mask = active_mask.view(-1)
                active_idx = (flat_active_mask > 0.5).nonzero(as_tuple=True)[0]

                if active_idx.numel() > 0:
                    flat_obs = obs.view(-1, self.actor.obs_dim)[active_idx]
                    flat_masks = {k: action_masks[k].view(-1, action_masks[k].shape[-1])[active_idx] for k in action_masks}
                    flat_actions = {k: actions[k].view(-1)[active_idx] for k in actions}
                    flat_old_log_probs = old_log_probs.view(-1)[active_idx]
                    flat_advantages = advantages.view(-1)[active_idx]

                    new_log_probs, entropy = self.actor.evaluate_actions(flat_obs, flat_masks, flat_actions)

                    # Importance sampling ratio r_t(theta)
                    log_ratio = new_log_probs - flat_old_log_probs
                    ratio = torch.exp(log_ratio)

                    # Schulman unbiased approx KL divergence
                    with torch.no_grad():
                        approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())

                    if self.target_kl is not None and epoch > 0 and approx_kl > 1.5 * self.target_kl:
                        early_stopped = True
                        break

                    # Clipped surrogate objective
                    surr1 = ratio * flat_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * flat_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    entropy_loss = -entropy.mean()

                    actor_loss = policy_loss + self.entropy_coef * entropy_loss

                    self.optimizer_actor.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.optimizer_actor.step()

                    total_policy_loss += policy_loss.item()
                    total_entropy_loss += entropy.mean().item()
                    total_approx_kl += approx_kl
                    num_actor_updates += 1

                # ----------------- 2. CENTRALIZED CRITIC UPDATE (Clipped Value Loss) -----------------
                predicted_values = self.critic(global_state)
                value_pred_clipped = old_values + torch.clamp(
                    predicted_values - old_values, -self.clip_param, self.clip_param
                )
                vf_loss_unclipped = (predicted_values - returns) ** 2
                vf_loss_clipped = (value_pred_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(vf_loss_unclipped, vf_loss_clipped).mean()

                self.optimizer_critic.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer_critic.step()

                total_value_loss += value_loss.item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / max(1, num_actor_updates),
            "value_loss": total_value_loss / max(1, num_updates),
            "entropy": total_entropy_loss / max(1, num_actor_updates),
            "approx_kl": total_approx_kl / max(1, num_actor_updates),
        }
