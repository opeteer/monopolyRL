"""Centralized Training with Decentralized Execution (CTDE) MAPPO Trainer.

Implements clipped PPO policy objective, centralized value loss on S_t^{global},
multi-head entropy regularization, and orthogonal gradient optimization.
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        clip_param: float = 0.15,
        value_clip_param: float = 8.0,
        ppo_epochs: int = 4,
        num_mini_batches: int = 8,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.0035,
        uniform_kl_coef: float = 0.020,
        logit_l2_coef: float = 0.0015,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = 0.018,
        weight_decay: float = 1e-5,
        device: torch.device = torch.device("cpu"),
    ):
        self.actor = actor
        self.critic = critic
        self.clip_param = clip_param
        self.value_clip_param = value_clip_param
        self.ppo_epochs = ppo_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.uniform_kl_coef = float(uniform_kl_coef)
        self.logit_l2_coef = float(logit_l2_coef)
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.device = device
        self.ret_std = 1.0

        self.optimizer_actor = optim.AdamW(
            self.actor.parameters(), lr=lr_actor, eps=1e-5, weight_decay=weight_decay
        )
        self.optimizer_critic = optim.AdamW(
            self.critic.parameters(), lr=lr_critic, eps=1e-5, weight_decay=weight_decay
        )

    def set_entropy_coef(self, new_coef: float):
        """Dynamically adjusts the entropy regularization coefficient."""
        self.entropy_coef = float(new_coef)

    def set_uniform_kl_coef(self, new_coef: float):
        """Dynamically adjusts the uniform-KL rare-phase exploration barrier coefficient."""
        self.uniform_kl_coef = float(new_coef)

    def set_lr(self, lr_actor: float, lr_critic: float):
        """Dynamically adjusts Actor and Critic learning rates for cosine annealing."""
        for pg in self.optimizer_actor.param_groups:
            pg["lr"] = float(lr_actor)
        for pg in self.optimizer_critic.param_groups:
            pg["lr"] = float(lr_critic)

    def _ppo_actor_step(
        self,
        target_actor: ActorNetwork,
        target_optimizer: optim.Optimizer,
        obs: torch.Tensor,
        action_masks: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        epoch: int,
    ) -> Tuple[Optional[Tuple[float, float, float]], bool]:
        """Executes one masked PPO actor step; returns ((policy_loss, entropy, approx_kl), hit_kl_limit)."""
        flat_mask = mask.reshape(-1)
        active_idx = (flat_mask > 0.5).nonzero(as_tuple=True)[0]
        if active_idx.numel() == 0:
            return None, False

        flat_obs = obs.reshape(-1, target_actor.obs_dim)[active_idx]
        flat_masks = {k: action_masks[k].reshape(-1, action_masks[k].shape[-1])[active_idx] for k in action_masks}
        flat_actions = {k: actions[k].reshape(-1)[active_idx] for k in actions}
        flat_old_log_probs = old_log_probs.reshape(-1)[active_idx]
        flat_advantages = advantages.reshape(-1)[active_idx]

        new_log_probs, entropy, uniform_kl, logit_l2 = target_actor.evaluate_actions_with_reg(
            flat_obs, flat_masks, flat_actions
        )

        log_ratio = new_log_probs - flat_old_log_probs
        ratio = torch.exp(log_ratio)

        with torch.no_grad():
            approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())

        if self.target_kl is not None and epoch > 0 and approx_kl > 1.5 * self.target_kl:
            return None, True

        surr1 = ratio * flat_advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * flat_advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        mean_entropy = entropy.mean()
        actor_loss = (
            policy_loss
            - self.entropy_coef * mean_entropy
            + self.uniform_kl_coef * uniform_kl.mean()
            + self.logit_l2_coef * logit_l2.mean()
        )

        target_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(target_actor.parameters(), self.max_grad_norm)
        target_optimizer.step()

        return (policy_loss.item(), mean_entropy.item(), approx_kl), False

    def train_step(
        self,
        buffer: RolloutBuffer,
        exploiter_actor: Optional[ActorNetwork] = None,
        exploiter_optimizer: Optional[optim.Optimizer] = None,
    ) -> Dict[str, float]:
        """Performs PPO CTDE update across mini-batches with decoupled critic updates & optional exploiter step."""
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_approx_kl = 0.0
        num_updates = 0
        num_actor_updates = 0
        actor_early_stopped = False
        exploiter_early_stopped = False

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
                exp_advantages = sample.get("exploiter_advantages", advantages)
                returns = sample["returns"]                  # (batch_size, 4)
                active_mask = sample.get("active_masks", torch.ones_like(advantages))
                exploiter_mask = sample.get("exploiter_masks", None)

                # ----------------- 1. MAIN ACTOR UPDATE (Active Decision Players Only) -----------------
                if not actor_early_stopped:
                    actor_res, hit_kl = self._ppo_actor_step(
                        self.actor, self.optimizer_actor,
                        obs, action_masks, actions, old_log_probs, advantages, active_mask, epoch,
                    )
                    if hit_kl:
                        actor_early_stopped = True
                    elif actor_res is not None:
                        p_loss, ent_val, approx_kl = actor_res
                        total_policy_loss += p_loss
                        total_entropy_loss += ent_val
                        total_approx_kl += approx_kl
                        num_actor_updates += 1

                # ----------------- 1b. EXPLOITER ACTOR UPDATE (League Exploiter Seats) -----------------
                if (
                    exploiter_actor is not None
                    and exploiter_optimizer is not None
                    and exploiter_mask is not None
                    and not exploiter_early_stopped
                ):
                    _, exp_hit_kl = self._ppo_actor_step(
                        exploiter_actor, exploiter_optimizer,
                        obs, action_masks, actions, old_log_probs, exp_advantages, exploiter_mask, epoch,
                    )
                    if exp_hit_kl:
                        exploiter_early_stopped = True

                # ----------------- 2. CENTRALIZED CRITIC UPDATE (Live-Masked Adaptive Huber + Calibrated Clip) -----------------
                predicted_values = self.critic(global_state)
                if obs.shape[-1] >= 24:
                    bankrupt_slice = obs[:, 0, 20:24]
                    if bool(((bankrupt_slice == 0.0) | (bankrupt_slice == 1.0)).all()):
                        alive_mask = (1.0 - bankrupt_slice).clamp(0.0, 1.0)
                        if not torch.any(alive_mask > 0.5):
                            alive_mask = torch.ones_like(returns)
                    else:
                        alive_mask = torch.ones_like(returns)
                else:
                    alive_mask = torch.ones_like(returns)

                with torch.no_grad():
                    live_rets = returns[alive_mask > 0.5]
                    if live_rets.numel() > 1:
                        batch_std = float(live_rets.std(unbiased=False).clamp(min=1.0, max=100.0).item())
                        self.ret_std = 0.95 * self.ret_std + 0.05 * batch_std

                scale = max(1.0, float(self.ret_std))
                eff_vclip = max(float(self.value_clip_param), 0.5 * scale)
                value_pred_clipped = old_values + torch.clamp(
                    predicted_values - old_values, -eff_vclip, eff_vclip
                )
                vf_loss_unclipped = F.huber_loss(predicted_values / scale, returns / scale, reduction="none", delta=2.0)
                vf_loss_clipped = F.huber_loss(value_pred_clipped / scale, returns / scale, reduction="none", delta=2.0)
                value_loss = (
                    0.5 * (torch.max(vf_loss_unclipped, vf_loss_clipped) * alive_mask).sum()
                    / alive_mask.sum().clamp(min=1.0)
                )

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
