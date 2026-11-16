"""Centralized Training with Decentralized Execution (CTDE) MAPPO Trainer for Model-3.

Features:
- Dual-Objective Actor Loss: Clipped PPO Surrogate + Search Policy Distillation KL(pi_search || pi_theta)
- Multi-Head Exploration & Conditioning Guards: Entropy + Uniform-KL + Masked Logit-L2
- Distributional QR-Critic Training: Adaptive EMA-Scaled Quantile Huber Loss (kappa = 1.0)
- Live-Seat Masking (global_state bankruptcy slots) + Full Optimizer State Save/Resume
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model_3.engine.monopoly_core import (
    TILE_FEAT_DIM,
    SCALAR_OBS_DIM,
    EVENT_RING_LEN,
    EVENT_FEAT_DIM,
    MASK_SHAPES,
    ACT_SHAPES,
)
from model_3.models.actor_network import ActorNetworkV3
from model_3.models.critic_network import QuantileCritic
from model_3.models.storage import RolloutBufferV3
from model_3.search.chance_lookahead import SEARCH_ID_TO_HEAD_NAME

# Bankrupt flag slots inside global_state (4 players x 12 starting at offset 360, field +5)
GLOBAL_BANKRUPT_INDICES = (365, 377, 389, 401)


class MAPPOTrainerV3:
    """Model-3 MAPPO + Quantile Regression + Search Distillation Trainer."""

    def __init__(
        self,
        actor: ActorNetworkV3,
        critic: QuantileCritic,
        lr_actor: float = 2.0e-4,
        lr_critic: float = 4.0e-4,
        clip_param: float = 0.15,
        ppo_epochs: int = 4,
        num_mini_batches: int = 8,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.0035,
        uniform_kl_coef: float = 0.020,
        logit_l2_coef: float = 0.0015,
        search_distill_coef: float = 0.05,
        quantile_huber_kappa: float = 1.0,
        cvar_alpha: float = 0.25,
        cvar_weight: float = 0.30,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = 0.018,
        weight_decay: float = 1e-5,
        device: torch.device = torch.device("cpu"),
    ):
        self.actor = actor
        self.critic = critic
        self.clip_param = float(clip_param)
        self.ppo_epochs = int(ppo_epochs)
        self.num_mini_batches = int(num_mini_batches)
        self.value_loss_coef = float(value_loss_coef)
        self.entropy_coef = float(entropy_coef)
        self.uniform_kl_coef = float(uniform_kl_coef)
        self.logit_l2_coef = float(logit_l2_coef)
        self.search_distill_coef = float(search_distill_coef)
        self.quantile_huber_kappa = float(quantile_huber_kappa)
        self.cvar_alpha = float(cvar_alpha)
        self.cvar_weight = float(cvar_weight)
        self.max_grad_norm = float(max_grad_norm)
        self.target_kl = target_kl
        self.device = device

        self.ret_std = 1.0
        self._ret_std_initialized = False

        self.optimizer_actor = optim.AdamW(
            self.actor.parameters(), lr=lr_actor, eps=1e-5, weight_decay=weight_decay
        )
        self.optimizer_critic = optim.AdamW(
            self.critic.parameters(), lr=lr_critic, eps=1e-5, weight_decay=weight_decay
        )

    def set_entropy_coef(self, new_coef: float):
        self.entropy_coef = float(new_coef)

    def set_uniform_kl_coef(self, new_coef: float):
        self.uniform_kl_coef = float(new_coef)

    def set_search_distill_coef(self, new_coef: float):
        self.search_distill_coef = float(new_coef)

    def set_lr(self, lr_actor: float, lr_critic: float):
        for pg in self.optimizer_actor.param_groups:
            pg["lr"] = float(lr_actor)
        for pg in self.optimizer_critic.param_groups:
            pg["lr"] = float(lr_critic)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "optimizer_actor": self.optimizer_actor.state_dict(),
            "optimizer_critic": self.optimizer_critic.state_dict(),
            "ret_std": float(self.ret_std),
            "ret_std_initialized": bool(self._ret_std_initialized),
            "entropy_coef": float(self.entropy_coef),
            "uniform_kl_coef": float(self.uniform_kl_coef),
            "search_distill_coef": float(self.search_distill_coef),
        }

    def load_state_dict(self, state: Dict[str, Any]):
        if "optimizer_actor" in state:
            self.optimizer_actor.load_state_dict(state["optimizer_actor"])
        if "optimizer_critic" in state:
            self.optimizer_critic.load_state_dict(state["optimizer_critic"])
        self.ret_std = float(state.get("ret_std", self.ret_std))
        self._ret_std_initialized = bool(state.get("ret_std_initialized", self._ret_std_initialized))

    @staticmethod
    def _slice_obs_by_flat_idx(
        tile_state: torch.Tensor,
        scalar_obs: torch.Tensor,
        event_ring: torch.Tensor,
        action_masks: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        flat_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        sub_tile = tile_state.reshape(-1, 40, TILE_FEAT_DIM)[flat_idx]
        sub_scalar = scalar_obs.reshape(-1, SCALAR_OBS_DIM)[flat_idx]
        sub_ev = event_ring.reshape(-1, EVENT_RING_LEN, EVENT_FEAT_DIM)[flat_idx]
        sub_masks = {
            k: action_masks[k].reshape((-1,) + MASK_SHAPES[k])[flat_idx]
            for k in action_masks
        }
        sub_acts = {
            k: (
                actions[k].reshape((-1,) + ACT_SHAPES[k])[flat_idx]
                if len(ACT_SHAPES[k]) > 0
                else actions[k].reshape(-1)[flat_idx]
            )
            for k in actions
        }
        return sub_tile, sub_scalar, sub_ev, sub_masks, sub_acts

    def _compute_search_distill_loss(
        self,
        target_actor: ActorNetworkV3,
        tile_state: torch.Tensor,
        scalar_obs: torch.Tensor,
        event_ring: torch.Tensor,
        action_masks: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        distill_idx: torch.Tensor,
        flat_search_head: torch.Tensor,
        flat_search_probs: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if distill_idx.numel() == 0:
            return None
        d_tile, d_scalar, d_ev, d_masks, _ = self._slice_obs_by_flat_idx(
            tile_state, scalar_obs, event_ring, action_masks, actions, distill_idx
        )
        d_heads = flat_search_head[distill_idx]
        d_probs = flat_search_probs[distill_idx]

        logits_dict = target_actor.get_head_logits(d_tile, d_scalar, d_ev, d_masks)
        losses = []
        for h_id, h_name in SEARCH_ID_TO_HEAD_NAME.items():
            sub_mask = d_heads == h_id
            if not torch.any(sub_mask):
                continue
            l_h = logits_dict[h_name][sub_mask]
            dim_h = l_h.shape[-1]
            t_p = torch.where(l_h > -1e4, d_probs[sub_mask, :dim_h], torch.zeros_like(l_h))
            valid_row = t_p.sum(dim=-1) > 1e-4
            if not torch.any(valid_row):
                continue
            l_h = l_h[valid_row]
            t_p = t_p[valid_row]
            t_p = t_p / t_p.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            log_p = F.log_softmax(l_h, dim=-1)
            safe_terms = torch.where((t_p > 1e-12) & (l_h > -1e4), t_p * log_p, torch.zeros_like(log_p))
            ce = -safe_terms.sum(dim=-1)
            losses.append(ce)

        if not losses:
            return None
        return torch.cat(losses, dim=0).mean()

    def _ppo_actor_step(
        self,
        target_actor: ActorNetworkV3,
        target_optimizer: optim.Optimizer,
        tile_state: torch.Tensor,
        scalar_obs: torch.Tensor,
        event_ring: torch.Tensor,
        action_masks: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        epoch: int,
        search_valid: Optional[torch.Tensor] = None,
        search_head: Optional[torch.Tensor] = None,
        search_probs: Optional[torch.Tensor] = None,
        apply_search_distill: bool = True,
    ) -> Tuple[Optional[Tuple[float, float, float, float]], bool]:
        flat_mask = mask.reshape(-1)
        active_idx = (flat_mask > 0.5).nonzero(as_tuple=True)[0]

        distill_idx = None
        flat_search_head = None
        flat_search_probs = None
        if apply_search_distill and self.search_distill_coef > 0.0 and search_valid is not None and search_head is not None and search_probs is not None:
            flat_sv = search_valid.reshape(-1)
            d_idx = (flat_sv > 0.5).nonzero(as_tuple=True)[0]
            if d_idx.numel() > 0:
                distill_idx = d_idx
                flat_search_head = search_head.reshape(-1)
                flat_search_probs = search_probs.reshape(-1, search_probs.shape[-1])

        if active_idx.numel() == 0 and distill_idx is None:
            return None, False

        actor_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        p_loss_val = 0.0
        ent_val = 0.0
        kl_val = 0.0
        distill_val = 0.0

        if active_idx.numel() > 0:
            sub_tile, sub_scalar, sub_ev, sub_masks, sub_acts = self._slice_obs_by_flat_idx(
                tile_state, scalar_obs, event_ring, action_masks, actions, active_idx
            )
            sub_old_lp = old_log_probs.reshape(-1)[active_idx]
            sub_adv = advantages.reshape(-1)[active_idx]

            new_lp, entropy, uniform_kl, logit_l2, _ = target_actor.evaluate_actions_with_reg(
                sub_tile, sub_scalar, sub_ev, sub_masks, sub_acts
            )
            log_ratio = new_lp - sub_old_lp
            ratio = torch.exp(log_ratio)

            with torch.no_grad():
                kl_val = float(((ratio - 1.0) - log_ratio).mean().item())

            if self.target_kl is not None and epoch > 0 and kl_val > 1.5 * self.target_kl:
                return None, True

            surr1 = ratio * sub_adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * sub_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            mean_entropy = entropy.mean()
            actor_loss = (
                policy_loss
                - self.entropy_coef * mean_entropy
                + self.uniform_kl_coef * uniform_kl.mean()
                + self.logit_l2_coef * logit_l2.mean()
            )
            p_loss_val = float(policy_loss.item())
            ent_val = float(mean_entropy.item())

        if distill_idx is not None:
            d_loss = self._compute_search_distill_loss(
                target_actor,
                tile_state,
                scalar_obs,
                event_ring,
                action_masks,
                actions,
                distill_idx,
                flat_search_head,
                flat_search_probs,
            )
            if d_loss is not None:
                actor_loss = actor_loss + self.search_distill_coef * d_loss
                distill_val = float(d_loss.item())

        if not actor_loss.requires_grad:
            return None, False

        target_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(target_actor.parameters(), self.max_grad_norm)
        target_optimizer.step()

        return (p_loss_val, ent_val, kl_val, distill_val), False

    @staticmethod
    def quantile_huber_loss(
        pred_quantiles: torch.Tensor,
        target_quantiles: torch.Tensor,
        taus: torch.Tensor,
        scale: float,
        alive_mask: torch.Tensor,
        kappa: float = 1.0,
    ) -> torch.Tensor:
        """Computes adaptive-scaled Quantile Huber Loss across 4 centralized seats using native C++ kernels."""
        s = max(1.0, float(scale))
        p_q = (pred_quantiles.unsqueeze(3) / s).expand(-1, -1, -1, target_quantiles.shape[-1])
        t_q = (target_quantiles.unsqueeze(2) / s).expand(-1, -1, pred_quantiles.shape[-1], -1)
        diff = t_q - p_q
        huber = F.huber_loss(p_q, t_q, delta=kappa, reduction="none")
        tau_view = taus.view(1, 1, -1, 1).to(diff.dtype)
        pinball = torch.abs(tau_view - (diff < 0.0).to(diff.dtype))
        per_seat = (pinball * huber).mean(dim=(-1, -2))
        return (per_seat * alive_mask).sum() / alive_mask.sum().clamp(min=1.0)

    def train_step(
        self,
        buffer: RolloutBufferV3,
        exploiter_actor: Optional[ActorNetworkV3] = None,
        exploiter_optimizer: Optional[optim.Optimizer] = None,
    ) -> Dict[str, float]:
        """Performs CTDE PPO + QR-Critic + Search Distillation update across epochs."""
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_distill_loss = 0.0
        num_critic_updates = 0
        num_actor_updates = 0
        num_distill_updates = 0
        actor_early_stopped = False
        exploiter_early_stopped = False

        for epoch in range(self.ppo_epochs):
            for sample in buffer.get_generator(self.num_mini_batches):
                tile_state = sample["tile_state"]
                scalar_obs = sample["scalar_obs"]
                event_ring = sample["event_ring"]
                global_state = sample["global_state"]
                actions = sample["actions"]
                action_masks = sample["action_masks"]
                old_log_probs = sample["log_probs"]
                advantages = sample["advantages"]
                exp_advantages = sample.get("exploiter_advantages", advantages)
                returns = sample["returns"]
                quantile_targets = sample["quantile_targets"]
                active_mask = sample.get("active_masks", torch.ones_like(advantages))
                exploiter_mask = sample.get("exploiter_masks", None)
                search_valid = sample.get("search_valid", None)
                search_head = sample.get("search_head", None)
                search_probs = sample.get("search_probs", None)

                # 1. Main Actor update (PPO + Search Distillation)
                if not actor_early_stopped:
                    res, hit_kl = self._ppo_actor_step(
                        self.actor,
                        self.optimizer_actor,
                        tile_state,
                        scalar_obs,
                        event_ring,
                        action_masks,
                        actions,
                        old_log_probs,
                        advantages,
                        active_mask,
                        epoch,
                        search_valid=search_valid,
                        search_head=search_head,
                        search_probs=search_probs,
                        apply_search_distill=True,
                    )
                    if hit_kl:
                        actor_early_stopped = True
                    elif res is not None:
                        p_l, e_l, k_l, d_l = res
                        total_policy_loss += p_l
                        total_entropy += e_l
                        total_kl += k_l
                        num_actor_updates += 1
                        if d_l > 0.0:
                            total_distill_loss += d_l
                            num_distill_updates += 1

                # 1b. MainExploiter update (strictly on exploiter seats, no search distill mixing)
                if (
                    exploiter_actor is not None
                    and exploiter_optimizer is not None
                    and exploiter_mask is not None
                    and not exploiter_early_stopped
                ):
                    _, exp_hit_kl = self._ppo_actor_step(
                        exploiter_actor,
                        exploiter_optimizer,
                        tile_state,
                        scalar_obs,
                        event_ring,
                        action_masks,
                        actions,
                        old_log_probs,
                        exp_advantages,
                        exploiter_mask,
                        epoch,
                        apply_search_distill=False,
                    )
                    if exp_hit_kl:
                        exploiter_early_stopped = True

                # 2. Centralized Quantile Regression Critic update
                bankrupt_flags = global_state[:, list(GLOBAL_BANKRUPT_INDICES)]
                alive_mask = (1.0 - bankrupt_flags).clamp(0.0, 1.0)
                if not torch.any(alive_mask > 0.5):
                    alive_mask = torch.ones_like(returns)

                with torch.no_grad():
                    live_rets = returns[alive_mask > 0.5]
                    if live_rets.numel() > 1:
                        batch_std = float(live_rets.std(unbiased=False).clamp(min=1.0, max=100.0).item())
                        if not self._ret_std_initialized:
                            self.ret_std = batch_std
                            self._ret_std_initialized = True
                        else:
                            self.ret_std = 0.95 * self.ret_std + 0.05 * batch_std

                scale = max(1.0, float(self.ret_std))
                pred_quantiles = self.critic(global_state)
                q_loss = self.quantile_huber_loss(
                    pred_quantiles=pred_quantiles,
                    target_quantiles=quantile_targets,
                    taus=self.critic.taus,
                    scale=scale,
                    alive_mask=alive_mask,
                    kappa=self.quantile_huber_kappa,
                )
                critic_loss = self.value_loss_coef * q_loss

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer_critic.step()

                total_value_loss += float(critic_loss.item())
                num_critic_updates += 1

        return {
            "policy_loss": total_policy_loss / max(1, num_actor_updates),
            "value_loss": total_value_loss / max(1, num_critic_updates),
            "entropy": total_entropy / max(1, num_actor_updates),
            "approx_kl": total_kl / max(1, num_actor_updates),
            "distill_loss": total_distill_loss / max(1, num_distill_updates),
            "ret_std": float(self.ret_std),
        }
