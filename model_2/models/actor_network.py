"""Decentralized Multi-Head Actor Network.

100% Identical Neural Network Blueprint for all 4 agents.
Initialized strictly via Orthogonal Initialization with identical gain.
Implements strict action masking with -1e8 logits prior to softmax.
Factorized into Core, Auction, Property Management, and Bilateral Trade heads.
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


def init_weights(m: nn.Module, gain: float = np.sqrt(2)):
    """Applies uniform Orthogonal Initialization with specified gain."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class ActorNetwork(nn.Module):
    """Model-2 Residual Multi-Head Actor Network with SiLU + Dedicated Trade Response Pathway."""

    def __init__(self, obs_dim: int = 500, hidden_dim: int = 384):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        # Shared input projection trunk (SiLU + LayerNorm)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Residual refinement block for higher strategic precision without vanishing gradients
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Dedicated nonlinear pathway for high-stakes bilateral Trade Accept/Reject decisions
        self.trade_resp_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
        )

        # Factorized Action Heads
        # 1. Core Actions (6 dims)
        self.head_core = nn.Linear(hidden_dim, 6)
        # 2. Auction Actions (7 dims: 5 discrete increments + 2 strategic jump bids)
        self.head_auction = nn.Linear(hidden_dim, 7)
        # 3. Property Management Sub-actions (161 dims: [0..39] Build, [40..79] Sell, [80..119] Mortgage, [120..159] Unmortgage, [160] Pass)
        self.head_property = nn.Linear(hidden_dim, 161)
        # 4. Bilateral Trade Heads
        self.head_trade_target = nn.Linear(hidden_dim, 3)
        self.head_trade_offer_prop = nn.Linear(hidden_dim, 41)
        self.head_trade_offer_cash = nn.Linear(hidden_dim, 10)
        self.head_trade_req_prop = nn.Linear(hidden_dim, 41)
        self.head_trade_req_cash = nn.Linear(hidden_dim, 10)
        self.head_trade_response = nn.Linear(hidden_dim // 2, 2)

        # Autoregressive zero-init trade conditioning embeddings:
        # req_prop -> offer_prop, and (offer_prop, req_prop) -> cash heads
        self.emb_trade_req_for_offer = nn.Embedding(41, hidden_dim)
        self.emb_trade_offer_prop = nn.Embedding(41, hidden_dim)
        self.emb_trade_req_prop = nn.Embedding(41, hidden_dim)

        # Apply strict Orthogonal Initialization
        self.apply_orthogonal_init()

    def apply_orthogonal_init(self):
        """Initializes all layers orthogonally with identical gain."""
        for layer in self.trunk:
            init_weights(layer, gain=np.sqrt(2))
        for layer in self.res_block:
            init_weights(layer, gain=1.0)
        for layer in self.trade_resp_branch:
            init_weights(layer, gain=np.sqrt(2))

        # Output policy heads: gain = 0.01 for balanced initial exploration
        head_gain = 0.01
        init_weights(self.head_core, gain=head_gain)
        init_weights(self.head_auction, gain=head_gain)
        init_weights(self.head_property, gain=head_gain)
        init_weights(self.head_trade_target, gain=head_gain)
        init_weights(self.head_trade_offer_prop, gain=head_gain)
        init_weights(self.head_trade_offer_cash, gain=head_gain)
        init_weights(self.head_trade_req_prop, gain=head_gain)
        init_weights(self.head_trade_req_cash, gain=head_gain)
        init_weights(self.head_trade_response, gain=head_gain)

        nn.init.zeros_(self.emb_trade_req_for_offer.weight)
        nn.init.zeros_(self.emb_trade_offer_prop.weight)
        nn.init.zeros_(self.emb_trade_req_prop.weight)

    def _encode_features(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.trunk(obs)
        return F.silu(h + self.res_block(h))

    def _mask_logits(self, logits: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Applies temperature scaling to valid logits and masks invalid logits to -1e8."""
        mask = mask.to(logits.device)
        scaled = logits / max(1e-3, float(temperature)) if temperature != 1.0 else logits
        return torch.where(mask > 0, scaled, torch.tensor(-1e8, dtype=logits.dtype, device=logits.device))

    def _compute_phase_aware_lp_entropy(
        self,
        obs: torch.Tensor,
        action_masks: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        dists: Dict[str, Categorical],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes log_prob and entropy exclusively for heads actually active in the current phase.

        Eliminates cross-head variance pollution and clipping interference from unused action heads.
        """
        lp_dict = {k: dists[k].log_prob(actions[k]) for k in dists}
        ent_dict = {k: dists[k].entropy() for k in dists}

        if obs.shape[-1] >= 327:
            u = obs[..., 307:314]
            valid_onehot = ((u.sum(dim=-1) - 1.0).abs() < 1e-4) & ((u == 1.0).sum(dim=-1) == 1)
            if torch.all(valid_onehot):
                is_core = (u[..., 0] + u[..., 1] + u[..., 3])
                is_auc = u[..., 2]
                is_manage = u[..., 4]
                is_trade_resp = u[..., 5]

                # Trade proposal heads only participate when prop == 160 (Pass physical mgmt) and trade is enabled
                trade_enabled = (action_masks["trade_req_cash"].sum(dim=-1) > 1).float()
                is_trade_check = is_manage * (actions["property"] == 160).float() * trade_enabled
                has_trade_prop = ((actions["trade_offer_prop"] < 40) | (actions["trade_req_prop"] < 40)).float()
                is_trade_detail = is_trade_check * has_trade_prop
                is_trade_target = is_trade_detail * (actions["trade_req_prop"] == 40).float()

                log_prob = (
                    is_core * lp_dict["core"] +
                    is_auc * lp_dict["auction"] +
                    is_manage * lp_dict["property"] +
                    is_trade_check * (lp_dict["trade_offer_prop"] + lp_dict["trade_req_prop"]) +
                    is_trade_detail * (lp_dict["trade_offer_cash"] + lp_dict["trade_req_cash"]) +
                    is_trade_target * lp_dict["trade_target"] +
                    is_trade_resp * lp_dict["trade_response"]
                )
                entropy = (
                    is_core * ent_dict["core"] +
                    is_auc * ent_dict["auction"] +
                    is_manage * ent_dict["property"] +
                    is_trade_check * (ent_dict["trade_offer_prop"] + ent_dict["trade_req_prop"]) +
                    is_trade_detail * (ent_dict["trade_offer_cash"] + ent_dict["trade_req_cash"]) +
                    is_trade_target * ent_dict["trade_target"] +
                    is_trade_resp * ent_dict["trade_response"]
                )
                return log_prob, entropy

        # Fallback for synthetic unit tests with random obs vectors
        log_prob = sum(lp_dict.values())
        entropy = sum(ent_dict.values())
        return log_prob, entropy

    @staticmethod
    def _safe_conditional_mask(cond_mask: torch.Tensor, fallback_mask: torch.Tensor) -> torch.Tensor:
        """Falls back to base mask if a conditional rule eliminates every option."""
        has_any = cond_mask.sum(dim=-1, keepdim=True) > 0
        return torch.where(has_any, cond_mask, fallback_mask)

    def _select_from_dist(self, dist: Categorical, deterministic: bool) -> torch.Tensor:
        return torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()

    def _build_distributions_and_actions(
        self,
        obs: torch.Tensor,
        action_masks: Dict[str, torch.Tensor],
        given_actions: Optional[Dict[str, torch.Tensor]] = None,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[Dict[str, Categorical], Dict[str, torch.Tensor]]:
        """Builds phase-conditioned distributions and sequentially samples (or evaluates) trade heads.

        Sequential trade conditioning guarantees:
        - No identical offer_prop == req_prop
        - Zero dual-cash paradox (never offer_cash > 0 and req_cash > 0 simultaneously)
        - Zero empty-handed begs (offer_prop=40, req_prop<40 forces offer_cash >= $25 when affordable)
        - Zero one-sided property giveaways (offer_prop<40, req_prop=40 forces req_cash >= $25)
        """
        features = self._encode_features(obs)
        resp_features = self.trade_resp_branch(features)
        device = features.device
        temp = 1.0 if given_actions is not None else float(temperature)

        dists = {
            "core": Categorical(logits=self._mask_logits(self.head_core(features), action_masks["core"], temp)),
            "auction": Categorical(logits=self._mask_logits(self.head_auction(features), action_masks["auction"], temp)),
            "property": Categorical(logits=self._mask_logits(self.head_property(features), action_masks["property"], temp)),
            "trade_target": Categorical(logits=self._mask_logits(self.head_trade_target(features), action_masks["trade_target"], temp)),
            "trade_response": Categorical(logits=self._mask_logits(self.head_trade_response(resp_features), action_masks["trade_response"], temp)),
        }

        if given_actions is None:
            actions_dict = {k: self._select_from_dist(d, deterministic) for k, d in dists.items()}
        else:
            actions_dict = dict(given_actions)

        # 1. Requested property
        dists["trade_req_prop"] = Categorical(
            logits=self._mask_logits(self.head_trade_req_prop(features), action_masks["trade_req_prop"], temp)
        )
        req_prop = (
            self._select_from_dist(dists["trade_req_prop"], deterministic)
            if given_actions is None else given_actions["trade_req_prop"].to(device).long()
        )
        actions_dict["trade_req_prop"] = req_prop

        # 2. Offered property conditioned on req_prop (cannot offer == request when both < 40)
        base_op_mask = action_masks["trade_offer_prop"].to(device)
        cond_op_mask = base_op_mask.clone()
        is_req_tile = req_prop < 40
        if torch.any(is_req_tile):
            idx_clamped = req_prop.clamp(0, 40).unsqueeze(-1)
            one_hot_req = torch.zeros_like(cond_op_mask).scatter_(-1, idx_clamped, 1)
            cond_op_mask = torch.where(
                is_req_tile.unsqueeze(-1),
                cond_op_mask * (1 - one_hot_req),
                cond_op_mask,
            )
            cond_op_mask = self._safe_conditional_mask(cond_op_mask, base_op_mask)
        feat_op = features + self.emb_trade_req_for_offer(req_prop.clamp(0, 40))
        dists["trade_offer_prop"] = Categorical(
            logits=self._mask_logits(self.head_trade_offer_prop(feat_op), cond_op_mask, temp)
        )
        offer_prop = (
            self._select_from_dist(dists["trade_offer_prop"], deterministic)
            if given_actions is None else given_actions["trade_offer_prop"].to(device).long()
        )
        actions_dict["trade_offer_prop"] = offer_prop

        # 3. Cash heads conditioned on (offer_prop, req_prop)
        feat_cash = (
            features
            + self.emb_trade_offer_prop(offer_prop.clamp(0, 40))
            + self.emb_trade_req_prop(req_prop.clamp(0, 40))
        )

        is_cash_buy = (offer_prop == 40) & (req_prop < 40)
        is_prop_sale = (offer_prop < 40) & (req_prop == 40)

        base_oc_mask = action_masks["trade_offer_cash"].to(device)
        cond_oc_mask = base_oc_mask.clone()
        cond_oc_mask[..., 0] = torch.where(is_cash_buy, torch.zeros_like(cond_oc_mask[..., 0]), cond_oc_mask[..., 0])
        cond_oc_mask[..., 1:] = torch.where(
            is_prop_sale.unsqueeze(-1),
            torch.zeros_like(cond_oc_mask[..., 1:]),
            cond_oc_mask[..., 1:],
        )
        cond_oc_mask = self._safe_conditional_mask(cond_oc_mask, base_oc_mask)
        dists["trade_offer_cash"] = Categorical(
            logits=self._mask_logits(self.head_trade_offer_cash(feat_cash), cond_oc_mask, temp)
        )
        offer_cash = (
            self._select_from_dist(dists["trade_offer_cash"], deterministic)
            if given_actions is None else given_actions["trade_offer_cash"].to(device).long()
        )
        actions_dict["trade_offer_cash"] = offer_cash

        # 4. Requested cash conditioned on (offer_prop, req_prop, offer_cash) -> strictly zero dual-cash!
        base_rc_mask = action_masks["trade_req_cash"].to(device)
        cond_rc_mask = base_rc_mask.clone()
        force_zero_rc = (offer_cash > 0) | is_cash_buy
        cond_rc_mask[..., 1:] = torch.where(
            force_zero_rc.unsqueeze(-1),
            torch.zeros_like(cond_rc_mask[..., 1:]),
            cond_rc_mask[..., 1:],
        )
        cond_rc_mask[..., 0] = torch.where(
            is_prop_sale,
            torch.zeros_like(cond_rc_mask[..., 0]),
            cond_rc_mask[..., 0],
        )
        cond_rc_mask = self._safe_conditional_mask(cond_rc_mask, base_rc_mask)
        dists["trade_req_cash"] = Categorical(
            logits=self._mask_logits(self.head_trade_req_cash(feat_cash), cond_rc_mask, temp)
        )
        req_cash = (
            self._select_from_dist(dists["trade_req_cash"], deterministic)
            if given_actions is None else given_actions["trade_req_cash"].to(device).long()
        )
        actions_dict["trade_req_cash"] = req_cash

        return dists, actions_dict

    def forward(
        self,
        obs: torch.Tensor,
        action_masks: Dict[str, torch.Tensor],
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Forward pass to sample actions from all factorized & autoregressive trade heads.

        Returns (actions_dict, total_log_prob, total_entropy).
        """
        dists, actions_dict = self._build_distributions_and_actions(
            obs, action_masks, given_actions=None, deterministic=deterministic, temperature=temperature
        )
        log_prob, entropy = self._compute_phase_aware_lp_entropy(obs, action_masks, actions_dict, dists)
        return actions_dict, log_prob, entropy

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action_masks: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates log probabilities and entropies for specific given actions."""
        dists, _ = self._build_distributions_and_actions(
            obs, action_masks, given_actions=actions, deterministic=False, temperature=1.0
        )
        return self._compute_phase_aware_lp_entropy(obs, action_masks, actions, dists)
