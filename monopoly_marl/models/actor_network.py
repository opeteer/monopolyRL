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
from torch.distributions.categorical import Categorical


def init_weights(m: nn.Module, gain: float = np.sqrt(2)):
    """Applies uniform Orthogonal Initialization with specified gain."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class ActorNetwork(nn.Module):
    """Identical Blueprint Decentralized Multi-Head Actor Network."""

    def __init__(self, obs_dim: int = 327, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        # Shared feature extraction trunk
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
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
        self.head_trade_response = nn.Linear(hidden_dim, 2)

        # Apply strict Orthogonal Initialization
        self.apply_orthogonal_init()

    def apply_orthogonal_init(self):
        """Initializes all layers orthogonally with identical gain."""
        # Hidden trunk layers: gain = sqrt(2)
        for layer in self.trunk:
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

    def _mask_logits(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Applies binary mask, setting invalid logits to -1e8."""
        mask = mask.to(logits.device)
        return torch.where(mask > 0, logits, torch.tensor(-1e8, dtype=logits.dtype, device=logits.device))

    def forward(
        self,
        obs: torch.Tensor,
        action_masks: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Forward pass to sample actions from all factorized heads.

        Returns (actions_dict, total_log_prob, total_entropy).
        """
        features = self.trunk(obs)

        # Compute raw logits
        logits_core = self._mask_logits(self.head_core(features), action_masks["core"])
        logits_auc = self._mask_logits(self.head_auction(features), action_masks["auction"])
        logits_prop = self._mask_logits(self.head_property(features), action_masks["property"])
        logits_tr_target = self._mask_logits(self.head_trade_target(features), action_masks["trade_target"])
        logits_tr_off_prop = self._mask_logits(self.head_trade_offer_prop(features), action_masks["trade_offer_prop"])
        logits_tr_off_cash = self._mask_logits(self.head_trade_offer_cash(features), action_masks["trade_offer_cash"])
        logits_tr_req_prop = self._mask_logits(self.head_trade_req_prop(features), action_masks["trade_req_prop"])
        logits_tr_req_cash = self._mask_logits(self.head_trade_req_cash(features), action_masks["trade_req_cash"])
        logits_tr_resp = self._mask_logits(self.head_trade_response(features), action_masks["trade_response"])

        # Categorical distributions
        dist_core = Categorical(logits=logits_core)
        dist_auc = Categorical(logits=logits_auc)
        dist_prop = Categorical(logits=logits_prop)
        dist_tr_target = Categorical(logits=logits_tr_target)
        dist_tr_off_prop = Categorical(logits=logits_tr_off_prop)
        dist_tr_off_cash = Categorical(logits=logits_tr_off_cash)
        dist_tr_req_prop = Categorical(logits=logits_tr_req_prop)
        dist_tr_req_cash = Categorical(logits=logits_tr_req_cash)
        dist_tr_resp = Categorical(logits=logits_tr_resp)

        # Sample or take argmax
        if deterministic:
            act_core = torch.argmax(logits_core, dim=-1)
            act_auc = torch.argmax(logits_auc, dim=-1)
            act_prop = torch.argmax(logits_prop, dim=-1)
            act_tr_target = torch.argmax(logits_tr_target, dim=-1)
            act_tr_off_prop = torch.argmax(logits_tr_off_prop, dim=-1)
            act_tr_off_cash = torch.argmax(logits_tr_off_cash, dim=-1)
            act_tr_req_prop = torch.argmax(logits_tr_req_prop, dim=-1)
            act_tr_req_cash = torch.argmax(logits_tr_req_cash, dim=-1)
            act_tr_resp = torch.argmax(logits_tr_resp, dim=-1)
        else:
            act_core = dist_core.sample()
            act_auc = dist_auc.sample()
            act_prop = dist_prop.sample()
            act_tr_target = dist_tr_target.sample()
            act_tr_off_prop = dist_tr_off_prop.sample()
            act_tr_off_cash = dist_tr_off_cash.sample()
            act_tr_req_prop = dist_tr_req_prop.sample()
            act_tr_req_cash = dist_tr_req_cash.sample()
            act_tr_resp = dist_tr_resp.sample()

        # Compute log probabilities and entropies
        log_prob = (
            dist_core.log_prob(act_core) +
            dist_auc.log_prob(act_auc) +
            dist_prop.log_prob(act_prop) +
            dist_tr_target.log_prob(act_tr_target) +
            dist_tr_off_prop.log_prob(act_tr_off_prop) +
            dist_tr_off_cash.log_prob(act_tr_off_cash) +
            dist_tr_req_prop.log_prob(act_tr_req_prop) +
            dist_tr_req_cash.log_prob(act_tr_req_cash) +
            dist_tr_resp.log_prob(act_tr_resp)
        )

        entropy = (
            dist_core.entropy() +
            dist_auc.entropy() +
            dist_prop.entropy() +
            dist_tr_target.entropy() +
            dist_tr_off_prop.entropy() +
            dist_tr_off_cash.entropy() +
            dist_tr_req_prop.entropy() +
            dist_tr_req_cash.entropy() +
            dist_tr_resp.entropy()
        )

        actions_dict = {
            "core": act_core,
            "auction": act_auc,
            "property": act_prop,
            "trade_target": act_tr_target,
            "trade_offer_prop": act_tr_off_prop,
            "trade_offer_cash": act_tr_off_cash,
            "trade_req_prop": act_tr_req_prop,
            "trade_req_cash": act_tr_req_cash,
            "trade_response": act_tr_resp,
        }

        return actions_dict, log_prob, entropy

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action_masks: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates log probabilities and entropies for specific given actions."""
        features = self.trunk(obs)

        logits_core = self._mask_logits(self.head_core(features), action_masks["core"])
        logits_auc = self._mask_logits(self.head_auction(features), action_masks["auction"])
        logits_prop = self._mask_logits(self.head_property(features), action_masks["property"])
        logits_tr_target = self._mask_logits(self.head_trade_target(features), action_masks["trade_target"])
        logits_tr_off_prop = self._mask_logits(self.head_trade_offer_prop(features), action_masks["trade_offer_prop"])
        logits_tr_off_cash = self._mask_logits(self.head_trade_offer_cash(features), action_masks["trade_offer_cash"])
        logits_tr_req_prop = self._mask_logits(self.head_trade_req_prop(features), action_masks["trade_req_prop"])
        logits_tr_req_cash = self._mask_logits(self.head_trade_req_cash(features), action_masks["trade_req_cash"])
        logits_tr_resp = self._mask_logits(self.head_trade_response(features), action_masks["trade_response"])

        dist_core = Categorical(logits=logits_core)
        dist_auc = Categorical(logits=logits_auc)
        dist_prop = Categorical(logits=logits_prop)
        dist_tr_target = Categorical(logits=logits_tr_target)
        dist_tr_off_prop = Categorical(logits=logits_tr_off_prop)
        dist_tr_off_cash = Categorical(logits=logits_tr_off_cash)
        dist_tr_req_prop = Categorical(logits=logits_tr_req_prop)
        dist_tr_req_cash = Categorical(logits=logits_tr_req_cash)
        dist_tr_resp = Categorical(logits=logits_tr_resp)

        log_prob = (
            dist_core.log_prob(actions["core"]) +
            dist_auc.log_prob(actions["auction"]) +
            dist_prop.log_prob(actions["property"]) +
            dist_tr_target.log_prob(actions["trade_target"]) +
            dist_tr_off_prop.log_prob(actions["trade_offer_prop"]) +
            dist_tr_off_cash.log_prob(actions["trade_offer_cash"]) +
            dist_tr_req_prop.log_prob(actions["trade_req_prop"]) +
            dist_tr_req_cash.log_prob(actions["trade_req_cash"]) +
            dist_tr_resp.log_prob(actions["trade_response"])
        )

        entropy = (
            dist_core.entropy() +
            dist_auc.entropy() +
            dist_prop.entropy() +
            dist_tr_target.entropy() +
            dist_tr_off_prop.entropy() +
            dist_tr_off_cash.entropy() +
            dist_tr_req_prop.entropy() +
            dist_tr_req_cash.entropy() +
            dist_tr_resp.entropy()
        )

        return log_prob, entropy
