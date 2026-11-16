"""Model-3 CPU-Optimized Multi-Head Actor Network.

Features:
- 3-Channel Encoder:
  * TileMixer (2-layer spatial-channel MLP-Mixer over 40 board tiles, ~10x faster on CPU than 2L Transformer)
    or optional LightTx (1-layer TransformerEncoder) selectable via arch="tile_mixer" | "light_tx".
  * EventMixer (temporal MLP-Mixer over the 24-event ring buffer, eliminating slow CPU GRU loops).
  * Scalar Telemetry projection + Residual Fusion Trunk.
- Dedicated Trade Response pathway pooling H_tiles over pending offered vs requested bundles (3-way: Reject/Accept/Counter).
- Target-Conditioned Multi-Bernoulli Set Heads for UNCONSTRAINED trade bundles (0..28 properties + 0..2 GOOJF)
  with Permutation-Invariant Set Sum Pooling (z_req, z_offer) and zero permutation ambiguity.
- Phase-aware exact log-probs, zero-validate_args fast distributions, Uniform-KL & masked logit L2 regularization.
"""

from typing import Dict, Tuple, Optional, Any
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_3.engine.monopoly_core import (
    TILE_FEAT_DIM,
    SCALAR_OBS_DIM,
    EVENT_RING_LEN,
    EVENT_FEAT_DIM,
    BUNDLE_DIM,
    TRADE_CASH_DIM,
    ACT_PROP_PROPOSE_TRADE,
    NUM_PHASES,
)


def init_weights(m: nn.Module, gain: float = np.sqrt(2)):
    """Orthogonal initialization for Linear/Conv layers."""
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class _TileMixerBlock(nn.Module):
    """Spatial-Channel MLP-Mixer block across fixed 40 board tiles (pure dense BLAS GEMMs)."""

    def __init__(self, d_model: int, num_tiles: int = 40):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.token_mix = nn.Sequential(
            nn.Linear(num_tiles, num_tiles),
            nn.SiLU(),
            nn.Linear(num_tiles, num_tiles),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.channel_mix = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        for layer in self.token_mix:
            init_weights(layer, gain=1.0)
        for layer in self.channel_mix:
            init_weights(layer, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 40, d_model)
        h = self.norm1(x).transpose(1, 2)
        x = x + self.token_mix(h).transpose(1, 2)
        x = x + self.channel_mix(self.norm2(x))
        return x


class _EventMixerEncoder(nn.Module):
    """Encodes (B, 24, 5) int16 event ring via factorized embeddings + temporal mixer."""

    def __init__(self, d_ev: int = 48, seq_len: int = EVENT_RING_LEN):
        super().__init__()
        self.emb_type = nn.Embedding(20, 16)
        self.emb_actor = nn.Embedding(6, 8)
        self.emb_target = nn.Embedding(6, 8)
        self.emb_tile = nn.Embedding(48, 16)
        self.emb_amt = nn.Embedding(20, 8)
        in_dim = 16 + 8 + 8 + 16 + 8  # 56

        self.proj = nn.Sequential(nn.Linear(in_dim, d_ev), nn.SiLU())
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_ev))
        self.norm = nn.LayerNorm(d_ev)
        self.time_mix = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.SiLU(),
            nn.Linear(seq_len, 4),
        )
        self.out_dim = d_ev * 4
        init_weights(self.proj[0], gain=np.sqrt(2))
        for layer in self.time_mix:
            init_weights(layer, gain=1.0)

    def forward(self, event_ring: torch.Tensor) -> torch.Tensor:
        ev = event_ring.long()
        emb = torch.cat([
            self.emb_type(ev[..., 0].clamp(0, 19)),
            self.emb_actor(ev[..., 1].clamp(0, 5)),
            self.emb_target(ev[..., 2].clamp(0, 5)),
            self.emb_tile(ev[..., 3].clamp(0, 47)),
            self.emb_amt(ev[..., 4].clamp(0, 19)),
        ], dim=-1)
        h = self.proj(emb) + self.pos_emb
        h = self.norm(h)
        # Project temporal dim 24 -> 4 summary slots: (B, d_ev, 4) -> (B, 4*d_ev)
        pooled = self.time_mix(h.transpose(1, 2))
        return pooled.reshape(event_ring.shape[0], -1)


class ActorNetworkV3(nn.Module):
    """Model-3 Structured CPU-Friendly Actor with Unconstrained Multi-Bernoulli Trade Bundles."""

    def __init__(
        self,
        tile_dim: int = TILE_FEAT_DIM,
        scalar_dim: int = SCALAR_OBS_DIM,
        d_model: int = 64,
        hidden_dim: int = 384,
        trade_cash_dim: int = TRADE_CASH_DIM,
        arch: str = "tile_mixer",
    ):
        super().__init__()
        self.tile_dim = tile_dim
        self.scalar_dim = scalar_dim
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.trade_cash_dim = trade_cash_dim
        self.arch = arch

        # 1. Board Encoder (40 tiles -> per-tile H_tiles (B, 40, d_model) + pooled z_board)
        self.tile_in = nn.Sequential(nn.Linear(tile_dim, d_model), nn.SiLU())
        self.tile_pos_emb = nn.Parameter(torch.zeros(1, 40, d_model))
        init_weights(self.tile_in[0], gain=np.sqrt(2))

        if arch == "light_tx":
            enc = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=d_model * 2,
                dropout=0.0, batch_first=True, norm_first=True, activation="gelu",
            )
            self.tile_encoder = nn.TransformerEncoder(enc, num_layers=1, enable_nested_tensor=False)
        else:
            self.tile_encoder = nn.Sequential(
                _TileMixerBlock(d_model, num_tiles=40),
                _TileMixerBlock(d_model, num_tiles=40),
            )
        self.tile_norm = nn.LayerNorm(d_model)

        # 2. Event Ring Encoder
        d_ev = max(32, min(64, d_model))
        self.event_encoder = _EventMixerEncoder(d_ev=d_ev, seq_len=EVENT_RING_LEN)

        # 3. Scalar Telemetry Encoder & Fusion Trunk
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
        )
        init_weights(self.scalar_encoder[0], gain=np.sqrt(2))

        fusion_in = (2 * d_model) + self.event_encoder.out_dim + 192
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
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
        init_weights(self.fusion_proj[0], gain=np.sqrt(2))
        for layer in self.res_block:
            init_weights(layer, gain=1.0)

        # 4. Trade Response Pathway (attends to H_tiles over pending offer & req bundles)
        self.trade_resp_branch = nn.Sequential(
            nn.Linear(hidden_dim + 2 * d_model, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
        )
        init_weights(self.trade_resp_branch[0], gain=np.sqrt(2))

        # 5. Primary Discrete Action Heads
        self.head_core = nn.Linear(hidden_dim, 6)
        self.head_auction = nn.Linear(hidden_dim, 7)
        self.head_property = nn.Linear(hidden_dim, 162)
        self.head_trade_response = nn.Linear(hidden_dim // 2, 3)

        # 6. Bilateral Bundle Trade Heads:
        #    trade_target -> req_bundle (42-bit) -> z_req -> offer_bundle (42-bit) -> z_offer -> offer_cash -> req_cash
        self.head_trade_target = nn.Linear(hidden_dim, 3)

        self.emb_target_for_req = nn.Embedding(3, hidden_dim)
        self.proj_req_tile_ctx = nn.Linear(hidden_dim, d_model)
        self.head_req_tile_score = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        self.head_req_goojf = nn.Linear(hidden_dim, 2)

        # Permutation-Invariant Set Sum Pooling Item Embeddings (42 -> hidden_dim)
        self.emb_item_req = nn.Parameter(torch.zeros(BUNDLE_DIM, hidden_dim))
        self.emb_target_for_offer = nn.Embedding(3, hidden_dim)
        self.proj_offer_tile_ctx = nn.Linear(hidden_dim, d_model)
        self.head_offer_tile_score = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        self.head_offer_goojf = nn.Linear(hidden_dim, 2)

        self.emb_item_offer = nn.Parameter(torch.zeros(BUNDLE_DIM, hidden_dim))
        self.emb_target_for_cash = nn.Embedding(3, hidden_dim)
        self.head_trade_offer_cash = nn.Linear(hidden_dim, trade_cash_dim)
        self.emb_offer_cash_for_req = nn.Embedding(trade_cash_dim, hidden_dim)
        self.head_trade_req_cash = nn.Linear(hidden_dim, trade_cash_dim)

        self._init_heads()

    def _init_heads(self):
        head_gain = 0.01
        for h in (
            self.head_core,
            self.head_auction,
            self.head_property,
            self.head_trade_response,
            self.head_trade_target,
            self.head_req_goojf,
            self.head_offer_goojf,
            self.head_trade_offer_cash,
            self.head_trade_req_cash,
        ):
            init_weights(h, gain=head_gain)

        init_weights(self.proj_req_tile_ctx, gain=1.0)
        init_weights(self.proj_offer_tile_ctx, gain=1.0)
        init_weights(self.head_req_tile_score[0], gain=np.sqrt(2))
        init_weights(self.head_req_tile_score[2], gain=head_gain)
        init_weights(self.head_offer_tile_score[0], gain=np.sqrt(2))
        init_weights(self.head_offer_tile_score[2], gain=head_gain)

        nn.init.zeros_(self.emb_target_for_req.weight)
        nn.init.zeros_(self.emb_target_for_offer.weight)
        nn.init.zeros_(self.emb_target_for_cash.weight)
        nn.init.zeros_(self.emb_offer_cash_for_req.weight)
        nn.init.zeros_(self.emb_item_req)
        nn.init.zeros_(self.emb_item_offer)

    def encode(
        self, tile_state: torch.Tensor, scalar_obs: torch.Tensor, event_ring: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes structured observation -> (trunk_features [B, 384], H_tiles [B, 40, d_model])."""
        t_in = self.tile_in(tile_state) + self.tile_pos_emb
        H_tiles = self.tile_norm(self.tile_encoder(t_in))
        z_board = torch.cat([H_tiles.mean(dim=1), H_tiles.max(dim=1).values], dim=-1)
        z_events = self.event_encoder(event_ring)
        z_scalar = self.scalar_encoder(scalar_obs)
        h = self.fusion_proj(torch.cat([z_board, z_events, z_scalar], dim=-1))
        features = F.silu(h + self.res_block(h))
        return features, H_tiles

    @staticmethod
    def _mask_categorical_logits(logits: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        mask_b = (mask > 0).to(logits.device)
        scaled = logits / max(1e-3, float(temperature)) if temperature != 1.0 else logits
        return torch.where(mask_b, scaled, torch.tensor(-1e8, dtype=logits.dtype, device=logits.device))

    @staticmethod
    def _sample_categorical(masked_logits: torch.Tensor, deterministic: bool) -> torch.Tensor:
        if deterministic:
            return torch.argmax(masked_logits, dim=-1)
        probs = F.softmax(masked_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @staticmethod
    def _categorical_lp_and_ent(
        masked_logits: torch.Tensor, mask: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_probs_all = F.log_softmax(masked_logits, dim=-1)
        lp = log_probs_all.gather(-1, action.long().unsqueeze(-1)).squeeze(-1)
        probs_all = log_probs_all.exp()
        m_f = (mask > 0).to(logits_dtype := masked_logits.dtype)
        # Zero invalid entries cleanly before sum so -1e8 never adds floating residual
        safe_lp = torch.where(m_f > 0.5, log_probs_all, torch.zeros(1, dtype=logits_dtype, device=masked_logits.device))
        ent = -((probs_all * safe_lp) * m_f).sum(dim=-1)
        return lp, ent

    @staticmethod
    def _bernoulli_lp_and_ent(
        logits: torch.Tensor, mask: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m_f = (mask > 0).to(logits.dtype).to(logits.device)
        act_f = action.to(logits.dtype) * m_f
        bce = F.binary_cross_entropy_with_logits(logits, act_f, reduction="none")
        lp = -(bce * m_f).sum(dim=-1)
        p = torch.sigmoid(logits)
        # Exact numerically stable binary entropy: H(p) = log(1+e^-z) + z * sigmoid(-z)... or softplus form
        # Equivalent exact form: softplus(z) - z * p = -p*log(p) - (1-p)*log(1-p) >= 0 everywhere!
        bit_ent = F.softplus(logits) - logits * p
        cnt = m_f.sum(dim=-1).clamp(min=1.0)
        # Normalize entropy contribution gently by sqrt(cnt) so multi-property turns don't overpower 1-choice turns
        ent = (bit_ent * m_f).sum(dim=-1) / torch.sqrt(cnt)
        return lp, ent

    @staticmethod
    def _valid_uniform_kl_cat(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = ((mask > 0) & (logits > -1e4)).to(logits.dtype).to(logits.device)
        cnt = m.sum(dim=-1, keepdim=True)
        has_choice = (cnt.squeeze(-1) > 1.5).to(logits.dtype)
        safe_cnt = cnt.clamp(min=1.0)
        masked_logits = torch.where(m > 0.5, logits, torch.tensor(-1e8, dtype=logits.dtype, device=logits.device))
        log_probs = F.log_softmax(masked_logits, dim=-1)
        mean_neg_logp = -((log_probs * m).sum(dim=-1) / safe_cnt.squeeze(-1))
        uni_kl = (mean_neg_logp - torch.log(safe_cnt.squeeze(-1))).clamp(min=0.0)
        return uni_kl * has_choice

    @staticmethod
    def _valid_uniform_kl_bern(logits: torch.Tensor, mask: torch.Tensor, p_prior: float = 0.15) -> torch.Tensor:
        """Exact KL(Bernoulli(p_prior) || Bernoulli(sigmoid(z))) preventing overly dense 50% bundle spam."""
        m = (mask > 0).to(logits.dtype).to(logits.device)
        cnt = m.sum(dim=-1)
        has_choice = (cnt > 0.5).to(logits.dtype)
        # KL(p0 || sig(z)) = softplus(z) - p0*z - H(p0), strictly >= 0 and minimized cleanly at sig(z) == p0
        h_prior = -(p_prior * math.log(p_prior) + (1.0 - p_prior) * math.log(1.0 - p_prior))
        kl_bits = F.softplus(logits) - p_prior * logits - h_prior
        return ((kl_bits.clamp(min=0.0) * m).sum(dim=-1) / cnt.clamp(min=1.0)) * has_choice

    @staticmethod
    def _valid_logit_l2_cat(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = ((mask > 0) & (logits > -1e4)).to(logits.dtype).to(logits.device)
        cnt = m.sum(dim=-1, keepdim=True)
        has_choice = (cnt.squeeze(-1) > 1.5).to(logits.dtype)
        safe_cnt = cnt.clamp(min=1.0)
        mean_z = (logits * m).sum(dim=-1, keepdim=True) / safe_cnt
        var_z = (((logits - mean_z) ** 2) * m).sum(dim=-1) / safe_cnt.squeeze(-1)
        return var_z * has_choice

    @staticmethod
    def _valid_logit_l2_bern(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = (mask > 0).to(logits.dtype).to(logits.device)
        cnt = m.sum(dim=-1)
        has_choice = (cnt > 0.5).to(logits.dtype)
        return (((logits ** 2) * m).sum(dim=-1) / cnt.clamp(min=1.0)) * has_choice

    @staticmethod
    def _safe_conditional_mask(cond_mask: torch.Tensor, fallback_mask: torch.Tensor) -> torch.Tensor:
        has_any = cond_mask.sum(dim=-1, keepdim=True) > 0
        return torch.where(has_any, cond_mask, fallback_mask)

    def _score_bundle_logits(
        self,
        H_tiles: torch.Tensor,
        ctx: torch.Tensor,
        proj_ctx: nn.Linear,
        tile_mlp: nn.Sequential,
        goojf_head: nn.Linear,
    ) -> torch.Tensor:
        tile_ctx = proj_ctx(ctx).unsqueeze(1)
        tile_logits = tile_mlp(F.silu(H_tiles + tile_ctx)).squeeze(-1)  # (B, 40)
        goojf_logits = goojf_head(ctx)                                  # (B, 2)
        return torch.cat([tile_logits, goojf_logits], dim=-1)           # (B, 42)

    def _build_outputs(
        self,
        tile_state: torch.Tensor,
        scalar_obs: torch.Tensor,
        event_ring: torch.Tensor,
        action_masks: Dict[str, torch.Tensor],
        given_actions: Optional[Dict[str, torch.Tensor]] = None,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        features, H_tiles = self.encode(tile_state, scalar_obs, event_ring)
        device = features.device
        B = features.shape[0]
        temp = 1.0 if given_actions is not None else max(1e-3, float(temperature))

        # Pending trade pool for PHASE_TRADE_RESPONSE from channels 14 (offer) & 15 (req)
        off_w = tile_state[..., 14:15]
        req_w = tile_state[..., 15:16]
        off_pool = (H_tiles * off_w).sum(dim=1) / off_w.sum(dim=1).clamp(min=1.0)
        req_pool = (H_tiles * req_w).sum(dim=1) / req_w.sum(dim=1).clamp(min=1.0)
        resp_features = self.trade_resp_branch(torch.cat([features, off_pool, req_pool], dim=-1))

        logits_dict: Dict[str, torch.Tensor] = {}
        cond_masks: Dict[str, torch.Tensor] = {}
        actions_out: Dict[str, torch.Tensor] = {}

        # 1. Fixed discrete heads
        for k, raw_l in (
            ("core", self.head_core(features)),
            ("auction", self.head_auction(features)),
            ("property", self.head_property(features)),
            ("trade_response", self.head_trade_response(resp_features)),
            ("trade_target", self.head_trade_target(features)),
        ):
            m_k = action_masks[k].to(device)
            ml = self._mask_categorical_logits(raw_l, m_k, temp)
            logits_dict[k] = ml
            cond_masks[k] = m_k
            if given_actions is None:
                actions_out[k] = self._sample_categorical(ml, deterministic)
            else:
                actions_out[k] = given_actions[k].to(device).long()

        # Fast-path during rollout inference: if no agent in batch is proposing trade / countering,
        # skip expensive bundle tile-MLP scoring and cash heads (zero impact on log_prob as is_trade_proposal==0)
        if given_actions is None:
            is_manage = scalar_obs[..., 44] > 0.5
            is_counter = scalar_obs[..., 46] > 0.5
            prop_proposes = actions_out["property"] == ACT_PROP_PROPOSE_TRADE
            is_trade_active = (is_manage & prop_proposes) | is_counter
            if not torch.any(is_trade_active):
                actions_out["trade_req_bundle"] = torch.zeros((B, BUNDLE_DIM), dtype=torch.int8, device=device)
                actions_out["trade_offer_bundle"] = torch.zeros((B, BUNDLE_DIM), dtype=torch.int8, device=device)
                actions_out["trade_offer_cash"] = torch.zeros((B,), dtype=torch.long, device=device)
                actions_out["trade_req_cash"] = torch.zeros((B,), dtype=torch.long, device=device)
                logits_dict["trade_req_bundle"] = torch.zeros((B, BUNDLE_DIM), dtype=features.dtype, device=device)
                logits_dict["trade_offer_bundle"] = torch.zeros((B, BUNDLE_DIM), dtype=features.dtype, device=device)
                logits_dict["trade_offer_cash"] = torch.zeros((B, self.trade_cash_dim), dtype=features.dtype, device=device)
                logits_dict["trade_req_cash"] = torch.zeros((B, self.trade_cash_dim), dtype=features.dtype, device=device)
                cond_masks["trade_req_bundle"] = action_masks["trade_req_bundle"][:, 0].to(device)
                cond_masks["trade_offer_bundle"] = action_masks["trade_offer_bundle"].to(device)
                cond_masks["trade_offer_cash"] = action_masks["trade_offer_cash"].to(device)
                cond_masks["trade_req_cash"] = action_masks["trade_req_cash"].to(device)
                return logits_dict, actions_out, cond_masks, {"features": features}

        # 2. Target-Conditioned Req Bundle ({0, 1}^42 via Multi-Bernoulli)
        target_idx = actions_out["trade_target"].clamp(0, 2)
        req_masks_all = action_masks["trade_req_bundle"].to(device)  # (B, 3, 42)
        cond_req_mask = req_masks_all[torch.arange(B, device=device), target_idx]  # (B, 42)
        cond_masks["trade_req_bundle"] = cond_req_mask

        ctx_req = features + self.emb_target_for_req(target_idx)
        raw_req_logits = self._score_bundle_logits(
            H_tiles, ctx_req, self.proj_req_tile_ctx, self.head_req_tile_score, self.head_req_goojf
        )
        req_logits = raw_req_logits / temp if temp != 1.0 else raw_req_logits
        logits_dict["trade_req_bundle"] = req_logits
        if given_actions is None:
            if deterministic:
                req_b = ((req_logits > 0.0) & (cond_req_mask > 0)).to(torch.int8)
            else:
                probs_req = torch.sigmoid(req_logits) * (cond_req_mask > 0).to(req_logits.dtype)
                req_b = torch.bernoulli(probs_req).to(torch.int8)
            actions_out["trade_req_bundle"] = req_b
        else:
            req_b = given_actions["trade_req_bundle"].to(device).to(torch.int8)
            actions_out["trade_req_bundle"] = req_b

        # Permutation-Invariant Set Sum Pooling of requested bundle
        req_active_f = (req_b * (cond_req_mask > 0).to(torch.int8)).to(features.dtype)
        z_req = torch.matmul(req_active_f, self.emb_item_req)

        # 3. Offer Bundle ({0, 1}^42 conditioned on target + z_req)
        cond_offer_mask = action_masks["trade_offer_bundle"].to(device)
        cond_masks["trade_offer_bundle"] = cond_offer_mask
        ctx_offer = features + self.emb_target_for_offer(target_idx) + z_req
        raw_offer_logits = self._score_bundle_logits(
            H_tiles, ctx_offer, self.proj_offer_tile_ctx, self.head_offer_tile_score, self.head_offer_goojf
        )
        offer_logits = raw_offer_logits / temp if temp != 1.0 else raw_offer_logits
        logits_dict["trade_offer_bundle"] = offer_logits
        if given_actions is None:
            if deterministic:
                offer_b = ((offer_logits > 0.0) & (cond_offer_mask > 0)).to(torch.int8)
            else:
                probs_off = torch.sigmoid(offer_logits) * (cond_offer_mask > 0).to(offer_logits.dtype)
                offer_b = torch.bernoulli(probs_off).to(torch.int8)
            actions_out["trade_offer_bundle"] = offer_b
        else:
            offer_b = given_actions["trade_offer_bundle"].to(device).to(torch.int8)
            actions_out["trade_offer_bundle"] = offer_b

        # Set Sum Pooling of offered bundle
        off_active_f = (offer_b * (cond_offer_mask > 0).to(torch.int8)).to(features.dtype)
        z_offer = torch.matmul(off_active_f, self.emb_item_offer)

        # 4. Trade Offer Cash & Req Cash (mutual exclusion & anti-empty-beg / anti-giveaway)
        ctx_cash = features + self.emb_target_for_cash(target_idx) + z_req + z_offer
        has_req = req_active_f.sum(dim=-1) > 0.5
        has_off = off_active_f.sum(dim=-1) > 0.5
        both_empty = (~has_req) & (~has_off)
        is_cash_buy = (~has_off) & has_req
        is_prop_sale = has_off & (~has_req)

        base_oc_mask = action_masks["trade_offer_cash"].to(device)
        cond_oc_mask = base_oc_mask.clone()
        cond_oc_mask[..., 0] = torch.where(is_cash_buy, torch.zeros_like(cond_oc_mask[..., 0]), cond_oc_mask[..., 0])
        cond_oc_mask[..., 1:] = torch.where(
            (is_prop_sale | both_empty).unsqueeze(-1),
            torch.zeros_like(cond_oc_mask[..., 1:]),
            cond_oc_mask[..., 1:],
        )
        cond_oc_mask = self._safe_conditional_mask(cond_oc_mask, base_oc_mask)
        cond_masks["trade_offer_cash"] = cond_oc_mask

        oc_logits = self._mask_categorical_logits(self.head_trade_offer_cash(ctx_cash), cond_oc_mask, temp)
        logits_dict["trade_offer_cash"] = oc_logits
        offer_cash = (
            self._sample_categorical(oc_logits, deterministic)
            if given_actions is None
            else given_actions["trade_offer_cash"].to(device).long()
        )
        actions_out["trade_offer_cash"] = offer_cash

        base_rc_mask = action_masks["trade_req_cash"].to(device)
        cond_rc_mask = base_rc_mask.clone()
        force_zero_rc = (offer_cash > 0) | is_cash_buy | both_empty
        cond_rc_mask[..., 1:] = torch.where(
            force_zero_rc.unsqueeze(-1),
            torch.zeros_like(cond_rc_mask[..., 1:]),
            cond_rc_mask[..., 1:],
        )
        cond_rc_mask[..., 0] = torch.where(
            is_prop_sale, torch.zeros_like(cond_rc_mask[..., 0]), cond_rc_mask[..., 0]
        )
        cond_rc_mask = self._safe_conditional_mask(cond_rc_mask, base_rc_mask)
        cond_masks["trade_req_cash"] = cond_rc_mask

        ctx_rc = ctx_cash + self.emb_offer_cash_for_req(offer_cash.clamp(0, self.trade_cash_dim - 1))
        rc_logits = self._mask_categorical_logits(self.head_trade_req_cash(ctx_rc), cond_rc_mask, temp)
        logits_dict["trade_req_cash"] = rc_logits
        req_cash = (
            self._sample_categorical(rc_logits, deterministic)
            if given_actions is None
            else given_actions["trade_req_cash"].to(device).long()
        )
        actions_out["trade_req_cash"] = req_cash

        return logits_dict, actions_out, cond_masks, {"features": features}

    def _compute_phase_aware_metrics(
        self,
        scalar_obs: torch.Tensor,
        action_masks: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        logits_dict: Dict[str, torch.Tensor],
        cond_masks: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes exact phase-masked log_prob, entropy, uniform_kl, and logit_l2."""
        # Phase one-hot resides at scalar_obs[..., 40:48]
        u = scalar_obs[..., 40 : 40 + NUM_PHASES]
        is_core = u[..., 0] + u[..., 1] + u[..., 3]
        is_buy = u[..., 1]
        is_auc = u[..., 2]
        is_manage = u[..., 4]
        is_trade_resp = u[..., 5]
        is_counter = u[..., 6]

        # Log-prob & entropy per head
        lp_dict: Dict[str, torch.Tensor] = {}
        ent_dict: Dict[str, torch.Tensor] = {}
        for k in ("core", "auction", "property", "trade_target", "trade_offer_cash", "trade_req_cash", "trade_response"):
            lp_dict[k], ent_dict[k] = self._categorical_lp_and_ent(logits_dict[k], cond_masks[k], actions[k])
        for k in ("trade_req_bundle", "trade_offer_bundle"):
            lp_dict[k], ent_dict[k] = self._bernoulli_lp_and_ent(logits_dict[k], cond_masks[k], actions[k])

        # Active trade proposal occurs on explicit property==161 in MANAGE, or any step in TRADE_COUNTER
        is_trade_proposal = (is_manage * (actions["property"] == ACT_PROP_PROPOSE_TRADE).float()) + is_counter

        log_prob = (
            is_core * lp_dict["core"]
            + is_auc * lp_dict["auction"]
            + is_manage * lp_dict["property"]
            + is_trade_proposal * (
                lp_dict["trade_target"]
                + lp_dict["trade_req_bundle"]
                + lp_dict["trade_offer_bundle"]
                + lp_dict["trade_offer_cash"]
                + lp_dict["trade_req_cash"]
            )
            + is_trade_resp * lp_dict["trade_response"]
        )

        entropy = (
            is_core * ent_dict["core"]
            + 2.5 * is_auc * ent_dict["auction"]
            + is_manage * ent_dict["property"]
            + is_trade_proposal * (
                ent_dict["trade_target"]
                + ent_dict["trade_req_bundle"]
                + ent_dict["trade_offer_bundle"]
                + ent_dict["trade_offer_cash"]
                + ent_dict["trade_req_cash"]
            )
            + 1.5 * is_trade_resp * ent_dict["trade_response"]
        )

        # Exploration regularizers on rare/high-leverage heads
        auc_kl = self._valid_uniform_kl_cat(logits_dict["auction"], cond_masks["auction"])
        tr_kl = self._valid_uniform_kl_cat(logits_dict["trade_response"], cond_masks["trade_response"])
        req_b_kl = self._valid_uniform_kl_bern(logits_dict["trade_req_bundle"], cond_masks["trade_req_bundle"])
        off_b_kl = self._valid_uniform_kl_bern(logits_dict["trade_offer_bundle"], cond_masks["trade_offer_bundle"])
        uniform_kl = (
            is_auc * auc_kl
            + 0.25 * is_trade_resp * tr_kl
            + 0.15 * is_trade_proposal * (req_b_kl + off_b_kl)
        )

        auc_l2 = self._valid_logit_l2_cat(logits_dict["auction"], cond_masks["auction"])
        tr_l2 = self._valid_logit_l2_cat(logits_dict["trade_response"], cond_masks["trade_response"])
        buy_l2 = self._valid_logit_l2_cat(logits_dict["core"], cond_masks["core"])
        oc_l2 = self._valid_logit_l2_cat(logits_dict["trade_offer_cash"], cond_masks["trade_offer_cash"])
        rc_l2 = self._valid_logit_l2_cat(logits_dict["trade_req_cash"], cond_masks["trade_req_cash"])
        req_b_l2 = self._valid_logit_l2_bern(logits_dict["trade_req_bundle"], cond_masks["trade_req_bundle"])
        off_b_l2 = self._valid_logit_l2_bern(logits_dict["trade_offer_bundle"], cond_masks["trade_offer_bundle"])
        logit_l2 = (
            is_auc * auc_l2
            + 0.50 * is_trade_resp * tr_l2
            + 0.10 * is_buy * buy_l2
            + 0.05 * is_trade_proposal * (oc_l2 + rc_l2 + 0.5 * (req_b_l2 + off_b_l2))
        )

        return log_prob, entropy, uniform_kl, logit_l2

    def get_head_logits(
        self,
        tile_state: torch.Tensor,
        scalar_obs: torch.Tensor,
        event_ring: torch.Tensor,
        action_masks: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Returns masked logits for discrete decision heads (used by Search & Distillation)."""
        logits_dict, _, _, _ = self._build_outputs(
            tile_state, scalar_obs, event_ring, action_masks, given_actions=None, deterministic=True, temperature=1.0
        )
        return logits_dict

    def forward(
        self,
        tile_state: torch.Tensor,
        scalar_obs: torch.Tensor,
        event_ring: torch.Tensor,
        action_masks: Dict[str, torch.Tensor],
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        logits_dict, actions_out, cond_masks, _ = self._build_outputs(
            tile_state, scalar_obs, event_ring, action_masks,
            given_actions=None, deterministic=deterministic, temperature=temperature,
        )
        log_prob, entropy, _, _ = self._compute_phase_aware_metrics(
            scalar_obs, action_masks, actions_out, logits_dict, cond_masks
        )
        return actions_out, log_prob, entropy

    def evaluate_actions_with_reg(
        self,
        tile_state: torch.Tensor,
        scalar_obs: torch.Tensor,
        event_ring: torch.Tensor,
        action_masks: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        logits_dict, _, cond_masks, _ = self._build_outputs(
            tile_state, scalar_obs, event_ring, action_masks,
            given_actions=actions, deterministic=False, temperature=1.0,
        )
        log_prob, entropy, uniform_kl, logit_l2 = self._compute_phase_aware_metrics(
            scalar_obs, action_masks, actions, logits_dict, cond_masks
        )
        return log_prob, entropy, uniform_kl, logit_l2, logits_dict
