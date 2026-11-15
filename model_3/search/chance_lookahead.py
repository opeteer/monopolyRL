"""Fast-snapshot Bounded Chance Lookahead for high-leverage Monopoly decisions.

Uses MonopolyCore.save_snapshot()/restore_snapshot() + fast_search_mode=True (~35x
faster than deepcopy), branches explicitly over the 11 dice outcomes weighted by
exact 2d6 probabilities when entering PHASE_ROLL, and evaluates leaf states in
one batched QuantileCritic.get_risk_value() call (CVaR risk-sensitive value).
"""

from typing import Dict, Any, Optional, Tuple, List, Callable
import numpy as np
import torch

from model_2.engine.board_data import DICE_OUTCOMES, DICE_PROBABILITIES
from model_3.engine.monopoly_core import (
    MonopolyCore,
    PHASE_ROLL,
    PHASE_BUY_OR_AUCTION,
    PHASE_AUCTION,
    PHASE_JAIL_DECISION,
    PHASE_PROPERTY_MANAGE,
    PHASE_TRADE_RESPONSE,
    ACT_PROP_END_MANAGE,
    ACT_PROP_PROPOSE_TRADE,
    ACTION_KEYS,
    ACT_SHAPES,
)
from model_3.models.critic_network import QuantileCritic
from model_3.models.storage import SEARCH_MAX_HEAD_DIM

SEARCH_HEAD_NAME_TO_ID: Dict[str, int] = {
    "core": 0,
    "auction": 1,
    "property": 2,
    "trade_response": 3,
}
SEARCH_ID_TO_HEAD_NAME: Dict[int, str] = {v: k for k, v in SEARCH_HEAD_NAME_TO_ID.items()}

HIGH_LEVERAGE_PHASES = {
    PHASE_BUY_OR_AUCTION,
    PHASE_AUCTION,
    PHASE_JAIL_DECISION,
    PHASE_PROPERTY_MANAGE,
    PHASE_TRADE_RESPONSE,
}

# Canonical (d1, d2) pairs covering sums 2..12 for 11-outcome chance branching
DICE_SUM_TO_PAIR: Dict[int, Tuple[int, int]] = {
    2: (1, 1),
    3: (1, 2),
    4: (1, 3),
    5: (2, 3),
    6: (2, 4),
    7: (3, 4),
    8: (3, 5),
    9: (4, 5),
    10: (4, 6),
    11: (5, 6),
    12: (6, 6),
}


class _FixedDiceRNG:
    """Thin RNG proxy injecting a fixed (d1, d2) roll once, then delegating remaining calls."""

    def __init__(self, base_rng, d1: int, d2: int):
        self._base_rng = base_rng
        self._pair = [d1, d2]

    def randint(self, a: int, b: int) -> int:
        if a == 1 and b == 6 and self._pair:
            return self._pair.pop(0)
        return self._base_rng.randint(a, b)

    def __getattr__(self, name: str):
        return getattr(self._base_rng, name)


def _active_head_for_phase(phase: int) -> Optional[str]:
    if phase in (PHASE_BUY_OR_AUCTION, PHASE_JAIL_DECISION):
        return "core"
    if phase == PHASE_AUCTION:
        return "auction"
    if phase == PHASE_PROPERTY_MANAGE:
        return "property"
    if phase == PHASE_TRADE_RESPONSE:
        return "trade_response"
    return None


def _clone_action_dict(actions: Dict[str, Any]) -> Dict[str, Any]:
    cloned: Dict[str, Any] = {}
    for k in ACTION_KEYS:
        v = actions[k]
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy() if v.ndim > 0 else int(v.item())
        if len(ACT_SHAPES[k]) > 0:
            cloned[k] = np.asarray(v, dtype=np.int8).copy()
        else:
            cloned[k] = int(v)
    return cloned


class BoundedChanceLookahead:
    """Evaluates top-K prior candidate actions via snapshot + 11-outcome chance lookahead."""

    def __init__(
        self,
        gamma: float = 0.997,
        cvar_alpha: float = 0.25,
        cvar_weight: float = 0.30,
        top_k: int = 4,
        temperature: float = 0.25,
        branch_dice: bool = True,
    ):
        self.gamma = float(gamma)
        self.cvar_alpha = float(cvar_alpha)
        self.cvar_weight = float(cvar_weight)
        self.top_k = int(max(2, top_k))
        self.temperature = float(max(1e-3, temperature))
        self.branch_dice = bool(branch_dice)

    @torch.no_grad()
    def _evaluate_leaf_states(
        self,
        states: List[np.ndarray],
        player_id: int,
        critic: Optional[QuantileCritic],
        leaf_value_fn: Optional[Callable[[np.ndarray, int], np.ndarray]],
    ) -> np.ndarray:
        if not states:
            return np.zeros((0,), dtype=np.float32)
        batch_gs = np.stack(states, axis=0).astype(np.float32, copy=False)
        if leaf_value_fn is not None:
            vals = leaf_value_fn(batch_gs, player_id)
            return np.asarray(vals, dtype=np.float32).reshape(-1)
        if critic is None:
            return np.zeros((len(states),), dtype=np.float32)
        gs_t = torch.from_numpy(batch_gs)
        risk_v, _ = critic.get_risk_value(gs_t, cvar_alpha=self.cvar_alpha, cvar_weight=self.cvar_weight)
        return risk_v[:, player_id].cpu().numpy().astype(np.float32, copy=False)

    def _select_candidates(
        self,
        head_name: str,
        head_mask: np.ndarray,
        actor_logits: Optional[Dict[str, torch.Tensor]],
        sampled_action_idx: int,
    ) -> List[int]:
        valid_indices = [int(i) for i in np.where(head_mask > 0)[0]]
        if len(valid_indices) <= 1:
            return valid_indices
        if len(valid_indices) <= self.top_k or actor_logits is None or head_name not in actor_logits:
            # Put sampled_action_idx first + keep up to top_k
            cands = valid_indices[: self.top_k]
            if sampled_action_idx in valid_indices and sampled_action_idx not in cands:
                cands[-1] = sampled_action_idx
            return cands

        logits_t = actor_logits[head_name]
        if isinstance(logits_t, torch.Tensor):
            logits_np = logits_t.detach().cpu().numpy().reshape(-1)
        else:
            logits_np = np.asarray(logits_t, dtype=np.float32).reshape(-1)

        valid_sorted = sorted(valid_indices, key=lambda idx: float(logits_np[idx]), reverse=True)
        cands = valid_sorted[: self.top_k]
        # Always include ACT_PROP_END_MANAGE and sampled_action_idx when legal on property head
        if head_name == "property":
            for anchor in (ACT_PROP_END_MANAGE, sampled_action_idx):
                if anchor in valid_indices and anchor not in cands:
                    cands[-1] = anchor
        elif sampled_action_idx in valid_indices and sampled_action_idx not in cands:
            cands[-1] = sampled_action_idx
        # Deduplicate while preserving order
        seen = set()
        dedup: List[int] = []
        for c in cands:
            if c not in seen:
                seen.add(c)
                dedup.append(c)
        return dedup

    def select_action(
        self,
        core: MonopolyCore,
        player_id: int,
        sampled_actions: Dict[str, Any],
        actor_logits: Optional[Dict[str, torch.Tensor]] = None,
        critic: Optional[QuantileCritic] = None,
        leaf_value_fn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        sample_from_search: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[Dict[str, Any], bool, int, np.ndarray]:
        """Runs bounded fast-snapshot lookahead and returns (best_act_dict, valid, head_id, target_probs)."""
        base_act = _clone_action_dict(sampled_actions)
        zero_probs = np.zeros(SEARCH_MAX_HEAD_DIM, dtype=np.float32)

        phase = core.phase
        if player_id != core.active_decision_player or phase not in HIGH_LEVERAGE_PHASES:
            return base_act, False, 0, zero_probs

        head_name = _active_head_for_phase(phase)
        if head_name is None:
            return base_act, False, 0, zero_probs
        head_id = SEARCH_HEAD_NAME_TO_ID[head_name]

        masks = core.get_action_masks(player_id)
        head_mask = masks[head_name]
        cands = self._select_candidates(head_name, head_mask, actor_logits, int(base_act[head_name]))
        if len(cands) <= 1:
            return base_act, False, head_id, zero_probs

        root_snap = core.save_snapshot()
        orig_fast = core.fast_search_mode
        orig_rng = core.rng
        core.fast_search_mode = True

        leaf_states: List[np.ndarray] = []
        # Each branch record tracks (cand_idx, immediate_reward, prob_weight, done, leaf_state_idx)
        branch_records: List[List[Tuple[float, float, bool, int]]] = [[] for _ in cands]

        try:
            for c_pos, cand_val in enumerate(cands):
                core.restore_snapshot(root_snap)
                core.rng = orig_rng

                cand_act = _clone_action_dict(base_act)
                cand_act[head_name] = int(cand_val)

                # If PROPERTY_MANAGE proposes a sub-action < 160 (e.g. build house / unmortgage),
                # evaluate resulting state and, if still in PROPERTY_MANAGE, close turn cleanly
                # so dice hazards/rewards can be compared symmetrically with ACT_PROP_END_MANAGE.
                p_key = f"player_{player_id}"
                _, rew1, done1, _ = core.step({p_key: cand_act}, skip_obs=True)
                r_first = float(rew1.get(p_key, 0.0))
                is_done1 = bool(done1.get(p_key, False) or core.bankrupt[player_id] or core.phase == 7)

                if not is_done1 and head_name == "property" and cand_val < 160 and core.phase == PHASE_PROPERTY_MANAGE and core.active_decision_player == player_id:
                    close_act = _clone_action_dict(cand_act)
                    close_act["property"] = ACT_PROP_END_MANAGE
                    _, rew_c, done_c, _ = core.step({p_key: close_act}, skip_obs=True)
                    r_first += self.gamma * float(rew_c.get(p_key, 0.0))
                    is_done1 = bool(done_c.get(p_key, False) or core.bankrupt[player_id] or core.phase == 7)

                if is_done1:
                    branch_records[c_pos].append((r_first, 1.0, True, -1))
                    continue

                if self.branch_dice and core.phase == PHASE_ROLL:
                    roller = core.active_decision_player
                    r_key = f"player_{roller}"
                    roll_snap = core.save_snapshot()
                    roll_act = _clone_action_dict(base_act)
                    roll_act["core"] = 0  # ACT_CORE_ROLL_PASS
                    for dice_sum, prob in zip(DICE_OUTCOMES, DICE_PROBABILITIES):
                        d1, d2 = DICE_SUM_TO_PAIR[int(dice_sum)]
                        core.restore_snapshot(roll_snap)
                        core.rng = _FixedDiceRNG(orig_rng, d1, d2)
                        _, rew2, done2, _ = core.step({r_key: roll_act}, skip_obs=True)
                        core.rng = orig_rng
                        r_second = float(rew2.get(p_key, 0.0))
                        is_done2 = bool(done2.get(p_key, False) or core.bankrupt[player_id] or core.phase == 7)
                        total_imm = r_first + self.gamma * r_second
                        if is_done2:
                            branch_records[c_pos].append((total_imm, float(prob), True, -1))
                        else:
                            gs_idx = len(leaf_states)
                            leaf_states.append(core.get_global_state())
                            branch_records[c_pos].append((total_imm, float(prob), False, gs_idx))
                else:
                    gs_idx = len(leaf_states)
                    leaf_states.append(core.get_global_state())
                    branch_records[c_pos].append((r_first, 1.0, False, gs_idx))
        finally:
            core.rng = orig_rng
            core.restore_snapshot(root_snap)
            core.fast_search_mode = orig_fast

        leaf_vals = self._evaluate_leaf_states(leaf_states, player_id, critic, leaf_value_fn)
        discount = self.gamma * self.gamma if self.branch_dice else self.gamma

        q_vals = np.zeros(len(cands), dtype=np.float64)
        for c_pos, recs in enumerate(branch_records):
            ev = 0.0
            for imm_r, prob, is_done, gs_idx in recs:
                if is_done or gs_idx < 0:
                    ev += prob * imm_r
                else:
                    ev += prob * (imm_r + discount * float(leaf_vals[gs_idx]))
            q_vals[c_pos] = ev

        # Compute temperature-scaled softmax target distribution over candidate actions
        scaled = (q_vals - q_vals.max()) / self.temperature
        exp_q = np.exp(np.clip(scaled, -40.0, 40.0))
        probs = (exp_q / max(1e-12, exp_q.sum())).astype(np.float32)

        target_probs = np.zeros(SEARCH_MAX_HEAD_DIM, dtype=np.float32)
        for c_pos, cand_val in enumerate(cands):
            target_probs[int(cand_val)] = probs[c_pos]

        if sample_from_search and rng is not None:
            chosen_idx = int(cands[int(rng.choice(len(cands), p=probs))])
        else:
            chosen_idx = int(cands[int(np.argmax(q_vals))])

        best_act = _clone_action_dict(base_act)
        best_act[head_name] = chosen_idx
        return best_act, True, head_id, target_probs
