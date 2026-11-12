"""Unit tests for the Offensive Monopoly Learning & Trade Observation Redesign.

Validates:
1. Monopoly starvation penalty: quadratic drop in potential when 0 monopolies as unowned tiles vanish.
2. Monopoly surge: massive leap in potential upon completing a street monopoly.
3. Single-head property management: action 160 (PASS) ends phase cleanly.
4. Non-vetoed building: building actions (prop_act < 40) are executed even if core_act is END_MANAGE.
5. Non-preemptive trading: initiator remains in PHASE_PROPERTY_MANAGE after trade to build immediately.
6. Trade observation discrimination: 500-dim obs sharply distinguishes good vs bad trade proposals.
7. Auto-routing target: requesting a property routes trade to its rightful owner deterministically.
8. Trade acceptance reward surge: accepting a starvation-breaking monopoly trade yields massive PBRS reward.
9. Property-first management order: valid property builds execute before trade proposals on the same step.
"""

import numpy as np
import torch

from monopoly_marl.engine.monopoly_core import (
    MonopolyCore, PHASE_PROPERTY_MANAGE, PHASE_ROLL, PHASE_BUY_OR_AUCTION,
    PHASE_TRADE_RESPONSE, ACT_CORE_ROLL_PASS, ACT_CORE_END_MANAGE, PURCHASABLE_TILES
)
from monopoly_marl.models.actor_network import ActorNetwork


def test_monopoly_starvation_and_completion_potential():
    """Validates round-based starvation penalty, anti-exploit Buy positivity, and monopoly surge."""
    core = MonopolyCore(seed=42)

    # 1. Verify buying an unowned property is ALWAYS positive (ZERO Cobra Effect!)
    core.round_count = 15
    pot_before_buy = core._compute_potential(0)
    core.property_owner[6] = 0  # Oriental Ave ($100)
    core.cash[0] -= 100
    pot_after_buy = core._compute_potential(0)
    assert pot_after_buy > pot_before_buy + 0.3, (
        f"Buying unowned property MUST yield strictly positive potential delta, got {pot_after_buy - pot_before_buy}"
    )

    # 2. Verify round-based starvation at round 40 with 0 monopolies
    core2 = MonopolyCore(seed=42)
    pot_early = core2._compute_potential(0)
    core2.round_count = 40
    pot_starving = core2._compute_potential(0)
    assert pot_starving < pot_early - 3.0, f"Expected round-based starvation penalty (~ -3.5), got {pot_starving} vs {pot_early}"

    # 3. Completing Brown monopoly (#1 and #3) removes starvation penalty + grants monopoly bonus
    core2.property_owner[1] = 0
    core2.property_owner[3] = 0
    pot_monopoly = core2._compute_potential(0)
    assert pot_monopoly > pot_starving + 4.0, f"Expected > +4.0 surge upon monopoly completion, got {pot_monopoly} vs {pot_starving}"


def test_property_pass_action_160():
    """Validates that action 160 in property head acts as clean explicit pass."""
    core = MonopolyCore(seed=42)
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0

    masks = core.get_action_masks(0)
    assert masks["property"][160] == 1, "Action 160 (PASS) must always be unmasked in property head"

    # Execute explicit pass
    act = {
        "core": ACT_CORE_ROLL_PASS,
        "auction": 0,
        "property": 160,
        "trade_target": 0,
        "trade_offer_prop": 40,
        "trade_offer_cash": 0,
        "trade_req_prop": 40,
        "trade_req_cash": 0,
        "trade_response": 0,
    }
    core.step({"player_0": act})

    # Turn should advance cleanly to Player 1
    assert core.current_player == 1
    assert core.phase == PHASE_ROLL


def test_property_building_not_vetoed_by_core():
    """Validates that building house is executed directly without core_act vetoing it."""
    core = MonopolyCore(seed=42)
    core.property_owner[1] = 0
    core.property_owner[3] = 0
    core.cash[0] = 2000
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0

    assert core.bank_houses == 32

    # Agent selects prop_act = 1 (build on #1), even if core selects ACT_CORE_END_MANAGE
    act = {
        "core": ACT_CORE_END_MANAGE,
        "auction": 0,
        "property": 1,
        "trade_target": 0,
        "trade_offer_prop": 40,
        "trade_offer_cash": 0,
        "trade_req_prop": 40,
        "trade_req_cash": 0,
        "trade_response": 0,
    }
    core.step({"player_0": act})

    assert core.property_houses[1] == 1
    assert core.bank_houses == 31
    assert core.cash[0] == 1950


def test_non_preemptive_trading_allows_immediate_building():
    """Validates that completing a trade does not kill turn, allowing immediate house construction."""
    core = MonopolyCore(seed=42)
    core.property_owner[3] = 0
    core.property_owner[1] = 1
    core.cash[0] = 2000
    core.cash[1] = 1500
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0

    # Player 0 proposes trade: offering cash $200 for Mediterranean #1
    act_trade = {
        "core": ACT_CORE_ROLL_PASS,
        "auction": 0,
        "property": 160,
        "trade_target": 0,
        "trade_offer_prop": 40,
        "trade_offer_cash": 2,
        "trade_req_prop": 1,
        "trade_req_cash": 0,
        "trade_response": 0,
    }
    core.step({"player_0": act_trade})

    assert core.phase == PHASE_TRADE_RESPONSE
    assert core.active_decision_player == 1

    # Player 1 accepts trade
    act_accept = {
        "core": ACT_CORE_ROLL_PASS,
        "auction": 0,
        "property": 160,
        "trade_target": 0,
        "trade_offer_prop": 40,
        "trade_offer_cash": 0,
        "trade_req_prop": 40,
        "trade_req_cash": 0,
        "trade_response": 1,
    }
    core.step({"player_1": act_accept})

    assert core.property_owner[1] == 0
    assert core.property_owner[3] == 0
    assert core.phase == PHASE_PROPERTY_MANAGE
    assert core.active_decision_player == 0

    # Player 0 immediately builds a house on Mediterranean Ave on the SAME turn
    act_build = {
        "core": ACT_CORE_ROLL_PASS,
        "auction": 0,
        "property": 1,
        "trade_target": 0,
        "trade_offer_prop": 40,
        "trade_offer_cash": 0,
        "trade_req_prop": 40,
        "trade_req_cash": 0,
        "trade_response": 0,
    }
    core.step({"player_0": act_build})

    assert core.property_houses[1] == 1
    assert core.bank_houses == 31


def test_trade_observation_discrimination():
    """Validates that target agent receives 500-dim obs that distinctly encodes trade terms & strategic value."""
    core = MonopolyCore(seed=42)
    core.property_owner[1] = 3
    core.property_owner[3] = 0
    core.property_owner[11] = 3
    core.property_owner[13] = 0
    core.property_owner[14] = 0
    core.property_owner[39] = 3

    # Case A: Generous mutual monopoly trade (P0 offers Baltic #3 for Pink #11)
    core.phase = PHASE_TRADE_RESPONSE
    core.active_decision_player = 3
    core.pending_trade = {
        "initiator": 0, "target": 3,
        "offer_prop": 3, "offer_cash": 0,
        "req_prop": 11, "req_cash": 0,
    }
    obs_good = core.get_actor_observation(3)
    assert obs_good.shape == (500,)
    trade_feats_good = core.get_trade_features(3)
    assert trade_feats_good.shape == (101,)
    assert trade_feats_good[-8] == 1.0, "completes_my_monopoly should be 1.0"
    assert trade_feats_good[-7] == 1.0, "completes_opp_monopoly should be 1.0"
    assert trade_feats_good[-4] == 1.0, "escapes_starvation should be 1.0"
    assert trade_feats_good[-3] == 1.0, "is_mutual_monopoly_trade should be 1.0"

    # Case B: Robbery trade (P0 offers nothing, requests Boardwalk #39 for $0)
    core.pending_trade = {
        "initiator": 0, "target": 3,
        "offer_prop": 40, "offer_cash": 0,
        "req_prop": 39, "req_cash": 0,
    }
    obs_robbery = core.get_actor_observation(3)
    trade_feats_robbery = core.get_trade_features(3)
    assert trade_feats_robbery[-8] == 0.0, "completes_my_monopoly should be 0.0"
    assert trade_feats_robbery[-4] == 0.0, "escapes_starvation should be 0.0"
    assert trade_feats_robbery[-1] < -0.3, f"net_equity_gain should be negative, got {trade_feats_robbery[-1]}"

    assert np.linalg.norm(obs_good - obs_robbery) > 1.5


def test_trade_target_auto_routing():
    """Validates that requesting a property auto-routes trade proposal to its true owner."""
    core = MonopolyCore(seed=42)
    core.property_owner[1] = 0
    core.property_owner[3] = 3
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0

    act_trade = {
        "core": ACT_CORE_ROLL_PASS,
        "auction": 0,
        "property": 160,
        "trade_target": 0,
        "trade_offer_prop": 40,
        "trade_offer_cash": 2,
        "trade_req_prop": 3,
        "trade_req_cash": 0,
        "trade_response": 0,
    }
    core.step({"player_0": act_trade})

    assert core.phase == PHASE_TRADE_RESPONSE
    assert core.active_decision_player == 3, "Must auto-route to Player 3 (owner of prop 3)"
    assert core.pending_trade["target"] == 3


def test_trade_acceptance_reward_surge_when_starving():
    """Validates that accepting a monopoly-completing trade yields massive PBRS reward surge vs 0 on reject."""
    core = MonopolyCore(seed=42)
    core.round_count = 40
    for idx, p in enumerate(PURCHASABLE_TILES):
        core.property_owner[p] = idx % 3
    core.property_owner[1] = 3
    core.property_owner[3] = 0
    core.prev_potentials = [core._compute_potential(pl) for pl in range(4)]

    core.phase = PHASE_TRADE_RESPONSE
    core.active_decision_player = 3
    core.pending_trade = {
        "initiator": 0, "target": 3,
        "offer_prop": 3, "offer_cash": 0,
        "req_prop": 40, "req_cash": 0,
    }

    act_accept = {
        "core": 0, "auction": 0, "property": 160,
        "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
        "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 1,
    }
    _, rewards, _, _ = core.step({"player_3": act_accept})
    assert rewards["player_3"] > 4.0, f"Expected > 4.0 reward surge on accept (erasing -3.5 starvation + monopoly), got {rewards['player_3']}"


def test_management_executes_property_build_before_trade():
    """Validates that valid physical property builds execute before trade proposals on same step."""
    core = MonopolyCore(seed=42)
    core.property_owner[1] = 0
    core.property_owner[3] = 0
    core.property_owner[11] = 1
    core.cash[0] = 2000
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0

    act = {
        "core": ACT_CORE_ROLL_PASS,
        "auction": 0,
        "property": 1,
        "trade_target": 0,
        "trade_offer_prop": 40,
        "trade_offer_cash": 1,
        "trade_req_prop": 11,
        "trade_req_cash": 0,
        "trade_response": 0,
    }
    core.step({"player_0": act})

    assert core.property_houses[1] == 1
    assert core.bank_houses == 31
    assert core.phase == PHASE_PROPERTY_MANAGE


def test_get_observations_zero_redundancy_equivalence():
    """Validates that optimized get_observations() produces bit-for-bit identical vectors to direct getters."""
    core = MonopolyCore(seed=123)
    core.property_owner[1] = 0
    core.property_owner[3] = 1
    core.phase = PHASE_TRADE_RESPONSE
    core.active_decision_player = 1
    core.pending_trade = {
        "initiator": 0, "target": 1,
        "offer_prop": 1, "offer_cash": 50,
        "req_prop": 3, "req_cash": 0,
    }
    obs_bundle = core.get_observations()
    for p in range(4):
        assert np.array_equal(obs_bundle["obs"][f"player_{p}"], core.get_actor_observation(p))
    assert np.array_equal(obs_bundle["global_state"], core.get_global_state())


def test_phase_aware_actor_log_prob_isolation():
    """Validates that unused heads do not pollute log_prob or entropy in single-head phases."""
    core = MonopolyCore(seed=42)
    core.phase = PHASE_BUY_OR_AUCTION
    core.landed_property = 6
    core.active_decision_player = 0

    actor = ActorNetwork(obs_dim=500)
    obs_np = core.get_actor_observation(0)
    masks = core.get_action_masks(0)
    obs_t = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
    masks_t = {k: torch.tensor(masks[k], dtype=torch.int8).unsqueeze(0) for k in masks}

    with torch.no_grad():
        actions, lp, ent = actor(obs_t, masks_t)
        # During PHASE_BUY_OR_AUCTION, only core head is active; entropy must be strictly <= ln(2) ~ 0.69315
        assert ent.item() <= 0.70, f"Expected single-head entropy <= ln(2), got {ent.item()}"

        # Corrupting unused trade head weights should NOT change evaluate_actions output for BUY_OR_AUCTION!
        actor.head_trade_offer_prop.weight.data.fill_(5.0)
        lp2, ent2 = actor.evaluate_actions(obs_t, masks_t, actions)
        assert torch.allclose(lp, lp2, atol=1e-6), "Unused trade head altered BUY_OR_AUCTION log_prob!"
        assert torch.allclose(ent, ent2, atol=1e-6), "Unused trade head altered BUY_OR_AUCTION entropy!"
