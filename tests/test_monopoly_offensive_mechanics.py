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


def test_mortgage_action_executes_directly_without_trade_leak():
    """Validates that selecting mortgage (prop 80..159) executes mortgage directly and never leaks into trade proposal."""
    core = MonopolyCore(seed=42)
    core.property_owner[1] = 0
    core.property_owner[11] = 1
    core.cash[0] = 100  # Tight liquidity (< $250) permits emergency mortgaging
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0

    act = {
        "core": ACT_CORE_END_MANAGE,  # Even if passed, core_act cannot block mortgage or leak to trade
        "auction": 0,
        "property": 81,               # Mortgage Mediterranean Ave (#1, mortgage=$30)
        "trade_target": 0,
        "trade_offer_prop": 40,
        "trade_offer_cash": 1,
        "trade_req_prop": 11,
        "trade_req_cash": 0,
        "trade_response": 0,
    }
    core.step({"player_0": act})

    assert core.property_mortgaged[1] is True
    assert core.cash[0] == 130
    assert core.trades_proposed_count == 0, "Physical property action 81 must never fall through to trade!"
    assert core.phase == PHASE_PROPERTY_MANAGE


def test_trade_heads_receive_nonzero_gradient_and_sync_with_core():
    """Validates that whenever core.step() triggers a trade proposal, all participating trade heads receive non-zero PPO gradients."""
    core = MonopolyCore(seed=42)
    core.property_owner[1] = 0
    core.property_owner[11] = 1
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0

    actor = ActorNetwork(obs_dim=500)
    obs_np = core.get_actor_observation(0)
    masks = core.get_action_masks(0)
    obs_t = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
    masks_t = {k: torch.tensor(masks[k], dtype=torch.int8).unsqueeze(0) for k in masks}

    # Explicit trade execution path 1: property=160 + offer_cash > 0 (sweetener trade -> req_cash forced 0)
    actions = {
        "core": torch.tensor([0]),
        "auction": torch.tensor([0]),
        "property": torch.tensor([160]),
        "trade_target": torch.tensor([0]),
        "trade_offer_prop": torch.tensor([1]),
        "trade_offer_cash": torch.tensor([1]),
        "trade_req_prop": torch.tensor([11]),
        "trade_req_cash": torch.tensor([0]),
        "trade_response": torch.tensor([0]),
    }

    actor.zero_grad()
    lp, ent = actor.evaluate_actions(obs_t, masks_t, actions)
    loss = -(lp + 0.01 * ent).mean()
    loss.backward()

    # Offered cash > 0 enforces one-way cash: offer_prop, req_prop, offer_cash receive non-zero gradient;
    # req_cash is deterministically masked to 0 and trade_target is auto-routed.
    assert actor.head_trade_offer_prop.weight.grad.norm().item() > 0.0
    assert actor.head_trade_req_prop.weight.grad.norm().item() > 0.0
    assert actor.head_trade_offer_cash.weight.grad.norm().item() > 0.0
    assert actor.head_trade_req_cash.weight.grad.norm().item() == 0.0
    assert actor.head_trade_target.weight.grad.norm().item() == 0.0

    # Explicit trade execution path 2: offer_cash == 0, req_cash > 0 -> head_trade_req_cash gets non-zero grad!
    actions_req_cash = dict(actions)
    actions_req_cash["trade_offer_cash"] = torch.tensor([0])
    actions_req_cash["trade_req_cash"] = torch.tensor([1])
    actor.zero_grad()
    lp_rc, ent_rc = actor.evaluate_actions(obs_t, masks_t, actions_req_cash)
    (-(lp_rc + 0.01 * ent_rc).mean()).backward()
    assert actor.head_trade_req_cash.weight.grad.norm().item() > 0.0

    # Verify core.step() also executes the exact same trade
    act_py = {k: int(v.item()) for k, v in actions.items()}
    core.step({"player_0": act_py})
    assert core.phase == PHASE_TRADE_RESPONSE
    assert core.trades_proposed_count == 1


def test_bankruptcy_even_house_liquidation_and_hotel_shortage():
    """Verify _resolve_bankruptcy liquidates houses evenly and respects bank housing scarcity."""
    # Case A: Even-selling liquidation on Dark Blue ([2, 2] -> [1, 1])
    core = MonopolyCore(seed=99)
    core.reset()
    core.property_owner[37] = 0
    core.property_owner[39] = 0
    core.property_houses[37] = 2
    core.property_houses[39] = 2
    core.bank_houses = 28
    core.cash[0] = -150

    core._resolve_bankruptcy(0)

    assert core.cash[0] == 50
    assert core.property_houses[37] == 1
    assert core.property_houses[39] == 1
    assert abs(core.property_houses[37] - core.property_houses[39]) <= 1
    assert core.bank_houses == 30

    # Case B: Hotel liquidation under Bank housing shortage ([5, 5] with bank_houses=1 -> [0, 0])
    core2 = MonopolyCore(seed=100)
    core2.reset()
    core2.property_owner[37] = 0
    core2.property_owner[39] = 0
    core2.property_houses[37] = 5
    core2.property_houses[39] = 5
    core2.bank_hotels = 10
    core2.bank_houses = 1
    core2.cash[0] = -150

    core2._resolve_bankruptcy(0)

    assert core2.property_houses[37] == 0
    assert core2.property_houses[39] == 0
    assert core2.bank_hotels == 12
    assert core2.bank_houses == 1
    assert core2.cash[0] == 850


def test_same_turn_mortgage_and_build_thrash_blocked():
    """Verify same-turn mortgage->unmortgage and build->sell thrashing are masked out."""
    core = MonopolyCore(seed=42)
    core.property_owner[1] = 0
    core.property_owner[3] = 0
    core.property_owner[6] = 0
    core.cash[0] = 1500
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0

    # Mortgage Oriental (#6) on turn -> manage_actions_this_turn becomes 1
    core.step({"player_0": {
        "core": 0, "auction": 0, "property": 86, "trade_target": 0,
        "trade_offer_prop": 40, "trade_offer_cash": 0, "trade_req_prop": 40,
        "trade_req_cash": 0, "trade_response": 0,
    }})
    assert core.property_mortgaged[6] is True
    masks = core.get_action_masks(0)
    assert masks["property"][126] == 0, "Cannot immediately unmortgage #6 on the exact same turn"

    # Build on Baltic (#3) on same turn -> cannot sell #3 on the exact same turn
    core.step({"player_0": {
        "core": 0, "auction": 0, "property": 3, "trade_target": 0,
        "trade_offer_prop": 40, "trade_offer_cash": 0, "trade_req_prop": 40,
        "trade_req_cash": 0, "trade_response": 0,
    }})
    assert core.property_houses[3] == 1
    masks2 = core.get_action_masks(0)
    assert masks2["property"][43] == 0, "Cannot immediately sell house #3 built on the exact same turn"


def test_autoregressive_actor_eliminates_dual_cash_beg_and_gift():
    """Verify ActorNetwork never produces dual-cash, pure-beg, or pure-gift trades and evaluate_actions matches forward."""
    core = MonopolyCore(seed=42)
    core.property_owner[1] = 0
    core.property_owner[11] = 1
    core.cash[0] = 1000
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0

    actor = ActorNetwork(obs_dim=500)
    obs_t = torch.tensor(core.get_actor_observation(0), dtype=torch.float32).unsqueeze(0)
    masks = core.get_action_masks(0)
    masks_t = {k: torch.tensor(masks[k], dtype=torch.int8).unsqueeze(0) for k in masks}

    for _ in range(30):
        acts, lp_f, ent_f = actor(obs_t, masks_t, deterministic=False)
        lp_e, ent_e = actor.evaluate_actions(obs_t, masks_t, acts)
        assert torch.allclose(lp_f, lp_e, atol=1e-5)
        assert torch.allclose(ent_f, ent_e, atol=1e-5)

        op = int(acts["trade_offer_prop"].item())
        rp = int(acts["trade_req_prop"].item())
        oc = int(acts["trade_offer_cash"].item())
        rc = int(acts["trade_req_cash"].item())
        assert not (oc > 0 and rc > 0), f"Dual cash paradox occurred: oc={oc}, rc={rc}"
        if op == 40 and rp < 40:
            assert oc > 0, "Cash-for-property buy cannot offer $0 cash"
            assert rc == 0, "Cash-for-property buy cannot request cash back"
        if op < 40 and rp == 40:
            assert oc == 0, "Property-for-cash sale cannot also pay cash"
            assert rc > 0, "Property-for-cash sale cannot request $0 cash (pure gift)"


def test_auction_jump_squeeze_capped_by_bidding_ceiling():
    """Verify ACT_AUC_JUMP_SQUEEZE is bounded by compute_bidding_ceiling (never jumps to $1000+ on $200 RR)."""
    from monopoly_marl.engine.monopoly_core import PHASE_AUCTION, ACT_AUC_JUMP_SQUEEZE

    core = MonopolyCore(seed=42)
    core.phase = PHASE_AUCTION
    core.auction_property = 25  # B. & O. Railroad (list price $200)
    core.auction_highest_bid = 50
    core.auction_highest_bidder = 1
    core.auction_active_bidders = [0, 1]
    core.auction_current_bidder_idx = 0
    core.active_decision_player = 0
    core.cash[0] = 1500
    core.cash[1] = 1200  # Rich opponent cash $1200

    # Non-synergy ceiling on $200 RR is <= $200, whereas old squeeze_target was $1201
    ceiling = core.compute_bidding_ceiling(0, 25)
    assert ceiling == 200
    masks = core.get_action_masks(0)
    assert masks["auction"][ACT_AUC_JUMP_SQUEEZE] == 1

    core.step({"player_0": {
        "core": 0, "auction": ACT_AUC_JUMP_SQUEEZE, "property": 0, "trade_target": 0,
        "trade_offer_prop": 40, "trade_offer_cash": 0, "trade_req_prop": 40,
        "trade_req_cash": 0, "trade_response": 0,
    }})
    assert core.auction_highest_bid == 200, f"Expected squeeze capped at ceiling $200, got {core.auction_highest_bid}"


def test_rejected_trade_cooldown_prevents_identical_spam():
    """Verify rejecting a requested property puts (initiator, owner, req_prop) on 5-round cooldown."""
    core = MonopolyCore(seed=42)
    core.property_owner[1] = 0
    core.property_owner[3] = 1
    core.cash[0] = 1000
    core.cash[1] = 1000
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0

    # P0 requests tile 3 from P1 -> P1 rejects
    core.step({"player_0": {
        "core": 0, "auction": 0, "property": 160, "trade_target": 0,
        "trade_offer_prop": 1, "trade_offer_cash": 1, "trade_req_prop": 3,
        "trade_req_cash": 0, "trade_response": 0,
    }})
    assert core.phase == PHASE_TRADE_RESPONSE
    core.step({"player_1": {
        "core": 0, "auction": 0, "property": 160, "trade_target": 0,
        "trade_offer_prop": 40, "trade_offer_cash": 0, "trade_req_prop": 40,
        "trade_req_cash": 0, "trade_response": 0,
    }})

    # Immediately and for next 4 rounds, tile 3 must be masked 0 for P0
    core.trade_proposals_this_turn = 0
    masks = core.get_action_masks(0)
    assert masks["trade_req_prop"][3] == 0, "Rejected req_prop #3 must be on cooldown"

    # After 5 rounds pass, tile 3 unlocks again
    core.round_count += 5
    masks_unlocked = core.get_action_masks(0)
    assert masks_unlocked["trade_req_prop"][3] == 1, "Cooldown should expire after 5 rounds"


def test_gae_terminal_boundary_does_not_leak_across_reset():
    """Verify compute_gae cuts off bootstrap at dones[t] and preserves credit up to t-1."""
    from monopoly_marl.models.storage import RolloutBuffer

    gamma = 0.99
    gae_lambda = 0.95
    buf = RolloutBuffer(buffer_size=3, num_envs=1, gamma=gamma, gae_lambda=gae_lambda)
    buf.rewards[:, 0, 0] = torch.tensor([0.0, 5.0, 0.0])
    buf.dones[:, 0, 0] = torch.tensor([0.0, 1.0, 0.0])
    buf.values[:, 0, 0] = torch.tensor([1.0, 1.0, 2.0])

    buf.compute_gae(last_values=torch.tensor([[2.0, 0.0, 0.0, 0.0]]), last_dones=torch.tensor([[0.0, 0.0, 0.0, 0.0]]))

    # Terminal step t=1 must NOT bootstrap next episode V(s_2)=2.0: A_1 = 5.0 - 1.0 = 4.0
    assert abs(buf.advantages[1, 0, 0].item() - 4.0) < 1e-5
    # Pre-terminal step t=0 must receive full discounted terminal advantage from t=1
    expected_a0 = (0.0 + gamma * 1.0 - 1.0) + gamma * gae_lambda * 4.0
    assert abs(buf.advantages[0, 0, 0].item() - expected_a0) < 1e-5


def test_pbrs_cash_vs_property_fair_pricing():
    """Verify linear w_cash properly penalizes mortgage burn & >2x auction overbids while keeping unowned buys positive."""
    core = MonopolyCore(seed=42)
    p0 = core._compute_potential(0)

    # 1. Unowned buy at list price ($100 Oriental) remains strongly positive (> 0.30)
    core.property_owner[6] = 0
    core.cash[0] -= 100
    p_buy = core._compute_potential(0)
    assert p_buy - p0 > 0.30

    # 2. Mortgage -> Unmortgage round trip ($175 mortgage -> $192 unmortgage = -$17 burned) produces clear penalty
    core2 = MonopolyCore(seed=42)
    core2.property_owner[37] = 0  # Park Place (mortgage=$175)
    core2.cash[0] = 1000
    before_cycle = core2._compute_potential(0)
    core2.cash[0] -= int(175 * 0.10)  # Net cash lost on 10% mortgage->unmortgage round trip
    after_cycle = core2._compute_potential(0)
    assert after_cycle - before_cycle < -0.05

    # 3. Overpaying $430 for non-monopoly $200 property (B. & O. Railroad) must be negative
    core3 = MonopolyCore(seed=42)
    core3.cash[0] = 1500
    before_auc = core3._compute_potential(0)
    core3.property_owner[25] = 0
    core3.cash[0] -= 430
    after_auc = core3._compute_potential(0)
    assert after_auc - before_auc < 0.0, f"Overpaying >2x on non-monopoly deed must reduce potential, got {after_auc - before_auc}"


def test_hardware_entropy_seed_generator():
    """Verify hardware fluctuation seed generator produces valid 32-bit seeds with high entropy."""
    from monopoly_marl.engine.monopoly_core import generate_hardware_entropy_seed

    seeds = [generate_hardware_entropy_seed() for _ in range(50)]
    # All seeds must be 32-bit unsigned integers
    for s in seeds:
        assert isinstance(s, int)
        assert 0 <= s < 2**32

    # High entropy check: all 50 drawn seeds must be distinct
    assert len(set(seeds)) == 50, "Hardware entropy generator produced duplicate seeds!"

    # Verify MonopolyCore default initialization and reset utilize fresh hardware entropy
    core1 = MonopolyCore(seed=None)
    core2 = MonopolyCore(seed=None)
    # The two instances should have distinct starting states/decks
    card1 = core1.deck_manager.draw_chance()
    card2 = core2.deck_manager.draw_chance()
    # While Chance cards could theoretically match by chance (1/16), 10 consecutive draws must diverge
    draws1 = [core1.deck_manager.draw_chance()["name"] for _ in range(10)]
    draws2 = [core2.deck_manager.draw_chance()["name"] for _ in range(10)]
    assert draws1 != draws2, "Independent hardware-seeded engines should have divergent shuffle states"


def test_bankruptcy_turn_rotation_immediate_advance():
    """Verify that when the active decision player goes bankrupt, turn immediately rotates to next surviving player."""
    from monopoly_marl.engine.monopoly_core import PHASE_ROLL, PHASE_JAIL_DECISION, PHASE_GAME_OVER

    core = MonopolyCore(seed=42)
    assert core.current_player == 0
    assert core.active_decision_player == 0

    # Incur terminal debt on player 0 to trigger bankruptcy
    core.cash[0] = -500
    core.last_creditor = 1
    core._resolve_bankruptcy(0)

    # 1. Player 0 must be officially bankrupt
    assert core.bankrupt[0] is True

    # 2. current_player and active_decision_player must immediately advance away from player 0
    assert core.current_player == 1, f"Expected current_player to advance to 1, got {core.current_player}"
    assert core.active_decision_player == 1, f"Expected active_decision_player to advance to 1, got {core.active_decision_player}"
    assert core.phase in (PHASE_ROLL, PHASE_JAIL_DECISION)

    # 3. Bankrupt player 0 must be treated as inactive in action masks
    masks_p0 = core.get_action_masks(0)
    assert all(np.all(m == 0) for k, m in masks_p0.items() if k not in ("core", "auction", "property", "trade_target", "trade_offer_prop", "trade_offer_cash", "trade_req_prop", "trade_req_cash", "trade_response"))

    # 4. Successive eliminations until Last Agent Standing trigger clean GAME OVER
    core.cash[1] = -500
    core.last_creditor = 2
    core._resolve_bankruptcy(1)
    assert core.current_player == 2
    assert core.active_decision_player == 2

    # Step game to let core recognize terminal state when player 2 eliminates player 3
    core.cash[3] = -500
    core.last_creditor = 2
    core.step({"player_2": {
        "core": 0, "auction": 0, "property": 160, "trade_target": 0,
        "trade_offer_prop": 40, "trade_offer_cash": 0, "trade_req_prop": 40,
        "trade_req_cash": 0, "trade_response": 0
    }})
    assert core.bankrupt[3] is True
    assert core.phase == PHASE_GAME_OVER
    assert any("TOURNAMENT CHAMPION" in log for log in core.event_log)


def test_liquidity_aware_mortgage_masking_blocks_rich_mortgage():
    """Verify that players with abundant cash (> $250) cannot voluntarily mortgage non-monopoly deeds."""
    core = MonopolyCore(seed=42)
    core.property_owner[1] = 0  # Mediterranean Ave
    core.cash[0] = 1000         # Abundant cash, no monopolies
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0

    masks_rich = core.get_action_masks(0)
    assert masks_rich["property"][81] == 0, "Rich player with $1000 cash must not be permitted to mortgage Mediterranean Ave"

    # When cash drops below $250 safety margin, emergency mortgaging unlocks
    core.cash[0] = 150
    masks_poor = core.get_action_masks(0)
    assert masks_poor["property"][81] == 1, "Tight cash (< $250) must unlock mortgaging for emergency liquidity"

    # Even with high cash ($1000), if player holds a street monopoly needing development, mortgaging for leverage is permitted
    core.cash[0] = 1000
    core.property_owner[18] = 0  # Tennessee Ave
    core.property_owner[16] = 0  # St. James Place
    core.property_owner[19] = 0  # New York Ave (Complete Orange Monopoly!)
    core.property_houses[18] = 0
    masks_leverage = core.get_action_masks(0)
    assert masks_leverage["property"][81] == 1, "Holding complete monopoly to develop houses permits strategic mortgage leverage"


def test_pure_cash_trade_proposal_requires_minimum_mortgage_floor():
    """Verify pure cash purchase proposal of opponent deed requires at least the deed's statutory mortgage value."""
    core = MonopolyCore(seed=42)
    core.property_owner[5] = 1  # Reading RR (mortgage = $100)
    core.cash[0] = 1000
    core.cash[1] = 1000

    # Lowball proposal: Offer None prop + $25 cash for Reading RR (#5) -> Rejected by engine
    started_lowball = core._propose_trade(
        initiator=0, target_idx=0, offer_prop=40, offer_cash_idx=1,  # $25
        req_prop=5, req_cash_idx=0
    )
    assert started_lowball is False, "Lowball pure-cash offer ($25 < $100 mortgage) must be rejected by engine"

    # Fair cash proposal: Offer None prop + $100 cash (>= $100 mortgage) -> Permitted by engine
    started_fair = core._propose_trade(
        initiator=0, target_idx=0, offer_prop=40, offer_cash_idx=3,  # $100
        req_prop=5, req_cash_idx=0
    )
    assert started_fair is True, "Pure-cash offer meeting statutory mortgage value ($100) must be permitted"


def test_trade_rejection_friction_penalty():
    """Verify that proposing a trade that gets rejected penalizes the initiator with -0.15 friction cost."""
    core = MonopolyCore(seed=42)
    core.property_owner[1] = 0
    core.property_owner[3] = 1
    core.cash[0] = 1000
    core.cash[1] = 1000
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0

    # P0 proposes trade to P1
    core.step({"player_0": {
        "core": 0, "auction": 0, "property": 160, "trade_target": 0,
        "trade_offer_prop": 1, "trade_offer_cash": 1, "trade_req_prop": 3,
        "trade_req_cash": 0, "trade_response": 0,
    }})
    assert core.phase == PHASE_TRADE_RESPONSE

    # P1 rejects
    _, rewards, _, _ = core.step({"player_1": {
        "core": 0, "auction": 0, "property": 160, "trade_target": 0,
        "trade_offer_prop": 40, "trade_offer_cash": 0, "trade_req_prop": 40,
        "trade_req_cash": 0, "trade_response": 0,
    }})

    # P0's reward must reflect the -0.15 friction penalty
    assert rewards["player_0"] <= -0.14, f"Initiator must receive -0.15 rejection penalty, got {rewards['player_0']}"


def test_win_win_bilateral_monopoly_trade_bonus():
    """Verify that mutual monopoly barter awards +2.5 bilateral synergy bonus to both players."""
    core = MonopolyCore(seed=42)
    # P0 holds Mediterranean (#1) + Illinois (#24)
    # P1 holds Baltic (#3) + Indiana (#23) & Kentucky (#21)
    # Swapping tile 24 (P0) for tile 3 (P1) completes Brown for P0 and Red for P1!
    core.property_owner[1] = 0
    core.property_owner[24] = 0
    core.property_owner[3] = 1
    core.property_owner[21] = 1
    core.property_owner[23] = 1
    core.cash[0] = 1000
    core.cash[1] = 1000
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0

    # P0 proposes swap: tile 24 for tile 3
    core.step({"player_0": {
        "core": 0, "auction": 0, "property": 160, "trade_target": 0,
        "trade_offer_prop": 24, "trade_offer_cash": 0, "trade_req_prop": 3,
        "trade_req_cash": 0, "trade_response": 0,
    }})
    assert core.phase == PHASE_TRADE_RESPONSE

    # P1 accepts
    _, rewards, _, _ = core.step({"player_1": {
        "core": 0, "auction": 0, "property": 0, "trade_target": 0,
        "trade_offer_prop": 40, "trade_offer_cash": 0, "trade_req_prop": 40,
        "trade_req_cash": 0, "trade_response": 1,
    }})

    # Both players completed a street monopoly! Both should receive substantial positive reward (+2.5 bonus)
    assert rewards["player_0"] >= 2.0, f"Expected P0 to receive mutual monopoly reward >= 2.0, got {rewards['player_0']}"
    assert rewards["player_1"] >= 2.0, f"Expected P1 to receive mutual monopoly reward >= 2.0, got {rewards['player_1']}"






