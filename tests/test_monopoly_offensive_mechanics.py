"""Unit tests for the Offensive Monopoly Learning Redesign.

Validates:
1. Monopoly starvation penalty: quadratic drop in potential when 0 monopolies as unowned tiles vanish.
2. Monopoly surge: massive leap in potential upon completing a street monopoly.
3. Single-head property management: action 160 (PASS) ends phase cleanly.
4. Non-vetoed building: building actions (prop_act < 40) are executed even if core_act is END_MANAGE.
5. Non-preemptive trading: initiator remains in PHASE_PROPERTY_MANAGE after trade to build immediately.
"""

import pytest
import numpy as np

from monopoly_marl.engine.monopoly_core import (
    MonopolyCore, PHASE_PROPERTY_MANAGE, PHASE_ROLL, PHASE_TRADE_RESPONSE,
    ACT_CORE_ROLL_PASS, ACT_CORE_END_MANAGE, PURCHASABLE_TILES
)


def test_monopoly_starvation_and_completion_potential():
    """Validates that holding 0 monopolies imposes escalating starvation penalty, and completing monopoly surges potential."""
    core = MonopolyCore(seed=42)
    
    # Early game: all 28 properties unowned
    pot_early = core._compute_potential(0)
    
    # Mid game: all 28 properties owned by players 1, 2, 3, player 0 has 0 monopolies
    for idx, p in enumerate(PURCHASABLE_TILES):
        core.property_owner[p] = (idx % 3) + 1  # Distributed among opponents
    
    # Player 0 has 0 monopolies, and 0 unowned properties remain
    pot_starving = core._compute_potential(0)
    # Starvation penalty should drop potential by beta_starve (25.0)
    assert pot_starving < pot_early - 20.0, f"Expected deep starvation penalty, got {pot_starving} vs {pot_early}"
    
    # Now give player 0 the Brown monopoly (Mediterranean #1 and Baltic #3)
    core.property_owner[1] = 0
    core.property_owner[3] = 0
    pot_monopoly = core._compute_potential(0)
    
    # Starvation penalty is eliminated and monopoly bonus is awarded!
    assert pot_monopoly > pot_starving + 25.0, f"Expected massive surge upon monopoly completion, got {pot_monopoly} vs {pot_starving}"


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
        "trade_response": 0
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
        "core": ACT_CORE_END_MANAGE,  # Formerly caused instant turn kill
        "auction": 0,
        "property": 1,                # Mediterranean Ave
        "trade_target": 0,
        "trade_offer_prop": 40,
        "trade_offer_cash": 0,
        "trade_req_prop": 40,
        "trade_req_cash": 0,
        "trade_response": 0
    }
    core.step({"player_0": act})
    
    # House should be built!
    assert core.property_houses[1] == 1
    assert core.bank_houses == 31
    assert core.cash[0] == 1950  # Deducted $50 house cost


def test_non_preemptive_trading_allows_immediate_building():
    """Validates that completing a trade does not kill turn, allowing immediate house construction."""
    core = MonopolyCore(seed=42)
    # Player 0 owns Baltic #3, Player 1 owns Mediterranean #1
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
        "trade_target": 0,           # Player 1
        "trade_offer_prop": 40,      # None
        "trade_offer_cash": 2,       # $200
        "trade_req_prop": 1,         # Mediterranean Ave
        "trade_req_cash": 0,         # $0
        "trade_response": 0
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
        "trade_response": 1          # ACCEPT
    }
    core.step({"player_1": act_accept})
    
    # Player 0 now owns both #1 and #3 (complete Brown monopoly)!
    assert core.property_owner[1] == 0
    assert core.property_owner[3] == 0
    
    # Non-preemptive: Player 0 MUST remain in PHASE_PROPERTY_MANAGE!
    assert core.phase == PHASE_PROPERTY_MANAGE
    assert core.active_decision_player == 0
    
    # Player 0 immediately builds a house on Mediterranean Ave on the SAME turn!
    act_build = {
        "core": ACT_CORE_ROLL_PASS,
        "auction": 0,
        "property": 1,
        "trade_target": 0,
        "trade_offer_prop": 40,
        "trade_offer_cash": 0,
        "trade_req_prop": 40,
        "trade_req_cash": 0,
        "trade_response": 0
    }
    core.step({"player_0": act_build})
    
    assert core.property_houses[1] == 1
    assert core.bank_houses == 31
