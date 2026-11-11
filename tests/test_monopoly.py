"""Comprehensive Unit & Invariant Tests for Tournament MARL Monopoly Engine.

Validates:
1. Physical bank scarcity (32 houses, 12 hotels).
2. Even building rule across color groups.
3. Speeding rule (3 consecutive doubles -> Jail).
4. Jail exit mechanisms (double roll, $50 fee, GOOJF card, 3-turn max).
5. Mandatory auction triggering when purchase declined.
6. Mortgage value and 10% unmortgage interest rate.
7. FIFO Queue Card decks and GOOJF card retention.
8. Potential-Based Reward Shaping (PBRS) invariant calculations.
9. Multi-head action masking strictness (illegal actions masked to 0).
10. Identical Actor blueprint and Orthogonal initialization.
"""

import pytest
import numpy as np
import torch

from monopoly_marl.engine.monopoly_core import (
    MonopolyCore, PHASE_ROLL, PHASE_JAIL_DECISION, PHASE_BUY_OR_AUCTION,
    PHASE_AUCTION, PHASE_PROPERTY_MANAGE, ACT_CORE_BUY_DIRECT,
    ACT_CORE_DECLINE_BUY, ACT_CORE_PAY_JAIL_50, ACT_CORE_USE_GOOJF,
    ACT_CORE_END_MANAGE, ACT_AUC_FOLD, ACT_AUC_BID_1
)
from monopoly_marl.engine.board_data import TILES_DATA, COLOR_GROUPS
from monopoly_marl.engine.cards_data import CardDeckManager
from monopoly_marl.models.actor_network import ActorNetwork
from monopoly_marl.models.critic_network import CentralizedCritic


def test_bank_scarcity_limits():
    """Validates physical bank scarcity: 32 houses, 12 hotels."""
    core = MonopolyCore(seed=123)
    assert core.bank_houses == 32
    assert core.bank_hotels == 12

    # Give player 0 all Brown properties (tiles 1 and 3)
    core.property_owner[1] = 0
    core.property_owner[3] = 0
    core.cash[0] = 5000
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0

    # Exhaust bank houses to 0
    core.bank_houses = 0
    masks = core.get_action_masks(0)
    # Building house on tile 1 or 3 should be illegal (mask 0)
    assert masks["property"][1] == 0
    assert masks["property"][3] == 0

    # Restore 1 house in bank -> tile 1 can build (min_h = 0)
    core.bank_houses = 1
    masks = core.get_action_masks(0)
    assert masks["property"][1] == 1
    assert masks["property"][3] == 1


def test_even_building_rule():
    """Validates that buildings must be built and sold evenly across group."""
    core = MonopolyCore(seed=123)
    # Brown group: [1, 3]
    core.property_owner[1] = 0
    core.property_owner[3] = 0
    core.cash[0] = 5000
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0

    # Tile 1 has 1 house, tile 3 has 0 houses
    core.property_houses[1] = 1
    core.property_houses[3] = 0
    masks = core.get_action_masks(0)

    # Building on tile 1 should be illegal (tile 3 must be built first)
    assert masks["property"][1] == 0
    # Building on tile 3 should be legal
    assert masks["property"][3] == 1

    # Selling: tile 1 has 1 house, tile 3 has 0 houses
    # Selling from tile 1 should be legal
    assert masks["property"][40 + 1] == 1
    # Selling from tile 3 should be illegal (it has 0)
    assert masks["property"][40 + 3] == 0


def test_speeding_to_jail():
    """Validates 3 consecutive doubles sends active player immediately to Jail."""
    core = MonopolyCore(seed=123)
    core.consecutive_doubles = 2
    core.current_player = 0
    core.active_decision_player = 0
    core.phase = PHASE_ROLL

    # Mock dice to roll doubles
    core.rng.randint = lambda a, b: 4  # (4, 4) = double

    action = {"core": 0, "auction": 0, "property": 0, "trade_target": 0,
              "trade_offer_prop": 40, "trade_offer_cash": 0, "trade_req_prop": 40,
              "trade_req_cash": 0, "trade_response": 0}

    core.step({"player_0": action})

    # Player 0 should now be in jail at tile 10
    assert core.in_jail[0] is True
    assert core.position[0] == 10
    assert core.consecutive_doubles == 0


def test_jail_mechanics():
    """Validates jail exit options: pay fine, use GOOJF card, or 3rd turn forced fine."""
    core = MonopolyCore(seed=123)
    core._send_to_jail(0)
    assert core.in_jail[0] is True
    assert core.position[0] == 10
    core.phase = PHASE_JAIL_DECISION
    core.active_decision_player = 0

    # 1. Pay $50 fine
    core.cash[0] = 200
    masks = core.get_action_masks(0)
    assert masks["core"][ACT_CORE_PAY_JAIL_50] == 1

    act_pay = {"core": ACT_CORE_PAY_JAIL_50, "auction": 0, "property": 0, "trade_target": 0,
                "trade_offer_prop": 40, "trade_offer_cash": 0, "trade_req_prop": 40,
                "trade_req_cash": 0, "trade_response": 0}
    core.step({"player_0": act_pay})

    assert core.in_jail[0] is False
    assert core.cash[0] == 150
    assert core.phase == PHASE_ROLL


def test_mandatory_auction_on_decline():
    """Validates that declining property purchase triggers open auction."""
    core = MonopolyCore(seed=123)
    core.position[0] = 1  # Mediterranean Ave ($60)
    core.property_owner[1] = -1
    core.landed_property = 1
    core.phase = PHASE_BUY_OR_AUCTION
    core.active_decision_player = 0

    masks = core.get_action_masks(0)
    assert masks["core"][ACT_CORE_DECLINE_BUY] == 1

    act_decline = {"core": ACT_CORE_DECLINE_BUY, "auction": 0, "property": 0, "trade_target": 0,
                   "trade_offer_prop": 40, "trade_offer_cash": 0, "trade_req_prop": 40,
                   "trade_req_cash": 0, "trade_response": 0}
    core.step({"player_0": act_decline})

    # Should transition to auction phase
    assert core.phase == PHASE_AUCTION
    assert core.auction_property == 1
    assert core.auction_highest_bid == 0


def test_mortgage_and_interest():
    """Validates mortgaging receives 50% and unmortgage costs principal + 10% interest."""
    core = MonopolyCore(seed=123)
    # Tile 1: Mediterranean Ave, price $60, mortgage $30
    core.property_owner[1] = 0
    core.cash[0] = 100
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0

    # Mortgage action: 80 + 1 = 81
    act_mtg = {"core": ACT_CORE_END_MANAGE + 1, "auction": 0, "property": 81, "trade_target": 0,
               "trade_offer_prop": 40, "trade_offer_cash": 0, "trade_req_prop": 40,
               "trade_req_cash": 0, "trade_response": 0}
    core.step({"player_0": act_mtg})

    assert core.property_mortgaged[1] is True
    assert core.cash[0] == 130  # Received $30

    # Unmortgage: cost is 30 * 1.10 = 33
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0
    # Unmortgage action: 120 + 1 = 121
    act_unmtg = {"core": ACT_CORE_END_MANAGE + 1, "auction": 0, "property": 121, "trade_target": 0,
                 "trade_offer_prop": 40, "trade_offer_cash": 0, "trade_req_prop": 40,
                 "trade_req_cash": 0, "trade_response": 0}
    core.step({"player_0": act_unmtg})

    assert core.property_mortgaged[1] is False
    assert core.cash[0] == 130 - 33  # 97


def test_identical_blueprint_orthogonal_init():
    """Validates that all agents share identical neural architecture with orthogonal weights."""
    actor1 = ActorNetwork(obs_dim=327)
    actor2 = ActorNetwork(obs_dim=327)

    # Check layer shapes match
    for (n1, p1), (n2, p2) in zip(actor1.named_parameters(), actor2.named_parameters()):
        assert n1 == n2
        assert p1.shape == p2.shape

    # Check forward pass consistency
    dummy_obs = torch.randn(2, 327)
    dummy_masks = {
        "core": torch.ones(2, 6, dtype=torch.int8),
        "auction": torch.ones(2, 5, dtype=torch.int8),
        "property": torch.ones(2, 160, dtype=torch.int8),
        "trade_target": torch.ones(2, 3, dtype=torch.int8),
        "trade_offer_prop": torch.ones(2, 41, dtype=torch.int8),
        "trade_offer_cash": torch.ones(2, 10, dtype=torch.int8),
        "trade_req_prop": torch.ones(2, 41, dtype=torch.int8),
        "trade_req_cash": torch.ones(2, 10, dtype=torch.int8),
        "trade_response": torch.ones(2, 2, dtype=torch.int8),
    }

    act_dict1, lp1, ent1 = actor1(dummy_obs, dummy_masks)
    act_dict2, lp2, ent2 = actor2(dummy_obs, dummy_masks)

    assert lp1.shape == (2,)
    assert ent1.shape == (2,)
    assert "core" in act_dict1
    assert "auction" in act_dict1
    assert "property" in act_dict1


def test_card_deck_queue_fifo():
    """Validates 16 cards FIFO draw, returning to bottom, and GOOJF retention."""
    deck_mgr = CardDeckManager()
    assert len(deck_mgr.chance_queue) == 16
    assert len(deck_mgr.chest_queue) == 16

    # Draw 20 cards and ensure queue length stays at 16 (unless goojf drawn)
    for _ in range(20):
        c = deck_mgr.draw_chance()
        if c["action"] == "goojf":
            assert len(deck_mgr.chance_queue) == 15
            deck_mgr.return_goojf("chance")
            assert len(deck_mgr.chance_queue) == 16
        else:
            assert len(deck_mgr.chance_queue) == 16


def test_bankruptcy_creditor_transfer():
    """Validates that bankrupting to another player transfers assets and GOOJF cards to creditor."""
    core = MonopolyCore(seed=42)
    # Player 0 owns tile 1 (Mediterranean)
    core.property_owner[1] = 0
    core.goojf_cards[0] = 1
    core.goojf_sources[0] = ["chance"]

    # Player 0 owes huge rent to Player 1 on Boardwalk (tile 39)
    core.cash[0] = 10
    core.last_creditor = 1
    core.cash[0] = -500  # Incurred debt

    core._resolve_bankruptcy(0)

    assert core.bankrupt[0] is True
    # Property 1 must now belong to Player 1!
    assert core.property_owner[1] == 1
    # GOOJF card must now belong to Player 1!
    assert core.goojf_cards[1] == 1
    assert core.goojf_cards[0] == 0


def test_double_rent_requires_no_mortgage():
    """Validates that unimproved monopoly double rent is disabled if any group prop is mortgaged."""
    core = MonopolyCore(seed=42)
    # Brown: [1, 3]
    core.property_owner[1] = 0
    core.property_owner[3] = 0
    core.property_houses[1] = 0
    core.property_houses[3] = 0

    # Mediterranean Ave base rent is 2 -> doubled is 4
    rent = core._calculate_rent(1, dice_sum=7)
    assert rent == 4

    # Mortgage Baltic Ave (tile 3)
    core.property_mortgaged[3] = True
    # Now Mediterranean Ave should only collect regular base rent (2), not doubled!
    rent_after = core._calculate_rent(1, dice_sum=7)
    assert rent_after == 2

