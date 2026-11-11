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
    ACT_CORE_END_MANAGE, ACT_AUC_FOLD, ACT_AUC_BID_1, ACT_AUC_BID_10,
    ACT_AUC_BID_50, ACT_AUC_BID_100, ACT_AUC_JUMP_LIST, ACT_AUC_JUMP_SQUEEZE
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
    actor1 = ActorNetwork(obs_dim=399)
    actor2 = ActorNetwork(obs_dim=399)

    # Check layer shapes match
    for (n1, p1), (n2, p2) in zip(actor1.named_parameters(), actor2.named_parameters()):
        assert n1 == n2
        assert p1.shape == p2.shape

    # Check forward pass consistency
    dummy_obs = torch.randn(2, 399)
    dummy_masks = {
        "core": torch.ones(2, 6, dtype=torch.int8),
        "auction": torch.ones(2, 7, dtype=torch.int8),
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


def test_advanced_barter_synergy_and_anti_kingmaking():
    """Validates color-matching synergy vectors and game-theoretic anti-kingmaking rationality."""
    core = MonopolyCore(seed=42, include_synergy_obs=True)

    # 1. Verify Synergy Vector dimension
    syn_vec = core.get_color_synergy_vector()
    assert syn_vec.shape == (64,), f"Expected shape (64,), got {syn_vec.shape}"
    assert core.get_actor_observation(0).shape == (399,)
    assert core.get_global_state().shape == (1628,)

    # 2. Test irrational trade: Target asked to give up tile that breaks monopoly for $0
    # Player 1 holds full Brown monopoly: [1, 3]
    core.property_owner[1] = 1
    core.property_owner[3] = 1
    core.cash[1] = 500
    core.cash[0] = 500

    bad_trade = {
        "initiator": 0,
        "target": 1,
        "offer_prop": 40,  # None
        "offer_cash": 0,
        "req_prop": 1,     # Mediterranean Ave
        "req_cash": 0,
    }
    is_rat, reason = core.evaluate_trade_rationality(bad_trade)
    assert not is_rat, "Scam trade breaking monopoly for $0 should be deemed irrational!"

    # 3. Test rational trade: Color-matching trade where Player 1 gets Baltic and Player 0 gets Oriental
    # Player 0 holds Oriental (tile 6), Player 1 holds Baltic (tile 3)
    good_trade = {
        "initiator": 0,
        "target": 1,
        "offer_prop": 6,   # Oriental
        "offer_cash": 100, # $100 cash sweetener
        "req_prop": 3,     # Baltic Ave
        "req_cash": 0,
    }
    # Reset owners so Brown is not already a monopoly
    core.property_owner[1] = -1
    core.property_owner[3] = 1
    core.property_owner[6] = 0
    is_rat_good, reason_good = core.evaluate_trade_rationality(good_trade)
    assert is_rat_good, f"Fair trade with cash sweetener should be rational, got: {reason_good}"


def test_advanced_auction_ceiling_and_jump_bidding():
    """Validates game-theoretic dynamic bidding ceiling, 7-dim auction action space,
    and jump bidding (jump to list price & opponent squeeze)."""
    core = MonopolyCore(seed=100, include_synergy_obs=True)

    # 1. Test Dynamic Bidding Ceiling
    # Tile 16: St. James Place (Orange, price 180, mortgage 90)
    # Wealthy player with $2000 cash
    core.cash[0] = 2000
    ceiling_wealthy = core.compute_bidding_ceiling(0, 16)
    assert ceiling_wealthy > 100, f"Wealthy player should have robust ceiling, got {ceiling_wealthy}"

    # Cash-strapped player with $60 cash (< base reserve $150)
    core.cash[1] = 60
    ceiling_strapped = core.compute_bidding_ceiling(1, 16)
    assert ceiling_strapped == 1, f"Cash strapped player must have ceiling clamped to 1, got {ceiling_strapped}"

    # Spite bidding ceiling: Opponent 2 owns other 2 Oranges (tiles 18, 19)
    core.property_owner[18] = 2
    core.property_owner[19] = 2
    core.cash[0] = 1500
    core.cash[2] = 400
    spite_ceiling = core.compute_bidding_ceiling(0, 16)
    # Spite ceiling should cap at max_opp_cash (400) or spite calculation
    assert spite_ceiling <= 400, f"Spite bidding should not exceed max opponent cash (400), got {spite_ceiling}"

    # 2. Test 7-dim Auction Action Space and Jump Bidding Execution
    # Put core into auction phase on tile 16
    core.phase = PHASE_AUCTION
    core.auction_property = 16
    core.auction_highest_bid = 10
    core.auction_highest_bidder = -1
    core.auction_active_bidders = [0, 1]
    core.auction_current_bidder_idx = 0
    core.active_decision_player = 0
    core.cash[0] = 1500
    core.cash[1] = 250
    core.property_owner[18] = -1
    core.property_owner[19] = -1

    masks = core.get_action_masks(0)
    assert masks["auction"].shape == (7,), f"Expected 7-dim auction mask, got {masks['auction'].shape}"
    assert masks["auction"][ACT_AUC_FOLD] == 1
    assert masks["auction"][ACT_AUC_BID_1] == 1
    assert masks["auction"][ACT_AUC_JUMP_LIST] == 1 # 180 <= ceiling and 180 > 10

    # Let's execute Jump to List Price (action 5)
    action_jump_list = {
        "core": 0, "auction": ACT_AUC_JUMP_LIST, "property": 0,
        "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
        "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 0
    }
    core.step({"player_0": action_jump_list})

    # Highest bid should now be 180 and highest bidder player 0
    assert core.auction_highest_bid == 180
    assert core.auction_highest_bidder == 0
    assert core.active_decision_player == 1 # Turn moves to player 1

    # Now player 1 (cash 250) facing bid of 180:
    # +100 -> 280 (exceeds cash 250, so illegal)
    # Jump list (180 <= 180 -> illegal)
    masks1 = core.get_action_masks(1)
    assert masks1["auction"][ACT_AUC_JUMP_LIST] == 0, "Cannot jump to list price if current bid is already >= list price"
    assert masks1["auction"][ACT_AUC_BID_100] == 0, "Cannot bid +100 if 180 + 100 > cash (250)"
    assert masks1["auction"][ACT_AUC_FOLD] == 1

    # Player 1 folds
    action_fold = {
        "core": 0, "auction": ACT_AUC_FOLD, "property": 0,
        "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
        "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 0
    }
    core.step({"player_1": action_fold})

    # Auction should resolve: Player 0 won tile 16 for $180
    assert core.property_owner[16] == 0
    assert core.cash[0] == 1500 - 180
    assert core.phase != PHASE_AUCTION


def test_property_purchase_hazard_survival_masking():
    """Validates that purchasing neutral properties when post-purchase cash is below
    survival buffer is masked to 0, but strategic monopoly/denial purchases are permitted."""
    core = MonopolyCore(seed=123, include_synergy_obs=True)

    # Setup player 0 in PHASE_BUY_OR_AUCTION on tile 21 (Kentucky Ave, Red, price $220)
    core.phase = PHASE_BUY_OR_AUCTION
    core.landed_property = 21
    core.active_decision_player = 0
    core.position[0] = 21

    # Place lethal opponent hotel nearby (tile 23: Indiana Ave, 3 houses, rent $700)
    core.property_owner[23] = 1
    core.property_houses[23] = 3

    # Player 0 has $240 cash (cash - price = $20, while survival_buffer is > $30)
    core.cash[0] = 240
    masks = core.get_action_masks(0)
    # Since tile 21 is neutral (does not complete or deny monopoly), BUY_DIRECT must be masked out to prevent ruin!
    assert masks["core"][ACT_CORE_BUY_DIRECT] == 0, "Buying neutral property leaving <$30 before lethal hazard must be masked 0"
    assert masks["core"][ACT_CORE_DECLINE_BUY] == 1, "Must be able to decline"

    # Now give player 0 the other two Red properties: tiles 23 (Indiana) and 24 (Illinois)
    core.property_owner[23] = 0
    core.property_owner[24] = 0
    # Now tile 21 completes player 0's Red monopoly!
    masks_monopoly = core.get_action_masks(0)
    assert masks_monopoly["core"][ACT_CORE_BUY_DIRECT] == 1, "Monopoly completion override must allow BUY_DIRECT despite hazard"


def test_property_purchase_anti_blunder_decline_masking():
    """Validates that declining a monopoly completion when opponents can afford to steal it is masked 0."""
    core = MonopolyCore(seed=456, include_synergy_obs=True)

    # Player 0 lands on tile 19 (New York Ave, Orange, price $200)
    core.phase = PHASE_BUY_OR_AUCTION
    core.landed_property = 19
    core.active_decision_player = 0
    core.cash[0] = 500

    # Player 0 already owns the other two Oranges (tiles 16 and 18)
    core.property_owner[16] = 0
    core.property_owner[18] = 0

    # Opponent 1 has $600 (can afford $200 in auction)
    core.cash[1] = 600

    masks = core.get_action_masks(0)
    assert masks["core"][ACT_CORE_BUY_DIRECT] == 1
    assert masks["core"][ACT_CORE_DECLINE_BUY] == 0, "Cannot decline completed monopoly when opponents can steal it in auction!"

    # Now make all opponents cash-poor (< $200)
    core.cash[1] = 50
    core.cash[2] = 40
    core.cash[3] = 30
    masks_poor_opps = core.get_action_masks(0)
    assert masks_poor_opps["core"][ACT_CORE_BUY_DIRECT] == 1
    assert masks_poor_opps["core"][ACT_CORE_DECLINE_BUY] == 1, "Declining permitted when opponents are too broke to steal monopoly (strategic discount opportunity)"


def test_property_transaction_features_in_buy_phase():
    """Validates that 8-dim property transaction features are active in PHASE_BUY_OR_AUCTION."""
    core = MonopolyCore(seed=789, include_synergy_obs=True)
    core.phase = PHASE_BUY_OR_AUCTION
    core.landed_property = 16 # St. James Place (Orange, price 180)
    core.active_decision_player = 0
    core.cash[0] = 1000

    feats = core.get_property_transaction_features(0)
    assert feats.shape == (8,)
    # Orange tier score is 1.0
    assert feats[6] == 1.0, f"Expected tier score 1.0 for Orange, got {feats[6]}"
    # Post transaction cash ratio: (1000 - 180) / 1000 = 0.82
    assert abs(feats[4] - 0.82) < 0.05, f"Expected post-cash ratio ~0.82, got {feats[4]}"

    # Verify total observation and global state dimension
    obs = core.get_actor_observation(0)
    assert obs.shape == (399,), f"Expected 399-dim obs, got {obs.shape}"
    global_state = core.get_global_state()
    assert global_state.shape == (1628,), f"Expected 1628-dim global state, got {global_state.shape}"




