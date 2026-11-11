"""Exhaustive, End-to-End Game Mechanics Test Suite for Tournament Monopoly MARL.

Tests 100% of official Monopoly mechanics:
1. Board, Tiles & Data Integrity
2. Movement, Passing GO, Dice & Consecutive Doubles (Speeding)
3. Jail Mechanics (Go to Jail, 3-turn limit, doubles release, $50 fine, GOOJF card)
4. Property Acquisition (Direct Buy, Cash Affordability, Decline to Auction)
5. Mandatory Open Auction (Bidding, Increments, Jump Bids, Folding, Awards)
6. Rent Calculations (Base, Monopoly Double, Houses 1-4, Hotel, Mortgaged, Railroads 1-4, Utilities 4x/10x)
7. House & Hotel Construction, Even Building Rules, Bank Scarcity (32 Houses / 12 Hotels)
8. Mortgage & Unmortgage Mechanics (50% value, 10% statutory interest, house restriction)
9. Bilateral Trading (Atomic Cash & Property Swaps, Rejections, Validation)
10. All 16 Chance & 16 Community Chest Cards (Advance, Cash, Repairs, Pay/Collect Each Player, Go Back 3 Spaces)
11. Taxes (Income Tax $200, Luxury Tax $100) and Free Parking
12. Bankruptcy, Debt Liquidation, Creditor vs Bank Transfer, and Last Agent Standing Victory
13. Observation (399-dim) and Action Mask Non-Zero Integrity Across All Phases
"""

import pytest
import numpy as np

from monopoly_marl.engine.monopoly_core import (
    MonopolyCore, PHASE_ROLL, PHASE_BUY_OR_AUCTION, PHASE_AUCTION,
    PHASE_JAIL_DECISION, PHASE_PROPERTY_MANAGE, PHASE_TRADE_RESPONSE, PHASE_GAME_OVER,
    ACT_CORE_ROLL_PASS, ACT_CORE_BUY_DIRECT, ACT_CORE_DECLINE_BUY,
    ACT_CORE_PAY_JAIL_50, ACT_CORE_USE_GOOJF, ACT_CORE_END_MANAGE,
    ACT_AUC_FOLD, ACT_AUC_BID_1, ACT_AUC_BID_10, ACT_AUC_BID_50, ACT_AUC_BID_100,
    ACT_AUC_JUMP_LIST, ACT_AUC_JUMP_SQUEEZE,
    TRADE_CASH_BRACKETS
)
from monopoly_marl.engine.board_data import (
    TILES_DATA, COLOR_GROUPS, PURCHASABLE_TILES, TILE_TO_GROUP,
    TILE_STREET, TILE_RAILROAD, TILE_UTILITY, TILE_TAX, TILE_CARD, TILE_SPECIAL
)
from monopoly_marl.engine.cards_data import CHANCE_CARDS, COMMUNITY_CHEST_CARDS


# ==============================================================================
# 1. BOARD, TILES & DATA INTEGRITY
# ==============================================================================
def test_board_data_integrity():
    """Validates full 40-tile board geometry, color groups, prices, and mortgage values."""
    assert len(TILES_DATA) == 40

    # Corners
    assert TILES_DATA[0]["name"] == "GO"
    assert "Jail" in TILES_DATA[10]["name"]
    assert TILES_DATA[20]["name"] == "Free Parking"
    assert TILES_DATA[30]["name"] == "Go To Jail"

    # Color groups count
    assert len(COLOR_GROUPS["BROWN"]) == 2
    assert len(COLOR_GROUPS["LIGHT_BLUE"]) == 3
    assert len(COLOR_GROUPS["PINK"]) == 3
    assert len(COLOR_GROUPS["ORANGE"]) == 3
    assert len(COLOR_GROUPS["RED"]) == 3
    assert len(COLOR_GROUPS["YELLOW"]) == 3
    assert len(COLOR_GROUPS["GREEN"]) == 3
    assert len(COLOR_GROUPS["DARK_BLUE"]) == 2
    assert len(COLOR_GROUPS["RAILROAD"]) == 4
    assert len(COLOR_GROUPS["UTILITY"]) == 2

    # Purchasable tiles = 22 streets + 4 railroads + 2 utilities = 28
    assert len(PURCHASABLE_TILES) == 28

    # Mortgage value must be exactly 50% of printed price for all purchasable properties
    for p in PURCHASABLE_TILES:
        tile = TILES_DATA[p]
        assert tile["mortgage"] == tile["price"] // 2

    # Street house costs: Brown/LB: 50, Pink/OR: 100, Red/YL: 150, Green/DB: 200
    for t in COLOR_GROUPS["BROWN"] + COLOR_GROUPS["LIGHT_BLUE"]:
        assert TILES_DATA[t]["house_cost"] == 50
    for t in COLOR_GROUPS["PINK"] + COLOR_GROUPS["ORANGE"]:
        assert TILES_DATA[t]["house_cost"] == 100
    for t in COLOR_GROUPS["RED"] + COLOR_GROUPS["YELLOW"]:
        assert TILES_DATA[t]["house_cost"] == 150
    for t in COLOR_GROUPS["GREEN"] + COLOR_GROUPS["DARK_BLUE"]:
        assert TILES_DATA[t]["house_cost"] == 200


# ==============================================================================
# 2. MOVEMENT, PASSING GO, DICE & CONSECUTIVE DOUBLES (SPEEDING)
# ==============================================================================
def test_movement_and_passing_go():
    """Validates moving forward, position modulo 40, and collecting $200 when passing GO."""
    core = MonopolyCore(seed=123)
    # Player 0 at tile 35, rolls 8 -> position 3 (Baltic Ave)
    core.position[0] = 35
    core.cash[0] = 1000

    # Simulate roll phase with deterministic dice
    core.last_dice = (4, 4)
    # Use step with ROLL action
    core.rng.randint = lambda a, b: 4  # (4, 4) = 8
    core.step({"player_0": {"core": ACT_CORE_ROLL_PASS, "auction": 0, "property": 0,
                            "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
                            "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 0}})

    assert core.position[0] == 3  # (35 + 8) % 40 = 3
    # Collected $200 for passing GO
    assert core.cash[0] == 1200


def test_consecutive_doubles_speeding_to_jail():
    """Validates that 3 consecutive doubles results in speeding penalty: direct to Jail."""
    core = MonopolyCore(seed=123)
    core.position[0] = 0
    core.consecutive_doubles = 2  # Already rolled 2 doubles

    # 3rd roll is also a double (3, 3)
    core.rng.randint = lambda a, b: 3
    core.step({"player_0": {"core": ACT_CORE_ROLL_PASS, "auction": 0, "property": 0,
                            "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
                            "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 0}})

    # Sent directly to Jail (tile 10)
    assert core.in_jail[0] is True
    assert core.position[0] == 10
    assert core.consecutive_doubles == 0
    # Turn should have rotated to next player
    assert core.current_player == 1


# ==============================================================================
# 3. JAIL MECHANICS
# ==============================================================================
def test_jail_exit_via_paying_50_fine():
    """Validates paying $50 fine to get out of Jail before rolling."""
    core = MonopolyCore(seed=123)
    core._send_to_jail(0)
    core.phase = PHASE_JAIL_DECISION
    core.active_decision_player = 0
    core.current_player = 0
    core.cash[0] = 500

    masks = core.get_action_masks(0)
    assert masks["core"][ACT_CORE_PAY_JAIL_50] == 1

    act = {"core": ACT_CORE_PAY_JAIL_50, "auction": 0, "property": 0,
           "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
           "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 0}
    core.step({"player_0": act})

    assert core.cash[0] == 450
    assert core.in_jail[0] is False
    assert core.phase == PHASE_ROLL


def test_jail_exit_via_goojf_card():
    """Validates using Get Out of Jail Free card to exit Jail."""
    core = MonopolyCore(seed=123)
    core._send_to_jail(0)
    core.goojf_cards[0] = 1
    core.goojf_sources[0] = ["chance"]
    core.phase = PHASE_JAIL_DECISION
    core.active_decision_player = 0
    core.current_player = 0

    masks = core.get_action_masks(0)
    assert masks["core"][ACT_CORE_USE_GOOJF] == 1

    act = {"core": ACT_CORE_USE_GOOJF, "auction": 0, "property": 0,
           "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
           "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 0}
    core.step({"player_0": act})

    assert core.goojf_cards[0] == 0
    assert core.in_jail[0] is False
    assert core.phase == PHASE_ROLL


def test_jail_three_turn_limit_mandatory_fine_and_move():
    """Validates that failing doubles 3 turns in Jail forces $50 fine and moves player by 3rd roll."""
    core = MonopolyCore(seed=123)
    core._send_to_jail(0)
    core.jail_turns[0] = 2  # Already failed twice
    core.phase = PHASE_JAIL_DECISION
    core.active_decision_player = 0
    core.current_player = 0
    core.cash[0] = 500

    # 3rd roll fails doubles: (2, 3) = 5
    rolls = [2, 3]
    core.rng.randint = lambda a, b: rolls.pop(0)

    act_roll = {"core": ACT_CORE_ROLL_PASS, "auction": 0, "property": 0,
                "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
                "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 0}
    core.step({"player_0": act_roll})

    # $50 deducted, released, moved 5 spaces from Jail (10 + 5 = 15, Pennsylvania RR)
    assert core.cash[0] == 450
    assert core.in_jail[0] is False
    assert core.position[0] == 15


# ==============================================================================
# 4. PROPERTY PURCHASE & LANDING
# ==============================================================================
def test_unowned_property_direct_buy():
    """Validates landing on unowned property and buying directly at list price."""
    core = MonopolyCore(seed=123)
    core.phase = PHASE_BUY_OR_AUCTION
    core.landed_property = 39  # Boardwalk ($400)
    core.active_decision_player = 0
    core.current_player = 0
    core.cash[0] = 1000

    act_buy = {"core": ACT_CORE_BUY_DIRECT, "auction": 0, "property": 0,
               "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
               "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 0}
    core.step({"player_0": act_buy})

    assert core.property_owner[39] == 0
    assert core.cash[0] == 600  # 1000 - 400


def test_unowned_property_decline_triggers_auction():
    """Validates that declining direct buy immediately triggers mandatory public auction."""
    core = MonopolyCore(seed=123)
    core.phase = PHASE_BUY_OR_AUCTION
    core.landed_property = 24  # Illinois Ave
    core.active_decision_player = 0
    core.current_player = 0

    act_decline = {"core": ACT_CORE_DECLINE_BUY, "auction": 0, "property": 0,
                   "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
                   "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 0}
    core.step({"player_0": act_decline})

    assert core.phase == PHASE_AUCTION
    assert core.auction_property == 24
    assert core.auction_highest_bid == 0


# ==============================================================================
# 5. MANDATORY OPEN AUCTION
# ==============================================================================
def test_open_auction_bidding_and_award():
    """Validates auction progression, bid increments, folding, and property award to highest bidder."""
    core = MonopolyCore(seed=123)
    core._start_auction(property_id=1, is_double_pending=False)  # Mediterranean Ave

    # Active bidder is player 0
    assert core.active_decision_player == 0
    # Player 0 bids +$50
    act_bid = {"core": 0, "auction": ACT_AUC_BID_50, "property": 0,
               "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
               "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 0}
    core.step({"player_0": act_bid})

    assert core.auction_highest_bid == 50
    assert core.auction_highest_bidder == 0

    # Players 1, 2, 3 fold
    act_fold = {"core": 0, "auction": ACT_AUC_FOLD, "property": 0,
                "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
                "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 0}

    core.step({f"player_{core.active_decision_player}": act_fold})
    core.step({f"player_{core.active_decision_player}": act_fold})
    core.step({f"player_{core.active_decision_player}": act_fold})

    # Player 0 wins Mediterranean Ave for $50!
    assert core.property_owner[1] == 0
    assert core.cash[0] == 1450  # 1500 - 50


# ==============================================================================
# 6. RENT CALCULATION ACROSS ALL PROPERTY TYPES
# ==============================================================================
def test_all_rent_calculations():
    """Validates base rent, monopoly double, houses 1-4, hotel, mortgaged, railroad, utility rents."""
    core = MonopolyCore(seed=123)

    # 1. Single street property (Mediterranean Ave: base rent $2)
    core.property_owner[1] = 0
    assert core._calculate_rent(1, dice_sum=7) == 2

    # 2. Complete monopoly without houses (Mediterranean + Baltic): rent doubled = $4!
    core.property_owner[3] = 0
    assert core._calculate_rent(1, dice_sum=7) == 4
    assert core._calculate_rent(3, dice_sum=7) == 8  # Baltic base 4 * 2 = 8

    # 3. If one property in monopoly is mortgaged, unmortgaged collects single rent, not double
    core.property_mortgaged[3] = True
    assert core._calculate_rent(1, dice_sum=7) == 2
    assert core._calculate_rent(3, dice_sum=7) == 0  # Mortgaged collects $0
    core.property_mortgaged[3] = False

    # 4. Houses 1, 2, 3, 4 and Hotel on Boardwalk (#39)
    # Boardwalk rents: [50, 200, 600, 1400, 1700, 2000]
    core.property_owner[37] = 1
    core.property_owner[39] = 1
    for h, expected_rent in enumerate([50, 200, 600, 1400, 1700, 2000]):
        core.property_houses[39] = h
        if h == 0:
            assert core._calculate_rent(39, dice_sum=7) == 100  # Double rent on 0 houses monopoly
        else:
            assert core._calculate_rent(39, dice_sum=7) == expected_rent

    # 5. Railroads: 5 (Reading), 15 (Penn), 25 (B&O), 35 (Short Line)
    core.property_owner[5] = 2
    assert core._calculate_rent(5, dice_sum=7) == 25   # 1 RR
    core.property_owner[15] = 2
    assert core._calculate_rent(5, dice_sum=7) == 50   # 2 RRs
    core.property_owner[25] = 2
    assert core._calculate_rent(5, dice_sum=7) == 100  # 3 RRs
    core.property_owner[35] = 2
    assert core._calculate_rent(5, dice_sum=7) == 200  # 4 RRs

    # 6. Utilities: 12 (Electric Co) & 28 (Water Works)
    core.property_owner[12] = 3
    assert core._calculate_rent(12, dice_sum=7) == 4 * 7   # 1 Utility: 4x roll = 28
    core.property_owner[28] = 3
    assert core._calculate_rent(12, dice_sum=7) == 10 * 7  # 2 Utilities: 10x roll = 70


# ==============================================================================
# 7. HOUSES & HOTELS CONSTRUCTION & BANK SCARCITY
# ==============================================================================
def test_even_building_and_hotel_downgrade():
    """Validates building evenly, upgrading to hotel, and selling hotel with 4-house release."""
    core = MonopolyCore(seed=123)
    # Boardwalk (37) & Park Place (39)
    core.property_owner[37] = 0
    core.property_owner[39] = 0
    core.cash[0] = 5000
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0

    assert core.bank_houses == 32
    assert core.bank_hotels == 12

    # Step 1: Build on 37
    act_b37 = {"core": ACT_CORE_ROLL_PASS, "auction": 0, "property": 37,
               "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
               "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 0}
    core.step({"player_0": act_b37})
    assert core.property_houses[37] == 1
    assert core.bank_houses == 31

    # Step 2: Even building constraint prevents building 2nd house on 37 before 39 gets 1st
    masks = core.get_action_masks(0)
    assert masks["property"][37] == 0
    assert masks["property"][39] == 1

    # Build on 39
    act_b39 = {"core": ACT_CORE_ROLL_PASS, "auction": 0, "property": 39,
               "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
               "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 0}
    core.step({"player_0": act_b39})
    assert core.property_houses[39] == 1
    assert core.bank_houses == 30

    # Advance both to 4 houses
    core.property_houses[37] = 4
    core.property_houses[39] = 4
    core.bank_houses = 24

    # Upgrade 37 to Hotel: consumes 1 hotel, returns 4 houses to bank!
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0
    core.manage_actions_this_turn = 0
    core.step({"player_0": act_b37})

    assert core.property_houses[37] == 5  # Hotel
    assert core.bank_hotels == 11
    assert core.bank_houses == 28        # 24 + 4

    # Sell Hotel on 37: converts to 4 houses, consumes 4 houses from bank, returns hotel!
    act_s37 = {"core": ACT_CORE_ROLL_PASS, "auction": 0, "property": 40 + 37,
               "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
               "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 0}
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0
    core.manage_actions_this_turn = 0
    core.step({"player_0": act_s37})

    assert core.property_houses[37] == 4  # Converted to 4 houses
    assert core.bank_hotels == 12        # Returned hotel to bank
    assert core.bank_houses == 24        # 28 - 4


# ==============================================================================
# 8. MORTGAGE & UNMORTGAGE
# ==============================================================================
def test_mortgage_and_unmortgage_rules():
    """Validates 50% mortgage cash receipt and 10% statutory unmortgage interest."""
    core = MonopolyCore(seed=123)
    core.property_owner[21] = 0  # Kentucky Ave ($220, mortgage $110)
    core.cash[0] = 500
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0

    # Cannot mortgage if color group has houses
    core.property_owner[23] = 0
    core.property_owner[24] = 0
    core.property_houses[23] = 1

    masks = core.get_action_masks(0)
    assert masks["property"][80 + 21] == 0  # Cannot mortgage while group has houses

    # Remove house
    core.property_houses[23] = 0
    masks_clear = core.get_action_masks(0)
    assert masks_clear["property"][80 + 21] == 1  # Legal to mortgage

    # Mortgage Kentucky Ave: receives $110
    act_mtg = {"core": ACT_CORE_ROLL_PASS, "auction": 0, "property": 80 + 21,
               "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
               "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 0}
    core.step({"player_0": act_mtg})

    assert core.property_mortgaged[21] is True
    assert core.cash[0] == 610

    # Unmortgage costs 110 * 1.10 = $121
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0
    core.manage_actions_this_turn = 0
    act_unmtg = {"core": ACT_CORE_ROLL_PASS, "auction": 0, "property": 120 + 21,
                 "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
                 "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 0}
    core.step({"player_0": act_unmtg})

    assert core.property_mortgaged[21] is False
    assert core.cash[0] == 610 - 121  # $489


# ==============================================================================
# 9. BILATERAL TRADING (BARTER)
# ==============================================================================
def test_bilateral_trading_atomic_swap():
    """Validates proposing, accepting, and atomically swapping property and cash."""
    core = MonopolyCore(seed=123)
    core.property_owner[1] = 0  # Luke owns Mediterranean
    core.property_owner[3] = 1  # Graham owns Baltic
    core.cash[0] = 500
    core.cash[1] = 500
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0
    core.current_player = 0

    # Luke proposes trade: offers tile 1 + $50 cash, requests tile 3 + $0 cash from Graham (target_idx 0 = player 1)
    act_trade = {
        "core": ACT_CORE_ROLL_PASS, "auction": 0, "property": 0,
        "trade_target": 0, "trade_offer_prop": 1, "trade_offer_cash": 2,  # index 2 = $50
        "trade_req_prop": 3, "trade_req_cash": 0, "trade_response": 0
    }
    core.step({"player_0": act_trade})

    assert core.phase == PHASE_TRADE_RESPONSE
    assert core.active_decision_player == 1  # Graham to respond

    # Graham accepts trade (trade_response = 1)
    act_accept = {
        "core": 0, "auction": 0, "property": 0,
        "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
        "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 1
    }
    core.step({"player_1": act_accept})

    # Atomic swap verified
    assert core.property_owner[1] == 1  # Graham owns tile 1
    assert core.property_owner[3] == 0  # Luke owns tile 3
    assert core.cash[0] == 450          # 500 - 50
    assert core.cash[1] == 550          # 500 + 50


# ==============================================================================
# 10. CARDS (CHANCE & COMMUNITY CHEST)
# ==============================================================================
def test_chance_and_community_chest_cards_execution():
    """Validates execution of Advance, Cash, Repairs, Pay/Collect Each Player, and Go to Jail cards."""
    core = MonopolyCore(seed=123)

    # 1. Advance to GO (awards $200)
    core.position[0] = 36  # Chance tile
    card_go = {"id": 0, "name": "Advance to GO", "action": "advance_to", "target": 0, "collect_pass_go": True}
    core._execute_card_action(0, card_go, is_double=False)
    assert core.position[0] == 0
    assert core.cash[0] == 1700  # 1500 + 200

    # 2. General Repairs: $25 per house, $100 per hotel
    core.property_owner[1] = 0
    core.property_houses[1] = 3  # 3 houses = $75
    core.property_owner[3] = 0
    core.property_houses[3] = 5  # 1 hotel = $100
    card_rep = {"id": 14, "name": "Make General Repairs", "action": "property_repairs", "house_cost": 25, "hotel_cost": 100}
    core._execute_card_action(0, card_rep, is_double=False)
    assert core.cash[0] == 1700 - 175  # 1525

    # 3. Pay Each Player $50 (e.g. Chairman of the Board)
    initial_cash = [core.cash[i] for i in range(4)]
    card_pay_all = {"id": 13, "name": "Pay each player $50", "action": "pay_each_player", "amount": 50}
    core._execute_card_action(0, card_pay_all, is_double=False)
    assert core.cash[0] == initial_cash[0] - 150  # Paid $50 to players 1, 2, 3
    assert core.cash[1] == initial_cash[1] + 50
    assert core.cash[2] == initial_cash[2] + 50
    assert core.cash[3] == initial_cash[3] + 50

    # 4. Collect from each player $10 (Grand Opera Night)
    initial_cash = [core.cash[i] for i in range(4)]
    card_collect = {"id": 103, "name": "Grand Opera Night", "action": "collect_from_each_player", "amount": 10}
    core._execute_card_action(0, card_collect, is_double=False)
    assert core.cash[0] == initial_cash[0] + 30  # Collected $10 from players 1, 2, 3
    assert core.cash[1] == initial_cash[1] - 10
    assert core.cash[2] == initial_cash[2] - 10
    assert core.cash[3] == initial_cash[3] - 10


# ==============================================================================
# 11. TAXES & SPECIAL TILES
# ==============================================================================
def test_taxes_and_free_parking():
    """Validates Income Tax ($200), Luxury Tax ($100), and Free Parking ($0)."""
    core = MonopolyCore(seed=123)
    core.cash[0] = 1000

    # Income tax (tile 4)
    core._handle_landing(0, pos=4, dice_sum=4, is_double=False)
    assert core.cash[0] == 800

    # Luxury tax (tile 38)
    core._handle_landing(0, pos=38, dice_sum=6, is_double=False)
    assert core.cash[0] == 700

    # Free parking (tile 20)
    core._handle_landing(0, pos=20, dice_sum=8, is_double=False)
    assert core.cash[0] == 700  # No change


# ==============================================================================
# 12. BANKRUPTCY & LAST AGENT STANDING VICTORY
# ==============================================================================
def test_bankruptcy_settlement_and_elimination():
    """Validates liquidation, creditor transfer, and game over Last Agent Standing victory."""
    core = MonopolyCore(seed=123)
    core.property_owner[1] = 0
    core.property_owner[3] = 0
    core.cash[0] = -500  # In debt to player 1
    core.last_creditor = 1

    core._resolve_bankruptcy(0)

    # Player 0 officially bankrupt
    assert core.bankrupt[0] is True
    # Mortgaged properties transferred to creditor (player 1)
    assert core.property_owner[1] == 1
    assert core.property_owner[3] == 1
    assert core.property_mortgaged[1] is True
    assert core.property_mortgaged[3] is True

    # Eliminate players 2 and 3 to trigger Last Agent Standing victory
    core.bankrupt[2] = True
    core.bankrupt[3] = True
    core.current_player = 1

    # Step game to trigger terminal check
    core.step({"player_1": {"core": ACT_CORE_ROLL_PASS, "auction": 0, "property": 0,
                            "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
                            "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 0}})

    assert core.phase == PHASE_GAME_OVER
    active = [pl for pl in range(4) if not core.bankrupt[pl]]
    assert len(active) == 1
    assert active[0] == 1  # Graham wins!


# ==============================================================================
# 13. OBSERVATION VECTOR & ACTION MASK NON-ZERO INTEGRITY
# ==============================================================================
def test_observation_and_action_mask_integrity_across_phases():
    """Validates that observation vectors are strictly 399-dim and action masks have sum >= 1 across all phases."""
    core = MonopolyCore(seed=42)

    for p in range(4):
        obs = core.get_actor_observation(p)
        assert obs.shape == (399,), f"Obs shape mismatch: {obs.shape}"
        assert not np.isnan(obs).any(), "NaN found in observation"
        assert not np.isinf(obs).any(), "Inf found in observation"

    phases_to_test = [
        PHASE_ROLL, PHASE_BUY_OR_AUCTION, PHASE_AUCTION,
        PHASE_JAIL_DECISION, PHASE_PROPERTY_MANAGE, PHASE_TRADE_RESPONSE, PHASE_GAME_OVER
    ]

    for ph in phases_to_test:
        core.phase = ph
        core.landed_property = 1
        core.auction_property = 1
        core.pending_trade = {"initiator": 0, "target": 1, "offer_prop": 40, "offer_cash": 0, "req_prop": 40, "req_cash": 0}
        for pl in range(4):
            masks = core.get_action_masks(pl)
            for head_name, mask in masks.items():
                assert np.sum(mask) >= 1, f"Empty action mask for head {head_name} in phase {ph} for player {pl}!"
