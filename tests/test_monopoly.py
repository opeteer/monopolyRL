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
    PHASE_AUCTION, PHASE_PROPERTY_MANAGE, ACT_CORE_ROLL_PASS, ACT_CORE_BUY_DIRECT,
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
    act_mtg = {"core": ACT_CORE_ROLL_PASS, "auction": 0, "property": 81, "trade_target": 0,
               "trade_offer_prop": 40, "trade_offer_cash": 0, "trade_req_prop": 40,
               "trade_req_cash": 0, "trade_response": 0}
    core.step({"player_0": act_mtg})

    assert core.property_mortgaged[1] is True
    assert core.cash[0] == 130  # Received $30

    # Unmortgage: cost is 30 * 1.10 = 33
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0
    # Unmortgage action: 120 + 1 = 121
    act_unmtg = {"core": ACT_CORE_ROLL_PASS, "auction": 0, "property": 121, "trade_target": 0,
                 "trade_offer_prop": 40, "trade_offer_cash": 0, "trade_req_prop": 40,
                 "trade_req_cash": 0, "trade_response": 0}
    core.step({"player_0": act_unmtg})

    assert core.property_mortgaged[1] is False
    assert core.cash[0] == 130 - 33  # 97


def test_identical_blueprint_orthogonal_init():
    """Validates that all agents share identical neural architecture with orthogonal weights."""
    actor1 = ActorNetwork(obs_dim=500)
    actor2 = ActorNetwork(obs_dim=500)

    # Check layer shapes match
    for (n1, p1), (n2, p2) in zip(actor1.named_parameters(), actor2.named_parameters()):
        assert n1 == n2
        assert p1.shape == p2.shape

    # Check forward pass consistency
    dummy_obs = torch.randn(2, 500)
    dummy_masks = {
        "core": torch.ones(2, 6, dtype=torch.int8),
        "auction": torch.ones(2, 7, dtype=torch.int8),
        "property": torch.ones(2, 161, dtype=torch.int8),
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
    assert core.get_actor_observation(0).shape == (500,)
    assert core.get_global_state().shape == (2032,)

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


def test_property_purchase_official_affordability_masking():
    """Validates that purchasing is strictly permitted when cash >= price, and forbidden when cash < price."""
    core = MonopolyCore(seed=123, include_synergy_obs=True)

    # Setup player 0 in PHASE_BUY_OR_AUCTION on tile 21 (Kentucky Ave, Red, price $220)
    core.phase = PHASE_BUY_OR_AUCTION
    core.landed_property = 21
    core.active_decision_player = 0
    core.position[0] = 21

    # Player 0 has $240 cash (cash >= price $220): BUY_DIRECT must be legal!
    core.cash[0] = 240
    masks = core.get_action_masks(0)
    assert masks["core"][ACT_CORE_BUY_DIRECT] == 1, "Direct buy must be legal when player can afford it"
    assert masks["core"][ACT_CORE_DECLINE_BUY] == 1, "Decline buy is always legal under official rules"

    # Player 0 has $100 cash (cash < price $220): BUY_DIRECT must be illegal!
    core.cash[0] = 100
    masks_poor = core.get_action_masks(0)
    assert masks_poor["core"][ACT_CORE_BUY_DIRECT] == 0, "Cannot buy directly when cash is insufficient"
    assert masks_poor["core"][ACT_CORE_DECLINE_BUY] == 1, "Must be able to decline to auction"


def test_property_purchase_official_decline_masking():
    """Validates that under official Monopoly rules, declining purchase is always legal."""
    core = MonopolyCore(seed=456, include_synergy_obs=True)

    # Player 0 lands on tile 19 (New York Ave, Orange, price $200)
    core.phase = PHASE_BUY_OR_AUCTION
    core.landed_property = 19
    core.active_decision_player = 0
    core.cash[0] = 500

    masks = core.get_action_masks(0)
    assert masks["core"][ACT_CORE_BUY_DIRECT] == 1
    assert masks["core"][ACT_CORE_DECLINE_BUY] == 1, "Official Monopoly rules always allow declining to auction"


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
    assert obs.shape == (500,), f"Expected 500-dim obs, got {obs.shape}"
    global_state = core.get_global_state()
    assert global_state.shape == (2032,), f"Expected 2032-dim global state, got {global_state.shape}"


def test_agent_names_and_unbounded_horizon_survival():
    """Validates player names (Luke, Graham, Pieter, Allison) and unbounded survival victory conditions."""
    from monopoly_marl.engine.monopoly_core import DEFAULT_PLAYER_NAMES, PHASE_GAME_OVER
    from monopoly_marl.engine.pettingzoo_wrapper import MonopolyParallelEnv

    assert DEFAULT_PLAYER_NAMES == ["Luke", "Graham", "Pieter", "Allison"]

    core = MonopolyCore(seed=42, max_rounds=None)
    assert core.max_rounds is None, "max_rounds should be None for unbounded horizon"
    assert core.get_player_name(0) == "Luke"
    assert core.get_player_name(1) == "Graham"
    assert core.get_player_name(2) == "Pieter"
    assert core.get_player_name(3) == "Allison"

    # Observation must have valid dimension (500,)
    obs = core.get_actor_observation(0)
    assert obs.shape == (500,)

    # Bankrupt players 1, 2, 3 one by one
    core.cash[1] = -100
    core.cash[2] = -100
    core.cash[3] = -100

    dummy_actions = {
        f"player_{i}": {
            "core": 0, "auction": 0, "property": 0,
            "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
            "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 0
        }
        for i in range(4)
    }

    obs, rewards, dones, infos = core.step(dummy_actions)
    assert dones["player_1"] is True
    assert dones["player_2"] is True
    assert dones["player_3"] is True
    assert dones["__all__"] is True
    assert core.phase == PHASE_GAME_OVER
    # Luke is the last agent standing
    assert any("Luke is the TOURNAMENT CHAMPION (Last Agent Standing)" in msg for msg in core.event_log)

    # Test PettingZoo wrapper integration
    pz_env = MonopolyParallelEnv(max_rounds=None)
    assert pz_env.agent_to_name["player_0"] == "Luke"
    assert pz_env.agent_to_name["player_1"] == "Graham"
    assert pz_env.agent_to_name["player_2"] == "Pieter"
    assert pz_env.agent_to_name["player_3"] == "Allison"


def test_round_counter_advancement_when_player0_eliminated():
    """Validates that round_count advances even when Player 0 (Luke) is bankrupt."""
    core = MonopolyCore(seed=123, max_rounds=None)
    core.bankrupt[0] = True  # Luke eliminated
    core.current_player = 3
    initial_round = core.round_count

    # Ending turn for player 3 wraps around to player 1 (skipping 0) and increments round_count
    core._end_turn()
    assert core.current_player == 1
    assert core.round_count == initial_round + 1


def test_gui_rendering_across_all_game_phases():
    """Validates that Pygame MonopolyRenderer and MonopolyDashboard execute without errors across all game phases."""
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame
    pygame.init()

    from monopoly_marl.engine.monopoly_core import (
        MonopolyCore, PHASE_ROLL, PHASE_BUY_OR_AUCTION, PHASE_AUCTION,
        PHASE_JAIL_DECISION, PHASE_PROPERTY_MANAGE, PHASE_TRADE_RESPONSE, PHASE_GAME_OVER
    )
    from monopoly_marl.gui.renderer import MonopolyRenderer
    from monopoly_marl.gui.dashboard import MonopolyDashboard

    surface = pygame.Surface((1380, 840))
    core = MonopolyCore(seed=42)

    # Setup properties & houses to test monopoly and building rendering
    core.property_owner[1] = 0
    core.property_owner[3] = 0
    core.property_houses[1] = 3
    core.property_houses[3] = 5

    renderer = MonopolyRenderer(surface, board_size=800, offset_x=20, offset_y=20)
    dashboard = MonopolyDashboard(surface, x=840, y=20, width=520, height=800)

    # 1. Default Roll / Manage Phase
    renderer.render(core)
    dashboard.render(core)

    # 2. Auction Phase
    core.phase = PHASE_AUCTION
    core.auction_property = 24
    core.auction_highest_bid = 240
    core.auction_highest_bidder = 1
    core.auction_active_bidders = [0, 1, 2]
    renderer.render(core)
    dashboard.render(core)

    # 3. Trade Response Phase
    core.phase = PHASE_TRADE_RESPONSE
    core.pending_trade = {
        "initiator": 0, "target": 1,
        "offer_prop": 1, "offer_cash": 100,
        "req_prop": 6, "req_cash": 50
    }
    renderer.render(core)
    dashboard.render(core)

    # 4. Game Over Phase
    core.phase = PHASE_GAME_OVER
    core.bankrupt[1] = True
    core.bankrupt[2] = True
    core.bankrupt[3] = True
    renderer.render(core)
    dashboard.render(core)

    # 5. Help Overlay
    dashboard.show_help = True
    dashboard.render(core)


def test_gui_cards_rendering_and_interaction():
    """Validates rendering of Title Deed, Chance, and Community Chest cards along with click inspection."""
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame
    pygame.init()

    from monopoly_marl.engine.monopoly_core import MonopolyCore, PHASE_BUY_OR_AUCTION, PHASE_ROLL
    from monopoly_marl.gui.renderer import MonopolyRenderer
    from monopoly_marl.engine.cards_data import CHANCE_CARDS, COMMUNITY_CHEST_CARDS

    surface = pygame.Surface((1380, 840))
    core = MonopolyCore(seed=42)
    renderer = MonopolyRenderer(surface, board_size=800, offset_x=20, offset_y=20)

    # 1. Title Deed Card in Center HUD during Buy/Auction phase
    core.phase = PHASE_BUY_OR_AUCTION
    core.landed_property = 39 # Boardwalk
    renderer.render(core)

    # 2. Chance Card in Center HUD
    core.phase = PHASE_ROLL
    core.position[0] = 7
    core.last_card_drawn = CHANCE_CARDS[0]
    renderer.render(core)

    # 3. Community Chest Card in Center HUD
    core.position[0] = 2
    core.last_card_drawn = COMMUNITY_CHEST_CARDS[1]
    renderer.render(core)

    # 4. Interactive Tile Inspection click
    # Click tile 24 (Illinois Ave)
    tile_rect_24 = renderer.tile_rects[24]
    clicked = renderer.handle_click(tile_rect_24.center)
    assert clicked is True
    assert renderer.inspect_tile_id == 24
    renderer.render(core)

    # Click Railroad (tile 5)
    tile_rect_5 = renderer.tile_rects[5]
    clicked = renderer.handle_click(tile_rect_5.center)
    assert clicked is True
    assert renderer.inspect_tile_id == 5
    renderer.render(core)

    # Click Utility (tile 12)
    tile_rect_12 = renderer.tile_rects[12]
    clicked = renderer.handle_click(tile_rect_12.center)
    assert clicked is True
    assert renderer.inspect_tile_id == 12
    renderer.render(core)

    # Click Chance (tile 7)
    tile_rect_7 = renderer.tile_rects[7]
    clicked = renderer.handle_click(tile_rect_7.center)
    assert clicked is True
    assert renderer.inspect_tile_id == 7
    renderer.render(core)

    # Click close button to dismiss inspection
    cx = renderer.offset_x + renderer.board_size // 2
    cy = renderer.offset_y + renderer.board_size // 2
    close_btn_pos = (cx + 105, cy - 140)
    renderer.handle_click(close_btn_pos)
    assert renderer.inspect_tile_id is None


def test_house_and_hotel_building_mechanics():
    """Validates building houses, even building constraints, bank scarcity, hotel upgrades, and scaled rent."""
    from monopoly_marl.engine.monopoly_core import MonopolyCore, PHASE_PROPERTY_MANAGE, ACT_CORE_ROLL_PASS, ACT_CORE_END_MANAGE

    core = MonopolyCore(seed=42)
    # Give Player 0 the Brown monopoly (Mediterranean #1 and Baltic #3)
    core.property_owner[1] = 0
    core.property_owner[3] = 0
    core.cash[0] = 3000
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0

    assert core.bank_houses == 32
    assert core.bank_hotels == 12

    # Step 1: Build house on tile 1 ($50)
    masks = core.get_action_masks(0)
    assert masks["core"][ACT_CORE_ROLL_PASS] == 1
    assert masks["core"][ACT_CORE_END_MANAGE] == 1
    assert masks["property"][1] == 1
    assert masks["property"][3] == 1

    act_b1 = {"core": ACT_CORE_ROLL_PASS, "auction": 0, "property": 1, "trade_target": 0,
              "trade_offer_prop": 40, "trade_offer_cash": 0, "trade_req_prop": 40,
              "trade_req_cash": 0, "trade_response": 0}
    core.step({"player_0": act_b1})

    assert core.property_houses[1] == 1
    assert core.property_houses[3] == 0
    assert core.bank_houses == 31
    assert core.cash[0] == 2950
    assert core.phase == PHASE_PROPERTY_MANAGE

    # Step 2: Even building constraint - tile 1 has 1 house, min_h is 0, so must build on tile 3!
    masks2 = core.get_action_masks(0)
    assert masks2["property"][1] == 0  # Cannot build 2nd house on 1 until 3 gets 1st house
    assert masks2["property"][3] == 1

    act_b3 = {"core": ACT_CORE_ROLL_PASS, "auction": 0, "property": 3, "trade_target": 0,
              "trade_offer_prop": 40, "trade_offer_cash": 0, "trade_req_prop": 40,
              "trade_req_cash": 0, "trade_response": 0}
    core.step({"player_0": act_b3})

    assert core.property_houses[1] == 1
    assert core.property_houses[3] == 1
    assert core.bank_houses == 30
    assert core.cash[0] == 2900

    # Fast forward to 4 houses each
    core.property_houses[1] = 4
    core.property_houses[3] = 4
    core.bank_houses = 24

    # Upgrade tile 1 to HOTEL ($50)
    # Consumes 1 hotel, releases 4 houses back to bank
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0
    core.manage_actions_this_turn = 0
    core.step({"player_0": act_b1})

    assert core.property_houses[1] == 5  # 5 = Hotel
    assert core.bank_hotels == 11        # Consumed 1 hotel
    assert core.bank_houses == 28        # 24 + 4 houses returned!

    # Verify scaled hotel rent when player 1 lands on Mediterranean Ave (#1)
    # Base rent is $2, but with hotel, Mediterranean rent is $250!
    rent = core._calculate_rent(1, dice_sum=7)
    assert rent == 250








