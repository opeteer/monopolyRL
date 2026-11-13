"""Unit tests for Modular House Rules system in Monopoly MARL."""

import pytest
from monopoly_marl.engine.board_data import TILES_DATA
from monopoly_marl.engine.monopoly_core import (
    MonopolyCore,
    PHASE_BUY_OR_AUCTION,
    PHASE_PROPERTY_MANAGE,
    ACT_CORE_ROLL_PASS,
    ACT_CORE_BUY_DIRECT,
    ACT_CORE_END_MANAGE,
)
from monopoly_marl.house_rules import (
    HouseRule,
    HouseRulesManager,
    HalfStartingCashRule,
    PropertyCostInflation25Rule,
    BUILTIN_RULES_FACTORY,
)


def test_default_no_house_rules_starting_cash():
    """Validates that default game initialization without house rules keeps standard $1,500 cash."""
    core = MonopolyCore(seed=42)
    assert core.cash == [1500, 1500, 1500, 1500]
    assert "initialized with $1500" in core.event_log[0]
    assert len(core.house_rules.active_rules) == 0


def test_half_starting_cash_rule_initialization():
    """Validates that HalfStartingCashRule sets initial cash to exactly $750."""
    rule = HalfStartingCashRule()
    assert rule.rule_id == "half_starting_cash"
    assert rule.modify_starting_cash(1500) == 750

    manager = HouseRulesManager([rule])
    assert manager.get_starting_cash(1500) == 750

    core = MonopolyCore(seed=42, house_rules=manager)
    assert core.cash == [750, 750, 750, 750]
    assert any("Half Starting Cash" in log for log in core.event_log)


def test_house_rules_from_string_aliases():
    """Validates that house rules can be specified via string aliases ('half_cash', 'half_starting_cash')."""
    core1 = MonopolyCore(seed=42, house_rules="half_cash")
    assert core1.cash == [750, 750, 750, 750]
    assert core1.house_rules.has_rule("half_starting_cash")

    core2 = MonopolyCore(seed=42, house_rules="half_starting_cash")
    assert core2.cash == [750, 750, 750, 750]
    assert core2.house_rules.has_rule("half_cash")


def test_house_rules_manager_composition_and_removal():
    """Validates dynamic registration, query, and removal in HouseRulesManager."""
    manager = HouseRulesManager()
    assert manager.get_starting_cash(1500) == 1500
    assert not manager.has_rule("half_starting_cash")

    rule = HalfStartingCashRule()
    manager.register_rule(rule)
    assert manager.has_rule("half_starting_cash")
    assert manager.get_starting_cash(1500) == 750
    assert "Half Starting Cash" in manager.get_rule_summary()

    # Removal
    removed = manager.remove_rule("half_starting_cash")
    assert removed is rule
    assert not manager.has_rule("half_starting_cash")
    assert manager.get_starting_cash(1500) == 1500


def test_invalid_house_rule_identifier_raises_error():
    """Validates that invalid rule names raise clear ValueError."""
    with pytest.raises(ValueError, match="Unknown house rule identifier"):
        HouseRulesManager.from_config("non_existent_rule_xyz")


def test_half_cash_gameplay_and_pass_go():
    """Validates that gameplay under $750 starting cash correctly steps and collects standard $200 on GO."""
    core = MonopolyCore(seed=100, house_rules="half_cash")
    assert core.cash[0] == 750

    # Advance player 0 across GO (e.g. from tile 35 to tile 5)
    core.position[0] = 35
    act_roll = {
        "core": 0, "auction": 0, "property": 160,
        "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
        "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 0
    }
    # Force dice roll of 10 to land on tile 5 (Reading RR)
    core.rng.randint = lambda a, b: 5  # 5 + 5 = 10
    core.step({"player_0": act_roll})

    assert core.position[0] == 5
    # Started with 750, collected $200 for passing GO -> 950
    assert core.cash[0] == 950


def _make_action(core_act=ACT_CORE_ROLL_PASS, prop_act=160):
    return {
        "core": core_act,
        "auction": 0,
        "property": prop_act,
        "trade_target": 0,
        "trade_offer_prop": 40,
        "trade_offer_cash": 0,
        "trade_req_prop": 40,
        "trade_req_cash": 0,
        "trade_response": 0,
    }


def test_property_cost_inflation_25_rule_initialization():
    """Validates PropertyCostInflation25Rule keeps $1500 cash and increases land/house costs +25%."""
    rule = PropertyCostInflation25Rule()
    assert rule.rule_id == "property_cost_plus_25"

    for alias in (
        "property_cost_plus_25",
        "property_inflation_25",
        "inflation_25",
        "plus_25_cost",
        "property_up_25",
    ):
        core = MonopolyCore(seed=42, house_rules=alias)
        assert core.cash == [1500, 1500, 1500, 1500]
        assert core.house_rules.has_rule("property_cost_plus_25")

    core = MonopolyCore(seed=42, house_rules="property_cost_plus_25")
    # Nominal price +25% (half-up rounding)
    assert core.get_property_price(1) == 75     # Mediterranean $60 -> $75
    assert core.get_property_price(12) == 188   # Electric Co $150 -> $188
    assert core.get_property_price(5) == 250    # Reading RR $200 -> $250
    assert core.get_property_price(37) == 438   # Park Place $350 -> $438
    assert core.get_property_price(39) == 500   # Boardwalk $400 -> $500

    # House / hotel cost +25%
    assert core.get_property_house_cost(1) == 63    # Brown $50 -> $63
    assert core.get_property_house_cost(11) == 125  # Pink $100 -> $125
    assert core.get_property_house_cost(21) == 188  # Red $150 -> $188
    assert core.get_property_house_cost(39) == 250  # Dark Blue $200 -> $250

    # Mortgage proportional (modified_price // 2)
    assert core.get_property_mortgage(1) == 37   # 75 // 2
    assert core.get_property_mortgage(12) == 94  # 188 // 2
    assert core.get_property_mortgage(5) == 125  # 250 // 2
    assert core.get_property_mortgage(37) == 219 # 438 // 2
    assert core.get_property_mortgage(39) == 250 # 500 // 2


def test_property_cost_inflation_25_gameplay_buy_build_mortgage_and_net_worth():
    """Validates inflated deductions for buying, building, selling, mortgaging, and net worth."""
    core = MonopolyCore(seed=123, house_rules="property_cost_plus_25")
    core.phase = PHASE_BUY_OR_AUCTION
    core.active_decision_player = 0
    core.landed_property = 1  # Mediterranean Ave ($60 -> $75)

    # Buy directly for $75
    core.step({"player_0": _make_action(core_act=ACT_CORE_BUY_DIRECT)})
    assert core.property_owner[1] == 0
    assert core.cash[0] == 1500 - 75  # 1425
    assert core.get_player_net_worth(0) == 1500

    # Grant Baltic Ave (tile 3, $75) to complete Brown monopoly, enter manage phase
    core.property_owner[3] = 0
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0
    core.current_player = 0

    # Build 1 house on Mediterranean (tile 1, cost $50 -> $63)
    core.step({"player_0": _make_action(core_act=ACT_CORE_END_MANAGE, prop_act=1)})
    assert core.property_houses[1] == 1
    assert core.cash[0] == 1425 - 63  # 1362
    assert core.get_player_net_worth(0) == 1362 + 75 + 75 + 63  # 1575

    # Sell the house on a subsequent turn (clear same-turn thrash guard; refund int(63 * 0.5) = 31)
    core.built_this_turn.clear()
    core.step({"player_0": _make_action(core_act=ACT_CORE_END_MANAGE, prop_act=41)})
    assert core.property_houses[1] == 0
    assert core.cash[0] == 1362 + 31  # 1393

    # Mortgage Baltic Ave (tile 3, mortgage = 37)
    core.step({"player_0": _make_action(core_act=ACT_CORE_END_MANAGE, prop_act=83)})
    assert core.property_mortgaged[3] is True
    assert core.cash[0] == 1393 + 37  # 1430
    assert core.get_player_net_worth(0) == 1430 + 75 + 37  # 1542

    # Reset same-turn lock and unmortgage Baltic (int(37 * 1.1) = 40)
    core.mortgaged_this_turn.clear()
    core.step({"player_0": _make_action(core_act=ACT_CORE_END_MANAGE, prop_act=123)})
    assert core.property_mortgaged[3] is False
    assert core.cash[0] == 1430 - 40  # 1390


def test_property_cost_inflation_25_masks_enforce_inflated_price():
    """Validates action masks block direct buy when cash is between standard and +25% price."""
    core = MonopolyCore(seed=42, house_rules="property_cost_plus_25")
    core.phase = PHASE_BUY_OR_AUCTION
    core.active_decision_player = 0
    core.landed_property = 1  # Standard $60, Inflated $75

    core.cash[0] = 65  # Enough for $60 standard, NOT enough for $75 inflated
    masks = core.get_action_masks(0)
    assert masks["core"][ACT_CORE_BUY_DIRECT] == 0

    core.cash[0] = 75
    masks = core.get_action_masks(0)
    assert masks["core"][ACT_CORE_BUY_DIRECT] == 1


def test_property_cost_inflation_25_full_episode_stability():
    """Runs random masked steps under PropertyCostInflation25Rule to ensure stability."""
    import random
    rng = random.Random(777)
    core = MonopolyCore(seed=777, house_rules="property_cost_plus_25")

    for _ in range(250):
        if core.phase == 6:  # PHASE_GAME_OVER
            break
        p = core.active_decision_player
        masks = core.get_action_masks(p)
        act = {}
        for head, arr in masks.items():
            valid_indices = [i for i, v in enumerate(arr) if v == 1]
            act[head] = rng.choice(valid_indices) if valid_indices else 0
        core.step({f"player_{p}": act})

    # Verify TILES_DATA global constant was not mutated
    assert TILES_DATA[1]["price"] == 60
    assert TILES_DATA[1]["house_cost"] == 50
    assert TILES_DATA[1]["mortgage"] == 30
