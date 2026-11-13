"""Unit tests for Modular House Rules system in Monopoly MARL."""

import pytest
from monopoly_marl.engine.monopoly_core import MonopolyCore
from monopoly_marl.house_rules import (
    HouseRule,
    HouseRulesManager,
    HalfStartingCashRule,
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
