"""Modular House Rules System for Tournament Monopoly MARL."""

from monopoly_marl.house_rules.base import HouseRule
from monopoly_marl.house_rules.manager import HouseRulesManager, BUILTIN_RULES_FACTORY
from monopoly_marl.house_rules.rules.half_starting_cash import HalfStartingCashRule
from monopoly_marl.house_rules.rules.property_cost_inflation import PropertyCostInflation25Rule

__all__ = [
    "HouseRule",
    "HouseRulesManager",
    "HalfStartingCashRule",
    "PropertyCostInflation25Rule",
    "BUILTIN_RULES_FACTORY",
]
