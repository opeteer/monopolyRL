"""Manager and Registry for Monopoly House Rules."""

from typing import Dict, List, Optional, Union, TYPE_CHECKING
from monopoly_marl.house_rules.base import HouseRule
from monopoly_marl.house_rules.rules.half_starting_cash import HalfStartingCashRule

if TYPE_CHECKING:
    from monopoly_marl.engine.monopoly_core import MonopolyCore


# Registry of built-in house rules with canonical IDs and aliases
BUILTIN_RULES_FACTORY = {
    "half_starting_cash": HalfStartingCashRule,
    "half_cash": HalfStartingCashRule,
    "austerity": HalfStartingCashRule,
}


class HouseRulesManager:
    """Manages active house rules, dispatches game hooks, and validates rule configurations."""

    def __init__(self, rules: Optional[List[HouseRule]] = None):
        self._rules: Dict[str, HouseRule] = {}
        if rules:
            for r in rules:
                self.register_rule(r)

    def register_rule(self, rule: HouseRule):
        """Adds or activates a house rule."""
        self._rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str) -> Optional[HouseRule]:
        """Deactivates a house rule by id."""
        return self._rules.pop(rule_id, None)

    def has_rule(self, rule_id: str) -> bool:
        """Returns True if rule_id or its canonical alias is currently active."""
        if rule_id in self._rules:
            return True
        if rule_id in BUILTIN_RULES_FACTORY:
            canon_id = BUILTIN_RULES_FACTORY[rule_id]().rule_id
            return canon_id in self._rules
        return False

    @property
    def active_rules(self) -> List[HouseRule]:
        """Returns a list of all currently active HouseRule instances."""
        return list(self._rules.values())

    def get_starting_cash(self, default_cash: int = 1500) -> int:
        """Calculates effective starting cash after applying all active modifiers."""
        cash = default_cash
        for rule in self.active_rules:
            cash = rule.modify_starting_cash(cash)
        return cash

    def apply_game_init(self, core: "MonopolyCore"):
        """Dispatches game initialization hooks to all active house rules."""
        for rule in self.active_rules:
            rule.on_game_init(core)

    def apply_pass_go_salary(self, player_id: int, standard_salary: int = 200) -> int:
        """Dispatches pass GO salary hook to active house rules."""
        salary = standard_salary
        for rule in self.active_rules:
            salary = rule.on_pass_go_salary(player_id, salary)
        return salary

    def get_rule_summary(self) -> str:
        """Returns a formatted summary string of all active house rules."""
        if not self._rules:
            return "Official Tournament Standard"
        return " + ".join(r.name for r in self.active_rules)

    @classmethod
    def from_config(cls, config: Optional[Union["HouseRulesManager", List[HouseRule], List[str], str]] = None) -> "HouseRulesManager":
        """Factory method to construct a HouseRulesManager from various input types.

        Supports:
        - None: returns empty manager (official standard rules)
        - HouseRulesManager: returns instance as-is
        - List[HouseRule]: registers each instance
        - List[str] / comma-separated str (e.g. 'half_cash', 'half_starting_cash'): instantiates from registry
        """
        if config is None:
            return cls([])

        if isinstance(config, HouseRulesManager):
            return config

        if isinstance(config, str):
            rule_tokens = [tok.strip().lower() for tok in config.split(",") if tok.strip()]
            rules_list = []
            for tok in rule_tokens:
                if tok in BUILTIN_RULES_FACTORY:
                    rules_list.append(BUILTIN_RULES_FACTORY[tok]())
                else:
                    raise ValueError(f"Unknown house rule identifier: '{tok}'. Available: {list(BUILTIN_RULES_FACTORY.keys())}")
            return cls(rules_list)

        if isinstance(config, (list, tuple)):
            rules_list = []
            for item in config:
                if isinstance(item, HouseRule):
                    rules_list.append(item)
                elif isinstance(item, str):
                    tok = item.strip().lower()
                    if tok in BUILTIN_RULES_FACTORY:
                        rules_list.append(BUILTIN_RULES_FACTORY[tok]())
                    else:
                        raise ValueError(f"Unknown house rule identifier: '{tok}'")
                else:
                    raise TypeError(f"Invalid house rule element: {item}")
            return cls(rules_list)

        raise TypeError(f"Unsupported house rules configuration type: {type(config)}")
