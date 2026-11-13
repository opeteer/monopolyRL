"""House Rule: Half Starting Cash (Austerity Mode).

Halves the starting cash of all players from the official standard $1,500
to $750. Creates immediate early-game liquidity pressure, forces disciplined
property acquisition, accelerates auction trade-offs, and precipitates earlier
bankruptcy dynamics.
"""

from typing import TYPE_CHECKING
from model_2.house_rules.base import HouseRule

if TYPE_CHECKING:
    from model_2.engine.monopoly_core import MonopolyCore


class HalfStartingCashRule(HouseRule):
    """House Rule: Players begin the game with $750 instead of $1,500."""

    @property
    def rule_id(self) -> str:
        return "half_starting_cash"

    @property
    def name(self) -> str:
        return "Half Starting Cash (Austerity Challenge)"

    @property
    def description(self) -> str:
        return (
            "Each player is initialized with only half the official standard cash "
            "($750 instead of $1,500). Severely limits early reckless property buying, "
            "punishes early jail fines or tax landing, and dramatically accelerates "
            "the emergence of early-game financial distress."
        )

    def modify_starting_cash(self, default_cash: int) -> int:
        """Divides starting cash by 2 (1500 -> 750)."""
        return default_cash // 2

    def on_game_init(self, core: "MonopolyCore"):
        """Logs the active austerity rule in game event history."""
        core.event_log.append("[HOUSE RULE] Half Starting Cash active: All players begin with $750.")
