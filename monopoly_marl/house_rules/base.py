"""Abstract Base Class for Modular Monopoly House Rules.

Defines the hook interface allowing custom variants, austerity challenges,
free parking jackpots, double GO salaries, and modified starting parameters.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from monopoly_marl.engine.monopoly_core import MonopolyCore


class HouseRule(ABC):
    """Abstract Base Class for a modular Monopoly House Rule."""

    @property
    @abstractmethod
    def rule_id(self) -> str:
        """Unique machine-readable identifier string (e.g. 'half_starting_cash')."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the house rule."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Comprehensive explanation of the house rule's effects and rationale."""
        pass

    def modify_starting_cash(self, default_cash: int) -> int:
        """Hook to modify player starting cash upon match initialization."""
        return default_cash

    def on_game_init(self, core: "MonopolyCore"):
        """Hook called when game state machine resets or initializes."""
        pass

    def on_pass_go_salary(self, player_id: int, standard_salary: int) -> int:
        """Hook to modify salary awarded when passing GO."""
        return standard_salary

    def on_turn_start(self, core: "MonopolyCore", player_id: int):
        """Hook called at the start of a player's turn."""
        pass

    def on_land_on_free_parking(self, core: "MonopolyCore", player_id: int):
        """Hook called when a player lands on Free Parking (Tile 20)."""
        pass
