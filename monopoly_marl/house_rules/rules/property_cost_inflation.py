"""Property & Land Cost +25% Inflation House Rule.

Increases the nominal purchase price of all purchasable properties (streets,
railroads, utilities) and house/hotel construction costs by 25%, while scaling
statutory mortgage values proportionally (50% of modified purchase price).
"""

from typing import Optional
from monopoly_marl.house_rules.base import HouseRule


def _scale_half_up(amount: int, pct: int) -> int:
    """Scales a positive integer amount by +(pct)% using half-up integer rounding."""
    if amount <= 0:
        return amount
    return (amount * (100 + pct) + 50) // 100


class PropertyCostInflation25Rule(HouseRule):
    """House rule that raises all property land prices and house/hotel costs by 25%."""

    def __init__(self, inflation_pct: int = 25):
        self.inflation_pct = int(inflation_pct)

    @property
    def rule_id(self) -> str:
        if self.inflation_pct == 25:
            return "property_cost_plus_25"
        return f"property_cost_plus_{self.inflation_pct}"

    @property
    def name(self) -> str:
        return f"Property & Land Cost +{self.inflation_pct}%"

    @property
    def description(self) -> str:
        return (
            f"Seluruh biaya pembelian tanah/properti (street, railroad, utility) "
            f"serta biaya konstruksi rumah/hotel naik sebesar +{self.inflation_pct}% "
            f"dari harga standar resmi (dengan nilai gadai proporsional 50% dari harga baru)."
        )

    def modify_property_price(self, tile_id: int, default_price: int) -> int:
        return _scale_half_up(default_price, self.inflation_pct)

    def modify_house_cost(self, tile_id: int, default_house_cost: int) -> int:
        return _scale_half_up(default_house_cost, self.inflation_pct)

    def modify_mortgage_value(
        self, tile_id: int, default_mortgage: int, modified_price: Optional[int] = None
    ) -> int:
        if default_mortgage <= 0:
            return default_mortgage
        if modified_price is not None:
            return modified_price // 2
        return _scale_half_up(default_mortgage, self.inflation_pct)
