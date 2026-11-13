"""Official Tournament Monopoly Board Data & Reference Specifications.

Defines all 40 tiles, color groups, property pricing, tiered rent schedules,
house costs, mortgage values, and dice roll distributions for PBRS calculation.
"""

from typing import Dict, List, Any
import numpy as np

# Tile Types
TILE_SPECIAL = "special"
TILE_STREET = "street"
TILE_RAILROAD = "railroad"
TILE_UTILITY = "utility"
TILE_TAX = "tax"
TILE_CARD = "card"

# Color group identifiers
COLOR_BROWN = "BROWN"
COLOR_LIGHT_BLUE = "LIGHT_BLUE"
COLOR_PINK = "PINK"
COLOR_ORANGE = "ORANGE"
COLOR_RED = "RED"
COLOR_YELLOW = "YELLOW"
COLOR_GREEN = "GREEN"
COLOR_DARK_BLUE = "DARK_BLUE"
COLOR_RAILROAD = "RAILROAD"
COLOR_UTILITY = "UTILITY"

COLOR_GROUPS: Dict[str, List[int]] = {
    COLOR_BROWN: [1, 3],
    COLOR_LIGHT_BLUE: [6, 8, 9],
    COLOR_PINK: [11, 13, 14],
    COLOR_ORANGE: [16, 18, 19],
    COLOR_RED: [21, 23, 24],
    COLOR_YELLOW: [26, 27, 29],
    COLOR_GREEN: [31, 32, 34],
    COLOR_DARK_BLUE: [37, 39],
    COLOR_RAILROAD: [5, 15, 25, 35],
    COLOR_UTILITY: [12, 28],
}

# Mapping of tile_id to color group
TILE_TO_GROUP: Dict[int, str] = {}
for grp, tiles in COLOR_GROUPS.items():
    for t in tiles:
        TILE_TO_GROUP[t] = grp

# Monopoly monopoly bonuses for PBRS (U_g)
GROUP_MONOPOLY_BONUS: Dict[str, float] = {
    COLOR_BROWN: 120.0,
    COLOR_LIGHT_BLUE: 320.0,
    COLOR_PINK: 440.0,
    COLOR_ORANGE: 560.0,
    COLOR_RED: 680.0,
    COLOR_YELLOW: 800.0,
    COLOR_GREEN: 920.0,
    COLOR_DARK_BLUE: 750.0,
    COLOR_RAILROAD: 400.0,
    COLOR_UTILITY: 200.0,
}

# Complete 40 board tiles definition
TILES_DATA: List[Dict[str, Any]] = [
    # 0
    {"id": 0, "name": "GO", "type": TILE_SPECIAL, "price": 0, "color": None},
    # 1
    {"id": 1, "name": "Mediterranean Avenue", "type": TILE_STREET, "color": COLOR_BROWN,
     "price": 60, "rent": [2, 10, 30, 90, 160, 250], "house_cost": 50, "mortgage": 30},
    # 2
    {"id": 2, "name": "Community Chest", "type": TILE_CARD, "deck": "chest", "price": 0, "color": None},
    # 3
    {"id": 3, "name": "Baltic Avenue", "type": TILE_STREET, "color": COLOR_BROWN,
     "price": 60, "rent": [4, 20, 60, 180, 320, 450], "house_cost": 50, "mortgage": 30},
    # 4
    {"id": 4, "name": "Income Tax", "type": TILE_TAX, "price": 0, "tax_amount": 200, "color": None},
    # 5
    {"id": 5, "name": "Reading Railroad", "type": TILE_RAILROAD, "color": COLOR_RAILROAD,
     "price": 200, "rent": [25, 50, 100, 200], "house_cost": 0, "mortgage": 100},
    # 6
    {"id": 6, "name": "Oriental Avenue", "type": TILE_STREET, "color": COLOR_LIGHT_BLUE,
     "price": 100, "rent": [6, 30, 90, 270, 400, 550], "house_cost": 50, "mortgage": 50},
    # 7
    {"id": 7, "name": "Chance", "type": TILE_CARD, "deck": "chance", "price": 0, "color": None},
    # 8
    {"id": 8, "name": "Vermont Avenue", "type": TILE_STREET, "color": COLOR_LIGHT_BLUE,
     "price": 100, "rent": [6, 30, 90, 270, 400, 550], "house_cost": 50, "mortgage": 50},
    # 9
    {"id": 9, "name": "Connecticut Avenue", "type": TILE_STREET, "color": COLOR_LIGHT_BLUE,
     "price": 120, "rent": [8, 40, 100, 300, 450, 600], "house_cost": 50, "mortgage": 60},
    # 10
    {"id": 10, "name": "Jail / Just Visiting", "type": TILE_SPECIAL, "price": 0, "color": None},
    # 11
    {"id": 11, "name": "St. Charles Place", "type": TILE_STREET, "color": COLOR_PINK,
     "price": 140, "rent": [10, 50, 150, 450, 625, 750], "house_cost": 100, "mortgage": 70},
    # 12
    {"id": 12, "name": "Electric Company", "type": TILE_UTILITY, "color": COLOR_UTILITY,
     "price": 150, "multipliers": [4, 10], "house_cost": 0, "mortgage": 75},
    # 13
    {"id": 13, "name": "States Avenue", "type": TILE_STREET, "color": COLOR_PINK,
     "price": 140, "rent": [10, 50, 150, 450, 625, 750], "house_cost": 100, "mortgage": 70},
    # 14
    {"id": 14, "name": "Virginia Avenue", "type": TILE_STREET, "color": COLOR_PINK,
     "price": 160, "rent": [12, 60, 180, 500, 700, 900], "house_cost": 100, "mortgage": 80},
    # 15
    {"id": 15, "name": "Pennsylvania Railroad", "type": TILE_RAILROAD, "color": COLOR_RAILROAD,
     "price": 200, "rent": [25, 50, 100, 200], "house_cost": 0, "mortgage": 100},
    # 16
    {"id": 16, "name": "St. James Place", "type": TILE_STREET, "color": COLOR_ORANGE,
     "price": 180, "rent": [14, 70, 200, 550, 750, 950], "house_cost": 100, "mortgage": 90},
    # 17
    {"id": 17, "name": "Community Chest", "type": TILE_CARD, "deck": "chest", "price": 0, "color": None},
    # 18
    {"id": 18, "name": "Tennessee Avenue", "type": TILE_STREET, "color": COLOR_ORANGE,
     "price": 180, "rent": [14, 70, 200, 550, 750, 950], "house_cost": 100, "mortgage": 90},
    # 19
    {"id": 19, "name": "New York Avenue", "type": TILE_STREET, "color": COLOR_ORANGE,
     "price": 200, "rent": [16, 80, 220, 600, 800, 1000], "house_cost": 100, "mortgage": 100},
    # 20
    {"id": 20, "name": "Free Parking", "type": TILE_SPECIAL, "price": 0, "color": None},
    # 21
    {"id": 21, "name": "Kentucky Avenue", "type": TILE_STREET, "color": COLOR_RED,
     "price": 220, "rent": [18, 90, 250, 700, 875, 1050], "house_cost": 150, "mortgage": 110},
    # 22
    {"id": 22, "name": "Chance", "type": TILE_CARD, "deck": "chance", "price": 0, "color": None},
    # 23
    {"id": 23, "name": "Indiana Avenue", "type": TILE_STREET, "color": COLOR_RED,
     "price": 220, "rent": [18, 90, 250, 700, 875, 1050], "house_cost": 150, "mortgage": 110},
    # 24
    {"id": 24, "name": "Illinois Avenue", "type": TILE_STREET, "color": COLOR_RED,
     "price": 240, "rent": [20, 100, 300, 750, 925, 1100], "house_cost": 150, "mortgage": 120},
    # 25
    {"id": 25, "name": "B. & O. Railroad", "type": TILE_RAILROAD, "color": COLOR_RAILROAD,
     "price": 200, "rent": [25, 50, 100, 200], "house_cost": 0, "mortgage": 100},
    # 26
    {"id": 26, "name": "Atlantic Avenue", "type": TILE_STREET, "color": COLOR_YELLOW,
     "price": 260, "rent": [22, 110, 330, 800, 975, 1150], "house_cost": 150, "mortgage": 130},
    # 27
    {"id": 27, "name": "Ventnor Avenue", "type": TILE_STREET, "color": COLOR_YELLOW,
     "price": 260, "rent": [22, 110, 330, 800, 975, 1150], "house_cost": 150, "mortgage": 130},
    # 28
    {"id": 28, "name": "Water Works", "type": TILE_UTILITY, "color": COLOR_UTILITY,
     "price": 150, "multipliers": [4, 10], "house_cost": 0, "mortgage": 75},
    # 29
    {"id": 29, "name": "Marvin Gardens", "type": TILE_STREET, "color": COLOR_YELLOW,
     "price": 280, "rent": [24, 120, 360, 850, 1025, 1200], "house_cost": 150, "mortgage": 140},
    # 30
    {"id": 30, "name": "Go To Jail", "type": TILE_SPECIAL, "price": 0, "color": None},
    # 31
    {"id": 31, "name": "Pacific Avenue", "type": TILE_STREET, "color": COLOR_GREEN,
     "price": 300, "rent": [26, 130, 390, 900, 1100, 1275], "house_cost": 200, "mortgage": 150},
    # 32
    {"id": 32, "name": "North Carolina Avenue", "type": TILE_STREET, "color": COLOR_GREEN,
     "price": 300, "rent": [26, 130, 390, 900, 1100, 1275], "house_cost": 200, "mortgage": 150},
    # 33
    {"id": 33, "name": "Community Chest", "type": TILE_CARD, "deck": "chest", "price": 0, "color": None},
    # 34
    {"id": 34, "name": "Pennsylvania Avenue", "type": TILE_STREET, "color": COLOR_GREEN,
     "price": 320, "rent": [28, 150, 450, 1000, 1200, 1400], "house_cost": 200, "mortgage": 160},
    # 35
    {"id": 35, "name": "Short Line", "type": TILE_RAILROAD, "color": COLOR_RAILROAD,
     "price": 200, "rent": [25, 50, 100, 200], "house_cost": 0, "mortgage": 100},
    # 36
    {"id": 36, "name": "Chance", "type": TILE_CARD, "deck": "chance", "price": 0, "color": None},
    # 37
    {"id": 37, "name": "Park Place", "type": TILE_STREET, "color": COLOR_DARK_BLUE,
     "price": 350, "rent": [35, 175, 500, 1100, 1300, 1500], "house_cost": 200, "mortgage": 175},
    # 38
    {"id": 38, "name": "Luxury Tax", "type": TILE_TAX, "price": 0, "tax_amount": 100, "color": None},
    # 39
    {"id": 39, "name": "Boardwalk", "type": TILE_STREET, "color": COLOR_DARK_BLUE,
     "price": 400, "rent": [50, 200, 600, 1400, 1700, 2000], "house_cost": 200, "mortgage": 200},
]

# Set of all purchasable property tile indices (28 total: 22 streets, 4 railroads, 2 utilities)
PURCHASABLE_TILES: List[int] = [
    t["id"] for t in TILES_DATA if t["type"] in (TILE_STREET, TILE_RAILROAD, TILE_UTILITY)
]

# Quick index lookup
TILE_LOOKUP: Dict[int, Dict[str, Any]] = {t["id"]: t for t in TILES_DATA}

# Two 6-sided dice roll probabilities P(D = d) for d in 2..12
DICE_PROBABILITIES: Dict[int, float] = {
    2: 1 / 36,
    3: 2 / 36,
    4: 3 / 36,
    5: 4 / 36,
    6: 5 / 36,
    7: 6 / 36,
    8: 5 / 36,
    9: 4 / 36,
    10: 3 / 36,
    11: 2 / 36,
    12: 1 / 36,
}
DICE_OUTCOMES: List[int] = list(range(2, 13))
DICE_PROB_VECTOR = np.array([DICE_PROBABILITIES[d] for d in DICE_OUTCOMES], dtype=np.float32)

# Unmortgage interest rate: 10%
UNMORTGAGE_INTEREST_RATE = 0.10

# House selling refund is 50% of original house cost
HOUSE_REFUND_RATE = 0.50
