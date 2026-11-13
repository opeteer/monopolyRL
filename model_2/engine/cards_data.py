"""Official Tournament Monopoly Card Decks: Chance and Community Chest.

Defines 16 Chance cards and 16 Community Chest cards, queue-based FIFO card
draw dynamics, returning cards to the bottom of the deck, and handling
Get Out of Jail Free (GOOJF) retention.
"""

from collections import deque
from typing import Dict, Any, List, Optional
import random

CHANCE_CARDS: List[Dict[str, Any]] = [
    {"id": 0, "name": "Advance to GO", "action": "advance_to", "target": 0, "collect_pass_go": True},
    {"id": 1, "name": "Advance to Illinois Ave", "action": "advance_to", "target": 24, "collect_pass_go": True},
    {"id": 2, "name": "Advance to St. Charles Place", "action": "advance_to", "target": 11, "collect_pass_go": True},
    {"id": 3, "name": "Advance to Nearest Utility", "action": "advance_nearest_utility"},
    {"id": 4, "name": "Advance to Nearest Railroad (1)", "action": "advance_nearest_railroad"},
    {"id": 5, "name": "Advance to Nearest Railroad (2)", "action": "advance_nearest_railroad"},
    {"id": 6, "name": "Bank Dividend", "action": "cash_from_bank", "amount": 50},
    {"id": 7, "name": "Get Out of Jail Free (Chance)", "action": "goojf", "deck": "chance"},
    {"id": 8, "name": "Go Back 3 Spaces", "action": "move_relative", "steps": -3},
    {"id": 9, "name": "Go to Jail", "action": "go_to_jail"},
    {"id": 10, "name": "General Repairs", "action": "property_repairs", "house_cost": 25, "hotel_cost": 100},
    {"id": 11, "name": "Speeding Fine", "action": "pay_bank", "amount": 15},
    {"id": 12, "name": "Take a Trip to Reading Railroad", "action": "advance_to", "target": 5, "collect_pass_go": True},
    {"id": 13, "name": "Elected Chairman of the Board", "action": "pay_each_player", "amount": 50},
    {"id": 14, "name": "Building Loan Matures", "action": "cash_from_bank", "amount": 150},
    {"id": 15, "name": "Advance to Boardwalk", "action": "advance_to", "target": 39, "collect_pass_go": False},
]

COMMUNITY_CHEST_CARDS: List[Dict[str, Any]] = [
    {"id": 100, "name": "Advance to GO", "action": "advance_to", "target": 0, "collect_pass_go": True},
    {"id": 101, "name": "Bank Error in Your Favor", "action": "cash_from_bank", "amount": 200},
    {"id": 102, "name": "Doctor's Fee", "action": "pay_bank", "amount": 50},
    {"id": 103, "name": "From Sale of Stock You Get $50", "action": "cash_from_bank", "amount": 50},
    {"id": 104, "name": "Get Out of Jail Free (Chest)", "action": "goojf", "deck": "chest"},
    {"id": 105, "name": "Go to Jail", "action": "go_to_jail"},
    {"id": 106, "name": "Holiday Fund Matures", "action": "cash_from_bank", "amount": 100},
    {"id": 107, "name": "Income Tax Refund", "action": "cash_from_bank", "amount": 20},
    {"id": 108, "name": "It is Your Birthday", "action": "collect_from_each_player", "amount": 10},
    {"id": 109, "name": "Life Insurance Matures", "action": "cash_from_bank", "amount": 100},
    {"id": 110, "name": "Hospital Fees", "action": "pay_bank", "amount": 100},
    {"id": 111, "name": "School Fees", "action": "pay_bank", "amount": 50},
    {"id": 112, "name": "Consultancy Fee", "action": "cash_from_bank", "amount": 25},
    {"id": 113, "name": "Street Repairs", "action": "property_repairs", "house_cost": 40, "hotel_cost": 115},
    {"id": 114, "name": "Beauty Contest", "action": "cash_from_bank", "amount": 10},
    {"id": 115, "name": "You Inherit $100", "action": "cash_from_bank", "amount": 100},
]


class CardDeckManager:
    """Manages the FIFO queue dynamics for Chance and Community Chest decks."""

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()
        self.chance_queue: deque = deque()
        self.chest_queue: deque = deque()
        self.chance_goojf_out: bool = False
        self.chest_goojf_out: bool = False
        self.reset()

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng.seed(seed)

        chance_cards = list(CHANCE_CARDS)
        chest_cards = list(COMMUNITY_CHEST_CARDS)
        self.rng.shuffle(chance_cards)
        self.rng.shuffle(chest_cards)

        self.chance_queue = deque(chance_cards)
        self.chest_queue = deque(chest_cards)
        self.chance_goojf_out = False
        self.chest_goojf_out = False

    def draw_chance(self) -> Dict[str, Any]:
        """Draws top card from Chance deck FIFO. Returns card data."""
        if not self.chance_queue:
            # Fallback if somehow empty
            return CHANCE_CARDS[0]
        card = self.chance_queue.popleft()
        if card["action"] == "goojf":
            self.chance_goojf_out = True
        else:
            # Place back at the bottom of queue
            self.chance_queue.append(card)
        return card

    def draw_chest(self) -> Dict[str, Any]:
        """Draws top card from Community Chest deck FIFO. Returns card data."""
        if not self.chest_queue:
            return COMMUNITY_CHEST_CARDS[0]
        card = self.chest_queue.popleft()
        if card["action"] == "goojf":
            self.chest_goojf_out = True
        else:
            # Place back at the bottom of queue
            self.chest_queue.append(card)
        return card

    def return_goojf(self, deck_name: str):
        """Returns a used Get Out of Jail Free card back to the bottom of its deck."""
        if deck_name == "chance":
            self.chance_goojf_out = False
            goojf_card = next(c for c in CHANCE_CARDS if c["action"] == "goojf")
            self.chance_queue.append(goojf_card)
        elif deck_name == "chest":
            self.chest_goojf_out = False
            goojf_card = next(c for c in COMMUNITY_CHEST_CARDS if c["action"] == "goojf")
            self.chest_queue.append(goojf_card)

    def get_state_vector(self) -> List[float]:
        """Returns normalized deck representation for Centralized Critic global state.

        Encodes card IDs in the current queue order.
        """
        # Encode top 4 cards of each deck (or all cards) normalized by 120.0
        chance_ids = [c["id"] / 120.0 for c in self.chance_queue]
        # Pad if goojf is out
        while len(chance_ids) < 16:
            chance_ids.append(-1.0)

        chest_ids = [c["id"] / 120.0 for c in self.chest_queue]
        while len(chest_ids) < 16:
            chest_ids.append(-1.0)

        return chance_ids + chest_ids
