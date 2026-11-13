"""Tournament-Grade Official Monopoly State Machine Engine.

Implements all official tournament rules:
- Physical bank scarcity: exactly 32 houses and 12 hotels.
- Mandatory auction on landed unowned property if active player declines purchase.
- Speeding penalty: 3 consecutive doubles in one turn sends player to Jail immediately.
- Jail mechanics: up to 3 turns, roll doubles, pay $50, or use GOOJF card.
- Even building rule across color group properties.
- Mortgages (50% value) and 10% unmortgage interest.
- FIFO Card decks for Chance and Community Chest.
- Bilateral trading protocol.
- Potential-Based Reward Shaping (PBRS) maintaining policy invariance.
"""

from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
import random
import math
import os
import time
import hashlib

from monopoly_marl.engine.board_data import (
    TILES_DATA, COLOR_GROUPS, TILE_TO_GROUP, GROUP_MONOPOLY_BONUS,
    PURCHASABLE_TILES, TILE_STREET, TILE_RAILROAD, TILE_UTILITY,
    TILE_TAX, TILE_CARD, TILE_SPECIAL, UNMORTGAGE_INTEREST_RATE,
    HOUSE_REFUND_RATE, DICE_OUTCOMES, DICE_PROBABILITIES
)
from monopoly_marl.engine.cards_data import CardDeckManager

# Phase Definitions
PHASE_ROLL = 0
PHASE_BUY_OR_AUCTION = 1
PHASE_AUCTION = 2
PHASE_JAIL_DECISION = 3
PHASE_PROPERTY_MANAGE = 4
PHASE_TRADE_RESPONSE = 5
PHASE_GAME_OVER = 6
NUM_PHASES = 7

# Core Action Head (6 dimensions)
ACT_CORE_ROLL_PASS = 0
ACT_CORE_BUY_DIRECT = 1
ACT_CORE_DECLINE_BUY = 2
ACT_CORE_PAY_JAIL_50 = 3
ACT_CORE_USE_GOOJF = 4
ACT_CORE_END_MANAGE = 5

# Auction Action Head (7 dimensions: discrete increments + strategic jump bids)
ACT_AUC_FOLD = 0
ACT_AUC_BID_1 = 1
ACT_AUC_BID_10 = 2
ACT_AUC_BID_50 = 3
ACT_AUC_BID_100 = 4
ACT_AUC_JUMP_LIST = 5      # Jump bid directly to property printed list price
ACT_AUC_JUMP_SQUEEZE = 6  # Jump bid to squeeze max active opponent cash (+ $1)

# Trade Cash Brackets (10 options)
TRADE_CASH_BRACKETS = [0, 25, 50, 100, 150, 200, 250, 300, 400, 500]

# Default Tournament Agent Names (P0: Luke, P1: Graham, P2: Pieter, P3: Allison)
DEFAULT_PLAYER_NAMES = ["Luke", "Graham", "Pieter", "Allison"]

# Precomputed static lookup tables for high-performance zero-overhead rollout
TILE_PRICES = tuple(float(TILES_DATA[p]["price"]) if "price" in TILES_DATA[p] else 0.0 for p in range(40))
TILE_MORTGAGES = tuple(float(TILES_DATA[p].get("mortgage", 0.0)) for p in range(40))
TILE_HOUSE_COSTS = tuple(float(TILES_DATA[p].get("house_cost", 0.0)) for p in range(40))
PURCHASABLE_TILES_TUPLE = tuple(PURCHASABLE_TILES)
COLOR_GROUPS_MONOPOLY = tuple(
    (tuple(tiles), float(GROUP_MONOPOLY_BONUS.get(grp_name, 0.0)), grp_name not in ("RAILROAD", "UTILITY"))
    for grp_name, tiles in COLOR_GROUPS.items()
)
STREET_GROUPS_SYNERGY = tuple(
    (grp_name, tuple(tiles), float(len(tiles)))
    for grp_name, tiles in COLOR_GROUPS.items()
    if grp_name not in ("RAILROAD", "UTILITY")
)
DICE_PROB_TUPLE = tuple((d, DICE_PROBABILITIES[d]) for d in DICE_OUTCOMES)

INACTIVE_MASKS_TEMPLATE = {
    "core": np.array([1, 0, 0, 0, 0, 0], dtype=np.int8),
    "auction": np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.int8),
    "property": np.zeros(161, dtype=np.int8),
    "trade_target": np.array([1, 0, 0], dtype=np.int8),
    "trade_offer_prop": np.zeros(41, dtype=np.int8),
    "trade_offer_cash": np.zeros(10, dtype=np.int8),
    "trade_req_prop": np.zeros(41, dtype=np.int8),
    "trade_req_cash": np.zeros(10, dtype=np.int8),
    "trade_response": np.array([1, 0], dtype=np.int8),
}
INACTIVE_MASKS_TEMPLATE["property"][160] = 1
INACTIVE_MASKS_TEMPLATE["trade_offer_prop"][40] = 1
INACTIVE_MASKS_TEMPLATE["trade_offer_cash"][0] = 1
INACTIVE_MASKS_TEMPLATE["trade_req_prop"][40] = 1
INACTIVE_MASKS_TEMPLATE["trade_req_cash"][0] = 1


def generate_hardware_entropy_seed() -> int:
    """Generates a cryptographically strong, high-entropy seed derived from real hardware fluctuations.

    Entropy sources:
    1. OS kernel cryptographic hardware entropy pool (/dev/urandom, CPU RDRAND/RDSEED, hardware interrupts, thermal noise)
    2. Sub-nanosecond CPU cycle counter fluctuation (time.perf_counter_ns)
    3. Monotonic system clock jitter (time.monotonic_ns)
    4. Wall clock timing jitter (time.time_ns)
    5. Process and runtime scheduling jitter (os.getpid)

    Hashed with SHA-256 and truncated to 32-bit unsigned integer (0 <= seed < 2**32)
    suitable for all standard PRNGs.
    """
    hw_bytes = os.urandom(32)
    perf_bytes = time.perf_counter_ns().to_bytes(8, byteorder="big", signed=False)
    mono_bytes = time.monotonic_ns().to_bytes(8, byteorder="big", signed=False)
    time_bytes = time.time_ns().to_bytes(8, byteorder="big", signed=False)
    pid_bytes = os.getpid().to_bytes(4, byteorder="big", signed=False)

    entropy_payload = hw_bytes + perf_bytes + mono_bytes + time_bytes + pid_bytes
    digest = hashlib.sha256(entropy_payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


class MonopolyCore:
    """Official Tournament-Grade Monopoly Game State Machine."""

    def __init__(
        self,
        seed: Optional[int] = None,
        max_rounds: Optional[int] = None,
        gamma: float = 0.997,
        w1: float = 0.050,
        w2: float = 0.0060,
        w3: float = 0.008,
        w4: float = 0.0005,
        w_syn: float = 0.020,
        beta_starve: float = 3.5,
        w_house: float = 0.012,
        w_cash: float = 0.0035,
        allow_trade: bool = True,
        include_synergy_obs: bool = True,
        player_names: Optional[List[str]] = None,
        house_rules: Optional[Any] = None,
        randomize_first_player: bool = False,
    ):
        if seed is None:
            seed = generate_hardware_entropy_seed()
        self.current_seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.max_rounds = max_rounds  # None indicates unbounded horizon (infinite survival)
        self.gamma = gamma
        self.randomize_first_player = bool(randomize_first_player)
        self.player_names = list(player_names) if player_names is not None else list(DEFAULT_PLAYER_NAMES)

        # Modular Custom House Rules System
        from monopoly_marl.house_rules.manager import HouseRulesManager
        self.house_rules = HouseRulesManager.from_config(house_rules)

        # Well-scaled, Anti-Exploit Offensive Arms-Race PBRS Weights
        self.w1 = w1           # Cash log safety potential (0.050)
        self.w2 = w2           # Property equity potential (0.0060: balanced against linear cash)
        self.w3 = w3           # Monopoly completion bonus potential (0.008: +0.96 to +7.36)
        self.w4 = w4           # Rent hazard paranoia (0.0005: low defensive paranoia)
        self.w_syn = w_syn     # Color-matching synergy quadratic potential (0.020)
        self.beta_starve = beta_starve  # Round-based monopoly starvation penalty (3.5: non-exploitable)
        self.w_house = w_house # Building house/hotel equity multiplier (0.012: strong development reward)
        self.w_cash = w_cash   # Linear cash equity weight (0.0035: prices 10% mortgage burn & auction/trade cash fairly)
        self.allow_trade = allow_trade
        self.include_synergy_obs = include_synergy_obs

        self.deck_manager = CardDeckManager(self.rng)
        self.reset(seed)

    def set_allow_trade(self, allow: bool):
        """Curriculum learning toggle: enable or disable trade proposals via action mask."""
        self.allow_trade = bool(allow)

    def get_player_name(self, player_id: int) -> str:
        """Returns the tournament agent name for player_id."""
        if 0 <= player_id < len(self.player_names):
            return self.player_names[player_id]
        return f"Player {player_id}"

    def _init_effective_tile_costs(self):
        """Precomputes per-tile effective purchase prices, house costs, and mortgages."""
        prices = [0] * 40
        house_costs = [0] * 40
        mortgages = [0] * 40
        prices_f = [0.0] * 40
        house_costs_f = [0.0] * 40
        mortgages_f = [0.0] * 40

        for tid in range(40):
            tile = TILES_DATA[tid]
            eff_p = None
            if "price" in tile:
                eff_p = self.house_rules.get_property_price(tid, int(tile["price"]))
                prices[tid] = eff_p
                prices_f[tid] = float(eff_p)

            if "house_cost" in tile:
                eff_hc = self.house_rules.get_house_cost(tid, int(tile["house_cost"]))
                house_costs[tid] = eff_hc
                house_costs_f[tid] = float(eff_hc)

            if "mortgage" in tile:
                eff_m = self.house_rules.get_mortgage_value(
                    tid, int(tile["mortgage"]), modified_price=eff_p
                )
                mortgages[tid] = eff_m
                mortgages_f[tid] = float(eff_m)

        self.tile_prices = tuple(prices)
        self.tile_house_costs = tuple(house_costs)
        self.tile_mortgages = tuple(mortgages)
        self._tile_prices_f = tuple(prices_f)
        self._tile_house_costs_f = tuple(house_costs_f)
        self._tile_mortgages_f = tuple(mortgages_f)

    def get_property_price(self, tile_id: int, default: int = 0) -> int:
        """Returns effective purchase price of tile_id after active House Rules."""
        if tile_id is not None and 0 <= tile_id < 40 and self.tile_prices[tile_id] > 0:
            return self.tile_prices[tile_id]
        return default

    def get_property_house_cost(self, tile_id: int, default: int = 0) -> int:
        """Returns effective house/hotel build cost of tile_id after active House Rules."""
        if tile_id is not None and 0 <= tile_id < 40 and self.tile_house_costs[tile_id] > 0:
            return self.tile_house_costs[tile_id]
        return default

    def get_property_mortgage(self, tile_id: int, default: int = 0) -> int:
        """Returns effective mortgage value of tile_id after active House Rules."""
        if tile_id is not None and 0 <= tile_id < 40 and self.tile_mortgages[tile_id] > 0:
            return self.tile_mortgages[tile_id]
        return default

    def reset(
        self, seed: Optional[int] = None, randomize_first_player: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Resets the environment to initial tournament starting state."""
        if seed is None:
            seed = generate_hardware_entropy_seed()

        self.current_seed = seed
        self.rng.seed(seed)
        self.np_rng = np.random.default_rng(seed)
        self.deck_manager.reset(seed)
        self._init_effective_tile_costs()

        # 4 Players: 0, 1, 2, 3 (Cash determined by official rules or custom House Rules)
        starting_cash = self.house_rules.get_starting_cash(default_cash=1500)
        self.cash = [starting_cash, starting_cash, starting_cash, starting_cash]
        self.position = [0, 0, 0, 0]
        self.in_jail = [False, False, False, False]
        self.jail_turns = [0, 0, 0, 0]
        self.goojf_cards = [0, 0, 0, 0]
        self.goojf_sources: List[List[str]] = [[], [], [], []]
        self.bankrupt = [False, False, False, False]

        # Bank Physical Scarcity
        self.bank_houses = 32
        self.bank_hotels = 12

        # 40 Tiles State
        # Owner: -1 if unowned / not purchasable, 0..3 if owned
        self.property_owner = [-1] * 40
        # Houses: 0 to 5 (5 = hotel)
        self.property_houses = [0] * 40
        # Mortgaged flags
        self.property_mortgaged = [False] * 40

        # Turn & State Machine Trackers
        use_rand_first = self.randomize_first_player if randomize_first_player is None else bool(randomize_first_player)
        self.starting_player = int(self.rng.randint(0, 3)) if use_rand_first else 0
        self.current_player = self.starting_player
        self.active_decision_player = self.starting_player
        self.last_creditor: Optional[int] = None
        self.round_count = 0
        self.turn_step = 0
        self.consecutive_doubles = 0
        self.last_dice: Tuple[int, int] = (1, 1)
        self.phase = PHASE_ROLL

        # Sub-phase specifics
        self.landed_property: Optional[int] = None
        
        # Auction state
        self.auction_property: Optional[int] = None
        self.auction_highest_bid = 0
        self.auction_highest_bidder = -1
        self.auction_active_bidders: List[int] = []
        self.auction_current_bidder_idx = 0

        # Trade proposal state & telemetry counters
        self.pending_trade: Optional[Dict[str, Any]] = None
        self.manage_actions_this_turn = 0
        self.trade_proposals_this_turn = 0
        self.trades_proposed_count = 0
        self.trades_accepted_count = 0
        self.houses_built_count = 0

        # Same-turn thrash guards & rejected-trade cooldowns
        self.mortgaged_this_turn: Set[int] = set()
        self.unmortgaged_this_turn: Set[int] = set()
        self.built_this_turn: Set[int] = set()
        self.sold_this_turn: Set[int] = set()
        self.rejected_req_prop_cooldown: Dict[Tuple[int, int, int], int] = {}
        self.rejected_offer_prop_cooldown: Dict[Tuple[int, int, int], int] = {}

        # Event log history for UI / replay
        if self.house_rules.active_rules:
            self.event_log: List[str] = [
                f"Tournament game started with House Rules: {self.house_rules.get_rule_summary()}. All players initialized with ${starting_cash}."
            ]
            self.house_rules.apply_game_init(self)
        else:
            self.event_log: List[str] = ["Tournament game started. All players initialized with $1500."]
        if self.starting_player != 0:
            self.event_log.append(f"First turn assigned to {self.get_player_name(self.starting_player)}.")
        self.last_card_drawn: Optional[Dict[str, Any]] = None

        # Precompute initial potentials for PBRS
        self.prev_potentials = [self._compute_potential(p) for p in range(4)]

        # If player 0 starts in jail (not standard, but handle general case)
        if self.in_jail[self.current_player]:
            self.phase = PHASE_JAIL_DECISION
        else:
            self.phase = PHASE_ROLL
        self.active_decision_player = self.current_player

        return self.get_observations()

    # ----------------------------------------------------------------------
    # PBRS & Value Calculations
    # ----------------------------------------------------------------------
    def _compute_potential(self, player_id: int) -> float:
        """Computes Φ^i(s) potential function as mathematically specified:

        Φ^i(s) = w1 * ln(c_i + 1)
                 + w2 * sum_{p in O_i} (V_base(p) + sum_{l=1}^{h_p} V_bld(p, l) - m_p * V_mtg(p))
                 + w3 * sum_{g in G} I(g subseteq O_i) * U_g
                 - w4 * sum_{j != i} sum_{d=2}^{12} P(D=d) * Rent_{j -> i}((x_i + d) % 40)
        """
        if self.bankrupt[player_id]:
            return 0.0

        # Term 1: w1 * ln(c_i + 1) + w_cash * c_i (log liquidity safety + linear cash value)
        c_i = max(0, self.cash[player_id])
        term1 = self.w1 * math.log(c_i + 1.0) + self.w_cash * float(c_i)

        # Term 2: Property equity + House/Hotel development value
        term2_prop = 0.0
        term2_house = 0.0
        for p in PURCHASABLE_TILES_TUPLE:
            if self.property_owner[p] == player_id:
                m_p = 1.0 if self.property_mortgaged[p] else 0.0
                term2_prop += (self._tile_prices_f[p] - m_p * self._tile_mortgages_f[p])
                term2_house += float(self.property_houses[p] * self._tile_house_costs_f[p])
        term2 = (self.w2 * term2_prop) + (self.w_house * term2_house)

        # Term 3: Monopoly complete groups
        term3_sum = 0.0
        street_monopolies_count = 0
        for tiles, bonus, is_street in COLOR_GROUPS_MONOPOLY:
            if all(self.property_owner[t] == player_id for t in tiles):
                term3_sum += bonus
                if is_street:
                    street_monopolies_count += 1
        term3 = self.w3 * term3_sum

        # Term 4: Expected rent risk from next roll (de-escalated to avoid paranoid trade embargo)
        pos_i = self.position[player_id]
        term4_sum = 0.0
        if not self.in_jail[player_id]:
            for d, prob in DICE_PROB_TUPLE:
                target_tile = (pos_i + d) % 40
                owner = self.property_owner[target_tile]
                if owner != -1 and owner != player_id and not self.bankrupt[owner] and not self.property_mortgaged[target_tile]:
                    rent = self._calculate_rent(target_tile, dice_sum=d)
                    term4_sum += prob * rent
        term4 = self.w4 * term4_sum

        # Term 5: Color-matching synergy quadratic potential
        term5_sum = 0.0
        for grp_name, tiles, n_tiles in STREET_GROUPS_SYNERGY:
            owned_count = sum(1 for t in tiles if self.property_owner[t] == player_id)
            frac = owned_count / n_tiles
            term5_sum += (frac ** 2) * 50.0
        term5 = self.w_syn * term5_sum

        # Term 6: Anti-Exploit Round-Based Monopoly Starvation Penalty
        # Keyed strictly to round_count (unstoppable time progression) rather than sold_frac,
        # ensuring purchasing unowned properties is NEVER penalized (eliminating the Cobra Effect).
        round_frac = min(1.0, self.round_count / 40.0)
        term6_starve = 0.0
        if street_monopolies_count == 0:
            term6_starve = self.beta_starve * (round_frac ** 2)

        return float(term1 + term2 + term3 - term4 + term5 - term6_starve)

    def get_color_synergy_vector(self) -> np.ndarray:
        """Constructs 64-dim color-matching synergy representation:
        - 32 dims: Fraction of color group owned (4 players x 8 street color groups)
        - 32 dims: Indicator whether player needs exactly 1 missing piece held by opponent (4 players x 8 color groups)
        """
        fractions = []
        needs_one = []

        for grp, tiles, n_tiles in STREET_GROUPS_SYNERGY:
            owners = [self.property_owner[t] for t in tiles]

            for pl in range(4):
                c = sum(1 for o in owners if o == pl)
                fractions.append(c / n_tiles)
                # Needs one: pl has n_tiles - 1, and missing tile is held by an opponent
                if c == int(n_tiles - 1):
                    missing_owner = next(o for o in owners if o != pl)
                    if missing_owner != -1:
                        needs_one.append(1.0)
                    else:
                        needs_one.append(0.0)
                else:
                    needs_one.append(0.0)

        return np.array(fractions + needs_one, dtype=np.float32)

    def evaluate_trade_rationality(self, trade: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluates game-theoretic rationality of trade for target player:
        1. Cash feasibility & post-trade liquidity safety margin
        2. Monopoly protection & anti-kingmaking constraint
        3. Color-matching synergy check
        """
        if not trade:
            return False, "No active trade"

        init = trade["initiator"]
        target = trade["target"]
        offer_p = trade["offer_prop"]
        req_p = trade["req_prop"]
        offer_c = trade["offer_cash"]
        req_c = trade["req_cash"]

        # 1. Cash Feasibility
        if self.cash[target] < req_c or self.cash[init] < offer_c:
            return False, "Insufficient cash for transaction"

        # Post-trade liquidity safety margin for target
        target_cash_after = self.cash[target] + offer_c - req_c
        if target_cash_after < 50:
            return False, "Post-trade liquidity squeeze: target cash would fall below $50 safety buffer"

        # 2. Check if target is giving away a monopoly piece
        if req_p < 40 and req_p in PURCHASABLE_TILES:
            req_grp = TILES_DATA[req_p].get("color")
            if req_grp and req_grp in COLOR_GROUPS:
                target_owns_group = all(self.property_owner[t] == target for t in COLOR_GROUPS[req_grp])
                if target_owns_group:
                    # Giving away part of an existing monopoly is irrational unless target gets a monopoly in return
                    target_gets_monopoly = False
                    if offer_p < 40 and offer_p in PURCHASABLE_TILES:
                        offer_grp = TILES_DATA[offer_p].get("color")
                        if offer_grp and offer_grp in COLOR_GROUPS:
                            target_owned = sum(1 for t in COLOR_GROUPS[offer_grp] if self.property_owner[t] == target)
                            if target_owned == len(COLOR_GROUPS[offer_grp]) - 1:
                                target_gets_monopoly = True
                    if not target_gets_monopoly:
                        return False, "Monopoly protection: Cannot break existing complete monopoly without receiving a monopoly"

        # 3. Anti-Kingmaking check: Does initiator get a monopoly while target gets nothing?
        init_gets_monopoly = False
        if req_p < 40 and req_p in PURCHASABLE_TILES:
            req_grp = TILES_DATA[req_p].get("color")
            if req_grp and req_grp in COLOR_GROUPS:
                init_owned = sum(1 for t in COLOR_GROUPS[req_grp] if self.property_owner[t] == init)
                if init_owned == len(COLOR_GROUPS[req_grp]) - 1:
                    init_gets_monopoly = True

        target_gets_monopoly = False
        if offer_p < 40 and offer_p in PURCHASABLE_TILES:
            offer_grp = TILES_DATA[offer_p].get("color")
            if offer_grp and offer_grp in COLOR_GROUPS:
                target_owned = sum(1 for t in COLOR_GROUPS[offer_grp] if self.property_owner[t] == target)
                if target_owned == len(COLOR_GROUPS[offer_grp]) - 1:
                    target_gets_monopoly = True

        if init_gets_monopoly and not target_gets_monopoly:
            # Initiator gets monopoly, but target does not.
            # Only acceptable if target receives substantial cash compensation (net gain >= $200)
            net_cash_gain = offer_c - req_c
            if net_cash_gain < 200:
                return False, "Anti-Kingmaking: Opponent would gain monopoly without sufficient cash compensation (net gain < $200)"

        return True, "Trade is game-theoretically rational"

    def compute_hazard_threat(self, player_id: int) -> float:
        """Computes expected immediate rent hazard threat in next roll (12 tiles ahead)."""
        if self.in_jail[player_id]:
            return 0.0
        hazard_threat = 0.0
        pos = self.position[player_id]
        for d in DICE_OUTCOMES:
            prob = DICE_PROBABILITIES[d]
            target_t = (pos + d) % 40
            owner = self.property_owner[target_t]
            if owner != -1 and owner != player_id and not self.bankrupt[owner] and not self.property_mortgaged[target_t]:
                h = self.property_houses[target_t]
                if h >= 2:
                    rent = self._calculate_rent(target_t, dice_sum=d)
                    hazard_threat += prob * rent
        return hazard_threat

    def compute_bidding_ceiling(self, player_id: int, tile_id: int) -> int:
        """Computes game-theoretic dynamic bidding ceiling for player_id on tile_id:
        1. Player cash liquidity
        2. Environmental hazard proximity (threat of opponent houses/hotels nearby)
        3. Round progress / Endgame status
        4. Bank house scarcity
        5. Color tier & synergy value vs denial/spite value
        """
        if tile_id is None or tile_id < 0 or tile_id >= len(TILES_DATA):
            return 0

        tile = TILES_DATA[tile_id]
        cash = self.cash[player_id]
        if cash <= 1:
            return 0

        price = self.get_property_price(tile_id, tile.get("price", 100))
        mortgage = self.get_property_mortgage(tile_id, price // 2)
        grp = tile.get("color")

        # 1. Environmental Factor A: Proximity to deadly opponent properties (Hazard Threat)
        hazard_threat = self.compute_hazard_threat(player_id)

        # Dynamic safety buffer based on cash wealth and hazard threat
        base_reserve = 50 if cash > 1000 else (100 if cash > 500 else 150)
        safety_reserve = max(base_reserve, int(hazard_threat * 0.8))
        available_budget = max(0, cash - safety_reserve)

        # 2. Environmental Factor B: Bank house scarcity
        scarcity_factor = 0.75 if self.bank_houses == 0 else 1.0

        # 3. Environmental Factor C: Color tier multiplier
        tier_mult = 1.4 if grp in ("ORANGE", "RED") else (0.6 if grp in ("UTILITY", "BROWN") else 1.0)

        # 4. Check personal synergy: does tile_id complete or advance player_id's monopoly?
        completes_my_monopoly = False
        advances_my_monopoly = False
        if grp and grp in COLOR_GROUPS:
            group_tiles = COLOR_GROUPS[grp]
            owned_by_me = sum(1 for t in group_tiles if self.property_owner[t] == player_id)
            if owned_by_me == len(group_tiles) - 1:
                completes_my_monopoly = True
            elif owned_by_me >= 1:
                advances_my_monopoly = True

        # 5. Check opponent threat: does tile_id complete an opponent's monopoly?
        opponent_gets_monopoly = False
        threat_opp_cash = 0
        for pl in range(4):
            if pl != player_id and not self.bankrupt[pl]:
                if grp and grp in COLOR_GROUPS:
                    group_tiles = COLOR_GROUPS[grp]
                    owned_by_opp = sum(1 for t in group_tiles if self.property_owner[t] == pl)
                    if owned_by_opp == len(group_tiles) - 1:
                        opponent_gets_monopoly = True
                        threat_opp_cash = max(threat_opp_cash, self.cash[pl])

        # Calculate Valuation Ceiling
        if completes_my_monopoly:
            val = int(price * 2.5 * tier_mult * scarcity_factor)
            ceiling = min(available_budget, max(price, val))
        elif advances_my_monopoly:
            val = int(price * 1.5 * tier_mult * scarcity_factor)
            ceiling = min(available_budget, max(mortgage + 20, val))
        elif opponent_gets_monopoly:
            aggression_pct = 0.40 if cash > 1000 else (0.25 if cash > 500 else 0.10)
            spite_ceiling = mortgage + int(aggression_pct * available_budget)
            # Spite bidding ceiling is strictly bounded by the threatening opponent's cash to avoid winner's curse
            ceiling = min(available_budget, spite_ceiling, threat_opp_cash)
        else:
            ceiling = min(available_budget, price)

        return max(1, ceiling)

    def get_property_transaction_features(self, player_id: int) -> np.ndarray:
        """Constructs 8-dim smart property transaction features active during
        both PHASE_BUY_OR_AUCTION (landed_property) and PHASE_AUCTION (auction_property):
        1. is_monopoly_piece_for_me (0.0 or 1.0)
        2. is_monopoly_piece_for_opponent (0.0 or 1.0)
        3. max_opponent_cash_ratio (max_opp_cash / price)
        4. current_bid_to_list_ratio (current_bid / price)
        5. post_transaction_cash_ratio ((cash - cost) / 1000.0)
        6. competitors_ratio (remaining_competitors / 4.0)
        7. property_tier_score (0.2 to 1.0)
        8. hazard_proximity_score (hazard_threat / 200.0)
        """
        target_prop = None
        if self.phase == PHASE_AUCTION:
            target_prop = self.auction_property
        elif self.phase == PHASE_BUY_OR_AUCTION:
            target_prop = self.landed_property

        if target_prop is None or target_prop < 0 or target_prop >= len(TILES_DATA):
            return np.zeros(8, dtype=np.float32)

        tile = TILES_DATA[target_prop]
        price = float(self.get_property_price(target_prop, tile.get("price", 100)))
        grp = tile.get("color")

        # 1. Monopoly for me
        for_me = 0.0
        if grp and grp in COLOR_GROUPS:
            group_tiles = COLOR_GROUPS[grp]
            if sum(1 for t in group_tiles if self.property_owner[t] == player_id) == len(group_tiles) - 1:
                for_me = 1.0

        # 2. Monopoly for opponent (or highest bidder in auction)
        for_opp = 0.0
        if self.phase == PHASE_AUCTION and self.auction_highest_bidder != -1:
            hb = self.auction_highest_bidder
            if grp and grp in COLOR_GROUPS:
                group_tiles = COLOR_GROUPS[grp]
                if sum(1 for t in group_tiles if self.property_owner[t] == hb) == len(group_tiles) - 1:
                    for_opp = 1.0
        elif grp and grp in COLOR_GROUPS:
            group_tiles = COLOR_GROUPS[grp]
            for pl in range(4):
                if pl != player_id and not self.bankrupt[pl]:
                    if sum(1 for t in group_tiles if self.property_owner[t] == pl) == len(group_tiles) - 1:
                        for_opp = 1.0
                        break

        # 3. Max opponent cash ratio
        max_opp_c = max([self.cash[pl] for pl in range(4) if pl != player_id and not self.bankrupt[pl]] or [0])
        opp_cash_ratio = min(3.0, max_opp_c / max(1.0, price))

        # 4. Current bid to list ratio
        if self.phase == PHASE_AUCTION:
            bid_ratio = min(4.0, self.auction_highest_bid / max(1.0, price))
        else:
            bid_ratio = 0.0

        # 5. Post transaction cash ratio
        cash = self.cash[player_id]
        trans_cost = self.auction_highest_bid if self.phase == PHASE_AUCTION else price
        post_cash = max(0.0, cash - trans_cost)
        post_cash_ratio = min(2.0, post_cash / 1000.0)

        # 6. Competitors ratio
        if self.phase == PHASE_AUCTION:
            competitors_ratio = len(self.auction_active_bidders) / 4.0
        else:
            affording_opps = sum(1 for pl in range(4) if pl != player_id and not self.bankrupt[pl] and self.cash[pl] >= price)
            competitors_ratio = affording_opps / 3.0

        # 7. Tier score
        tier_score = 1.0 if grp in ("ORANGE", "RED") else (0.8 if grp in ("YELLOW", "DARK_BLUE", "LIGHT_BLUE") else 0.5)

        # 8. Hazard threat
        hazard = self.compute_hazard_threat(player_id)
        hazard_score = min(2.0, hazard / 200.0)

        return np.array([
            for_me, for_opp, opp_cash_ratio, bid_ratio,
            post_cash_ratio, competitors_ratio, tier_score, hazard_score
        ], dtype=np.float32)

    def get_auction_features(self, player_id: int) -> np.ndarray:
        """Alias for backward compatibility."""
        return self.get_property_transaction_features(player_id)

    def get_trade_features(self, player_id: int) -> np.ndarray:
        """Constructs 101-dim trade context vector during PHASE_TRADE_RESPONSE:
        - 1 dim: trade_active flag
        - 4 dims: initiator one-hot
        - 4 dims: target one-hot
        - 41 dims: offered property one-hot (0..39 tile, 40 = None)
        - 41 dims: requested property one-hot (0..39 tile, 40 = None)
        - 2 dims: [offer_cash / 1000.0, req_cash / 1000.0]
        - 8 dims: objective strategic trade impact descriptors from player_id's perspective
        Total = 1 + 4 + 4 + 41 + 41 + 2 + 8 = 101 dims.
        """
        if self.phase != PHASE_TRADE_RESPONSE or self.pending_trade is None:
            return np.zeros(101, dtype=np.float32)

        t = self.pending_trade
        init = t["initiator"]
        target = t["target"]
        offer_p = t["offer_prop"]
        req_p = t["req_prop"]
        offer_c = float(t["offer_cash"])
        req_c = float(t["req_cash"])

        trade_active = [1.0]
        init_oh = [0.0] * 4
        if 0 <= init < 4:
            init_oh[init] = 1.0
        targ_oh = [0.0] * 4
        if 0 <= target < 4:
            targ_oh[target] = 1.0

        off_p_oh = [0.0] * 41
        off_p_oh[offer_p if (0 <= offer_p < 40) else 40] = 1.0

        req_p_oh = [0.0] * 41
        req_p_oh[req_p if (0 <= req_p < 40) else 40] = 1.0

        cash_terms = [offer_c / 1000.0, req_c / 1000.0]

        # Perspective-relative gain/give
        if player_id == target:
            gain_p, give_p = offer_p, req_p
            gain_c, give_c = offer_c, req_c
            other_p = init
        else:
            gain_p, give_p = req_p, offer_p
            gain_c, give_c = req_c, offer_c
            other_p = target

        # Hypothetical ownership post-trade
        sim_owner = list(self.property_owner)
        if 0 <= offer_p < 40 and sim_owner[offer_p] == init:
            sim_owner[offer_p] = target
        if 0 <= req_p < 40 and sim_owner[req_p] == target:
            sim_owner[req_p] = init

        def _count_street_monopolies(pl: int, owners: List[int]) -> int:
            cnt = 0
            for grp_name, tiles in COLOR_GROUPS.items():
                if grp_name in ("RAILROAD", "UTILITY"):
                    continue
                if all(owners[tile_idx] == pl for tile_idx in tiles):
                    cnt += 1
            return cnt

        my_mono_before = _count_street_monopolies(player_id, self.property_owner)
        my_mono_after = _count_street_monopolies(player_id, sim_owner)
        opp_mono_before = _count_street_monopolies(other_p, self.property_owner) if 0 <= other_p < 4 else 0
        opp_mono_after = _count_street_monopolies(other_p, sim_owner) if 0 <= other_p < 4 else 0

        completes_my_monopoly = 1.0 if my_mono_after > my_mono_before else 0.0
        completes_opp_monopoly = 1.0 if opp_mono_after > opp_mono_before else 0.0

        advances_my_monopoly = 0.0
        if 0 <= gain_p < 40:
            grp_g = TILES_DATA[gain_p].get("color")
            if grp_g and grp_g in COLOR_GROUPS:
                if sum(1 for tile_idx in COLOR_GROUPS[grp_g] if self.property_owner[tile_idx] == player_id) >= 1:
                    advances_my_monopoly = 1.0

        breaks_my_pair_or_monopoly = 0.0
        if 0 <= give_p < 40:
            grp_gv = TILES_DATA[give_p].get("color")
            if grp_gv and grp_gv in COLOR_GROUPS:
                if sum(1 for tile_idx in COLOR_GROUPS[grp_gv] if self.property_owner[tile_idx] == player_id) >= 2:
                    breaks_my_pair_or_monopoly = 1.0

        escapes_starvation = 1.0 if (my_mono_before == 0 and my_mono_after >= 1) else 0.0
        is_mutual_monopoly_trade = 1.0 if (completes_my_monopoly > 0.5 and completes_opp_monopoly > 0.5) else 0.0
        net_cash_gain = (gain_c - give_c) / 1000.0
        gain_val = float(self.get_property_price(gain_p, 0)) if (0 <= gain_p < 40) else 0.0
        give_val = float(self.get_property_price(give_p, 0)) if (0 <= give_p < 40) else 0.0
        net_equity_gain = ((gain_val + gain_c) - (give_val + give_c)) / 1000.0

        strategic_impact = [
            completes_my_monopoly,
            completes_opp_monopoly,
            advances_my_monopoly,
            breaks_my_pair_or_monopoly,
            escapes_starvation,
            is_mutual_monopoly_trade,
            net_cash_gain,
            net_equity_gain,
        ]

        return np.array(
            trade_active + init_oh + targ_oh + off_p_oh + req_p_oh + cash_terms + strategic_impact,
            dtype=np.float32
        )

    def _calculate_rent(self, tile_id: int, dice_sum: int, multiplier_override: Optional[int] = None) -> int:
        """Calculates official rent for tile given current state."""
        tile = TILES_DATA[tile_id]
        owner = self.property_owner[tile_id]
        if owner == -1 or self.property_mortgaged[tile_id]:
            return 0

        t_type = tile["type"]
        if t_type == TILE_STREET:
            h = self.property_houses[tile_id]
            rents = tile["rent"]
            if h == 0:
                # Double rent if owner holds entire color group AND none are mortgaged
                grp = tile["color"]
                group_tiles = COLOR_GROUPS[grp]
                all_owned = all(self.property_owner[t] == owner for t in group_tiles)
                none_mortgaged = all(not self.property_mortgaged[t] for t in group_tiles)
                return rents[0] * 2 if (all_owned and none_mortgaged) else rents[0]
            else:
                return rents[h]

        elif t_type == TILE_RAILROAD:
            rr_tiles = COLOR_GROUPS["RAILROAD"]
            rr_owned = sum(1 for t in rr_tiles if self.property_owner[t] == owner and not self.property_mortgaged[t])
            base_rent = 25 * (2 ** (max(1, rr_owned) - 1))
            if multiplier_override:
                return base_rent * multiplier_override
            return base_rent

        elif t_type == TILE_UTILITY:
            ut_tiles = COLOR_GROUPS["UTILITY"]
            ut_owned = sum(1 for t in ut_tiles if self.property_owner[t] == owner and not self.property_mortgaged[t])
            mult = 10 if ut_owned == 2 or multiplier_override == 10 else 4
            return mult * dice_sum

        return 0

    # ----------------------------------------------------------------------
    # Action Masking Generators
    # ----------------------------------------------------------------------
    def get_action_masks(self, player_id: int) -> Dict[str, np.ndarray]:
        """Strict Action Masking generator:

        Returns binary masks for each factorized head:
        - core: (6,)
        - auction: (7,)
        - property: (161,)
        - trade: (3 + 41 + 10 + 41 + 10 + 2 = 107,)
        Invalid actions have 0, valid actions have 1.
        """
        # Fast path for inactive players or game over (zero array allocations)
        if player_id != self.active_decision_player or self.bankrupt[player_id] or self.phase == PHASE_GAME_OVER:
            return {k: v.copy() for k, v in INACTIVE_MASKS_TEMPLATE.items()}

        mask_core = np.zeros(6, dtype=np.int8)
        mask_auction = np.zeros(7, dtype=np.int8)
        mask_property = np.zeros(161, dtype=np.int8)
        
        # Trade masks: target(3), offer_prop(41), offer_cash(10), req_prop(41), req_cash(10), respond(2)
        mask_trade_target = np.zeros(3, dtype=np.int8)
        mask_trade_offer_prop = np.zeros(41, dtype=np.int8)
        mask_trade_offer_cash = np.zeros(10, dtype=np.int8)
        mask_trade_req_prop = np.zeros(41, dtype=np.int8)
        mask_trade_req_cash = np.zeros(10, dtype=np.int8)
        mask_trade_response = np.zeros(2, dtype=np.int8)

        # 1. PHASE_ROLL
        if self.phase == PHASE_ROLL:
            mask_core[ACT_CORE_ROLL_PASS] = 1

        # 2. PHASE_JAIL_DECISION
        elif self.phase == PHASE_JAIL_DECISION:
            # Can always roll for doubles
            mask_core[ACT_CORE_ROLL_PASS] = 1
            # Pay $50 fine if has enough cash
            if self.cash[player_id] >= 50:
                mask_core[ACT_CORE_PAY_JAIL_50] = 1
            # Use GOOJF if owns card
            if self.goojf_cards[player_id] > 0:
                mask_core[ACT_CORE_USE_GOOJF] = 1

        # 3. PHASE_BUY_OR_AUCTION
        elif self.phase == PHASE_BUY_OR_AUCTION:
            tile_id = self.landed_property
            price = self.get_property_price(tile_id, 999999) if tile_id is not None else 999999
            cash = self.cash[player_id]

            # Official Monopoly Rule: Direct purchase is legal whenever player has enough cash
            if cash >= price:
                mask_core[ACT_CORE_BUY_DIRECT] = 1

            # Official Monopoly Rule: Player can always decline purchase (sending property to public auction)
            mask_core[ACT_CORE_DECLINE_BUY] = 1

        # 4. PHASE_AUCTION
        elif self.phase == PHASE_AUCTION:
            # Official Monopoly Rule: Fold (0) is always valid
            mask_auction[ACT_AUC_FOLD] = 1
            if self.auction_property is not None:
                current_bid = self.auction_highest_bid
                cash = self.cash[player_id]

                # Official Monopoly Rule: Any active player can bid any increment they have cash for
                increments = [1, 10, 50, 100]
                for act_idx, inc in enumerate(increments, start=1):
                    if cash >= current_bid + inc:
                        mask_auction[act_idx] = 1

                # Jump Bids (allowed if bidder has cash)
                tile = TILES_DATA[self.auction_property]
                price = self.get_property_price(self.auction_property, tile.get("price", 100))
                if price > current_bid and cash >= price:
                    mask_auction[ACT_AUC_JUMP_LIST] = 1

                other_bidders = [b for b in self.auction_active_bidders if b != player_id and not self.bankrupt[b]]
                if other_bidders:
                    max_opp_c = max(self.cash[b] for b in other_bidders)
                    ceiling = self.compute_bidding_ceiling(player_id, self.auction_property)
                    squeeze_target = min(max_opp_c + 1, ceiling)
                    if squeeze_target > current_bid and cash >= squeeze_target:
                        mask_auction[ACT_AUC_JUMP_SQUEEZE] = 1

        # 5. PHASE_PROPERTY_MANAGE
        elif self.phase == PHASE_PROPERTY_MANAGE:
            # Core head is fixed to single dummy option in PHASE_PROPERTY_MANAGE;
            # Property head (0..160) exclusively controls physical actions & pass/trade (160).
            mask_core[ACT_CORE_ROLL_PASS] = 1
            mask_property[160] = 1  # 160: Explicit PASS / END MANAGEMENT / TRADE CHECK action

            same_turn_guard = getattr(self, "manage_actions_this_turn", 0) > 0
            mortgaged_turn = getattr(self, "mortgaged_this_turn", set()) if same_turn_guard else set()
            unmortgaged_turn = getattr(self, "unmortgaged_this_turn", set()) if same_turn_guard else set()
            built_turn = getattr(self, "built_this_turn", set()) if same_turn_guard else set()
            sold_turn = getattr(self, "sold_this_turn", set()) if same_turn_guard else set()

            # Property Management Sub-actions:
            # [0..39]: Build House
            # [40..79]: Sell House
            # [80..119]: Mortgage
            # [120..159]: Unmortgage
            for p in PURCHASABLE_TILES:
                tile = TILES_DATA[p]
                if self.property_owner[p] == player_id:
                    grp = tile.get("color")
                    h = self.property_houses[p]
                    h_cost = self.get_property_house_cost(p, 0)

                    # BUILD HOUSE / HOTEL
                    if tile["type"] == TILE_STREET and grp and p not in sold_turn:
                        group_tiles = COLOR_GROUPS[grp]
                        all_owned = all(self.property_owner[t] == player_id for t in group_tiles)
                        none_mortgaged = all(not self.property_mortgaged[t] for t in group_tiles)
                        if all_owned and none_mortgaged:
                            min_h = min(self.property_houses[t] for t in group_tiles)
                            # Even building: current tile house count must equal min_h and be < 5
                            if h == min_h and h < 5 and self.cash[player_id] >= h_cost:
                                # Physical bank scarcity check
                                if h < 4 and self.bank_houses > 0:
                                    mask_property[p] = 1
                                elif h == 4 and self.bank_hotels > 0:
                                    # Hotel needs bank hotel
                                    mask_property[p] = 1

                    # SELL HOUSE / HOTEL
                    if tile["type"] == TILE_STREET and grp and h > 0 and p not in built_turn:
                        group_tiles = COLOR_GROUPS[grp]
                        max_h = max(self.property_houses[t] for t in group_tiles)
                        if h == max_h:
                            if h <= 4:
                                mask_property[40 + p] = 1
                            elif h == 5:
                                # Selling hotel: bank must have 4 houses or downgrade cleanly
                                if self.bank_houses >= 4:
                                    mask_property[40 + p] = 1

                    # MORTGAGE
                    if not self.property_mortgaged[p] and p not in unmortgaged_turn:
                        # Cannot mortgage if any tile in color group has houses
                        grp = tile.get("color")
                        has_houses = False
                        if grp and tile["type"] == TILE_STREET:
                            has_houses = any(self.property_houses[t] > 0 for t in COLOR_GROUPS[grp])
                        if not has_houses:
                            # Liquidity-aware & strategic leverage mortgage condition:
                            # Only permit voluntary mortgaging during management if:
                            # 1. Player has tight cash liquidity (< $250), OR
                            # 2. Player holds a complete street monopoly and can develop houses on it
                            has_monopoly = any(
                                all(self.property_owner[t] == player_id for t in tiles)
                                for g_name, tiles in COLOR_GROUPS.items()
                                if g_name not in ("RAILROAD", "UTILITY")
                            )
                            can_develop_monopoly = any(
                                TILES_DATA[t]["type"] == TILE_STREET
                                and self.property_owner[t] == player_id
                                and self.property_houses[t] < 5
                                for t in PURCHASABLE_TILES
                            ) if has_monopoly else False

                            if self.cash[player_id] < 250 or (has_monopoly and can_develop_monopoly):
                                mask_property[80 + p] = 1

                    # UNMORTGAGE
                    if self.property_mortgaged[p] and p not in mortgaged_turn:
                        unmtg_cost = int(self.get_property_mortgage(p, tile.get("mortgage", 0)) * (1.0 + UNMORTGAGE_INTEREST_RATE))
                        if self.cash[player_id] >= unmtg_cost:
                            mask_property[120 + p] = 1

            # Bilateral Trade Proposals (max 1 trade proposal per turn)
            if self.allow_trade and getattr(self, "trade_proposals_this_turn", 0) == 0:
                # Target players (3 other players)
                other_players = [pl for pl in range(4) if pl != player_id and not self.bankrupt[pl]]
                for idx, pl in enumerate(other_players):
                    mask_trade_target[idx] = 1

                # Offer properties: owned by player with 0 houses in group
                mask_trade_offer_prop[40] = 1  # 40 = None
                has_tradable_offer_prop = False
                for p in PURCHASABLE_TILES:
                    if self.property_owner[p] == player_id:
                        grp = TILES_DATA[p].get("color")
                        if not (grp and any(self.property_houses[t] > 0 for t in COLOR_GROUPS.get(grp, []))):
                            mask_trade_offer_prop[p] = 1
                            has_tradable_offer_prop = True

                # Offer cash: brackets <= player cash
                for b_idx, b_val in enumerate(TRADE_CASH_BRACKETS):
                    if self.cash[player_id] >= b_val:
                        mask_trade_offer_cash[b_idx] = 1

                # Request properties: owned by someone else with 0 houses in group (and not on rejection cooldown)
                mask_trade_req_prop[40] = 1  # 40 = None
                can_afford_any_offer = has_tradable_offer_prop or (self.cash[player_id] >= TRADE_CASH_BRACKETS[1])
                req_cooldowns = getattr(self, "rejected_req_prop_cooldown", {})
                if can_afford_any_offer:
                    for p in PURCHASABLE_TILES:
                        owner = self.property_owner[p]
                        if owner != -1 and owner != player_id and not self.bankrupt[owner]:
                            if req_cooldowns.get((player_id, owner, p), -1) > self.round_count:
                                continue
                            grp = TILES_DATA[p].get("color")
                            if not (grp and any(self.property_houses[t] > 0 for t in COLOR_GROUPS.get(grp, []))):
                                mask_trade_req_prop[p] = 1

                # Request cash: all brackets (target will decide if they have it)
                mask_trade_req_cash[:] = 1
            else:
                # Curriculum Phase 1: Trades disabled via mask
                mask_trade_target[0] = 1
                mask_trade_offer_prop[40] = 1
                mask_trade_offer_cash[0] = 1
                mask_trade_req_prop[40] = 1
                mask_trade_req_cash[0] = 1

        # 6. PHASE_TRADE_RESPONSE
        elif self.phase == PHASE_TRADE_RESPONSE:
            # 0: Reject (always valid)
            mask_trade_response[0] = 1
            if self.pending_trade:
                req_cash = self.pending_trade["req_cash"]
                init = self.pending_trade["initiator"]
                offer_cash = self.pending_trade["offer_cash"]
                if self.cash[player_id] >= req_cash and self.cash[init] >= offer_cash:
                    mask_trade_response[1] = 1

        # Safety: guarantee at least one valid action for each head
        if not np.any(mask_core):
            mask_core[0] = 1
        if not np.any(mask_auction):
            mask_auction[0] = 1
        if not np.any(mask_property):
            mask_property[160] = 1
        if not np.any(mask_trade_target):
            mask_trade_target[0] = 1
        if not np.any(mask_trade_offer_prop):
            mask_trade_offer_prop[40] = 1
        if not np.any(mask_trade_offer_cash):
            mask_trade_offer_cash[0] = 1
        if not np.any(mask_trade_req_prop):
            mask_trade_req_prop[40] = 1
        if not np.any(mask_trade_req_cash):
            mask_trade_req_cash[0] = 1
        if not np.any(mask_trade_response):
            mask_trade_response[0] = 1

        return {
            "core": mask_core,
            "auction": mask_auction,
            "property": mask_property,
            "trade_target": mask_trade_target,
            "trade_offer_prop": mask_trade_offer_prop,
            "trade_offer_cash": mask_trade_offer_cash,
            "trade_req_prop": mask_trade_req_prop,
            "trade_req_cash": mask_trade_req_cash,
            "trade_response": mask_trade_response,
        }

    # ----------------------------------------------------------------------
    # Observations & Centralized Critic Global State
    # ----------------------------------------------------------------------
    def _build_shared_obs_parts(self) -> Tuple[List[float], List[float], Optional[np.ndarray]]:
        """Computes shared board observation components once per step for all 4 agents."""
        X = [self.position[p] / 40.0 for p in range(4)]
        C = [self.cash[p] / 2000.0 for p in range(4)]
        J_in = [1.0 if self.in_jail[p] else 0.0 for p in range(4)]
        J_turns = [self.jail_turns[p] / 3.0 for p in range(4)]
        J_goojf = [self.goojf_cards[p] / 2.0 for p in range(4)]
        K = [1.0 if self.bankrupt[p] else 0.0 for p in range(4)]

        O = [0.0] * 200
        for p in range(40):
            owner = self.property_owner[p]
            idx = (p * 5) + (0 if owner == -1 else (owner + 1))
            O[idx] = 1.0

        H = [self.property_houses[p] / 5.0 for p in range(40)]
        M = [1.0 if self.property_mortgaged[p] else 0.0 for p in range(40)]
        B_h = [self.bank_houses / 32.0]
        B_ht = [self.bank_hotels / 12.0]
        tau = [0.0 if self.max_rounds is None or self.max_rounds <= 0 else self.round_count / float(self.max_rounds)]

        u = [0.0] * NUM_PHASES
        if 0 <= self.phase < NUM_PHASES:
            u[self.phase] = 1.0

        dice_feat = [self.last_dice[0] / 6.0, self.last_dice[1] / 6.0]
        auc_active = 1.0 if self.phase == PHASE_AUCTION else 0.0
        auc_prop = (self.auction_property or 0) / 40.0
        auc_bid = self.auction_highest_bid / 1000.0
        auc_bidder = [0.0] * 4
        if 0 <= self.auction_highest_bidder < 4:
            auc_bidder[self.auction_highest_bidder] = 1.0

        prefix_314 = X + C + J_in + J_turns + J_goojf + K + O + H + M + B_h + B_ht + tau + u
        suffix_9 = dice_feat + [auc_active, auc_prop, auc_bid] + auc_bidder
        syn_vec = self.get_color_synergy_vector() if self.include_synergy_obs else None
        return prefix_314, suffix_9, syn_vec

    def _assemble_actor_obs(
        self, player_id: int, prefix_314: List[float], suffix_9: List[float], syn_vec: Optional[np.ndarray]
    ) -> np.ndarray:
        """Assembles complete private actor observation vector from shared components."""
        id_onehot = [0.0] * 4
        id_onehot[player_id] = 1.0
        base_vec = np.array(prefix_314 + id_onehot + suffix_9, dtype=np.float32)
        if syn_vec is not None:
            auc_vec = self.get_auction_features(player_id)
            trade_vec = self.get_trade_features(player_id)
            return np.concatenate([base_vec, syn_vec, auc_vec, trade_vec])
        return base_vec

    def get_actor_observation(self, player_id: int) -> np.ndarray:
        """Constructs private actor observation vector s_t^i."""
        prefix_314, suffix_9, syn_vec = self._build_shared_obs_parts()
        return self._assemble_actor_obs(player_id, prefix_314, suffix_9, syn_vec)

    def get_global_state(self) -> np.ndarray:
        """Constructs Centralized Critic Global State S_t^{global}:

        S_t^{global} = [ s_t^1, s_t^2, s_t^3, s_t^4, Deck_chance, Deck_chest ]
        """
        prefix_314, suffix_9, syn_vec = self._build_shared_obs_parts()
        actor_obs = [self._assemble_actor_obs(p, prefix_314, suffix_9, syn_vec) for p in range(4)]
        deck_state = np.array(self.deck_manager.get_state_vector(), dtype=np.float32)
        return np.concatenate(actor_obs + [deck_state])

    def get_observations(self) -> Dict[str, Any]:
        """Returns observation dictionary for all agents and global critic state (zero-redundancy)."""
        prefix_314, suffix_9, syn_vec = self._build_shared_obs_parts()
        obs_dict = {}
        masks_dict = {}
        actor_obs_list = []
        for p in range(4):
            p_name = f"player_{p}"
            p_obs = self._assemble_actor_obs(p, prefix_314, suffix_9, syn_vec)
            obs_dict[p_name] = p_obs
            actor_obs_list.append(p_obs)
            masks_dict[p_name] = self.get_action_masks(p)

        deck_state = np.array(self.deck_manager.get_state_vector(), dtype=np.float32)
        global_state = np.concatenate(actor_obs_list + [deck_state])

        return {
            "obs": obs_dict,
            "action_mask": masks_dict,
            "global_state": global_state,
            "active_player": self.active_decision_player,
            "phase": self.phase,
        }

    # ----------------------------------------------------------------------
    # Game Step Transition & Execution
    # ----------------------------------------------------------------------
    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        """Executes a micro-step given factorized actions from active agent.

        Returns (observations, rewards, dones, infos).
        """
        p = self.active_decision_player
        p_act = actions.get(f"player_{p}", actions)
        
        # Parse action heads
        core_act = int(p_act.get("core", 0))
        auc_act = int(p_act.get("auction", 0))
        prop_act = int(p_act.get("property", 0))
        
        trade_target_idx = int(p_act.get("trade_target", 0))
        trade_offer_prop = int(p_act.get("trade_offer_prop", 40))
        trade_offer_cash_idx = int(p_act.get("trade_offer_cash", 0))
        trade_req_prop = int(p_act.get("trade_req_prop", 40))
        trade_req_cash_idx = int(p_act.get("trade_req_cash", 0))
        trade_response = int(p_act.get("trade_response", 0))

        # Check action mask validity
        masks = self.get_action_masks(p)

        # Environment rewards before potential shaping
        env_rewards = [0.0, 0.0, 0.0, 0.0]

        # ----------------- EXECUTE ACCORDING TO PHASE -----------------
        if self.phase == PHASE_ROLL:
            self._handle_roll_phase(p)

        elif self.phase == PHASE_JAIL_DECISION:
            self._handle_jail_phase(p, core_act, masks)

        elif self.phase == PHASE_BUY_OR_AUCTION:
            self._handle_buy_auction_phase(p, core_act)

        elif self.phase == PHASE_AUCTION:
            self._handle_auction_phase(p, auc_act)

        elif self.phase == PHASE_PROPERTY_MANAGE:
            self._handle_management_phase(
                p, core_act, prop_act, masks,
                trade_target_idx, trade_offer_prop, trade_offer_cash_idx,
                trade_req_prop, trade_req_cash_idx
            )

        elif self.phase == PHASE_TRADE_RESPONSE:
            self._handle_trade_response(p, trade_response, env_rewards)

        # ----------------- CHECK BANKRUPTCY & VICTORY -----------------
        for pl in range(4):
            if not self.bankrupt[pl] and self.cash[pl] < 0:
                self._resolve_bankruptcy(pl)
                if self.bankrupt[pl]:
                    env_rewards[pl] -= 5.0

        # Check terminal conditions (Unbounded horizon survival or max rounds reached)
        active_players = [pl for pl in range(4) if not self.bankrupt[pl]]
        dones = {f"player_{pl}": self.bankrupt[pl] for pl in range(4)}
        
        max_rounds_reached = (self.max_rounds is not None) and (self.round_count >= self.max_rounds)
        if len(active_players) <= 1 or max_rounds_reached:
            self.phase = PHASE_GAME_OVER
            for pl in range(4):
                dones[f"player_{pl}"] = True
            dones["__all__"] = True

            if len(active_players) == 1:
                winner = active_players[0]
                env_rewards[winner] += 5.0
                self.event_log.append(f"Game Over! {self.get_player_name(winner)} is the TOURNAMENT CHAMPION (Last Agent Standing)!")
            elif len(active_players) == 0:
                self.event_log.append("Game Over! All players eliminated simultaneously.")
            else:
                # Max rounds reached (if finite horizon was configured): player with highest net worth wins
                net_worths = [self.get_player_net_worth(pl) if not self.bankrupt[pl] else -99999 for pl in range(4)]
                winner = int(np.argmax(net_worths))
                env_rewards[winner] += 2.0
                self.event_log.append(f"Max rounds ({self.max_rounds}) reached. {self.get_player_name(winner)} wins on Net Worth!")
        else:
            dones["__all__"] = False

        # ----------------- POTENTIAL-BASED REWARD SHAPING (PBRS) -----------------
        # R'^i(s, a, s') = R_{env}^i + gamma * Phi^i(s') - Phi^i(s)
        shaped_rewards = {}
        for pl in range(4):
            new_phi = self._compute_potential(pl)
            pbrs = self.gamma * new_phi - self.prev_potentials[pl]
            shaped_r = env_rewards[pl] + pbrs
            shaped_rewards[f"player_{pl}"] = float(shaped_r)
            self.prev_potentials[pl] = new_phi

        total_monopolies = sum(
            1 for grp_name, tiles in COLOR_GROUPS.items()
            if grp_name not in ("RAILROAD", "UTILITY")
            and self.property_owner[tiles[0]] != -1
            and all(self.property_owner[t] == self.property_owner[tiles[0]] for t in tiles)
        )
        observations = self.get_observations()
        infos = {
            "event_log": list(self.event_log[-5:]),
            "round": self.round_count,
            "turn_step": self.turn_step,
            "bank_houses": self.bank_houses,
            "bank_hotels": self.bank_hotels,
            "last_dice": self.last_dice,
            "trades_proposed": self.trades_proposed_count,
            "trades_accepted": self.trades_accepted_count,
            "houses_built": self.houses_built_count,
            "monopolies_count": total_monopolies,
        }

        return observations, shaped_rewards, dones, infos

    # ----------------------------------------------------------------------
    # Core Phase Handlers
    # ----------------------------------------------------------------------
    def _handle_roll_phase(self, p: int):
        """Rolls dice, handles Speeding (3x doubles), and moves player."""
        d1 = self.rng.randint(1, 6)
        d2 = self.rng.randint(1, 6)
        self.last_dice = (d1, d2)
        is_double = (d1 == d2)

        if is_double:
            self.consecutive_doubles += 1
        else:
            self.consecutive_doubles = 0

        # SPEEDING RULE: 3 consecutive doubles -> immediately to Jail!
        if self.consecutive_doubles >= 3:
            self.event_log.append(f"{self.get_player_name(p)} rolled doubles 3 times in a row! SPEEDING -> Sent directly to JAIL!")
            self._send_to_jail(p)
            self._end_turn()
            return

        dice_sum = d1 + d2
        old_pos = self.position[p]
        new_pos = (old_pos + dice_sum) % 40
        self.position[p] = new_pos

        self.event_log.append(f"{self.get_player_name(p)} rolled ({d1}, {d2}) = {dice_sum}. Moved from {old_pos} to {new_pos} ({TILES_DATA[new_pos]['name']}).")

        # Check Passing GO
        if new_pos < old_pos:
            salary = self.house_rules.apply_pass_go_salary(p, 200)
            self.cash[p] += salary
            self.event_log.append(f"{self.get_player_name(p)} passed GO and collected ${salary}.")

        self._handle_landing(p, new_pos, dice_sum, is_double)

    def _handle_jail_phase(self, p: int, core_act: int, masks: Dict[str, np.ndarray]):
        """Handles player in jail: roll doubles, pay $50, or use GOOJF card."""
        if core_act == ACT_CORE_PAY_JAIL_50 and masks["core"][ACT_CORE_PAY_JAIL_50]:
            self.cash[p] -= 50
            self.last_creditor = None
            self.in_jail[p] = False
            self.jail_turns[p] = 0
            self.event_log.append(f"{self.get_player_name(p)} paid $50 fine to get out of Jail.")
            self.phase = PHASE_ROLL
            self.active_decision_player = p
            return

        elif core_act == ACT_CORE_USE_GOOJF and masks["core"][ACT_CORE_USE_GOOJF]:
            self.goojf_cards[p] -= 1
            source_deck = self.goojf_sources[p].pop()
            self.deck_manager.return_goojf(source_deck)
            self.in_jail[p] = False
            self.jail_turns[p] = 0
            self.event_log.append(f"{self.get_player_name(p)} used a Get Out of Jail Free card.")
            self.phase = PHASE_ROLL
            self.active_decision_player = p
            return

        else:
            # Action 0: Roll for doubles
            d1 = self.rng.randint(1, 6)
            d2 = self.rng.randint(1, 6)
            self.last_dice = (d1, d2)
            is_double = (d1 == d2)

            if is_double:
                self.in_jail[p] = False
                self.jail_turns[p] = 0
                self.consecutive_doubles = 0
                self.event_log.append(f"{self.get_player_name(p)} rolled doubles ({d1}, {d2}) and got out of Jail!")
                # Move by dice sum, but rolling out of jail does not grant another turn
                old_pos = self.position[p]
                new_pos = (old_pos + d1 + d2) % 40
                self.position[p] = new_pos
                self._handle_landing(p, new_pos, d1 + d2, is_double=False)
            else:
                self.jail_turns[p] += 1
                self.event_log.append(f"{self.get_player_name(p)} rolled ({d1}, {d2}) (no doubles). Remains in Jail (turn {self.jail_turns[p]}/3).")
                if self.jail_turns[p] >= 3:
                    # Turn 3 failed double: must pay $50 fine and move!
                    self.cash[p] -= 50
                    self.last_creditor = None
                    self.in_jail[p] = False
                    self.jail_turns[p] = 0
                    self.event_log.append(f"{self.get_player_name(p)} served 3 turns in Jail. Paid $50 mandatory fine and moves {d1 + d2} spaces.")
                    old_pos = self.position[p]
                    new_pos = (old_pos + d1 + d2) % 40
                    self.position[p] = new_pos
                    self._handle_landing(p, new_pos, d1 + d2, is_double=False)
                else:
                    self._end_turn()

    def _handle_landing(self, p: int, pos: int, dice_sum: int, is_double: bool):
        """Resolves landing on tile pos."""
        tile = TILES_DATA[pos]
        t_type = tile["type"]

        if pos == 30:  # Go To Jail
            self.event_log.append(f"{self.get_player_name(p)} landed on Go To Jail!")
            self._send_to_jail(p)
            self._end_turn()
            return

        if t_type == TILE_TAX:
            tax = tile["tax_amount"]
            self.cash[p] -= tax
            self.last_creditor = None
            self.event_log.append(f"{self.get_player_name(p)} paid tax of ${tax} on {tile['name']}.")
            self._proceed_to_manage_or_next(p, is_double)
            return

        if t_type == TILE_CARD:
            deck_name = tile["deck"]
            card = self.deck_manager.draw_chance() if deck_name == "chance" else self.deck_manager.draw_chest()
            self.last_card_drawn = card
            self.event_log.append(f"{self.get_player_name(p)} drew {deck_name.upper()} card: '{card['name']}'.")
            self._execute_card_action(p, card, is_double)
            return

        if t_type in (TILE_STREET, TILE_RAILROAD, TILE_UTILITY):
            owner = self.property_owner[pos]
            if owner == -1:
                # Unowned property: Player can buy directly or decline (triggering auction)
                self.landed_property = pos
                self.phase = PHASE_BUY_OR_AUCTION
                self.active_decision_player = p
                self.pending_is_double = is_double
                return
            elif owner != p and not self.property_mortgaged[pos]:
                # Pay rent
                rent = self._calculate_rent(pos, dice_sum)
                self.cash[p] -= rent
                self.cash[owner] += rent
                self.last_creditor = owner
                self.event_log.append(f"{self.get_player_name(p)} paid ${rent} rent to {self.get_player_name(owner)} on {tile['name']}.")
                self._proceed_to_manage_or_next(p, is_double)
                return
            else:
                # Owned by self or mortgaged
                self._proceed_to_manage_or_next(p, is_double)
                return

        # Special / Free Parking / GO / Jail visiting
        self._proceed_to_manage_or_next(p, is_double)

    def _handle_buy_auction_phase(self, p: int, core_act: int):
        """Active player chooses to buy directly or decline (triggers auction)."""
        tile_id = self.landed_property
        if tile_id is None:
            self._proceed_to_manage_or_next(p, getattr(self, "pending_is_double", False))
            return

        tile = TILES_DATA[tile_id]
        price = self.get_property_price(tile_id, tile["price"])

        if core_act == ACT_CORE_BUY_DIRECT and self.cash[p] >= price:
            self.cash[p] -= price
            self.property_owner[tile_id] = p
            self.event_log.append(f"{self.get_player_name(p)} purchased {tile['name']} directly for ${price}.")
            self.landed_property = None
            self._proceed_to_manage_or_next(p, getattr(self, "pending_is_double", False))
        else:
            # DECLINE BUY -> MANDATORY OPEN AUCTION TRIGGERED!
            self.event_log.append(f"{self.get_player_name(p)} declined buying {tile['name']}. Mandatory AUCTION begins!")
            self._start_auction(tile_id, getattr(self, "pending_is_double", False))

    # ----------------------------------------------------------------------
    # Auction Engine
    # ----------------------------------------------------------------------
    def _start_auction(self, property_id: int, is_double_pending: bool):
        """Initializes open ascending auction among all non-bankrupt players."""
        self.auction_property = property_id
        self.auction_highest_bid = 0
        self.auction_highest_bidder = -1
        start_p = self.current_player
        self.auction_active_bidders = [
            (start_p + off) % 4
            for off in range(4)
            if not self.bankrupt[(start_p + off) % 4]
        ]
        self.auction_current_bidder_idx = 0
        self.pending_auction_double = is_double_pending
        self.phase = PHASE_AUCTION
        self.active_decision_player = self.auction_active_bidders[0]

    def _handle_auction_phase(self, p: int, auc_act: int):
        """Processes auction bid, strategic jump bid, or fold for active bidder."""
        current_bid = self.auction_highest_bid
        cash = self.cash[p]
        tile = TILES_DATA[self.auction_property]
        price = self.get_property_price(self.auction_property, tile.get("price", 100))

        bid_inc = 0
        if auc_act == ACT_AUC_BID_1:
            bid_inc = 1
        elif auc_act == ACT_AUC_BID_10:
            bid_inc = 10
        elif auc_act == ACT_AUC_BID_50:
            bid_inc = 50
        elif auc_act == ACT_AUC_BID_100:
            bid_inc = 100
        elif auc_act == ACT_AUC_JUMP_LIST:
            if price > current_bid and cash >= price:
                bid_inc = price - current_bid
        elif auc_act == ACT_AUC_JUMP_SQUEEZE:
            other_bidders = [b for b in self.auction_active_bidders if b != p and not self.bankrupt[b]]
            if other_bidders:
                max_opp_c = max(self.cash[b] for b in other_bidders)
                ceiling = self.compute_bidding_ceiling(p, self.auction_property)
                squeeze_target = min(max_opp_c + 1, ceiling)
                if squeeze_target > current_bid and cash >= squeeze_target:
                    bid_inc = squeeze_target - current_bid

        if bid_inc > 0 and cash >= current_bid + bid_inc:
            self.auction_highest_bid = current_bid + bid_inc
            self.auction_highest_bidder = p
            self.event_log.append(f"{self.get_player_name(p)} bids ${self.auction_highest_bid} (+${bid_inc}) on {TILES_DATA[self.auction_property]['name']}.")
            # Advance to next bidder
            self.auction_current_bidder_idx = (self.auction_current_bidder_idx + 1) % len(self.auction_active_bidders)
            self.active_decision_player = self.auction_active_bidders[self.auction_current_bidder_idx]
        else:
            # Fold
            self.event_log.append(f"{self.get_player_name(p)} folded in auction.")
            self.auction_active_bidders.remove(p)
            if len(self.auction_active_bidders) == 0 or (len(self.auction_active_bidders) == 1 and self.auction_highest_bidder != -1):
                self._finish_auction()
                return
            else:
                self.auction_current_bidder_idx %= len(self.auction_active_bidders)
                self.active_decision_player = self.auction_active_bidders[self.auction_current_bidder_idx]

    def _finish_auction(self):
        """Concludes auction and awards property to highest bidder."""
        tile_id = self.auction_property
        tile = TILES_DATA[tile_id]
        if self.auction_highest_bidder != -1 and self.auction_highest_bid > 0:
            winner = self.auction_highest_bidder
            price = self.auction_highest_bid
            self.cash[winner] -= price
            self.property_owner[tile_id] = winner
            self.event_log.append(f"AUCTION WON! {self.get_player_name(winner)} won {tile['name']} for ${price}!")
        else:
            self.event_log.append(f"Auction ended with no bids. {tile['name']} remains with Bank.")

        self.auction_property = None
        self.auction_highest_bid = 0
        self.auction_highest_bidder = -1
        self.auction_active_bidders = []
        is_double = getattr(self, "pending_auction_double", False)
        self._proceed_to_manage_or_next(self.current_player, is_double)

    # ----------------------------------------------------------------------
    # Property Management & Trading
    # ----------------------------------------------------------------------
    def _clear_turn_manage_guards(self):
        """Resets same-turn physical management thrash guards."""
        self.mortgaged_this_turn.clear()
        self.unmortgaged_this_turn.clear()
        self.built_this_turn.clear()
        self.sold_this_turn.clear()

    def _finish_manage_or_next(self, p: int, is_double: bool):
        """Finishes management phase and either executes double roll or rotates turn."""
        self.manage_actions_this_turn = 0
        self.trade_proposals_this_turn = 0
        self._clear_turn_manage_guards()
        if is_double and not self.in_jail[p] and not self.bankrupt[p]:
            self.event_log.append(f"{self.get_player_name(p)} rolled doubles! Gets another roll.")
            self.phase = PHASE_ROLL
            self.active_decision_player = p
        else:
            self._end_turn()

    def _proceed_to_manage_or_next(self, p: int, is_double: bool):
        """Enters property management phase before ending turn or re-rolling."""
        self.phase = PHASE_PROPERTY_MANAGE
        self.active_decision_player = p
        self.pending_manage_double = is_double
        self.manage_actions_this_turn = 0
        self._clear_turn_manage_guards()

    def _handle_management_phase(
        self, p: int, core_act: int, prop_act: int, masks: Dict[str, np.ndarray],
        trade_target_idx: int, trade_offer_prop: int, trade_offer_cash_idx: int,
        trade_req_prop: int, trade_req_cash_idx: int
    ):
        """Handles house/hotel building/selling, mortgaging, and trade proposals."""
        is_double = getattr(self, "pending_manage_double", False)
        self.manage_actions_this_turn = getattr(self, "manage_actions_this_turn", 0) + 1

        # 1. Execute physical property management actions FIRST (0..159) directly without core_act interference
        if 0 <= prop_act < 160 and bool(masks["property"][prop_act]):
            # [0..39]: Build House / Hotel
            if 0 <= prop_act < 40:
                tile = TILES_DATA[prop_act]
                if tile["type"] == TILE_STREET and "house_cost" in tile:
                    h_cost = self.get_property_house_cost(prop_act, tile["house_cost"])
                    h = self.property_houses[prop_act]
                    if h < 4:
                        self.property_houses[prop_act] += 1
                        self.bank_houses -= 1
                        self.cash[p] -= h_cost
                        self.houses_built_count += 1
                        self.built_this_turn.add(prop_act)
                        self.event_log.append(f"{self.get_player_name(p)} built a house on {tile['name']} for ${h_cost}. (Bank houses left: {self.bank_houses})")
                    elif h == 4:
                        self.property_houses[prop_act] = 5
                        self.bank_hotels -= 1
                        self.bank_houses += 4
                        self.cash[p] -= h_cost
                        self.houses_built_count += 1
                        self.built_this_turn.add(prop_act)
                        self.event_log.append(f"{self.get_player_name(p)} upgraded to a HOTEL on {tile['name']} for ${h_cost}! (4 houses returned to bank).")

            # [40..79]: Sell House / Hotel
            elif 40 <= prop_act < 80:
                p_idx = prop_act - 40
                tile = TILES_DATA[p_idx]
                if tile["type"] == TILE_STREET and "house_cost" in tile:
                    h_cost = self.get_property_house_cost(p_idx, tile["house_cost"])
                    h = self.property_houses[p_idx]
                    refund = int(h_cost * HOUSE_REFUND_RATE)
                    if h <= 4:
                        self.property_houses[p_idx] -= 1
                        self.bank_houses += 1
                        self.cash[p] += refund
                        self.sold_this_turn.add(p_idx)
                        self.event_log.append(f"{self.get_player_name(p)} sold a house on {tile['name']} for ${refund}.")
                    elif h == 5:
                        self.property_houses[p_idx] = 4
                        self.bank_hotels += 1
                        self.bank_houses -= 4
                        self.cash[p] += refund
                        self.sold_this_turn.add(p_idx)
                        self.event_log.append(f"{self.get_player_name(p)} sold hotel on {tile['name']} for ${refund} (converted to 4 houses).")

            # [80..119]: Mortgage
            elif 80 <= prop_act < 120:
                p_idx = prop_act - 80
                tile = TILES_DATA[p_idx]
                if "mortgage" in tile and p_idx in PURCHASABLE_TILES:
                    mtg_val = self.get_property_mortgage(p_idx, tile["mortgage"])
                    self.property_mortgaged[p_idx] = True
                    self.cash[p] += mtg_val
                    self.mortgaged_this_turn.add(p_idx)
                    self.event_log.append(f"{self.get_player_name(p)} mortgaged {tile['name']} for ${mtg_val}.")

            # [120..159]: Unmortgage
            elif 120 <= prop_act < 160:
                p_idx = prop_act - 120
                tile = TILES_DATA[p_idx]
                if "mortgage" in tile and p_idx in PURCHASABLE_TILES:
                    cost = int(self.get_property_mortgage(p_idx, tile["mortgage"]) * (1.0 + UNMORTGAGE_INTEREST_RATE))
                    self.property_mortgaged[p_idx] = False
                    self.cash[p] -= cost
                    self.unmortgaged_this_turn.add(p_idx)
                    self.event_log.append(f"{self.get_player_name(p)} unmortgaged {tile['name']} for ${cost} (incl. 10% interest).")

            if self.manage_actions_this_turn >= 5:
                self._finish_manage_or_next(p, is_double)
            return

        # 2. Check bilateral trade proposal (when not executing physical property action; max 1 trade per turn)
        has_trade_intent = (
            self.allow_trade and
            getattr(self, "trade_proposals_this_turn", 0) == 0 and
            (trade_offer_prop < 40 or trade_req_prop < 40)
        )

        if has_trade_intent:
            # Synchronize uncoordinated categorical action heads:
            # If agent intends a pure-cash purchase of opponent deed (trade_offer_prop == 40, trade_req_prop < 40)
            # but the cash head chose a bracket below the deed's statutory mortgage floor,
            # snap offer_cash_idx to a competitive legal bracket the agent can afford.
            if trade_offer_prop >= 40 and 0 <= trade_req_prop < 40:
                min_cash = self.get_property_mortgage(trade_req_prop, TILES_DATA[trade_req_prop]["mortgage"])
                cur_cash = TRADE_CASH_BRACKETS[min(trade_offer_cash_idx, len(TRADE_CASH_BRACKETS) - 1)]
                if cur_cash < min_cash:
                    target_cash = 500 if self.cash[p] >= 500 else (250 if self.cash[p] >= 250 else min_cash)
                    legal_brackets = [idx for idx, b in enumerate(TRADE_CASH_BRACKETS) if b >= target_cash and self.cash[p] >= b]
                    if not legal_brackets:
                        legal_brackets = [idx for idx, b in enumerate(TRADE_CASH_BRACKETS) if b >= min_cash and self.cash[p] >= b]
                    if legal_brackets:
                        trade_offer_cash_idx = legal_brackets[0]

            trade_started = self._propose_trade(
                p, trade_target_idx, trade_offer_prop, trade_offer_cash_idx, trade_req_prop, trade_req_cash_idx
            )
            if trade_started:
                self.trade_proposals_this_turn = 1
                self.trades_proposed_count += 1
                return

        # 3. End Management -> Proceed to Next Turn or Double Roll
        self._finish_manage_or_next(p, is_double)

    def _propose_trade(
        self, initiator: int, target_idx: int, offer_prop: int,
        offer_cash_idx: int, req_prop: int, req_cash_idx: int
    ) -> bool:
        """Initiates a bilateral trade negotiation. Returns True if trade proposal started."""
        other_players = [pl for pl in range(4) if pl != initiator and not self.bankrupt[pl]]
        if not other_players:
            return False

        # Auto-route target deterministically to the rightful owner of req_prop when requested
        if 0 <= req_prop < 40:
            prop_owner = self.property_owner[req_prop]
            if prop_owner == -1 or prop_owner == initiator or self.bankrupt[prop_owner]:
                return False
            if self.rejected_req_prop_cooldown.get((initiator, prop_owner, req_prop), -1) > self.round_count:
                return False
            grp_req = TILES_DATA[req_prop].get("color")
            if grp_req and any(self.property_houses[t] > 0 for t in COLOR_GROUPS.get(grp_req, [])):
                return False
            target = prop_owner
        else:
            target = other_players[target_idx % len(other_players)]
            if 0 <= offer_prop < 40 and self.rejected_offer_prop_cooldown.get((initiator, target, offer_prop), -1) > self.round_count:
                return False

        offer_cash = TRADE_CASH_BRACKETS[min(offer_cash_idx, len(TRADE_CASH_BRACKETS) - 1)]
        req_cash = TRADE_CASH_BRACKETS[min(req_cash_idx, len(TRADE_CASH_BRACKETS) - 1)]

        # Canonicalize dual-cash paradox into strict one-way net cash
        if offer_cash > 0 and req_cash > 0:
            if offer_cash >= req_cash:
                offer_cash -= req_cash
                req_cash = 0
            else:
                req_cash -= offer_cash
                offer_cash = 0

        # Validate offer & eliminate degenerate one-sided giveaways / empty-handed begs
        if 0 <= offer_prop < 40:
            if offer_prop == req_prop:
                return False
            if self.property_owner[offer_prop] != initiator:
                return False
            grp_off = TILES_DATA[offer_prop].get("color")
            if grp_off and any(self.property_houses[t] > 0 for t in COLOR_GROUPS.get(grp_off, [])):
                return False
        elif offer_cash == 0:
            # Pure beg: offering neither property nor cash
            return False
        else:
            # Pure cash purchase: If requesting a property without offering a property in return,
            # cash offered must be at least the property's statutory mortgage value
            if 0 <= req_prop < 40:
                min_cash_required = self.get_property_mortgage(req_prop, TILES_DATA[req_prop]["mortgage"])
                if offer_cash < min_cash_required:
                    return False

        # Pure cash property sale: ensure requested cash meets at least statutory mortgage value
        if 0 <= offer_prop < 40 and req_prop >= 40:
            min_cash_required = self.get_property_mortgage(offer_prop, TILES_DATA[offer_prop]["mortgage"])
            if req_cash < min_cash_required:
                valid_brackets = [b for b in TRADE_CASH_BRACKETS if b >= min_cash_required]
                if valid_brackets:
                    req_cash = valid_brackets[0]

        if req_prop >= 40 and req_cash == 0:
            # Pure gift: giving away property (and/or cash) for nothing in return
            return False

        if self.cash[initiator] < offer_cash:
            return False

        self.pending_trade = {
            "initiator": initiator,
            "target": target,
            "offer_prop": offer_prop,
            "offer_cash": offer_cash,
            "req_prop": req_prop,
            "req_cash": req_cash,
        }
        self.phase = PHASE_TRADE_RESPONSE
        self.active_decision_player = target
        self.event_log.append(f"TRADE PROPOSAL: {self.get_player_name(initiator)} -> {self.get_player_name(target)}: Offer(Prop={offer_prop if offer_prop<40 else 'None'}, Cash=${offer_cash}) for Request(Prop={req_prop if req_prop<40 else 'None'}, Cash=${req_cash}).")
        return True

    def _handle_trade_response(self, target: int, trade_response: int, env_rewards: Optional[List[float]] = None):
        """Processes acceptance or rejection of bilateral trade."""
        init = self.current_player
        if self.pending_trade:
            init = self.pending_trade["initiator"]
            if trade_response == 1:
                trade = self.pending_trade
                req_cash = trade["req_cash"]
                offer_cash = trade["offer_cash"]
                offer_p = trade["offer_prop"]
                req_p = trade["req_prop"]

                if self.cash[target] >= req_cash and self.cash[init] >= offer_cash:
                    # Helper to track street monopolies
                    def _get_street_monopolies(pl: int) -> Set[str]:
                        return {
                            grp_name for grp_name, tiles in COLOR_GROUPS.items()
                            if grp_name not in ("RAILROAD", "UTILITY")
                            and all(self.property_owner[t] == pl for t in tiles)
                        }

                    init_mono_before = _get_street_monopolies(init)
                    target_mono_before = _get_street_monopolies(target)

                    # Atomic transfer
                    self.cash[init] -= offer_cash
                    self.cash[target] += offer_cash
                    self.cash[target] -= req_cash
                    self.cash[init] += req_cash

                    if offer_p < 40 and self.property_owner[offer_p] == init:
                        self.property_owner[offer_p] = target
                    if req_p < 40 and self.property_owner[req_p] == target:
                        self.property_owner[req_p] = init

                    self.trades_accepted_count += 1
                    self.event_log.append(f"TRADE ACCEPTED between {self.get_player_name(init)} and {self.get_player_name(target)}!")

                    # Anti-churning cooldown: Prevent immediately ping-ponging the exact same property back
                    if 0 <= req_p < 40:
                        self.rejected_req_prop_cooldown[(target, init, req_p)] = self.round_count + 8
                        self.rejected_req_prop_cooldown[(init, target, req_p)] = self.round_count + 8
                    if 0 <= offer_p < 40:
                        self.rejected_offer_prop_cooldown[(target, init, offer_p)] = self.round_count + 8
                        self.rejected_offer_prop_cooldown[(init, target, offer_p)] = self.round_count + 8

                    # Reward shaping for successful monopoly-advancing trades
                    init_mono_after = _get_street_monopolies(init)
                    target_mono_after = _get_street_monopolies(target)

                    init_completed_mono = len(init_mono_after) > len(init_mono_before)
                    target_completed_mono = len(target_mono_after) > len(target_mono_before)

                    if env_rewards is not None:
                        if init_completed_mono:
                            env_rewards[init] += 1.5
                        if target_completed_mono:
                            env_rewards[target] += 1.5
                        if init_completed_mono and target_completed_mono:
                            # Bilateral Win-Win Synergy bonus!
                            env_rewards[init] += 1.0
                            env_rewards[target] += 1.0
                else:
                    self.event_log.append("Trade failed due to insufficient funds.")
            else:
                trade = self.pending_trade
                req_p = trade["req_prop"]
                offer_p = trade["offer_prop"]
                if 0 <= req_p < 40:
                    self.rejected_req_prop_cooldown[(init, target, req_p)] = self.round_count + 5
                elif 0 <= offer_p < 40:
                    self.rejected_offer_prop_cooldown[(init, target, offer_p)] = self.round_count + 5
                self.event_log.append(f"Trade REJECTED by {self.get_player_name(target)}.")

                # Friction penalty on initiator for proposing a rejected trade
                if env_rewards is not None:
                    env_rewards[init] -= 0.15

        self.pending_trade = None
        is_double = getattr(self, "pending_manage_double", False)
        # Non-preemptive trading: Return to PHASE_PROPERTY_MANAGE for initiator
        # If the initiator just acquired a monopoly via trade, they can immediately build houses!
        if not self.bankrupt[init] and self.manage_actions_this_turn < 5:
            self.phase = PHASE_PROPERTY_MANAGE
            self.active_decision_player = init
        else:
            self._finish_manage_or_next(init, is_double)

    # ----------------------------------------------------------------------
    # Cards & Actions Handling
    # ----------------------------------------------------------------------
    def _execute_card_action(self, p: int, card: Dict[str, Any], is_double: bool):
        """Executes official Chance or Community Chest card effects."""
        action = card["action"]

        if action == "advance_to":
            target = card["target"]
            old_pos = self.position[p]
            self.position[p] = target
            if card.get("collect_pass_go", False) and target < old_pos:
                salary = self.house_rules.apply_pass_go_salary(p, 200)
                self.cash[p] += salary
                self.event_log.append(f"{self.get_player_name(p)} passed GO and collected ${salary}.")
            self._handle_landing(p, target, dice_sum=self.last_dice[0] + self.last_dice[1], is_double=is_double)

        elif action == "advance_nearest_utility":
            # Utilities are 12 and 28
            curr = self.position[p]
            target = 12 if curr < 12 or curr >= 28 else 28
            if target < curr:
                salary = self.house_rules.apply_pass_go_salary(p, 200)
                self.cash[p] += salary
                self.event_log.append(f"{self.get_player_name(p)} passed GO and collected ${salary}.")
            self.position[p] = target
            owner = self.property_owner[target]
            if owner == -1:
                self.landed_property = target
                self.phase = PHASE_BUY_OR_AUCTION
                self.active_decision_player = p
                self.pending_is_double = is_double
            elif owner != p and not self.property_mortgaged[target]:
                # Official Card Rule: throw fresh dice and pay 10x
                d_roll = self.rng.randint(1, 6) + self.rng.randint(1, 6)
                rent = 10 * d_roll
                self.cash[p] -= rent
                self.cash[owner] += rent
                self.last_creditor = owner
                self.event_log.append(f"{self.get_player_name(p)} threw {d_roll} and paid ${rent} (10x fresh roll) to {self.get_player_name(owner)} on {TILES_DATA[target]['name']}.")
                self._proceed_to_manage_or_next(p, is_double)
            else:
                self._proceed_to_manage_or_next(p, is_double)

        elif action == "advance_nearest_railroad":
            # Railroads are 5, 15, 25, 35
            curr = self.position[p]
            rr_tiles = [5, 15, 25, 35]
            target = min([r for r in rr_tiles if r > curr] or [5])
            if target < curr:
                self.cash[p] += 200
            self.position[p] = target
            owner = self.property_owner[target]
            if owner == -1:
                self.landed_property = target
                self.phase = PHASE_BUY_OR_AUCTION
                self.active_decision_player = p
                self.pending_is_double = is_double
            elif owner != p and not self.property_mortgaged[target]:
                # 2x normal rent
                rent = self._calculate_rent(target, dice_sum=self.last_dice[0] + self.last_dice[1], multiplier_override=2)
                self.cash[p] -= rent
                self.cash[owner] += rent
                self.last_creditor = owner
                self.event_log.append(f"{self.get_player_name(p)} paid ${rent} (2x railroad rent) to {self.get_player_name(owner)} on {TILES_DATA[target]['name']}.")
                self._proceed_to_manage_or_next(p, is_double)
            else:
                self._proceed_to_manage_or_next(p, is_double)

        elif action == "cash_from_bank":
            amount = card["amount"]
            self.cash[p] += amount
            self._proceed_to_manage_or_next(p, is_double)

        elif action == "pay_bank":
            amount = card["amount"]
            self.cash[p] -= amount
            self.last_creditor = None
            self._proceed_to_manage_or_next(p, is_double)

        elif action == "goojf":
            deck = card["deck"]
            self.goojf_cards[p] += 1
            self.goojf_sources[p].append(deck)
            self._proceed_to_manage_or_next(p, is_double)

        elif action == "go_to_jail":
            self._send_to_jail(p)
            self._end_turn()

        elif action == "move_relative":
            steps = card["steps"]
            new_pos = (self.position[p] + steps) % 40
            self.position[p] = new_pos
            self._handle_landing(p, new_pos, dice_sum=self.last_dice[0] + self.last_dice[1], is_double=is_double)

        elif action == "property_repairs":
            h_cost = card["house_cost"]
            hotel_cost = card["hotel_cost"]
            owned = [t for t in PURCHASABLE_TILES if self.property_owner[t] == p]
            total_houses = sum(self.property_houses[t] for t in owned if 1 <= self.property_houses[t] <= 4)
            total_hotels = sum(1 for t in owned if self.property_houses[t] == 5)
            repairs = total_houses * h_cost + total_hotels * hotel_cost
            self.cash[p] -= repairs
            self.last_creditor = None
            self.event_log.append(f"{self.get_player_name(p)} paid ${repairs} in property repairs.")
            self._proceed_to_manage_or_next(p, is_double)

        elif action == "pay_each_player":
            amt = card["amount"]
            for pl in range(4):
                if pl != p and not self.bankrupt[pl]:
                    self.cash[p] -= amt
                    self.cash[pl] += amt
            self.last_creditor = None
            self._proceed_to_manage_or_next(p, is_double)

        elif action == "collect_from_each_player":
            amt = card["amount"]
            for pl in range(4):
                if pl != p and not self.bankrupt[pl]:
                    self.cash[pl] -= amt
                    self.cash[p] += amt
            self.last_creditor = p
            self._proceed_to_manage_or_next(p, is_double)

        else:
            self._proceed_to_manage_or_next(p, is_double)

    # ----------------------------------------------------------------------
    # Bankruptcy & End Turn
    # ----------------------------------------------------------------------
    def _send_to_jail(self, p: int):
        """Sends player p directly to Jail."""
        self.position[p] = 10
        self.in_jail[p] = True
        self.jail_turns[p] = 0
        self.consecutive_doubles = 0

    def _resolve_bankruptcy(self, p: int):
        """Liquidates assets or eliminates bankrupt player under official tournament rules."""
        owned_props = [t for t in PURCHASABLE_TILES if self.property_owner[t] == p]
        creditor = getattr(self, "last_creditor", None)
        if creditor == p or (creditor is not None and self.bankrupt[creditor]):
            creditor = None

        # 1. Sell houses/hotels evenly first at 50% refund to raise cash (strictly obeying even-selling & bank scarcity)
        while self.cash[p] < 0:
            sellable = [t for t in owned_props if self.property_houses[t] > 0]
            if not sellable:
                break
            # Pick highest house count tile; find its color group max
            target_t = max(sellable, key=lambda t: self.property_houses[t])
            grp = TILES_DATA[target_t].get("color")
            group_tiles = COLOR_GROUPS[grp] if (grp and grp in COLOR_GROUPS) else [target_t]
            max_h = max(self.property_houses[t] for t in group_tiles)
            top_tiles = [t for t in group_tiles if self.property_houses[t] == max_h]
            t_sell = top_tiles[0]
            cost = self.get_property_house_cost(t_sell, TILES_DATA[t_sell]["house_cost"])
            refund = int(cost * HOUSE_REFUND_RATE)

            if max_h <= 4:
                self.property_houses[t_sell] -= 1
                self.bank_houses += 1
                self.cash[p] += refund
            else:
                # max_h == 5 (Hotel)
                if self.bank_houses >= 4:
                    self.property_houses[t_sell] = 4
                    self.bank_hotels += 1
                    self.bank_houses -= 4
                    self.cash[p] += refund
                else:
                    # Bank housing shortage: liquidate all hotels in group down to highest uniform level bank can supply
                    cur_houses_in_grp = sum(self.property_houses[t] for t in group_tiles if self.property_houses[t] <= 4)
                    cur_hotels_in_grp = sum(1 for t in group_tiles if self.property_houses[t] == 5)
                    total_levels_before = sum(self.property_houses[t] for t in group_tiles)
                    n_grp = len(group_tiles)
                    target_h = 0
                    for cand_h in (3, 2, 1, 0):
                        if n_grp * cand_h <= self.bank_houses + cur_houses_in_grp:
                            target_h = cand_h
                            break
                    self.bank_hotels += cur_hotels_in_grp
                    self.bank_houses += cur_houses_in_grp - (n_grp * target_h)
                    for t_g in group_tiles:
                        self.property_houses[t_g] = target_h
                    levels_sold = total_levels_before - (n_grp * target_h)
                    self.cash[p] += levels_sold * refund

        # 2. Mortgage properties if still in debt
        for t in owned_props:
            if not self.property_mortgaged[t] and self.cash[p] < 0:
                mtg = self.get_property_mortgage(t, TILES_DATA[t]["mortgage"])
                self.property_mortgaged[t] = True
                self.cash[p] += mtg

        # If still in debt: PLAYER IS OFFICIALLY BANKRUPT!
        if self.cash[p] < 0:
            self.bankrupt[p] = True
            if creditor is not None:
                # Official Tournament Rule: Bankrupt to another player!
                # All remaining properties (mortgaged) and GOOJF cards transfer to creditor!
                for t in owned_props:
                    self.property_owner[t] = creditor
                self.event_log.append(f"{self.get_player_name(p).upper()} IS BANKRUPT! All properties transferred to creditor {self.get_player_name(creditor)}.")

                # Transfer GOOJF cards to creditor
                while self.goojf_cards[p] > 0:
                    self.goojf_cards[p] -= 1
                    src = self.goojf_sources[p].pop()
                    self.goojf_cards[creditor] += 1
                    self.goojf_sources[creditor].append(src)
            else:
                # Official Tournament Rule: Bankrupt to the Bank (taxes, fees)
                self.event_log.append(f"{self.get_player_name(p).upper()} IS BANKRUPT to the Bank! Properties returned to Bank.")
                for t in owned_props:
                    # Return houses/hotels
                    h = self.property_houses[t]
                    if h == 5:
                        self.bank_hotels += 1
                    elif h > 0:
                        self.bank_houses += h
                    self.property_houses[t] = 0
                    self.property_mortgaged[t] = False
                    self.property_owner[t] = -1

                # Return GOOJF cards to decks
                while self.goojf_cards[p] > 0:
                    self.goojf_cards[p] -= 1
                    src = self.goojf_sources[p].pop()
                    self.deck_manager.return_goojf(src)

            # Clear pending trade if bankrupt player was involved
            if self.pending_trade and (self.pending_trade["initiator"] == p or self.pending_trade["target"] == p):
                self.pending_trade = None

            # Remove bankrupt player from active auction bidders
            if p in self.auction_active_bidders:
                self.auction_active_bidders.remove(p)
                if self.auction_highest_bidder == p:
                    self.auction_highest_bidder = -1
                    self.auction_highest_bid = 0

            # Immediately end turn and rotate to next surviving player if bankrupt player was the active turn player
            if self.current_player == p:
                self.pending_manage_double = False
                self.pending_auction_double = False
                self.pending_is_double = False
                self._end_turn()
            elif self.active_decision_player == p:
                self.active_decision_player = self.current_player

        self.last_creditor = None

    def _end_turn(self):
        """Rotates to next active player."""
        self.consecutive_doubles = 0
        self.landed_property = None
        self.last_creditor = None
        self.turn_step += 1
        self.manage_actions_this_turn = 0
        self.trade_proposals_this_turn = 0
        self._clear_turn_manage_guards()

        # Check if all players are eliminated
        if all(self.bankrupt):
            self.phase = PHASE_GAME_OVER
            return

        # Advance current_player to next non-bankrupt player
        prev_player = self.current_player
        for _ in range(4):
            self.current_player = (self.current_player + 1) % 4
            if not self.bankrupt[self.current_player]:
                break

        start_p = getattr(self, "starting_player", 0)
        if ((self.current_player - start_p) % 4) <= ((prev_player - start_p) % 4):
            self.round_count += 1

        # If next player is in jail, prompt jail decision; otherwise roll phase
        if self.in_jail[self.current_player]:
            self.phase = PHASE_JAIL_DECISION
        else:
            self.phase = PHASE_ROLL

        self.active_decision_player = self.current_player

    def get_player_net_worth(self, pl: int) -> int:
        """Calculates total net worth: Cash + Property Purchase Prices + Building Equity."""
        if self.bankrupt[pl]:
            return 0
        nw = max(0, self.cash[pl])
        for p in PURCHASABLE_TILES:
            if self.property_owner[p] == pl:
                tile = TILES_DATA[p]
                if not self.property_mortgaged[p]:
                    nw += self.get_property_price(p, tile["price"])
                else:
                    nw += self.get_property_mortgage(p, tile["mortgage"])
                h = self.property_houses[p]
                if h > 0:
                    nw += h * self.get_property_house_cost(p, tile.get("house_cost", 0))
        return nw
