"""Model-3 Tournament Monopoly State Machine Engine.

Upgrades over Model-2:
- fast_snapshot() / restore_snapshot() (~35x faster than copy.deepcopy) + fast_search_mode
- Unconstrained Multi-Property Bundle Trade ({0, 1}^42 sets: 0..28 properties + 0..2 GOOJF)
- Finer 32-bracket Trade Cash resolution ($0..$2,500, starting at $25)
- Bounded 1x Counter-Offer protocol (PHASE_TRADE_COUNTER, loop-free)
- 3-Channel Structured Actor Observation: tile_state (40, 16), scalar_telemetry (192,), event_ring (24, 5)
- Compact Non-Redundant Centralized Critic Global State (592,)
- Per-opponent Online Behavioral Telemetry + PBRS v3 (Expected Rent Receivable & 4-House Lock)
"""

from collections import deque
from typing import Dict, List, Tuple, Optional, Any, Set
import math
import random
import numpy as np

from model_2.engine.board_data import (
    TILES_DATA,
    COLOR_GROUPS,
    TILE_TO_GROUP,
    GROUP_MONOPOLY_BONUS,
    PURCHASABLE_TILES,
    TILE_STREET,
    TILE_RAILROAD,
    TILE_UTILITY,
    TILE_TAX,
    TILE_CARD,
    TILE_SPECIAL,
    UNMORTGAGE_INTEREST_RATE,
    HOUSE_REFUND_RATE,
    DICE_OUTCOMES,
    DICE_PROBABILITIES,
)
from model_2.engine.cards_data import CardDeckManager
from model_2.engine.monopoly_core import (
    DEFAULT_PLAYER_NAMES,
    generate_hardware_entropy_seed,
    TILE_PRICES,
    TILE_MORTGAGES,
    TILE_HOUSE_COSTS,
    PURCHASABLE_TILES_TUPLE,
    COLOR_GROUPS_MONOPOLY,
    MEAN_STREET_MONOPOLY_BONUS,
    STREET_GROUPS_SYNERGY,
    DICE_PROB_TUPLE,
    ACT_CORE_ROLL_PASS,
    ACT_CORE_BUY_DIRECT,
    ACT_CORE_DECLINE_BUY,
    ACT_CORE_PAY_JAIL_50,
    ACT_CORE_USE_GOOJF,
    ACT_CORE_END_MANAGE,
    ACT_AUC_FOLD,
    ACT_AUC_BID_1,
    ACT_AUC_BID_10,
    ACT_AUC_BID_50,
    ACT_AUC_BID_100,
    ACT_AUC_JUMP_LIST,
    ACT_AUC_JUMP_SQUEEZE,
)
from model_2.house_rules.manager import HouseRulesManager

# Phase Definitions (8 phases, backwards-compatible 0..5 ordering + TRADE_COUNTER=6, GAME_OVER=7)
PHASE_ROLL = 0
PHASE_BUY_OR_AUCTION = 1
PHASE_AUCTION = 2
PHASE_JAIL_DECISION = 3
PHASE_PROPERTY_MANAGE = 4
PHASE_TRADE_RESPONSE = 5
PHASE_TRADE_COUNTER = 6
PHASE_GAME_OVER = 7
NUM_PHASES = 8

# Property Head Sub-Actions (162 dims)
ACT_PROP_END_MANAGE = 160
ACT_PROP_PROPOSE_TRADE = 161

# Finer Trade Cash Brackets ($0..$2,500 -> 32 discrete brackets)
TRADE_CASH_BRACKETS: List[int] = [
    0, 25, 50, 75, 100, 125, 150, 175,
    200, 250, 300, 350, 400, 450, 500, 550,
    600, 700, 800, 900, 1000, 1100, 1200, 1300,
    1400, 1500, 1650, 1800, 2000, 2200, 2400, 2500,
]
TRADE_CASH_DIM = len(TRADE_CASH_BRACKETS)  # 32

# Multi-Property Bundle Space (0..39 tile indices + 40: GOOJF #1 + 41: GOOJF #2)
BUNDLE_DIM = 42
GOOJF_SLOT_1 = 40
GOOJF_SLOT_2 = 41

# Structured Observation & Global State Constants
TILE_FEAT_DIM = 16
SCALAR_OBS_DIM = 192
EVENT_RING_LEN = 24
EVENT_FEAT_DIM = 5
GLOBAL_STATE_DIM = 592

# Event Types for Episodic Ring Buffer (0..15)
EV_NONE = 0
EV_ROLL = 1
EV_BUY = 2
EV_AUCTION_BID = 3
EV_AUCTION_WIN = 4
EV_BUILD = 5
EV_SELL = 6
EV_MORTGAGE = 7
EV_UNMORTGAGE = 8
EV_JAIL_IN = 9
EV_JAIL_OUT = 10
EV_RENT = 11
EV_TRADE_PROPOSE = 12
EV_TRADE_ACCEPT = 13
EV_TRADE_REJECT = 14
EV_BANKRUPT = 15

ACTION_KEYS: List[str] = [
    "core",
    "auction",
    "property",
    "trade_target",
    "trade_req_bundle",
    "trade_offer_bundle",
    "trade_offer_cash",
    "trade_req_cash",
    "trade_response",
]

MASK_SHAPES: Dict[str, Tuple[int, ...]] = {
    "core": (6,),
    "auction": (7,),
    "property": (162,),
    "trade_target": (3,),
    "trade_req_bundle": (3, BUNDLE_DIM),
    "trade_offer_bundle": (BUNDLE_DIM,),
    "trade_offer_cash": (TRADE_CASH_DIM,),
    "trade_req_cash": (TRADE_CASH_DIM,),
    "trade_response": (3,),
}

ACT_SHAPES: Dict[str, Tuple[int, ...]] = {
    "core": (),
    "auction": (),
    "property": (),
    "trade_target": (),
    "trade_req_bundle": (BUNDLE_DIM,),
    "trade_offer_bundle": (BUNDLE_DIM,),
    "trade_offer_cash": (),
    "trade_req_cash": (),
    "trade_response": (),
}

STREET_GROUP_LIST = tuple(
    (grp_name, tuple(tiles), float(len(tiles)))
    for grp_name, tiles in COLOR_GROUPS.items()
    if grp_name not in ("RAILROAD", "UTILITY")
)
STREET_GROUP_TO_IDX: Dict[str, int] = {
    grp_name: idx for idx, (grp_name, _, _) in enumerate(STREET_GROUP_LIST)
}
TILE_IS_PURCHASABLE = tuple(p in set(PURCHASABLE_TILES) for p in range(40))
TILE_COLORS = tuple(TILES_DATA[p].get("color") for p in range(40))
TILE_TYPES = tuple(TILES_DATA[p]["type"] for p in range(40))
RAILROAD_TILES = tuple(COLOR_GROUPS["RAILROAD"])
UTILITY_TILES = tuple(COLOR_GROUPS["UTILITY"])
MAX_UNMORTGAGE_COSTS = tuple(
    int(TILES_DATA[p].get("mortgage", 0) * (1.0 + UNMORTGAGE_INTEREST_RATE)) for p in range(40)
)

INACTIVE_MASKS_TEMPLATE: Dict[str, np.ndarray] = {
    "core": np.array([1, 0, 0, 0, 0, 0], dtype=np.int8),
    "auction": np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.int8),
    "property": np.zeros(162, dtype=np.int8),
    "trade_target": np.array([1, 0, 0], dtype=np.int8),
    "trade_req_bundle": np.zeros((3, BUNDLE_DIM), dtype=np.int8),
    "trade_offer_bundle": np.zeros(BUNDLE_DIM, dtype=np.int8),
    "trade_offer_cash": np.zeros(TRADE_CASH_DIM, dtype=np.int8),
    "trade_req_cash": np.zeros(TRADE_CASH_DIM, dtype=np.int8),
    "trade_response": np.array([1, 0, 0], dtype=np.int8),
}
INACTIVE_MASKS_TEMPLATE["property"][ACT_PROP_END_MANAGE] = 1
INACTIVE_MASKS_TEMPLATE["trade_offer_cash"][0] = 1
INACTIVE_MASKS_TEMPLATE["trade_req_cash"][0] = 1


def _amount_bucket(amount: float) -> int:
    """Quantizes dollar amount into compact bucket index [0..15] for event_ring."""
    a = abs(float(amount))
    if a <= 0:
        return 0
    if a < 50:
        return 1
    if a < 100:
        return 2
    if a < 150:
        return 3
    if a < 200:
        return 4
    if a < 300:
        return 5
    if a < 450:
        return 6
    if a < 600:
        return 7
    if a < 800:
        return 8
    if a < 1000:
        return 9
    if a < 1250:
        return 10
    if a < 1500:
        return 11
    if a < 1800:
        return 12
    if a < 2200:
        return 13
    if a < 2800:
        return 14
    return 15


class MonopolyCore:
    """Model-3 Official Tournament Monopoly Engine with Fast Snapshot, Bundle Trade, & Telemetry."""

    def __init__(
        self,
        seed: Optional[int] = None,
        max_rounds: Optional[int] = 80,
        gamma: float = 0.997,
        w1: float = 0.025,
        w2: float = 0.0034,
        w3: float = 0.014,
        w4: float = 0.0006,
        w_syn: float = 0.030,
        beta_starve: float = 4.0,
        w_house: float = 0.0145,
        w_cash: float = 0.0034,
        w_opp_mono: float = 0.85,
        w_lead: float = 0.80,
        w_income: float = 0.0005,
        w_house_lock: float = 0.35,
        pbrs_scale: float = 1.0,
        allow_trade: bool = True,
        player_names: Optional[List[str]] = None,
        house_rules: Optional[Any] = None,
        randomize_first_player: bool = False,
        fast_search_mode: bool = False,
    ):
        if seed is None:
            seed = generate_hardware_entropy_seed()
        self.current_seed = seed
        self.rng = random.Random(seed)
        self.max_rounds = max_rounds
        self.gamma = float(gamma)
        self.randomize_first_player = bool(randomize_first_player)
        self.fast_search_mode = bool(fast_search_mode)
        self.player_names = list(player_names) if player_names is not None else list(DEFAULT_PLAYER_NAMES)
        self.house_rules = HouseRulesManager.from_config(house_rules)

        # PBRS weights (Model-2 balanced + Model-3 Expected Rent Income & House Scarcity Control)
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.w3 = float(w3)
        self.w4 = float(w4)
        self.w_syn = float(w_syn)
        self.beta_starve = float(beta_starve)
        self.w_house = float(w_house)
        self.w_cash = float(w_cash)
        self.w_opp_mono = float(w_opp_mono)
        self.w_lead = float(w_lead)
        self.w_income = float(w_income)
        self.w_house_lock = float(w_house_lock)
        self.pbrs_scale = float(pbrs_scale)
        self.allow_trade = bool(allow_trade)

        self.deck_manager = CardDeckManager(self.rng)
        self.reset(seed)

    def set_allow_trade(self, allow: bool):
        self.allow_trade = bool(allow)

    def set_pbrs_scale(self, scale: float):
        """Updates PBRS scale between iterations and syncs stored prev_potentials cleanly."""
        self.pbrs_scale = max(0.0, float(scale))
        self.prev_potentials = [self._compute_potential(p) for p in range(4)]

    def get_player_name(self, player_id: int) -> str:
        if 0 <= player_id < len(self.player_names):
            return self.player_names[player_id]
        return f"Player {player_id}"

    def _init_effective_tile_costs(self):
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
                eff_m = self.house_rules.get_mortgage_value(tid, int(tile["mortgage"]), modified_price=eff_p)
                mortgages[tid] = eff_m
                mortgages_f[tid] = float(eff_m)

        self.tile_prices = tuple(prices)
        self.tile_house_costs = tuple(house_costs)
        self.tile_mortgages = tuple(mortgages)
        self._tile_prices_f = tuple(prices_f)
        self._tile_house_costs_f = tuple(house_costs_f)
        self._tile_mortgages_f = tuple(mortgages_f)

    def get_property_price(self, tile_id: int, default: int = 0) -> int:
        if tile_id is not None and 0 <= tile_id < 40 and self.tile_prices[tile_id] > 0:
            return self.tile_prices[tile_id]
        return default

    def get_property_house_cost(self, tile_id: int, default: int = 0) -> int:
        if tile_id is not None and 0 <= tile_id < 40 and self.tile_house_costs[tile_id] > 0:
            return self.tile_house_costs[tile_id]
        return default

    def get_property_mortgage(self, tile_id: int, default: int = 0) -> int:
        if tile_id is not None and 0 <= tile_id < 40 and self.tile_mortgages[tile_id] > 0:
            return self.tile_mortgages[tile_id]
        return default

    def reset(
        self, seed: Optional[int] = None, randomize_first_player: Optional[bool] = None
    ) -> Dict[str, Any]:
        if seed is None:
            seed = generate_hardware_entropy_seed()
        self.current_seed = seed
        self.rng.seed(seed)
        self.deck_manager.reset(seed)
        self._init_effective_tile_costs()

        starting_cash = self.house_rules.get_starting_cash(default_cash=1500)
        self.cash = [starting_cash, starting_cash, starting_cash, starting_cash]
        self.position = [0, 0, 0, 0]
        self.in_jail = [False, False, False, False]
        self.jail_turns = [0, 0, 0, 0]
        self.goojf_cards = [0, 0, 0, 0]
        self.goojf_sources: List[List[str]] = [[], [], [], []]
        self.bankrupt = [False, False, False, False]
        self.elimination_order: List[int] = []

        self.bank_houses = 32
        self.bank_hotels = 12
        self.property_owner = [-1] * 40
        self.property_houses = [0] * 40
        self.property_mortgaged = [False] * 40

        use_rand_first = (
            self.randomize_first_player if randomize_first_player is None else bool(randomize_first_player)
        )
        self.starting_player = int(self.rng.randint(0, 3)) if use_rand_first else 0
        self.current_player = self.starting_player
        self.active_decision_player = self.starting_player
        self.last_creditor: Optional[int] = None
        self.round_count = 0
        self.turn_step = 0
        self.consecutive_doubles = 0
        self.last_dice: Tuple[int, int] = (1, 1)
        self.phase = PHASE_JAIL_DECISION if self.in_jail[self.current_player] else PHASE_ROLL

        self.landed_property: Optional[int] = None
        self.pending_is_double: bool = False
        self.pending_manage_double: bool = False
        self.pending_auction_double: bool = False

        # Auction tracking
        self.auction_property: Optional[int] = None
        self.auction_highest_bid = 0
        self.auction_highest_bidder = -1
        self.auction_active_bidders: List[int] = []
        self.auction_current_bidder_idx = 0
        self._auction_has_acted: List[bool] = [False, False, False, False]

        # Trade & Turn counters
        self.pending_trade: Optional[Dict[str, Any]] = None
        self.manage_actions_this_turn = 0
        self.trade_proposals_this_turn = 0
        self.trades_proposed_count = 0
        self.trades_accepted_count = 0
        self.trades_countered_count = 0
        self.houses_built_count = 0

        # Same-turn physical thrash guards
        self.mortgaged_this_turn: Set[int] = set()
        self.unmortgaged_this_turn: Set[int] = set()
        self.built_this_turn: Set[int] = set()
        self.sold_this_turn: Set[int] = set()

        # Episodic event ring (last 24 structured events)
        self.event_ring: deque = deque([(0, -1, -1, 0, 0)] * EVENT_RING_LEN, maxlen=EVENT_RING_LEN)

        # Online per-player behavioral telemetry (EMA ratios + counts)
        self.tel_bid_ratio_ema = [1.0, 1.0, 1.0, 1.0]
        self.tel_overbid_ema = [0.0, 0.0, 0.0, 0.0]
        self.tel_fold_ema = [0.0, 0.0, 0.0, 0.0]
        self.tel_trade_prop_cnt = [0, 0, 0, 0]
        self.tel_trade_rej_cnt = [0, 0, 0, 0]
        self.tel_trade_recv_cnt = [0, 0, 0, 0]
        self.tel_cash_ema = [float(starting_cash)] * 4
        self.tel_jail_stay_cnt = [0, 0, 0, 0]
        self.tel_jail_opp_cnt = [0, 0, 0, 0]
        self.tel_build_cnt = [0, 0, 0, 0]
        self.tel_mortgage_cnt = [0, 0, 0, 0]

        self.event_log: List[str] = []
        self.last_card_drawn: Optional[Dict[str, Any]] = None
        if not self.fast_search_mode:
            self.event_log.append(f"Game started (seed={seed}, cash=${starting_cash}).")
        if self.house_rules.active_rules:
            self.house_rules.apply_game_init(self)

        self.prev_potentials = [self._compute_potential(p) for p in range(4)]
        return self.get_observations()

    # ----------------------------------------------------------------------
    # Fast Deterministic Snapshot / Restore (~35x faster than copy.deepcopy)
    # ----------------------------------------------------------------------
    def save_snapshot(self) -> Tuple[Any, ...]:
        """Captures lightweight deterministic state snapshot for Lookahead Search."""
        pt = self.pending_trade
        pt_copy = None if pt is None else {
            "initiator": pt["initiator"],
            "target": pt["target"],
            "orig_initiator": pt["orig_initiator"],
            "offer_tiles": tuple(pt["offer_tiles"]),
            "req_tiles": tuple(pt["req_tiles"]),
            "offer_goojf": pt["offer_goojf"],
            "req_goojf": pt["req_goojf"],
            "offer_cash": pt["offer_cash"],
            "req_cash": pt["req_cash"],
            "is_counter": pt["is_counter"],
        }
        return (
            self.cash.copy(),
            self.position.copy(),
            self.in_jail.copy(),
            self.jail_turns.copy(),
            self.goojf_cards.copy(),
            [s.copy() for s in self.goojf_sources],
            self.bankrupt.copy(),
            self.elimination_order.copy(),
            self.bank_houses,
            self.bank_hotels,
            self.property_owner.copy(),
            self.property_houses.copy(),
            self.property_mortgaged.copy(),
            self.current_player,
            self.active_decision_player,
            self.last_creditor,
            self.round_count,
            self.turn_step,
            self.consecutive_doubles,
            self.last_dice,
            self.phase,
            self.landed_property,
            self.pending_is_double,
            self.pending_manage_double,
            self.pending_auction_double,
            self.auction_property,
            self.auction_highest_bid,
            self.auction_highest_bidder,
            self.auction_active_bidders.copy(),
            self.auction_current_bidder_idx,
            self._auction_has_acted.copy(),
            pt_copy,
            self.manage_actions_this_turn,
            self.trade_proposals_this_turn,
            self.trades_proposed_count,
            self.trades_accepted_count,
            self.trades_countered_count,
            self.houses_built_count,
            self.mortgaged_this_turn.copy(),
            self.unmortgaged_this_turn.copy(),
            self.built_this_turn.copy(),
            self.sold_this_turn.copy(),
            deque(self.event_ring, maxlen=EVENT_RING_LEN),
            self.tel_bid_ratio_ema.copy(),
            self.tel_overbid_ema.copy(),
            self.tel_fold_ema.copy(),
            self.tel_trade_prop_cnt.copy(),
            self.tel_trade_rej_cnt.copy(),
            self.tel_trade_recv_cnt.copy(),
            self.tel_cash_ema.copy(),
            self.tel_jail_stay_cnt.copy(),
            self.tel_jail_opp_cnt.copy(),
            self.tel_build_cnt.copy(),
            self.tel_mortgage_cnt.copy(),
            self.prev_potentials.copy(),
            self.rng.getstate(),
            deque(self.deck_manager.chance_queue),
            deque(self.deck_manager.chest_queue),
            self.deck_manager.chance_goojf_out,
            self.deck_manager.chest_goojf_out,
            self.starting_player,
            self.allow_trade,
            self.pbrs_scale,
        )

    def restore_snapshot(self, snap: Tuple[Any, ...]):
        """Restores state deterministically from save_snapshot()."""
        (
            cash, position, in_jail, jail_turns, goojf_cards, goojf_sources,
            bankrupt, elimination_order, bank_houses, bank_hotels,
            property_owner, property_houses, property_mortgaged,
            current_player, active_decision_player, last_creditor,
            round_count, turn_step, consecutive_doubles, last_dice, phase,
            landed_property, pending_is_double, pending_manage_double,
            pending_auction_double, auction_property, auction_highest_bid,
            auction_highest_bidder, auction_active_bidders, auction_current_bidder_idx,
            auction_has_acted, pt_copy, manage_actions_this_turn,
            trade_proposals_this_turn, trades_proposed_count, trades_accepted_count,
            trades_countered_count, houses_built_count, mortgaged_this_turn,
            unmortgaged_this_turn, built_this_turn, sold_this_turn, event_ring,
            tel_bid, tel_over, tel_fold, tel_tp, tel_tr, tel_trv,
            tel_cash, tel_js, tel_jo, tel_bld, tel_mtg,
            prev_potentials, rng_state, ch_q, cc_q, ch_out, cc_out,
            starting_player, allow_trade, pbrs_scale,
        ) = snap

        self.cash = cash.copy()
        self.position = position.copy()
        self.in_jail = in_jail.copy()
        self.jail_turns = jail_turns.copy()
        self.goojf_cards = goojf_cards.copy()
        self.goojf_sources = [s.copy() for s in goojf_sources]
        self.bankrupt = bankrupt.copy()
        self.elimination_order = elimination_order.copy()
        self.bank_houses = bank_houses
        self.bank_hotels = bank_hotels
        self.property_owner = property_owner.copy()
        self.property_houses = property_houses.copy()
        self.property_mortgaged = property_mortgaged.copy()
        self.current_player = current_player
        self.active_decision_player = active_decision_player
        self.last_creditor = last_creditor
        self.round_count = round_count
        self.turn_step = turn_step
        self.consecutive_doubles = consecutive_doubles
        self.last_dice = last_dice
        self.phase = phase
        self.landed_property = landed_property
        self.pending_is_double = pending_is_double
        self.pending_manage_double = pending_manage_double
        self.pending_auction_double = pending_auction_double
        self.auction_property = auction_property
        self.auction_highest_bid = auction_highest_bid
        self.auction_highest_bidder = auction_highest_bidder
        self.auction_active_bidders = auction_active_bidders.copy()
        self.auction_current_bidder_idx = auction_current_bidder_idx
        self._auction_has_acted = auction_has_acted.copy()
        self.pending_trade = None if pt_copy is None else dict(pt_copy)
        self.manage_actions_this_turn = manage_actions_this_turn
        self.trade_proposals_this_turn = trade_proposals_this_turn
        self.trades_proposed_count = trades_proposed_count
        self.trades_accepted_count = trades_accepted_count
        self.trades_countered_count = trades_countered_count
        self.houses_built_count = houses_built_count
        self.mortgaged_this_turn = mortgaged_this_turn.copy()
        self.unmortgaged_this_turn = unmortgaged_this_turn.copy()
        self.built_this_turn = built_this_turn.copy()
        self.sold_this_turn = sold_this_turn.copy()
        self.event_ring = deque(event_ring, maxlen=EVENT_RING_LEN)
        self.tel_bid_ratio_ema = tel_bid.copy()
        self.tel_overbid_ema = tel_over.copy()
        self.tel_fold_ema = tel_fold.copy()
        self.tel_trade_prop_cnt = tel_tp.copy()
        self.tel_trade_rej_cnt = tel_tr.copy()
        self.tel_trade_recv_cnt = tel_trv.copy()
        self.tel_cash_ema = tel_cash.copy()
        self.tel_jail_stay_cnt = tel_js.copy()
        self.tel_jail_opp_cnt = tel_jo.copy()
        self.tel_build_cnt = tel_bld.copy()
        self.tel_mortgage_cnt = tel_mtg.copy()
        self.prev_potentials = prev_potentials.copy()
        self.rng.setstate(rng_state)
        self.deck_manager.chance_queue = deque(ch_q)
        self.deck_manager.chest_queue = deque(cc_q)
        self.deck_manager.chance_goojf_out = ch_out
        self.deck_manager.chest_goojf_out = cc_out
        self.starting_player = starting_player
        self.allow_trade = allow_trade
        self.pbrs_scale = pbrs_scale

    # ----------------------------------------------------------------------
    # Event Ring & Logging Helpers
    # ----------------------------------------------------------------------
    def _push_event(self, ev_type: int, actor: int, target: int, tile_or_grp: int, amount: float = 0.0):
        self.event_ring.append((int(ev_type), int(actor), int(target), int(tile_or_grp), _amount_bucket(amount)))

    def _log(self, msg: str):
        if not self.fast_search_mode:
            self.event_log.append(msg)

    # ----------------------------------------------------------------------
    # PBRS v3 & Analytics
    # ----------------------------------------------------------------------
    def _calculate_rent(self, tile_id: int, dice_sum: int, multiplier_override: Optional[int] = None) -> int:
        tile = TILES_DATA[tile_id]
        owner = self.property_owner[tile_id]
        if owner == -1 or self.property_mortgaged[tile_id]:
            return 0

        t_type = tile["type"]
        if t_type == TILE_STREET:
            h = self.property_houses[tile_id]
            rents = tile["rent"]
            if h == 0:
                grp = tile["color"]
                group_tiles = COLOR_GROUPS[grp]
                all_owned = all(self.property_owner[t] == owner for t in group_tiles)
                none_mortgaged = all(not self.property_mortgaged[t] for t in group_tiles)
                return rents[0] * 2 if (all_owned and none_mortgaged) else rents[0]
            return rents[h]
        elif t_type == TILE_RAILROAD:
            rr_owned = sum(
                1 for t in RAILROAD_TILES if self.property_owner[t] == owner and not self.property_mortgaged[t]
            )
            base_rent = 25 * (2 ** (max(1, rr_owned) - 1))
            return base_rent * multiplier_override if multiplier_override else base_rent
        elif t_type == TILE_UTILITY:
            ut_owned = sum(
                1 for t in UTILITY_TILES if self.property_owner[t] == owner and not self.property_mortgaged[t]
            )
            mult = 10 if ut_owned == 2 or multiplier_override == 10 else 4
            return mult * dice_sum
        return 0

    def _compute_expected_rent_flows(self) -> Tuple[List[float], List[float]]:
        """Computes per-player next-roll expected rent outflow and receivable inflow."""
        outflow = [0.0, 0.0, 0.0, 0.0]
        inflow = [0.0, 0.0, 0.0, 0.0]
        for pl in range(4):
            if self.bankrupt[pl] or self.in_jail[pl]:
                continue
            pos_pl = self.position[pl]
            for d, prob in DICE_PROB_TUPLE:
                t_id = (pos_pl + d) % 40
                owner = self.property_owner[t_id]
                if owner != -1 and owner != pl and not self.bankrupt[owner] and not self.property_mortgaged[t_id]:
                    r = float(self._calculate_rent(t_id, dice_sum=d)) * prob
                    outflow[pl] += r
                    inflow[owner] += r
        return outflow, inflow

    def _compute_potential(self, player_id: int) -> float:
        """Model-3 PBRS Φ^i(s): Model-2 Terms 1..8 + Term 9 (Expected Rent Receivable) + Term 10 (House-Lock Control)."""
        if self.bankrupt[player_id] or self.pbrs_scale <= 0.0:
            return 0.0

        c_i = max(0, self.cash[player_id])
        term1 = self.w1 * math.log(c_i + 1.0) + self.w_cash * float(c_i)

        term2_prop = 0.0
        term2_house = 0.0
        my_non_hotel_houses = 0
        for p in PURCHASABLE_TILES_TUPLE:
            if self.property_owner[p] == player_id:
                m_p = 1.0 if self.property_mortgaged[p] else 0.0
                term2_prop += self._tile_prices_f[p] - m_p * 1.12 * self._tile_mortgages_f[p]
                h_p = self.property_houses[p]
                term2_house += float(h_p * self._tile_house_costs_f[p])
                if 1 <= h_p <= 4:
                    my_non_hotel_houses += h_p
        term2 = (self.w2 * term2_prop) + (self.w_house * term2_house)

        term3_sum = 0.0
        street_monopolies_count = 0
        opp_has_underbuilt_monopoly = False
        opp_threat_score = 0.0

        for tiles, bonus, is_street in COLOR_GROUPS_MONOPOLY:
            first_owner = self.property_owner[tiles[0]]
            if first_owner != -1 and all(self.property_owner[t] == first_owner for t in tiles):
                if first_owner == player_id:
                    term3_sum += bonus
                    if is_street:
                        street_monopolies_count += 1
                elif is_street and not self.bankrupt[first_owner]:
                    avg_h = sum(self.property_houses[t] for t in tiles) / float(len(tiles))
                    value_scale = 0.5 + 0.5 * (bonus / MEAN_STREET_MONOPOLY_BONUS)
                    opp_threat_score += value_scale * (1.0 + 0.15 * avg_h)
                    if avg_h < 4.0:
                        opp_has_underbuilt_monopoly = True

        term3 = self.w3 * term3_sum
        term7_opp_threat = self.w_opp_mono * opp_threat_score

        # Expected rent hazard (Term 4) & Expected rent receivable (Term 9)
        term4_sum = 0.0
        if not self.in_jail[player_id]:
            pos_i = self.position[player_id]
            for d, prob in DICE_PROB_TUPLE:
                t_tile = (pos_i + d) % 40
                owner = self.property_owner[t_tile]
                if owner != -1 and owner != player_id and not self.bankrupt[owner] and not self.property_mortgaged[t_tile]:
                    term4_sum += prob * self._calculate_rent(t_tile, dice_sum=d)
        term4 = self.w4 * term4_sum

        term9_sum = 0.0
        if self.w_income > 0.0:
            for j in range(4):
                if j == player_id or self.bankrupt[j] or self.in_jail[j]:
                    continue
                pos_j = self.position[j]
                for d, prob in DICE_PROB_TUPLE:
                    t_tile = (pos_j + d) % 40
                    if self.property_owner[t_tile] == player_id and not self.property_mortgaged[t_tile]:
                        term9_sum += prob * self._calculate_rent(t_tile, dice_sum=d)
        term9_income = self.w_income * term9_sum

        # Term 5: Color-matching synergy
        term5_sum = 0.0
        for _, tiles, n_tiles in STREET_GROUPS_SYNERGY:
            owned_count = sum(1 for t in tiles if self.property_owner[t] == player_id)
            frac = owned_count / n_tiles
            term5_sum += (frac ** 2) * 50.0
        term5 = self.w_syn * term5_sum

        # Term 6: Monopoly starvation penalty
        round_frac = min(2.5, self.round_count / 40.0)
        term6_starve = self.beta_starve * (round_frac ** 2) if street_monopolies_count == 0 else 0.0

        # Term 8: Relative net-worth lead
        alive_opps = [pl for pl in range(4) if pl != player_id and not self.bankrupt[pl]]
        term8_lead = 0.0
        if alive_opps and self.w_lead > 0.0:
            my_nw = float(self.get_player_net_worth(player_id))
            avg_opp_nw = sum(float(self.get_player_net_worth(pl)) for pl in alive_opps) / float(len(alive_opps))
            term8_lead = self.w_lead * math.tanh((my_nw - avg_opp_nw) / 1500.0)

        # Term 10: Housing Shortage Lock Control (incentivize holding 4-house locks when bank houses are tight)
        term10_house_lock = 0.0
        if self.w_house_lock > 0.0 and self.bank_houses <= 16 and opp_has_underbuilt_monopoly:
            term10_house_lock = self.w_house_lock * (my_non_hotel_houses / 32.0)

        raw_phi = (
            term1 + term2 + term3 - term4 + term5 - term6_starve
            - term7_opp_threat + term8_lead + term9_income + term10_house_lock
        )
        return float(self.pbrs_scale * raw_phi)

    def _effective_trade_prop_value(self, tile_id: int) -> float:
        if not (0 <= tile_id < 40):
            return 0.0
        base_price = float(self.get_property_price(tile_id, 0))
        if self.property_mortgaged[tile_id]:
            m = float(self.get_property_mortgage(tile_id, TILES_DATA[tile_id].get("mortgage", 0)))
            return max(0.0, base_price - (1.0 + UNMORTGAGE_INTEREST_RATE) * m)
        return base_price

    def compute_hazard_threat(self, player_id: int) -> float:
        if self.in_jail[player_id]:
            return 0.0
        hazard_threat = 0.0
        pos = self.position[player_id]
        for d in DICE_OUTCOMES:
            prob = DICE_PROBABILITIES[d]
            target_t = (pos + d) % 40
            owner = self.property_owner[target_t]
            if owner != -1 and owner != player_id and not self.bankrupt[owner] and not self.property_mortgaged[target_t]:
                if self.property_houses[target_t] >= 2:
                    hazard_threat += prob * self._calculate_rent(target_t, dice_sum=d)
        return hazard_threat

    def _get_squeeze_bid_target(self, player_id: int) -> Optional[int]:
        other_bidders = [b for b in self.auction_active_bidders if b != player_id and not self.bankrupt[b]]
        if not other_bidders:
            return None
        squeeze_target = max(self.cash[b] for b in other_bidders) + 1
        if squeeze_target > self.auction_highest_bid and self.cash[player_id] >= squeeze_target:
            return squeeze_target
        return None

    def get_player_net_worth(self, pl: int) -> int:
        if self.bankrupt[pl]:
            return 0
        nw = max(0, self.cash[pl])
        for p in PURCHASABLE_TILES_TUPLE:
            if self.property_owner[p] == pl:
                if not self.property_mortgaged[p]:
                    nw += self.get_property_price(p, 0)
                else:
                    nw += self.get_property_mortgage(p, 0)
                h = self.property_houses[p]
                if h > 0:
                    nw += h * self.get_property_house_cost(p, 0)
        return nw

    # ----------------------------------------------------------------------
    # Action Masks Generator (Strict Official Rules + Unconstrained Bundles)
    # ----------------------------------------------------------------------
    def _is_tile_tradable_by(self, tile_id: int, owner_id: int) -> bool:
        if not (0 <= tile_id < 40 and TILE_IS_PURCHASABLE[tile_id]):
            return False
        if self.property_owner[tile_id] != owner_id:
            return False
        grp = TILE_COLORS[tile_id]
        if grp is not None and any(self.property_houses[t] > 0 for t in COLOR_GROUPS[grp]):
            return False
        return True

    def get_action_masks(self, player_id: int) -> Dict[str, np.ndarray]:
        if player_id != self.active_decision_player or self.bankrupt[player_id] or self.phase == PHASE_GAME_OVER:
            return {k: v.copy() for k, v in INACTIVE_MASKS_TEMPLATE.items()}

        mask_core = np.zeros(6, dtype=np.int8)
        mask_auction = np.zeros(7, dtype=np.int8)
        mask_property = np.zeros(162, dtype=np.int8)
        mask_trade_target = np.zeros(3, dtype=np.int8)
        mask_trade_req_bundle = np.zeros((3, BUNDLE_DIM), dtype=np.int8)
        mask_trade_offer_bundle = np.zeros(BUNDLE_DIM, dtype=np.int8)
        mask_trade_offer_cash = np.zeros(TRADE_CASH_DIM, dtype=np.int8)
        mask_trade_req_cash = np.zeros(TRADE_CASH_DIM, dtype=np.int8)
        mask_trade_response = np.zeros(3, dtype=np.int8)

        if self.phase == PHASE_ROLL:
            mask_core[ACT_CORE_ROLL_PASS] = 1

        elif self.phase == PHASE_JAIL_DECISION:
            mask_core[ACT_CORE_ROLL_PASS] = 1
            if self.cash[player_id] >= 50:
                mask_core[ACT_CORE_PAY_JAIL_50] = 1
            if self.goojf_cards[player_id] > 0:
                mask_core[ACT_CORE_USE_GOOJF] = 1

        elif self.phase == PHASE_BUY_OR_AUCTION:
            tile_id = self.landed_property
            price = self.get_property_price(tile_id, 999999) if tile_id is not None else 999999
            if self.cash[player_id] >= price:
                mask_core[ACT_CORE_BUY_DIRECT] = 1
            mask_core[ACT_CORE_DECLINE_BUY] = 1

        elif self.phase == PHASE_AUCTION:
            mask_auction[ACT_AUC_FOLD] = 1
            if self.auction_property is not None:
                current_bid = self.auction_highest_bid
                cash = self.cash[player_id]
                for act_idx, inc in enumerate((1, 10, 50, 100), start=1):
                    if cash >= current_bid + inc:
                        mask_auction[act_idx] = 1
                price = self.get_property_price(self.auction_property, 100)
                if price > current_bid and cash >= price:
                    mask_auction[ACT_AUC_JUMP_LIST] = 1
                if self._get_squeeze_bid_target(player_id) is not None:
                    mask_auction[ACT_AUC_JUMP_SQUEEZE] = 1

        elif self.phase in (PHASE_PROPERTY_MANAGE, PHASE_TRADE_COUNTER):
            is_counter_phase = self.phase == PHASE_TRADE_COUNTER
            mask_core[ACT_CORE_ROLL_PASS] = 1

            if is_counter_phase:
                # Counter-offer phase strictly constructs one counter bundle back to orig_initiator
                mask_property[ACT_PROP_END_MANAGE] = 1
                self._populate_trade_proposal_masks(
                    proposer=player_id,
                    locked_target=self.pending_trade["orig_initiator"] if self.pending_trade else None,
                    mask_trade_target=mask_trade_target,
                    mask_trade_req_bundle=mask_trade_req_bundle,
                    mask_trade_offer_bundle=mask_trade_offer_bundle,
                    mask_trade_offer_cash=mask_trade_offer_cash,
                    mask_trade_req_cash=mask_trade_req_cash,
                )
            else:
                mask_property[ACT_PROP_END_MANAGE] = 1
                same_turn_guard = self.manage_actions_this_turn > 0
                mortgaged_turn = self.mortgaged_this_turn if same_turn_guard else set()
                unmortgaged_turn = self.unmortgaged_this_turn if same_turn_guard else set()
                built_turn = self.built_this_turn if same_turn_guard else set()
                sold_turn = self.sold_this_turn if same_turn_guard else set()

                for p in PURCHASABLE_TILES_TUPLE:
                    if self.property_owner[p] != player_id:
                        continue
                    grp = TILE_COLORS[p]
                    h = self.property_houses[p]
                    h_cost = self.get_property_house_cost(p, 0)

                    # Build house / hotel
                    if TILE_TYPES[p] == TILE_STREET and grp and p not in sold_turn:
                        group_tiles = COLOR_GROUPS[grp]
                        if all(self.property_owner[t] == player_id and not self.property_mortgaged[t] for t in group_tiles):
                            min_h = min(self.property_houses[t] for t in group_tiles)
                            if h == min_h and h < 5 and self.cash[player_id] >= h_cost:
                                if (h < 4 and self.bank_houses > 0) or (h == 4 and self.bank_hotels > 0):
                                    mask_property[p] = 1

                    # Sell house / hotel
                    if TILE_TYPES[p] == TILE_STREET and grp and h > 0 and p not in built_turn:
                        group_tiles = COLOR_GROUPS[grp]
                        if h == max(self.property_houses[t] for t in group_tiles):
                            if h <= 4 or (h == 5 and self.bank_houses >= 4):
                                mask_property[40 + p] = 1

                    # Mortgage
                    if not self.property_mortgaged[p] and p not in unmortgaged_turn:
                        has_houses = bool(grp and TILE_TYPES[p] == TILE_STREET and any(self.property_houses[t] > 0 for t in COLOR_GROUPS[grp]))
                        if not has_houses:
                            mask_property[80 + p] = 1

                    # Unmortgage
                    if self.property_mortgaged[p] and p not in mortgaged_turn:
                        unmtg_cost = int(self.get_property_mortgage(p, 0) * (1.0 + UNMORTGAGE_INTEREST_RATE))
                        if self.cash[player_id] >= unmtg_cost:
                            mask_property[120 + p] = 1

                # Trade initiation gate (161)
                if self.allow_trade and self.trade_proposals_this_turn == 0:
                    can_propose = self._populate_trade_proposal_masks(
                        proposer=player_id,
                        locked_target=None,
                        mask_trade_target=mask_trade_target,
                        mask_trade_req_bundle=mask_trade_req_bundle,
                        mask_trade_offer_bundle=mask_trade_offer_bundle,
                        mask_trade_offer_cash=mask_trade_offer_cash,
                        mask_trade_req_cash=mask_trade_req_cash,
                    )
                    if can_propose:
                        mask_property[ACT_PROP_PROPOSE_TRADE] = 1

        elif self.phase == PHASE_TRADE_RESPONSE:
            mask_trade_response[0] = 1  # Reject always legal
            if self.pending_trade:
                req_c = self.pending_trade["req_cash"]
                off_c = self.pending_trade["offer_cash"]
                init = self.pending_trade["initiator"]
                if self.cash[player_id] >= req_c and self.cash[init] >= off_c:
                    mask_trade_response[1] = 1  # Accept
                if not self.pending_trade.get("is_counter", False) and not self.bankrupt[init]:
                    mask_trade_response[2] = 1  # 1x Counter-Offer

        # Safety invariants (guaranteed at least 1 legal option in scalar heads)
        if not np.any(mask_core):
            mask_core[0] = 1
        if not np.any(mask_auction):
            mask_auction[0] = 1
        if not np.any(mask_property):
            mask_property[ACT_PROP_END_MANAGE] = 1
        if not np.any(mask_trade_target):
            mask_trade_target[0] = 1
        if not np.any(mask_trade_offer_cash):
            mask_trade_offer_cash[0] = 1
        if not np.any(mask_trade_req_cash):
            mask_trade_req_cash[0] = 1
        if not np.any(mask_trade_response):
            mask_trade_response[0] = 1

        return {
            "core": mask_core,
            "auction": mask_auction,
            "property": mask_property,
            "trade_target": mask_trade_target,
            "trade_req_bundle": mask_trade_req_bundle,
            "trade_offer_bundle": mask_trade_offer_bundle,
            "trade_offer_cash": mask_trade_offer_cash,
            "trade_req_cash": mask_trade_req_cash,
            "trade_response": mask_trade_response,
        }

    def _populate_trade_proposal_masks(
        self,
        proposer: int,
        locked_target: Optional[int],
        mask_trade_target: np.ndarray,
        mask_trade_req_bundle: np.ndarray,
        mask_trade_offer_bundle: np.ndarray,
        mask_trade_offer_cash: np.ndarray,
        mask_trade_req_cash: np.ndarray,
    ) -> bool:
        """Fills unconstrained target-indexed bundle masks & cash masks; returns True if >=1 meaningful trade exists."""
        # Own offerable assets (0..39 properties without houses in group + up to 2 GOOJF cards)
        has_offer_asset = False
        for p in PURCHASABLE_TILES_TUPLE:
            if self._is_tile_tradable_by(p, proposer):
                mask_trade_offer_bundle[p] = 1
                has_offer_asset = True
        if self.goojf_cards[proposer] >= 1:
            mask_trade_offer_bundle[GOOJF_SLOT_1] = 1
            has_offer_asset = True
        if self.goojf_cards[proposer] >= 2:
            mask_trade_offer_bundle[GOOJF_SLOT_2] = 1

        proposer_can_offer_cash = self.cash[proposer] >= TRADE_CASH_BRACKETS[1]
        for b_idx, b_val in enumerate(TRADE_CASH_BRACKETS):
            if self.cash[proposer] >= b_val:
                mask_trade_offer_cash[b_idx] = 1

        any_valid_target = False
        for rel_idx in range(3):
            target_pl = (proposer + 1 + rel_idx) % 4
            if self.bankrupt[target_pl]:
                continue
            if locked_target is not None and target_pl != locked_target:
                continue

            has_target_asset = False
            for p in PURCHASABLE_TILES_TUPLE:
                if self._is_tile_tradable_by(p, target_pl):
                    mask_trade_req_bundle[rel_idx, p] = 1
                    has_target_asset = True
            if self.goojf_cards[target_pl] >= 1:
                mask_trade_req_bundle[rel_idx, GOOJF_SLOT_1] = 1
                has_target_asset = True
            if self.goojf_cards[target_pl] >= 2:
                mask_trade_req_bundle[rel_idx, GOOJF_SLOT_2] = 1

            target_can_pay = self.cash[target_pl] >= TRADE_CASH_BRACKETS[1]
            # Target is valid if at least one non-empty bilateral exchange can occur
            can_buy_from_target = has_target_asset and (has_offer_asset or proposer_can_offer_cash)
            can_sell_to_target = has_offer_asset and (has_target_asset or target_can_pay)
            if can_buy_from_target or can_sell_to_target:
                mask_trade_target[rel_idx] = 1
                any_valid_target = True

        mask_trade_req_cash[:] = 1
        return any_valid_target

    # ----------------------------------------------------------------------
    # Structured 3-Channel Actor Observation & Compact Global Critic State
    # ----------------------------------------------------------------------
    def get_actor_structured_obs(
        self,
        player_id: int,
        outflow: Optional[List[float]] = None,
        inflow: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Builds (tile_state[40,16], scalar_telemetry[192], event_ring[24,5]) relative to player_id."""
        if outflow is None or inflow is None:
            outflow, inflow = self._compute_expected_rent_flows()

        tile_state = np.zeros((40, TILE_FEAT_DIM), dtype=np.float32)
        pt = self.pending_trade
        pt_offer_tiles = set(pt["offer_tiles"]) if (pt and self.phase == PHASE_TRADE_RESPONSE) else set()
        pt_req_tiles = set(pt["req_tiles"]) if (pt and self.phase == PHASE_TRADE_RESPONSE) else set()

        # Precompute relative owner mapping: abs -> 0(unowned), 1(self), 2(+1), 3(+2), 4(+3)
        rel_slot = [(player_id + off) % 4 for off in range(4)]
        abs_to_rel = {rel_slot[off]: off + 1 for off in range(4)}

        # Precompute "completes group for player_id" flags per tile
        completes_my_group = [False] * 40
        for _, g_tiles, n_t in STREET_GROUP_LIST:
            owned_cnt = sum(1 for t in g_tiles if self.property_owner[t] == player_id)
            if owned_cnt == int(n_t) - 1:
                for t in g_tiles:
                    if self.property_owner[t] != player_id:
                        completes_my_group[t] = True

        for tid in range(40):
            own = self.property_owner[tid]
            tile_state[tid, 0 if own == -1 else abs_to_rel[own]] = 1.0
            tile_state[tid, 5] = self.property_houses[tid] / 5.0
            tile_state[tid, 6] = 1.0 if self.property_mortgaged[tid] else 0.0
            tile_state[tid, 7] = self._tile_prices_f[tid] / 400.0
            if own != -1 and not self.property_mortgaged[tid]:
                tile_state[tid, 8] = self._calculate_rent(tid, dice_sum=7) / 2000.0
            if completes_my_group[tid]:
                tile_state[tid, 13] = 1.0
            if tid in pt_offer_tiles:
                tile_state[tid, 14] = 1.0
            if tid in pt_req_tiles:
                tile_state[tid, 15] = 1.0

        for off, abs_p in enumerate(rel_slot):
            if not self.bankrupt[abs_p]:
                tile_state[self.position[abs_p], 9 + off] = 1.0

        # Scalar Telemetry (192 dims exact)
        scalar = np.zeros(SCALAR_OBS_DIM, dtype=np.float32)

        # [0..39]: 4 relative players x 10 features
        max_board_rent = 50.0
        opp_needs_houses = False
        for off, abs_p in enumerate(rel_slot):
            base = off * 10
            nw = float(self.get_player_net_worth(abs_p)) if not self.bankrupt[abs_p] else 0.0
            mono_cnt = 0
            for _, g_tiles, _ in STREET_GROUP_LIST:
                if all(self.property_owner[t] == abs_p for t in g_tiles):
                    mono_cnt += 1
                    if off > 0 and any(self.property_houses[t] < 4 for t in g_tiles):
                        opp_needs_houses = True
            scalar[base + 0] = self.position[abs_p] / 40.0
            scalar[base + 1] = self.cash[abs_p] / 2000.0
            scalar[base + 2] = 1.0 if self.in_jail[abs_p] else 0.0
            scalar[base + 3] = self.jail_turns[abs_p] / 3.0
            scalar[base + 4] = self.goojf_cards[abs_p] / 2.0
            scalar[base + 5] = 1.0 if self.bankrupt[abs_p] else 0.0
            scalar[base + 6] = nw / 5000.0
            scalar[base + 7] = mono_cnt / 8.0
            scalar[base + 8] = outflow[abs_p] / 500.0
            scalar[base + 9] = inflow[abs_p] / 500.0

        for p_id in PURCHASABLE_TILES_TUPLE:
            own = self.property_owner[p_id]
            if own != -1 and own != player_id and not self.bankrupt[own] and not self.property_mortgaged[p_id]:
                r = self._calculate_rent(p_id, dice_sum=7)
                if r > max_board_rent:
                    max_board_rent = float(r)

        # [40..47]: Phase one-hot (8 dims) - referenced explicitly by ActorNetwork
        if 0 <= self.phase < NUM_PHASES:
            scalar[40 + self.phase] = 1.0

        # [48..55]: Bank & Scarcity (8 dims)
        total_needed_h4 = 0
        for _, g_tiles, _ in STREET_GROUP_LIST:
            own0 = self.property_owner[g_tiles[0]]
            if own0 != -1 and not self.bankrupt[own0] and all(self.property_owner[t] == own0 for t in g_tiles):
                total_needed_h4 += sum(max(0, 4 - self.property_houses[t]) for t in g_tiles)
        shortage_ratio = self.bank_houses / float(max(1, total_needed_h4))
        scalar[48] = self.bank_houses / 32.0
        scalar[49] = self.bank_hotels / 12.0
        scalar[50] = min(2.0, shortage_ratio)
        scalar[51] = 0.0 if not self.max_rounds else min(2.0, self.round_count / float(self.max_rounds))
        scalar[52] = self.last_dice[0] / 6.0
        scalar[53] = self.last_dice[1] / 6.0
        scalar[54] = 1.0 if self.allow_trade else 0.0
        scalar[55] = 1.0 if (pt is not None and pt.get("is_counter", False)) else 0.0

        # [56..63]: 4-House Lock hotel-release warning per street color group (8 dims)
        for g_idx, (_, g_tiles, _) in enumerate(STREET_GROUP_LIST):
            if opp_needs_houses and all(
                self.property_owner[t] == player_id and self.property_houses[t] == 4 for t in g_tiles
            ):
                scalar[56 + g_idx] = 1.0

        # [64..71]: Property transaction / auction context (8 dims)
        scalar[64:72] = self._get_property_transaction_features(player_id)

        # [72..79]: Risk & Buffer descriptors (8 dims)
        hazard = self.compute_hazard_threat(player_id)
        my_non_hotel = sum(self.property_houses[p] for p in PURCHASABLE_TILES_TUPLE if self.property_owner[p] == player_id and 1 <= self.property_houses[p] <= 4)
        scalar[72] = min(2.0, hazard / 200.0)
        scalar[73] = min(5.0, max(0.0, self.cash[player_id]) / max(100.0, max_board_rent))
        scalar[74] = my_non_hotel / 32.0
        scalar[75] = 1.0 if opp_needs_houses else 0.0
        scalar[76] = min(2.0, max_board_rent / 1000.0)
        scalar[77] = 1.0 if self.trade_proposals_this_turn > 0 else 0.0
        scalar[78] = min(1.0, self.manage_actions_this_turn / 12.0)
        scalar[79] = 1.0 if self.current_player == player_id else 0.0

        # [80..109]: Opponent Behavioral Telemetry (3 relative opponents x 10 dims = 30 dims)
        for opp_off in range(3):
            opp_id = rel_slot[1 + opp_off]
            b = 80 + opp_off * 10
            tp_cnt = max(1, self.tel_trade_recv_cnt[opp_id])
            jo_cnt = max(1, self.tel_jail_opp_cnt[opp_id])
            ts = max(1, self.turn_step)
            scalar[b + 0] = min(3.0, self.tel_bid_ratio_ema[opp_id])
            scalar[b + 1] = self.tel_overbid_ema[opp_id]
            scalar[b + 2] = self.tel_fold_ema[opp_id]
            scalar[b + 3] = min(1.0, self.tel_trade_prop_cnt[opp_id] / float(ts))
            scalar[b + 4] = self.tel_trade_rej_cnt[opp_id] / float(tp_cnt)
            scalar[b + 5] = min(3.0, self.tel_cash_ema[opp_id] / 1000.0)
            scalar[b + 6] = self.tel_jail_stay_cnt[opp_id] / float(jo_cnt)
            scalar[b + 7] = min(1.0, self.tel_build_cnt[opp_id] / 20.0)
            scalar[b + 8] = min(1.0, self.tel_mortgage_cnt[opp_id] / float(ts))
            scalar[b + 9] = min(3.0, outflow[opp_id] / 500.0)

        # [110..173]: Perspective-relative Color Synergy Vector (64 dims)
        for g_idx, (_, g_tiles, n_tiles) in enumerate(STREET_GROUP_LIST):
            owners = [self.property_owner[t] for t in g_tiles]
            for off, abs_p in enumerate(rel_slot):
                cnt = sum(1 for o in owners if o == abs_p)
                scalar[110 + g_idx * 4 + off] = cnt / n_tiles
                if cnt == int(n_tiles) - 1 and any(o != abs_p and o != -1 for o in owners):
                    scalar[142 + g_idx * 4 + off] = 1.0

        # [174..191]: Pending Trade Scalars (18 dims)
        if pt is not None and self.phase in (PHASE_TRADE_RESPONSE, PHASE_TRADE_COUNTER):
            init = pt["initiator"]
            targ = pt["target"]
            scalar[174] = 1.0
            scalar[175] = 1.0 if pt.get("is_counter", False) else 0.0
            scalar[176 + (abs_to_rel[init] - 1)] = 1.0
            scalar[180 + (abs_to_rel[targ] - 1)] = 1.0
            scalar[184] = pt["offer_cash"] / 1000.0
            scalar[185] = pt["req_cash"] / 1000.0
            scalar[186] = pt["offer_goojf"] / 2.0
            scalar[187] = pt["req_goojf"] / 2.0
            other_p = init if player_id == targ else targ
            my_gain, my_give = self._compute_trade_impact(player_id, other_p, pt)
            scalar[188] = my_gain[0]   # completes_my_mono
            scalar[189] = my_gain[1]   # completes_opp_mono
            scalar[190] = my_give[0]   # net_cash_gain / 1000
            scalar[191] = my_give[1]   # net_equity_gain / 1000

        # Event Ring translated to perspective-relative IDs: (24, 5) int16
        ev_arr = np.empty((EVENT_RING_LEN, EVENT_FEAT_DIM), dtype=np.int16)
        for i, (ev_t, act, tgt, tile_g, amt_b) in enumerate(self.event_ring):
            ev_arr[i, 0] = ev_t
            ev_arr[i, 1] = abs_to_rel[act] if act in abs_to_rel else 0
            ev_arr[i, 2] = abs_to_rel[tgt] if tgt in abs_to_rel else 0
            ev_arr[i, 3] = tile_g
            ev_arr[i, 4] = amt_b

        return tile_state, scalar, ev_arr

    def _compute_trade_impact(
        self, me: int, other: int, trade: Dict[str, Any]
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        init = trade["initiator"]
        if me == trade["target"]:
            gain_tiles, give_tiles = trade["offer_tiles"], trade["req_tiles"]
            gain_c, give_c = float(trade["offer_cash"]), float(trade["req_cash"])
            gain_g, give_g = trade["offer_goojf"], trade["req_goojf"]
        else:
            gain_tiles, give_tiles = trade["req_tiles"], trade["offer_tiles"]
            gain_c, give_c = float(trade["req_cash"]), float(trade["offer_cash"])
            gain_g, give_g = trade["req_goojf"], trade["offer_goojf"]

        sim_owner = list(self.property_owner)
        for tid in trade["offer_tiles"]:
            if 0 <= tid < 40 and sim_owner[tid] == init:
                sim_owner[tid] = trade["target"]
        for tid in trade["req_tiles"]:
            if 0 <= tid < 40 and sim_owner[tid] == trade["target"]:
                sim_owner[tid] = init

        def _mono_cnt(pl: int, owners: List[int]) -> int:
            return sum(1 for _, gt, _ in STREET_GROUP_LIST if all(owners[t] == pl for t in gt))

        my_comp = 1.0 if _mono_cnt(me, sim_owner) > _mono_cnt(me, self.property_owner) else 0.0
        opp_comp = 1.0 if (0 <= other < 4 and _mono_cnt(other, sim_owner) > _mono_cnt(other, self.property_owner)) else 0.0

        gain_val = sum(self._effective_trade_prop_value(t) for t in gain_tiles) + gain_g * 50.0
        give_val = sum(self._effective_trade_prop_value(t) for t in give_tiles) + give_g * 50.0
        net_cash = (gain_c - give_c) / 1000.0
        net_eq = ((gain_val + gain_c) - (give_val + give_c)) / 1000.0
        return (my_comp, opp_comp), (net_cash, net_eq)

    def _get_property_transaction_features(self, player_id: int) -> np.ndarray:
        target_prop = self.auction_property if self.phase == PHASE_AUCTION else (
            self.landed_property if self.phase == PHASE_BUY_OR_AUCTION else None
        )
        if target_prop is None or not (0 <= target_prop < 40):
            return np.zeros(8, dtype=np.float32)

        grp = TILE_COLORS[target_prop]
        price = float(self.get_property_price(target_prop, 100))

        for_me = 0.0
        if grp and grp in COLOR_GROUPS:
            gt = COLOR_GROUPS[grp]
            if sum(1 for t in gt if self.property_owner[t] == player_id) == len(gt) - 1:
                for_me = 1.0

        for_opp = 0.0
        if grp and grp in COLOR_GROUPS:
            gt = COLOR_GROUPS[grp]
            if self.phase == PHASE_AUCTION and self.auction_highest_bidder != -1:
                hb = self.auction_highest_bidder
                if hb != player_id and sum(1 for t in gt if self.property_owner[t] == hb) == len(gt) - 1:
                    for_opp = 1.0
            else:
                for pl in range(4):
                    if pl != player_id and not self.bankrupt[pl]:
                        if sum(1 for t in gt if self.property_owner[t] == pl) == len(gt) - 1:
                            for_opp = 1.0
                            break

        max_opp_c = max([self.cash[pl] for pl in range(4) if pl != player_id and not self.bankrupt[pl]] or [0])
        opp_cash_ratio = min(3.0, max_opp_c / max(1.0, price))
        bid_ratio = min(4.0, self.auction_highest_bid / max(1.0, price)) if self.phase == PHASE_AUCTION else 0.0
        trans_cost = self.auction_highest_bid if self.phase == PHASE_AUCTION else price
        post_cash_ratio = min(2.0, max(0.0, self.cash[player_id] - trans_cost) / 1000.0)
        if self.phase == PHASE_AUCTION:
            comp_ratio = len(self.auction_active_bidders) / 4.0
        else:
            comp_ratio = sum(1 for pl in range(4) if pl != player_id and not self.bankrupt[pl] and self.cash[pl] >= price) / 3.0
        tier_score = 1.0 if grp in ("ORANGE", "RED") else (0.8 if grp in ("YELLOW", "DARK_BLUE", "LIGHT_BLUE") else 0.5)
        hazard_score = min(2.0, self.compute_hazard_threat(player_id) / 200.0)

        return np.array(
            [for_me, for_opp, opp_cash_ratio, bid_ratio, post_cash_ratio, comp_ratio, tier_score, hazard_score],
            dtype=np.float32,
        )

    def get_global_state(
        self,
        outflow: Optional[List[float]] = None,
        inflow: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Builds compact 592-dim non-redundant Centralized Critic state S_t^global."""
        if outflow is None or inflow is None:
            outflow, inflow = self._compute_expected_rent_flows()

        gs = np.zeros(GLOBAL_STATE_DIM, dtype=np.float32)
        # 1. Canonical 40 tiles x 9 features = [0..360)
        for tid in range(40):
            base = tid * 9
            own = self.property_owner[tid]
            gs[base + (0 if own == -1 else (own + 1))] = 1.0
            gs[base + 5] = self.property_houses[tid] / 5.0
            gs[base + 6] = 1.0 if self.property_mortgaged[tid] else 0.0
            gs[base + 7] = self._tile_prices_f[tid] / 400.0
            if own != -1 and not self.property_mortgaged[tid]:
                gs[base + 8] = self._calculate_rent(tid, dice_sum=7) / 2000.0

        # 2. 4 players x 12 = [360..408)
        for pl in range(4):
            base = 360 + pl * 12
            nw = float(self.get_player_net_worth(pl)) if not self.bankrupt[pl] else 0.0
            mono_cnt = sum(1 for _, gt, _ in STREET_GROUP_LIST if all(self.property_owner[t] == pl for t in gt))
            gs[base + 0] = self.position[pl] / 40.0
            gs[base + 1] = self.cash[pl] / 2000.0
            gs[base + 2] = 1.0 if self.in_jail[pl] else 0.0
            gs[base + 3] = self.jail_turns[pl] / 3.0
            gs[base + 4] = self.goojf_cards[pl] / 2.0
            gs[base + 5] = 1.0 if self.bankrupt[pl] else 0.0
            gs[base + 6] = nw / 5000.0
            gs[base + 7] = mono_cnt / 8.0
            gs[base + 8] = outflow[pl] / 500.0
            gs[base + 9] = inflow[pl] / 500.0
            gs[base + 10] = 1.0 if self.current_player == pl else 0.0
            gs[base + 11] = 1.0 if self.active_decision_player == pl else 0.0

        # 3. Global phase, bank, auction = [408..432)
        if 0 <= self.phase < NUM_PHASES:
            gs[408 + self.phase] = 1.0
        gs[416] = self.bank_houses / 32.0
        gs[417] = self.bank_hotels / 12.0
        gs[418] = 0.0 if not self.max_rounds else min(2.0, self.round_count / float(self.max_rounds))
        gs[419] = self.last_dice[0] / 6.0
        gs[420] = self.last_dice[1] / 6.0
        gs[421] = 1.0 if self.allow_trade else 0.0
        gs[422] = 1.0 if (self.pending_trade is not None and self.pending_trade.get("is_counter", False)) else 0.0
        gs[423] = 1.0 if self.phase == PHASE_AUCTION else 0.0
        gs[424] = (self.auction_property or 0) / 40.0
        gs[425] = self.auction_highest_bid / 1000.0
        if 0 <= self.auction_highest_bidder < 4:
            gs[426 + self.auction_highest_bidder] = 1.0
        gs[430] = self.starting_player / 4.0
        gs[431] = self.manage_actions_this_turn / 12.0

        # 4. Online behavioral telemetry 4 x 8 = [432..464)
        for pl in range(4):
            base = 432 + pl * 8
            tp_cnt = max(1, self.tel_trade_recv_cnt[pl])
            jo_cnt = max(1, self.tel_jail_opp_cnt[pl])
            ts = max(1, self.turn_step)
            gs[base + 0] = min(3.0, self.tel_bid_ratio_ema[pl])
            gs[base + 1] = self.tel_overbid_ema[pl]
            gs[base + 2] = self.tel_fold_ema[pl]
            gs[base + 3] = min(1.0, self.tel_trade_prop_cnt[pl] / float(ts))
            gs[base + 4] = self.tel_trade_rej_cnt[pl] / float(tp_cnt)
            gs[base + 5] = min(3.0, self.tel_cash_ema[pl] / 1000.0)
            gs[base + 6] = self.tel_jail_stay_cnt[pl] / float(jo_cnt)
            gs[base + 7] = min(1.0, self.tel_build_cnt[pl] / 20.0)

        # 5. Pending trade canonical [464..560): 12 scalars + 42 offer_bundle + 42 req_bundle
        pt = self.pending_trade
        if pt is not None and self.phase in (PHASE_TRADE_RESPONSE, PHASE_TRADE_COUNTER):
            gs[464] = 1.0
            gs[465] = 1.0 if pt.get("is_counter", False) else 0.0
            gs[466 + pt["initiator"]] = 1.0
            gs[470 + pt["target"]] = 1.0
            gs[474] = pt["offer_cash"] / 1000.0
            gs[475] = pt["req_cash"] / 1000.0
            for tid in pt["offer_tiles"]:
                gs[476 + tid] = 1.0
            if pt["offer_goojf"] >= 1:
                gs[476 + GOOJF_SLOT_1] = 1.0
            if pt["offer_goojf"] >= 2:
                gs[476 + GOOJF_SLOT_2] = 1.0
            for tid in pt["req_tiles"]:
                gs[518 + tid] = 1.0
            if pt["req_goojf"] >= 1:
                gs[518 + GOOJF_SLOT_1] = 1.0
            if pt["req_goojf"] >= 2:
                gs[518 + GOOJF_SLOT_2] = 1.0

        # 6. Hidden FIFO Chance & Community Chest card order [560..592)
        gs[560:592] = np.asarray(self.deck_manager.get_state_vector(), dtype=np.float32)
        return gs

    def get_observations(self, skip_obs: bool = False) -> Dict[str, Any]:
        """Returns 3-channel observations, action masks, and compact global state."""
        if skip_obs:
            return {
                "tile_state": None,
                "scalar_obs": None,
                "event_ring": None,
                "action_mask": {f"player_{self.active_decision_player}": self.get_action_masks(self.active_decision_player)},
                "global_state": None,
                "active_player": self.active_decision_player,
                "phase": self.phase,
            }

        outflow, inflow = self._compute_expected_rent_flows()
        tile_dict: Dict[str, np.ndarray] = {}
        scalar_dict: Dict[str, np.ndarray] = {}
        event_dict: Dict[str, np.ndarray] = {}
        masks_dict: Dict[str, Dict[str, np.ndarray]] = {}

        for p in range(4):
            p_name = f"player_{p}"
            t_st, s_obs, e_ring = self.get_actor_structured_obs(p, outflow=outflow, inflow=inflow)
            tile_dict[p_name] = t_st
            scalar_dict[p_name] = s_obs
            event_dict[p_name] = e_ring
            masks_dict[p_name] = self.get_action_masks(p)

        global_state = self.get_global_state(outflow=outflow, inflow=inflow)
        return {
            "tile_state": tile_dict,
            "scalar_obs": scalar_dict,
            "event_ring": event_dict,
            "action_mask": masks_dict,
            "global_state": global_state,
            "active_player": self.active_decision_player,
            "phase": self.phase,
        }

    # ----------------------------------------------------------------------
    # Game Step Transition & Execution
    # ----------------------------------------------------------------------
    def step(
        self, actions: Dict[str, Any], skip_obs: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        p = self.active_decision_player
        p_act = actions.get(f"player_{p}", actions.get(p, actions))

        core_act = int(p_act.get("core", 0))
        auc_act = int(p_act.get("auction", 0))
        prop_act = int(p_act.get("property", ACT_PROP_END_MANAGE))
        trade_target_idx = int(p_act.get("trade_target", 0))
        trade_req_bundle = p_act.get("trade_req_bundle", None)
        trade_offer_bundle = p_act.get("trade_offer_bundle", None)
        trade_offer_cash_idx = int(p_act.get("trade_offer_cash", 0))
        trade_req_cash_idx = int(p_act.get("trade_req_cash", 0))
        trade_response = int(p_act.get("trade_response", 0))

        masks = self.get_action_masks(p)
        env_rewards = [0.0, 0.0, 0.0, 0.0]

        if self.phase == PHASE_ROLL:
            self._handle_roll_phase(p)
        elif self.phase == PHASE_JAIL_DECISION:
            self._handle_jail_phase(p, core_act, masks)
        elif self.phase == PHASE_BUY_OR_AUCTION:
            self._handle_buy_auction_phase(p, core_act)
        elif self.phase == PHASE_AUCTION:
            self._handle_auction_phase(p, auc_act)
        elif self.phase in (PHASE_PROPERTY_MANAGE, PHASE_TRADE_COUNTER):
            self._handle_management_or_counter_phase(
                p, prop_act, masks,
                trade_target_idx, trade_req_bundle, trade_offer_bundle,
                trade_offer_cash_idx, trade_req_cash_idx,
            )
        elif self.phase == PHASE_TRADE_RESPONSE:
            self._handle_trade_response(p, trade_response)

        # Resolve negative cash / bankruptcy
        batch_creditor = self.last_creditor
        for pl in range(4):
            if not self.bankrupt[pl] and self.cash[pl] < 0:
                self._resolve_bankruptcy(pl, creditor_override=batch_creditor)
                if self.bankrupt[pl]:
                    env_rewards[pl] -= 10.0
        self.last_creditor = None

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
                env_rewards[winner] += 10.0
                self._log(f"Game Over! {self.get_player_name(winner)} is CHAMPION!")
            elif len(active_players) > 1:
                net_worths = [self.get_player_net_worth(pl) if not self.bankrupt[pl] else -99999 for pl in range(4)]
                max_nw = max(net_worths[pl] for pl in active_players)
                leaders = [pl for pl in active_players if net_worths[pl] == max_nw]
                if len(leaders) == 1:
                    winner = leaders[0]
                    env_rewards[winner] += 1.0
                    for pl in active_players:
                        if pl != winner:
                            env_rewards[pl] -= 10.0
                else:
                    for pl in active_players:
                        env_rewards[pl] -= 10.0
        else:
            dones["__all__"] = False

        # PBRS v3 shaping
        shaped_rewards = {}
        for pl in range(4):
            new_phi = self._compute_potential(pl)
            pbrs = self.gamma * new_phi - self.prev_potentials[pl]
            shaped_rewards[f"player_{pl}"] = float(env_rewards[pl] + pbrs)
            self.prev_potentials[pl] = new_phi

        observations = self.get_observations(skip_obs=skip_obs)
        infos: Dict[str, Any] = {} if self.fast_search_mode else {
            "event_log": list(self.event_log[-5:]),
            "round": self.round_count,
            "turn_step": self.turn_step,
            "bank_houses": self.bank_houses,
            "bank_hotels": self.bank_hotels,
            "last_dice": self.last_dice,
            "trades_proposed": self.trades_proposed_count,
            "trades_accepted": self.trades_accepted_count,
            "trades_countered": self.trades_countered_count,
            "houses_built": self.houses_built_count,
        }
        return observations, shaped_rewards, dones, infos

    # ----------------------------------------------------------------------
    # Phase Handlers
    # ----------------------------------------------------------------------
    def _handle_roll_phase(self, p: int):
        d1 = self.rng.randint(1, 6)
        d2 = self.rng.randint(1, 6)
        self.last_dice = (d1, d2)
        is_double = d1 == d2
        self.consecutive_doubles = (self.consecutive_doubles + 1) if is_double else 0

        if self.consecutive_doubles >= 3:
            self._log(f"{self.get_player_name(p)} sped (3x doubles) -> JAIL")
            self._send_to_jail(p)
            self._end_turn()
            return

        dice_sum = d1 + d2
        old_pos = self.position[p]
        new_pos = (old_pos + dice_sum) % 40
        self.position[p] = new_pos
        self._push_event(EV_ROLL, p, -1, new_pos, dice_sum)
        if new_pos < old_pos:
            self.cash[p] += self.house_rules.apply_pass_go_salary(p, 200)
        self._handle_landing(p, new_pos, dice_sum, is_double)

    def _handle_jail_phase(self, p: int, core_act: int, masks: Dict[str, np.ndarray]):
        self.tel_jail_opp_cnt[p] += 1
        if core_act == ACT_CORE_PAY_JAIL_50 and masks["core"][ACT_CORE_PAY_JAIL_50]:
            self.cash[p] -= 50
            self.last_creditor = None
            self.in_jail[p] = False
            self.jail_turns[p] = 0
            self._push_event(EV_JAIL_OUT, p, -1, 10, 50)
            self._log(f"{self.get_player_name(p)} paid $50 to leave Jail.")
            self.phase = PHASE_ROLL
            self.active_decision_player = p
            return

        if core_act == ACT_CORE_USE_GOOJF and masks["core"][ACT_CORE_USE_GOOJF]:
            self.goojf_cards[p] -= 1
            src = self.goojf_sources[p].pop()
            self.deck_manager.return_goojf(src)
            self.in_jail[p] = False
            self.jail_turns[p] = 0
            self._push_event(EV_JAIL_OUT, p, -1, GOOJF_SLOT_1, 0)
            self._log(f"{self.get_player_name(p)} used GOOJF.")
            self.phase = PHASE_ROLL
            self.active_decision_player = p
            return

        # Attempt doubles (strategic jail stay tracked for opponent telemetry)
        self.tel_jail_stay_cnt[p] += 1
        d1 = self.rng.randint(1, 6)
        d2 = self.rng.randint(1, 6)
        self.last_dice = (d1, d2)
        if d1 == d2:
            self.in_jail[p] = False
            self.jail_turns[p] = 0
            self.consecutive_doubles = 0
            self._push_event(EV_JAIL_OUT, p, -1, 10, 0)
            new_pos = (self.position[p] + d1 + d2) % 40
            self.position[p] = new_pos
            self._handle_landing(p, new_pos, d1 + d2, is_double=False)
        else:
            self.jail_turns[p] += 1
            if self.jail_turns[p] >= 3:
                self.cash[p] -= 50
                self.last_creditor = None
                self.in_jail[p] = False
                self.jail_turns[p] = 0
                new_pos = (self.position[p] + d1 + d2) % 40
                self.position[p] = new_pos
                self._handle_landing(p, new_pos, d1 + d2, is_double=False)
            else:
                self._end_turn()

    def _handle_landing(self, p: int, pos: int, dice_sum: int, is_double: bool):
        if pos == 30:
            self._send_to_jail(p)
            self._end_turn()
            return

        t_type = TILE_TYPES[pos]
        if t_type == TILE_TAX:
            self.cash[p] -= TILES_DATA[pos]["tax_amount"]
            self.last_creditor = None
            self._proceed_to_manage_or_next(p, is_double)
            return

        if t_type == TILE_CARD:
            deck_name = TILES_DATA[pos]["deck"]
            card = self.deck_manager.draw_chance() if deck_name == "chance" else self.deck_manager.draw_chest()
            self.last_card_drawn = card
            self._execute_card_action(p, card, is_double)
            return

        if t_type in (TILE_STREET, TILE_RAILROAD, TILE_UTILITY):
            owner = self.property_owner[pos]
            if owner == -1:
                self.landed_property = pos
                self.phase = PHASE_BUY_OR_AUCTION
                self.active_decision_player = p
                self.pending_is_double = is_double
                return
            elif owner != p and not self.property_mortgaged[pos]:
                rent = self._calculate_rent(pos, dice_sum)
                self.cash[p] -= rent
                self.cash[owner] += rent
                self.last_creditor = owner
                self._push_event(EV_RENT, p, owner, pos, rent)
                self._log(f"{self.get_player_name(p)} paid ${rent} rent to {self.get_player_name(owner)}.")
                self._proceed_to_manage_or_next(p, is_double)
                return

        self._proceed_to_manage_or_next(p, is_double)

    def _handle_buy_auction_phase(self, p: int, core_act: int):
        tile_id = self.landed_property
        if tile_id is None:
            self._proceed_to_manage_or_next(p, self.pending_is_double)
            return
        price = self.get_property_price(tile_id, 100)
        if core_act == ACT_CORE_BUY_DIRECT and self.cash[p] >= price:
            self.cash[p] -= price
            self.property_owner[tile_id] = p
            self._push_event(EV_BUY, p, -1, tile_id, price)
            self._log(f"{self.get_player_name(p)} bought {TILES_DATA[tile_id]['name']} for ${price}.")
            self.landed_property = None
            self._proceed_to_manage_or_next(p, self.pending_is_double)
        else:
            self._log(f"{self.get_player_name(p)} declined {TILES_DATA[tile_id]['name']} -> AUCTION!")
            self._start_auction(tile_id, self.pending_is_double)

    def _start_auction(self, property_id: int, is_double_pending: bool):
        self.auction_property = property_id
        self.auction_highest_bid = 0
        self.auction_highest_bidder = -1
        start_p = self.current_player
        self.auction_active_bidders = [
            (start_p + off) % 4 for off in range(4) if not self.bankrupt[(start_p + off) % 4]
        ]
        self.auction_current_bidder_idx = 0
        self._auction_has_acted = [False, False, False, False]
        self.pending_auction_double = is_double_pending
        self.phase = PHASE_AUCTION
        self.active_decision_player = self.auction_active_bidders[0]

    def _handle_auction_phase(self, p: int, auc_act: int):
        current_bid = self.auction_highest_bid
        cash = self.cash[p]
        list_price = float(self.get_property_price(self.auction_property, 100))

        bid_inc = 0
        if auc_act == ACT_AUC_BID_1:
            bid_inc = 1
        elif auc_act == ACT_AUC_BID_10:
            bid_inc = 10
        elif auc_act == ACT_AUC_BID_50:
            bid_inc = 50
        elif auc_act == ACT_AUC_BID_100:
            bid_inc = 100
        elif auc_act == ACT_AUC_JUMP_LIST and list_price > current_bid and cash >= list_price:
            bid_inc = int(list_price) - current_bid
        elif auc_act == ACT_AUC_JUMP_SQUEEZE:
            sq = self._get_squeeze_bid_target(p)
            if sq is not None:
                bid_inc = sq - current_bid

        if bid_inc > 0 and cash >= current_bid + bid_inc:
            self.auction_highest_bid = current_bid + bid_inc
            self.auction_highest_bidder = p
            self._auction_has_acted[p] = True
            ratio = self.auction_highest_bid / max(1.0, list_price)
            self.tel_bid_ratio_ema[p] = 0.8 * self.tel_bid_ratio_ema[p] + 0.2 * ratio
            self.tel_overbid_ema[p] = 0.8 * self.tel_overbid_ema[p] + 0.2 * (1.0 if ratio > 1.25 else 0.0)
            self.tel_fold_ema[p] = 0.9 * self.tel_fold_ema[p]
            self._push_event(EV_AUCTION_BID, p, -1, self.auction_property, self.auction_highest_bid)
            self.auction_current_bidder_idx = (self.auction_current_bidder_idx + 1) % len(self.auction_active_bidders)
            self.active_decision_player = self.auction_active_bidders[self.auction_current_bidder_idx]
        else:
            self.tel_fold_ema[p] = 0.85 * self.tel_fold_ema[p] + 0.15
            self.auction_active_bidders.remove(p)
            if len(self.auction_active_bidders) == 0 or (
                len(self.auction_active_bidders) == 1 and self.auction_highest_bidder != -1
            ):
                self._finish_auction()
            else:
                self.auction_current_bidder_idx %= len(self.auction_active_bidders)
                self.active_decision_player = self.auction_active_bidders[self.auction_current_bidder_idx]

    def _finish_auction(self):
        tile_id = self.auction_property
        if self.auction_highest_bidder != -1 and self.auction_highest_bid > 0:
            winner = self.auction_highest_bidder
            price = self.auction_highest_bid
            self.cash[winner] -= price
            self.property_owner[tile_id] = winner
            self._push_event(EV_AUCTION_WIN, winner, -1, tile_id, price)
            self._log(f"AUCTION WON! {self.get_player_name(winner)} won tile {tile_id} for ${price}.")
        self.auction_property = None
        self.auction_highest_bid = 0
        self.auction_highest_bidder = -1
        self.auction_active_bidders = []
        self._proceed_to_manage_or_next(self.current_player, self.pending_auction_double)

    # ----------------------------------------------------------------------
    # Property Management & Multi-Property Bundle Trade + 1x Counter-Offer
    # ----------------------------------------------------------------------
    def _clear_turn_manage_guards(self):
        self.mortgaged_this_turn.clear()
        self.unmortgaged_this_turn.clear()
        self.built_this_turn.clear()
        self.sold_this_turn.clear()

    def _proceed_to_manage_or_next(self, p: int, is_double: bool):
        self.phase = PHASE_PROPERTY_MANAGE
        self.active_decision_player = p
        self.pending_manage_double = is_double
        self.manage_actions_this_turn = 0
        self._clear_turn_manage_guards()

    def _finish_manage_or_next(self, p: int, is_double: bool):
        self.manage_actions_this_turn = 0
        self.trade_proposals_this_turn = 0
        self._clear_turn_manage_guards()
        if is_double and not self.in_jail[p] and not self.bankrupt[p]:
            self.phase = PHASE_ROLL
            self.active_decision_player = p
        else:
            self._end_turn()

    @staticmethod
    def _decode_bundle(vec: Any, mask: np.ndarray) -> Tuple[Tuple[int, ...], int]:
        """Decodes a {0, 1}^42 binary bundle masked strictly by valid items -> (tiles_tuple, goojf_count)."""
        if vec is None:
            return (), 0
        arr = np.asarray(vec, dtype=np.int8)
        if arr.shape != (BUNDLE_DIM,):
            return (), 0
        valid = (arr > 0) & (mask > 0)
        tiles = tuple(int(i) for i in range(40) if valid[i] and TILE_IS_PURCHASABLE[i])
        goojf_cnt = int(valid[GOOJF_SLOT_1]) + int(valid[GOOJF_SLOT_2])
        return tiles, goojf_cnt

    def _handle_management_or_counter_phase(
        self,
        p: int,
        prop_act: int,
        masks: Dict[str, np.ndarray],
        trade_target_idx: int,
        trade_req_bundle: Any,
        trade_offer_bundle: Any,
        trade_offer_cash_idx: int,
        trade_req_cash_idx: int,
    ):
        is_double = self.pending_manage_double

        # Counter-Offer submission step
        if self.phase == PHASE_TRADE_COUNTER:
            pt_prev = self.pending_trade
            orig_init = pt_prev["orig_initiator"] if pt_prev else self.current_player
            target_rel = int(trade_target_idx) % 3
            if not masks["trade_target"][target_rel]:
                # Pick the sole unmasked relative index for orig_initiator
                valid_t = np.where(masks["trade_target"] > 0)[0]
                if len(valid_t) == 0:
                    self.pending_trade = None
                    self._return_to_initiator_manage(orig_init, is_double)
                    return
                target_rel = int(valid_t[0])

            started = self._propose_bundle_trade(
                initiator=p,
                target_rel_idx=target_rel,
                req_bundle=trade_req_bundle,
                offer_bundle=trade_offer_bundle,
                offer_cash_idx=trade_offer_cash_idx,
                req_cash_idx=trade_req_cash_idx,
                masks=masks,
                is_counter=True,
                orig_initiator=orig_init,
            )
            if started:
                self.trades_countered_count += 1
                return
            # Empty/invalid counter cancels counter cleanly back to original initiator
            self.pending_trade = None
            self._push_event(EV_TRADE_REJECT, p, orig_init, 0, 0)
            self._return_to_initiator_manage(orig_init, is_double)
            return

        # Physical management actions [0..159]
        self.manage_actions_this_turn += 1
        if 0 <= prop_act < 160 and bool(masks["property"][prop_act]):
            self._execute_property_manage_op(p, prop_act)
            if self.manage_actions_this_turn >= 12:
                self._finish_manage_or_next(p, is_double)
            return

        # Trade initiation (explicit prop_act == ACT_PROP_PROPOSE_TRADE == 161)
        if (
            prop_act == ACT_PROP_PROPOSE_TRADE
            and bool(masks["property"][ACT_PROP_PROPOSE_TRADE])
            and self.allow_trade
            and self.trade_proposals_this_turn == 0
        ):
            started = self._propose_bundle_trade(
                initiator=p,
                target_rel_idx=int(trade_target_idx) % 3,
                req_bundle=trade_req_bundle,
                offer_bundle=trade_offer_bundle,
                offer_cash_idx=trade_offer_cash_idx,
                req_cash_idx=trade_req_cash_idx,
                masks=masks,
                is_counter=False,
                orig_initiator=p,
            )
            if started:
                self.trade_proposals_this_turn = 1
                self.trades_proposed_count += 1
                return

        # End management phase
        self._finish_manage_or_next(p, is_double)

    def _execute_property_manage_op(self, p: int, prop_act: int):
        if 0 <= prop_act < 40:
            h_cost = self.get_property_house_cost(prop_act, 0)
            h = self.property_houses[prop_act]
            if h < 4:
                self.property_houses[prop_act] += 1
                self.bank_houses -= 1
                self.cash[p] -= h_cost
                self.houses_built_count += 1
                self.tel_build_cnt[p] += 1
                self.built_this_turn.add(prop_act)
                self._push_event(EV_BUILD, p, -1, prop_act, h_cost)
            elif h == 4:
                self.property_houses[prop_act] = 5
                self.bank_hotels -= 1
                self.bank_houses += 4
                self.cash[p] -= h_cost
                self.houses_built_count += 1
                self.tel_build_cnt[p] += 1
                self.built_this_turn.add(prop_act)
                self._push_event(EV_BUILD, p, -1, prop_act, h_cost)
        elif 40 <= prop_act < 80:
            p_idx = prop_act - 40
            h_cost = self.get_property_house_cost(p_idx, 0)
            refund = int(h_cost * HOUSE_REFUND_RATE)
            if self.property_houses[p_idx] <= 4:
                self.property_houses[p_idx] -= 1
                self.bank_houses += 1
            else:
                self.property_houses[p_idx] = 4
                self.bank_hotels += 1
                self.bank_houses -= 4
            self.cash[p] += refund
            self.sold_this_turn.add(p_idx)
            self._push_event(EV_SELL, p, -1, p_idx, refund)
        elif 80 <= prop_act < 120:
            p_idx = prop_act - 80
            mtg_val = self.get_property_mortgage(p_idx, 0)
            self.property_mortgaged[p_idx] = True
            self.cash[p] += mtg_val
            self.mortgaged_this_turn.add(p_idx)
            self.tel_mortgage_cnt[p] += 1
            self._push_event(EV_MORTGAGE, p, -1, p_idx, mtg_val)
        elif 120 <= prop_act < 160:
            p_idx = prop_act - 120
            cost = int(self.get_property_mortgage(p_idx, 0) * (1.0 + UNMORTGAGE_INTEREST_RATE))
            self.property_mortgaged[p_idx] = False
            self.cash[p] -= cost
            self.unmortgaged_this_turn.add(p_idx)
            self._push_event(EV_UNMORTGAGE, p, -1, p_idx, cost)

    def _propose_bundle_trade(
        self,
        initiator: int,
        target_rel_idx: int,
        req_bundle: Any,
        offer_bundle: Any,
        offer_cash_idx: int,
        req_cash_idx: int,
        masks: Dict[str, np.ndarray],
        is_counter: bool,
        orig_initiator: int,
    ) -> bool:
        target_rel_idx = int(target_rel_idx) % 3
        if not masks["trade_target"][target_rel_idx]:
            return False
        target = (initiator + 1 + target_rel_idx) % 4
        if self.bankrupt[target] or target == initiator:
            return False

        req_tiles, req_goojf = self._decode_bundle(req_bundle, masks["trade_req_bundle"][target_rel_idx])
        offer_tiles, offer_goojf = self._decode_bundle(offer_bundle, masks["trade_offer_bundle"])

        # Same tile cannot appear in both sides (disjoint owners normally guarantees this)
        if set(req_tiles) & set(offer_tiles):
            return False

        offer_cash = TRADE_CASH_BRACKETS[max(0, min(offer_cash_idx, TRADE_CASH_DIM - 1))]
        req_cash = TRADE_CASH_BRACKETS[max(0, min(req_cash_idx, TRADE_CASH_DIM - 1))]

        # Eliminate dual-cash paradox if present
        if offer_cash > 0 and req_cash > 0:
            if offer_cash >= req_cash:
                offer_cash -= req_cash
                req_cash = 0
            else:
                req_cash -= offer_cash
                offer_cash = 0

        has_offer_side = len(offer_tiles) > 0 or offer_goojf > 0 or offer_cash > 0
        has_req_side = len(req_tiles) > 0 or req_goojf > 0 or req_cash > 0
        has_any_asset = len(offer_tiles) > 0 or offer_goojf > 0 or len(req_tiles) > 0 or req_goojf > 0

        # Reject empty-beg / 1-way giveaway / cash-for-cash no-ops cleanly
        if not has_any_asset or not has_offer_side or not has_req_side:
            return False
        if self.cash[initiator] < offer_cash:
            return False

        self.pending_trade = {
            "initiator": initiator,
            "target": target,
            "orig_initiator": orig_initiator,
            "offer_tiles": offer_tiles,
            "req_tiles": req_tiles,
            "offer_goojf": offer_goojf,
            "req_goojf": req_goojf,
            "offer_cash": offer_cash,
            "req_cash": req_cash,
            "is_counter": bool(is_counter),
        }
        self.tel_trade_prop_cnt[initiator] += 1
        self.tel_trade_recv_cnt[target] += 1
        first_tile = req_tiles[0] if req_tiles else (offer_tiles[0] if offer_tiles else GOOJF_SLOT_1)
        self._push_event(EV_TRADE_PROPOSE, initiator, target, first_tile, offer_cash + req_cash)
        self.phase = PHASE_TRADE_RESPONSE
        self.active_decision_player = target
        self._log(
            f"{'COUNTER ' if is_counter else ''}TRADE: {self.get_player_name(initiator)} -> {self.get_player_name(target)} "
            f"(Offer={offer_tiles}+G{offer_goojf}+${offer_cash} vs Req={req_tiles}+G{req_goojf}+${req_cash})"
        )
        return True

    def _return_to_initiator_manage(self, orig_init: int, is_double: bool):
        if not self.bankrupt[orig_init] and self.manage_actions_this_turn < 12:
            self.phase = PHASE_PROPERTY_MANAGE
            self.active_decision_player = orig_init
        else:
            self._finish_manage_or_next(orig_init, is_double)

    def _handle_trade_response(self, responder: int, trade_response: int):
        trade = self.pending_trade
        if trade is None:
            self._return_to_initiator_manage(self.current_player, self.pending_manage_double)
            return

        init = trade["initiator"]
        orig_init = trade.get("orig_initiator", init)
        is_double = self.pending_manage_double

        # Counter-offer requested (only allowed once, when is_counter == False)
        if trade_response == 2 and not trade.get("is_counter", False) and not self.bankrupt[orig_init]:
            self.phase = PHASE_TRADE_COUNTER
            self.active_decision_player = responder
            self._log(f"{self.get_player_name(responder)} chooses to COUNTER-OFFER {self.get_player_name(orig_init)}.")
            return

        if trade_response == 1:
            req_cash = trade["req_cash"]
            offer_cash = trade["offer_cash"]
            if self.cash[responder] >= req_cash and self.cash[init] >= offer_cash:
                self.cash[init] -= offer_cash
                self.cash[responder] += offer_cash
                self.cash[responder] -= req_cash
                self.cash[init] += req_cash

                for tid in trade["offer_tiles"]:
                    if 0 <= tid < 40 and self.property_owner[tid] == init:
                        self.property_owner[tid] = responder
                for tid in trade["req_tiles"]:
                    if 0 <= tid < 40 and self.property_owner[tid] == responder:
                        self.property_owner[tid] = init

                for _ in range(min(trade["offer_goojf"], self.goojf_cards[init])):
                    self.goojf_cards[init] -= 1
                    src = self.goojf_sources[init].pop()
                    self.goojf_cards[responder] += 1
                    self.goojf_sources[responder].append(src)
                for _ in range(min(trade["req_goojf"], self.goojf_cards[responder])):
                    self.goojf_cards[responder] -= 1
                    src = self.goojf_sources[responder].pop()
                    self.goojf_cards[init] += 1
                    self.goojf_sources[init].append(src)

                self.trades_accepted_count += 1
                self._push_event(EV_TRADE_ACCEPT, init, responder, 0, offer_cash + req_cash)
                self._log(f"TRADE ACCEPTED ({self.get_player_name(init)} <-> {self.get_player_name(responder)}).")
        else:
            self.tel_trade_rej_cnt[responder] += 1
            self._push_event(EV_TRADE_REJECT, responder, init, 0, trade["offer_cash"] + trade["req_cash"])
            self._log(f"TRADE REJECTED by {self.get_player_name(responder)}.")

        self.pending_trade = None
        self._return_to_initiator_manage(orig_init, is_double)

    # ----------------------------------------------------------------------
    # Cards, Jail, Bankruptcy & Turn Rotation
    # ----------------------------------------------------------------------
    def _execute_card_action(self, p: int, card: Dict[str, Any], is_double: bool):
        action = card["action"]
        if action == "advance_to":
            target = card["target"]
            old_pos = self.position[p]
            self.position[p] = target
            if card.get("collect_pass_go", False) and target < old_pos:
                self.cash[p] += self.house_rules.apply_pass_go_salary(p, 200)
            self._handle_landing(p, target, dice_sum=self.last_dice[0] + self.last_dice[1], is_double=is_double)
        elif action == "advance_nearest_utility":
            curr = self.position[p]
            target = 12 if curr < 12 or curr >= 28 else 28
            if target < curr:
                self.cash[p] += self.house_rules.apply_pass_go_salary(p, 200)
            self.position[p] = target
            owner = self.property_owner[target]
            if owner == -1:
                self.landed_property = target
                self.phase = PHASE_BUY_OR_AUCTION
                self.active_decision_player = p
                self.pending_is_double = is_double
            elif owner != p and not self.property_mortgaged[target]:
                rent = 10 * (self.rng.randint(1, 6) + self.rng.randint(1, 6))
                self.cash[p] -= rent
                self.cash[owner] += rent
                self.last_creditor = owner
                self._proceed_to_manage_or_next(p, is_double)
            else:
                self._proceed_to_manage_or_next(p, is_double)
        elif action == "advance_nearest_railroad":
            curr = self.position[p]
            target = min([r for r in (5, 15, 25, 35) if r > curr] or [5])
            if target < curr:
                self.cash[p] += self.house_rules.apply_pass_go_salary(p, 200)
            self.position[p] = target
            owner = self.property_owner[target]
            if owner == -1:
                self.landed_property = target
                self.phase = PHASE_BUY_OR_AUCTION
                self.active_decision_player = p
                self.pending_is_double = is_double
            elif owner != p and not self.property_mortgaged[target]:
                rent = self._calculate_rent(target, dice_sum=self.last_dice[0] + self.last_dice[1], multiplier_override=2)
                self.cash[p] -= rent
                self.cash[owner] += rent
                self.last_creditor = owner
                self._proceed_to_manage_or_next(p, is_double)
            else:
                self._proceed_to_manage_or_next(p, is_double)
        elif action == "cash_from_bank":
            self.cash[p] += card["amount"]
            self._proceed_to_manage_or_next(p, is_double)
        elif action == "pay_bank":
            self.cash[p] -= card["amount"]
            self.last_creditor = None
            self._proceed_to_manage_or_next(p, is_double)
        elif action == "goojf":
            self.goojf_cards[p] += 1
            self.goojf_sources[p].append(card["deck"])
            self._proceed_to_manage_or_next(p, is_double)
        elif action == "go_to_jail":
            self._send_to_jail(p)
            self._end_turn()
        elif action == "move_relative":
            new_pos = (self.position[p] + card["steps"]) % 40
            self.position[p] = new_pos
            self._handle_landing(p, new_pos, dice_sum=self.last_dice[0] + self.last_dice[1], is_double=is_double)
        elif action == "property_repairs":
            owned = [t for t in PURCHASABLE_TILES_TUPLE if self.property_owner[t] == p]
            h_cnt = sum(self.property_houses[t] for t in owned if 1 <= self.property_houses[t] <= 4)
            ht_cnt = sum(1 for t in owned if self.property_houses[t] == 5)
            self.cash[p] -= h_cnt * card["house_cost"] + ht_cnt * card["hotel_cost"]
            self.last_creditor = None
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

    def _send_to_jail(self, p: int):
        self.position[p] = 10
        self.in_jail[p] = True
        self.jail_turns[p] = 0
        self.consecutive_doubles = 0
        self._push_event(EV_JAIL_IN, p, -1, 10, 0)

    def _resolve_bankruptcy(self, p: int, creditor_override: Optional[int] = None):
        owned_props = [t for t in PURCHASABLE_TILES_TUPLE if self.property_owner[t] == p]
        creditor = creditor_override if creditor_override is not None else self.last_creditor
        if creditor == p or (creditor is not None and self.bankrupt[creditor]):
            creditor = None

        while self.cash[p] < 0:
            sellable = [t for t in owned_props if self.property_houses[t] > 0]
            if not sellable:
                break
            target_t = max(sellable, key=lambda t: self.property_houses[t])
            grp = TILE_COLORS[target_t]
            group_tiles = COLOR_GROUPS[grp] if (grp and grp in COLOR_GROUPS) else [target_t]
            max_h = max(self.property_houses[t] for t in group_tiles)
            t_sell = next(t for t in group_tiles if self.property_houses[t] == max_h)
            cost = self.get_property_house_cost(t_sell, 0)
            refund = int(cost * HOUSE_REFUND_RATE)
            if max_h <= 4:
                self.property_houses[t_sell] -= 1
                self.bank_houses += 1
                self.cash[p] += refund
            elif self.bank_houses >= 4:
                self.property_houses[t_sell] = 4
                self.bank_hotels += 1
                self.bank_houses -= 4
                self.cash[p] += refund
            else:
                cur_h = sum(self.property_houses[t] for t in group_tiles if self.property_houses[t] <= 4)
                cur_ht = sum(1 for t in group_tiles if self.property_houses[t] == 5)
                tot_before = sum(self.property_houses[t] for t in group_tiles)
                n_grp = len(group_tiles)
                target_h = next((c for c in (3, 2, 1, 0) if n_grp * c <= self.bank_houses + cur_h), 0)
                self.bank_hotels += cur_ht
                self.bank_houses += cur_h - (n_grp * target_h)
                for t_g in group_tiles:
                    self.property_houses[t_g] = target_h
                self.cash[p] += (tot_before - n_grp * target_h) * refund

        for t in owned_props:
            if not self.property_mortgaged[t] and self.cash[p] < 0:
                self.property_mortgaged[t] = True
                self.cash[p] += self.get_property_mortgage(t, 0)

        if self.cash[p] < 0:
            self.bankrupt[p] = True
            if p not in self.elimination_order:
                self.elimination_order.append(p)
            self._push_event(EV_BANKRUPT, p, creditor if creditor is not None else -1, 0, 0)

            if creditor is not None:
                for t in owned_props:
                    self.property_owner[t] = creditor
                while self.goojf_cards[p] > 0:
                    self.goojf_cards[p] -= 1
                    src = self.goojf_sources[p].pop()
                    self.goojf_cards[creditor] += 1
                    self.goojf_sources[creditor].append(src)
            else:
                for t in owned_props:
                    h = self.property_houses[t]
                    if h == 5:
                        self.bank_hotels += 1
                    elif h > 0:
                        self.bank_houses += h
                    self.property_houses[t] = 0
                    self.property_mortgaged[t] = False
                    self.property_owner[t] = -1
                while self.goojf_cards[p] > 0:
                    self.goojf_cards[p] -= 1
                    self.deck_manager.return_goojf(self.goojf_sources[p].pop())

            if self.pending_trade and (self.pending_trade["initiator"] == p or self.pending_trade["target"] == p):
                self.pending_trade = None
            if p in self.auction_active_bidders:
                self.auction_active_bidders.remove(p)
                if self.auction_highest_bidder == p:
                    self.auction_highest_bidder = -1
                    self.auction_highest_bid = 0

            if self.current_player == p:
                self.pending_manage_double = False
                self.pending_auction_double = False
                self.pending_is_double = False
                self._end_turn()
            elif self.active_decision_player == p:
                self.active_decision_player = self.current_player

        self.last_creditor = None

    def _end_turn(self):
        self.consecutive_doubles = 0
        self.landed_property = None
        self.last_creditor = None
        self.turn_step += 1
        self.manage_actions_this_turn = 0
        self.trade_proposals_this_turn = 0
        self._clear_turn_manage_guards()

        # Track cash reserve EMA once per turn
        for pl in range(4):
            if not self.bankrupt[pl]:
                self.tel_cash_ema[pl] = 0.92 * self.tel_cash_ema[pl] + 0.08 * float(max(0, self.cash[pl]))

        if all(self.bankrupt):
            self.phase = PHASE_GAME_OVER
            return

        prev_player = self.current_player
        for _ in range(4):
            self.current_player = (self.current_player + 1) % 4
            if not self.bankrupt[self.current_player]:
                break

        if ((self.current_player - self.starting_player) % 4) <= ((prev_player - self.starting_player) % 4):
            self.round_count += 1

        self.phase = PHASE_JAIL_DECISION if self.in_jail[self.current_player] else PHASE_ROLL
        self.active_decision_player = self.current_player
