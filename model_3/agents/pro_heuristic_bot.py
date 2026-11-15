"""External tactical Monopoly sparring bots (strictly external to RL action masks).

- ProHeuristicBot: Tournament-pro strategy featuring cash reserve buffers,
  strategic late-game jail stalling, 4-house shortage locks, fair bundle trade
  negotiation, and bounded 1x counter-offers.
- OverbidderBot: Aggressive Model-1-style auction overbidder & hotel-rusher for
  PSRO opponent diversity.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from model_2.engine.board_data import (
    TILES_DATA,
    COLOR_GROUPS,
    TILE_TO_GROUP,
    PURCHASABLE_TILES,
    TILE_STREET,
    TILE_RAILROAD,
    TILE_UTILITY,
    UNMORTGAGE_INTEREST_RATE,
    COLOR_BROWN,
    COLOR_LIGHT_BLUE,
    COLOR_PINK,
    COLOR_ORANGE,
    COLOR_RED,
    COLOR_YELLOW,
    COLOR_GREEN,
    COLOR_DARK_BLUE,
)
from model_2.engine.monopoly_core import (
    ACT_CORE_ROLL_PASS,
    ACT_CORE_BUY_DIRECT,
    ACT_CORE_DECLINE_BUY,
    ACT_CORE_PAY_JAIL_50,
    ACT_CORE_USE_GOOJF,
    ACT_AUC_FOLD,
    ACT_AUC_BID_1,
    ACT_AUC_BID_10,
    ACT_AUC_BID_50,
    ACT_AUC_BID_100,
    ACT_AUC_JUMP_LIST,
    ACT_AUC_JUMP_SQUEEZE,
)
from model_3.engine.monopoly_core import (
    MonopolyCore,
    PHASE_ROLL,
    PHASE_BUY_OR_AUCTION,
    PHASE_AUCTION,
    PHASE_JAIL_DECISION,
    PHASE_PROPERTY_MANAGE,
    PHASE_TRADE_RESPONSE,
    PHASE_TRADE_COUNTER,
    ACT_PROP_END_MANAGE,
    ACT_PROP_PROPOSE_TRADE,
    TRADE_CASH_BRACKETS,
    TRADE_CASH_DIM,
    BUNDLE_DIM,
    GOOJF_SLOT_1,
    GOOJF_SLOT_2,
    ACT_SHAPES,
)

# Color group strategic priority weights (Orange/Red > LightBlue/Pink > Yellow/Green > Brown/DarkBlue)
GROUP_VALUE_WEIGHT: Dict[str, float] = {
    COLOR_ORANGE: 1.35,
    COLOR_RED: 1.30,
    COLOR_LIGHT_BLUE: 1.18,
    COLOR_PINK: 1.15,
    COLOR_YELLOW: 1.08,
    COLOR_GREEN: 0.95,
    COLOR_DARK_BLUE: 0.95,
    COLOR_BROWN: 0.88,
}


def make_noop_action_v3() -> Dict[str, Any]:
    """Returns a zero-initialized valid Model-3 action dictionary matching ACT_SHAPES."""
    return {
        k: (np.zeros(shp, dtype=np.int8) if len(shp) > 0 else 0)
        for k, shp in ACT_SHAPES.items()
    }


def _nearest_cash_bracket(amount: float, max_cash: int) -> int:
    """Returns index in TRADE_CASH_BRACKETS closest to `amount` and <= `max_cash`."""
    target = max(0.0, min(float(amount), float(max_cash)))
    best_idx = 0
    best_diff = 1e12
    for idx, val in enumerate(TRADE_CASH_BRACKETS):
        if val > max_cash:
            break
        diff = abs(float(val) - target)
        if diff < best_diff:
            best_diff = diff
            best_idx = idx
    return best_idx


class ProHeuristicBot:
    """Tournament-pro heuristic agent reading public board state and Model-3 masks."""

    def __init__(self, aggression: float = 1.0, seed: int = 0):
        self.aggression = float(aggression)
        self.rng = np.random.RandomState(seed)
        self._pending_counter: Optional[Dict[str, Any]] = None

    def reset(self):
        self._pending_counter = None

    @staticmethod
    def _max_board_rent(core: MonopolyCore, player_id: int) -> int:
        max_rent = 0
        for t in PURCHASABLE_TILES:
            owner = core.property_owner[t]
            if owner != -1 and owner != player_id and not core.property_mortgaged[t]:
                r = core._calculate_rent(t, dice_sum=7)
                if r > max_rent:
                    max_rent = r
        return max_rent

    def _desired_reserve(self, core: MonopolyCore, player_id: int) -> int:
        max_rent = self._max_board_rent(core, player_id)
        base = 120 if core.round_count < 12 else 200
        return int(min(900, max(base, 0.75 * max_rent + 60)))

    @staticmethod
    def _count_unowned_streets(core: MonopolyCore) -> int:
        return sum(
            1
            for t in PURCHASABLE_TILES
            if core.property_owner[t] == -1 and TILES_DATA[t]["type"] == TILE_STREET
        )

    @staticmethod
    def _owns_monopoly(core: MonopolyCore, player_id: int, grp: str) -> bool:
        return all(core.property_owner[t] == player_id for t in COLOR_GROUPS[grp])

    def _tile_strategic_value(self, core: MonopolyCore, tile_id: int, player_id: int) -> float:
        td = TILES_DATA[tile_id]
        base_price = float(core.get_property_price(tile_id, td["price"]))
        if td["type"] == TILE_RAILROAD:
            owned_rr = sum(
                1 for t in PURCHASABLE_TILES
                if TILES_DATA[t]["type"] == TILE_RAILROAD and core.property_owner[t] == player_id
            )
            return base_price * (1.05 + 0.25 * owned_rr) * self.aggression
        if td["type"] == TILE_UTILITY:
            owned_u = sum(
                1 for t in PURCHASABLE_TILES
                if TILES_DATA[t]["type"] == TILE_UTILITY and core.property_owner[t] == player_id
            )
            return base_price * (0.75 + 0.35 * owned_u)

        grp = TILE_TO_GROUP.get(tile_id, "")
        grp_tiles = COLOR_GROUPS.get(grp, [tile_id])
        n_grp = len(grp_tiles)
        own_cnt = sum(1 for t in grp_tiles if core.property_owner[t] == player_id)
        will_complete = (own_cnt + (1 if core.property_owner[tile_id] != player_id else 0)) == n_grp

        # Check if any opponent sits 1 away from completing this color
        opp_blocking = False
        for opp in range(4):
            if opp == player_id or core.bankrupt[opp]:
                continue
            opp_cnt = sum(1 for t in grp_tiles if core.property_owner[t] == opp)
            if opp_cnt == n_grp - 1 and core.property_owner[tile_id] != opp:
                opp_blocking = True
                break

        gw = GROUP_VALUE_WEIGHT.get(grp, 1.0)
        mult = 1.05 * gw
        if will_complete:
            mult = 2.45 * gw
        elif opp_blocking:
            mult = 1.70 * gw
        elif own_cnt > 0:
            mult = 1.25 * gw
        if core.property_mortgaged[tile_id]:
            mult *= 0.72
        return base_price * mult * self.aggression

    def _should_hold_four_house_lock(self, core: MonopolyCore, player_id: int) -> bool:
        """Returns True if opponents hold potential monopolies and upgrading 4->hotel weakens shortage pressure."""
        if core.bank_houses <= 16:
            return True
        for opp in range(4):
            if opp == player_id or core.bankrupt[opp]:
                continue
            for grp, grp_tiles in COLOR_GROUPS.items():
                opp_cnt = sum(1 for t in grp_tiles if core.property_owner[t] == opp)
                if opp_cnt == len(grp_tiles):
                    min_h = min(core.property_houses[t] for t in grp_tiles)
                    if min_h < 4:
                        return True
        return False

    def act(self, core: MonopolyCore, player_id: int) -> Dict[str, Any]:
        """Selects a legal Model-3 action dict given `core` and `player_id`."""
        masks = core.get_action_masks(player_id)
        out = make_noop_action_v3()
        phase = core.phase

        if phase == PHASE_ROLL:
            out["core"] = ACT_CORE_ROLL_PASS
            return out

        if phase == PHASE_JAIL_DECISION:
            out["core"] = self._decide_jail(core, player_id, masks["core"])
            return out

        if phase == PHASE_BUY_OR_AUCTION:
            out["core"] = self._decide_buy(core, player_id, masks["core"])
            return out

        if phase == PHASE_AUCTION:
            out["auction"] = self._decide_auction(core, player_id, masks["auction"])
            return out

        if phase == PHASE_TRADE_RESPONSE:
            out["trade_response"] = self._decide_trade_response(core, player_id, masks["trade_response"])
            return out

        if phase == PHASE_TRADE_COUNTER:
            return self._decide_trade_counter(core, player_id, masks, out)

        if phase == PHASE_PROPERTY_MANAGE:
            return self._decide_property_manage(core, player_id, masks, out)

        return out

    def _decide_jail(self, core: MonopolyCore, player_id: int, mask_core: np.ndarray) -> int:
        unowned = self._count_unowned_streets(core)
        max_rent = self._max_board_rent(core, player_id)
        # Early/mid game when unowned streets remain: exit Jail quickly to claim equity
        if unowned >= 3 or (core.round_count < 14 and max_rent < 350):
            if mask_core[ACT_CORE_USE_GOOJF]:
                return ACT_CORE_USE_GOOJF
            if mask_core[ACT_CORE_PAY_JAIL_50] and core.cash[player_id] >= 200:
                return ACT_CORE_PAY_JAIL_50
        # Late game with dangerous developed boards: stall in Jail!
        return ACT_CORE_ROLL_PASS

    def _decide_buy(self, core: MonopolyCore, player_id: int, mask_core: np.ndarray) -> int:
        if not mask_core[ACT_CORE_BUY_DIRECT]:
            return ACT_CORE_DECLINE_BUY
        tile_id = core.landed_property
        if tile_id is None:
            return ACT_CORE_DECLINE_BUY
        price = core.get_property_price(tile_id, TILES_DATA[tile_id]["price"])
        cash = core.cash[player_id]
        val = self._tile_strategic_value(core, tile_id, player_id)
        reserve = self._desired_reserve(core, player_id)
        # Buy if completes monopoly / blocks opponent or affordable above reserve (or cheap street)
        if val >= price * 1.4 or (cash - price >= reserve * 0.45) or (cash >= price + 60 and val >= price * 0.95):
            return ACT_CORE_BUY_DIRECT
        return ACT_CORE_DECLINE_BUY

    def _decide_auction(self, core: MonopolyCore, player_id: int, mask_auc: np.ndarray) -> int:
        tile_id = core.auction_property
        if tile_id is None:
            return ACT_AUC_FOLD
        if core.auction_highest_bidder == player_id:
            return ACT_AUC_FOLD
        val = self._tile_strategic_value(core, tile_id, player_id)
        reserve = self._desired_reserve(core, player_id)
        cash = core.cash[player_id]
        # Allow dipping into reserve for monopoly-completers or key blocks
        effective_budget = cash if val >= 1.5 * core.get_property_price(tile_id, 100) else max(0, cash - int(0.4 * reserve))
        max_bid = min(int(val), effective_budget)
        cur = core.auction_highest_bid
        if cur >= max_bid:
            return ACT_AUC_FOLD

        price = core.get_property_price(tile_id, TILES_DATA[tile_id]["price"])
        squeeze = core._get_squeeze_bid_target(player_id)
        candidates: List[Tuple[int, int]] = []
        inc_map = {
            ACT_AUC_BID_1: cur + 1,
            ACT_AUC_BID_10: cur + 10,
            ACT_AUC_BID_50: cur + 50,
            ACT_AUC_BID_100: cur + 100,
        }
        if mask_auc[ACT_AUC_JUMP_LIST]:
            inc_map[ACT_AUC_JUMP_LIST] = price
        if mask_auc[ACT_AUC_JUMP_SQUEEZE] and squeeze is not None:
            inc_map[ACT_AUC_JUMP_SQUEEZE] = squeeze

        for act_idx, resulting_bid in inc_map.items():
            if mask_auc[act_idx] and resulting_bid <= max_bid:
                candidates.append((act_idx, resulting_bid))

        if not candidates:
            return ACT_AUC_FOLD
        # Prefer JUMP_LIST or BID_50/10 when well below max_bid, or small increment near ceiling
        if max_bid - cur >= 90 and mask_auc[ACT_AUC_BID_50] and cur + 50 <= max_bid:
            return ACT_AUC_BID_50
        if max_bid - cur >= 25 and mask_auc[ACT_AUC_BID_10] and cur + 10 <= max_bid:
            return ACT_AUC_BID_10
        # Otherwise pick smallest legal increment
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    def _score_bundle_for_player(
        self,
        core: MonopolyCore,
        tiles: Tuple[int, ...],
        goojf: int,
        cash_delta: int,
        player_id: int,
        receiving: bool,
    ) -> float:
        """Estimates strategic equity gain/loss of a bundle for `player_id`."""
        total = float(cash_delta) + float(goojf) * 38.0 * (1.0 if receiving else -1.0)
        for t in tiles:
            # Evaluate base value and monopoly shift
            base = float(core.get_property_price(t, TILES_DATA[t]["price"]))
            if core.property_mortgaged[t]:
                base *= 0.65
            grp = TILE_TO_GROUP.get(t, "")
            mult = 1.05
            if TILES_DATA[t]["type"] == TILE_STREET and grp:
                grp_tiles = COLOR_GROUPS[grp]
                n_grp = len(grp_tiles)
                gw = GROUP_VALUE_WEIGHT.get(grp, 1.0)
                cur_own = sum(1 for gt in grp_tiles if core.property_owner[gt] == player_id)
                gained = sum(1 for gt in tiles if TILE_TO_GROUP.get(gt, "") == grp)
                if receiving:
                    after = cur_own + gained
                    mult = (2.35 * gw) if after == n_grp else (1.25 * gw if after >= 2 else 1.0 * gw)
                else:
                    after = max(0, cur_own - gained)
                    if cur_own == n_grp and after < n_grp:
                        mult = 2.60 * gw
                    else:
                        # Giving opponent a monopoly hurts more than losing an isolated card!
                        mult = 1.25 * gw
            elif TILES_DATA[t]["type"] == TILE_RAILROAD:
                mult = 1.25
            else:
                mult = 0.80
            signed_v = base * mult * (1.0 if receiving else -1.0)
            total += signed_v
        return total

    def _decide_trade_response(self, core: MonopolyCore, player_id: int, mask_resp: np.ndarray) -> int:
        pt = core.pending_trade
        if pt is None:
            return 0
        # Incoming trade perspective for target `player_id`:
        # receives: pt["offer_tiles"], pt["offer_goojf"], pt["offer_cash"]
        # gives:    pt["req_tiles"], pt["req_goojf"], pt["req_cash"]
        in_val = self._score_bundle_for_player(
            core, pt["offer_tiles"], pt["offer_goojf"], pt["offer_cash"], player_id, receiving=True
        )
        out_val = self._score_bundle_for_player(
            core, pt["req_tiles"], pt["req_goojf"], -pt["req_cash"], player_id, receiving=False
        )
        net_self = in_val + out_val

        # Also check strategic opponent empowerment penalty
        init = pt["initiator"]
        opp_gain = self._score_bundle_for_player(
            core, pt["req_tiles"], pt["req_goojf"], pt["req_cash"], init, receiving=True
        ) - self._score_bundle_for_player(
            core, pt["offer_tiles"], pt["offer_goojf"], -pt["offer_cash"], init, receiving=False
        )

        # Completes monopoly cleanly and leaves safe reserve?
        reserve = self._desired_reserve(core, player_id)
        cash_after = core.cash[player_id] + pt["offer_cash"] - pt["req_cash"]

        # Reject giving away an Orange/Red monopoly unless receiving equal/stronger monopoly + cash
        if mask_resp[1]:
            if net_self >= 40.0 and cash_after >= max(60, int(0.35 * reserve)):
                if opp_gain <= net_self * 1.45 or net_self >= 180.0:
                    return 1

        # Consider 1x Counter-Offer if close enough (-300 < net_self < 60)
        if mask_resp[2] and net_self > -320.0:
            ask_extra = max(25.0, (40.0 - net_self) + 0.25 * max(0.0, opp_gain - net_self))
            new_offer_cash_for_init = max(0, pt["req_cash"] - int(ask_extra))
            needed_cash_from_init = pt["offer_cash"] + max(0, int(ask_extra) - pt["req_cash"])
            if core.cash[init] >= needed_cash_from_init and (needed_cash_from_init > 0 or new_offer_cash_for_init >= 0):
                self._pending_counter = {
                    "target_abs": pt["orig_initiator"],
                    "req_tiles": pt["offer_tiles"],
                    "req_goojf": pt["offer_goojf"],
                    "offer_tiles": pt["req_tiles"],
                    "offer_goojf": pt["req_goojf"],
                    "req_cash": needed_cash_from_init,
                    "offer_cash": new_offer_cash_for_init,
                }
                return 2

        return 0

    def _decide_trade_counter(
        self, core: MonopolyCore, player_id: int, masks: Dict[str, np.ndarray], out: Dict[str, Any]
    ) -> Dict[str, Any]:
        pc = self._pending_counter
        self._pending_counter = None
        valid_targets = np.where(masks["trade_target"] > 0)[0]
        if pc is None or len(valid_targets) == 0:
            out["property"] = ACT_PROP_END_MANAGE
            return out

        t_rel = int(valid_targets[0])
        out["trade_target"] = t_rel
        req_b = np.zeros(BUNDLE_DIM, dtype=np.int8)
        off_b = np.zeros(BUNDLE_DIM, dtype=np.int8)
        for t in pc["req_tiles"]:
            if masks["trade_req_bundle"][t_rel, t]:
                req_b[t] = 1
        if pc["req_goojf"] >= 1 and masks["trade_req_bundle"][t_rel, GOOJF_SLOT_1]:
            req_b[GOOJF_SLOT_1] = 1
        if pc["req_goojf"] >= 2 and masks["trade_req_bundle"][t_rel, GOOJF_SLOT_2]:
            req_b[GOOJF_SLOT_2] = 1
        for t in pc["offer_tiles"]:
            if masks["trade_offer_bundle"][t]:
                off_b[t] = 1
        if pc["offer_goojf"] >= 1 and masks["trade_offer_bundle"][GOOJF_SLOT_1]:
            off_b[GOOJF_SLOT_1] = 1
        if pc["offer_goojf"] >= 2 and masks["trade_offer_bundle"][GOOJF_SLOT_2]:
            off_b[GOOJF_SLOT_2] = 1

        out["trade_req_bundle"] = req_b
        out["trade_offer_bundle"] = off_b
        out["trade_offer_cash"] = _nearest_cash_bracket(pc["offer_cash"], core.cash[player_id])
        opp_id = (player_id + 1 + t_rel) % 4
        out["trade_req_cash"] = _nearest_cash_bracket(pc["req_cash"], core.cash[opp_id])
        return out

    def _find_best_trade_proposal(
        self, core: MonopolyCore, player_id: int, masks: Dict[str, np.ndarray]
    ) -> Optional[Tuple[int, np.ndarray, np.ndarray, int, int]]:
        """Finds a mutually beneficial monopoly-completing bundle trade if one is legal."""
        if not masks["property"][ACT_PROP_PROPOSE_TRADE]:
            return None

        my_singles = [
            t for t in PURCHASABLE_TILES
            if masks["trade_offer_bundle"][t] and not self._owns_monopoly(core, player_id, TILE_TO_GROUP.get(t, ""))
        ]
        cash = core.cash[player_id]
        reserve = self._desired_reserve(core, player_id)

        best_score = -1e9
        best_plan = None

        for rel in range(3):
            if not masks["trade_target"][rel]:
                continue
            opp = (player_id + 1 + rel) % 4
            req_mask = masks["trade_req_bundle"][rel]

            # Look for color groups where self already owns >=1 and opp holds the missing pieces (usually 1 or 2)
            for grp, grp_tiles in COLOR_GROUPS.items():
                missing_from_opp = [t for t in grp_tiles if core.property_owner[t] == opp and req_mask[t]]
                owned_by_me = [t for t in grp_tiles if core.property_owner[t] == player_id]
                if not missing_from_opp or len(owned_by_me) + len(missing_from_opp) != len(grp_tiles):
                    continue

                # Estimate value needed to persuade opponent
                opp_loss = sum(self._tile_strategic_value(core, t, opp) for t in missing_from_opp)
                # Try giving opponent a card they want + top-up cash (avoid giving away stronger monopoly freely)
                sweetener_tiles: List[int] = []
                for mt in my_singles:
                    mgrp = TILE_TO_GROUP.get(mt, "")
                    if mgrp and mgrp != grp:
                        m_tiles = COLOR_GROUPS[mgrp]
                        if any(core.property_owner[x] == opp for x in m_tiles):
                            sweetener_tiles.append(mt)
                            if len(sweetener_tiles) >= 2:
                                break
                if not sweetener_tiles and cash < reserve + int(opp_loss):
                    continue

                sweet_val_to_opp = sum(self._tile_strategic_value(core, t, opp) for t in sweetener_tiles)
                cash_diff = (opp_loss * 1.15) - sweet_val_to_opp
                off_cash_idx = 0
                req_cash_idx = 0
                if cash_diff >= 0:
                    max_spend = max(0, cash - int(0.5 * reserve))
                    if cash_diff > max_spend:
                        continue
                    off_cash_idx = _nearest_cash_bracket(cash_diff, max_spend)
                else:
                    req_cash_idx = _nearest_cash_bracket(-cash_diff * 0.85, core.cash[opp])

                req_b = np.zeros(BUNDLE_DIM, dtype=np.int8)
                off_b = np.zeros(BUNDLE_DIM, dtype=np.int8)
                for t in missing_from_opp:
                    req_b[t] = 1
                for t in sweetener_tiles:
                    off_b[t] = 1

                if not np.any(off_b) and off_cash_idx == 0:
                    continue

                gw = GROUP_VALUE_WEIGHT.get(grp, 1.0)
                score = gw * 400.0 - TRADE_CASH_BRACKETS[off_cash_idx] + TRADE_CASH_BRACKETS[req_cash_idx]
                if score > best_score:
                    best_score = score
                    best_plan = (rel, req_b, off_b, off_cash_idx, req_cash_idx)

        return best_plan

    def _decide_property_manage(
        self, core: MonopolyCore, player_id: int, masks: Dict[str, Any], out: Dict[str, Any]
    ) -> Dict[str, Any]:
        mask_p = masks["property"]
        cash = core.cash[player_id]
        reserve = self._desired_reserve(core, player_id)

        # 1. Liquidity defense: if cash dangerously low (< 40% reserve), raise funds carefully
        if cash < int(0.35 * reserve) and core.manage_actions_this_turn < 10:
            for t in PURCHASABLE_TILES:
                if mask_p[80 + t] and TILES_DATA[t]["type"] != TILE_RAILROAD:
                    out["property"] = 80 + t
                    return out
            for t in PURCHASABLE_TILES:
                if mask_p[80 + t]:
                    out["property"] = 80 + t
                    return out

        # 2. Build houses with 4-house shortage lock & reserve protection
        lock_4h = self._should_hold_four_house_lock(core, player_id)
        best_build = None
        best_build_score = -1e9
        for t in PURCHASABLE_TILES:
            if not mask_p[t]:
                continue
            cur_h = core.property_houses[t]
            h_cost = core.get_property_house_cost(t, TILES_DATA[t]["house_cost"])
            if cur_h == 4 and lock_4h:
                continue  # Explicit Pro 4-House Shortage Lock!
            keep_buf = int(0.35 * reserve) if cur_h < 3 else int(0.65 * reserve)
            if cash - h_cost < keep_buf and cur_h >= 1:
                continue
            if cash < h_cost:
                continue
            grp = TILE_TO_GROUP.get(t, "")
            gw = GROUP_VALUE_WEIGHT.get(grp, 1.0)
            # Sweet spot is house 1 -> 3 first across monopolies
            stage_bonus = 3.0 if cur_h < 3 else (1.5 if cur_h == 3 else 0.4)
            sc = gw * 100.0 + stage_bonus * 30.0 - h_cost * 0.05
            if sc > best_build_score:
                best_build_score = sc
                best_build = t

        if best_build is not None:
            out["property"] = best_build
            return out

        # 3. Unmortgage high-value / monopoly tiles when holding excess liquidity
        best_unmtg = None
        best_unmtg_score = -1e9
        for t in PURCHASABLE_TILES:
            if mask_p[120 + t]:
                cost = int(core.get_property_mortgage(t, 0) * (1.0 + UNMORTGAGE_INTEREST_RATE))
                if cash - cost >= int(0.85 * reserve):
                    val = self._tile_strategic_value(core, t, player_id)
                    if val > best_unmtg_score:
                        best_unmtg_score = val
                        best_unmtg = 120 + t
        if best_unmtg is not None:
            out["property"] = best_unmtg
            return out

        # 4. Propose a monopoly-completing bundle trade if available
        plan = self._find_best_trade_proposal(core, player_id, masks)
        if plan is not None:
            rel, req_b, off_b, off_c_idx, req_c_idx = plan
            out["property"] = ACT_PROP_PROPOSE_TRADE
            out["trade_target"] = rel
            out["trade_req_bundle"] = req_b
            out["trade_offer_bundle"] = off_b
            out["trade_offer_cash"] = off_c_idx
            out["trade_req_cash"] = req_c_idx
            return out

        out["property"] = ACT_PROP_END_MANAGE
        return out


class OverbidderBot(ProHeuristicBot):
    """Aggressive auction shark / hotel-rusher for PSRO adversary diversity."""

    def __init__(self, aggression: float = 1.48, seed: int = 17):
        super().__init__(aggression=aggression, seed=seed)

    def _desired_reserve(self, core: MonopolyCore, player_id: int) -> int:
        return 60 if core.round_count < 18 else 120

    def _should_hold_four_house_lock(self, core: MonopolyCore, player_id: int) -> bool:
        return False  # Always rushes hotels aggressively

    def _decide_buy(self, core: MonopolyCore, player_id: int, mask_core: np.ndarray) -> int:
        if mask_core[ACT_CORE_BUY_DIRECT]:
            return ACT_CORE_BUY_DIRECT
        return ACT_CORE_DECLINE_BUY

    def _decide_auction(self, core: MonopolyCore, player_id: int, mask_auc: np.ndarray) -> int:
        tile_id = core.auction_property
        if tile_id is None or core.auction_highest_bidder == player_id:
            return ACT_AUC_FOLD
        price = core.get_property_price(tile_id, TILES_DATA[tile_id]["price"])
        val = max(price * 1.42, self._tile_strategic_value(core, tile_id, player_id))
        cur = core.auction_highest_bid
        max_bid = min(int(val), max(0, core.cash[player_id] - 25))
        if cur >= max_bid:
            return ACT_AUC_FOLD
        if mask_auc[ACT_AUC_JUMP_LIST] and price <= max_bid:
            return ACT_AUC_JUMP_LIST
        if mask_auc[ACT_AUC_BID_50] and cur + 50 <= max_bid:
            return ACT_AUC_BID_50
        if mask_auc[ACT_AUC_BID_10] and cur + 10 <= max_bid:
            return ACT_AUC_BID_10
        if mask_auc[ACT_AUC_BID_1] and cur + 1 <= max_bid:
            return ACT_AUC_BID_1
        return ACT_AUC_FOLD
