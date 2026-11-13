"""Interactive Pygame Renderer for 4-Agent Monopoly MARL.

Renders high-fidelity 11x11 Monopoly board with authentic tile badges,
vibrant color bands, complete monopoly indicators, non-overlapping
3D player tokens (Luke, Graham, Pieter, Allison), realistic tumbling 3D dice,
smooth hop-by-hop parabolic jump animations, landing shockwave ripples,
active tile spotlight, floating action banners (rent, trades, hotel construction),
authentic Title Deed cards, Chance cards, Community Chest cards,
interactive tile inspection, and dynamic context-sensitive center HUD.
"""

from typing import Dict, List, Tuple, Optional, Any
import pygame
import math
import random
import numpy as np

from model_2.engine.board_data import (
    TILES_DATA, COLOR_GROUPS, TILE_TO_GROUP, PURCHASABLE_TILES,
    COLOR_BROWN, COLOR_LIGHT_BLUE, COLOR_PINK, COLOR_ORANGE,
    COLOR_RED, COLOR_YELLOW, COLOR_GREEN, COLOR_DARK_BLUE,
    COLOR_RAILROAD, COLOR_UTILITY,
    TILE_STREET, TILE_RAILROAD, TILE_UTILITY, TILE_TAX, TILE_CARD, TILE_SPECIAL
)
from model_2.engine.cards_data import CHANCE_CARDS, COMMUNITY_CHEST_CARDS
from model_2.engine.monopoly_core import (
    MonopolyCore, PHASE_ROLL, PHASE_BUY_OR_AUCTION, PHASE_AUCTION,
    PHASE_JAIL_DECISION, PHASE_PROPERTY_MANAGE, PHASE_TRADE_RESPONSE, PHASE_GAME_OVER
)

# Color Palette
BG_COLOR = (18, 22, 30)
BOARD_BG = (236, 240, 246)
TILE_BG = (255, 255, 255)
TILE_BORDER = (180, 188, 202)
TEXT_COLOR = (24, 28, 38)
TEXT_MUTED = (95, 105, 122)
TEXT_LIGHT = (240, 244, 252)

COLOR_HEADER_MAP = {
    COLOR_BROWN: (136, 68, 24),
    COLOR_LIGHT_BLUE: (115, 195, 238),
    COLOR_PINK: (212, 95, 175),
    COLOR_ORANGE: (245, 130, 15),
    COLOR_RED: (224, 30, 48),
    COLOR_YELLOW: (248, 210, 18),
    COLOR_GREEN: (30, 150, 60),
    COLOR_DARK_BLUE: (16, 40, 120),
}

# Player Identity Colors
PLAYER_COLORS = [
    (239, 71, 111),   # Player 0: Luke (Coral Red)
    (17, 138, 178),   # Player 1: Graham (Ocean Blue)
    (6, 214, 160),    # Player 2: Pieter (Mint Green)
    (255, 209, 102),  # Player 3: Allison (Warm Yellow)
]

PLAYER_INITIALS = ["L", "G", "P", "A"]

HOUSE_COLOR = (34, 153, 84)     # Forest Green
HOTEL_COLOR = (214, 48, 49)     # Crimson Red
GOLD_COLOR = (241, 196, 15)     # Shiny Gold
JAIL_BAR_COLOR = (55, 60, 72)

# Curated 2-line clean display labels and icons for all 40 tiles
TILE_DISPLAY_CONFIG = {
    0: {"line1": "COLLECT", "line2": "$200", "sub": "GO", "icon": "go"},
    1: {"line1": "MEDITER.", "line2": "AVE", "price": "$60"},
    2: {"line1": "COMMUNITY", "line2": "CHEST", "icon": "chest"},
    3: {"line1": "BALTIC", "line2": "AVE", "price": "$60"},
    4: {"line1": "INCOME", "line2": "TAX", "price": "Pay $200", "icon": "tax"},
    5: {"line1": "READING", "line2": "RAILROAD", "price": "$200", "icon": "rr"},
    6: {"line1": "ORIENTAL", "line2": "AVE", "price": "$100"},
    7: {"line1": "CHANCE", "line2": "", "icon": "chance"},
    8: {"line1": "VERMONT", "line2": "AVE", "price": "$100"},
    9: {"line1": "CONNECT.", "line2": "AVE", "price": "$120"},
    10: {"line1": "JAIL", "line2": "CELL", "sub": "JUST VISITING", "icon": "jail"},
    11: {"line1": "ST. CHARLES", "line2": "PLACE", "price": "$140"},
    12: {"line1": "ELECTRIC", "line2": "CO.", "price": "$150", "icon": "utility_elec"},
    13: {"line1": "STATES", "line2": "AVE", "price": "$140"},
    14: {"line1": "VIRGINIA", "line2": "AVE", "price": "$160"},
    15: {"line1": "PENN.", "line2": "RAILROAD", "price": "$200", "icon": "rr"},
    16: {"line1": "ST. JAMES", "line2": "PLACE", "price": "$180"},
    17: {"line1": "COMMUNITY", "line2": "CHEST", "icon": "chest"},
    18: {"line1": "TENNESSEE", "line2": "AVE", "price": "$180"},
    19: {"line1": "NEW YORK", "line2": "AVE", "price": "$200"},
    20: {"line1": "FREE", "line2": "PARKING", "icon": "parking"},
    21: {"line1": "KENTUCKY", "line2": "AVE", "price": "$220"},
    22: {"line1": "CHANCE", "line2": "", "icon": "chance"},
    23: {"line1": "INDIANA", "line2": "AVE", "price": "$220"},
    24: {"line1": "ILLINOIS", "line2": "AVE", "price": "$240"},
    25: {"line1": "B. & O.", "line2": "RAILROAD", "price": "$200", "icon": "rr"},
    26: {"line1": "ATLANTIC", "line2": "AVE", "price": "$260"},
    27: {"line1": "VENTNOR", "line2": "AVE", "price": "$260"},
    28: {"line1": "WATER", "line2": "WORKS", "price": "$150", "icon": "utility_water"},
    29: {"line1": "MARVIN", "line2": "GARDENS", "price": "$280"},
    30: {"line1": "GO TO", "line2": "JAIL", "icon": "gotojail"},
    31: {"line1": "PACIFIC", "line2": "AVE", "price": "$300"},
    32: {"line1": "NORTH", "line2": "CAROLINA", "price": "$300"},
    33: {"line1": "COMMUNITY", "line2": "CHEST", "icon": "chest"},
    34: {"line1": "PENN.", "line2": "AVE", "price": "$320"},
    35: {"line1": "SHORT", "line2": "LINE", "price": "$200", "icon": "rr"},
    36: {"line1": "CHANCE", "line2": "", "icon": "chance"},
    37: {"line1": "PARK", "line2": "PLACE", "price": "$350"},
    38: {"line1": "LUXURY", "line2": "TAX", "price": "Pay $100", "icon": "tax"},
    39: {"line1": "BOARDWALK", "line2": "", "price": "$400"},
}

CARD_DESCRIPTIONS: Dict[int, str] = {
    # Chance Cards (0..15)
    0: "Advance directly to GO. Collect $200.",
    1: "Advance to Illinois Avenue. If you pass GO, collect $200.",
    2: "Advance to St. Charles Place. If you pass GO, collect $200.",
    3: "Advance to nearest Utility. If unowned, buy from Bank. If owned, throw dice and pay 10x throw.",
    4: "Advance to nearest Railroad. If unowned, buy it. If owned, pay owner 2x normal rent.",
    5: "Advance to nearest Railroad. If unowned, buy it. If owned, pay owner 2x normal rent.",
    6: "Bank pays you dividend of $50.",
    7: "This card may be kept until needed or traded: GET OUT OF JAIL FREE.",
    8: "Go back 3 spaces.",
    9: "Go directly to Jail. Do not pass GO, do not collect $200.",
    10: "Make general repairs on all your property: Pay $25 per house, $100 per hotel.",
    11: "Speeding fine. Pay $15 to the Bank.",
    12: "Take a trip to Reading Railroad. If you pass GO, collect $200.",
    13: "You have been elected Chairman of the Board. Pay each player $50.",
    14: "Your building loan matures. Collect $150 from the Bank.",
    15: "Advance directly to Boardwalk.",

    # Community Chest Cards (100..115)
    100: "Advance to GO. Collect $200.",
    101: "Bank error in your favor. Collect $200 from the Bank.",
    102: "Doctor's fees. Pay $50 to the Bank.",
    103: "From sale of stock you get $50.",
    104: "This card may be kept until needed or traded: GET OUT OF JAIL FREE.",
    105: "Go directly to Jail. Do not pass GO, do not collect $200.",
    106: "Holiday fund matures. Receive $100.",
    107: "Income tax refund. Collect $20 from the Bank.",
    108: "It is your birthday. Collect $10 from every player.",
    109: "Life insurance matures. Collect $100 from the Bank.",
    110: "Hospital fees. Pay $100 to the Bank.",
    111: "School fees. Pay $50 to the Bank.",
    112: "Receive $25 consultancy fee.",
    113: "Street repairs: Pay $40 per house and $115 per hotel.",
    114: "Won second prize in a beauty contest. Collect $10.",
    115: "You inherit $100.",
}


class MonopolyRenderer:
    """Renders the Monopoly 11x11 game board, visual tokens, cards, and center HUD."""

    def __init__(self, surface: pygame.Surface, board_size: int = 800, offset_x: int = 20, offset_y: int = 20):
        self.surface = surface
        self.board_size = board_size
        self.offset_x = offset_x
        self.offset_y = offset_y

        # Layout metrics
        self.corner_size = 100
        self.tile_count_per_side = 9
        self.tile_step = (self.board_size - 2 * self.corner_size) / float(self.tile_count_per_side)

        # Precompute tile bounding rects
        self.tile_rects = self._compute_tile_rects()

        # Fonts - upgraded sizing for superior legibility
        pygame.font.init()
        self.font_tile = pygame.font.SysFont("Arial", 10, bold=True)
        self.font_price = pygame.font.SysFont("Arial", 10, bold=True)
        self.font_corner_title = pygame.font.SysFont("Arial", 13, bold=True)
        self.font_corner_sub = pygame.font.SysFont("Arial", 9, bold=True)
        self.font_token = pygame.font.SysFont("Arial", 12, bold=True)
        self.font_center_title = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_center_sub = pygame.font.SysFont("Arial", 12)
        self.font_dice = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_badge = pygame.font.SysFont("Arial", 11, bold=True)
        self.font_victory_title = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_victory_sub = pygame.font.SysFont("Arial", 12, bold=True)
        self.font_toast_title = pygame.font.SysFont("Arial", 13, bold=True)
        self.font_toast_sub = pygame.font.SysFont("Arial", 10, bold=False)

        # Card specific fonts
        self.font_deed_header = pygame.font.SysFont("Arial", 11, bold=True)
        self.font_deed_sub = pygame.font.SysFont("Arial", 8, bold=True)
        self.font_deed_rent = pygame.font.SysFont("Arial", 10, bold=True)
        self.font_deed_row = pygame.font.SysFont("Arial", 9)
        self.font_deed_footer = pygame.font.SysFont("Arial", 8)
        self.font_card_title = pygame.font.SysFont("Arial", 12, bold=True)
        self.font_card_body = pygame.font.SysFont("Arial", 10)

        self.animation_tick = 0
        self.simulation_speed = 5
        self.inspect_tile_id: Optional[int] = None

        # --- Dynamic Token Hopping System ---
        self.token_positions: Dict[int, int] = {p: 0 for p in range(4)}
        self.token_in_jail: Dict[int, bool] = {p: False for p in range(4)}
        self.hop_queues: Dict[int, List[int]] = {p: [] for p in range(4)}
        self.current_hop_from: Dict[int, int] = {p: 0 for p in range(4)}
        self.current_hop_to: Dict[int, int] = {p: 0 for p in range(4)}
        self.hop_progress: Dict[int, float] = {p: 1.0 for p in range(4)}
        self.landing_ripples: List[Dict[str, Any]] = []

        # --- Animated 3D Rolling Dice ---
        self.last_dice_seen: Tuple[int, int] = (1, 2)
        self.dice_roll_timer: int = 0
        self.dice_roll_values: Tuple[int, int] = (1, 2)

        # --- Floating Event Toast Banners ---
        self.last_event_count: int = 0
        self.active_toasts: List[Dict[str, Any]] = []

        # Simulation pause status
        self.is_paused: bool = True

    def reset(self, core: MonopolyCore):
        """Resets visual animation state upon match restart."""
        self.token_positions = {p: core.position[p] for p in range(4)}
        self.token_in_jail = {p: core.in_jail[p] for p in range(4)}
        self.hop_queues = {p: [] for p in range(4)}
        self.current_hop_from = {p: core.position[p] for p in range(4)}
        self.current_hop_to = {p: core.position[p] for p in range(4)}
        self.hop_progress = {p: 1.0 for p in range(4)}
        self.landing_ripples = []
        self.dice_roll_timer = 0
        self.dice_roll_values = core.last_dice
        self.last_dice_seen = core.last_dice
        self.active_toasts = []
        self.last_event_count = len(core.event_log)
        self.inspect_tile_id = None
        self.is_paused = True

    def _compute_tile_rects(self) -> Dict[int, pygame.Rect]:
        """Calculates precise screen coordinates for all 40 tiles."""
        rects = {}
        bs = self.board_size
        cs = self.corner_size
        ts = self.tile_step
        ox, oy = self.offset_x, self.offset_y

        # Tile 0: Bottom-right corner (GO)
        rects[0] = pygame.Rect(ox + bs - cs, oy + bs - cs, cs, cs)

        # Bottom row: Tiles 1 to 9 (from right to left)
        for i in range(1, 10):
            rx = ox + bs - cs - i * ts
            ry = oy + bs - cs
            rects[i] = pygame.Rect(rx, ry, ts + 1, cs)

        # Tile 10: Bottom-left corner (Jail)
        rects[10] = pygame.Rect(ox, oy + bs - cs, cs, cs)

        # Left row: Tiles 11 to 19 (from bottom to top)
        for i in range(1, 10):
            rx = ox
            ry = oy + bs - cs - i * ts
            rects[10 + i] = pygame.Rect(rx, ry, cs, ts + 1)

        # Tile 20: Top-left corner (Free Parking)
        rects[20] = pygame.Rect(ox, oy, cs, cs)

        # Top row: Tiles 21 to 29 (from left to right)
        for i in range(1, 10):
            rx = ox + cs + (i - 1) * ts
            ry = oy
            rects[20 + i] = pygame.Rect(rx, ry, ts + 1, cs)

        # Tile 30: Top-right corner (Go To Jail)
        rects[30] = pygame.Rect(ox + bs - cs, oy, cs, cs)

        # Right row: Tiles 31 to 39 (from top to bottom)
        for i in range(1, 10):
            rx = ox + bs - cs
            ry = oy + cs + (i - 1) * ts
            rects[30 + i] = pygame.Rect(rx, ry, cs, ts + 1)

        return rects

    def handle_click(self, pos: Tuple[int, int]) -> bool:
        """Handles mouse click on board tiles for interactive card inspection."""
        if self.inspect_tile_id is not None:
            cx = self.offset_x + self.board_size // 2
            cy = self.offset_y + self.board_size // 2
            close_btn = pygame.Rect(cx + 100, cy - 145, 24, 24)
            if close_btn.collidepoint(pos):
                self.inspect_tile_id = None
                return True

        for t_id, rect in self.tile_rects.items():
            if rect.collidepoint(pos):
                if self.inspect_tile_id == t_id:
                    self.inspect_tile_id = None
                else:
                    self.inspect_tile_id = t_id
                return True

        board_rect = pygame.Rect(self.offset_x, self.offset_y, self.board_size, self.board_size)
        if board_rect.collidepoint(pos) and self.inspect_tile_id is not None:
            self.inspect_tile_id = None
            return True

        return False

    def is_animating(self) -> bool:
        """Returns True if tokens are actively hopping or dice are rolling."""
        if self.dice_roll_timer > 0:
            return True
        for p in range(4):
            if len(self.hop_queues[p]) > 0 or self.hop_progress[p] < 1.0:
                return True
        return False

    def get_token_tile_coords(self, p_id: int, tile_id: int, in_jail: bool = False) -> Tuple[int, int]:
        """Calculates precise non-overlapping center coordinates for player token at any tile."""
        rect = self.tile_rects[tile_id]

        if tile_id == 0:  # GO corner: neatly arranged in open space below text
            offsets = [(-18, 15), (18, 15), (-18, 28), (18, 28)]
            return (rect.centerx + offsets[p_id][0], rect.centery + offsets[p_id][1])
        elif tile_id == 10:  # JAIL
            if in_jail:
                cell_cx = rect.right - 26
                cell_cy = rect.y + 26
                offsets = [(-9, -9), (9, -9), (-9, 9), (9, 9)]
                return (cell_cx + offsets[p_id][0], cell_cy + offsets[p_id][1])
            else:
                vis_offsets = [(-22, 16), (2, 16), (-22, 34), (2, 34)]
                return (rect.centerx + vis_offsets[p_id][0], rect.centery + vis_offsets[p_id][1])
        elif tile_id == 20:  # FREE PARKING: middle zone between emblem and text
            offsets = [(-16, -2), (16, -2), (-16, 14), (16, 14)]
            return (rect.centerx + offsets[p_id][0], rect.centery + offsets[p_id][1])
        elif tile_id == 30:  # GO TO JAIL: middle zone between badge and text
            offsets = [(-16, -2), (16, -2), (-16, 14), (16, 14)]
            return (rect.centerx + offsets[p_id][0], rect.centery + offsets[p_id][1])

        # Regular side tiles
        if 0 < tile_id < 10:  # Bottom row
            offsets = [(-14, 8), (14, 8), (-14, 23), (14, 23)]
            return (rect.centerx + offsets[p_id][0], rect.centery + offsets[p_id][1])
        elif 10 < tile_id < 20:  # Left row
            offsets = [(2, -14), (18, -14), (2, 14), (18, 14)]
            return (rect.centerx + offsets[p_id][0], rect.centery + offsets[p_id][1])
        elif 20 < tile_id < 30:  # Top row
            offsets = [(-14, -8), (14, -8), (-14, 8), (14, 8)]
            return (rect.centerx + offsets[p_id][0], rect.centery + offsets[p_id][1])
        else:  # Right row (31..39)
            offsets = [(-18, -14), (0, -14), (-18, 14), (0, 14)]
            return (rect.centerx + offsets[p_id][0], rect.centery + offsets[p_id][1])

    def render(self, core: MonopolyCore, speed: Optional[int] = None, is_paused: Optional[bool] = None):
        """Renders complete board state onto surface with animations and toast banners."""
        self.animation_tick += 1
        if speed is not None:
            self.simulation_speed = speed
        if is_paused is not None:
            self.is_paused = is_paused

        # 1. Update event toasts & dice animation
        self._process_new_events(core)
        self._update_dice(core)

        # 2. Update token hopping trajectory
        self._update_token_hops(core)

        # 3. Draw board background & authentic felt decorations
        board_rect = pygame.Rect(self.offset_x, self.offset_y, self.board_size, self.board_size)
        pygame.draw.rect(self.surface, BOARD_BG, board_rect, border_radius=10)
        pygame.draw.rect(self.surface, (45, 52, 65), board_rect, width=3, border_radius=10)
        self._render_board_decorations()

        # 4. Draw all 40 tiles with active spotlight
        active_tile = core.position[core.active_decision_player]
        for tile_id in range(40):
            is_active_spotlight = (tile_id == active_tile and core.phase != PHASE_GAME_OVER)
            self._render_tile(tile_id, core, is_active=is_active_spotlight)

        # 5. Draw landing shockwaves
        self._render_ripples()

        # 6. Draw player tokens with 3D drop shadow & hopping arc
        self._render_tokens(core)

        # 7. Draw center board HUD (dice, active player, card, auction, game over)
        self._render_center_hud(core)

        # 8. Draw floating event toast notifications
        self._render_toast_banners()

        # 9. Inspection modal overlay if tile clicked
        if self.inspect_tile_id is not None:
            self._render_inspection_overlay(core)

    def _update_dice(self, core: MonopolyCore):
        """Tracks dice rolls and triggers tumbling animation."""
        if core.last_dice != self.last_dice_seen:
            self.last_dice_seen = core.last_dice
            # Roll for 14 frames at speeds <= 5, instant at higher speeds
            self.dice_roll_timer = 14 if self.simulation_speed <= 5 else 1

        if self.dice_roll_timer > 0:
            self.dice_roll_timer -= 1
            if self.dice_roll_timer > 0:
                self.dice_roll_values = (random.randint(1, 6), random.randint(1, 6))
            else:
                self.dice_roll_values = core.last_dice
        else:
            self.dice_roll_values = core.last_dice

    def _process_new_events(self, core: MonopolyCore):
        """Scans core.event_log for impactful game events and creates floating action toasts."""
        curr_len = len(core.event_log)
        if curr_len > self.last_event_count:
            new_events = core.event_log[self.last_event_count:curr_len]
            self.last_event_count = curr_len

            for ev in new_events:
                if "paid $" in ev and "rent to" in ev:
                    self.active_toasts.append({
                        "title": "RENT PAYMENT", "sub": ev,
                        "accent_color": (231, 76, 60), "icon": "$",
                        "timer": 90, "max_timer": 90
                    })
                elif "bought " in ev or "won auction for " in ev:
                    self.active_toasts.append({
                        "title": "PROPERTY PURCHASED", "sub": ev,
                        "accent_color": (46, 204, 113), "icon": "+",
                        "timer": 80, "max_timer": 80
                    })
                elif "HOTEL" in ev and ("built" in ev or "upgraded" in ev):
                    self.active_toasts.append({
                        "title": "HOTEL CONSTRUCTED!", "sub": ev,
                        "accent_color": (214, 48, 49), "icon": "H",
                        "timer": 100, "max_timer": 100
                    })
                elif "TRADE ACCEPTED" in ev:
                    self.active_toasts.append({
                        "title": "TRADE DEAL COMPLETED!", "sub": ev,
                        "accent_color": (52, 152, 219), "icon": "<->",
                        "timer": 110, "max_timer": 110
                    })
                elif "IS BANKRUPT" in ev:
                    self.active_toasts.append({
                        "title": "PLAYER ELIMINATED!", "sub": ev,
                        "accent_color": (192, 57, 43), "icon": "X",
                        "timer": 120, "max_timer": 120
                    })
                elif "sent to Jail" in ev:
                    self.active_toasts.append({
                        "title": "ARRESTED -> GO TO JAIL", "sub": ev,
                        "accent_color": (155, 89, 182), "icon": "!",
                        "timer": 90, "max_timer": 90
                    })
                elif "passed GO" in ev:
                    self.active_toasts.append({
                        "title": "PASSED GO -> SALARY", "sub": "+$200 collected for passing GO",
                        "accent_color": (39, 174, 96), "icon": "$",
                        "timer": 75, "max_timer": 75
                    })

    def _update_token_hops(self, core: MonopolyCore):
        """Maintains smooth step-by-step hopping queues and updates fractional jump progress."""
        for p in range(4):
            if core.bankrupt[p]:
                continue

            actual_pos = core.position[p]
            actual_jail = core.in_jail[p]
            tracked_pos = self.token_positions[p]
            tracked_jail = self.token_in_jail[p]

            # Detect new movement
            if actual_pos != tracked_pos or actual_jail != tracked_jail:
                self.token_positions[p] = actual_pos
                self.token_in_jail[p] = actual_jail

                if actual_jail and not tracked_jail:
                    # Direct swift transfer to jail cell
                    self.hop_queues[p] = [10]
                elif (tracked_pos - actual_pos) % 40 == 37:
                    # Chance card: Go back 3 spaces
                    self.hop_queues[p] = [
                        (tracked_pos - 1) % 40,
                        (tracked_pos - 2) % 40,
                        actual_pos
                    ]
                else:
                    # Normal forward advancement along perimeter
                    hops = []
                    curr = tracked_pos
                    while curr != actual_pos:
                        curr = (curr + 1) % 40
                        hops.append(curr)
                    self.hop_queues[p] = hops

                if self.hop_progress[p] >= 1.0 and self.hop_queues[p]:
                    self.current_hop_from[p] = tracked_pos
                    self.current_hop_to[p] = self.hop_queues[p].pop(0)
                    self.hop_progress[p] = 0.0

            # Advance current hop progress
            if self.hop_progress[p] < 1.0:
                if self.simulation_speed == 1:
                    step = 0.125   # 8 frames per hop
                elif self.simulation_speed == 2:
                    step = 0.17    # 6 frames per hop
                elif self.simulation_speed == 5:
                    step = 0.28    # 3-4 frames per hop
                else:
                    step = 1.0     # instant at speeds >= 20

                self.hop_progress[p] = min(1.0, self.hop_progress[p] + step)

                if self.hop_progress[p] >= 1.0:
                    # Destination reached for this hop step -> trigger landing shockwave
                    land_tile = self.current_hop_to[p]
                    lx, ly = self.get_token_tile_coords(p, land_tile, in_jail=(land_tile == 10 and actual_jail))
                    self.landing_ripples.append({
                        "x": lx, "y": ly, "r": 6.0, "max_r": 24.0,
                        "color": PLAYER_COLORS[p], "alpha": 220.0
                    })

                    # If more intermediate hops remain in queue, pop next
                    if self.hop_queues[p]:
                        self.current_hop_from[p] = self.current_hop_to[p]
                        self.current_hop_to[p] = self.hop_queues[p].pop(0)
                        self.hop_progress[p] = 0.0
                    else:
                        self.current_hop_from[p] = actual_pos
                        self.current_hop_to[p] = actual_pos

    def _render_ripples(self):
        """Draws expanding landing pulse shockwaves."""
        active = []
        for rip in self.landing_ripples:
            rip["r"] += 2.0
            rip["alpha"] -= 16.0
            if rip["alpha"] > 0 and rip["r"] < rip["max_r"]:
                surf = pygame.Surface((int(rip["r"] * 2 + 6), int(rip["r"] * 2 + 6)), pygame.SRCALPHA)
                col = rip["color"]
                a = max(0, min(255, int(rip["alpha"])))
                pygame.draw.circle(surf, (col[0], col[1], col[2], a),
                                   (int(rip["r"] + 3), int(rip["r"] + 3)), int(rip["r"]), width=2)
                self.surface.blit(surf, (rip["x"] - int(rip["r"] + 3), rip["y"] - int(rip["r"] + 3)))
                active.append(rip)
        self.landing_ripples = active

    def _render_toast_banners(self):
        """Renders floating event toast banners in the open interior area above the center HUD."""
        if not self.active_toasts:
            return

        cx = self.offset_x + self.board_size // 2
        cy = self.offset_y + self.board_size // 2
        base_y = cy - 180  # Sits in open space (y ~ 240) with ample clearance above HUD

        active = []
        for idx, toast in enumerate(self.active_toasts[-2:]):
            toast["timer"] -= 1
            timer = toast["timer"]
            max_timer = toast["max_timer"]

            if timer <= 0:
                continue

            active.append(toast)
            life_ratio = timer / float(max_timer)

            y_float = int((1.0 - life_ratio) * 16)
            ty = base_y - (idx * 46) - y_float

            if (max_timer - timer) < 10:
                alpha = int(255 * (max_timer - timer) / 10.0)
            elif timer < 20:
                alpha = int(255 * timer / 20.0)
            else:
                alpha = 255

            banner_w, banner_h = 320, 40
            banner_rect = pygame.Rect(0, 0, banner_w, banner_h)
            banner_surf = pygame.Surface((banner_w, banner_h), pygame.SRCALPHA)

            # Dark card with colored accent border
            ac = toast["accent_color"]
            pygame.draw.rect(banner_surf, (20, 26, 38, int(alpha * 0.94)), banner_rect, border_radius=8)
            pygame.draw.rect(banner_surf, (ac[0], ac[1], ac[2], alpha), banner_rect, width=2, border_radius=8)
            pygame.draw.rect(banner_surf, (ac[0], ac[1], ac[2], alpha),
                             pygame.Rect(0, 0, 7, banner_h), border_top_left_radius=8, border_bottom_left_radius=8)

            t_surf = self.font_toast_title.render(f"[{toast['icon']}] {toast['title']}", True, (255, 255, 255))
            banner_surf.blit(t_surf, (14, 3))

            sub_txt = toast["sub"]
            if len(sub_txt) > 42:
                sub_txt = sub_txt[:39] + "..."
            s_surf = self.font_toast_sub.render(sub_txt, True, (215, 225, 240))
            banner_surf.blit(s_surf, (14, 21))

            self.surface.blit(banner_surf, (cx - banner_w // 2, ty))

        self.active_toasts = active

    def _is_monopoly_tile(self, tile_id: int, core: MonopolyCore) -> bool:
        """Returns True if the tile belongs to a completely owned color group monopoly."""
        owner = core.property_owner[tile_id]
        if owner == -1:
            return False
        grp = TILE_TO_GROUP.get(tile_id)
        if not grp or grp in ("RAILROAD", "UTILITY"):
            return False
        tiles = COLOR_GROUPS[grp]
        return all(core.property_owner[t] == owner for t in tiles)

    def _render_tile(self, tile_id: int, core: MonopolyCore, is_active: bool = False):
        """Draws individual tile background, color band, labels, icons, spotlight, and buildings."""
        rect = self.tile_rects[tile_id]
        tile = TILES_DATA[tile_id]
        cfg = TILE_DISPLAY_CONFIG.get(tile_id, {"line1": tile["name"], "line2": ""})

        # Base tile rectangle
        pygame.draw.rect(self.surface, TILE_BG, rect)
        pygame.draw.rect(self.surface, TILE_BORDER, rect, width=1)

        # Active tile spotlight: glowing aura & corner brackets
        if is_active:
            spot_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            alpha_pulse = int(45 + 30 * math.sin(self.animation_tick * 0.15))
            spot_surf.fill((255, 215, 0, alpha_pulse))
            self.surface.blit(spot_surf, (rect.x, rect.y))
            pygame.draw.rect(self.surface, (255, 200, 30), rect, width=2)

            clen = 9
            # Top-left
            pygame.draw.line(self.surface, (255, 235, 80), (rect.x, rect.y), (rect.x + clen, rect.y), 3)
            pygame.draw.line(self.surface, (255, 235, 80), (rect.x, rect.y), (rect.x, rect.y + clen), 3)
            # Top-right
            pygame.draw.line(self.surface, (255, 235, 80), (rect.right, rect.y), (rect.right - clen, rect.y), 3)
            pygame.draw.line(self.surface, (255, 235, 80), (rect.right, rect.y), (rect.right, rect.y + clen), 3)
            # Bottom-left
            pygame.draw.line(self.surface, (255, 235, 80), (rect.x, rect.bottom), (rect.x + clen, rect.bottom), 3)
            pygame.draw.line(self.surface, (255, 235, 80), (rect.x, rect.bottom), (rect.x, rect.bottom - clen), 3)
            # Bottom-right
            pygame.draw.line(self.surface, (255, 235, 80), (rect.right, rect.bottom), (rect.right - clen, rect.bottom), 3)
            pygame.draw.line(self.surface, (255, 235, 80), (rect.right, rect.bottom), (rect.right, rect.bottom - clen), 3)

        # Highlight if selected for inspection
        if self.inspect_tile_id == tile_id:
            pygame.draw.rect(self.surface, (52, 152, 219), rect, width=3)

        # 1. Corner tiles special handling
        if tile_id in (0, 10, 20, 30):
            self._render_corner_tile(tile_id, rect, core)
            return

        # 2. Color header band for street properties (22px deep with beveled border)
        color_grp = tile.get("color")
        has_color = color_grp in COLOR_HEADER_MAP
        is_monopoly = self._is_monopoly_tile(tile_id, core)

        if has_color:
            header_color = COLOR_HEADER_MAP[color_grp]
            band_size = 22
            if 0 < tile_id < 10:  # Bottom: band at top
                band = pygame.Rect(rect.x, rect.y, rect.width, band_size)
            elif 10 < tile_id < 20:  # Left: band at right
                band = pygame.Rect(rect.right - band_size, rect.y, band_size, rect.height)
            elif 20 < tile_id < 30:  # Top: band at bottom
                band = pygame.Rect(rect.x, rect.bottom - band_size, rect.width, band_size)
            else:  # Right: band at left
                band = pygame.Rect(rect.x, rect.y, band_size, rect.height)

            pygame.draw.rect(self.surface, header_color, band)
            pygame.draw.rect(self.surface, (40, 40, 50), band, width=1)

            # If full monopoly: golden crown/star badge in header band
            if is_monopoly:
                pygame.draw.rect(self.surface, GOLD_COLOR, band, width=2)
                pygame.draw.circle(self.surface, GOLD_COLOR, (band.centerx, band.centery), 5)
                pygame.draw.circle(self.surface, (255, 255, 255), (band.centerx, band.centery), 2)

        # 3. Property Ownership Ribbon (6px prominent stripe + 20x15 owner initial badge)
        owner = core.property_owner[tile_id]
        if owner != -1:
            owner_col = PLAYER_COLORS[owner]
            ribbon_w = 6
            if 0 < tile_id < 10:
                rib_rect = pygame.Rect(rect.x, rect.bottom - ribbon_w, rect.width, ribbon_w)
            elif 10 < tile_id < 20:
                rib_rect = pygame.Rect(rect.x, rect.y, ribbon_w, rect.height)
            elif 20 < tile_id < 30:
                rib_rect = pygame.Rect(rect.x, rect.y, rect.width, ribbon_w)
            else:
                rib_rect = pygame.Rect(rect.right - ribbon_w, rect.y, ribbon_w, rect.height)

            pygame.draw.rect(self.surface, owner_col, rib_rect)

            by = rect.bottom - 19 if 0 < tile_id < 10 else (rect.y + 4 if 20 < tile_id < 30 else rect.centery - 7)
            bx = rect.centerx - 10 if (0 < tile_id < 10 or 20 < tile_id < 30) else (rect.x + 4 if 10 < tile_id < 20 else rect.right - 24)
            init_rect = pygame.Rect(bx, by, 20, 15)
            pygame.draw.rect(self.surface, (20, 24, 34), init_rect.move(1, 1), border_radius=3)
            pygame.draw.rect(self.surface, owner_col, init_rect, border_radius=3)
            init_txt = self.font_corner_sub.render(PLAYER_INITIALS[owner], True, (255, 255, 255))
            self.surface.blit(init_txt, (init_rect.centerx - init_txt.get_width() // 2, init_rect.centery - init_txt.get_height() // 2))

        # 4. Render Tile Text and Icon with zero overlap
        icon_type = cfg.get("icon")
        l1 = cfg.get("line1", "")
        l2 = cfg.get("line2", "")
        eff_price = core.get_property_price(tile_id, int(tile.get("price", 0))) if "price" in tile else 0
        price_str = f"${eff_price}" if eff_price > 0 else cfg.get("price", "")

        if 0 < tile_id < 10:  # Bottom Row
            text_y_offset = 12 if core.property_houses[tile_id] > 0 else 0
            if icon_type:
                self._draw_tile_icon(icon_type, rect.centerx, rect.y + 26)
                t1 = self.font_tile.render(l1, True, TEXT_COLOR)
                t2 = self.font_tile.render(l2, True, TEXT_COLOR)
                self.surface.blit(t1, (rect.centerx - t1.get_width() // 2, rect.y + 44))
                if l2:
                    self.surface.blit(t2, (rect.centerx - t2.get_width() // 2, rect.y + 56))
            else:
                t1 = self.font_tile.render(l1, True, TEXT_COLOR)
                t2 = self.font_tile.render(l2, True, TEXT_COLOR)
                self.surface.blit(t1, (rect.centerx - t1.get_width() // 2, rect.y + 26 + text_y_offset))
                if l2:
                    self.surface.blit(t2, (rect.centerx - t2.get_width() // 2, rect.y + 38 + text_y_offset))

            if price_str:
                p_surf = self.font_price.render(price_str, True, (40, 50, 70))
                self.surface.blit(p_surf, (rect.centerx - p_surf.get_width() // 2, rect.bottom - 18))

        elif 10 < tile_id < 20:  # Left Row
            if icon_type:
                self._draw_tile_icon(icon_type, rect.x + 68, rect.centery)
            t1 = self.font_tile.render(l1, True, TEXT_COLOR)
            t2 = self.font_tile.render(l2, True, TEXT_COLOR)
            self.surface.blit(t1, (rect.x + 8, rect.centery - 16))
            if l2:
                self.surface.blit(t2, (rect.x + 8, rect.centery - 5))
            if price_str:
                p_surf = self.font_price.render(price_str, True, (40, 50, 70))
                self.surface.blit(p_surf, (rect.x + 8, rect.centery + 8))

        elif 20 < tile_id < 30:  # Top Row
            if icon_type:
                self._draw_tile_icon(icon_type, rect.centerx, rect.y + 44)
            t1 = self.font_tile.render(l1, True, TEXT_COLOR)
            t2 = self.font_tile.render(l2, True, TEXT_COLOR)
            self.surface.blit(t1, (rect.centerx - t1.get_width() // 2, rect.y + 6))
            if l2:
                self.surface.blit(t2, (rect.centerx - t2.get_width() // 2, rect.y + 18))
            if price_str:
                p_surf = self.font_price.render(price_str, True, (40, 50, 70))
                self.surface.blit(p_surf, (rect.centerx - p_surf.get_width() // 2, rect.bottom - 34))

        elif 30 < tile_id < 40:  # Right Row
            if icon_type:
                self._draw_tile_icon(icon_type, rect.x + 36, rect.centery - 10)
                t1 = self.font_tile.render(l1, True, TEXT_COLOR)
                t2 = self.font_tile.render(l2, True, TEXT_COLOR)
                self.surface.blit(t1, (rect.x + 48, rect.centery - 14))
                if l2:
                    self.surface.blit(t2, (rect.x + 48, rect.centery - 3))
            else:
                t1 = self.font_tile.render(l1, True, TEXT_COLOR)
                t2 = self.font_tile.render(l2, True, TEXT_COLOR)
                self.surface.blit(t1, (rect.x + 26, rect.centery - 16))
                if l2:
                    self.surface.blit(t2, (rect.x + 26, rect.centery - 5))
            if price_str:
                p_surf = self.font_price.render(price_str, True, (40, 50, 70))
                self.surface.blit(p_surf, (rect.x + 26, rect.centery + 8))

        # 5. Mortgaged Overlay
        if core.property_mortgaged[tile_id]:
            mtg_badge = pygame.Rect(rect.centerx - 28, rect.centery - 9, 56, 18)
            pygame.draw.rect(self.surface, (220, 35, 35), mtg_badge, border_radius=4)
            mtg_surf = self.font_corner_sub.render("MORTGAGE", True, (255, 255, 255))
            self.surface.blit(mtg_surf, (mtg_badge.centerx - mtg_surf.get_width() // 2, mtg_badge.centery - mtg_surf.get_height() // 2))

        # 6. Render 3D Houses / Hotel
        houses = core.property_houses[tile_id]
        if houses > 0 and not core.property_mortgaged[tile_id]:
            self._render_buildings(tile_id, rect, houses)

    def _render_board_decorations(self):
        """Draws classic authentic Monopoly board felt decorations (Logo, Chance & Chest card deck slots)."""
        ox, oy = self.offset_x, self.offset_y
        bs = self.board_size

        # 1. Classic Red MONOPOLY Banner in Top-Left interior
        logo_rect = pygame.Rect(ox + 115, oy + 125, 135, 38)
        pygame.draw.rect(self.surface, (20, 25, 35), logo_rect.move(2, 3), border_radius=6)
        pygame.draw.rect(self.surface, (220, 35, 45), logo_rect, border_radius=6)
        pygame.draw.rect(self.surface, (255, 215, 0), logo_rect, width=2, border_radius=6)
        t_logo = self.font_victory_title.render("MONOPOLY", True, (255, 255, 255))
        self.surface.blit(t_logo, (logo_rect.centerx - t_logo.get_width() // 2, logo_rect.centery - t_logo.get_height() // 2))

        # 2. Chance Card Deck Slot in Top-Right interior
        chance_slot = pygame.Rect(ox + bs - 245, oy + 125, 125, 72)
        pygame.draw.rect(self.surface, (255, 248, 238), chance_slot, border_radius=8)
        pygame.draw.rect(self.surface, (230, 126, 34), chance_slot, width=2, border_radius=8)
        inner_ch = chance_slot.inflate(-6, -6)
        pygame.draw.rect(self.surface, (245, 180, 110), inner_ch, width=1, border_radius=6)
        ch_t1 = self.font_badge.render("? CHANCE ?", True, (230, 126, 34))
        ch_t2 = self.font_corner_sub.render("DRAW DECK", True, TEXT_MUTED)
        self.surface.blit(ch_t1, (chance_slot.centerx - ch_t1.get_width() // 2, chance_slot.y + 15))
        self.surface.blit(ch_t2, (chance_slot.centerx - ch_t2.get_width() // 2, chance_slot.y + 38))

        # 3. Community Chest Card Deck Slot in Bottom-Left interior
        chest_slot = pygame.Rect(ox + 115, oy + bs - 195, 130, 72)
        pygame.draw.rect(self.surface, (245, 250, 255), chest_slot, border_radius=8)
        pygame.draw.rect(self.surface, (41, 128, 185), chest_slot, width=2, border_radius=8)
        inner_cs = chest_slot.inflate(-6, -6)
        pygame.draw.rect(self.surface, (140, 190, 235), inner_cs, width=1, border_radius=6)
        cs_t1 = self.font_badge.render("COMMUNITY CHEST", True, (41, 128, 185))
        cs_t2 = self.font_corner_sub.render("DRAW DECK", True, TEXT_MUTED)
        self.surface.blit(cs_t1, (chest_slot.centerx - cs_t1.get_width() // 2, chest_slot.y + 15))
        self.surface.blit(cs_t2, (chest_slot.centerx - cs_t2.get_width() // 2, chest_slot.y + 38))

    def _render_corner_tile(self, tile_id: int, rect: pygame.Rect, core: MonopolyCore):
        """Draws rich detailed corner tiles (GO, Jail, Free Parking, Go To Jail)."""
        if tile_id == 0:  # GO
            arrow_rect = pygame.Rect(rect.x + 8, rect.y + 6, rect.width - 16, 24)
            pygame.draw.rect(self.surface, (231, 76, 60), arrow_rect, border_radius=4)
            t_go = self.font_victory_title.render("GO ->", True, (255, 255, 255))
            self.surface.blit(t_go, (arrow_rect.centerx - t_go.get_width() // 2, arrow_rect.centery - t_go.get_height() // 2))

            t_col = self.font_corner_title.render("COLLECT $200", True, (46, 125, 50))
            self.surface.blit(t_col, (rect.centerx - t_col.get_width() // 2, rect.y + 34))
            t_pass = self.font_corner_sub.render("AS YOU PASS", True, TEXT_MUTED)
            self.surface.blit(t_pass, (rect.centerx - t_pass.get_width() // 2, rect.y + 48))

        elif tile_id == 10:  # JAIL
            cell_w, cell_h = 52, 52
            cell_rect = pygame.Rect(rect.right - cell_w, rect.y, cell_w, cell_h)
            pygame.draw.rect(self.surface, (220, 140, 40), cell_rect)
            pygame.draw.rect(self.surface, JAIL_BAR_COLOR, cell_rect, width=2)
            for bx in range(cell_rect.x + 8, cell_rect.right - 4, 10):
                pygame.draw.line(self.surface, JAIL_BAR_COLOR, (bx, cell_rect.y), (bx, cell_rect.bottom), 2)

            t_jail = self.font_corner_sub.render("IN JAIL", True, (255, 255, 255))
            self.surface.blit(t_jail, (cell_rect.centerx - t_jail.get_width() // 2, cell_rect.y + 6))

            t_vis1 = self.font_corner_sub.render("JUST", True, TEXT_COLOR)
            t_vis2 = self.font_corner_sub.render("VISITING", True, TEXT_COLOR)
            self.surface.blit(t_vis1, (rect.x + 8, rect.bottom - 32))
            self.surface.blit(t_vis2, (rect.x + 8, rect.bottom - 18))

        elif tile_id == 20:  # FREE PARKING
            emblem_rect = pygame.Rect(rect.centerx - 16, rect.y + 12, 32, 32)
            pygame.draw.rect(self.surface, (231, 76, 60), emblem_rect, border_radius=6)
            p_surf = self.font_victory_title.render("P", True, (255, 255, 255))
            self.surface.blit(p_surf, (emblem_rect.centerx - p_surf.get_width() // 2, emblem_rect.centery - p_surf.get_height() // 2))

            t_fp1 = self.font_corner_title.render("FREE", True, TEXT_COLOR)
            t_fp2 = self.font_corner_title.render("PARKING", True, TEXT_COLOR)
            self.surface.blit(t_fp1, (rect.centerx - t_fp1.get_width() // 2, rect.bottom - 36))
            self.surface.blit(t_fp2, (rect.centerx - t_fp2.get_width() // 2, rect.bottom - 22))

        elif tile_id == 30:  # GO TO JAIL
            badge_rect = pygame.Rect(rect.centerx - 16, rect.y + 12, 32, 32)
            pygame.draw.rect(self.surface, (52, 73, 94), badge_rect, border_radius=6)
            pygame.draw.circle(self.surface, GOLD_COLOR, (badge_rect.centerx, badge_rect.centery), 9)
            pygame.draw.circle(self.surface, (52, 73, 94), (badge_rect.centerx, badge_rect.centery), 4)

            t_gj1 = self.font_corner_title.render("GO TO", True, (214, 48, 49))
            t_gj2 = self.font_corner_title.render("JAIL!", True, (214, 48, 49))
            self.surface.blit(t_gj1, (rect.centerx - t_gj1.get_width() // 2, rect.bottom - 36))
            self.surface.blit(t_gj2, (rect.centerx - t_gj2.get_width() // 2, rect.bottom - 22))

    def _draw_tile_icon(self, icon_type: str, cx: int, cy: int):
        """Draws clean geometric mini vector icons for Railroads, Chance, Chest, Tax, Utilities."""
        if icon_type == "rr":
            body = pygame.Rect(cx - 9, cy - 5, 18, 10)
            pygame.draw.rect(self.surface, (45, 52, 65), body, border_radius=2)
            pygame.draw.rect(self.surface, (45, 52, 65), (cx + 3, cy - 8, 4, 4))
            pygame.draw.circle(self.surface, (25, 25, 30), (cx - 5, cy + 6), 3)
            pygame.draw.circle(self.surface, (25, 25, 30), (cx + 5, cy + 6), 3)

        elif icon_type == "chance":
            pts = [(cx, cy - 9), (cx + 9, cy), (cx, cy + 9), (cx - 9, cy)]
            pygame.draw.polygon(self.surface, (230, 126, 34), pts)
            q_surf = self.font_corner_sub.render("?", True, (255, 255, 255))
            self.surface.blit(q_surf, (cx - q_surf.get_width() // 2, cy - q_surf.get_height() // 2))

        elif icon_type == "chest":
            c_rect = pygame.Rect(cx - 9, cy - 5, 18, 10)
            pygame.draw.rect(self.surface, (41, 128, 185), c_rect, border_radius=2)
            pygame.draw.line(self.surface, GOLD_COLOR, (cx, cy - 5), (cx, cy + 4), 2)

        elif icon_type == "tax":
            pts = [(cx, cy - 8), (cx + 8, cy - 2), (cx, cy + 8), (cx - 8, cy - 2)]
            pygame.draw.polygon(self.surface, (142, 68, 173), pts)

        elif icon_type == "utility_elec":
            pygame.draw.circle(self.surface, (241, 196, 15), (cx, cy - 3), 5)
            pygame.draw.rect(self.surface, (120, 120, 130), (cx - 3, cy + 2, 6, 3))

        elif icon_type == "utility_water":
            pts = [(cx, cy - 7), (cx + 5, cy + 2), (cx, cy + 6), (cx - 5, cy + 2)]
            pygame.draw.polygon(self.surface, (52, 152, 219), pts)

    def _render_buildings(self, tile_id: int, rect: pygame.Rect, houses: int):
        """Draws 3D architectural house cottages (green) or grand hotel (crimson) with roof peaks."""
        if houses == 5:
            # Hotel: 28x15 3D building block
            if 0 < tile_id < 10:    # Bottom: right below header band
                hx, hy = rect.centerx - 14, rect.y + 24
            elif 10 < tile_id < 20: # Left: left of header band
                hx, hy = rect.right - 38, rect.centery - 7
            elif 20 < tile_id < 30: # Top: above bottom header band
                hx, hy = rect.centerx - 14, rect.bottom - 38
            else:                   # Right: right of left header band
                hx, hy = rect.x + 24, rect.centery - 7

            hotel_rect = pygame.Rect(hx, hy, 28, 15)
            pygame.draw.rect(self.surface, (20, 25, 35), hotel_rect.move(1, 2), border_radius=3)
            pygame.draw.rect(self.surface, HOTEL_COLOR, hotel_rect, border_radius=3)
            pygame.draw.rect(self.surface, (255, 130, 130), hotel_rect, width=1, border_radius=3)
            pygame.draw.line(self.surface, (255, 215, 0), (hotel_rect.x + 2, hotel_rect.y + 1), (hotel_rect.right - 2, hotel_rect.y + 1), 2)
            h_text = self.font_corner_sub.render("HOTEL", True, (255, 255, 255))
            self.surface.blit(h_text, (hotel_rect.centerx - h_text.get_width() // 2, hotel_rect.centery - h_text.get_height() // 2))
        else:
            # 1 to 4 Houses: 3D cottages with peaked triangular roofs
            house_w, house_h = 8, 9
            spacing = 9
            total_w = houses * spacing

            if 0 < tile_id < 10:  # Bottom: horizontal row under header band
                start_x = rect.centerx - total_w // 2 + 1
                base_y = rect.y + 24
                orient = "h"
            elif 10 < tile_id < 20: # Left: vertical column next to header band
                start_y = rect.centery - total_w // 2 + 1
                base_x = rect.right - 34
                orient = "v"
            elif 20 < tile_id < 30: # Top: horizontal row above header band
                start_x = rect.centerx - total_w // 2 + 1
                base_y = rect.bottom - 34
                orient = "h"
            else:                   # Right: vertical column next to header band
                start_y = rect.centery - total_w // 2 + 1
                base_x = rect.x + 24
                orient = "v"

            for h_i in range(houses):
                if orient == "h":
                    cx = start_x + h_i * spacing
                    cy = base_y
                else:
                    cx = base_x
                    cy = start_y + h_i * spacing

                h_rect = pygame.Rect(cx, cy + 3, house_w, house_h - 3)
                pygame.draw.rect(self.surface, (20, 25, 35), h_rect.move(1, 1), border_radius=1)
                pygame.draw.rect(self.surface, HOUSE_COLOR, h_rect, border_radius=1)
                roof_pts = [(cx, cy + 3), (cx + house_w // 2, cy), (cx + house_w, cy + 3)]
                pygame.draw.polygon(self.surface, (46, 204, 113), roof_pts)
                pygame.draw.polygon(self.surface, (20, 70, 30), roof_pts, width=1)

    def _render_tokens(self, core: MonopolyCore):
        """Renders 4 distinct 3D player tokens with initials, drop shadows, and hopping arcs."""
        active_player = core.active_decision_player

        for p_id in range(4):
            if core.bankrupt[p_id]:
                continue

            actual_pos = core.position[p_id]
            is_active = (p_id == active_player and core.phase != PHASE_GAME_OVER)

            # Interpolate coordinates if in middle of hopping
            if self.hop_progress[p_id] < 1.0:
                t = self.hop_progress[p_id]
                f_tile = self.current_hop_from[p_id]
                t_tile = self.current_hop_to[p_id]
                p1 = self.get_token_tile_coords(p_id, f_tile, in_jail=(f_tile == 10 and core.in_jail[p_id]))
                p2 = self.get_token_tile_coords(p_id, t_tile, in_jail=(t_tile == 10 and core.in_jail[p_id]))

                ground_x = (1.0 - t) * p1[0] + t * p2[0]
                ground_y = (1.0 - t) * p1[1] + t * p2[1]
                jump_arc = -15.0 * math.sin(math.pi * t)
                tx = int(ground_x)
                ty = int(ground_y + jump_arc)
                gx = int(ground_x)
                gy = int(ground_y)
            else:
                p_coord = self.get_token_tile_coords(p_id, actual_pos, in_jail=core.in_jail[p_id])
                tx, ty = p_coord
                gx, gy = p_coord

            color = PLAYER_COLORS[p_id]

            # 1. Active player pulsing golden halo
            if is_active:
                pulse_r = 16 + int(math.sin(self.animation_tick * 0.15) * 3)
                pygame.draw.circle(self.surface, GOLD_COLOR, (tx, ty), pulse_r, width=2)
                pygame.draw.circle(self.surface, (255, 255, 255), (tx, ty), pulse_r + 2, width=1)

            # 2. Ground shadow (stays on board surface during hop)
            shadow_surf = pygame.Surface((30, 30), pygame.SRCALPHA)
            pygame.draw.ellipse(shadow_surf, (15, 20, 30, 140), pygame.Rect(3, 8, 24, 14))
            self.surface.blit(shadow_surf, (gx - 15, gy - 8))

            # 3. 3D Token body: 24px diameter with metallic dark rim
            pygame.draw.circle(self.surface, (25, 30, 42), (tx, ty), 12)
            pygame.draw.circle(self.surface, color, (tx, ty), 10)
            # Specular 3D highlight gloss
            pygame.draw.circle(self.surface, (255, 255, 255), (tx - 3, ty - 3), 3)

            # 4. Bold white centered initial letter
            letter = PLAYER_INITIALS[p_id]
            t_surf = self.font_token.render(letter, True, (255, 255, 255))
            self.surface.blit(t_surf, (tx - t_surf.get_width() // 2, ty - t_surf.get_height() // 2))

    def _render_center_hud(self, core: MonopolyCore):
        """Renders center board display contextually (Default turn, Buy/Auction, Auction, Trade, Victory, Cards)."""
        cx = self.offset_x + self.board_size // 2
        cy = self.offset_y + self.board_size // 2

        # ----------------- PHASE: GAME OVER (VICTORY CHAMPION) -----------------
        if core.phase == PHASE_GAME_OVER:
            hud_box = pygame.Rect(cx - 200, cy - 130, 400, 260)
            pygame.draw.rect(self.surface, (244, 246, 250), hud_box, border_radius=14)
            pygame.draw.rect(self.surface, (195, 202, 215), hud_box, width=2, border_radius=14)
            self._render_victory_hud(core, hud_box, cx, cy)
            return

        # ----------------- PHASE: AUCTION -----------------
        if core.phase == PHASE_AUCTION and core.auction_property is not None:
            hud_box = pygame.Rect(cx - 200, cy - 130, 400, 260)
            pygame.draw.rect(self.surface, (244, 246, 250), hud_box, border_radius=14)
            pygame.draw.rect(self.surface, (195, 202, 215), hud_box, width=2, border_radius=14)
            self._render_auction_hud(core, hud_box, cx, cy)
            return

        # ----------------- PHASE: TRADE PROPOSAL -----------------
        if core.phase == PHASE_TRADE_RESPONSE and getattr(core, "pending_trade", None) is not None:
            hud_box = pygame.Rect(cx - 200, cy - 130, 400, 260)
            pygame.draw.rect(self.surface, (244, 246, 250), hud_box, border_radius=14)
            pygame.draw.rect(self.surface, (195, 202, 215), hud_box, width=2, border_radius=14)
            self._render_trade_hud(core, hud_box, cx, cy)
            return

        # ----------------- PHASE: BUY OR AUCTION (TITLE DEED CARD DISPLAY) -----------------
        if core.phase == PHASE_BUY_OR_AUCTION and core.landed_property is not None:
            self._render_buy_or_auction_hud(core, cx, cy)
            return

        # ----------------- CARD DRAWN DISPLAY (CHANCE / COMMUNITY CHEST) -----------------
        if getattr(core, "last_card_drawn", None) is not None and core.position[core.current_player] in (2, 7, 17, 22, 33, 36):
            self._render_drawn_card_hud(core, cx, cy)
            return

        # ----------------- DEFAULT TURN HUD -----------------
        hud_box = pygame.Rect(cx - 205, cy - 128, 410, 256)
        pygame.draw.rect(self.surface, (20, 25, 35), hud_box.move(2, 4), border_radius=14)
        pygame.draw.rect(self.surface, (246, 248, 252), hud_box, border_radius=14)
        pygame.draw.rect(self.surface, (190, 198, 212), hud_box, width=2, border_radius=14)

        # Title
        t_surf = self.font_center_title.render("TOURNAMENT MARL MONOPOLY", True, (28, 36, 50))
        self.surface.blit(t_surf, (cx - t_surf.get_width() // 2, hud_box.y + 14))

        # Round and Turn step
        round_str = f"Round: {core.round_count} (Unbounded Survival)   |   Turn Step: {core.turn_step}" if core.max_rounds is None else f"Round: {core.round_count}/{core.max_rounds}   |   Turn Step: {core.turn_step}"
        r_surf = self.font_center_sub.render(round_str, True, TEXT_MUTED)
        self.surface.blit(r_surf, (cx - r_surf.get_width() // 2, hud_box.y + 38))

        # Active Player Badge
        p_act = core.active_decision_player
        p_name = core.get_player_name(p_act)
        p_col = PLAYER_COLORS[p_act]
        phase_names = ['Roll Dice', 'Buy / Auction', 'Open Auction', 'Jail Decision', 'Property Manage', 'Trade Deal', 'Game Over']
        p_phase = phase_names[min(core.phase, 6)]

        act_text = f"Turn: {p_name} ({p_phase})"
        act_surf = self.font_badge.render(act_text, True, (255, 255, 255))
        badge_w = max(230, act_surf.get_width() + 24)
        badge_rect = pygame.Rect(cx - badge_w // 2, hud_box.y + 64, badge_w, 25)
        pygame.draw.rect(self.surface, p_col, badge_rect, border_radius=6)
        self.surface.blit(act_surf, (cx - act_surf.get_width() // 2, badge_rect.y + 4))

        # 3D Animated Dice Display
        d1, d2 = self.dice_roll_values
        is_rolling = (self.dice_roll_timer > 0)
        self._render_dice(cx - 55, hud_box.y + 102, d1, is_rolling=is_rolling, roll_tick=self.animation_tick)
        self._render_dice(cx + 12, hud_box.y + 102, d2, is_rolling=is_rolling, roll_tick=self.animation_tick + 5)

        # Consecutive doubles banner or paused status banner
        if getattr(self, "is_paused", False) and core.phase != PHASE_GAME_OVER:
            pause_txt = "⏸ PAUSED (Press Space or PLAY to Resume)" if core.turn_step > 0 else "⏸ READY (Press Space or PLAY to Start)"
            p_surf = self.font_badge.render(pause_txt, True, (211, 84, 0))
            self.surface.blit(p_surf, (cx - p_surf.get_width() // 2, hud_box.y + 154))
        elif getattr(core, "consecutive_doubles", 0) > 0:
            dbl_txt = f"Doubles Rolled! ({core.consecutive_doubles}/3) -> Extra Roll"
            d_surf = self.font_corner_sub.render(dbl_txt, True, (214, 48, 49))
            self.surface.blit(d_surf, (cx - d_surf.get_width() // 2, hud_box.y + 156))

        # Surviving Agents Counter
        active_count = sum(1 for b in core.bankrupt if not b)
        surv_txt = f"Survival Status: {active_count} of 4 Agents Remaining"
        s_surf = self.font_center_sub.render(surv_txt, True, (46, 125, 50) if active_count > 1 else (210, 40, 40))
        self.surface.blit(s_surf, (cx - s_surf.get_width() // 2, hud_box.bottom - 46))

        # Latest event snippet
        if core.event_log:
            last_ev = core.event_log[-1]
            if len(last_ev) > 52:
                last_ev = last_ev[:49] + "..."
            ev_surf = self.font_price.render(last_ev, True, (75, 85, 100))
            self.surface.blit(ev_surf, (cx - ev_surf.get_width() // 2, hud_box.bottom - 22))

    def _render_buy_or_auction_hud(self, core: MonopolyCore, cx: int, cy: int):
        """Displays authentic Title Deed Card on left and buy/auction options on right."""
        prop_id = core.landed_property
        p = core.active_decision_player
        p_name = core.get_player_name(p)
        tile = TILES_DATA[prop_id]
        price = core.get_property_price(prop_id, tile["price"])

        hud_w, hud_h = 490, 270
        hud_box = pygame.Rect(cx - hud_w // 2, cy - hud_h // 2, hud_w, hud_h)
        pygame.draw.rect(self.surface, (244, 246, 250), hud_box, border_radius=12)
        pygame.draw.rect(self.surface, (195, 202, 215), hud_box, width=2, border_radius=12)

        # Left: Authentic Title Deed Card
        card_rect = pygame.Rect(hud_box.x + 16, hud_box.y + 12, 195, 246)
        self._render_title_deed_card(prop_id, core, card_rect)

        # Right: Purchase / Auction Decision Panel
        rx = card_rect.right + 20
        ry = hud_box.y + 16

        t_h = self.font_card_title.render("UNOWNED PROPERTY LANDING", True, (30, 40, 55))
        self.surface.blit(t_h, (rx, ry))

        buyer_txt = f"Buyer: {p_name} (Cash: ${core.cash[p]:,})"
        t_b = self.font_center_sub.render(buyer_txt, True, PLAYER_COLORS[p])
        self.surface.blit(t_b, (rx, ry + 24))

        pr_txt = f"List Purchase Price: ${price}"
        t_p = self.font_card_title.render(pr_txt, True, (46, 125, 50))
        self.surface.blit(t_p, (rx, ry + 46))

        # Decision Options
        opt1_box = pygame.Rect(rx, ry + 78, 230, 44)
        opt2_box = pygame.Rect(rx, ry + 132, 230, 44)
        can_afford = (core.cash[p] >= price)

        pygame.draw.rect(self.surface, (230, 245, 235) if can_afford else (245, 230, 230), opt1_box, border_radius=6)
        pygame.draw.rect(self.surface, (46, 125, 50) if can_afford else (180, 50, 50), opt1_box, width=1, border_radius=6)
        o1_t1 = self.font_badge.render("[1] BUY DIRECTLY", True, (46, 125, 50) if can_afford else (180, 50, 50))
        o1_t2 = self.font_price.render(f"Pay ${price} and claim Title Deed", True, TEXT_MUTED)
        self.surface.blit(o1_t1, (opt1_box.x + 10, opt1_box.y + 6))
        self.surface.blit(o1_t2, (opt1_box.x + 10, opt1_box.y + 24))

        pygame.draw.rect(self.surface, (255, 245, 230), opt2_box, border_radius=6)
        pygame.draw.rect(self.surface, (215, 100, 20), opt2_box, width=1, border_radius=6)
        o2_t1 = self.font_badge.render("[2] DECLINE -> OPEN AUCTION", True, (215, 100, 20))
        o2_t2 = self.font_price.render("Triggers open bidding for all agents", True, TEXT_MUTED)
        self.surface.blit(o2_t1, (opt2_box.x + 10, opt2_box.y + 6))
        self.surface.blit(o2_t2, (opt2_box.x + 10, opt2_box.y + 24))

        hint_txt = "Awaiting Actor Policy decision..."
        self.surface.blit(self.font_price.render(hint_txt, True, TEXT_MUTED), (rx, hud_box.bottom - 28))

    def _render_drawn_card_hud(self, core: MonopolyCore, cx: int, cy: int):
        """Displays drawn Chance or Community Chest card in center HUD."""
        card = core.last_card_drawn
        is_chance = (card["id"] < 100)

        card_w, card_h = 360, 210
        card_rect = pygame.Rect(cx - card_w // 2, cy - card_h // 2, card_w, card_h)

        if is_chance:
            self._render_chance_card(card, card_rect)
        else:
            self._render_community_chest_card(card, card_rect)

    def _render_title_deed_card(self, tile_id: int, core: MonopolyCore, rect: pygame.Rect):
        """Renders authentic official Title Deed Card with color banner and full rent schedule."""
        tile = TILES_DATA[tile_id]
        t_type = tile.get("type", TILE_STREET)

        pygame.draw.rect(self.surface, (255, 255, 255), rect, border_radius=6)
        pygame.draw.rect(self.surface, (45, 52, 65), rect, width=2, border_radius=6)
        inner_rect = rect.inflate(-6, -6)
        pygame.draw.rect(self.surface, (215, 220, 230), inner_rect, width=1, border_radius=4)

        if t_type == TILE_STREET:
            grp = tile.get("color")
            header_col = COLOR_HEADER_MAP.get(grp, (60, 70, 85))

            hdr_rect = pygame.Rect(inner_rect.x, inner_rect.y, inner_rect.width, 38)
            pygame.draw.rect(self.surface, header_col, hdr_rect, border_top_left_radius=4, border_top_right_radius=4)
            pygame.draw.rect(self.surface, (45, 52, 65), hdr_rect, width=1)

            t_sub = self.font_deed_sub.render("TITLE DEED", True, (255, 255, 255))
            self.surface.blit(t_sub, (hdr_rect.centerx - t_sub.get_width() // 2, hdr_rect.y + 4))

            p_title = tile["name"].upper().replace(" AVENUE", " AVE").replace(" PLACE", " PL")
            t_name = self.font_deed_header.render(p_title, True, (255, 255, 255))
            self.surface.blit(t_name, (hdr_rect.centerx - t_name.get_width() // 2, hdr_rect.y + 18))

            rents = tile.get("rent", [0, 0, 0, 0, 0, 0])
            ry = hdr_rect.bottom + 6

            r_base = self.font_deed_rent.render(f"RENT ${rents[0]}", True, TEXT_COLOR)
            self.surface.blit(r_base, (inner_rect.centerx - r_base.get_width() // 2, ry))
            ry += 18

            rent_rows = [
                ("With 1 House", f"${rents[1]}"),
                ("With 2 Houses", f"${rents[2]}"),
                ("With 3 Houses", f"${rents[3]}"),
                ("With 4 Houses", f"${rents[4]}"),
                ("With HOTEL", f"${rents[5]}"),
            ]

            for label, val in rent_rows:
                col = (214, 48, 49) if "HOTEL" in label else TEXT_COLOR
                fnt = self.font_deed_rent if "HOTEL" in label else self.font_deed_row
                l_surf = fnt.render(label, True, col)
                v_surf = fnt.render(val, True, col)
                self.surface.blit(l_surf, (inner_rect.x + 12, ry))
                self.surface.blit(v_surf, (inner_rect.right - 12 - v_surf.get_width(), ry))
                ry += 15

            pygame.draw.line(self.surface, (200, 205, 215), (inner_rect.x + 8, ry + 2), (inner_rect.right - 8, ry + 2), 1)
            ry += 6

            h_cost = core.get_property_house_cost(tile_id, tile.get("house_cost", 50))
            mtg = core.get_property_mortgage(tile_id, tile.get("mortgage", 30))

            mtg_txt = f"Mortgage Value ${mtg}"
            h_txt = f"Houses cost ${h_cost} each"
            ht_txt = f"Hotels, ${h_cost} plus 4 houses"

            self.surface.blit(self.font_deed_footer.render(mtg_txt, True, TEXT_MUTED), (inner_rect.centerx - 45, ry))
            ry += 13
            self.surface.blit(self.font_deed_footer.render(h_txt, True, TEXT_MUTED), (inner_rect.centerx - 45, ry))
            ry += 13
            self.surface.blit(self.font_deed_footer.render(ht_txt, True, TEXT_MUTED), (inner_rect.centerx - 55, ry))

            owner = core.property_owner[tile_id]
            owner_name = core.get_player_name(owner) if owner != -1 else "Unowned (Bank)"
            o_col = PLAYER_COLORS[owner] if owner != -1 else TEXT_MUTED
            status_txt = f"Owner: {owner_name}"
            s_surf = self.font_deed_sub.render(status_txt, True, o_col)
            self.surface.blit(s_surf, (inner_rect.centerx - s_surf.get_width() // 2, inner_rect.bottom - 18))

        elif t_type == TILE_RAILROAD:
            hdr_rect = pygame.Rect(inner_rect.x, inner_rect.y, inner_rect.width, 38)
            pygame.draw.rect(self.surface, (45, 52, 65), hdr_rect, border_top_left_radius=4, border_top_right_radius=4)

            t_sub = self.font_deed_sub.render("RAILROAD", True, (255, 255, 255))
            self.surface.blit(t_sub, (hdr_rect.centerx - t_sub.get_width() // 2, hdr_rect.y + 4))
            t_name = self.font_deed_header.render(tile["name"].upper(), True, (255, 255, 255))
            self.surface.blit(t_name, (hdr_rect.centerx - t_name.get_width() // 2, hdr_rect.y + 18))

            ry = hdr_rect.bottom + 12
            rr_rows = [
                ("Rent", "$25"),
                ("If 2 Railroads owned", "$50"),
                ("If 3 Railroads owned", "$100"),
                ("If 4 Railroads owned", "$200"),
            ]
            for label, val in rr_rows:
                l_surf = self.font_deed_row.render(label, True, TEXT_COLOR)
                v_surf = self.font_deed_row.render(val, True, TEXT_COLOR)
                self.surface.blit(l_surf, (inner_rect.x + 12, ry))
                self.surface.blit(v_surf, (inner_rect.right - 12 - v_surf.get_width(), ry))
                ry += 18

            pygame.draw.line(self.surface, (200, 205, 215), (inner_rect.x + 8, ry + 4), (inner_rect.right - 8, ry + 4), 1)
            ry += 12
            mtg_rr = core.get_property_mortgage(tile_id, tile.get("mortgage", 100))
            mtg_txt = f"Mortgage Value ${mtg_rr}"
            self.surface.blit(self.font_deed_footer.render(mtg_txt, True, TEXT_MUTED), (inner_rect.centerx - 45, ry))

            owner = core.property_owner[tile_id]
            owner_name = core.get_player_name(owner) if owner != -1 else "Unowned (Bank)"
            o_col = PLAYER_COLORS[owner] if owner != -1 else TEXT_MUTED
            self.surface.blit(self.font_deed_sub.render(f"Owner: {owner_name}", True, o_col), (inner_rect.centerx - 45, inner_rect.bottom - 18))

        elif t_type == TILE_UTILITY:
            hdr_rect = pygame.Rect(inner_rect.x, inner_rect.y, inner_rect.width, 38)
            pygame.draw.rect(self.surface, (215, 140, 20), hdr_rect, border_top_left_radius=4, border_top_right_radius=4)

            t_sub = self.font_deed_sub.render("UTILITY", True, (255, 255, 255))
            self.surface.blit(t_sub, (hdr_rect.centerx - t_sub.get_width() // 2, hdr_rect.y + 4))
            t_name = self.font_deed_header.render(tile["name"].upper(), True, (255, 255, 255))
            self.surface.blit(t_name, (hdr_rect.centerx - t_name.get_width() // 2, hdr_rect.y + 18))

            ry = hdr_rect.bottom + 14
            u_text1 = "If 1 Utility is owned, rent is"
            u_text2 = "4 times amount shown on dice."
            u_text3 = "If both Utilities are owned, rent is"
            u_text4 = "10 times amount shown on dice."

            for line in [u_text1, u_text2, "", u_text3, u_text4]:
                if line:
                    l_s = self.font_deed_row.render(line, True, TEXT_COLOR)
                    self.surface.blit(l_s, (inner_rect.centerx - l_s.get_width() // 2, ry))
                ry += 14

            pygame.draw.line(self.surface, (200, 205, 215), (inner_rect.x + 8, ry + 4), (inner_rect.right - 8, ry + 4), 1)
            ry += 10
            mtg_util = core.get_property_mortgage(tile_id, tile.get("mortgage", 75))
            mtg_txt = f"Mortgage Value ${mtg_util}"
            self.surface.blit(self.font_deed_footer.render(mtg_txt, True, TEXT_MUTED), (inner_rect.centerx - 45, ry))

            owner = core.property_owner[tile_id]
            owner_name = core.get_player_name(owner) if owner != -1 else "Unowned (Bank)"
            o_col = PLAYER_COLORS[owner] if owner != -1 else TEXT_MUTED
            self.surface.blit(self.font_deed_sub.render(f"Owner: {owner_name}", True, o_col), (inner_rect.centerx - 45, inner_rect.bottom - 18))

    def _render_chance_card(self, card: Dict[str, Any], rect: pygame.Rect):
        """Renders authentic official Chance Card (Vibrant Orange & Cream)."""
        pygame.draw.rect(self.surface, (255, 248, 238), rect, border_radius=8)
        pygame.draw.rect(self.surface, (230, 126, 34), rect, width=3, border_radius=8)
        inner = rect.inflate(-8, -8)
        pygame.draw.rect(self.surface, (240, 160, 80), inner, width=1, border_radius=6)

        h_badge = pygame.Rect(inner.centerx - 60, inner.y + 10, 120, 24)
        pygame.draw.rect(self.surface, (230, 126, 34), h_badge, border_radius=4)
        h_txt = self.font_card_title.render("? CHANCE ?", True, (255, 255, 255))
        self.surface.blit(h_txt, (h_badge.centerx - h_txt.get_width() // 2, h_badge.centery - h_txt.get_height() // 2))

        title_txt = card["name"].upper()
        t_surf = self.font_card_title.render(title_txt, True, (45, 52, 65))
        self.surface.blit(t_surf, (inner.centerx - t_surf.get_width() // 2, inner.y + 48))

        desc = CARD_DESCRIPTIONS.get(card["id"], card["name"])
        words = desc.split()
        lines = []
        curr_line = []
        for w in words:
            curr_line.append(w)
            if len(" ".join(curr_line)) > 34:
                lines.append(" ".join(curr_line[:-1]))
                curr_line = [w]
        if curr_line:
            lines.append(" ".join(curr_line))

        dy = inner.y + 80
        for l in lines:
            l_s = self.font_card_body.render(l, True, (60, 70, 85))
            self.surface.blit(l_s, (inner.centerx - l_s.get_width() // 2, dy))
            dy += 18

        footer_txt = f"Tournament Chance Card • ID #{card['id']}"
        f_s = self.font_deed_footer.render(footer_txt, True, TEXT_MUTED)
        self.surface.blit(f_s, (inner.centerx - f_s.get_width() // 2, inner.bottom - 16))

    def _render_community_chest_card(self, card: Dict[str, Any], rect: pygame.Rect):
        """Renders authentic official Community Chest Card (Royal Blue & Golden Cream)."""
        pygame.draw.rect(self.surface, (254, 250, 232), rect, border_radius=8)
        pygame.draw.rect(self.surface, (41, 128, 185), rect, width=3, border_radius=8)
        inner = rect.inflate(-8, -8)
        pygame.draw.rect(self.surface, (100, 160, 220), inner, width=1, border_radius=6)

        h_badge = pygame.Rect(inner.centerx - 85, inner.y + 10, 170, 24)
        pygame.draw.rect(self.surface, (41, 128, 185), h_badge, border_radius=4)
        h_txt = self.font_card_title.render("COMMUNITY CHEST", True, (255, 255, 255))
        self.surface.blit(h_txt, (h_badge.centerx - h_txt.get_width() // 2, h_badge.centery - h_txt.get_height() // 2))

        title_txt = card["name"].upper()
        t_surf = self.font_card_title.render(title_txt, True, (45, 52, 65))
        self.surface.blit(t_surf, (inner.centerx - t_surf.get_width() // 2, inner.y + 48))

        desc = CARD_DESCRIPTIONS.get(card["id"], card["name"])
        words = desc.split()
        lines = []
        curr_line = []
        for w in words:
            curr_line.append(w)
            if len(" ".join(curr_line)) > 34:
                lines.append(" ".join(curr_line[:-1]))
                curr_line = [w]
        if curr_line:
            lines.append(" ".join(curr_line))

        dy = inner.y + 80
        for l in lines:
            l_s = self.font_card_body.render(l, True, (60, 70, 85))
            self.surface.blit(l_s, (inner.centerx - l_s.get_width() // 2, dy))
            dy += 18

        footer_txt = f"Tournament Community Chest Card • ID #{card['id']}"
        f_s = self.font_deed_footer.render(footer_txt, True, TEXT_MUTED)
        self.surface.blit(f_s, (inner.centerx - f_s.get_width() // 2, inner.bottom - 16))

    def _render_inspection_overlay(self, core: MonopolyCore):
        """Renders full card inspection dialog when user clicks any tile on board."""
        t_id = self.inspect_tile_id
        if t_id is None:
            return

        cx = self.offset_x + self.board_size // 2
        cy = self.offset_y + self.board_size // 2

        backdrop = pygame.Surface((self.board_size, self.board_size), pygame.SRCALPHA)
        backdrop.fill((10, 12, 18, 140))
        self.surface.blit(backdrop, (self.offset_x, self.offset_y))

        tile = TILES_DATA[t_id]
        t_type = tile.get("type", TILE_SPECIAL)

        if t_type in (TILE_STREET, TILE_RAILROAD, TILE_UTILITY):
            card_w, card_h = 240, 310
            card_rect = pygame.Rect(cx - card_w // 2, cy - card_h // 2, card_w, card_h)
            self._render_title_deed_card(t_id, core, card_rect)
        elif t_type == TILE_CARD:
            card_w, card_h = 360, 210
            card_rect = pygame.Rect(cx - card_w // 2, cy - card_h // 2, card_w, card_h)
            deck_name = tile.get("deck", "chance")
            if deck_name == "chance":
                sample_card = CHANCE_CARDS[0] if not core.deck_manager.chance_queue else core.deck_manager.chance_queue[0]
                self._render_chance_card(sample_card, card_rect)
            else:
                sample_card = COMMUNITY_CHEST_CARDS[0] if not core.deck_manager.chest_queue else core.deck_manager.chest_queue[0]
                self._render_community_chest_card(sample_card, card_rect)
        else:
            card_w, card_h = 260, 200
            card_rect = pygame.Rect(cx - card_w // 2, cy - card_h // 2, card_w, card_h)
            pygame.draw.rect(self.surface, (255, 255, 255), card_rect, border_radius=8)
            pygame.draw.rect(self.surface, (45, 52, 65), card_rect, width=2, border_radius=8)
            t_h = self.font_card_title.render(tile["name"].upper(), True, (30, 40, 55))
            self.surface.blit(t_h, (card_rect.centerx - t_h.get_width() // 2, card_rect.y + 20))
            if t_type == TILE_TAX:
                desc = f"Mandatory Tournament Tax: Pay ${tile['tax_amount']} to Bank."
            elif t_id == 0:
                desc = "GO: Collect $200 salary as you pass."
            elif t_id == 10:
                desc = "JAIL: Roll doubles, pay $50 fine, or use GOOJF card."
            elif t_id == 20:
                desc = "FREE PARKING: Rest area, no rent or action penalty."
            else:
                desc = "GO TO JAIL: Move directly to Jail. Do not pass GO."

            d_s = self.font_card_body.render(desc, True, TEXT_MUTED)
            self.surface.blit(d_s, (card_rect.centerx - d_s.get_width() // 2, card_rect.centery - 10))

        close_rect = pygame.Rect(cx + 105, cy - 145, 22, 22)
        pygame.draw.rect(self.surface, (214, 48, 49), close_rect, border_radius=4)
        x_surf = self.font_corner_sub.render("X", True, (255, 255, 255))
        self.surface.blit(x_surf, (close_rect.centerx - x_surf.get_width() // 2, close_rect.centery - x_surf.get_height() // 2))

    def _render_auction_hud(self, core: MonopolyCore, hud_box: pygame.Rect, cx: int, cy: int):
        """Renders dedicated interactive auction block."""
        prop_id = core.auction_property
        tile = TILES_DATA[prop_id]
        p_name = tile["name"]

        h_surf = self.font_victory_sub.render("MANDATORY OPEN AUCTION", True, (215, 100, 20))
        self.surface.blit(h_surf, (cx - h_surf.get_width() // 2, hud_box.y + 14))

        color_grp = tile.get("color")
        band_col = COLOR_HEADER_MAP.get(color_grp, (70, 80, 95))
        p_badge = pygame.Rect(cx - 140, hud_box.y + 38, 280, 26)
        pygame.draw.rect(self.surface, band_col, p_badge, border_radius=4)
        eff_price = core.get_property_price(prop_id, tile["price"])
        t_prop = self.font_badge.render(f"{p_name} (List: ${eff_price})", True, (255, 255, 255))
        self.surface.blit(t_prop, (cx - t_prop.get_width() // 2, p_badge.y + 5))

        b_box = pygame.Rect(cx - 130, hud_box.y + 72, 260, 52)
        pygame.draw.rect(self.surface, (255, 255, 255), b_box, border_radius=8)
        pygame.draw.rect(self.surface, (230, 235, 245), b_box, width=2, border_radius=8)

        bidder = core.auction_highest_bidder
        b_name = core.get_player_name(bidder) if bidder != -1 else "No Bids Yet"
        b_col = PLAYER_COLORS[bidder] if bidder != -1 else TEXT_MUTED

        b_txt = self.font_victory_title.render(f"${core.auction_highest_bid}", True, (214, 48, 49) if core.auction_highest_bid > 0 else TEXT_MUTED)
        self.surface.blit(b_txt, (cx - b_txt.get_width() // 2, b_box.y + 5))
        by_txt = self.font_center_sub.render(f"High Bidder: {b_name}", True, b_col)
        self.surface.blit(by_txt, (cx - by_txt.get_width() // 2, b_box.y + 29))

        act_bidder = core.active_decision_player
        turn_txt = f"Awaiting Bid From: {core.get_player_name(act_bidder)}"
        t_surf = self.font_badge.render(turn_txt, True, PLAYER_COLORS[act_bidder])
        self.surface.blit(t_surf, (cx - t_surf.get_width() // 2, hud_box.y + 134))

        bx = cx - 110
        for b_id in range(4):
            is_active = b_id in core.auction_active_bidders
            b_pill = pygame.Rect(bx, hud_box.y + 158, 50, 20)
            p_c = PLAYER_COLORS[b_id] if is_active else (190, 195, 205)
            pygame.draw.rect(self.surface, p_c, b_pill, border_radius=4)
            p_txt = self.font_corner_sub.render(f"{PLAYER_INITIALS[b_id]} ({'BID' if is_active else 'OUT'})", True, (255, 255, 255))
            self.surface.blit(p_txt, (b_pill.centerx - p_txt.get_width() // 2, b_pill.centery - p_txt.get_height() // 2))
            bx += 60

        info_txt = "Discrete increments (+$1, +$10, +$50, +$100) or strategic Jump Bids"
        i_surf = self.font_price.render(info_txt, True, TEXT_MUTED)
        self.surface.blit(i_surf, (cx - i_surf.get_width() // 2, hud_box.bottom - 22))

    def _render_trade_hud(self, core: MonopolyCore, hud_box: pygame.Rect, cx: int, cy: int):
        """Renders bilateral trade negotiation deal card."""
        trade = core.pending_trade
        init = trade["initiator"]
        target = trade["target"]
        init_name = core.get_player_name(init)
        target_name = core.get_player_name(target)

        h_surf = self.font_victory_sub.render("BILATERAL TRADE PROPOSAL", True, (46, 125, 50))
        self.surface.blit(h_surf, (cx - h_surf.get_width() // 2, hud_box.y + 14))

        offer_p = TILES_DATA[trade["offer_prop"]]["name"] if trade["offer_prop"] < 40 else "None"
        req_p = TILES_DATA[trade["req_prop"]]["name"] if trade["req_prop"] < 40 else "None"

        card1 = pygame.Rect(cx - 170, hud_box.y + 42, 160, 92)
        card2 = pygame.Rect(cx + 10, hud_box.y + 42, 160, 92)
        pygame.draw.rect(self.surface, (255, 255, 255), card1, border_radius=6)
        pygame.draw.rect(self.surface, (255, 255, 255), card2, border_radius=6)
        pygame.draw.rect(self.surface, PLAYER_COLORS[init], card1, width=2, border_radius=6)
        pygame.draw.rect(self.surface, PLAYER_COLORS[target], card2, width=2, border_radius=6)

        o_h = self.font_badge.render(f"{init_name} OFFERS:", True, PLAYER_COLORS[init])
        self.surface.blit(o_h, (card1.x + 8, card1.y + 8))
        o_p = self.font_price.render(f"Prop: {offer_p[:14]}", True, TEXT_COLOR)
        self.surface.blit(o_p, (card1.x + 8, card1.y + 34))
        o_c = self.font_price.render(f"Cash: ${trade['offer_cash']}", True, (46, 125, 50))
        self.surface.blit(o_c, (card1.x + 8, card1.y + 56))

        r_h = self.font_badge.render(f"REQUESTS FROM {target_name}:", True, PLAYER_COLORS[target])
        self.surface.blit(r_h, (card2.x + 8, card2.y + 8))
        r_p = self.font_price.render(f"Prop: {req_p[:14]}", True, TEXT_COLOR)
        self.surface.blit(r_p, (card2.x + 8, card2.y + 34))
        r_c = self.font_price.render(f"Cash: ${trade['req_cash']}", True, (46, 125, 50))
        self.surface.blit(r_c, (card2.x + 8, card2.y + 56))

        status_txt = f"Awaiting Game-Theoretic Response from {target_name}..."
        st_surf = self.font_center_sub.render(status_txt, True, (40, 50, 70))
        self.surface.blit(st_surf, (cx - st_surf.get_width() // 2, hud_box.bottom - 46))

    def _render_victory_hud(self, core: MonopolyCore, hud_box: pygame.Rect, cx: int, cy: int):
        """Renders celebration championship banner when match concludes."""
        active = [pl for pl in range(4) if not core.bankrupt[pl]]
        if len(active) == 1:
            winner = active[0]
            w_name = core.get_player_name(winner)
            w_col = PLAYER_COLORS[winner]
            w_nw = core.get_player_net_worth(winner)
            sub_title = "LAST AGENT STANDING • SURVIVAL CHAMPION"
        else:
            net_worths = [core.get_player_net_worth(pl) if not core.bankrupt[pl] else -99999 for pl in range(4)]
            winner = int(np.argmax(net_worths))
            w_name = core.get_player_name(winner)
            w_col = PLAYER_COLORS[winner]
            w_nw = net_worths[winner]
            sub_title = "WINNER ON NET WORTH"

        t_surf = self.font_victory_title.render(f"*** {w_name.upper()} IS THE CHAMPION! ***", True, w_col)
        self.surface.blit(t_surf, (cx - t_surf.get_width() // 2, hud_box.y + 18))

        sub_surf = self.font_center_sub.render(sub_title, True, (190, 140, 20))
        self.surface.blit(sub_surf, (cx - sub_surf.get_width() // 2, hud_box.y + 48))

        w_card = pygame.Rect(cx - 150, hud_box.y + 74, 300, 72)
        pygame.draw.rect(self.surface, (255, 255, 255), w_card, border_radius=8)
        pygame.draw.rect(self.surface, w_col, w_card, width=2, border_radius=8)

        props_owned = sum(1 for p in PURCHASABLE_TILES if core.property_owner[p] == winner)
        stats1 = f"Final Net Worth: ${w_nw:,}   |   Properties: {props_owned}"
        stats2 = f"Surviving Rounds: {core.round_count}   |   Turn Step: {core.turn_step}"

        s1 = self.font_badge.render(stats1, True, (46, 125, 50))
        s2 = self.font_center_sub.render(stats2, True, TEXT_MUTED)
        self.surface.blit(s1, (cx - s1.get_width() // 2, w_card.y + 14))
        self.surface.blit(s2, (cx - s2.get_width() // 2, w_card.y + 40))

        rst_surf = self.font_center_sub.render("Press 'R' or click RESET on Dashboard to start new game", True, (60, 70, 85))
        self.surface.blit(rst_surf, (cx - rst_surf.get_width() // 2, hud_box.bottom - 36))

    def _render_dice(self, x: int, y: int, value: int, is_rolling: bool = False, roll_tick: int = 0):
        """Draws a clean 3D dice face with drop shadow, border bevel, and pips."""
        if is_rolling:
            wobble_x = int(math.sin(roll_tick * 2.3) * 3)
            wobble_y = int(math.cos(roll_tick * 2.8) * 3)
            x += wobble_x
            y += wobble_y

        dice_rect = pygame.Rect(x, y, 44, 44)
        pygame.draw.rect(self.surface, (140, 145, 160), dice_rect.move(2, 4), border_radius=8)
        pygame.draw.rect(self.surface, (255, 255, 255), dice_rect, border_radius=8)
        pygame.draw.line(self.surface, (245, 250, 255), (dice_rect.x + 4, dice_rect.y + 2), (dice_rect.right - 4, dice_rect.y + 2), 2)

        border_col = (241, 196, 15) if is_rolling else (80, 90, 105)
        pygame.draw.rect(self.surface, border_col, dice_rect, width=2, border_radius=8)

        cx, cy = dice_rect.centerx, dice_rect.centery
        dot_r = 3.5
        pip_col = (28, 32, 42)

        pip_coords = {
            1: [(cx, cy)],
            2: [(cx - 10, cy - 10), (cx + 10, cy + 10)],
            3: [(cx - 10, cy - 10), (cx, cy), (cx + 10, cy + 10)],
            4: [(cx - 10, cy - 10), (cx + 10, cy - 10), (cx - 10, cy + 10), (cx + 10, cy + 10)],
            5: [(cx - 10, cy - 10), (cx + 10, cy - 10), (cx, cy), (cx - 10, cy + 10), (cx + 10, cy + 10)],
            6: [(cx - 10, cy - 11), (cx + 10, cy - 11), (cx - 10, cy), (cx + 10, cy), (cx - 10, cy + 11), (cx + 10, cy + 11)],
        }

        for px, py in pip_coords.get(value, [(cx, cy)]):
            pygame.draw.circle(self.surface, pip_col, (px, py), dot_r)
            pygame.draw.circle(self.surface, (255, 255, 255), (px - 1, py - 1), 1)
