"""Interactive Pygame Renderer for 4-Agent Monopoly MARL.

Renders high-fidelity 11x11 Monopoly board, tile color banners, non-overlapping
player tokens, houses and hotels, dice roll animations, and center HUD.
"""

from typing import Dict, List, Tuple, Optional, Any
import pygame
import math
import numpy as np

from monopoly_marl.engine.board_data import (
    TILES_DATA, COLOR_GROUPS, PURCHASABLE_TILES,
    COLOR_BROWN, COLOR_LIGHT_BLUE, COLOR_PINK, COLOR_ORANGE,
    COLOR_RED, COLOR_YELLOW, COLOR_GREEN, COLOR_DARK_BLUE,
    TILE_STREET, TILE_RAILROAD, TILE_UTILITY, TILE_TAX, TILE_CARD, TILE_SPECIAL
)
from monopoly_marl.engine.monopoly_core import MonopolyCore, PHASE_AUCTION

# Color Palette
BG_COLOR = (24, 27, 34)
BOARD_BG = (235, 238, 242)
TILE_BORDER = (180, 185, 195)
TEXT_COLOR = (30, 35, 45)
TEXT_LIGHT = (220, 225, 235)

COLOR_HEADER_MAP = {
    COLOR_BROWN: (139, 69, 19),
    COLOR_LIGHT_BLUE: (135, 206, 250),
    COLOR_PINK: (218, 112, 214),
    COLOR_ORANGE: (255, 140, 0),
    COLOR_RED: (220, 20, 60),
    COLOR_YELLOW: (255, 215, 0),
    COLOR_GREEN: (34, 139, 34),
    COLOR_DARK_BLUE: (25, 25, 112),
}

PLAYER_COLORS = [
    (239, 71, 111),   # Player 0: Red
    (17, 138, 178),   # Player 1: Blue
    (6, 214, 160),    # Player 2: Green
    (255, 209, 102),  # Player 3: Yellow
]

HOUSE_COLOR = (46, 204, 113)  # Bright Green
HOTEL_COLOR = (231, 76, 60)   # Vivid Red


class MonopolyRenderer:
    """Renders the Monopoly 11x11 game board and center game state."""

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

        # Fonts
        pygame.font.init()
        self.font_tile = pygame.font.SysFont("Arial", 9, bold=True)
        self.font_price = pygame.font.SysFont("Arial", 9)
        self.font_center_title = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_center_sub = pygame.font.SysFont("Arial", 13)
        self.font_dice = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_badge = pygame.font.SysFont("Arial", 11, bold=True)

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

    def render(self, core: MonopolyCore):
        """Renders complete board state onto surface."""
        # Draw board background
        board_rect = pygame.Rect(self.offset_x, self.offset_y, self.board_size, self.board_size)
        pygame.draw.rect(self.surface, BOARD_BG, board_rect, border_radius=8)
        pygame.draw.rect(self.surface, (50, 55, 65), board_rect, width=3, border_radius=8)

        # Draw all 40 tiles
        for tile_id in range(40):
            self._render_tile(tile_id, core)

        # Draw player tokens
        self._render_tokens(core)

        # Draw center board HUD (dice, active player, card, auction)
        self._render_center_hud(core)

    def _render_tile(self, tile_id: int, core: MonopolyCore):
        """Draws individual tile background, color band, title, price, and houses."""
        rect = self.tile_rects[tile_id]
        tile = TILES_DATA[tile_id]

        # Base tile rectangle
        pygame.draw.rect(self.surface, (245, 247, 250), rect)
        pygame.draw.rect(self.surface, TILE_BORDER, rect, width=1)

        # Color header band for street properties
        color_grp = tile.get("color")
        if color_grp in COLOR_HEADER_MAP:
            header_color = COLOR_HEADER_MAP[color_grp]
            # Determine orientation of band
            if 0 < tile_id < 10:  # Bottom: band at top of tile
                band = pygame.Rect(rect.x, rect.y, rect.width, 18)
            elif 10 < tile_id < 20:  # Left: band at right of tile
                band = pygame.Rect(rect.right - 18, rect.y, 18, rect.height)
            elif 20 < tile_id < 30:  # Top: band at bottom of tile
                band = pygame.Rect(rect.x, rect.bottom - 18, rect.width, 18)
            elif 30 < tile_id < 40:  # Right: band at left of tile
                band = pygame.Rect(rect.x, rect.y, 18, rect.height)
            else:
                band = pygame.Rect(rect.x, rect.y, rect.width, 15)
            pygame.draw.rect(self.surface, header_color, band)
            pygame.draw.rect(self.surface, (60, 60, 60), band, width=1)

        # Highlight owner if owned
        owner = core.property_owner[tile_id]
        if owner != -1:
            owner_col = PLAYER_COLORS[owner]
            pygame.draw.rect(self.surface, owner_col, rect, width=2)
            # Small owner marker
            pygame.draw.circle(self.surface, owner_col, (rect.centerx, rect.centery), 4)

        # Mortgaged indicator
        if core.property_mortgaged[tile_id]:
            mtg_surf = self.font_price.render("MORTGAGE", True, (200, 40, 40))
            self.surface.blit(mtg_surf, (rect.centerx - mtg_surf.get_width() // 2, rect.centery - 6))

        # Tile name label
        name = tile["name"]
        short_name = name.split()[0] if len(name) > 10 else name
        text_surf = self.font_tile.render(short_name, True, TEXT_COLOR)

        # Price label if purchasable
        price_text = f"${tile['price']}" if tile.get("price", 0) > 0 else ""
        price_surf = self.font_price.render(price_text, True, (80, 85, 95))

        # Position labels
        if 0 < tile_id < 10:
            self.surface.blit(text_surf, (rect.centerx - text_surf.get_width() // 2, rect.y + 22))
            if price_text:
                self.surface.blit(price_surf, (rect.centerx - price_surf.get_width() // 2, rect.bottom - 15))
        elif 10 < tile_id < 20:
            self.surface.blit(text_surf, (rect.x + 6, rect.centery - 10))
            if price_text:
                self.surface.blit(price_surf, (rect.x + 6, rect.centery + 4))
        elif 20 < tile_id < 30:
            self.surface.blit(text_surf, (rect.centerx - text_surf.get_width() // 2, rect.y + 6))
            if price_text:
                self.surface.blit(price_surf, (rect.centerx - price_surf.get_width() // 2, rect.bottom - 28))
        elif 30 < tile_id < 40:
            self.surface.blit(text_surf, (rect.x + 22, rect.centery - 10))
            if price_text:
                self.surface.blit(price_surf, (rect.x + 22, rect.centery + 4))
        else:
            # Corner tiles
            self.surface.blit(text_surf, (rect.centerx - text_surf.get_width() // 2, rect.centery - 8))

        # Render Houses / Hotel
        houses = core.property_houses[tile_id]
        if houses > 0:
            self._render_buildings(tile_id, rect, houses)

    def _render_buildings(self, tile_id: int, rect: pygame.Rect, houses: int):
        """Draws house blocks (green) or hotel block (red)."""
        if houses == 5:
            # Hotel
            hotel_rect = pygame.Rect(rect.centerx - 8, rect.centery - 6, 16, 12)
            pygame.draw.rect(self.surface, HOTEL_COLOR, hotel_rect, border_radius=2)
            h_text = self.font_tile.render("H", True, (255, 255, 255))
            self.surface.blit(h_text, (hotel_rect.centerx - h_text.get_width() // 2, hotel_rect.centery - h_text.get_height() // 2))
        else:
            # 1 to 4 houses
            for h_i in range(houses):
                spacing = 7
                hx = rect.centerx - (houses * spacing) // 2 + h_i * spacing
                hy = rect.centery - 4
                h_rect = pygame.Rect(hx, hy, 6, 8)
                pygame.draw.rect(self.surface, HOUSE_COLOR, h_rect, border_radius=1)

    def _render_tokens(self, core: MonopolyCore):
        """Renders 4 distinct player tokens arranged neatly without overlapping."""
        # 2x2 grid offsets inside tile
        offsets = [
            (-12, -12),  # P0: top-left
            (12, -12),   # P1: top-right
            (-12, 12),   # P2: bottom-left
            (12, 12),    # P3: bottom-right
        ]

        for p_id in range(4):
            if core.bankrupt[p_id]:
                continue
            pos = core.position[p_id]
            rect = self.tile_rects[pos]
            ox, oy = offsets[p_id]
            tx = rect.centerx + ox
            ty = rect.centery + oy

            color = PLAYER_COLORS[p_id]
            # Outer dark ring
            pygame.draw.circle(self.surface, (20, 20, 25), (tx, ty), 9)
            # Token body
            pygame.draw.circle(self.surface, color, (tx, ty), 7)
            # Token ID text
            p_text = self.font_tile.render(str(p_id), True, (255, 255, 255))
            self.surface.blit(p_text, (tx - p_text.get_width() // 2, ty - p_text.get_height() // 2))

    def _render_center_hud(self, core: MonopolyCore):
        """Renders central board dashboard: round info, active player, dice, auction."""
        cx = self.offset_x + self.board_size // 2
        cy = self.offset_y + self.board_size // 2

        # Center box background
        box_w, box_h = 360, 240
        hud_box = pygame.Rect(cx - box_w // 2, cy - box_h // 2, box_w, box_h)
        pygame.draw.rect(self.surface, (240, 243, 248), hud_box, border_radius=12)
        pygame.draw.rect(self.surface, (200, 205, 215), hud_box, width=2, border_radius=12)

        # Title: Tournament Monopoly
        title_surf = self.font_center_title.render("TOURNAMENT MARL MONOPOLY", True, (35, 45, 60))
        self.surface.blit(title_surf, (cx - title_surf.get_width() // 2, hud_box.y + 14))

        # Round and Turn step
        round_text = f"Round: {core.round_count}/{core.max_rounds}   |   Turn Step: {core.turn_step}"
        r_surf = self.font_center_sub.render(round_text, True, (80, 90, 105))
        self.surface.blit(r_surf, (cx - r_surf.get_width() // 2, hud_box.y + 42))

        # Active Player Indicator
        p_act = core.active_decision_player
        p_col = PLAYER_COLORS[p_act]
        act_text = f"Active Player: P{p_act} ({['Roll', 'Buy/Auction', 'Auction', 'Jail', 'Manage', 'Trade', 'Over'][min(core.phase, 6)]})"
        act_surf = self.font_badge.render(act_text, True, (255, 255, 255))
        badge_rect = pygame.Rect(cx - 100, hud_box.y + 68, 200, 24)
        pygame.draw.rect(self.surface, p_col, badge_rect, border_radius=6)
        self.surface.blit(act_surf, (cx - act_surf.get_width() // 2, badge_rect.y + 4))

        # Render 3D Dice Display
        d1, d2 = core.last_dice
        self._render_dice(cx - 55, hud_box.y + 104, d1)
        self._render_dice(cx + 15, hud_box.y + 104, d2)

        # Auction Badge if active
        if core.phase == PHASE_AUCTION and core.auction_property is not None:
            prop_name = TILES_DATA[core.auction_property]["name"]
            auc_text = f"AUCTION: {prop_name}  |  Bid: ${core.auction_highest_bid} by P{core.auction_highest_bidder}"
            auc_surf = self.font_center_sub.render(auc_text, True, (200, 50, 50))
            self.surface.blit(auc_surf, (cx - auc_surf.get_width() // 2, hud_box.bottom - 48))

        # Latest event banner
        if core.event_log:
            last_ev = core.event_log[-1]
            if len(last_ev) > 48:
                last_ev = last_ev[:45] + "..."
            ev_surf = self.font_price.render(last_ev, True, (60, 70, 85))
            self.surface.blit(ev_surf, (cx - ev_surf.get_width() // 2, hud_box.bottom - 22))

    def _render_dice(self, x: int, y: int, value: int):
        """Draws a clean dice face with pips."""
        dice_rect = pygame.Rect(x, y, 40, 40)
        # Shadow
        pygame.draw.rect(self.surface, (180, 185, 195), dice_rect.move(2, 2), border_radius=6)
        # Body
        pygame.draw.rect(self.surface, (255, 255, 255), dice_rect, border_radius=6)
        pygame.draw.rect(self.surface, (100, 110, 125), dice_rect, width=2, border_radius=6)

        # Draw Pips
        cx, cy = dice_rect.centerx, dice_rect.centery
        dot_r = 3
        pip_col = (40, 45, 55)

        pip_coords = {
            1: [(cx, cy)],
            2: [(cx - 9, cy - 9), (cx + 9, cy + 9)],
            3: [(cx - 9, cy - 9), (cx, cy), (cx + 9, cy + 9)],
            4: [(cx - 9, cy - 9), (cx + 9, cy - 9), (cx - 9, cy + 9), (cx + 9, cy + 9)],
            5: [(cx - 9, cy - 9), (cx + 9, cy - 9), (cx, cy), (cx - 9, cy + 9), (cx + 9, cy + 9)],
            6: [(cx - 9, cy - 10), (cx + 9, cy - 10), (cx - 9, cy), (cx + 9, cy), (cx - 9, cy + 10), (cx + 9, cy + 10)],
        }

        for px, py in pip_coords.get(value, [(cx, cy)]):
            pygame.draw.circle(self.surface, pip_col, (px, py), dot_r)
