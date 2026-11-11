"""Interactive Pygame Renderer for 4-Agent Monopoly MARL.

Renders high-fidelity 11x11 Monopoly board with authentic tile badges,
property color bands, complete monopoly indicators, non-overlapping
player tokens (Luke, Graham, Pieter, Allison), realistic 3D dice,
and dynamic context-sensitive center HUD (Auctions, Trades, Victory Champion).
"""

from typing import Dict, List, Tuple, Optional, Any
import pygame
import math
import numpy as np

from monopoly_marl.engine.board_data import (
    TILES_DATA, COLOR_GROUPS, TILE_TO_GROUP, PURCHASABLE_TILES,
    COLOR_BROWN, COLOR_LIGHT_BLUE, COLOR_PINK, COLOR_ORANGE,
    COLOR_RED, COLOR_YELLOW, COLOR_GREEN, COLOR_DARK_BLUE,
    TILE_STREET, TILE_RAILROAD, TILE_UTILITY, TILE_TAX, TILE_CARD, TILE_SPECIAL
)
from monopoly_marl.engine.monopoly_core import (
    MonopolyCore, PHASE_ROLL, PHASE_BUY_OR_AUCTION, PHASE_AUCTION,
    PHASE_JAIL_DECISION, PHASE_PROPERTY_MANAGE, PHASE_TRADE_RESPONSE, PHASE_GAME_OVER
)

# Color Palette
BG_COLOR = (20, 24, 32)
BOARD_BG = (238, 241, 246)
TILE_BG = (248, 250, 253)
TILE_BORDER = (185, 192, 204)
TEXT_COLOR = (28, 32, 42)
TEXT_MUTED = (90, 100, 115)
TEXT_LIGHT = (235, 240, 248)

COLOR_HEADER_MAP = {
    COLOR_BROWN: (140, 75, 30),
    COLOR_LIGHT_BLUE: (130, 200, 240),
    COLOR_PINK: (215, 105, 185),
    COLOR_ORANGE: (250, 140, 20),
    COLOR_RED: (225, 35, 55),
    COLOR_YELLOW: (250, 215, 25),
    COLOR_GREEN: (35, 155, 65),
    COLOR_DARK_BLUE: (20, 45, 125),
}

# Player Identity Colors
PLAYER_COLORS = [
    (239, 71, 111),   # Player 0: Luke (Coral Red)
    (17, 138, 178),   # Player 1: Graham (Ocean Blue)
    (6, 214, 160),    # Player 2: Pieter (Mint Green)
    (255, 209, 102),  # Player 3: Allison (Warm Yellow)
]

PLAYER_INITIALS = ["L", "G", "P", "A"]

HOUSE_COLOR = (39, 174, 96)    # Forest Green
HOTEL_COLOR = (214, 48, 49)    # Crimson Red
GOLD_COLOR = (241, 196, 15)    # Shiny Gold
JAIL_BAR_COLOR = (60, 65, 75)

# Curated 2-line clean display labels and icons for all 40 tiles
TILE_DISPLAY_CONFIG = {
    0: {"line1": "COLLECT", "line2": "$200", "sub": "GO", "icon": "go"},
    1: {"line1": "Mediter.", "line2": "Avenue", "price": "$60"},
    2: {"line1": "Community", "line2": "Chest", "icon": "chest"},
    3: {"line1": "Baltic", "line2": "Avenue", "price": "$60"},
    4: {"line1": "Income", "line2": "Tax", "price": "Pay $200", "icon": "tax"},
    5: {"line1": "Reading", "line2": "Railroad", "price": "$200", "icon": "rr"},
    6: {"line1": "Oriental", "line2": "Avenue", "price": "$100"},
    7: {"line1": "Chance", "line2": "", "icon": "chance"},
    8: {"line1": "Vermont", "line2": "Avenue", "price": "$100"},
    9: {"line1": "Connect.", "line2": "Avenue", "price": "$120"},
    10: {"line1": "JAIL", "line2": "CELL", "sub": "JUST VISITING", "icon": "jail"},
    11: {"line1": "St. Charles", "line2": "Place", "price": "$140"},
    12: {"line1": "Electric", "line2": "Company", "price": "$150", "icon": "utility_elec"},
    13: {"line1": "States", "line2": "Avenue", "price": "$140"},
    14: {"line1": "Virginia", "line2": "Avenue", "price": "$160"},
    15: {"line1": "Penn.", "line2": "Railroad", "price": "$200", "icon": "rr"},
    16: {"line1": "St. James", "line2": "Place", "price": "$180"},
    17: {"line1": "Community", "line2": "Chest", "icon": "chest"},
    18: {"line1": "Tennessee", "line2": "Avenue", "price": "$180"},
    19: {"line1": "New York", "line2": "Avenue", "price": "$200"},
    20: {"line1": "FREE", "line2": "PARKING", "icon": "parking"},
    21: {"line1": "Kentucky", "line2": "Avenue", "price": "$220"},
    22: {"line1": "Chance", "line2": "", "icon": "chance"},
    23: {"line1": "Indiana", "line2": "Avenue", "price": "$220"},
    24: {"line1": "Illinois", "line2": "Avenue", "price": "$240"},
    25: {"line1": "B. & O.", "line2": "Railroad", "price": "$200", "icon": "rr"},
    26: {"line1": "Atlantic", "line2": "Avenue", "price": "$260"},
    27: {"line1": "Ventnor", "line2": "Avenue", "price": "$260"},
    28: {"line1": "Water", "line2": "Works", "price": "$150", "icon": "utility_water"},
    29: {"line1": "Marvin", "line2": "Gardens", "price": "$280"},
    30: {"line1": "GO TO", "line2": "JAIL", "icon": "gotojail"},
    31: {"line1": "Pacific", "line2": "Avenue", "price": "$300"},
    32: {"line1": "North", "line2": "Carolina", "price": "$300"},
    33: {"line1": "Community", "line2": "Chest", "icon": "chest"},
    34: {"line1": "Penn.", "line2": "Avenue", "price": "$320"},
    35: {"line1": "Short", "line2": "Line", "price": "$200", "icon": "rr"},
    36: {"line1": "Chance", "line2": "", "icon": "chance"},
    37: {"line1": "Park", "line2": "Place", "price": "$350"},
    38: {"line1": "Luxury", "line2": "Tax", "price": "Pay $100", "icon": "tax"},
    39: {"line1": "Boardwalk", "line2": "", "price": "$400"},
}


class MonopolyRenderer:
    """Renders the Monopoly 11x11 game board, visual tokens, and center game HUD."""

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
        self.font_price = pygame.font.SysFont("Arial", 9, bold=False)
        self.font_corner_title = pygame.font.SysFont("Arial", 11, bold=True)
        self.font_corner_sub = pygame.font.SysFont("Arial", 8, bold=True)
        self.font_token = pygame.font.SysFont("Arial", 10, bold=True)
        self.font_center_title = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_center_sub = pygame.font.SysFont("Arial", 12)
        self.font_dice = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_badge = pygame.font.SysFont("Arial", 11, bold=True)
        self.font_victory_title = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_victory_sub = pygame.font.SysFont("Arial", 12, bold=True)

        self.animation_tick = 0

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
        self.animation_tick += 1

        # Draw board background
        board_rect = pygame.Rect(self.offset_x, self.offset_y, self.board_size, self.board_size)
        pygame.draw.rect(self.surface, BOARD_BG, board_rect, border_radius=10)
        pygame.draw.rect(self.surface, (45, 52, 65), board_rect, width=3, border_radius=10)

        # Draw all 40 tiles
        for tile_id in range(40):
            self._render_tile(tile_id, core)

        # Draw player tokens
        self._render_tokens(core)

        # Draw center board HUD (dice, active player, card, auction, game over)
        self._render_center_hud(core)

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

    def _render_tile(self, tile_id: int, core: MonopolyCore):
        """Draws individual tile background, color band, labels, icons, and buildings."""
        rect = self.tile_rects[tile_id]
        tile = TILES_DATA[tile_id]
        cfg = TILE_DISPLAY_CONFIG.get(tile_id, {"line1": tile["name"], "line2": ""})

        # Base tile rectangle
        pygame.draw.rect(self.surface, TILE_BG, rect)
        pygame.draw.rect(self.surface, TILE_BORDER, rect, width=1)

        # 1. Corner tiles special handling
        if tile_id in (0, 10, 20, 30):
            self._render_corner_tile(tile_id, rect, core)
            return

        # 2. Color header band for street properties
        color_grp = tile.get("color")
        has_color = color_grp in COLOR_HEADER_MAP
        is_monopoly = self._is_monopoly_tile(tile_id, core)

        if has_color:
            header_color = COLOR_HEADER_MAP[color_grp]
            band_size = 18
            if 0 < tile_id < 10:  # Bottom: band at top
                band = pygame.Rect(rect.x, rect.y, rect.width, band_size)
            elif 10 < tile_id < 20:  # Left: band at right
                band = pygame.Rect(rect.right - band_size, rect.y, band_size, rect.height)
            elif 20 < tile_id < 30:  # Top: band at bottom
                band = pygame.Rect(rect.x, rect.bottom - band_size, rect.width, band_size)
            else:  # Right: band at left
                band = pygame.Rect(rect.x, rect.y, band_size, rect.height)

            pygame.draw.rect(self.surface, header_color, band)
            pygame.draw.rect(self.surface, (50, 50, 50), band, width=1)

            # If full monopoly: golden star marker drawn in header band
            if is_monopoly:
                pygame.draw.rect(self.surface, GOLD_COLOR, band, width=2)
                # Draw small golden star dot
                pygame.draw.circle(self.surface, GOLD_COLOR, (band.centerx, band.centery), 4)
                pygame.draw.circle(self.surface, (255, 255, 255), (band.centerx, band.centery), 2)

        # 3. Property Ownership Ribbon
        owner = core.property_owner[tile_id]
        if owner != -1:
            owner_col = PLAYER_COLORS[owner]
            ribbon_w = 4
            # Draw ribbon along inner edge
            if 0 < tile_id < 10:
                rib_rect = pygame.Rect(rect.x, rect.bottom - ribbon_w, rect.width, ribbon_w)
            elif 10 < tile_id < 20:
                rib_rect = pygame.Rect(rect.x, rect.y, ribbon_w, rect.height)
            elif 20 < tile_id < 30:
                rib_rect = pygame.Rect(rect.x, rect.y, rect.width, ribbon_w)
            else:
                rib_rect = pygame.Rect(rect.right - ribbon_w, rect.y, ribbon_w, rect.height)

            pygame.draw.rect(self.surface, owner_col, rib_rect)

            # Small owner initial badge
            by = rect.bottom - 16 if 0 < tile_id < 10 else (rect.y + 4 if 20 < tile_id < 30 else rect.centery - 6)
            bx = rect.centerx - 7 if (0 < tile_id < 10 or 20 < tile_id < 30) else (rect.x + 4 if 10 < tile_id < 20 else rect.right - 18)
            init_rect = pygame.Rect(bx, by, 14, 12)
            pygame.draw.rect(self.surface, owner_col, init_rect, border_radius=3)
            init_txt = self.font_corner_sub.render(PLAYER_INITIALS[owner], True, (255, 255, 255))
            self.surface.blit(init_txt, (init_rect.centerx - init_txt.get_width() // 2, init_rect.centery - init_txt.get_height() // 2))

        # 4. Render Tile Text and Icon with zero overlap
        icon_type = cfg.get("icon")
        l1 = cfg.get("line1", "")
        l2 = cfg.get("line2", "")
        price_str = cfg.get("price", f"${tile['price']}" if tile.get("price", 0) > 0 else "")

        if 0 < tile_id < 10:  # Bottom Row
            if icon_type:
                # Icon placed in upper half, text below
                self._draw_tile_icon(icon_type, rect.centerx, rect.y + 24)
                t1 = self.font_tile.render(l1, True, TEXT_COLOR)
                t2 = self.font_tile.render(l2, True, TEXT_COLOR)
                self.surface.blit(t1, (rect.centerx - t1.get_width() // 2, rect.y + 44))
                if l2:
                    self.surface.blit(t2, (rect.centerx - t2.get_width() // 2, rect.y + 56))
            else:
                # Street properties: band on top (y: 0..18), text below
                t1 = self.font_tile.render(l1, True, TEXT_COLOR)
                t2 = self.font_tile.render(l2, True, TEXT_COLOR)
                self.surface.blit(t1, (rect.centerx - t1.get_width() // 2, rect.y + 22))
                if l2:
                    self.surface.blit(t2, (rect.centerx - t2.get_width() // 2, rect.y + 34))

            if price_str:
                p_surf = self.font_price.render(price_str, True, TEXT_MUTED)
                self.surface.blit(p_surf, (rect.centerx - p_surf.get_width() // 2, rect.bottom - 16))

        elif 10 < tile_id < 20:  # Left Row
            if icon_type:
                self._draw_tile_icon(icon_type, rect.x + 68, rect.centery)
            t1 = self.font_tile.render(l1, True, TEXT_COLOR)
            t2 = self.font_tile.render(l2, True, TEXT_COLOR)
            self.surface.blit(t1, (rect.x + 8, rect.centery - 16))
            if l2:
                self.surface.blit(t2, (rect.x + 8, rect.centery - 5))
            if price_str:
                p_surf = self.font_price.render(price_str, True, TEXT_MUTED)
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
                p_surf = self.font_price.render(price_str, True, TEXT_MUTED)
                self.surface.blit(p_surf, (rect.centerx - p_surf.get_width() // 2, rect.bottom - 32))

        elif 30 < tile_id < 40:  # Right Row
            if icon_type:
                self._draw_tile_icon(icon_type, rect.x + 24, rect.centery)
            t1 = self.font_tile.render(l1, True, TEXT_COLOR)
            t2 = self.font_tile.render(l2, True, TEXT_COLOR)
            self.surface.blit(t1, (rect.x + 40, rect.centery - 16))
            if l2:
                self.surface.blit(t2, (rect.x + 40, rect.centery - 5))
            if price_str:
                p_surf = self.font_price.render(price_str, True, TEXT_MUTED)
                self.surface.blit(p_surf, (rect.x + 40, rect.centery + 8))

        # 5. Mortgaged Overlay
        if core.property_mortgaged[tile_id]:
            mtg_badge = pygame.Rect(rect.centerx - 26, rect.centery - 8, 52, 16)
            pygame.draw.rect(self.surface, (230, 45, 45), mtg_badge, border_radius=3)
            mtg_surf = self.font_corner_sub.render("MORTGAGE", True, (255, 255, 255))
            self.surface.blit(mtg_surf, (mtg_badge.centerx - mtg_surf.get_width() // 2, mtg_badge.centery - mtg_surf.get_height() // 2))

        # 6. Render Houses / Hotel
        houses = core.property_houses[tile_id]
        if houses > 0 and not core.property_mortgaged[tile_id]:
            self._render_buildings(tile_id, rect, houses)

    def _render_corner_tile(self, tile_id: int, rect: pygame.Rect, core: MonopolyCore):
        """Draws rich detailed corner tiles (GO, Jail, Free Parking, Go To Jail)."""
        if tile_id == 0:  # GO
            # Red header banner
            arrow_rect = pygame.Rect(rect.x + 8, rect.y + 8, rect.width - 16, 26)
            pygame.draw.rect(self.surface, (231, 76, 60), arrow_rect, border_radius=4)
            t_go = self.font_victory_title.render("GO ->", True, (255, 255, 255))
            self.surface.blit(t_go, (arrow_rect.centerx - t_go.get_width() // 2, arrow_rect.centery - t_go.get_height() // 2))

            t_col = self.font_corner_title.render("COLLECT $200", True, (46, 125, 50))
            self.surface.blit(t_col, (rect.centerx - t_col.get_width() // 2, rect.y + 40))
            t_pass = self.font_corner_sub.render("AS YOU PASS", True, TEXT_MUTED)
            self.surface.blit(t_pass, (rect.centerx - t_pass.get_width() // 2, rect.y + 56))

        elif tile_id == 10:  # JAIL
            # Inner Jail Cell
            cell_w, cell_h = 52, 52
            cell_rect = pygame.Rect(rect.right - cell_w, rect.y, cell_w, cell_h)
            pygame.draw.rect(self.surface, (220, 140, 40), cell_rect)
            pygame.draw.rect(self.surface, JAIL_BAR_COLOR, cell_rect, width=2)
            # Vertical iron bars
            for bx in range(cell_rect.x + 8, cell_rect.right - 4, 10):
                pygame.draw.line(self.surface, JAIL_BAR_COLOR, (bx, cell_rect.y), (bx, cell_rect.bottom), 2)

            t_jail = self.font_corner_sub.render("IN JAIL", True, (255, 255, 255))
            self.surface.blit(t_jail, (cell_rect.centerx - t_jail.get_width() // 2, cell_rect.y + 6))

            # Outer Visiting Area
            t_vis1 = self.font_corner_sub.render("JUST", True, TEXT_COLOR)
            t_vis2 = self.font_corner_sub.render("VISITING", True, TEXT_COLOR)
            self.surface.blit(t_vis1, (rect.x + 8, rect.bottom - 32))
            self.surface.blit(t_vis2, (rect.x + 8, rect.bottom - 18))

        elif tile_id == 20:  # FREE PARKING
            # Red/White Parking Emblem
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
            # Draw police star emblem
            pygame.draw.circle(self.surface, GOLD_COLOR, (badge_rect.centerx, badge_rect.centery), 9)
            pygame.draw.circle(self.surface, (52, 73, 94), (badge_rect.centerx, badge_rect.centery), 4)

            t_gj1 = self.font_corner_title.render("GO TO", True, (214, 48, 49))
            t_gj2 = self.font_corner_title.render("JAIL!", True, (214, 48, 49))
            self.surface.blit(t_gj1, (rect.centerx - t_gj1.get_width() // 2, rect.bottom - 36))
            self.surface.blit(t_gj2, (rect.centerx - t_gj2.get_width() // 2, rect.bottom - 22))

    def _draw_tile_icon(self, icon_type: str, cx: int, cy: int):
        """Draws clean geometric mini vector icons for Railroads, Chance, Chest, Tax, Utilities."""
        if icon_type == "rr":
            # Locomotive train icon
            body = pygame.Rect(cx - 9, cy - 5, 18, 10)
            pygame.draw.rect(self.surface, (45, 52, 65), body, border_radius=2)
            # Chimney
            pygame.draw.rect(self.surface, (45, 52, 65), (cx + 3, cy - 8, 4, 4))
            # Wheels
            pygame.draw.circle(self.surface, (25, 25, 30), (cx - 5, cy + 6), 3)
            pygame.draw.circle(self.surface, (25, 25, 30), (cx + 5, cy + 6), 3)

        elif icon_type == "chance":
            # Orange diamond with question mark
            pts = [(cx, cy - 9), (cx + 9, cy), (cx, cy + 9), (cx - 9, cy)]
            pygame.draw.polygon(self.surface, (230, 126, 34), pts)
            q_surf = self.font_corner_sub.render("?", True, (255, 255, 255))
            self.surface.blit(q_surf, (cx - q_surf.get_width() // 2, cy - q_surf.get_height() // 2))

        elif icon_type == "chest":
            # Blue Treasure chest
            c_rect = pygame.Rect(cx - 9, cy - 5, 18, 10)
            pygame.draw.rect(self.surface, (41, 128, 185), c_rect, border_radius=2)
            pygame.draw.line(self.surface, GOLD_COLOR, (cx, cy - 5), (cx, cy + 4), 2)

        elif icon_type == "tax":
            # Diamond tax shape
            pts = [(cx, cy - 8), (cx + 8, cy - 2), (cx, cy + 8), (cx - 8, cy - 2)]
            pygame.draw.polygon(self.surface, (142, 68, 173), pts)

        elif icon_type == "utility_elec":
            # Yellow lightbulb glyph
            pygame.draw.circle(self.surface, (241, 196, 15), (cx, cy - 3), 5)
            pygame.draw.rect(self.surface, (120, 120, 130), (cx - 3, cy + 2, 6, 3))

        elif icon_type == "utility_water":
            # Blue water drop glyph
            pts = [(cx, cy - 7), (cx + 5, cy + 2), (cx, cy + 6), (cx - 5, cy + 2)]
            pygame.draw.polygon(self.surface, (52, 152, 219), pts)

    def _render_buildings(self, tile_id: int, rect: pygame.Rect, houses: int):
        """Draws house blocks (green) or hotel block (crimson red) with 3D bevels."""
        if houses == 5:
            # Hotel
            hx = rect.centerx - 10
            hy = rect.centery - 6
            hotel_rect = pygame.Rect(hx, hy, 20, 12)
            pygame.draw.rect(self.surface, HOTEL_COLOR, hotel_rect, border_radius=2)
            pygame.draw.rect(self.surface, (255, 120, 120), hotel_rect, width=1, border_radius=2)
            h_text = self.font_corner_sub.render("HOTEL", True, (255, 255, 255))
            self.surface.blit(h_text, (hotel_rect.centerx - h_text.get_width() // 2, hotel_rect.centery - h_text.get_height() // 2))
        else:
            # 1 to 4 houses
            spacing = 8
            start_x = rect.centerx - (houses * spacing) // 2
            hy = rect.centery - 5
            for h_i in range(houses):
                h_rect = pygame.Rect(start_x + h_i * spacing, hy, 6, 8)
                pygame.draw.rect(self.surface, HOUSE_COLOR, h_rect, border_radius=1)
                pygame.draw.rect(self.surface, (120, 240, 150), h_rect, width=1, border_radius=1)

    def _render_tokens(self, core: MonopolyCore):
        """Renders 4 distinct player tokens with initials (Luke=L, Graham=G, Pieter=P, Allison=A)."""
        offsets = [
            (-12, -12),  # P0 Luke: top-left
            (12, -12),   # P1 Graham: top-right
            (-12, 12),   # P2 Pieter: bottom-left
            (12, 12),    # P3 Allison: bottom-right
        ]

        jail_rect = self.tile_rects[10]
        jail_cell_cx = jail_rect.right - 26
        jail_cell_cy = jail_rect.y + 26
        jail_offsets = [(-8, -8), (8, -8), (-8, 8), (8, 8)]

        # On Tile 0 (GO), arrange tokens in bottom area so they don't block the GO banner
        go_rect = self.tile_rects[0]
        go_offsets = [
            (-14, 22),   # Luke
            (14, 22),    # Graham
            (-14, 38),   # Pieter
            (14, 38),    # Allison
        ]

        active_player = core.active_decision_player

        for p_id in range(4):
            if core.bankrupt[p_id]:
                continue

            pos = core.position[p_id]
            rect = self.tile_rects[pos]

            if pos == 0:
                ox, oy = go_offsets[p_id]
                tx = rect.centerx + ox
                ty = rect.centery + oy
            elif pos == 10 and core.in_jail[p_id]:
                ox, oy = jail_offsets[p_id]
                tx = jail_cell_cx + ox
                ty = jail_cell_cy + oy
            elif pos == 10:  # Just visiting
                vis_offsets = [(-20, 20), (0, 20), (-20, 36), (0, 36)]
                ox, oy = vis_offsets[p_id]
                tx = rect.centerx + ox
                ty = rect.centery + oy
            else:
                ox, oy = offsets[p_id]
                tx = rect.centerx + ox
                ty = rect.centery + oy

            color = PLAYER_COLORS[p_id]

            # Active player animated golden pulse ring
            if p_id == active_player and core.phase != PHASE_GAME_OVER:
                pulse_r = 11 + int(math.sin(self.animation_tick * 0.15) * 2)
                pygame.draw.circle(self.surface, GOLD_COLOR, (tx, ty), pulse_r, width=2)

            # Drop shadow
            pygame.draw.circle(self.surface, (15, 20, 28), (tx + 1, ty + 2), 9)
            # Outer dark ring
            pygame.draw.circle(self.surface, (30, 35, 45), (tx, ty), 9)
            # Token body
            pygame.draw.circle(self.surface, color, (tx, ty), 7)
            # Gloss highlight
            pygame.draw.circle(self.surface, (255, 255, 255), (tx - 2, ty - 2), 2)

            # Initial letter: L, G, P, A
            letter = PLAYER_INITIALS[p_id]
            t_surf = self.font_token.render(letter, True, (255, 255, 255))
            self.surface.blit(t_surf, (tx - t_surf.get_width() // 2, ty - t_surf.get_height() // 2))

    def _render_center_hud(self, core: MonopolyCore):
        """Renders center board display contextually (Default turn, Auction, Trade, Victory Champion)."""
        cx = self.offset_x + self.board_size // 2
        cy = self.offset_y + self.board_size // 2

        # Center box background
        box_w, box_h = 400, 250
        hud_box = pygame.Rect(cx - box_w // 2, cy - box_h // 2, box_w, box_h)
        pygame.draw.rect(self.surface, (244, 246, 250), hud_box, border_radius=14)
        pygame.draw.rect(self.surface, (195, 202, 215), hud_box, width=2, border_radius=14)

        # ----------------- PHASE: GAME OVER (VICTORY CHAMPION) -----------------
        if core.phase == PHASE_GAME_OVER:
            self._render_victory_hud(core, hud_box, cx, cy)
            return

        # ----------------- PHASE: AUCTION -----------------
        if core.phase == PHASE_AUCTION and core.auction_property is not None:
            self._render_auction_hud(core, hud_box, cx, cy)
            return

        # ----------------- PHASE: TRADE PROPOSAL -----------------
        if core.phase == PHASE_TRADE_RESPONSE and getattr(core, "pending_trade", None) is not None:
            self._render_trade_hud(core, hud_box, cx, cy)
            return

        # ----------------- DEFAULT TURN HUD -----------------
        # Title
        t_surf = self.font_center_title.render("TOURNAMENT MARL MONOPOLY", True, (30, 40, 55))
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
        badge_w = max(220, act_surf.get_width() + 24)
        badge_rect = pygame.Rect(cx - badge_w // 2, hud_box.y + 64, badge_w, 24)
        pygame.draw.rect(self.surface, p_col, badge_rect, border_radius=6)
        self.surface.blit(act_surf, (cx - act_surf.get_width() // 2, badge_rect.y + 4))

        # 3D Dice Display
        d1, d2 = core.last_dice
        self._render_dice(cx - 55, hud_box.y + 100, d1)
        self._render_dice(cx + 15, hud_box.y + 100, d2)

        # Consecutive doubles banner if in rolling streak
        if getattr(core, "consecutive_doubles", 0) > 0:
            dbl_txt = f"Doubles Rolled! ({core.consecutive_doubles}/3) -> Extra Roll"
            d_surf = self.font_corner_sub.render(dbl_txt, True, (214, 48, 49))
            self.surface.blit(d_surf, (cx - d_surf.get_width() // 2, hud_box.y + 152))

        # Surviving Agents Counter
        active_count = sum(1 for b in core.bankrupt if not b)
        surv_txt = f"Survival Status: {active_count} of 4 Agents Remaining"
        s_surf = self.font_center_sub.render(surv_txt, True, (50, 140, 80) if active_count > 1 else (210, 40, 40))
        self.surface.blit(s_surf, (cx - s_surf.get_width() // 2, hud_box.bottom - 46))

        # Latest event snippet
        if core.event_log:
            last_ev = core.event_log[-1]
            if len(last_ev) > 52:
                last_ev = last_ev[:49] + "..."
            ev_surf = self.font_price.render(last_ev, True, (75, 85, 100))
            self.surface.blit(ev_surf, (cx - ev_surf.get_width() // 2, hud_box.bottom - 22))

    def _render_auction_hud(self, core: MonopolyCore, hud_box: pygame.Rect, cx: int, cy: int):
        """Renders dedicated interactive auction block."""
        prop_id = core.auction_property
        tile = TILES_DATA[prop_id]
        p_name = tile["name"]

        # Header
        h_surf = self.font_victory_sub.render("MANDATORY OPEN AUCTION", True, (215, 100, 20))
        self.surface.blit(h_surf, (cx - h_surf.get_width() // 2, hud_box.y + 14))

        # Property Badge
        color_grp = tile.get("color")
        band_col = COLOR_HEADER_MAP.get(color_grp, (70, 80, 95))
        p_badge = pygame.Rect(cx - 140, hud_box.y + 38, 280, 26)
        pygame.draw.rect(self.surface, band_col, p_badge, border_radius=4)
        t_prop = self.font_badge.render(f"{p_name} (List: ${tile['price']})", True, (255, 255, 255))
        self.surface.blit(t_prop, (cx - t_prop.get_width() // 2, p_badge.y + 5))

        # Current High Bid Box
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

        # Active Bidders Ring
        act_bidder = core.active_decision_player
        turn_txt = f"Awaiting Bid From: {core.get_player_name(act_bidder)}"
        t_surf = self.font_badge.render(turn_txt, True, PLAYER_COLORS[act_bidder])
        self.surface.blit(t_surf, (cx - t_surf.get_width() // 2, hud_box.y + 134))

        # Bidders status row
        bx = cx - 110
        for b_id in range(4):
            is_active = b_id in core.auction_active_bidders
            b_pill = pygame.Rect(bx, hud_box.y + 158, 50, 20)
            p_c = PLAYER_COLORS[b_id] if is_active else (190, 195, 205)
            pygame.draw.rect(self.surface, p_c, b_pill, border_radius=4)
            p_txt = self.font_corner_sub.render(f"{PLAYER_INITIALS[b_id]} ({'BID' if is_active else 'OUT'})", True, (255, 255, 255))
            self.surface.blit(p_txt, (b_pill.centerx - p_txt.get_width() // 2, b_pill.centery - p_txt.get_height() // 2))
            bx += 60

        # Sub info
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

        # Card 1: Offer
        o_h = self.font_badge.render(f"{init_name} OFFERS:", True, PLAYER_COLORS[init])
        self.surface.blit(o_h, (card1.x + 8, card1.y + 8))
        o_p = self.font_price.render(f"Prop: {offer_p[:14]}", True, TEXT_COLOR)
        self.surface.blit(o_p, (card1.x + 8, card1.y + 34))
        o_c = self.font_price.render(f"Cash: ${trade['offer_cash']}", True, (46, 125, 50))
        self.surface.blit(o_c, (card1.x + 8, card1.y + 56))

        # Card 2: Request
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

        # Championship Title
        t_surf = self.font_victory_title.render(f"*** {w_name.upper()} IS THE CHAMPION! ***", True, w_col)
        self.surface.blit(t_surf, (cx - t_surf.get_width() // 2, hud_box.y + 18))

        sub_surf = self.font_center_sub.render(sub_title, True, (190, 140, 20))
        self.surface.blit(sub_surf, (cx - sub_surf.get_width() // 2, hud_box.y + 48))

        # Winner Details Card
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

    def _render_dice(self, x: int, y: int, value: int):
        """Draws a clean 3D dice face with drop shadow, border bevel, and pips."""
        dice_rect = pygame.Rect(x, y, 40, 40)
        pygame.draw.rect(self.surface, (170, 175, 185), dice_rect.move(2, 3), border_radius=6)
        pygame.draw.rect(self.surface, (255, 255, 255), dice_rect, border_radius=6)
        pygame.draw.rect(self.surface, (90, 100, 115), dice_rect, width=2, border_radius=6)

        cx, cy = dice_rect.centerx, dice_rect.centery
        dot_r = 3
        pip_col = (35, 40, 50)

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
