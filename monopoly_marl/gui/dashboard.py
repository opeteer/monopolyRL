"""Real-Time Interactive Dashboard for MARL Monopoly.

Displays live agent statistics (Cash, Net Worth, Liquidity Health Bar,
Monopoly & Color Group Portfolio Swatches, Scarcity Meters, Event Ticker),
and interactive controls (Play/Pause, Step, Reset, Screenshot, Speed, Help).
"""

from typing import Dict, List, Optional, Tuple, Any
import pygame
import numpy as np

from monopoly_marl.engine.monopoly_core import MonopolyCore
from monopoly_marl.engine.board_data import (
    PURCHASABLE_TILES, COLOR_GROUPS, COLOR_BROWN, COLOR_LIGHT_BLUE,
    COLOR_PINK, COLOR_ORANGE, COLOR_RED, COLOR_YELLOW, COLOR_GREEN,
    COLOR_DARK_BLUE, COLOR_RAILROAD, COLOR_UTILITY
)
from monopoly_marl.gui.renderer import PLAYER_COLORS, HOUSE_COLOR, HOTEL_COLOR, GOLD_COLOR, COLOR_HEADER_MAP

DASH_BG = (18, 22, 30)
CARD_BG = (28, 34, 46)
CARD_BG_ACTIVE = (38, 48, 66)
CARD_BORDER = (45, 54, 72)
TEXT_WHITE = (245, 248, 252)
TEXT_MUTED = (145, 155, 175)
TEXT_GREEN = (46, 204, 113)
TEXT_RED = (231, 76, 60)
TEXT_YELLOW = (241, 196, 15)

BTN_BG = (38, 46, 62)
BTN_HOVER = (55, 68, 92)
BTN_ACTIVE = (52, 152, 219)

# Group order for dashboard swatch tracker
SWATCH_GROUPS = [
    (COLOR_BROWN, "BR", 2),
    (COLOR_LIGHT_BLUE, "LB", 3),
    (COLOR_PINK, "PK", 3),
    (COLOR_ORANGE, "OR", 3),
    (COLOR_RED, "RD", 3),
    (COLOR_YELLOW, "YL", 3),
    (COLOR_GREEN, "GR", 3),
    (COLOR_DARK_BLUE, "DB", 2),
    (COLOR_RAILROAD, "RR", 4),
    (COLOR_UTILITY, "UT", 2),
]


class DashboardButton:
    """Clickable UI Button with hover, active state, and modern styling."""

    def __init__(self, rect: pygame.Rect, label: str, action_id: str, accent_color: Optional[Tuple[int, int, int]] = None):
        self.rect = rect
        self.label = label
        self.action_id = action_id
        self.accent_color = accent_color
        self.is_hovered = False
        self.is_active = False

    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return self.action_id
        return None

    def draw(self, surface: pygame.Surface, font: pygame.font.Font):
        if self.is_active:
            col = self.accent_color if self.accent_color else BTN_ACTIVE
        elif self.is_hovered:
            col = BTN_HOVER
        else:
            col = BTN_BG

        pygame.draw.rect(surface, col, self.rect, border_radius=6)
        border_col = (90, 110, 140) if self.is_hovered or self.is_active else (55, 65, 85)
        pygame.draw.rect(surface, border_col, self.rect, width=1, border_radius=6)

        txt_surf = font.render(self.label, True, TEXT_WHITE)
        surface.blit(txt_surf, (self.rect.centerx - txt_surf.get_width() // 2, self.rect.centery - txt_surf.get_height() // 2))


class MonopolyDashboard:
    """Right-Hand Side Dashboard Panel with rich agent tracking and controls."""

    def __init__(self, surface: pygame.Surface, x: int = 840, y: int = 20, width: int = 520, height: int = 800, initially_paused: bool = True, agent_labels: Optional[List[str]] = None):
        self.surface = surface
        self.rect = pygame.Rect(x, y, width, height)

        # Fonts
        pygame.font.init()
        self.font_header = pygame.font.SysFont("Arial", 15, bold=True)
        self.font_card_title = pygame.font.SysFont("Arial", 12, bold=True)
        self.font_stat = pygame.font.SysFont("Arial", 11)
        self.font_btn = pygame.font.SysFont("Arial", 11, bold=True)
        self.font_badge = pygame.font.SysFont("Arial", 9, bold=True)
        self.font_log = pygame.font.SysFont("Arial", 10)
        self.font_help_title = pygame.font.SysFont("Arial", 14, bold=True)
        self.font_help_body = pygame.font.SysFont("Arial", 11)

        # Interactive states: start in paused state by default
        self.is_paused = initially_paused
        self.agent_labels: List[str] = list(agent_labels) if agent_labels else []
        self.speed = 2  # 1, 2, 5, 20, 100
        self.speed_options = [1, 2, 5, 20, 100]
        self.show_help = False

        # Setup Buttons
        self.buttons: List[DashboardButton] = []
        self._init_buttons()

    def _init_buttons(self):
        bx = self.rect.x + 20
        # Row 1: Actions (y = rect.y + 458)
        by = self.rect.y + 458
        bw = 82
        bh = 32
        gap = 10

        self.btn_pause = DashboardButton(
            pygame.Rect(bx, by, bw, bh),
            "PLAY" if self.is_paused else "PAUSE",
            "toggle_pause",
            accent_color=(39, 174, 96) if self.is_paused else None
        )
        self.btn_pause.is_active = self.is_paused
        self.btn_step = DashboardButton(pygame.Rect(bx + bw + gap, by, bw, bh), "STEP", "step")
        self.btn_reset = DashboardButton(pygame.Rect(bx + 2 * (bw + gap), by, bw, bh), "RESET", "reset", accent_color=(192, 57, 43))
        self.btn_screen = DashboardButton(pygame.Rect(bx + 3 * (bw + gap), by, 110, bh), "SCREENSHOT", "screenshot", accent_color=(39, 174, 96))
        self.buttons.extend([self.btn_pause, self.btn_step, self.btn_reset, self.btn_screen])

        # Row 2: Speed & Help (y = rect.y + 498)
        sy = self.rect.y + 498
        sx = self.rect.x + 20
        sw = 52
        for spd in self.speed_options:
            btn = DashboardButton(pygame.Rect(sx, sy, sw, 26), f"{spd}x", f"speed_{spd}")
            if spd == self.speed:
                btn.is_active = True
            self.buttons.append(btn)
            sx += sw + 7

        self.btn_help = DashboardButton(pygame.Rect(sx + 8, sy, 78, 26), "HELP (?)", "toggle_help", accent_color=(142, 68, 173))
        self.buttons.append(self.btn_help)

    def set_paused(self, paused: bool):
        """Sets the pause state and synchronizes pause/play button UI."""
        self.is_paused = paused
        self.btn_pause.label = "PLAY" if self.is_paused else "PAUSE"
        self.btn_pause.is_active = self.is_paused
        self.btn_pause.accent_color = (39, 174, 96) if self.is_paused else None

    def toggle_pause(self):
        """Toggles the pause state."""
        self.set_paused(not self.is_paused)

    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        """Handles mouse interactions and returns triggered action string."""
        for btn in self.buttons:
            action = btn.handle_event(event)
            if action:
                if action == "toggle_pause":
                    self.toggle_pause()
                elif action == "toggle_help":
                    self.show_help = not self.show_help
                    self.btn_help.is_active = self.show_help
                elif action.startswith("speed_"):
                    self.speed = int(action.split("_")[1])
                    for b in self.buttons:
                        if b.action_id.startswith("speed_"):
                            b.is_active = (b.action_id == action)
                return action
        return None

    def render(self, core: MonopolyCore):
        """Draws dashboard contents onto surface."""
        # Panel background
        pygame.draw.rect(self.surface, DASH_BG, self.rect, border_radius=10)
        pygame.draw.rect(self.surface, CARD_BORDER, self.rect, width=2, border_radius=10)

        # Header with tournament icon
        h_surf = self.font_header.render("TOURNAMENT RL AGENTS LEAGUE", True, TEXT_WHITE)
        self.surface.blit(h_surf, (self.rect.x + 20, self.rect.y + 14))

        if getattr(core, "house_rules", None) and core.house_rules.active_rules:
            sub_txt = f"House Rule: {core.house_rules.get_rule_summary()}"
            sub_col = (250, 204, 21)
        else:
            sub_txt = "MAPPO CTDE League • Unbounded Horizon Survival"
            sub_col = TEXT_MUTED
        sub_h = self.font_stat.render(sub_txt, True, sub_col)
        self.surface.blit(sub_h, (self.rect.x + 20, self.rect.y + 32))

        # Compute Net Worth Rankings (#1 to #4)
        net_worths = [core.get_player_net_worth(p) if not core.bankrupt[p] else -99999 for p in range(4)]
        sorted_indices = list(np.argsort(net_worths)[::-1])
        ranks = {p_id: rank + 1 for rank, p_id in enumerate(sorted_indices)}

        # 1. Four Player Agent Cards (y: 52 to 396)
        cy = self.rect.y + 52
        card_w = self.rect.width - 40
        card_h = 78
        card_gap = 8
        for p in range(4):
            self._render_agent_card(core, p, self.rect.x + 20, cy, card_w, card_h, ranks[p])
            cy += card_h + card_gap

        # 2. Bank Physical Scarcity Meter (y: 404 to 448)
        self._render_bank_scarcity(core, self.rect.x + 20, self.rect.y + 404, card_w, 44)

        # 3. Interactive Buttons (y: 458 to 524)
        for btn in self.buttons:
            btn.draw(self.surface, self.font_btn)

        # 4. Live Event Log Ticker (y: 536 to 788)
        log_y = self.rect.y + 536
        log_h = self.rect.bottom - log_y - 12
        self._render_event_log(core, self.rect.x + 20, log_y, card_w, log_h)

        # 5. Help Overlay Modal (if toggled)
        if self.show_help:
            self._render_help_overlay()

    def _render_agent_card(self, core: MonopolyCore, p: int, x: int, y: int, w: int, h: int, rank: int):
        """Draws individual player status card with financial stats, health bar, and monopoly portfolio."""
        card_rect = pygame.Rect(x, y, w, h)
        is_active = (core.active_decision_player == p and not core.bankrupt[p])

        # Card Background
        bg = CARD_BG_ACTIVE if is_active else CARD_BG
        pygame.draw.rect(self.surface, bg, card_rect, border_radius=6)
        border_col = PLAYER_COLORS[p] if is_active else CARD_BORDER
        pygame.draw.rect(self.surface, border_col, card_rect, width=2 if is_active else 1, border_radius=6)

        # 1. Name Pill (Luke, Graham, Pieter, Allison)
        p_name = core.get_player_name(p)
        p_txt = self.font_card_title.render(p_name, True, (255, 255, 255))
        pill_w = max(72, p_txt.get_width() + 16)
        pill_rect = pygame.Rect(x + 10, y + 6, pill_w, 20)
        pygame.draw.rect(self.surface, PLAYER_COLORS[p], pill_rect, border_radius=4)
        self.surface.blit(p_txt, (pill_rect.centerx - p_txt.get_width() // 2, pill_rect.centery - p_txt.get_height() // 2))

        # Rank Pill (#1, #2, #3, #4)
        if not core.bankrupt[p]:
            rank_col = GOLD_COLOR if rank == 1 else (170, 180, 195)
            r_txt = self.font_badge.render(f"#{rank}", True, (30, 35, 45))
            r_rect = pygame.Rect(pill_rect.right + 6, y + 6, 24, 20)
            pygame.draw.rect(self.surface, rank_col, r_rect, border_radius=3)
            self.surface.blit(r_txt, (r_rect.centerx - r_txt.get_width() // 2, r_rect.centery - r_txt.get_height() // 2))

        # Model / Policy Badge (e.g. 50M vs 100M)
        if hasattr(self, "agent_labels") and self.agent_labels and p < len(self.agent_labels) and self.agent_labels[p]:
            lbl = self.agent_labels[p]
            lbl_surf = self.font_badge.render(lbl, True, (255, 255, 255))
            lbl_bg = (142, 68, 173) if ("100M" in lbl or "100" in lbl) else (41, 128, 185)
            lbl_rect = pygame.Rect(x + w - lbl_surf.get_width() - 14, y + 6, lbl_surf.get_width() + 10, 20)
            pygame.draw.rect(self.surface, lbl_bg, lbl_rect, border_radius=4)
            self.surface.blit(lbl_surf, (lbl_rect.centerx - lbl_surf.get_width() // 2, lbl_rect.centery - lbl_surf.get_height() // 2))

        # Status Badges
        if core.bankrupt[p]:
            b_txt = self.font_card_title.render("BANKRUPT • ELIMINATED", True, TEXT_RED)
            self.surface.blit(b_txt, (pill_rect.right + 10, y + 8))
            pygame.draw.line(self.surface, (180, 50, 50), (x + 10, y + h // 2), (x + w - 10, y + h // 2), 2)
            return

        if core.in_jail[p]:
            j_txt = self.font_badge.render(f"JAIL ({core.jail_turns[p]}/3)", True, (240, 160, 40))
            self.surface.blit(j_txt, (pill_rect.right + 36, y + 9))

        # Active turn indicator badge
        if is_active:
            act_txt = self.font_badge.render("* ACTIVE TURN", True, PLAYER_COLORS[p])
            self.surface.blit(act_txt, (x + w - act_txt.get_width() - 10, y + 9))

        # 2. Financial Stats: Cash & Net Worth
        cash = core.cash[p]
        nw = core.get_player_net_worth(p)
        cash_col = TEXT_GREEN if cash >= 500 else (TEXT_YELLOW if cash >= 150 else TEXT_RED)

        stat1 = f"Cash: ${cash:,}   |   Net Worth: ${nw:,}"
        s1_surf = self.font_stat.render(stat1, True, cash_col)
        self.surface.blit(s1_surf, (x + 10, y + 29))

        # Mini Liquidity Bar
        bar_w = 110
        bar_h = 4
        bar_x = x + w - bar_w - 12
        bar_y = y + 33
        pygame.draw.rect(self.surface, (45, 52, 68), (bar_x, bar_y, bar_w, bar_h), border_radius=2)
        fill_ratio = min(1.0, max(0.0, cash / 2000.0))
        pygame.draw.rect(self.surface, cash_col, (bar_x, bar_y, int(bar_w * fill_ratio), bar_h), border_radius=2)

        # 3. Property & Building Counts
        props_owned = [t for t in PURCHASABLE_TILES if core.property_owner[t] == p]
        houses_count = sum(core.property_houses[t] for t in props_owned if 1 <= core.property_houses[t] <= 4)
        hotels_count = sum(1 for t in props_owned if core.property_houses[t] == 5)
        mtg_count = sum(1 for t in props_owned if core.property_mortgaged[t])

        stat2 = f"Props: {len(props_owned)}  |  Houses: {houses_count}  |  Hotels: {hotels_count}  |  Mtg: {mtg_count}"
        s2_surf = self.font_stat.render(stat2, True, TEXT_MUTED)
        self.surface.blit(s2_surf, (x + 10, y + 44))

        # 4. Monopoly / Color Group Swatch Portfolio Bar
        self._render_swatches_bar(core, p, x + 10, y + 59)

    def _render_swatches_bar(self, core: MonopolyCore, p: int, sx: int, sy: int):
        """Draws 10 color group swatches showing progress towards monopolies."""
        bx = sx
        sw_w = 40
        sw_h = 13

        for grp_key, label, total_tiles in SWATCH_GROUPS:
            tiles = COLOR_GROUPS[grp_key]
            owned = sum(1 for t in tiles if core.property_owner[t] == p)
            is_mono = (owned == total_tiles)

            base_color = COLOR_HEADER_MAP.get(grp_key, (100, 110, 130))
            swatch_rect = pygame.Rect(bx, sy, sw_w, sw_h)

            if is_mono:
                # Complete monopoly: glowing gold outline + full color fill
                pygame.draw.rect(self.surface, base_color, swatch_rect, border_radius=2)
                pygame.draw.rect(self.surface, GOLD_COLOR, swatch_rect, width=1, border_radius=2)
                lbl = self.font_badge.render(f"{label}*", True, (255, 255, 255))
            elif owned > 0:
                # Partial ownership: muted fill with count
                pygame.draw.rect(self.surface, (35, 42, 55), swatch_rect, border_radius=2)
                pygame.draw.rect(self.surface, base_color, swatch_rect, width=1, border_radius=2)
                lbl = self.font_badge.render(f"{label}:{owned}", True, (215, 225, 240))
            else:
                # 0 owned: dim outline
                pygame.draw.rect(self.surface, (26, 31, 42), swatch_rect, border_radius=2)
                pygame.draw.rect(self.surface, (45, 52, 66), swatch_rect, width=1, border_radius=2)
                lbl = self.font_badge.render(label, True, (80, 90, 110))

            self.surface.blit(lbl, (swatch_rect.centerx - lbl.get_width() // 2, swatch_rect.centery - lbl.get_height() // 2))
            bx += sw_w + 5

    def _render_bank_scarcity(self, core: MonopolyCore, x: int, y: int, w: int, h: int):
        """Draws visual bank physical inventory scarcity meters."""
        box = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.surface, CARD_BG, box, border_radius=6)
        pygame.draw.rect(self.surface, CARD_BORDER, box, width=1, border_radius=6)

        h_rem = core.bank_houses
        ht_rem = core.bank_hotels

        # Title
        t_surf = self.font_card_title.render("BANK PHYSICAL SCARCITY METERS", True, TEXT_WHITE)
        self.surface.blit(t_surf, (x + 12, y + 6))

        # Houses Bar
        h_ratio = h_rem / 32.0
        h_col = TEXT_RED if h_rem <= 4 else (TEXT_YELLOW if h_rem <= 12 else HOUSE_COLOR)
        hx = x + 12
        hy = y + 24
        hw = 210
        pygame.draw.rect(self.surface, (40, 48, 62), (hx, hy, hw, 10), border_radius=3)
        pygame.draw.rect(self.surface, h_col, (hx, hy, int(hw * h_ratio), 10), border_radius=3)
        h_lbl = self.font_stat.render(f"Houses: {h_rem}/32", True, h_col)
        self.surface.blit(h_lbl, (hx + hw + 8, hy - 2))

        # Hotels Bar
        ht_ratio = ht_rem / 12.0
        ht_col = TEXT_RED if ht_rem <= 2 else HOTEL_COLOR
        htx = x + 310
        htw = 90
        pygame.draw.rect(self.surface, (40, 48, 62), (htx, hy, htw, 10), border_radius=3)
        pygame.draw.rect(self.surface, ht_col, (htx, hy, int(htw * ht_ratio), 10), border_radius=3)
        ht_lbl = self.font_stat.render(f"Hotels: {ht_rem}/12", True, ht_col)
        self.surface.blit(ht_lbl, (htx + htw + 8, hy - 2))

    def _render_event_log(self, core: MonopolyCore, x: int, y: int, w: int, h: int):
        """Draws live scrolling event log ticker with category badge highlights."""
        box = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.surface, (16, 19, 26), box, border_radius=6)
        pygame.draw.rect(self.surface, CARD_BORDER, box, width=1, border_radius=6)

        title = self.font_card_title.render("LIVE EVENT TICKER", True, (130, 142, 165))
        self.surface.blit(title, (x + 12, y + 8))

        # Display last 12 events
        logs = core.event_log[-12:]
        ly = y + 26
        for log in logs:
            badge_str, col = self._categorize_log(log)

            # Badge pill
            b_txt = self.font_badge.render(badge_str, True, col)
            self.surface.blit(b_txt, (x + 10, ly))

            # Log message
            disp_log = log
            if len(disp_log) > 60:
                disp_log = disp_log[:57] + "..."
            l_surf = self.font_log.render(disp_log, True, (215, 220, 230))
            self.surface.blit(l_surf, (x + 60, ly))
            ly += 18

    def _categorize_log(self, log: str) -> Tuple[str, Tuple[int, int, int]]:
        """Categorizes log message into icon tag and highlight color."""
        if "CHAMPION" in log or "wins" in log:
            return "[WIN]", GOLD_COLOR
        elif "BANKRUPT" in log:
            return "[BANKRUPT]", TEXT_RED
        elif "TRADE" in log:
            return "[TRADE]", (46, 204, 113)
        elif "AUCTION" in log or "bids" in log:
            return "[AUCT]", (243, 156, 18)
        elif "JAIL" in log:
            return "[JAIL]", (241, 196, 15)
        elif "built a house" in log or "HOTEL" in log:
            return "[BUILD]", (46, 204, 113)
        elif "rent" in log:
            return "[RENT]", (231, 76, 60)
        elif "rolled" in log:
            return "[ROLL]", (52, 152, 219)
        elif "purchased" in log:
            return "[BUY]", (155, 89, 182)
        return "[INFO]", TEXT_MUTED

    def _render_help_overlay(self):
        """Renders semi-transparent help dialog overlaying shortcuts."""
        overlay = pygame.Surface((self.surface.get_width(), self.surface.get_height()), pygame.SRCALPHA)
        overlay.fill((10, 12, 16, 210))

        dialog_w, dialog_h = 460, 280
        cx = self.surface.get_width() // 2
        cy = self.surface.get_height() // 2
        d_rect = pygame.Rect(cx - dialog_w // 2, cy - dialog_h // 2, dialog_w, dialog_h)

        pygame.draw.rect(overlay, (24, 28, 38), d_rect, border_radius=10)
        pygame.draw.rect(overlay, (70, 85, 115), d_rect, width=2, border_radius=10)

        # Title
        t_surf = self.font_help_title.render("KEYBOARD SHORTCUTS & CONTROLS", True, GOLD_COLOR)
        overlay.blit(t_surf, (cx - t_surf.get_width() // 2, d_rect.y + 16))

        shortcuts = [
            ("Space", "Pause / Play simulation"),
            ("S", "Step forward by 1 turn decision"),
            ("R", "Reset game to initial tournament state"),
            ("1, 2, 3, 4, 5", "Set simulation speed multiplier (1x, 2x, 5x, 20x, 100x)"),
            ("C / SCREENSHOT", "Save high-resolution screenshot"),
            ("H / HELP", "Toggle this shortcuts guide"),
            ("Esc", "Quit application cleanly"),
        ]

        sy = d_rect.y + 48
        for key_s, desc in shortcuts:
            k_surf = self.font_btn.render(f"[{key_s}]", True, (52, 152, 219))
            d_surf = self.font_help_body.render(desc, True, TEXT_WHITE)
            overlay.blit(k_surf, (d_rect.x + 20, sy))
            overlay.blit(d_surf, (d_rect.x + 130, sy))
            sy += 25

        hint = self.font_stat.render("Click HELP or press 'H' to close", True, TEXT_MUTED)
        overlay.blit(hint, (cx - hint.get_width() // 2, d_rect.bottom - 26))

        self.surface.blit(overlay, (0, 0))
