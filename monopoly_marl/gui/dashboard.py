"""Real-Time Interactive Dashboard for MARL Monopoly.

Displays live agent statistics (Cash, Net Worth, Properties, Jail, Bankrupt),
physical bank scarcity meters, scrolling event log ticker, and interactive controls
(Play/Pause, Step, Speed selector, Reset).
"""

from typing import Dict, List, Optional, Tuple, Any
import pygame

from monopoly_marl.engine.monopoly_core import MonopolyCore
from monopoly_marl.engine.board_data import PURCHASABLE_TILES
from monopoly_marl.gui.renderer import PLAYER_COLORS, HOUSE_COLOR, HOTEL_COLOR

DASH_BG = (18, 21, 28)
CARD_BG = (28, 33, 44)
CARD_BORDER = (45, 52, 68)
TEXT_WHITE = (240, 243, 248)
TEXT_MUTED = (145, 155, 175)
BTN_BG = (40, 48, 64)
BTN_HOVER = (60, 72, 96)
BTN_ACTIVE = (52, 152, 219)


class DashboardButton:
    """Clickable UI Button."""

    def __init__(self, rect: pygame.Rect, label: str, action_id: str):
        self.rect = rect
        self.label = label
        self.action_id = action_id
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
        color = BTN_ACTIVE if self.is_active else (BTN_HOVER if self.is_hovered else BTN_BG)
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        pygame.draw.rect(surface, (70, 80, 105), self.rect, width=1, border_radius=6)

        txt_surf = font.render(self.label, True, TEXT_WHITE)
        surface.blit(txt_surf, (self.rect.centerx - txt_surf.get_width() // 2, self.rect.centery - txt_surf.get_height() // 2))


class MonopolyDashboard:
    """Right-Hand Side Dashboard Panel."""

    def __init__(self, surface: pygame.Surface, x: int = 850, y: int = 20, width: int = 500, height: int = 800):
        self.surface = surface
        self.rect = pygame.Rect(x, y, width, height)

        # Fonts
        pygame.font.init()
        self.font_header = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_card_title = pygame.font.SysFont("Arial", 13, bold=True)
        self.font_stat = pygame.font.SysFont("Arial", 12)
        self.font_btn = pygame.font.SysFont("Arial", 12, bold=True)
        self.font_log = pygame.font.SysFont("Arial", 11)

        # Interactive states
        self.is_paused = False
        self.speed = 2  # 1, 2, 5, 20, 100
        self.speed_options = [1, 2, 5, 20, 100]

        # Setup Buttons
        self.buttons: List[DashboardButton] = []
        self._init_buttons()

    def _init_buttons(self):
        bx = self.rect.x + 20
        by = self.rect.y + 440
        bw = 85
        bh = 32

        self.btn_pause = DashboardButton(pygame.Rect(bx, by, bw, bh), "PAUSE", "toggle_pause")
        self.btn_step = DashboardButton(pygame.Rect(bx + bw + 10, by, bw, bh), "STEP", "step")
        self.btn_reset = DashboardButton(pygame.Rect(bx + 2 * (bw + 10), by, bw, bh), "RESET", "reset")
        self.buttons.extend([self.btn_pause, self.btn_step, self.btn_reset])

        # Speed buttons
        sx = self.rect.x + 20
        sy = by + bh + 14
        sw = 50
        for spd in self.speed_options:
            btn = DashboardButton(pygame.Rect(sx, sy, sw, 26), f"{spd}x", f"speed_{spd}")
            if spd == self.speed:
                btn.is_active = True
            self.buttons.append(btn)
            sx += sw + 8

    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        """Handles mouse interactions and returns triggered action string."""
        for btn in self.buttons:
            action = btn.handle_event(event)
            if action:
                if action == "toggle_pause":
                    self.is_paused = not self.is_paused
                    self.btn_pause.label = "PLAY" if self.is_paused else "PAUSE"
                    self.btn_pause.is_active = self.is_paused
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
        pygame.draw.rect(self.surface, DASH_BG, self.rect, border_radius=8)
        pygame.draw.rect(self.surface, CARD_BORDER, self.rect, width=2, border_radius=8)

        # Header
        h_surf = self.font_header.render("TOURNAMENT RL AGENTS LEAGUE", True, TEXT_WHITE)
        self.surface.blit(h_surf, (self.rect.x + 20, self.rect.y + 16))

        # 1. Four Player Agent Cards
        cy = self.rect.y + 48
        card_w = self.rect.width - 40
        card_h = 76
        for p in range(4):
            self._render_agent_card(core, p, self.rect.x + 20, cy, card_w, card_h)
            cy += card_h + 10

        # 2. Bank Physical Scarcity Meter
        self._render_bank_scarcity(core, self.rect.x + 20, cy + 4, card_w, 42)

        # 3. Interactive Buttons
        for btn in self.buttons:
            btn.draw(self.surface, self.font_btn)

        # 4. Live Event Log Ticker
        log_y = self.rect.y + 535
        log_h = self.rect.bottom - log_y - 20
        self._render_event_log(core, self.rect.x + 20, log_y, card_w, log_h)

    def _render_agent_card(self, core: MonopolyCore, p: int, x: int, y: int, w: int, h: int):
        """Draws individual player status card."""
        card_rect = pygame.Rect(x, y, w, h)
        is_active = (core.active_decision_player == p and not core.bankrupt[p])

        # Card bg
        bg = (38, 45, 60) if is_active else CARD_BG
        pygame.draw.rect(self.surface, bg, card_rect, border_radius=6)
        border_col = PLAYER_COLORS[p] if is_active else CARD_BORDER
        pygame.draw.rect(self.surface, border_col, card_rect, width=2 if is_active else 1, border_radius=6)

        # Player Pill
        pill_rect = pygame.Rect(x + 10, y + 10, 65, 20)
        pygame.draw.rect(self.surface, PLAYER_COLORS[p], pill_rect, border_radius=4)
        p_name = f"Agent {p}"
        p_txt = self.font_card_title.render(p_name, True, (255, 255, 255))
        self.surface.blit(p_txt, (pill_rect.centerx - p_txt.get_width() // 2, pill_rect.centery - p_txt.get_height() // 2))

        # Bankruptcy Badge
        if core.bankrupt[p]:
            b_txt = self.font_card_title.render("BANKRUPT", True, (230, 60, 60))
            self.surface.blit(b_txt, (x + 85, y + 12))
            return

        # Jail badge
        if core.in_jail[p]:
            j_txt = self.font_stat.render(f"JAIL ({core.jail_turns[p]}/3)", True, (240, 160, 40))
            self.surface.blit(j_txt, (x + 85, y + 12))

        # Financial Stats: Cash & Net Worth
        cash = core.cash[p]
        nw = core.get_player_net_worth(p)
        stat1 = f"Cash: ${cash:,}   |   Net Worth: ${nw:,}"
        s1_surf = self.font_stat.render(stat1, True, (46, 204, 113) if cash >= 0 else (231, 76, 60))
        self.surface.blit(s1_surf, (x + 10, y + 36))

        # Property & Building counts
        props_owned = [t for t in PURCHASABLE_TILES if core.property_owner[t] == p]
        houses_count = sum(core.property_houses[t] for t in props_owned if 1 <= core.property_houses[t] <= 4)
        hotels_count = sum(1 for t in props_owned if core.property_houses[t] == 5)
        mtg_count = sum(1 for t in props_owned if core.property_mortgaged[t])

        stat2 = f"Props: {len(props_owned)}  |  Houses: {houses_count}  |  Hotels: {hotels_count}  |  Mtg: {mtg_count}"
        s2_surf = self.font_stat.render(stat2, True, TEXT_MUTED)
        self.surface.blit(s2_surf, (x + 10, y + 54))

    def _render_bank_scarcity(self, core: MonopolyCore, x: int, y: int, w: int, h: int):
        """Draws physical bank scarcity box."""
        box = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.surface, CARD_BG, box, border_radius=6)
        pygame.draw.rect(self.surface, CARD_BORDER, box, width=1, border_radius=6)

        h_rem = core.bank_houses
        ht_rem = core.bank_hotels
        h_color = (231, 76, 60) if h_rem == 0 else HOUSE_COLOR
        ht_color = (231, 76, 60) if ht_rem == 0 else HOTEL_COLOR

        txt = f"BANK SCARCITY:   Houses: {h_rem}/32   |   Hotels: {ht_rem}/12"
        surf = self.font_card_title.render(txt, True, TEXT_WHITE)
        self.surface.blit(surf, (box.centerx - surf.get_width() // 2, box.centery - surf.get_height() // 2))

    def _render_event_log(self, core: MonopolyCore, x: int, y: int, w: int, h: int):
        """Draws live scrolling event log ticker."""
        box = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.surface, (22, 26, 35), box, border_radius=6)
        pygame.draw.rect(self.surface, CARD_BORDER, box, width=1, border_radius=6)

        title = self.font_stat.render("EVENT LOG TICKER", True, (120, 130, 150))
        self.surface.blit(title, (x + 10, y + 8))

        # Display last 9 events
        logs = core.event_log[-9:]
        ly = y + 28
        for log in logs:
            col = (200, 205, 215)
            if "BANKRUPT" in log:
                col = (235, 87, 87)
            elif "AUCTION" in log:
                col = (242, 153, 74)
            elif "TRADE" in log:
                col = (111, 207, 151)
            elif "JAIL" in log:
                col = (242, 201, 76)

            # Truncate text if long
            disp_log = log if len(log) <= 65 else log[:62] + "..."
            l_surf = self.font_log.render(disp_log, True, col)
            self.surface.blit(l_surf, (x + 10, ly))
            ly += 18
