"""Interactive Real-Time Pygame GUI for Tournament MARL Monopoly.

Visualizes 4 autonomous RL agents playing tournament Monopoly under MAPPO CTDE.
Renders 11x11 board, property color bands, non-overlapping tokens, buildings,
3D dice rolls, physical scarcity, and real-time interactive dashboard.
"""

import sys
import os
import argparse
import time
import pygame
import torch
import numpy as np

from monopoly_marl.engine.monopoly_core import MonopolyCore
from monopoly_marl.models.actor_network import ActorNetwork
from monopoly_marl.gui.renderer import MonopolyRenderer, BG_COLOR
from monopoly_marl.gui.dashboard import MonopolyDashboard


def parse_args():
    parser = argparse.ArgumentParser(description="Monopoly MARL Real-Time Pygame GUI")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained Actor checkpoint")
    parser.add_argument("--max_rounds", type=int, default=None, help="Max game rounds (None for unbounded horizon survival)")
    parser.add_argument("--speed", type=int, default=5, choices=[1, 2, 5, 20, 100], help="Initial simulation speed multiplier")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic policy actions (default: True)")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false", help="Use stochastic action sampling")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    return parser.parse_args()


def load_agent_policies(checkpoint_path: str = None, target_obs_dim: int = 399, device: torch.device = torch.device("cpu")):
    """Loads 4 identical blueprint actor networks, optionally restored from checkpoint with weight adaptation."""
    policies = []
    state_dict = None
    ckpt_obs_dim = target_obs_dim

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading policy checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)
        if "trunk.0.weight" in state_dict:
            ckpt_obs_dim = state_dict["trunk.0.weight"].shape[1]

    for p_id in range(4):
        actor = ActorNetwork(obs_dim=target_obs_dim).to(device)
        if state_dict is not None:
            new_sd = actor.state_dict()
            old_auc_dim = state_dict["head_auction.weight"].shape[0] if "head_auction.weight" in state_dict else 5
            curr_auc_dim = new_sd["head_auction.weight"].shape[0]

            for k in state_dict:
                if k == "trunk.0.weight" and ckpt_obs_dim != target_obs_dim:
                    new_sd[k][:, :ckpt_obs_dim] = state_dict[k]
                    new_sd[k][:, ckpt_obs_dim:] = 0.0
                elif k == "head_auction.weight" and old_auc_dim != curr_auc_dim:
                    new_sd[k][:old_auc_dim, :] = state_dict[k]
                    new_sd[k][old_auc_dim:, :] = 0.0
                elif k == "head_auction.bias" and old_auc_dim != curr_auc_dim:
                    new_sd[k][:old_auc_dim] = state_dict[k]
                    new_sd[k][old_auc_dim:] = -0.1
                elif k in new_sd and new_sd[k].shape == state_dict[k].shape:
                    new_sd[k] = state_dict[k]
            actor.load_state_dict(new_sd)
        actor.eval()
        policies.append(actor)

    return policies


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Initialize Pygame Window
    pygame.init()
    pygame.font.init()

    win_width = 1380
    win_height = 840
    screen = pygame.display.set_mode((win_width, win_height))
    pygame.display.set_caption("Tournament MARL Monopoly (MAPPO CTDE + League Training)")
    clock = pygame.time.Clock()

    # Core Environment
    core = MonopolyCore(seed=args.seed, max_rounds=args.max_rounds)

    # Renderer & Dashboard
    renderer = MonopolyRenderer(screen, board_size=800, offset_x=20, offset_y=20)
    dashboard = MonopolyDashboard(screen, x=840, y=20, width=520, height=800)
    dashboard.speed = args.speed
    for b in dashboard.buttons:
        if b.action_id.startswith("speed_"):
            b.is_active = (b.action_id == f"speed_{dashboard.speed}")

    # Load 4 Identical Blueprint Autonomous Agents
    target_obs_dim = core.get_actor_observation(0).shape[0]
    policies = load_agent_policies(args.checkpoint, target_obs_dim=target_obs_dim, device=device)

    action_keys = [
        "core", "auction", "property", "trade_target",
        "trade_offer_prop", "trade_offer_cash", "trade_req_prop",
        "trade_req_cash", "trade_response"
    ]

    running = True
    step_requested = False
    last_step_time = time.time()
    game_over_captured = False
    last_bankrupt_count = 0

    print("=== Pygame MARL Monopoly GUI Running ===")
    print(f"Agents: {', '.join(core.player_names)} (P0..P3)")
    print("Rule: Unbounded Horizon Survival (Winner: Last Agent Standing)")
    print("Controls: Space: Pause/Play | S: Step | R: Reset | 1-5: Speed | C: Screenshot | Esc: Quit")

    while running:
        current_time = time.time()

        # Handle Pygame Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                elif event.key == pygame.K_SPACE:
                    dashboard.is_paused = not dashboard.is_paused
                    dashboard.btn_pause.label = "PLAY" if dashboard.is_paused else "PAUSE"
                    dashboard.btn_pause.is_active = dashboard.is_paused
                elif event.key == pygame.K_s:
                    step_requested = True
                elif event.key == pygame.K_r:
                    core.reset(seed=args.seed)
                    game_over_captured = False
                    last_bankrupt_count = 0
                elif event.key == pygame.K_1:
                    dashboard.speed = 1
                elif event.key == pygame.K_2:
                    dashboard.speed = 2
                elif event.key == pygame.K_3:
                    dashboard.speed = 5
                elif event.key == pygame.K_4:
                    dashboard.speed = 20
                elif event.key == pygame.K_5:
                    dashboard.speed = 100
                elif event.key == pygame.K_h:
                    dashboard.show_help = not dashboard.show_help
                    dashboard.btn_help.is_active = dashboard.show_help
                elif event.key == pygame.K_c:
                    # Save screenshot
                    os.makedirs("screenshots", exist_ok=True)
                    s_path = f"screenshots/monopoly_gui_{int(time.time())}.png"
                    pygame.image.save(screen, s_path)
                    print(f"Screenshot saved to: {s_path}")
                    core.event_log.append(f"Screenshot saved: {os.path.basename(s_path)}")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Board tile card inspection click
                if renderer.handle_click(event.pos):
                    continue

            # Check dashboard button clicks
            action = dashboard.handle_event(event)
            if action == "step":
                step_requested = True
            elif action == "reset":
                core.reset(seed=args.seed)
                game_over_captured = False
                last_bankrupt_count = 0
            elif action == "screenshot":
                os.makedirs("screenshots", exist_ok=True)
                s_path = f"screenshots/monopoly_gui_{int(time.time())}.png"
                pygame.image.save(screen, s_path)
                print(f"Screenshot saved to: {s_path}")
                core.event_log.append(f"Screenshot saved: {os.path.basename(s_path)}")

        # Decide whether to step simulation
        step_interval = 1.0 / max(1, dashboard.speed * 2)
        should_step = (not dashboard.is_paused and (current_time - last_step_time >= step_interval)) or step_requested

        if should_step:
            step_requested = False
            last_step_time = current_time

            # Active decision agent takes turn
            p_act = core.active_decision_player
            if not core.bankrupt[p_act] and core.phase != 6:  # 6 = PHASE_GAME_OVER
                obs_vec = core.get_actor_observation(p_act)
                masks = core.get_action_masks(p_act)

                # Format tensors for policy forward pass
                obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
                masks_t = {
                    k: torch.tensor(masks[k], dtype=torch.int8, device=device).unsqueeze(0)
                    for k in action_keys
                }

                policy = policies[p_act]
                with torch.no_grad():
                    actions_dict, _, _ = policy(obs_t, masks_t, deterministic=args.deterministic)

                # Convert to integer actions
                act = {k: int(actions_dict[k].item()) for k in action_keys}

                # Step state machine
                core.step({f"player_{p_act}": act})

                # Check for new bankruptcies
                curr_bankrupt_count = sum(1 for b in core.bankrupt if b)
                if curr_bankrupt_count > last_bankrupt_count:
                    eliminated = [core.get_player_name(pl) for pl in range(4) if core.bankrupt[pl]]
                    survivors = [core.get_player_name(pl) for pl in range(4) if not core.bankrupt[pl]]
                    print(f"[ROUND {core.round_count} - STEP {core.turn_step}] Player eliminated! Eliminated: {', '.join(eliminated)} | Survivors remaining ({len(survivors)}): {', '.join(survivors)}")
                    last_bankrupt_count = curr_bankrupt_count

        # Clear background
        screen.fill(BG_COLOR)

        # Render Board & Dashboard
        renderer.render(core)
        dashboard.render(core)

        # Check Tournament Game Over (Last Agent Standing)
        if core.phase == 6 and not game_over_captured:
            game_over_captured = True
            active = [pl for pl in range(4) if not core.bankrupt[pl]]
            winner_name = core.get_player_name(active[0]) if len(active) == 1 else "Unknown"
            winner_nw = core.get_player_net_worth(active[0]) if len(active) == 1 else 0
            print("\n" + "=" * 55)
            print(f"*** TOURNAMENT CHAMPION: {winner_name.upper()} ***")
            print("Rule Condition: Last Agent Standing (Survival Victory)")
            print(f"Surviving Rounds: {core.round_count} | Total Turn Steps: {core.turn_step}")
            print(f"Final Net Worth: ${winner_nw:,}")
            print("=" * 55 + "\n")
            os.makedirs("screenshots", exist_ok=True)
            v_path = f"screenshots/tournament_victory_{int(time.time())}.png"
            pygame.image.save(screen, v_path)
            print(f"Victory screenshot captured: {v_path}")
            core.event_log.append(f"Auto-saved: {os.path.basename(v_path)}")

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
