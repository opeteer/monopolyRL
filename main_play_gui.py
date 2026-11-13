"""Interactive Real-Time Pygame GUI for Tournament MARL Monopoly.

Visualizes 4 autonomous RL agents playing tournament Monopoly under MAPPO CTDE.
Renders 11x11 board, property color bands, non-overlapping tokens, buildings,
3D dice rolls, physical scarcity, and real-time interactive dashboard.
"""

import sys
import os
import argparse
import time
from typing import Optional, List, Union
import pygame
import torch
import numpy as np

from monopoly_marl.engine.monopoly_core import MonopolyCore
from monopoly_marl.models.actor_network import ActorNetwork
from monopoly_marl.gui.renderer import MonopolyRenderer, BG_COLOR
from monopoly_marl.gui.dashboard import MonopolyDashboard


def parse_args():
    parser = argparse.ArgumentParser(description="Monopoly MARL Real-Time Pygame GUI")
    parser.add_argument("--checkpoint", type=str, default=None, help="Default path to trained Actor checkpoint")
    parser.add_argument("--checkpoint_p0", type=str, default=None, help="Checkpoint for Player 0 (Luke)")
    parser.add_argument("--checkpoint_p1", type=str, default=None, help="Checkpoint for Player 1 (Graham)")
    parser.add_argument("--checkpoint_p2", type=str, default=None, help="Checkpoint for Player 2 (Pieter)")
    parser.add_argument("--checkpoint_p3", type=str, default=None, help="Checkpoint for Player 3 (Allison)")
    parser.add_argument("--player_labels", type=str, default=None, help="Comma-separated model labels e.g. '50M,100M,50M,50M'")
    parser.add_argument("--max_rounds", type=int, default=None, help="Max game rounds (None for unbounded horizon survival)")
    parser.add_argument("--speed", type=int, default=5, choices=[1, 2, 5, 20, 100], help="Initial simulation speed multiplier")
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use strict argmax deterministic actions (default: False / stochastic mixed-strategy)")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false", help="Use stochastic policy sampling (default)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--auto_quit_delay", type=float, default=None, help="Seconds to show victory screen before auto-quitting (None to stay open)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--house_rules", type=str, default=None, help="Custom house rules (e.g. 'half_cash', 'half_starting_cash')")
    parser.add_argument("--half_cash", action="store_true", default=False, help="House rule convenience flag: start with $750 cash (half of standard $1,500)")
    parser.add_argument("--auto_play", action="store_true", default=False, help="Start playing immediately without waiting for manual unpause")
    parser.add_argument("--fixed_first_player", action="store_false", dest="randomize_first_player", default=True, help="Keep Player 0 as fixed starting player instead of randomizing first seat")
    return parser.parse_args()


def load_single_policy(checkpoint_path: Optional[str] = None, target_obs_dim: int = 500, device: torch.device = torch.device("cpu")) -> ActorNetwork:
    """Loads a single actor network, adapting dimensions if needed from checkpoint."""
    actor = ActorNetwork(obs_dim=target_obs_dim).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)
        ckpt_obs_dim = target_obs_dim
        if "trunk.0.weight" in state_dict:
            ckpt_obs_dim = state_dict["trunk.0.weight"].shape[1]
        new_sd = actor.state_dict()
        old_auc_dim = state_dict["head_auction.weight"].shape[0] if "head_auction.weight" in state_dict else 5
        curr_auc_dim = new_sd["head_auction.weight"].shape[0]
        old_prop_dim = state_dict["head_property.weight"].shape[0] if "head_property.weight" in state_dict else 160
        curr_prop_dim = new_sd["head_property.weight"].shape[0]

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
            elif k == "head_property.weight" and old_prop_dim != curr_prop_dim:
                new_sd[k][:old_prop_dim, :] = state_dict[k]
                new_sd[k][old_prop_dim:, :] = 0.0
            elif k == "head_property.bias" and old_prop_dim != curr_prop_dim:
                new_sd[k][:old_prop_dim] = state_dict[k]
                new_sd[k][old_prop_dim:] = 0.0
            elif k in new_sd and new_sd[k].shape == state_dict[k].shape:
                new_sd[k] = state_dict[k]
        actor.load_state_dict(new_sd)
    actor.eval()
    return actor


def load_agent_policies(checkpoint_paths=None, target_obs_dim: int = 500, device: torch.device = torch.device("cpu")):
    """Loads 4 blueprint actor networks, supporting uniform or per-player distinct checkpoints."""
    if isinstance(checkpoint_paths, (list, tuple)):
        paths = list(checkpoint_paths)
        while len(paths) < 4:
            paths.append(paths[0] if paths else None)
    else:
        paths = [checkpoint_paths] * 4

    policies = []
    for p_id in range(4):
        p_ckpt = paths[p_id]
        if p_ckpt:
            print(f"Loading Player {p_id} policy from: {p_ckpt}")
        policies.append(load_single_policy(p_ckpt, target_obs_dim=target_obs_dim, device=device))
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

    # Assemble custom house rules if specified
    active_house_rules = args.house_rules
    if args.half_cash:
        active_house_rules = "half_cash" if not active_house_rules else f"{active_house_rules},half_cash"

    # Core Environment (randomize first-mover seat fairly in GUI tournaments unless --fixed_first_player)
    core = MonopolyCore(
        seed=args.seed,
        max_rounds=args.max_rounds,
        house_rules=active_house_rules,
        randomize_first_player=args.randomize_first_player,
    )

    # Resolve per-player checkpoints
    ckpt_paths = [
        args.checkpoint_p0 or args.checkpoint,
        args.checkpoint_p1 or args.checkpoint,
        args.checkpoint_p2 or args.checkpoint,
        args.checkpoint_p3 or args.checkpoint,
    ]

    # Resolve agent model labels
    if args.player_labels:
        agent_labels = [s.strip() for s in args.player_labels.split(",")]
    else:
        agent_labels = []
        for idx, cp in enumerate(ckpt_paths):
            if cp and ("04882" in cp or "100" in cp):
                agent_labels.append("100M")
            elif cp and ("02441" in cp or "50" in cp):
                agent_labels.append("50M")
            elif cp:
                agent_labels.append(os.path.basename(cp).split(".")[0])
            else:
                agent_labels.append(f"P{idx}")

    # Renderer & Dashboard
    renderer = MonopolyRenderer(screen, board_size=800, offset_x=20, offset_y=20)
    dashboard = MonopolyDashboard(screen, x=840, y=20, width=520, height=800, agent_labels=agent_labels)
    dashboard.speed = args.speed
    for b in dashboard.buttons:
        if b.action_id.startswith("speed_"):
            b.is_active = (b.action_id == f"speed_{dashboard.speed}")
    if args.auto_play:
        dashboard.set_paused(False)

    # Load Autonomous Agents (Uniform or Distinct Models)
    target_obs_dim = core.get_actor_observation(0).shape[0]
    policies = load_agent_policies(ckpt_paths, target_obs_dim=target_obs_dim, device=device)

    action_keys = [
        "core", "auction", "property", "trade_target",
        "trade_offer_prop", "trade_offer_cash", "trade_req_prop",
        "trade_req_cash", "trade_response"
    ]

    running = True
    step_requested = False
    last_step_time = time.time()
    game_over_captured = False
    victory_time = None
    last_bankrupt_count = 0
    last_logged_ev_idx = 0

    print("=== Pygame MARL Monopoly GUI Running ===")
    print(f"Active Match Seed (Hardware Fluctuation Entropy): {core.current_seed}")
    print("=== Agent Model Lineup ===")
    for p_id in range(4):
        p_name = core.get_player_name(p_id)
        ckpt_disp = ckpt_paths[p_id] if ckpt_paths[p_id] else "Default Initialized"
        print(f"  P{p_id} ({p_name}): [{agent_labels[p_id]}] -> {ckpt_disp}")
    print("Rule: Unbounded Horizon Survival (Winner: Last Agent Standing)")
    print("Controls: Space: Pause/Play | S: Step | R: Reset | 1-5: Speed | C: Screenshot | Esc: Quit")
    print("Initial State: PAUSED (Game starts in paused mode; press PLAY or Space to begin)")
    core.event_log.append("Game initialized in PAUSED state. Press PLAY or Space to begin.")

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
                    dashboard.toggle_pause()
                elif event.key == pygame.K_s:
                    step_requested = True
                elif event.key == pygame.K_r:
                    core.reset(seed=args.seed)
                    renderer.reset(core)
                    dashboard.set_paused(True)
                    game_over_captured = False
                    last_bankrupt_count = 0
                    print(f"[RESET] New Match Seed (Hardware Fluctuation Entropy): {core.current_seed}")
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
                renderer.reset(core)
                dashboard.set_paused(True)
                game_over_captured = False
                last_bankrupt_count = 0
                print(f"[RESET] New Match Seed (Hardware Fluctuation Entropy): {core.current_seed}")
            elif action == "screenshot":
                os.makedirs("screenshots", exist_ok=True)
                s_path = f"screenshots/monopoly_gui_{int(time.time())}.png"
                pygame.image.save(screen, s_path)
                print(f"Screenshot saved to: {s_path}")
                core.event_log.append(f"Screenshot saved: {os.path.basename(s_path)}")

        # Decide whether to step simulation (smooth pacing allows hop animation to complete at speeds <= 5)
        step_interval = 1.0 / max(1, dashboard.speed * 2)
        is_anim = renderer.is_animating() if dashboard.speed <= 5 else False
        should_step = (not dashboard.is_paused and (current_time - last_step_time >= step_interval) and not is_anim) or step_requested

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

                while last_logged_ev_idx < len(core.event_log):
                    ev = core.event_log[last_logged_ev_idx]
                    last_logged_ev_idx += 1
                    if any(k in ev for k in ["TRADE ACCEPTED", "HOTEL", "IS BANKRUPT"]):
                        print(f"[R{core.round_count:<3} S{core.turn_step:<4}] {ev}")

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
        renderer.render(core, speed=dashboard.speed, is_paused=dashboard.is_paused)
        dashboard.render(core)

        # Check Tournament Game Over (Last Agent Standing)
        if core.phase == 6 and not game_over_captured:
            game_over_captured = True
            victory_time = current_time
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

            with open("tournament_gui_match.log", "w") as f_log:
                f_log.write(f"=== TOURNAMENT GUI MATCH LOG (Seed: {core.current_seed}) ===\n")
                for ev_line in core.event_log:
                    f_log.write(ev_line + "\n")
            print(f"Full game event log saved to: tournament_gui_match.log ({len(core.event_log)} events)")

        if game_over_captured and args.auto_quit_delay is not None and victory_time is not None:
            if current_time - victory_time >= args.auto_quit_delay:
                print(f"Auto-closing GUI tournament after {args.auto_quit_delay}s victory display.")
                running = False

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
