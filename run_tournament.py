"""Autonomous Tournament Runner for Tournament MARL Monopoly.

Executes tournament matches between 4 MARL agents until 1 single sole winner remains.
Always driven 100% by the real-time hardware fluctuation entropy generator.
"""

import os
import sys
import argparse
import time
import torch
import numpy as np

from monopoly_marl.engine.monopoly_core import MonopolyCore, PHASE_GAME_OVER, generate_hardware_entropy_seed
from monopoly_marl.engine.board_data import COLOR_GROUPS
from monopoly_marl.models.actor_network import ActorNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous Tournament Runner")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/main_iter_02441.pt", help="Path to Actor checkpoint")
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use argmax greedy actions")
    parser.add_argument("--max_steps", type=int, default=5000, help="Maximum steps guard")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--house_rules", type=str, default=None, help="Custom house rules (e.g. 'half_cash', 'half_starting_cash')")
    parser.add_argument("--half_cash", action="store_true", default=False, help="House rule convenience flag: start with $750 cash (half of standard $1,500)")
    parser.add_argument("--fixed_first_player", action="store_false", dest="randomize_first_player", default=True, help="Keep Player 0 as fixed starting player instead of randomizing first seat")
    return parser.parse_args()


def load_actor(checkpoint_path: str, obs_dim: int = 500, device: torch.device = torch.device("cpu")):
    actor = ActorNetwork(obs_dim=obs_dim).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        actor.load_state_dict(state_dict, strict=False)
    actor.eval()
    return actor


def run_tournament_match(args):
    device = torch.device(args.device)
    actor = load_actor(args.checkpoint, device=device)

    # Assemble custom house rules if specified
    active_house_rules = args.house_rules
    if args.half_cash:
        active_house_rules = "half_cash" if not active_house_rules else f"{active_house_rules},half_cash"

    # Core environment initialized strictly with seed=None -> generate_hardware_entropy_seed()
    core = MonopolyCore(
        seed=None,
        max_rounds=None,
        house_rules=active_house_rules,
        randomize_first_player=args.randomize_first_player,
    )
    hw_seed = core.current_seed

    print("=" * 65)
    print("      TOURNAMENT MARL MONOPOLY: SOLE SURVIVOR CHAMPIONSHIP")
    print("=" * 65)
    print(f"Active Seed (Hardware Fluctuation Entropy): {hw_seed} (0x{hw_seed:08X})")
    print(f"Policy Checkpoint : {args.checkpoint}")
    print(f"Sampling Mode     : {'Deterministic (Argmax)' if args.deterministic else 'Stochastic (Mixed-Strategy)'}")
    print(f"First Mover       : P{core.starting_player} ({core.get_player_name(core.starting_player)}) [{'Randomized' if args.randomize_first_player else 'Fixed P0'}]")
    print(f"Tournament Rules  : Unbounded Horizon Survival (Plays until 1 sole winner)")
    print("-" * 65)

    action_keys = [
        "core", "auction", "property", "trade_target",
        "trade_offer_prop", "trade_offer_cash", "trade_req_prop",
        "trade_req_cash", "trade_response"
    ]

    step_count = 0
    last_bankrupt_count = 0
    trade_props = 0
    trade_accs = 0

    start_time = time.time()

    while core.phase != PHASE_GAME_OVER and step_count < args.max_steps:
        step_count += 1
        p_act = core.active_decision_player
        if core.bankrupt[p_act]:
            continue

        obs_vec = core.get_actor_observation(p_act)
        masks = core.get_action_masks(p_act)

        obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
        masks_t = {k: torch.tensor(masks[k], dtype=torch.int8, device=device).unsqueeze(0) for k in action_keys}

        with torch.no_grad():
            actions_dict, _, _ = actor(obs_t, masks_t, deterministic=args.deterministic)
        act = {k: int(actions_dict[k].item()) for k in action_keys}

        prev_len = len(core.event_log)
        core.step({f"player_{p_act}": act})

        for i in range(prev_len, len(core.event_log)):
            ev = core.event_log[i]
            if "TRADE PROPOSAL:" in ev:
                trade_props += 1
            elif "TRADE ACCEPTED" in ev:
                trade_accs += 1
                print(f"[R{core.round_count:<3} S{step_count:<4}] {ev}")
            elif "upgraded to a HOTEL" in ev:
                print(f"[R{core.round_count:<3} S{step_count:<4}] {ev}")
            elif "IS BANKRUPT" in ev:
                print(f"[R{core.round_count:<3} S{step_count:<4}] 💥 {ev}")

        curr_bankrupt_count = sum(1 for b in core.bankrupt if b)
        if curr_bankrupt_count > last_bankrupt_count:
            last_bankrupt_count = curr_bankrupt_count
            survivors = [core.get_player_name(pl) for pl in range(4) if not core.bankrupt[pl]]
            print(f"      --> Survivors remaining ({len(survivors)}): {', '.join(survivors)}")

    elapsed = time.time() - start_time
    survivors = [pl for pl in range(4) if not core.bankrupt[pl]]

    print("\n" + "=" * 65)
    if len(survivors) == 1:
        champion_id = survivors[0]
        champ_name = core.get_player_name(champion_id)
        champ_cash = core.cash[champion_id]
        champ_nw = core.get_player_net_worth(champion_id)
        print(f"🏆 TOURNAMENT CHAMPION: {champ_name.upper()} (Sole Survivor)")
        print(f"   Final Cash     : ${champ_cash:,}")
        print(f"   Total Net Worth: ${champ_nw:,}")
        print(f"   Total Rounds   : {core.round_count} (Steps: {step_count})")
        print(f"   Elapsed Time   : {elapsed:.2f} seconds ({step_count/elapsed:.1f} steps/s)")
        print(f"   Trades (Prop/Acc): {trade_props}/{trade_accs}")

        champ_monos = []
        for g, tids in COLOR_GROUPS.items():
            if all(core.property_owner[t] == champion_id for t in tids):
                champ_monos.append(g)
        print(f"   Monopolies Won : {len(champ_monos)}/10 {champ_monos}")
    else:
        print(f"Game concluded without sole survivor at step {step_count} (Round {core.round_count})")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    args = parse_args()
    run_tournament_match(args)
