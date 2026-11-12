"""Evaluation and Intelligence Benchmark for 10M Steps Monopoly MARL Agents."""
import os
import torch
import numpy as np
from monopoly_marl.engine.monopoly_core import (
    MonopolyCore, PHASE_AUCTION, PHASE_BUY_OR_AUCTION,
    ACT_AUC_JUMP_LIST, ACT_AUC_JUMP_SQUEEZE,
    ACT_CORE_BUY_DIRECT, ACT_CORE_DECLINE_BUY
)
from monopoly_marl.engine.board_data import COLOR_GROUPS
from monopoly_marl.models.actor_network import ActorNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"=== Benchmarking Monopoly MARL Intelligence (10M Steps) ===")
print(f"Device: {device}")

def load_policy(path):
    ckpt = torch.load(path, map_location=device)
    sd = ckpt.get("state_dict", ckpt)
    actor = ActorNetwork(obs_dim=500).to(device)
    actor.load_state_dict(sd)
    actor.eval()
    return actor

trained_policy = load_policy("checkpoints/main_iter_02442.pt")
untrained_policy = load_policy("checkpoints/main_iter_00000.pt")

def evaluate_lineup(policies, num_games=5):
    stats = {
        "game_lengths": [],
        "bankruptcies": [],
        "monopolies_per_game": [],
        "houses_built_per_game": [],
        "auctions_per_game": [],
        "jump_bids_per_game": [],
        "buys_direct_per_game": [],
        "declines_per_game": [],
        "wins": [0, 0, 0, 0],
        "final_net_worth": [[], [], [], []]
    }

    for g in range(num_games):
        core = MonopolyCore(seed=1000 + g, max_rounds=150)
        buys = 0
        declines = 0
        auctions = 0
        jumps = 0

        while core.phase != 6 and core.round_count < 150:
            p = core.active_decision_player
            obs = torch.tensor(core.get_actor_observation(p), dtype=torch.float32, device=device).unsqueeze(0)
            masks = {k: torch.tensor(v, dtype=torch.int8, device=device).unsqueeze(0) for k, v in core.get_action_masks(p).items()}

            with torch.no_grad():
                act_dict, _, _ = policies[p](obs, masks)

            act_dict_int = {k: int(v.squeeze(0).cpu().item()) for k, v in act_dict.items()}

            if core.phase == PHASE_BUY_OR_AUCTION:
                if act_dict_int["core"] == ACT_CORE_BUY_DIRECT:
                    buys += 1
                elif act_dict_int["core"] == ACT_CORE_DECLINE_BUY:
                    declines += 1
            elif core.phase == PHASE_AUCTION:
                auctions += 1
                if act_dict_int["auction"] in (ACT_AUC_JUMP_LIST, ACT_AUC_JUMP_SQUEEZE):
                    jumps += 1

            core.step({f"player_{p}": act_dict_int})

        stats["game_lengths"].append(core.round_count)
        bankrupt_cnt = sum(1 for b in core.bankrupt if b)
        stats["bankruptcies"].append(bankrupt_cnt)

        # Count monopolies
        monopolies = 0
        for pl in range(4):
            for grp, tiles in COLOR_GROUPS.items():
                if all(core.property_owner[t] == pl for t in tiles):
                    monopolies += 1
        stats["monopolies_per_game"].append(monopolies)

        h_on_board = sum(core.property_houses)
        stats["houses_built_per_game"].append(h_on_board)
        stats["auctions_per_game"].append(auctions)
        stats["jump_bids_per_game"].append(jumps)
        stats["buys_direct_per_game"].append(buys)
        stats["declines_per_game"].append(declines)

        nws = [core.get_player_net_worth(pl) if not core.bankrupt[pl] else -999999 for pl in range(4)]
        winner = int(np.argmax(nws))
        stats["wins"][winner] += 1
        for pl in range(4):
            stats["final_net_worth"][pl].append(core.get_player_net_worth(pl))

    return stats

print("\n[1/3] Benchmarking 4 Trained Agents (10M Steps)...")
trained_stats = evaluate_lineup([trained_policy] * 4, num_games=5)
print(f"  Avg Game Length: {np.mean(trained_stats['game_lengths']):.1f} rounds")
print(f"  Avg Bankruptcies / Game: {np.mean(trained_stats['bankruptcies']):.2f}")
print(f"  Avg Monopolies Formed: {np.mean(trained_stats['monopolies_per_game']):.2f}")
print(f"  Avg Houses Built: {np.mean(trained_stats['houses_built_per_game']):.2f}")
print(f"  Buys Direct vs Declines: {np.mean(trained_stats['buys_direct_per_game']):.1f} buys / {np.mean(trained_stats['declines_per_game']):.1f} declines")
print(f"  Auctions & Jump Bids: {np.mean(trained_stats['auctions_per_game']):.1f} total bids, {np.mean(trained_stats['jump_bids_per_game']):.1f} jump bids")
print(f"  Average Net Worth: ${int(np.mean(trained_stats['final_net_worth']))}")

print("\n[2/3] Benchmarking 4 Untrained Random Agents (0M Steps)...")
untrained_stats = evaluate_lineup([untrained_policy] * 4, num_games=5)
print(f"  Avg Game Length: {np.mean(untrained_stats['game_lengths']):.1f} rounds")
print(f"  Avg Bankruptcies / Game: {np.mean(untrained_stats['bankruptcies']):.2f}")
print(f"  Avg Monopolies Formed: {np.mean(untrained_stats['monopolies_per_game']):.2f}")
print(f"  Avg Houses Built: {np.mean(untrained_stats['houses_built_per_game']):.2f}")
print(f"  Buys Direct vs Declines: {np.mean(untrained_stats['buys_direct_per_game']):.1f} buys / {np.mean(untrained_stats['declines_per_game']):.1f} declines")
print(f"  Auctions & Jump Bids: {np.mean(untrained_stats['auctions_per_game']):.1f} total bids, {np.mean(untrained_stats['jump_bids_per_game']):.1f} jump bids")
print(f"  Average Net Worth: ${int(np.mean(untrained_stats['final_net_worth']))}")

print("\n[3/3] Head-to-Head Tournament (10 Games): Trained (P0, P2) vs Untrained (P1, P3)...")
h2h_policies = [trained_policy, untrained_policy, trained_policy, untrained_policy]
h2h_stats = evaluate_lineup(h2h_policies, num_games=10)
t_wins = h2h_stats["wins"][0] + h2h_stats["wins"][2]
u_wins = h2h_stats["wins"][1] + h2h_stats["wins"][3]
print(f"  Trained Agents (10M): {t_wins}/10 wins ({t_wins/10*100:.1f}%)")
print(f"  Untrained Agents (0M): {u_wins}/10 wins ({u_wins/10*100:.1f}%)")

t_nw = np.mean(h2h_stats["final_net_worth"][0] + h2h_stats["final_net_worth"][2])
u_nw = np.mean(h2h_stats["final_net_worth"][1] + h2h_stats["final_net_worth"][3])
print(f"  Average End-Game Net Worth: Trained ${int(t_nw)} vs Untrained ${int(u_nw)}")
