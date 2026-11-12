"""Headless Multi-Environment MAPPO League Training for Monopoly MARL.

Executes CTDE training with strict action masking, identical neural blueprint,
potential-based reward shaping, and league matchmaking (Main Agent vs Exploiter
vs Historical Checkpoints).
"""

import os
import argparse
import time
import json
import torch
import numpy as np

from monopoly_marl.engine.pettingzoo_wrapper import MonopolyParallelEnv
from monopoly_marl.engine.vec_env import SubprocMonopolyVecEnv
from monopoly_marl.models.actor_network import ActorNetwork
from monopoly_marl.models.critic_network import CentralizedCritic
from monopoly_marl.models.storage import RolloutBuffer
from monopoly_marl.training.mappo_trainer import MAPPOTrainer
from monopoly_marl.training.league_manager import LeagueManager


def parse_args():
    parser = argparse.ArgumentParser(description="Monopoly MARL Tournament Headless Training")
    parser.add_argument("--total_steps", type=int, default=1000000, help="Total environment steps to train (default: 1,000,000)")
    parser.add_argument("--num_iterations", type=int, default=None, help="Number of training iterations (overrides total_steps if set)")
    parser.add_argument("--rollout_steps", type=int, default=256, help="Rollout steps per iteration (default: 256 for extended GAE horizon)")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of multiprocessing env workers (>1 uses SubprocMonopolyVecEnv)")
    parser.add_argument("--num_threads", type=int, default=min(6, os.cpu_count() or 6), help="PyTorch CPU OpenMP threads (default: min(6, cpu_count))")
    parser.add_argument("--checkpoint_interval", type=int, default=25, help="Iterations between checkpoints")
    parser.add_argument("--eval_interval", type=int, default=10, help="Iterations between league evaluations")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="PPO entropy loss coefficient (default: 0.01)")
    parser.add_argument("--entropy_min", type=float, default=0.001, help="Minimum entropy coefficient when annealing (default: 0.001)")
    parser.add_argument("--no_entropy_anneal", action="store_true", help="Disable linear entropy annealing")
    parser.add_argument("--league_rollout_fraction", type=float, default=0.25, help="Fraction of environments reserved for heterogeneous league rollout (default: 0.25)")
    parser.add_argument("--curriculum", action="store_true", default=True, help="Enable 2-phase curriculum learning (default: True)")
    parser.add_argument("--no_curriculum", action="store_false", dest="curriculum", help="Disable curriculum learning")
    parser.add_argument("--curriculum_phase_2_frac", type=float, default=0.15, help="Fraction of total iterations to unlock Phase 2 full trading (default: 0.15)")
    parser.add_argument("--gamma", type=float, default=0.997, help="GAE discount factor (default: 0.997 for long-horizon Monopoly)")
    parser.add_argument("--gae_lambda", type=float, default=0.97, help="GAE lambda (default: 0.97)")
    parser.add_argument("--w1", type=float, default=0.050, help="PBRS cash log potential weight (default: 0.050)")
    parser.add_argument("--w2", type=float, default=0.005, help="PBRS property equity weight (default: 0.005)")
    parser.add_argument("--w3", type=float, default=0.008, help="PBRS monopoly completion bonus weight (default: 0.008)")
    parser.add_argument("--w4", type=float, default=0.0005, help="PBRS rent hazard paranoia weight (default: 0.0005)")
    parser.add_argument("--w_syn", type=float, default=0.020, help="PBRS color synergy potential weight (default: 0.020)")
    parser.add_argument("--beta_starve", type=float, default=3.5, help="PBRS round-based monopoly starvation penalty weight (default: 3.5)")
    parser.add_argument("--w_house", type=float, default=0.015, help="PBRS house/hotel building equity weight (default: 0.015)")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume actor weights from")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cpu" and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    # Compute iterations from total_steps if not explicitly given
    steps_per_iter = args.num_envs * args.rollout_steps
    if args.num_iterations is None:
        args.num_iterations = max(1, args.total_steps // steps_per_iter)
    actual_total_steps = args.num_iterations * steps_per_iter

    print(f"=== Starting Monopoly MARL CTDE League Training ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else f'CPU, Threads={torch.get_num_threads()}'})")
    print(f"Parallel Envs: {args.num_envs} (Workers: {args.num_workers}) | Rollout Steps: {args.rollout_steps} | Steps/Iter: {steps_per_iter:,}")
    print(f"Target Iterations: {args.num_iterations} | Total Environment Steps: {actual_total_steps:,}")
    print(f"PBRS Anti-Exploit Weights: w1={args.w1}, w2={args.w2}, w3={args.w3}, w4={args.w4}, w_house={args.w_house}, beta_starve={args.beta_starve}, gamma={args.gamma}")
    num_league_envs = max(0, int(args.num_envs * args.league_rollout_fraction))
    print(f"Heterogeneous League Rollout: {num_league_envs}/{args.num_envs} envs ({args.league_rollout_fraction * 100:.0f}%)")
    print(f"Curriculum 2-Phase Learning: {'Enabled (Phase 2 at ' + str(int(args.curriculum_phase_2_frac * 100)) + '%)' if args.curriculum else 'Disabled'}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Initialize Environments (Single-process or Multiprocessing)
    env_configs = [
        dict(
            max_rounds=120,
            seed=42 + i,
            gamma=args.gamma,
            w1=args.w1,
            w2=args.w2,
            w3=args.w3,
            w4=args.w4,
            w_syn=args.w_syn,
            beta_starve=args.beta_starve,
            w_house=args.w_house,
            allow_trade=not args.curriculum,
        )
        for i in range(args.num_envs)
    ]

    use_vec = args.num_workers > 1
    if use_vec:
        vec_env = SubprocMonopolyVecEnv(env_configs, num_workers=args.num_workers)
        obs_dim = vec_env.obs_dim
        state_dim = vec_env.state_dim
        envs = None
    else:
        envs = [MonopolyParallelEnv(**cfg) for cfg in env_configs]
        obs_dim = envs[0].obs_dim
        state_dim = envs[0].state_dim
        vec_env = None
    print(f"Observation Dimension: {obs_dim} (Synergy Features: {'Enabled' if obs_dim > 327 else 'Disabled'}) | Global State: {state_dim}")
    main_actor = ActorNetwork(obs_dim=obs_dim).to(device)
    main_critic = CentralizedCritic(state_dim=state_dim).to(device)
    exploiter_actor = ActorNetwork(obs_dim=obs_dim).to(device)

    # Resume checkpoint if provided (with seamless weight transfer if obs_dim or head dims changed)
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"--> [Resume] Loading weights from {args.resume_checkpoint}")
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt.get("actor_state_dict", None))
        if state_dict is not None:
            new_sd = main_actor.state_dict()
            old_in_dim = state_dict["trunk.0.weight"].shape[1]
            old_auc_dim = state_dict["head_auction.weight"].shape[0] if "head_auction.weight" in state_dict else 5
            curr_auc_dim = new_sd["head_auction.weight"].shape[0]
            old_prop_dim = state_dict["head_property.weight"].shape[0] if "head_property.weight" in state_dict else 160
            curr_prop_dim = new_sd["head_property.weight"].shape[0]

            for k in state_dict:
                if k == "trunk.0.weight" and old_in_dim != obs_dim:
                    new_sd[k][:, :old_in_dim] = state_dict[k]
                    new_sd[k][:, old_in_dim:] = 0.0
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
            main_actor.load_state_dict(new_sd)
            print(f"--> [Resume] Actor weights successfully adapted and restored (in: {old_in_dim}->{obs_dim}, auc: {old_auc_dim}->{curr_auc_dim}, prop: {old_prop_dim}->{curr_prop_dim}).")

    # 3. Initialize Trainer & Storage
    trainer = MAPPOTrainer(
        actor=main_actor,
        critic=main_critic,
        lr_actor=3e-4,
        lr_critic=5e-4,
        clip_param=0.2,
        ppo_epochs=4,
        num_mini_batches=4,
        entropy_coef=args.entropy_coef,
        device=device,
    )

    buffer = RolloutBuffer(
        buffer_size=args.rollout_steps,
        num_envs=args.num_envs,
        obs_dim=obs_dim,
        state_dim=state_dim,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        device=device,
    )

    league_mgr = LeagueManager(checkpoint_dir=args.checkpoint_dir, device=device)

    # Save initial milestone checkpoint
    league_mgr.save_checkpoint(main_actor, iteration=0, role="main")

    # Reset environments
    if use_vec:
        current_obs_list, current_infos_list = vec_env.reset()
    else:
        current_obs_list = []
        current_infos_list = []
        for env in envs:
            o, info = env.reset()
            current_obs_list.append(o)
            current_infos_list.append(info)

    # Preallocate fast batched numpy arrays for zero-overhead transfer
    batch_obs_np = np.zeros((args.num_envs, 4, obs_dim), dtype=np.float32)
    batch_states_np = np.zeros((args.num_envs, state_dim), dtype=np.float32)
    batch_masks_np = {k: np.zeros((args.num_envs, 4, buffer.mask_dims[k]), dtype=np.int8) for k in buffer.action_keys}
    active_players_np = np.zeros(args.num_envs, dtype=np.int64)
    rewards_np = np.zeros((args.num_envs, 4), dtype=np.float32)
    dones_np = np.zeros((args.num_envs, 4), dtype=np.float32)
    env_indices_t = torch.arange(args.num_envs, device=device)

    start_time = time.time()
    total_steps_executed = 0
    telemetry_window = {
        "episodes": 0,
        "trades_proposed": 0,
        "trades_accepted": 0,
        "houses_built": 0,
        "bank_houses_end": [],
    }

    try:
        # Main Training Loop
        for iteration in range(1, args.num_iterations + 1):
            # Dynamic linear entropy annealing
            if not args.no_entropy_anneal and args.num_iterations > 1:
                frac = (iteration - 1) / float(args.num_iterations - 1)
                current_ent_coef = args.entropy_coef - frac * (args.entropy_coef - args.entropy_min)
                trainer.set_entropy_coef(current_ent_coef)

            # Curriculum Learning: Phase 1 (Foundation without trade) vs Phase 2 (Full trade unlocked)
            is_curriculum_phase_1 = args.curriculum and (iteration <= int(args.curriculum_phase_2_frac * args.num_iterations))
            allow_trade = not is_curriculum_phase_1
            if use_vec:
                vec_env.set_allow_trade(allow_trade)
            else:
                for env in envs:
                    env.set_allow_trade(allow_trade)

            main_actor.train()
            main_critic.train()

            # Step rollout collection
            for step in range(args.rollout_steps):
                # Gather numpy data across environments
                for e_idx in range(args.num_envs):
                    info_0 = current_infos_list[e_idx]["player_0"]
                    batch_states_np[e_idx] = info_0["global_state"]
                    active_players_np[e_idx] = info_0["active_player"]
                    obs_e = current_obs_list[e_idx]
                    for p_idx in range(4):
                        p_name = f"player_{p_idx}"
                        batch_obs_np[e_idx, p_idx] = obs_e[p_name]["obs"]
                        for k in buffer.action_keys:
                            batch_masks_np[k][e_idx, p_idx] = obs_e[p_name]["action_mask"][k]

                # Fast vectorized transfer
                batch_obs = torch.as_tensor(batch_obs_np, device=device)
                batch_states = torch.as_tensor(batch_states_np, device=device)
                batch_masks = {k: torch.as_tensor(batch_masks_np[k], device=device) for k in buffer.action_keys}
                act_p_t = torch.as_tensor(active_players_np, device=device)

                # Evaluate critic & sample ONLY for the 1 active decision player per env (4x faster Actor inference!)
                with torch.no_grad():
                    values = main_critic(batch_states)  # (num_envs, 4)

                    active_obs = batch_obs[env_indices_t, act_p_t]  # (num_envs, obs_dim)
                    active_masks_in = {k: batch_masks[k][env_indices_t, act_p_t] for k in buffer.action_keys}
                    act_dict_active, lp_active, _ = main_actor(active_obs, active_masks_in)

                    actions_per_env = {
                        k: torch.zeros((args.num_envs, 4), dtype=torch.long, device=device)
                        for k in buffer.action_keys
                    }
                    # Default fallback for passive property head is 160 (Pass), offer/req prop is 40 (None)
                    actions_per_env["property"].fill_(160)
                    actions_per_env["trade_offer_prop"].fill_(40)
                    actions_per_env["trade_req_prop"].fill_(40)
                    for k in buffer.action_keys:
                        actions_per_env[k][env_indices_t, act_p_t] = act_dict_active[k]

                    log_probs_per_env = torch.zeros((args.num_envs, 4), dtype=torch.float32, device=device)
                    log_probs_per_env[env_indices_t, act_p_t] = lp_active

                    # Active mask = 1.0 strictly for active_decision_player controlled by main_actor
                    active_masks_step = torch.zeros((args.num_envs, 4), dtype=torch.float32, device=device)
                    active_masks_step[env_indices_t, act_p_t] = 1.0

                    # Heterogeneous League Rollout: players 2 and 3 in league envs use historical policy
                    if num_league_envs > 0 and league_mgr.historical_records:
                        hist_policy = league_mgr.get_random_historical_policy()
                        if hist_policy is not None:
                            hist_env_mask = (torch.arange(args.num_envs, device=device) < num_league_envs) & (act_p_t >= 2)
                            hist_idx = hist_env_mask.nonzero(as_tuple=True)[0]
                            if hist_idx.numel() > 0:
                                hist_p = act_p_t[hist_idx]
                                opp_obs = batch_obs[hist_idx, hist_p]
                                if hasattr(hist_policy, "obs_dim") and hist_policy.obs_dim < obs_dim:
                                    opp_obs = opp_obs[:, :hist_policy.obs_dim]
                                opp_masks = {k: batch_masks[k][hist_idx, hist_p] for k in buffer.action_keys}
                                opp_acts, opp_lps, _ = hist_policy(opp_obs, opp_masks)
                                for k in buffer.action_keys:
                                    actions_per_env[k][hist_idx, hist_p] = opp_acts[k]
                                log_probs_per_env[hist_idx, hist_p] = opp_lps
                                active_masks_step[hist_idx, hist_p] = 0.0

                # Move actions to CPU numpy once for stepping
                a_cpu = {k: actions_per_env[k].cpu().numpy() for k in buffer.action_keys}

                # Build action dicts for all environments
                all_env_actions = [
                    {
                        f"player_{p_idx}": {k: int(a_cpu[k][e_idx, p_idx]) for k in buffer.action_keys}
                        for p_idx in range(4)
                    }
                    for e_idx in range(args.num_envs)
                ]

                if use_vec:
                    step_results = vec_env.step(all_env_actions)
                    for e_idx, (next_obs, rewards, terminations, truncations, infos, ep_stats) in enumerate(step_results):
                        for p_idx in range(4):
                            p_name = f"player_{p_idx}"
                            rewards_np[e_idx, p_idx] = rewards.get(p_name, 0.0)
                            done = terminations.get(p_name, False) or truncations.get(p_name, False)
                            dones_np[e_idx, p_idx] = 1.0 if done else 0.0
                        current_obs_list[e_idx] = next_obs
                        current_infos_list[e_idx] = infos
                        if ep_stats is not None:
                            telemetry_window["episodes"] += 1
                            telemetry_window["trades_proposed"] += ep_stats["trades_proposed"]
                            telemetry_window["trades_accepted"] += ep_stats["trades_accepted"]
                            telemetry_window["houses_built"] += ep_stats["houses_built"]
                            telemetry_window["bank_houses_end"].append(ep_stats["bank_houses"])
                else:
                    for e_idx, env in enumerate(envs):
                        next_obs, rewards, terminations, truncations, infos = env.step(all_env_actions[e_idx])
                        for p_idx in range(4):
                            p_name = f"player_{p_idx}"
                            rewards_np[e_idx, p_idx] = rewards.get(p_name, 0.0)
                            done = terminations.get(p_name, False) or truncations.get(p_name, False)
                            dones_np[e_idx, p_idx] = 1.0 if done else 0.0
                        current_obs_list[e_idx] = next_obs
                        current_infos_list[e_idx] = infos
                        if any(truncations.values()) or all(terminations.values()):
                            telemetry_window["episodes"] += 1
                            telemetry_window["trades_proposed"] += env.core.trades_proposed_count
                            telemetry_window["trades_accepted"] += env.core.trades_accepted_count
                            telemetry_window["houses_built"] += env.core.houses_built_count
                            telemetry_window["bank_houses_end"].append(env.core.bank_houses)
                            o, info = env.reset()
                            current_obs_list[e_idx] = o
                            current_infos_list[e_idx] = info

                rewards_tensor = torch.as_tensor(rewards_np, device=device)
                dones_tensor = torch.as_tensor(dones_np, device=device)

                # Insert into buffer with active_masks for turn-level & heterogeneous CTDE filtering
                buffer.insert(
                    obs=batch_obs,
                    global_state=batch_states,
                    actions=actions_per_env,
                    action_masks=batch_masks,
                    rewards=rewards_tensor,
                    dones=dones_tensor,
                    values=values,
                    log_probs=log_probs_per_env,
                    active_masks=active_masks_step,
                )

            total_steps_executed += steps_per_iter

            # Compute GAE
            with torch.no_grad():
                last_states_np = np.zeros((args.num_envs, state_dim), dtype=np.float32)
                for e_idx in range(args.num_envs):
                    last_states_np[e_idx] = current_infos_list[e_idx]["player_0"]["global_state"]
                last_states = torch.as_tensor(last_states_np, device=device)
                last_values = main_critic(last_states)
                last_dones = dones_tensor

            buffer.compute_gae(last_values, last_dones)

            # Update MAPPO policy & centralized value function
            train_metrics = trainer.train_step(buffer)

            # Log progress
            elapsed = time.time() - start_time
            fps = int(total_steps_executed / elapsed)
            progress_pct = (iteration / args.num_iterations) * 100.0

            if iteration % 5 == 0 or iteration == 1 or iteration == args.num_iterations:
                curriculum_tag = "P1: Foundation" if is_curriculum_phase_1 else "P2: Full Trade"
                ep_cnt = telemetry_window["episodes"]
                if ep_cnt > 0:
                    tr_prop = telemetry_window["trades_proposed"]
                    tr_acc = telemetry_window["trades_accepted"]
                    acc_rate = (tr_acc / tr_prop * 100.0) if tr_prop > 0 else 0.0
                    h_built = telemetry_window["houses_built"] / ep_cnt
                    avg_bh = sum(telemetry_window["bank_houses_end"]) / len(telemetry_window["bank_houses_end"])
                    telemetry_str = f" | Trades(Acc/Prop): {tr_acc}/{tr_prop} ({acc_rate:.1f}%) | Houses/Ep: {h_built:.1f} | EndBankH: {avg_bh:.1f}"
                else:
                    live_stats = vec_env.get_live_stats() if use_vec else [
                        {
                            "trades_proposed": e.core.trades_proposed_count,
                            "trades_accepted": e.core.trades_accepted_count,
                            "houses_built": e.core.houses_built_count,
                            "bank_houses": e.core.bank_houses,
                        }
                        for e in envs
                    ]
                    tr_prop = sum(s["trades_proposed"] for s in live_stats)
                    tr_acc = sum(s["trades_accepted"] for s in live_stats)
                    h_built = sum(s["houses_built"] for s in live_stats) / max(1, len(live_stats))
                    avg_bh = sum(s["bank_houses"] for s in live_stats) / max(1, len(live_stats))
                    telemetry_str = f" | Live Trades: {tr_acc}/{tr_prop} | Houses/Env: {h_built:.1f} | BankH: {avg_bh:.1f}"

                print(
                    f"[{iteration:04d}/{args.num_iterations} | {progress_pct:5.1f}% | {curriculum_tag}] "
                    f"Steps: {total_steps_executed:,}/{actual_total_steps:,} | "
                    f"PLoss: {train_metrics['policy_loss']:.4f} | "
                    f"VLoss: {train_metrics['value_loss']:.4f} | "
                    f"Ent: {train_metrics['entropy']:.3f} ({trainer.entropy_coef:.5f}) | "
                    f"FPS: {fps:,}{telemetry_str}"
                )
                telemetry_window = {
                    "episodes": 0, "trades_proposed": 0, "trades_accepted": 0,
                    "houses_built": 0, "bank_houses_end": [],
                }

            # Checkpointing & League Updates
            if iteration % args.checkpoint_interval == 0 or iteration == args.num_iterations:
                ckpt_path = league_mgr.save_checkpoint(
                    main_actor,
                    iteration=iteration,
                    role="main",
                    metadata={"train_metrics": train_metrics, "iteration": iteration, "total_steps": total_steps_executed},
                )
                print(f"--> [Checkpoint Saved] {ckpt_path} (Total Steps: {total_steps_executed:,})")

            # League Evaluation Matches
            if iteration % args.eval_interval == 0 and league_mgr.historical_records:
                eval_env = MonopolyParallelEnv(max_rounds=100, seed=1000 + iteration)
                eval_obs, eval_infos = eval_env.reset()
                lineup_policies, lineup_records = league_mgr.sample_match_lineup(main_actor, exploiter_actor, mode="league")

                done_eval = False
                step_count = 0
                while not done_eval and step_count < 150:
                    step_count += 1
                    actions = {}
                    for p_idx in range(4):
                        p_name = f"player_{p_idx}"
                        pol = lineup_policies[p_idx]
                        o_raw = eval_obs[p_name]["obs"]
                        if hasattr(pol, "obs_dim") and pol.obs_dim < len(o_raw):
                            o_raw = o_raw[:pol.obs_dim]
                        o_t = torch.tensor(o_raw, dtype=torch.float32, device=device).unsqueeze(0)
                        m_t = {
                            k: torch.tensor(eval_obs[p_name]["action_mask"][k], dtype=torch.int8, device=device).unsqueeze(0)
                            for k in buffer.action_keys
                        }
                        with torch.no_grad():
                            a_dict, _, _ = pol(o_t, m_t, deterministic=True)
                        actions[p_name] = {k: int(a_dict[k].item()) for k in buffer.action_keys}

                    eval_obs, _, terminations, truncations, eval_infos = eval_env.step(actions)
                    done_eval = any(truncations.values()) or all(terminations.values())

                # Compute ranks and update Elo
                net_worths = [eval_env.core.get_player_net_worth(p) for p in range(4)]
                order = np.argsort(net_worths)[::-1]
                ranks = [0] * 4
                for rank_pos, p_idx in enumerate(order):
                    ranks[p_idx] = rank_pos + 1

                league_mgr.update_match_results(lineup_records, ranks, net_worths)
                league_mgr.save_league_summary()
                print(
                    f"--> [League Match] Ranks: {ranks} | "
                    f"Main Elo: {league_mgr.main_agent_record.elo:.1f} | "
                    f"Win Rate: {league_mgr.main_agent_record.win_rate * 100:.1f}%"
                )
    finally:
        if use_vec and vec_env is not None:
            vec_env.close()

    total_time = time.time() - start_time
    print(f"\n=======================================================")
    print(f"=== Training {total_steps_executed:,} Steps Finished Successfully! ===")
    print(f"Total Steps: {total_steps_executed:,} | Elapsed Time: {total_time:.2f}s ({total_time/60:.1f} min)")
    print(f"Average Speed: {int(total_steps_executed/total_time):,} steps/sec")
    print(f"Checkpoints and League Summary saved in: {args.checkpoint_dir}/")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
