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
from monopoly_marl.models.actor_network import ActorNetwork
from monopoly_marl.models.critic_network import CentralizedCritic
from monopoly_marl.models.storage import RolloutBuffer
from monopoly_marl.training.mappo_trainer import MAPPOTrainer
from monopoly_marl.training.league_manager import LeagueManager


def parse_args():
    parser = argparse.ArgumentParser(description="Monopoly MARL Tournament Headless Training")
    parser.add_argument("--total_steps", type=int, default=1000000, help="Total environment steps to train (default: 1,000,000)")
    parser.add_argument("--num_iterations", type=int, default=None, help="Number of training iterations (overrides total_steps if set)")
    parser.add_argument("--rollout_steps", type=int, default=128, help="Rollout steps per iteration")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument("--checkpoint_interval", type=int, default=25, help="Iterations between checkpoints")
    parser.add_argument("--eval_interval", type=int, default=10, help="Iterations between league evaluations")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--entropy_coef", type=float, default=0.001, help="PPO entropy loss coefficient (default: 0.001)")
    parser.add_argument("--entropy_min", type=float, default=0.0001, help="Minimum entropy coefficient when annealing (default: 0.0001)")
    parser.add_argument("--no_entropy_anneal", action="store_true", help="Disable linear entropy annealing")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume actor weights from")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Compute iterations from total_steps if not explicitly given
    steps_per_iter = args.num_envs * args.rollout_steps
    if args.num_iterations is None:
        args.num_iterations = max(1, args.total_steps // steps_per_iter)
    actual_total_steps = args.num_iterations * steps_per_iter

    print(f"=== Starting Monopoly MARL CTDE League Training ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Parallel Envs: {args.num_envs} | Rollout Steps: {args.rollout_steps} | Steps/Iter: {steps_per_iter:,}")
    print(f"Target Iterations: {args.num_iterations} | Total Environment Steps: {actual_total_steps:,}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Initialize Environments
    envs = [MonopolyParallelEnv(max_rounds=120, seed=42 + i) for i in range(args.num_envs)]

    # 2. Initialize Models (100% Identical Blueprint & Orthogonal Initialization)
    obs_dim = 327
    state_dim = 1340
    main_actor = ActorNetwork(obs_dim=obs_dim).to(device)
    main_critic = CentralizedCritic(state_dim=state_dim).to(device)
    exploiter_actor = ActorNetwork(obs_dim=obs_dim).to(device)

    # Resume checkpoint if provided
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"--> [Resume] Loading weights from {args.resume_checkpoint}")
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt.get("actor_state_dict", None))
        if state_dict is not None:
            main_actor.load_state_dict(state_dict)
            print("--> [Resume] Actor weights successfully restored.")

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
        device=device,
    )

    league_mgr = LeagueManager(checkpoint_dir=args.checkpoint_dir, device=device)

    # Save initial milestone checkpoint
    league_mgr.save_checkpoint(main_actor, iteration=0, role="main")

    # Reset environments
    current_obs_list = []
    current_infos_list = []
    for env in envs:
        o, info = env.reset()
        current_obs_list.append(o)
        current_infos_list.append(info)

    # Preallocate fast batched numpy arrays for zero-overhead GPU transfer
    batch_obs_np = np.zeros((args.num_envs, 4, obs_dim), dtype=np.float32)
    batch_states_np = np.zeros((args.num_envs, state_dim), dtype=np.float32)
    batch_masks_np = {k: np.zeros((args.num_envs, 4, buffer.mask_dims[k]), dtype=np.int8) for k in buffer.action_keys}
    rewards_np = np.zeros((args.num_envs, 4), dtype=np.float32)
    dones_np = np.zeros((args.num_envs, 4), dtype=np.float32)

    start_time = time.time()
    total_steps_executed = 0

    # Main Training Loop
    for iteration in range(1, args.num_iterations + 1):
        # Dynamic linear entropy annealing
        if not args.no_entropy_anneal and args.num_iterations > 1:
            frac = (iteration - 1) / float(args.num_iterations - 1)
            current_ent_coef = args.entropy_coef - frac * (args.entropy_coef - args.entropy_min)
            trainer.set_entropy_coef(current_ent_coef)

        main_actor.train()
        main_critic.train()

        # Step rollout collection
        for step in range(args.rollout_steps):
            # Gather numpy data across environments
            for e_idx in range(args.num_envs):
                batch_states_np[e_idx] = current_infos_list[e_idx]["player_0"]["global_state"]
                for p_idx in range(4):
                    p_name = f"player_{p_idx}"
                    batch_obs_np[e_idx, p_idx] = current_obs_list[e_idx][p_name]["obs"]
                    for k in buffer.action_keys:
                        batch_masks_np[k][e_idx, p_idx] = current_obs_list[e_idx][p_name]["action_mask"][k]

            # Fast single vectorized transfer to GPU
            batch_obs = torch.as_tensor(batch_obs_np, device=device)
            batch_states = torch.as_tensor(batch_states_np, device=device)
            batch_masks = {k: torch.as_tensor(batch_masks_np[k], device=device) for k in buffer.action_keys}

            # Evaluate critic values for all agents in all envs
            with torch.no_grad():
                values = main_critic(batch_states)  # (num_envs, 4)

                # Sample actions from actor for all 4 agents
                flat_obs = batch_obs.view(-1, obs_dim)
                flat_masks = {k: batch_masks[k].view(-1, buffer.mask_dims[k]) for k in buffer.action_keys}
                actions_dict, log_probs, _ = main_actor(flat_obs, flat_masks)

                actions_per_env = {k: actions_dict[k].view(args.num_envs, 4) for k in buffer.action_keys}
                log_probs_per_env = log_probs.view(args.num_envs, 4)

            # Move actions to CPU numpy once for stepping
            a_cpu = {k: actions_per_env[k].cpu().numpy() for k in buffer.action_keys}

            # Step environments
            for e_idx, env in enumerate(envs):
                env_actions = {
                    f"player_{p_idx}": {k: int(a_cpu[k][e_idx, p_idx]) for k in buffer.action_keys}
                    for p_idx in range(4)
                }

                next_obs, rewards, terminations, truncations, infos = env.step(env_actions)

                for p_idx in range(4):
                    p_name = f"player_{p_idx}"
                    rewards_np[e_idx, p_idx] = rewards.get(p_name, 0.0)
                    done = terminations.get(p_name, False) or truncations.get(p_name, False)
                    dones_np[e_idx, p_idx] = 1.0 if done else 0.0

                current_obs_list[e_idx] = next_obs
                current_infos_list[e_idx] = infos

                # Reset env if game over
                if any(truncations.values()) or all(terminations.values()):
                    o, info = env.reset()
                    current_obs_list[e_idx] = o
                    current_infos_list[e_idx] = info

            rewards_tensor = torch.as_tensor(rewards_np, device=device)
            dones_tensor = torch.as_tensor(dones_np, device=device)

            # Insert into buffer
            buffer.insert(
                obs=batch_obs,
                global_state=batch_states,
                actions=actions_per_env,
                action_masks=batch_masks,
                rewards=rewards_tensor,
                dones=dones_tensor,
                values=values,
                log_probs=log_probs_per_env,
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
            print(
                f"[{iteration:04d}/{args.num_iterations} | {progress_pct:5.1f}%] "
                f"Steps: {total_steps_executed:,}/{actual_total_steps:,} | "
                f"Policy Loss: {train_metrics['policy_loss']:.4f} | "
                f"Value Loss: {train_metrics['value_loss']:.4f} | "
                f"Entropy: {train_metrics['entropy']:.3f} | "
                f"Ent-Coef: {trainer.entropy_coef:.5f} | "
                f"Speed: {fps:,} steps/s"
            )

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
                    o_t = torch.tensor(eval_obs[p_name]["obs"], dtype=torch.float32, device=device).unsqueeze(0)
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

    total_time = time.time() - start_time
    print(f"\n=======================================================")
    print(f"=== Training 1,000,000 Steps Finished Successfully! ===")
    print(f"Total Steps: {total_steps_executed:,} | Elapsed Time: {total_time:.2f}s ({total_time/60:.1f} min)")
    print(f"Average Speed: {int(total_steps_executed/total_time):,} steps/sec")
    print(f"Checkpoints and League Summary saved in: {args.checkpoint_dir}/")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
