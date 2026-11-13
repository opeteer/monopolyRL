"""High-Performance Multi-Environment MAPPO League Training for Monopoly MARL.

Executes CTDE training with strict action masking, identical neural blueprint,
potential-based reward shaping, and league matchmaking (Main Agent vs Exploiter
vs Historical Checkpoints).

Features:
- Zero-copy Shared Memory vectorized parallel rollout across physical CPU cores
- Hardware auto-detection and optimal thread calibration
- Configurable training presets (smoke_test, fast_dev, standard_1m, tournament_5m, overnight_10m)
- Isolated session management, JSONL telemetry, and graceful Ctrl+C interruption
"""

import os
import sys
import math
import argparse
import time
import json
import glob
import torch
import numpy as np
from typing import Optional, Dict, Any

from monopoly_marl.engine.pettingzoo_wrapper import MonopolyParallelEnv
from monopoly_marl.engine.vec_env import SubprocMonopolyVecEnv, SharedMemoryMonopolyVecEnv
from monopoly_marl.models.actor_network import ActorNetwork
from monopoly_marl.models.critic_network import CentralizedCritic
from monopoly_marl.models.storage import RolloutBuffer
from monopoly_marl.training.mappo_trainer import MAPPOTrainer
from monopoly_marl.training.league_manager import LeagueManager
from monopoly_marl.system.hardware import detect_hardware, print_hardware_summary
from monopoly_marl.system.config_presets import PRESETS, get_preset, list_presets, TrainingConfig
from monopoly_marl.system.session_manager import SessionManager


def parse_args():
    # Pre-parse to check preset or auto_hardware
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--preset", type=str, default=None, choices=list(PRESETS.keys()), help="Training preset name")
    pre_parser.add_argument("--auto_hardware", action="store_true", help="Auto-configure workers, envs, and threads for this PC")
    pre_args, remaining_argv = pre_parser.parse_known_args()

    # Determine base defaults from preset or hardware
    hw = detect_hardware()
    defaults = {}

    if pre_args.auto_hardware:
        defaults["num_workers"] = hw.recommended_workers
        defaults["num_envs"] = hw.recommended_envs
        defaults["num_threads"] = hw.recommended_torch_threads
        defaults["rollout_steps"] = hw.recommended_rollout_steps

    if pre_args.preset:
        preset_cfg = get_preset(pre_args.preset)
        if preset_cfg:
            defaults.update(preset_cfg.to_dict())

    parser = argparse.ArgumentParser(description="Monopoly MARL Tournament Headless Training")
    parser.add_argument("--preset", type=str, default=pre_args.preset, choices=list(PRESETS.keys()), help="Training preset name")
    parser.add_argument("--auto_hardware", action="store_true", default=pre_args.auto_hardware, help="Auto-configure workers and threads based on system resources")
    parser.add_argument("--session_name", type=str, default=None, help="Name of training session (creates sessions/<session_name>)")
    parser.add_argument("--base_session_dir", type=str, default="sessions", help="Base directory for sessions")
    parser.add_argument("--total_steps", type=int, default=defaults.get("total_steps", 1000000), help="Total environment steps to train")
    parser.add_argument("--num_iterations", type=int, default=None, help="Number of training iterations (overrides total_steps if set)")
    parser.add_argument("--rollout_steps", type=int, default=defaults.get("rollout_steps", 128), help="Rollout steps per iteration")
    parser.add_argument("--num_envs", type=int, default=defaults.get("num_envs", hw.recommended_envs), help="Number of parallel environments")
    parser.add_argument("--num_workers", type=int, default=defaults.get("num_workers", hw.recommended_workers), help="Number of multiprocessing env workers (>1 uses vectorized multiprocessing)")
    parser.add_argument("--num_threads", type=int, default=defaults.get("num_threads", hw.recommended_torch_threads), help="PyTorch CPU OpenMP threads")
    parser.add_argument("--use_shm", action="store_true", default=True, help="Enable zero-copy shared memory vectorization (default: True)")
    parser.add_argument("--no_shm", action="store_false", dest="use_shm", help="Disable shared memory vectorization (use standard Pipe Subproc)")
    parser.add_argument("--checkpoint_interval", type=int, default=defaults.get("checkpoint_interval", 25), help="Iterations between checkpoints")
    parser.add_argument("--eval_interval", type=int, default=defaults.get("eval_interval", 10), help="Iterations between league evaluations")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Custom checkpoint directory (overrides session dir if set)")
    parser.add_argument("--entropy_coef", type=float, default=defaults.get("entropy_coef", 0.004), help="PPO entropy loss coefficient")
    parser.add_argument("--entropy_min", type=float, default=defaults.get("entropy_min", 0.0002), help="Minimum entropy coefficient when annealing")
    parser.add_argument("--no_entropy_anneal", action="store_true", default=defaults.get("no_entropy_anneal", False), help="Disable cosine entropy annealing")
    parser.add_argument("--league_rollout_fraction", type=float, default=defaults.get("league_rollout_fraction", 0.25), help="Fraction of environments reserved for heterogeneous league rollout")
    parser.add_argument("--curriculum", action="store_true", default=defaults.get("curriculum", True), help="Enable 2-phase curriculum learning")
    parser.add_argument("--no_curriculum", action="store_false", dest="curriculum", help="Disable curriculum learning")
    parser.add_argument("--curriculum_phase_2_frac", type=float, default=defaults.get("curriculum_phase_2_frac", 0.15), help="Fraction of total iterations to unlock Phase 2 full trading")
    parser.add_argument("--gamma", type=float, default=defaults.get("gamma", 0.997), help="GAE discount factor")
    parser.add_argument("--gae_lambda", type=float, default=defaults.get("gae_lambda", 0.97), help="GAE lambda")
    parser.add_argument("--lr_actor", type=float, default=defaults.get("lr_actor", 2.5e-4), help="Actor learning rate")
    parser.add_argument("--lr_critic", type=float, default=defaults.get("lr_critic", 5e-4), help="Critic learning rate")
    parser.add_argument("--ppo_epochs", type=int, default=defaults.get("ppo_epochs", 4), help="PPO optimization epochs per iteration")
    parser.add_argument("--num_mini_batches", type=int, default=defaults.get("num_mini_batches", 4), help="Number of mini-batches per epoch")
    parser.add_argument("--clip_param", type=float, default=defaults.get("clip_param", 0.15), help="PPO clip parameter (default: 0.15)")
    parser.add_argument("--w1", type=float, default=defaults.get("w1", 0.050), help="PBRS cash log potential weight")
    parser.add_argument("--w2", type=float, default=defaults.get("w2", 0.0060), help="PBRS property equity weight")
    parser.add_argument("--w3", type=float, default=defaults.get("w3", 0.008), help="PBRS monopoly completion bonus weight")
    parser.add_argument("--w4", type=float, default=defaults.get("w4", 0.0005), help="PBRS rent hazard paranoia weight")
    parser.add_argument("--w_syn", type=float, default=defaults.get("w_syn", 0.020), help="PBRS color synergy potential weight")
    parser.add_argument("--beta_starve", type=float, default=defaults.get("beta_starve", 3.5), help="PBRS round-based monopoly starvation penalty weight")
    parser.add_argument("--w_house", type=float, default=defaults.get("w_house", 0.012), help="PBRS house/hotel building equity weight")
    parser.add_argument("--w_cash", type=float, default=defaults.get("w_cash", 0.0035), help="PBRS linear cash equity weight")
    parser.add_argument("--resume", type=str, default=None, help="Resume training: 'latest' or path to checkpoint file")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Legacy alias for --resume")
    parser.add_argument("--fixed_first_player", action="store_false", dest="randomize_first_player", default=defaults.get("randomize_first_player", True), help="Keep Player 0 as fixed starting player instead of randomizing first seat")
    parser.add_argument("--device", type=str, default=defaults.get("device", "cuda" if torch.cuda.is_available() else "cpu"), help="Device")

    return parser.parse_args(remaining_argv)


def find_latest_global_checkpoint() -> Optional[str]:
    """Scans sessions/ and checkpoints/ for the latest checkpoint."""
    patterns = [
        "sessions/*/checkpoints/main_iter_*.pt",
        "checkpoints/main_iter_*.pt",
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    if not candidates:
        return None

    def extract_iteration(p: str) -> int:
        try:
            base = os.path.basename(p)
            return int(base.split("_iter_")[1].split(".pt")[0])
        except Exception:
            return 0

    candidates.sort(key=lambda x: (extract_iteration(x), os.path.getmtime(x)))
    return candidates[-1]


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cpu" and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    # Initialize Session Manager
    session_mgr = SessionManager(
        session_name=args.session_name,
        base_dir=args.base_session_dir,
        config_dict=vars(args),
    )
    checkpoint_dir = args.checkpoint_dir or session_mgr.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Iteration and steps computation
    steps_per_iter = args.num_envs * args.rollout_steps
    if args.num_iterations is None:
        args.num_iterations = max(1, args.total_steps // steps_per_iter)
    actual_total_steps = args.num_iterations * steps_per_iter

    # Diagnostics printout
    hw = detect_hardware()
    print_hardware_summary(hw)
    print(f"\n=== Starting Monopoly MARL CTDE League Training Session ===")
    print(f"Session Name: {session_mgr.session_name}")
    print(f"Session Directory: {session_mgr.session_dir}/")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else f'CPU, Threads={torch.get_num_threads()}'})")
    print(f"Parallel Envs: {args.num_envs} (Workers: {args.num_workers}, Mode: {'SharedMemory (Zero-Copy)' if (args.num_workers > 1 and args.use_shm) else 'Subprocess/Single'})")
    print(f"Rollout Steps: {args.rollout_steps} | Steps/Iteration: {steps_per_iter:,}")
    print(f"Target Iterations: {args.num_iterations} | Total Environment Steps: {actual_total_steps:,}")
    num_league_envs = max(0, int(args.num_envs * args.league_rollout_fraction))
    print(f"Heterogeneous League Rollout: {num_league_envs}/{args.num_envs} envs ({args.league_rollout_fraction * 100:.0f}%)")
    print(f"Curriculum 2-Phase Learning: {'Enabled (Phase 2 unlock at ' + str(int(args.curriculum_phase_2_frac * 100)) + '%)' if args.curriculum else 'Disabled'}")
    print(f"First Mover Seat: {'Randomized [0..3]' if args.randomize_first_player else 'Fixed (Player 0)'}")

    # Environment Configurations (Seed driven by hardware fluctuation entropy)
    env_configs = [
        dict(
            max_rounds=120,
            seed=None,
            gamma=args.gamma,
            w1=args.w1,
            w2=args.w2,
            w3=args.w3,
            w4=args.w4,
            w_syn=args.w_syn,
            beta_starve=args.beta_starve,
            w_house=args.w_house,
            w_cash=args.w_cash,
            allow_trade=not args.curriculum,
            randomize_first_player=args.randomize_first_player,
        )
        for i in range(args.num_envs)
    ]

    use_shm = args.num_workers > 1 and args.use_shm
    use_subproc = args.num_workers > 1 and not args.use_shm

    vec_env = None
    envs = None

    if use_shm:
        vec_env = SharedMemoryMonopolyVecEnv(env_configs, num_workers=args.num_workers)
        obs_dim = vec_env.obs_dim
        state_dim = vec_env.state_dim
    elif use_subproc:
        vec_env = SubprocMonopolyVecEnv(env_configs, num_workers=args.num_workers)
        obs_dim = vec_env.obs_dim
        state_dim = vec_env.state_dim
    else:
        envs = [MonopolyParallelEnv(**cfg) for cfg in env_configs]
        obs_dim = envs[0].obs_dim
        state_dim = envs[0].state_dim

    print(f"Observation Dimension: {obs_dim} (Synergy Features: {'Enabled' if obs_dim > 327 else 'Disabled'}) | Global State: {state_dim}")

    # Models & League Manager
    main_actor = ActorNetwork(obs_dim=obs_dim).to(device)
    main_critic = CentralizedCritic(state_dim=state_dim).to(device)
    exploiter_actor = ActorNetwork(obs_dim=obs_dim).to(device)

    # Resume handling
    resume_target = args.resume or args.resume_checkpoint
    if resume_target == "latest":
        resume_target = session_mgr.find_latest_checkpoint() or find_latest_global_checkpoint()

    start_iteration = 1
    total_steps_executed = 0

    if resume_target and os.path.exists(resume_target):
        print(f"--> [Resume] Loading weights and state from {resume_target}")
        ckpt = torch.load(resume_target, map_location=device)
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

        # Restore iteration offset if present
        meta = ckpt.get("metadata", {})
        if "iteration" in meta:
            start_iteration = meta["iteration"] + 1
        if "total_steps" in meta:
            total_steps_executed = meta["total_steps"]
        print(f"--> [Resume] Successfully restored. Resuming from iteration {start_iteration} (Total Steps: {total_steps_executed:,})")

    initial_total_steps = total_steps_executed

    # Trainer & Buffer
    trainer = MAPPOTrainer(
        actor=main_actor,
        critic=main_critic,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        clip_param=args.clip_param,
        ppo_epochs=args.ppo_epochs,
        num_mini_batches=args.num_mini_batches,
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

    league_mgr = LeagueManager(checkpoint_dir=checkpoint_dir, device=device)
    if start_iteration <= 1:
        league_mgr.save_checkpoint(main_actor, iteration=0, role="main")

    env_indices_t = torch.arange(args.num_envs, device=device)
    actions_per_env = {
        k: torch.zeros((args.num_envs, 4), dtype=torch.long, device=device)
        for k in buffer.action_keys
    }
    log_probs_per_env = torch.zeros((args.num_envs, 4), dtype=torch.float32, device=device)
    active_masks_step = torch.zeros((args.num_envs, 4), dtype=torch.float32, device=device)

    # Fallback preallocated numpy arrays for non-SHM mode
    batch_obs_np = np.zeros((args.num_envs, 4, obs_dim), dtype=np.float32)
    batch_states_np = np.zeros((args.num_envs, state_dim), dtype=np.float32)
    batch_masks_np = {k: np.zeros((args.num_envs, 4, buffer.mask_dims[k]), dtype=np.int8) for k in buffer.action_keys}
    active_players_np = np.zeros(args.num_envs, dtype=np.int64)
    rewards_np = np.zeros((args.num_envs, 4), dtype=np.float32)
    dones_np = np.zeros((args.num_envs, 4), dtype=np.float32)

    # Initial environment reset
    if use_shm:
        vec_env.reset()
        shm_obs = torch.from_numpy(vec_env.obs_buf)
        shm_state = torch.from_numpy(vec_env.state_buf)
        shm_rewards = torch.from_numpy(vec_env.rewards_buf)
        shm_dones = torch.from_numpy(vec_env.dones_buf)
        shm_act_p = torch.from_numpy(vec_env.active_player_buf)
        shm_masks = {k: torch.from_numpy(vec_env.mask_bufs[k]) for k in buffer.action_keys}
    elif use_subproc:
        current_obs_list, current_infos_list = vec_env.reset()
    else:
        current_obs_list = []
        current_infos_list = []
        for env in envs:
            o, info = env.reset()
            current_obs_list.append(o)
            current_infos_list.append(info)

    start_time = time.time()
    telemetry_window = {
        "episodes": 0,
        "trades_proposed": 0,
        "trades_accepted": 0,
        "houses_built": 0,
        "bank_houses_end": [],
    }

    try:
        # Main Training Loop
        for iteration in range(start_iteration, args.num_iterations + 1):
            if session_mgr.interrupted:
                break

            # Dynamic cosine entropy annealing (sustains exploration early, enforces crisp economic play late)
            if not args.no_entropy_anneal and args.num_iterations > 1:
                frac = min(1.0, max(0.0, (iteration - 1) / float(args.num_iterations - 1)))
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * frac))
                current_ent_coef = args.entropy_min + cosine_decay * (args.entropy_coef - args.entropy_min)
                trainer.set_entropy_coef(current_ent_coef)

            # Curriculum Learning toggle
            is_curriculum_phase_1 = args.curriculum and (iteration <= int(args.curriculum_phase_2_frac * args.num_iterations))
            allow_trade = not is_curriculum_phase_1
            if use_shm or use_subproc:
                vec_env.set_allow_trade(allow_trade)
            else:
                for env in envs:
                    env.set_allow_trade(allow_trade)

            main_actor.train()
            main_critic.train()

            # Rollout collection loop
            for step in range(args.rollout_steps):
                if use_shm:
                    # ZERO-COPY FAST PATH: Data is already in shm_obs, shm_state, shm_masks
                    with torch.inference_mode():
                        values = main_critic(shm_state)
                        act_p = shm_act_p
                        active_obs = shm_obs[env_indices_t, act_p]
                        active_masks_in = {k: shm_masks[k][env_indices_t, act_p] for k in buffer.action_keys}
                        act_dict_active, lp_active, _ = main_actor(active_obs, active_masks_in)

                        for k in buffer.action_keys:
                            actions_per_env[k].fill_(0)
                        actions_per_env["property"].fill_(160)
                        actions_per_env["trade_offer_prop"].fill_(40)
                        actions_per_env["trade_req_prop"].fill_(40)

                        for k in buffer.action_keys:
                            actions_per_env[k][env_indices_t, act_p] = act_dict_active[k]

                        log_probs_per_env.zero_()
                        log_probs_per_env[env_indices_t, act_p] = lp_active

                        active_masks_step.zero_()
                        active_masks_step[env_indices_t, act_p] = 1.0

                        # Heterogeneous League Rollout
                        if num_league_envs > 0 and league_mgr.historical_records:
                            hist_policy = league_mgr.get_random_historical_policy()
                            if hist_policy is not None:
                                hist_env_mask = (env_indices_t < num_league_envs) & (act_p >= 2)
                                hist_idx = hist_env_mask.nonzero(as_tuple=True)[0]
                                if hist_idx.numel() > 0:
                                    hist_p = act_p[hist_idx]
                                    opp_obs = shm_obs[hist_idx, hist_p]
                                    if hasattr(hist_policy, "obs_dim") and hist_policy.obs_dim < obs_dim:
                                        opp_obs = opp_obs[:, :hist_policy.obs_dim]
                                    opp_masks = {k: shm_masks[k][hist_idx, hist_p] for k in buffer.action_keys}
                                    opp_acts, opp_lps, _ = hist_policy(opp_obs, opp_masks)
                                    for k in buffer.action_keys:
                                        actions_per_env[k][hist_idx, hist_p] = opp_acts[k]
                                    log_probs_per_env[hist_idx, hist_p] = opp_lps
                                    active_masks_step[hist_idx, hist_p] = 0.0

                        # Direct write to shared action buffers
                        for k in buffer.action_keys:
                            vec_env.act_bufs[k][:] = actions_per_env[k].cpu().numpy()

                    # Insert state snapshot into buffer prior to stepping
                    buffer.insert(
                        obs=shm_obs,
                        global_state=shm_state,
                        actions=actions_per_env,
                        action_masks=shm_masks,
                        rewards=shm_rewards,  # updated immediately after step
                        dones=shm_dones,
                        values=values,
                        log_probs=log_probs_per_env,
                        active_masks=active_masks_step,
                    )

                    # Step parallel environments
                    _, _, _, _, _, _, step_stats = vec_env.step()
                    # Store exact post-step rewards and terminal flags
                    buffer.rewards[buffer.step - 1] = shm_rewards
                    buffer.dones[buffer.step - 1] = shm_dones

                    if step_stats["episodes"] > 0:
                        telemetry_window["episodes"] += step_stats["episodes"]
                        telemetry_window["trades_proposed"] += step_stats["trades_proposed"]
                        telemetry_window["trades_accepted"] += step_stats["trades_accepted"]
                        telemetry_window["houses_built"] += step_stats["houses_built"]
                        telemetry_window["bank_houses_end"].extend(step_stats["bank_houses"])

                else:
                    # STANDARD FALLBACK PATH (for Subproc / Single-Process)
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

                    batch_obs = torch.as_tensor(batch_obs_np, device=device)
                    batch_states = torch.as_tensor(batch_states_np, device=device)
                    batch_masks = {k: torch.as_tensor(batch_masks_np[k], device=device) for k in buffer.action_keys}
                    act_p_t = torch.as_tensor(active_players_np, device=device)

                    with torch.inference_mode():
                        values = main_critic(batch_states)
                        active_obs = batch_obs[env_indices_t, act_p_t]
                        active_masks_in = {k: batch_masks[k][env_indices_t, act_p_t] for k in buffer.action_keys}
                        act_dict_active, lp_active, _ = main_actor(active_obs, active_masks_in)

                        for k in buffer.action_keys:
                            actions_per_env[k].fill_(0)
                        actions_per_env["property"].fill_(160)
                        actions_per_env["trade_offer_prop"].fill_(40)
                        actions_per_env["trade_req_prop"].fill_(40)
                        for k in buffer.action_keys:
                            actions_per_env[k][env_indices_t, act_p_t] = act_dict_active[k]

                        log_probs_per_env.zero_()
                        log_probs_per_env[env_indices_t, act_p_t] = lp_active
                        active_masks_step.zero_()
                        active_masks_step[env_indices_t, act_p_t] = 1.0

                        if num_league_envs > 0 and league_mgr.historical_records:
                            hist_policy = league_mgr.get_random_historical_policy()
                            if hist_policy is not None:
                                hist_env_mask = (env_indices_t < num_league_envs) & (act_p_t >= 2)
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

                    a_cpu = {k: actions_per_env[k].cpu().numpy() for k in buffer.action_keys}
                    all_env_actions = [
                        {
                            f"player_{p_idx}": {k: int(a_cpu[k][e_idx, p_idx]) for k in buffer.action_keys}
                            for p_idx in range(4)
                        }
                        for e_idx in range(args.num_envs)
                    ]

                    if use_subproc:
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
            with torch.inference_mode():
                if use_shm:
                    last_values = main_critic(shm_state)
                    last_dones = shm_dones
                else:
                    for e_idx in range(args.num_envs):
                        batch_states_np[e_idx] = current_infos_list[e_idx]["player_0"]["global_state"]
                    last_states = torch.as_tensor(batch_states_np, device=device)
                    last_values = main_critic(last_states)
                    last_dones = dones_tensor

            buffer.compute_gae(last_values, last_dones)

            # MAPPO gradient update
            train_metrics = trainer.train_step(buffer)

            # Telemetry & Logging
            elapsed = time.time() - start_time
            fps = int((total_steps_executed - initial_total_steps) / max(0.001, elapsed))
            progress_pct = (iteration / args.num_iterations) * 100.0

            log_entry = {
                "iteration": iteration,
                "total_steps": total_steps_executed,
                "progress_pct": round(progress_pct, 2),
                "fps": fps,
                "policy_loss": train_metrics["policy_loss"],
                "value_loss": train_metrics["value_loss"],
                "entropy": train_metrics["entropy"],
                "entropy_coef": trainer.entropy_coef,
            }

            if iteration % 5 == 0 or iteration == 1 or iteration == args.num_iterations:
                curriculum_tag = "P1: Foundation" if is_curriculum_phase_1 else "P2: Full Trade"
                ep_cnt = telemetry_window["episodes"]
                if ep_cnt > 0:
                    tr_prop = telemetry_window["trades_proposed"]
                    tr_acc = telemetry_window["trades_accepted"]
                    acc_rate = (tr_acc / tr_prop * 100.0) if tr_prop > 0 else 0.0
                    h_built = telemetry_window["houses_built"] / ep_cnt
                    avg_bh = sum(telemetry_window["bank_houses_end"]) / len(telemetry_window["bank_houses_end"])
                    telemetry_str = f" | Trades: {tr_acc}/{tr_prop} ({acc_rate:.1f}%) | Houses/Ep: {h_built:.1f} | BankH: {avg_bh:.1f}"
                    log_entry.update({
                        "episodes": ep_cnt, "trades_proposed": tr_prop, "trades_accepted": tr_acc,
                        "trade_acc_rate": acc_rate, "houses_per_ep": h_built, "end_bank_houses": avg_bh
                    })
                else:
                    live_stats = vec_env.get_live_stats() if (use_shm or use_subproc) else [
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

            session_mgr.log_iteration(log_entry)

            # Checkpointing
            if iteration % args.checkpoint_interval == 0 or iteration == args.num_iterations:
                ckpt_path = league_mgr.save_checkpoint(
                    main_actor,
                    iteration=iteration,
                    role="main",
                    metadata={"train_metrics": train_metrics, "iteration": iteration, "total_steps": total_steps_executed},
                )
                print(f"--> [Checkpoint Saved] {ckpt_path} (Total Steps: {total_steps_executed:,})")

            # League Evaluation Match (Driven by hardware fluctuation entropy)
            if iteration % args.eval_interval == 0 and league_mgr.historical_records:
                eval_env = MonopolyParallelEnv(
                    max_rounds=100,
                    seed=None,
                    randomize_first_player=args.randomize_first_player,
                )
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
                        with torch.inference_mode():
                            a_dict, _, _ = pol(o_t, m_t, deterministic=True)
                        actions[p_name] = {k: int(a_dict[k].item()) for k in buffer.action_keys}

                    eval_obs, _, terminations, truncations, eval_infos = eval_env.step(actions)
                    done_eval = any(truncations.values()) or all(terminations.values())

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

    except KeyboardInterrupt:
        print("\n[Training Loop] User requested termination.")
        session_mgr.interrupted = True

    finally:
        if session_mgr.interrupted:
            intr_path = os.path.join(checkpoint_dir, "interrupted_checkpoint.pt")
            torch.save({
                "state_dict": main_actor.state_dict(),
                "critic_state_dict": main_critic.state_dict(),
                "metadata": {"total_steps": total_steps_executed, "iteration": iteration if 'iteration' in locals() else 0},
            }, intr_path)
            print(f"--> [Interrupted Checkpoint Saved] {intr_path}")

        if (use_shm or use_subproc) and vec_env is not None:
            vec_env.close()

    total_time = time.time() - start_time
    session_steps = total_steps_executed - initial_total_steps
    avg_speed = int(session_steps / max(0.001, total_time))

    summary_file = session_mgr.save_summary({
        "total_steps": total_steps_executed,
        "session_steps": session_steps,
        "average_fps": avg_speed,
        "final_elo": league_mgr.main_agent_record.elo if hasattr(league_mgr, "main_agent_record") else 1200.0,
        "final_win_rate": league_mgr.main_agent_record.win_rate if hasattr(league_mgr, "main_agent_record") else 0.0,
    })

    print(f"\n=======================================================")
    print(f"=== Training Session Finished! ===")
    print(f"Total Steps: {total_steps_executed:,} (Session: {session_steps:,}) | Elapsed: {total_time:.2f}s ({total_time/60:.1f} min)")
    print(f"Average Speed: {avg_speed:,} steps/sec")
    print(f"Session Summary: {summary_file}")
    print(f"Checkpoints and Metrics in: {checkpoint_dir}/")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
