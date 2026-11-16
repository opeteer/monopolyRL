"""Model-3 MAPPO + Quantile Regression + Bounded Chance Lookahead + PSRO Training Entry.

Designed for 24/7 CPU-Only Multi-Core Servers with Large RAM:
- Zero GPU requirement; batched CPU inference + float16 RAM rollout storage.
- TileMixer + EventMixer structured board & event encoders (~10.3x faster on CPU).
- Unconstrained {0, 1}^42 Multi-Property Bundle Trades + 1x Counter-Offer protocol.
- Selective parallel Search Workers (fast_snapshot + 11-outcome dice lookahead) with
  off-policy-safe Search Distillation (KL(pi_search || pi_theta)).
- Full PSRO League with ProHeuristicBot & OverbidderBot sparring + atomic auto-resume.
"""

import os
import sys
import math
import time
import argparse
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch

from model_3.engine.monopoly_core import (
    MonopolyCore,
    ACTION_KEYS,
    MASK_SHAPES,
    ACT_SHAPES,
    TILE_FEAT_DIM,
    SCALAR_OBS_DIM,
    EVENT_RING_LEN,
    EVENT_FEAT_DIM,
    PHASE_GAME_OVER,
)
from model_3.engine.vec_env import SharedMemoryMonopolyVecEnvV3
from model_3.models.actor_network import ActorNetworkV3
from model_3.models.critic_network import QuantileCritic
from model_3.models.storage import RolloutBufferV3
from model_3.agents.pro_heuristic_bot import ProHeuristicBot, OverbidderBot
from model_3.search.chance_lookahead import _active_head_for_phase
from model_3.training.mappo_trainer import MAPPOTrainerV3
from model_3.training.psro_league import (
    PSROLeagueManager,
    PSROAgentRecord,
    SEAT_KIND_RL,
    SEAT_KIND_PRO_BOT,
    SEAT_KIND_OVERBIDDER_BOT,
)
from model_3.system.hardware import detect_hardware, print_hardware_summary
from model_3.system.config_presets import PRESETS, TrainingConfigV3, get_preset
from model_3.system.session_manager import SessionManagerV3, resolve_latest_checkpoint


def _sample_policy_actions_for_seats(
    policy: ActorNetworkV3,
    tile_np: np.ndarray,
    scalar_np: np.ndarray,
    ev_np: np.ndarray,
    mask_bufs: Dict[str, np.ndarray],
    env_indices: List[int],
    seat_indices: List[int],
    deterministic: bool = False,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Runs one batched forward pass for the subset (env_indices, seat_indices) under inference_mode."""
    with torch.inference_mode():
        t_sub = np.ascontiguousarray(tile_np[env_indices, seat_indices])
        s_sub = np.ascontiguousarray(scalar_np[env_indices, seat_indices])
        e_sub = np.ascontiguousarray(ev_np[env_indices, seat_indices])
        t_t = torch.from_numpy(t_sub)
        s_t = torch.from_numpy(s_sub)
        e_t = torch.from_numpy(e_sub)
        m_t = {
            k: torch.from_numpy(np.ascontiguousarray(mask_bufs[k][env_indices, seat_indices]))
            for k in ACTION_KEYS
        }
        acts, lps, _ = policy(t_t, s_t, e_t, m_t, deterministic=deterministic)

    out_acts: Dict[str, np.ndarray] = {}
    for k in ACTION_KEYS:
        arr = acts[k].cpu().numpy()
        out_acts[k] = arr.astype(np.int8 if len(ACT_SHAPES[k]) > 0 else np.int64, copy=False)
    return out_acts, lps.cpu().numpy().astype(np.float32, copy=False)


def save_named_model(
    actor: ActorNetworkV3,
    critic: torch.nn.Module,
    exploiter_actor: ActorNetworkV3,
    exploiter_optimizer: torch.optim.Optimizer,
    trainer: MAPPOTrainerV3,
    total_steps: int,
    it: int,
    model_name: Optional[str],
    checkpoint_dir: str,
    league: Optional[PSROLeagueManager] = None,
):
    """Saves designated named model checkpoint (e.g. Model-3 Dragonfly)."""
    if not model_name:
        return
    os.makedirs(checkpoint_dir, exist_ok=True)
    sanitized = model_name.replace(" ", "_")
    named_primary = os.path.join(checkpoint_dir, f"{sanitized}.pt")
    named_direct = os.path.join(checkpoint_dir, f"{model_name}.pt")
    payload = {
        "model_name": model_name,
        "iteration": int(it),
        "total_steps": int(total_steps),
        "model_config": {
            "arch": actor.arch,
            "d_model": actor.d_model,
            "hidden_dim": actor.hidden_dim,
            "critic_hidden_dim": getattr(critic, "hidden_dim", 512),
            "num_quantiles": getattr(critic, "num_quantiles", 32),
        },
        "state_dict": {k: v.detach().cpu() for k, v in actor.state_dict().items()},
        "actor_state_dict": {k: v.detach().cpu() for k, v in actor.state_dict().items()},
        "critic_state_dict": {k: v.detach().cpu() for k, v in critic.state_dict().items()},
        "exploiter_state_dict": {k: v.detach().cpu() for k, v in exploiter_actor.state_dict().items()},
        "exploiter_optimizer": exploiter_optimizer.state_dict(),
        "trainer_state": trainer.state_dict(),
        "metadata": {
            "model_name": model_name,
            "total_steps": int(total_steps),
            "final_elo": league.main_record.elo if (league and hasattr(league, "main_record")) else 1200.0,
            "final_win_rate": league.main_record.win_rate if (league and hasattr(league, "main_record")) else 0.0,
            "timestamp": time.time(),
        },
    }
    tmp = named_primary + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, named_primary)
    if named_direct != named_primary:
        tmp_dir = named_direct + ".tmp"
        torch.save(payload, tmp_dir)
        os.replace(tmp_dir, named_direct)
    print(f"--> [Designated Named Model Saved] {named_primary}")


def evaluate_gauntlet(
    actor: ActorNetworkV3,
    league: PSROLeagueManager,
    num_games: int = 8,
    max_rounds: int = 100,
    seed_offset: int = 90000,
) -> Dict[str, float]:
    """Runs balanced-seat gauntlet games: 1 MainAgent vs 2 ProHeuristicBot vs 1 OverbidderBot."""
    actor.eval()
    wins = 0
    ranks_all = []
    net_worths = []
    trades_prop_total = 0
    trades_acc_total = 0
    locks_4h_total = 0

    for g in range(num_games):
        core = MonopolyCore(max_rounds=max_rounds, randomize_first_player=False, seed=seed_offset + g)
        core.reset(seed=seed_offset + g)
        main_seat = g % 4
        bots = {}
        seats: List[PSROAgentRecord] = []
        for p in range(4):
            rel = (p - main_seat) % 4
            if rel == 0:
                seats.append(league.main_record)
            elif rel in (1, 2):
                bots[p] = ProHeuristicBot(seed=seed_offset + g * 10 + p)
                seats.append(league.pro_bot_record)
            else:
                bots[p] = OverbidderBot(seed=seed_offset + g * 10 + p)
                seats.append(league.overbidder_bot_record)

        while core.phase != PHASE_GAME_OVER:
            ap = core.active_decision_player
            if ap == main_seat:
                tile, scalar, ev = core.get_actor_structured_obs(ap)
                masks = core.get_action_masks(ap)
                with torch.no_grad():
                    t_t = torch.from_numpy(tile).unsqueeze(0)
                    s_t = torch.from_numpy(scalar).unsqueeze(0)
                    e_t = torch.from_numpy(ev.astype(np.int64)).unsqueeze(0)
                    m_t = {k: torch.from_numpy(masks[k].astype(np.float32)).unsqueeze(0) for k in ACTION_KEYS}
                    acts_t, _, _ = actor(t_t, s_t, e_t, m_t, deterministic=False)
                act = {
                    k: (acts_t[k][0].cpu().numpy().astype(np.int8) if len(ACT_SHAPES[k]) > 0 else int(acts_t[k][0].item()))
                    for k in ACTION_KEYS
                }
            else:
                act = bots[ap].act(core, ap)
            _, _, dones, _ = core.step({f"player_{ap}": act}, skip_obs=True)
            if all(dones.values()):
                break

        nws = [int(core.get_player_net_worth(p)) for p in range(4)]
        survivors = sorted([p for p in range(4) if not core.bankrupt[p]], key=lambda x: nws[x], reverse=True)
        finish_order = survivors + list(reversed(core.elimination_order))
        ranks = [4, 4, 4, 4]
        for rk_i, p_i in enumerate(finish_order):
            ranks[p_i] = rk_i + 1

        if ranks[main_seat] == 1:
            wins += 1
        ranks_all.append(ranks[main_seat])
        net_worths.append(nws[main_seat])
        trades_prop_total += core.trades_proposed_count
        trades_acc_total += core.trades_accepted_count
        if core.bank_houses <= 16:
            locks_4h_total += 1
        league.record_match_result(seats, ranks, nws, record_for_mle=True)

    ratings = league.calibrate_mle_elos()
    return {
        "eval_win_rate": wins / max(1, num_games),
        "eval_avg_rank": float(np.mean(ranks_all)) if ranks_all else 4.0,
        "eval_avg_nw": float(np.mean(net_worths)) if net_worths else 0.0,
        "eval_trades_prop": trades_prop_total / max(1, num_games),
        "eval_trades_acc": trades_acc_total / max(1, num_games),
        "eval_4h_lock_rate": locks_4h_total / max(1, num_games),
        "main_elo": float(ratings.get("MainAgent", league.main_record.elo)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model-3 CPU-Only MARL Monopoly Training")
    p.add_argument("--preset", type=str, default="server_cpu_standard", choices=list(PRESETS.keys()))
    p.add_argument("--session_name", type=str, default=None)
    p.add_argument("--base_dir", type=str, default="runs_model_3")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints_model_3", help="Directory for persistent named model checkpoints")
    p.add_argument("--model_name", type=str, default="Model-3 Dragonfly", help="Designated model name saved on completion (default: 'Model-3 Dragonfly')")
    p.add_argument("--auto_hardware", action="store_true")
    p.add_argument("--arch", type=str, default=None, choices=["tile_mixer", "light_tx"])
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--total_steps", type=int, default=None)
    p.add_argument("--rollout_steps", type=int, default=None)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--num_threads", type=int, default=None)
    p.add_argument("--ppo_epochs", type=int, default=None, help="PPO update epochs (default from preset)")
    p.add_argument("--critic_extra_epochs", type=int, default=None, help="Extra Critic-only epochs per iteration")
    p.add_argument("--num_mini_batches", type=int, default=None, help="Number of PPO mini-batches (default from preset)")
    p.add_argument("--search_worker_ratio", type=float, default=None)
    p.add_argument("--pbrs_anneal", dest="pbrs_anneal", action="store_true")
    p.add_argument("--no_pbrs_anneal", dest="pbrs_anneal", action="store_false")
    p.set_defaults(pbrs_anneal=None)
    p.add_argument("--resume", type=str, default="none", help="'none', 'latest', or explicit .pt path")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfigV3:
    cfg = get_preset(args.preset)
    if args.auto_hardware:
        hw = detect_hardware()
        print_hardware_summary(hw)
        cfg.num_workers = hw.recommended_workers
        cfg.num_envs = hw.recommended_envs
        cfg.num_threads = hw.recommended_torch_threads
        cfg.rollout_steps = hw.recommended_rollout_steps

    for attr in ("arch", "d_model", "hidden_dim", "total_steps", "rollout_steps", "num_envs", "num_workers", "num_threads", "ppo_epochs", "critic_extra_epochs", "num_mini_batches", "search_worker_ratio", "pbrs_anneal"):
        val = getattr(args, attr, None)
        if val is not None:
            setattr(cfg, attr, val)

    # Ensure batch size matches mini-batches cleanly
    cfg.num_workers = min(max(1, cfg.num_workers), cfg.num_envs)
    return cfg


def train(args: argparse.Namespace):
    cfg = build_config(args)
    torch.set_num_threads(max(1, int(cfg.num_threads)))
    device = torch.device("cpu")
    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    resume_ckpt_path = resolve_latest_checkpoint(args.resume, base_dir=args.base_dir)
    ckpt_payload = None
    if resume_ckpt_path is not None:
        print(f"[Resume] Loading checkpoint from: {resume_ckpt_path}")
        ckpt_payload = PSROLeagueManager.load_training_checkpoint(resume_ckpt_path, map_location=device)
        m_cfg = ckpt_payload.get("model_config", {})
        cfg.arch = m_cfg.get("arch", cfg.arch)
        cfg.d_model = int(m_cfg.get("d_model", cfg.d_model))
        cfg.hidden_dim = int(m_cfg.get("hidden_dim", cfg.hidden_dim))
        cfg.critic_hidden_dim = int(m_cfg.get("critic_hidden_dim", cfg.critic_hidden_dim))
        cfg.num_quantiles = int(m_cfg.get("num_quantiles", cfg.num_quantiles))

    effective_session_name = args.session_name
    if resume_ckpt_path is not None and effective_session_name is None:
        parent_dir = os.path.dirname(os.path.abspath(resume_ckpt_path))
        if os.path.basename(parent_dir) == "checkpoints":
            effective_session_name = os.path.basename(os.path.dirname(parent_dir))

    session = SessionManagerV3(
        session_name=effective_session_name,
        base_dir=args.base_dir,
        config_dict=cfg.to_dict(),
    )
    league = PSROLeagueManager(checkpoint_dir=session.checkpoint_dir, device=device)
    if args.model_name and hasattr(league, "main_record"):
        league.main_record.name = args.model_name

    actor = ActorNetworkV3(
        d_model=cfg.d_model,
        hidden_dim=cfg.hidden_dim,
        arch=cfg.arch,
    ).to(device)
    critic = QuantileCritic(
        hidden_dim=cfg.critic_hidden_dim,
        num_quantiles=cfg.num_quantiles,
    ).to(device)
    exploiter_actor = ActorNetworkV3(
        d_model=cfg.d_model,
        hidden_dim=cfg.hidden_dim,
        arch=cfg.arch,
    ).to(device)
    exploiter_actor.load_state_dict(actor.state_dict())
    exploiter_optimizer = torch.optim.AdamW(exploiter_actor.parameters(), lr=cfg.lr_actor, eps=1e-5, weight_decay=1e-5)

    trainer = MAPPOTrainerV3(
        actor=actor,
        critic=critic,
        lr_actor=cfg.lr_actor,
        lr_critic=cfg.lr_critic,
        clip_param=cfg.clip_param,
        ppo_epochs=cfg.ppo_epochs,
        critic_extra_epochs=cfg.critic_extra_epochs,
        num_mini_batches=cfg.num_mini_batches,
        value_loss_coef=cfg.value_loss_coef,
        entropy_coef=cfg.entropy_coef,
        uniform_kl_coef=cfg.uniform_kl_coef,
        logit_l2_coef=cfg.logit_l2_coef,
        search_distill_coef=cfg.search_distill_coef,
        quantile_huber_kappa=cfg.quantile_huber_kappa,
        cvar_alpha=cfg.cvar_alpha,
        cvar_weight=cfg.cvar_weight,
        target_kl=cfg.target_kl,
        device=device,
    )

    start_iter = 0
    total_steps = 0
    if ckpt_payload is not None:
        actor.load_state_dict(ckpt_payload["actor_state_dict"])
        critic.load_state_dict(ckpt_payload["critic_state_dict"])
        if "exploiter_state_dict" in ckpt_payload:
            exploiter_actor.load_state_dict(ckpt_payload["exploiter_state_dict"])
        if "exploiter_optimizer" in ckpt_payload:
            exploiter_optimizer.load_state_dict(ckpt_payload["exploiter_optimizer"])
        if "trainer_state" in ckpt_payload:
            trainer.load_state_dict(ckpt_payload["trainer_state"])
        start_iter = int(ckpt_payload.get("iteration", 0)) + 1
        total_steps = int(ckpt_payload.get("total_steps", 0))
        print(f"[Resume] Restored iter={start_iter}, total_steps={total_steps:,}")

    if args.eval_only:
        eval_stats = evaluate_gauntlet(actor, league, num_games=cfg.eval_games, max_rounds=cfg.gauntlet_max_rounds)
        print(f"[EvalOnly] {eval_stats}")
        session.restore_signals()
        return

    num_search_envs = int(round(cfg.num_envs * max(0.0, min(1.0, cfg.search_worker_ratio))))
    if cfg.search_worker_ratio > 0 and num_search_envs == 0:
        num_search_envs = 1
    search_env_indices = list(range(num_search_envs))
    search_env_set = set(search_env_indices)

    initial_allow_trade = (not cfg.curriculum) or (cfg.curriculum_trade_frac <= 0.0) or ((total_steps / max(1, cfg.total_steps)) >= cfg.curriculum_trade_frac)
    env_configs = [
        {
            "max_rounds": cfg.max_rounds,
            "allow_trade": initial_allow_trade,
            "randomize_first_player": cfg.randomize_first_player,
            "seed": args.seed + 1000 + i,
            "w_income": cfg.w_income,
            "w_house_lock": cfg.w_house_lock,
        }
        for i in range(cfg.num_envs)
    ]

    search_cfg = {
        "critic_hidden_dim": cfg.critic_hidden_dim,
        "num_quantiles": cfg.num_quantiles,
        "gamma": cfg.gamma,
        "cvar_alpha": cfg.cvar_alpha,
        "cvar_weight": cfg.cvar_weight,
        "top_k": cfg.search_top_k,
        "temperature": cfg.search_temperature,
        "branch_dice": True,
    }
    vec_env = SharedMemoryMonopolyVecEnvV3(
        env_configs=env_configs,
        num_workers=cfg.num_workers,
        search_env_indices=search_env_indices,
        search_cfg=search_cfg,
    )
    buffer = RolloutBufferV3(
        buffer_size=cfg.rollout_steps,
        num_envs=cfg.num_envs,
        num_quantiles=cfg.num_quantiles,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        cvar_alpha=cfg.cvar_alpha,
        cvar_weight=cfg.cvar_weight,
        device=device,
    )

    param_cnt = sum(p.numel() for p in actor.parameters())
    print(f"[Model-3] Session={session.session_name} | Arch={cfg.arch}(d={cfg.d_model}, h={cfg.hidden_dim}, params={param_cnt:,})")
    print(f"[Model-3] CPU Envs={cfg.num_envs} ({cfg.num_workers} workers) | SearchEnvs={num_search_envs}/{cfg.num_envs} | Threads={cfg.num_threads}")

    steps_per_iter = cfg.rollout_steps * cfg.num_envs
    num_iters = max(1, int(math.ceil(cfg.total_steps / steps_per_iter)))
    current_allow_trade = initial_allow_trade
    tile, scalar, ev, gs, masks, ap = vec_env.reset()

    assignments, bot_kinds = league.sample_matchups(
        cfg.num_envs, search_env_set=search_env_set, rng=np_rng, sparring_prob=cfg.sparring_prob
    )
    vec_env.bot_seat_types_buf[:] = bot_kinds

    last_iter = start_iter - 1
    try:
        for it in range(start_iter, num_iters):
            if session.interrupted:
                break

            progress = min(1.0, total_steps / max(1, cfg.total_steps))
            if not cfg.no_lr_anneal:
                cos_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
                cur_lr_a = cfg.lr_actor_min + (cfg.lr_actor - cfg.lr_actor_min) * cos_mult
                cur_lr_c = cfg.lr_critic_min + (cfg.lr_critic - cfg.lr_critic_min) * cos_mult
                trainer.set_lr(cur_lr_a, cur_lr_c)

            if not cfg.no_entropy_anneal:
                trainer.set_entropy_coef(cfg.entropy_coef + (cfg.entropy_min - cfg.entropy_coef) * progress)
                trainer.set_uniform_kl_coef(cfg.uniform_kl_coef + (cfg.uniform_kl_min - cfg.uniform_kl_coef) * progress)

            if not current_allow_trade and cfg.curriculum and progress >= cfg.curriculum_trade_frac:
                current_allow_trade = True
                vec_env.set_allow_trade(True)

            if cfg.pbrs_anneal:
                p_scale = max(cfg.pbrs_min_scale, 1.0 - (1.0 - cfg.pbrs_min_scale) * progress)
                vec_env.set_pbrs_scale(p_scale)

            vec_env.sync_search_critic(critic.state_dict())

            actor.eval()
            exploiter_actor.eval()
            critic.eval()
            t_roll_start = time.time()

            ep_count = 0
            trades_prop = 0
            trades_acc = 0
            trades_cnt = 0
            houses_built = 0

            for _ in range(cfg.rollout_steps):
                step_idx = buffer.insert_obs(tile, scalar, ev, gs, masks)

                with torch.inference_mode():
                    rv, qs = critic.get_risk_value(
                        buffer.global_states[step_idx],
                        cvar_alpha=cfg.cvar_alpha,
                        cvar_weight=cfg.cvar_weight,
                    )
                rv_np = rv.cpu().numpy()
                qs_np = qs.cpu().numpy()

                # Group ONLY active decision RL seats by policy instance (4x fewer actor passes on CPU!)
                policy_groups: Dict[int, Tuple[ActorNetworkV3, List[int], List[int]]] = {}
                active_mask_step = np.zeros((cfg.num_envs, 4), dtype=np.float32)
                exploiter_mask_step = np.zeros((cfg.num_envs, 4), dtype=np.float32)

                for e_i in range(cfg.num_envs):
                    a_p = int(ap[e_i])
                    rec = assignments[e_i][a_p]
                    if rec.bot_kind != SEAT_KIND_RL:
                        continue
                    if rec.role == "main":
                        pol = actor
                        active_mask_step[e_i, a_p] = 1.0
                    elif rec.role == "exploiter":
                        pol = exploiter_actor
                        exploiter_mask_step[e_i, a_p] = 1.0
                    else:
                        pol = league.load_historical_policy(rec)
                    pid = id(pol)
                    if pid not in policy_groups:
                        policy_groups[pid] = (pol, [], [])
                    policy_groups[pid][1].append(e_i)
                    policy_groups[pid][2].append(a_p)

                # Clear action buffers and fill ONLY active RL seats
                for k in ACTION_KEYS:
                    vec_env.act_bufs[k].fill(0)
                lps_step = np.zeros((cfg.num_envs, 4), dtype=np.float32)

                for pol, e_list, p_list in policy_groups.values():
                    sampled, lps_sub = _sample_policy_actions_for_seats(
                        pol, tile, scalar, ev, masks, e_list, p_list, deterministic=False
                    )
                    for k in ACTION_KEYS:
                        vec_env.act_bufs[k][e_list, p_list] = sampled[k]
                    lps_step[e_list, p_list] = lps_sub

                # Provide actor priors to Search Envs when active phase is high-leverage
                if search_env_indices:
                    vec_env.search_prior_logits_buf.fill(0.0)
                    s_envs_hi = []
                    s_ps_hi = []
                    s_heads_hi = []
                    for s_e in search_env_indices:
                        a_p = int(ap[s_e])
                        ph_vec = scalar[s_e, a_p, 40:48]
                        ph = int(np.argmax(ph_vec)) if np.any(ph_vec > 0.5) else 0
                        h_name = _active_head_for_phase(ph)
                        if h_name is not None:
                            s_envs_hi.append(s_e)
                            s_ps_hi.append(a_p)
                            s_heads_hi.append(h_name)
                    if s_envs_hi:
                        with torch.inference_mode():
                            l_dict = actor.get_head_logits(
                                torch.from_numpy(tile[s_envs_hi, s_ps_hi]),
                                torch.from_numpy(scalar[s_envs_hi, s_ps_hi]),
                                torch.from_numpy(ev[s_envs_hi, s_ps_hi].astype(np.int64)),
                                {k: torch.from_numpy(masks[k][s_envs_hi, s_ps_hi]) for k in ACTION_KEYS},
                            )
                        for idx_s, (s_e, h_name) in enumerate(zip(s_envs_hi, s_heads_hi)):
                            vec_env.search_prior_logits_buf[s_e, : l_dict[h_name].shape[-1]] = (
                                l_dict[h_name][idx_s].cpu().numpy()
                            )

                acts_step = {k: vec_env.act_bufs[k].copy() for k in ACTION_KEYS}
                tile, scalar, ev, gs, masks, rews, dones, ap, step_stats = vec_env.step(use_search=bool(search_env_indices))

                # On-policy PPO guard: zero PPO mask wherever BoundedChanceLookahead overrode the sampled action
                overridden = vec_env.search_overridden_buf
                if np.any(overridden > 0.5):
                    active_mask_step[overridden > 0.5] = 0.0

                buffer.insert_step(
                    step_idx,
                    acts_step,
                    rews,
                    dones,
                    rv_np,
                    qs_np,
                    lps_step,
                    active_masks=active_mask_step,
                    exploiter_masks=exploiter_mask_step,
                    search_valid=vec_env.search_valid_buf,
                    search_head=vec_env.search_head_buf,
                    search_probs=vec_env.search_probs_buf,
                )

                ep_count += step_stats["episodes"]
                trades_prop += step_stats["trades_proposed"]
                trades_acc += step_stats["trades_accepted"]
                trades_cnt += step_stats["trades_countered"]
                houses_built += step_stats["houses_built"]

                for g_idx, ranks, nws in step_stats["completed_matches"]:
                    league.record_match_result(assignments[g_idx], ranks, nws, record_for_mle=False)
                    new_assign, new_bk = league.sample_matchups(
                        1, search_env_set={0} if g_idx in search_env_set else set(), rng=np_rng, sparring_prob=cfg.sparring_prob
                    )
                    assignments[g_idx] = new_assign[0]
                    vec_env.bot_seat_types_buf[g_idx] = new_bk[0]

            last_gs_copy = gs.copy()
            last_dones_copy = dones.copy()
            with torch.no_grad():
                _, last_qs = critic.get_risk_value(torch.from_numpy(last_gs_copy), cvar_alpha=cfg.cvar_alpha, cvar_weight=cfg.cvar_weight)
            buffer.compute_gae_and_quantile_targets(last_qs.cpu().numpy(), last_dones_copy)

            t_upd_start = time.time()
            actor.train()
            exploiter_actor.train()
            critic.train()
            metrics = trainer.train_step(buffer, exploiter_actor=exploiter_actor, exploiter_optimizer=exploiter_optimizer)
            t_end = time.time()

            total_steps += steps_per_iter
            last_iter = it
            sps = int(steps_per_iter / max(1e-4, t_end - t_roll_start))

            log_rec = {
                "iteration": it,
                "total_steps": total_steps,
                "sps": sps,
                "rollout_s": round(t_upd_start - t_roll_start, 2),
                "update_s": round(t_end - t_upd_start, 2),
                "episodes": ep_count,
                "trades_prop": trades_prop,
                "trades_acc": trades_acc,
                "trades_cnt": trades_cnt,
                "houses_built": houses_built,
                "main_elo": round(league.main_record.elo, 1),
                **metrics,
            }

            if (it + 1) % cfg.eval_interval == 0 or it == num_iters - 1:
                ev_stats = evaluate_gauntlet(
                    actor, league, num_games=cfg.eval_games, max_rounds=cfg.gauntlet_max_rounds, seed_offset=args.seed + 50000 + it * 100
                )
                log_rec.update(ev_stats)
                print(
                    f"[Iter {it+1}/{num_iters} | Steps {total_steps:,}] "
                    f"SPS={sps} | p_loss={metrics['policy_loss']:.4f} | q_loss={metrics['value_loss']:.4f} | "
                    f"distill={metrics['distill_loss']:.4f} | WinVsPro={ev_stats['eval_win_rate']*100:.1f}% | Elo={ev_stats['main_elo']:.1f}"
                )
            else:
                print(
                    f"[Iter {it+1}/{num_iters} | Steps {total_steps:,}] "
                    f"SPS={sps} | p_loss={metrics['policy_loss']:.4f} | q_loss={metrics['value_loss']:.4f} | "
                    f"distill={metrics['distill_loss']:.4f} | ent={metrics['entropy']:.3f}"
                )

            session.log_iteration(log_rec)

            if (it + 1) % cfg.historical_save_interval == 0:
                league.save_historical_snapshot(actor, f"main_hist_{it+1}", it)

            if (it + 1) % cfg.checkpoint_interval == 0 or it == num_iters - 1:
                league.save_training_checkpoint(
                    session.latest_checkpoint_path,
                    iteration=it,
                    total_steps=total_steps,
                    actor=actor,
                    critic=critic,
                    exploiter_actor=exploiter_actor,
                    exploiter_optimizer=exploiter_optimizer,
                    trainer_state=trainer.state_dict(),
                )
                save_named_model(
                    actor=actor,
                    critic=critic,
                    exploiter_actor=exploiter_actor,
                    exploiter_optimizer=exploiter_optimizer,
                    trainer=trainer,
                    total_steps=total_steps,
                    it=it,
                    model_name=args.model_name,
                    checkpoint_dir=args.checkpoint_dir,
                    league=league,
                )
    finally:
        if last_iter >= start_iter:
            target_ckpt = (
                os.path.join(session.checkpoint_dir, "interrupted_checkpoint.pt")
                if session.interrupted
                else session.latest_checkpoint_path
            )
            league.save_training_checkpoint(
                target_ckpt,
                iteration=max(0, last_iter),
                total_steps=total_steps,
                actor=actor,
                critic=critic,
                exploiter_actor=exploiter_actor,
                exploiter_optimizer=exploiter_optimizer,
                trainer_state=trainer.state_dict(),
            )
            save_named_model(
                actor=actor,
                critic=critic,
                exploiter_actor=exploiter_actor,
                exploiter_optimizer=exploiter_optimizer,
                trainer=trainer,
                total_steps=total_steps,
                it=last_iter,
                model_name=args.model_name,
                checkpoint_dir=args.checkpoint_dir,
                league=league,
            )
        vec_env.close()
        session.save_summary(
            {
                "last_iteration": max(0, last_iter),
                "total_steps": total_steps,
                "final_main_elo": round(league.main_record.elo, 1),
                "latest_checkpoint": session.latest_checkpoint_path,
            }
        )
        session.restore_signals()
        print(f"[Model-3] Saved session summary -> {session.summary_file}")


if __name__ == "__main__":
    train(parse_args())
