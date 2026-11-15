"""Zero-Copy Shared-Memory Vectorized Environment for Model-3 (CPU-Only Optimized).

Features:
- Enforces single-thread BLAS/PyTorch per worker process to prevent OpenMP thread oversubscription.
- Transfers 3-channel structured observations (tile_state, scalar_obs, event_ring)
  plus unconstrained 42-bit trade bundles without Python pickle IPC during rollout.
- Supports dynamic PBRS scale updates and trade curriculum toggles across all workers.
"""

import os
import atexit
import multiprocessing as mp
from multiprocessing import shared_memory
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from model_3.engine.monopoly_core import (
    TILE_FEAT_DIM,
    SCALAR_OBS_DIM,
    EVENT_RING_LEN,
    EVENT_FEAT_DIM,
    GLOBAL_STATE_DIM,
    ACTION_KEYS,
    MASK_SHAPES,
    ACT_SHAPES,
)
from model_3.engine.pettingzoo_wrapper import MonopolyParallelEnvV3


def _configure_worker_single_thread():
    """Prevents BLAS/OpenMP oversubscription when running dozens of CPU workers."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _shm_worker_loop(
    remote,
    env_configs: List[Dict[str, Any]],
    shard_indices: List[int],
    shm_names: Dict[str, str],
    shapes: Dict[str, Tuple[int, ...]],
    dtypes: Dict[str, Any],
    search_env_set: set,
    search_cfg: Dict[str, Any],
):
    """Worker process writing 3-channel observations & reading actions directly via SharedMemory."""
    _configure_worker_single_thread()
    shms = {k: shared_memory.SharedMemory(name=shm_names[k]) for k in shm_names}
    arrays = {k: np.ndarray(shapes[k], dtype=dtypes[k], buffer=shms[k].buf) for k in shm_names}
    envs = [MonopolyParallelEnvV3(**cfg) for cfg in env_configs]
    from model_3.agents.pro_heuristic_bot import ProHeuristicBot, OverbidderBot
    pro_bot = ProHeuristicBot(seed=shard_indices[0] + 101 if shard_indices else 101)
    over_bot = OverbidderBot(seed=shard_indices[0] + 202 if shard_indices else 202)

    lookahead = None
    search_critic = None
    has_search = any(g_idx in search_env_set for g_idx in shard_indices)
    if has_search:
        from model_3.models.critic_network import QuantileCritic
        from model_3.search.chance_lookahead import (
            BoundedChanceLookahead,
            _active_head_for_phase,
        )

        search_critic = QuantileCritic(
            state_dim=GLOBAL_STATE_DIM,
            hidden_dim=int(search_cfg.get("critic_hidden_dim", 512)),
            num_quantiles=int(search_cfg.get("num_quantiles", 32)),
        )
        search_critic.eval()
        lookahead = BoundedChanceLookahead(
            gamma=float(search_cfg.get("gamma", 0.997)),
            cvar_alpha=float(search_cfg.get("cvar_alpha", 0.25)),
            cvar_weight=float(search_cfg.get("cvar_weight", 0.30)),
            top_k=int(search_cfg.get("top_k", 4)),
            temperature=float(search_cfg.get("temperature", 0.25)),
            branch_dice=bool(search_cfg.get("branch_dice", True)),
        )

    try:
        while True:
            cmd = remote.recv()
            if cmd == "reset":
                for env, g_idx in zip(envs, shard_indices):
                    obs_dict, info_dict = env.reset()
                    info_0 = info_dict["player_0"]
                    arrays["state"][g_idx] = info_0["global_state"]
                    arrays["active_player"][g_idx] = info_0["active_player"]
                    arrays["search_valid"][g_idx].fill(0.0)
                    arrays["search_head"][g_idx].fill(0)
                    arrays["search_overridden"][g_idx].fill(0.0)
                    for p in range(4):
                        p_name = f"player_{p}"
                        p_obs = obs_dict[p_name]
                        arrays["tile_state"][g_idx, p] = p_obs["tile_state"]
                        arrays["scalar_obs"][g_idx, p] = p_obs["scalar_obs"]
                        arrays["event_ring"][g_idx, p] = p_obs["event_ring"]
                        for k in ACTION_KEYS:
                            arrays[f"mask_{k}"][g_idx, p] = p_obs["action_mask"][k]
                remote.send(1)

            elif cmd == "step":
                use_search = bool(remote.recv())
                episodes_done = 0
                trades_prop = 0
                trades_acc = 0
                trades_cnt = 0
                houses_built = 0
                bank_houses_list = []
                completed_matches = []

                for env, g_idx in zip(envs, shard_indices):
                    actions = {}
                    for p in range(4):
                        p_act = {}
                        for k in ACTION_KEYS:
                            if ACT_SHAPES[k] == ():
                                p_act[k] = int(arrays[f"act_{k}"][g_idx, p])
                            else:
                                p_act[k] = arrays[f"act_{k}"][g_idx, p]
                        actions[f"player_{p}"] = p_act

                    arrays["search_valid"][g_idx].fill(0.0)
                    arrays["search_head"][g_idx].fill(0)
                    arrays["search_overridden"][g_idx].fill(0.0)

                    ap = env.core.active_decision_player
                    bot_kind = int(arrays["bot_seat_types"][g_idx, ap])
                    if bot_kind == 1:
                        actions[f"player_{ap}"] = pro_bot.act(env.core, ap)
                    elif bot_kind == 2:
                        actions[f"player_{ap}"] = over_bot.act(env.core, ap)
                    elif use_search and has_search and g_idx in search_env_set and lookahead is not None:
                        head_name = _active_head_for_phase(env.core.phase)
                        if head_name is not None:
                            ap_name = f"player_{ap}"
                            priors = {head_name: arrays["search_prior_logits"][g_idx]}
                            best_act, s_valid, s_head, s_probs = lookahead.select_action(
                                env.core,
                                ap,
                                actions[ap_name],
                                actor_logits=priors,
                                critic=search_critic,
                            )
                            if s_valid:
                                arrays["search_valid"][g_idx, ap] = 1.0
                                arrays["search_head"][g_idx, ap] = s_head
                                arrays["search_probs"][g_idx, ap] = s_probs
                                if int(best_act[head_name]) != int(actions[ap_name][head_name]):
                                    actions[ap_name][head_name] = int(best_act[head_name])
                                    arrays[f"act_{head_name}"][g_idx, ap] = int(best_act[head_name])
                                    arrays["search_overridden"][g_idx, ap] = 1.0

                    next_obs, rewards, terminations, truncations, infos = env.step(actions)
                    is_done = any(truncations.values()) or all(terminations.values())
                    if is_done:
                        episodes_done += 1
                        trades_prop += env.core.trades_proposed_count
                        trades_acc += env.core.trades_accepted_count
                        trades_cnt += env.core.trades_countered_count
                        houses_built += env.core.houses_built_count
                        bank_houses_list.append(env.core.bank_houses)

                        nws = [int(env.core.get_player_net_worth(p_i)) for p_i in range(4)]
                        elim_order = list(env.core.elimination_order)
                        survivors = sorted([p_i for p_i in range(4) if not env.core.bankrupt[p_i]], key=lambda x: nws[x], reverse=True)
                        finish_order = survivors + list(reversed(elim_order))
                        ranks = [4, 4, 4, 4]
                        for rk_idx, p_i in enumerate(finish_order):
                            ranks[p_i] = rk_idx + 1
                        completed_matches.append((g_idx, ranks, nws))

                        pro_bot.reset()
                        over_bot.reset()
                        next_obs, infos = env.reset()

                    info_0 = infos["player_0"]
                    arrays["state"][g_idx] = info_0["global_state"]
                    arrays["active_player"][g_idx] = info_0["active_player"]

                    for p in range(4):
                        p_name = f"player_{p}"
                        p_obs = next_obs[p_name]
                        arrays["tile_state"][g_idx, p] = p_obs["tile_state"]
                        arrays["scalar_obs"][g_idx, p] = p_obs["scalar_obs"]
                        arrays["event_ring"][g_idx, p] = p_obs["event_ring"]
                        arrays["rewards"][g_idx, p] = rewards.get(p_name, 0.0)
                        done_flag = terminations.get(p_name, False) or truncations.get(p_name, False)
                        arrays["dones"][g_idx, p] = 1.0 if done_flag else 0.0
                        for k in ACTION_KEYS:
                            arrays[f"mask_{k}"][g_idx, p] = p_obs["action_mask"][k]

                remote.send((episodes_done, trades_prop, trades_acc, trades_cnt, houses_built, bank_houses_list, completed_matches))

            elif cmd == "set_allow_trade":
                allow = remote.recv()
                for env, g_idx in zip(envs, shard_indices):
                    env.set_allow_trade(allow)
                    obs_bundle = env.core.get_observations(skip_obs=False)
                    arrays["state"][g_idx] = obs_bundle["global_state"]
                    arrays["active_player"][g_idx] = obs_bundle["active_player"]
                    for p in range(4):
                        p_name = f"player_{p}"
                        arrays["tile_state"][g_idx, p] = obs_bundle["tile_state"][p_name]
                        arrays["scalar_obs"][g_idx, p] = obs_bundle["scalar_obs"][p_name]
                        arrays["event_ring"][g_idx, p] = obs_bundle["event_ring"][p_name]
                        for k in ACTION_KEYS:
                            arrays[f"mask_{k}"][g_idx, p] = obs_bundle["action_mask"][p_name][k]
                remote.send(1)

            elif cmd == "set_pbrs_scale":
                scale = float(remote.recv())
                for env in envs:
                    env.set_pbrs_scale(scale)
                remote.send(1)

            elif cmd == "sync_search_critic":
                np_sd = remote.recv()
                if search_critic is not None and np_sd is not None:
                    import torch
                    sd = {k: torch.from_numpy(v) for k, v in np_sd.items()}
                    search_critic.load_state_dict(sd, strict=True)
                remote.send(1)

            elif cmd == "get_live_stats":
                stats = [
                    {
                        "trades_proposed": env.core.trades_proposed_count,
                        "trades_accepted": env.core.trades_accepted_count,
                        "trades_countered": env.core.trades_countered_count,
                        "houses_built": env.core.houses_built_count,
                        "bank_houses": env.core.bank_houses,
                    }
                    for env in envs
                ]
                remote.send(stats)

            elif cmd == "close":
                break
    except (EOFError, BrokenPipeError):
        pass
    finally:
        for s in shms.values():
            try:
                s.close()
            except Exception:
                pass


class SharedMemoryMonopolyVecEnvV3:
    """Zero-Copy Shared-Memory Vectorized Environment for Model-3."""

    def __init__(
        self,
        env_configs: List[Dict[str, Any]],
        num_workers: int,
        search_env_indices: Optional[List[int]] = None,
        search_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.num_envs = len(env_configs)
        self.num_workers = min(max(1, num_workers), self.num_envs)
        self.search_env_set = set(int(i) for i in (search_env_indices or []) if 0 <= int(i) < self.num_envs)
        self.search_cfg = dict(search_cfg or {})

        self.tile_shape = (40, TILE_FEAT_DIM)
        self.scalar_dim = SCALAR_OBS_DIM
        self.event_shape = (EVENT_RING_LEN, EVENT_FEAT_DIM)
        self.state_dim = GLOBAL_STATE_DIM
        self.action_keys = list(ACTION_KEYS)
        self.mask_shapes = dict(MASK_SHAPES)
        self.act_shapes = dict(ACT_SHAPES)

        self.shapes: Dict[str, Tuple[int, ...]] = {
            "tile_state": (self.num_envs, 4, 40, TILE_FEAT_DIM),
            "scalar_obs": (self.num_envs, 4, SCALAR_OBS_DIM),
            "event_ring": (self.num_envs, 4, EVENT_RING_LEN, EVENT_FEAT_DIM),
            "state": (self.num_envs, GLOBAL_STATE_DIM),
            "rewards": (self.num_envs, 4),
            "dones": (self.num_envs, 4),
            "active_player": (self.num_envs,),
            "search_valid": (self.num_envs, 4),
            "search_head": (self.num_envs, 4),
            "search_probs": (self.num_envs, 4, 162),
            "search_overridden": (self.num_envs, 4),
            "search_prior_logits": (self.num_envs, 162),
            "bot_seat_types": (self.num_envs, 4),
        }
        self.dtypes: Dict[str, Any] = {
            "tile_state": np.float32,
            "scalar_obs": np.float32,
            "event_ring": np.int16,
            "state": np.float32,
            "rewards": np.float32,
            "dones": np.float32,
            "active_player": np.int64,
            "search_valid": np.float32,
            "search_head": np.int64,
            "search_probs": np.float32,
            "search_overridden": np.float32,
            "search_prior_logits": np.float32,
            "bot_seat_types": np.int8,
        }

        for k in self.action_keys:
            self.shapes[f"mask_{k}"] = (self.num_envs, 4) + self.mask_shapes[k]
            self.dtypes[f"mask_{k}"] = np.int8
            if self.act_shapes[k] == ():
                self.shapes[f"act_{k}"] = (self.num_envs, 4)
                self.dtypes[f"act_{k}"] = np.int64
            else:
                self.shapes[f"act_{k}"] = (self.num_envs, 4) + self.act_shapes[k]
                self.dtypes[f"act_{k}"] = np.int8

        self.shms: Dict[str, shared_memory.SharedMemory] = {}
        self.arrays: Dict[str, np.ndarray] = {}
        self.shm_names: Dict[str, str] = {}

        try:
            for k in self.shapes:
                nbytes = int(np.prod(self.shapes[k]) * np.dtype(self.dtypes[k]).itemsize)
                shm = shared_memory.SharedMemory(create=True, size=nbytes)
                self.shms[k] = shm
                self.shm_names[k] = shm.name
                arr = np.ndarray(self.shapes[k], dtype=self.dtypes[k], buffer=shm.buf)
                arr.fill(0)
                self.arrays[k] = arr
        except Exception:
            self._unlink_shms()
            raise

        self.tile_buf = self.arrays["tile_state"]
        self.scalar_buf = self.arrays["scalar_obs"]
        self.event_buf = self.arrays["event_ring"]
        self.state_buf = self.arrays["state"]
        self.rewards_buf = self.arrays["rewards"]
        self.dones_buf = self.arrays["dones"]
        self.active_player_buf = self.arrays["active_player"]
        self.search_valid_buf = self.arrays["search_valid"]
        self.search_head_buf = self.arrays["search_head"]
        self.search_probs_buf = self.arrays["search_probs"]
        self.search_overridden_buf = self.arrays["search_overridden"]
        self.search_prior_logits_buf = self.arrays["search_prior_logits"]
        self.bot_seat_types_buf = self.arrays["bot_seat_types"]
        self.mask_bufs = {k: self.arrays[f"mask_{k}"] for k in self.action_keys}
        self.act_bufs = {k: self.arrays[f"act_{k}"] for k in self.action_keys}

        shards: List[List[Dict[str, Any]]] = [[] for _ in range(self.num_workers)]
        self.shard_indices: List[List[int]] = [[] for _ in range(self.num_workers)]
        for i, cfg in enumerate(env_configs):
            shards[i % self.num_workers].append(cfg)
            self.shard_indices[i % self.num_workers].append(i)

        ctx = mp.get_context("spawn")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_workers)])
        self.processes = []
        for w_idx in range(self.num_workers):
            p = ctx.Process(
                target=_shm_worker_loop,
                args=(
                    self.work_remotes[w_idx],
                    shards[w_idx],
                    self.shard_indices[w_idx],
                    self.shm_names,
                    self.shapes,
                    self.dtypes,
                    self.search_env_set,
                    self.search_cfg,
                ),
                daemon=True,
            )
            p.start()
            self.work_remotes[w_idx].close()
            self.processes.append(p)

        self._closed = False
        atexit.register(self.close)

    def reset(self):
        for r in self.remotes:
            r.send("reset")
        for r in self.remotes:
            r.recv()
        return (
            self.tile_buf,
            self.scalar_buf,
            self.event_buf,
            self.state_buf,
            self.mask_bufs,
            self.active_player_buf,
        )

    def sync_search_critic(self, critic_state_dict: Dict[str, Any]):
        if not self.search_env_set:
            return
        np_sd = {k: v.detach().cpu().numpy() for k, v in critic_state_dict.items()}
        active_remotes = []
        for w_idx, r in enumerate(self.remotes):
            if any(g in self.search_env_set for g in self.shard_indices[w_idx]):
                r.send("sync_search_critic")
                r.send(np_sd)
                active_remotes.append(r)
        for r in active_remotes:
            r.recv()

    def step(self, use_search: bool = True):
        flag = bool(use_search and self.search_env_set)
        for r in self.remotes:
            r.send("step")
            r.send(flag)

        episodes = 0
        trades_prop = 0
        trades_acc = 0
        trades_cnt = 0
        houses_built = 0
        bank_houses = []
        completed_matches = []

        for r in self.remotes:
            ep_done, t_p, t_a, t_c, h_b, bh_list, matches = r.recv()
            episodes += ep_done
            trades_prop += t_p
            trades_acc += t_a
            trades_cnt += t_c
            houses_built += h_b
            bank_houses.extend(bh_list)
            completed_matches.extend(matches)

        stats = {
            "episodes": episodes,
            "trades_proposed": trades_prop,
            "trades_accepted": trades_acc,
            "trades_countered": trades_cnt,
            "houses_built": houses_built,
            "bank_houses": bank_houses,
            "completed_matches": completed_matches,
        }
        return (
            self.tile_buf,
            self.scalar_buf,
            self.event_buf,
            self.state_buf,
            self.mask_bufs,
            self.rewards_buf,
            self.dones_buf,
            self.active_player_buf,
            stats,
        )

    def set_allow_trade(self, allow: bool):
        for r in self.remotes:
            r.send("set_allow_trade")
            r.send(bool(allow))
        for r in self.remotes:
            r.recv()

    def set_pbrs_scale(self, scale: float):
        for r in self.remotes:
            r.send("set_pbrs_scale")
            r.send(float(scale))
        for r in self.remotes:
            r.recv()

    def get_live_stats(self) -> List[Dict[str, Any]]:
        for r in self.remotes:
            r.send("get_live_stats")
        stats = []
        for r in self.remotes:
            stats.extend(r.recv())
        return stats

    def _unlink_shms(self):
        for s in self.shms.values():
            try:
                s.close()
                s.unlink()
            except Exception:
                pass

    def close(self):
        if self._closed:
            return
        self._closed = True
        for r in self.remotes:
            try:
                r.send("close")
            except Exception:
                pass
        for p in self.processes:
            try:
                p.join(timeout=1.0)
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass
        self._unlink_shms()
