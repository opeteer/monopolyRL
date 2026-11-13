"""Multiprocessing Vectorized Environment Wrapper for Monopoly MARL.

Provides both:
1. SharedMemoryMonopolyVecEnv: Zero-copy, high-performance shared-memory multiprocessing
   vectorized environment scaling across physical CPU cores without pickle IPC overhead.
2. SubprocMonopolyVecEnv: Standard Pipe-IPC multiprocessing fallback.
"""

import os
import atexit
import signal
import multiprocessing as mp
from multiprocessing import shared_memory
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from model_2.engine.monopoly_core import TRADE_CASH_DIM
from model_2.engine.pettingzoo_wrapper import MonopolyParallelEnv

ACTION_KEYS = [
    "core", "auction", "property", "trade_target",
    "trade_offer_prop", "trade_offer_cash", "trade_req_prop",
    "trade_req_cash", "trade_response"
]

MASK_DIMS = {
    "core": 6,
    "auction": 7,
    "property": 161,
    "trade_target": 3,
    "trade_offer_prop": 41,
    "trade_offer_cash": TRADE_CASH_DIM,
    "trade_req_prop": 41,
    "trade_req_cash": TRADE_CASH_DIM,
    "trade_response": 2,
}


def _shm_worker_loop(
    remote,
    env_configs: List[Dict[str, Any]],
    shard_indices: List[int],
    shm_names: Dict[str, str],
    shapes: Dict[str, Tuple],
    dtypes: Dict[str, np.dtype],
):
    """Worker process writing directly into shared memory buffers."""
    # Attach to shared memory blocks
    shms = {k: shared_memory.SharedMemory(name=shm_names[k]) for k in shm_names}
    arrays = {k: np.ndarray(shapes[k], dtype=dtypes[k], buffer=shms[k].buf) for k in shm_names}
    envs = [MonopolyParallelEnv(**cfg) for cfg in env_configs]

    try:
        while True:
            cmd = remote.recv()
            if cmd == "reset":
                for env, g_idx in zip(envs, shard_indices):
                    obs_dict, info_dict = env.reset()
                    info_0 = info_dict["player_0"]
                    arrays["state"][g_idx] = info_0["global_state"]
                    arrays["active_player"][g_idx] = info_0["active_player"]
                    for p in range(4):
                        p_name = f"player_{p}"
                        arrays["obs"][g_idx, p] = obs_dict[p_name]["obs"]
                        for k in ACTION_KEYS:
                            arrays[f"mask_{k}"][g_idx, p] = obs_dict[p_name]["action_mask"][k]
                remote.send(1)

            elif cmd == "step":
                episodes_done = 0
                trades_prop = 0
                trades_acc = 0
                houses_built = 0
                bank_houses_list = []

                for env, g_idx in zip(envs, shard_indices):
                    # Direct reading from shared memory action buffers
                    actions = {
                        f"player_{p}": {k: int(arrays[f"act_{k}"][g_idx, p]) for k in ACTION_KEYS}
                        for p in range(4)
                    }
                    next_obs, rewards, terminations, truncations, infos = env.step(actions)

                    is_done = any(truncations.values()) or all(terminations.values())
                    if is_done:
                        episodes_done += 1
                        trades_prop += env.core.trades_proposed_count
                        trades_acc += env.core.trades_accepted_count
                        houses_built += env.core.houses_built_count
                        bank_houses_list.append(env.core.bank_houses)
                        next_obs, infos = env.reset()

                    info_0 = infos["player_0"]
                    arrays["state"][g_idx] = info_0["global_state"]
                    arrays["active_player"][g_idx] = info_0["active_player"]

                    for p in range(4):
                        p_name = f"player_{p}"
                        arrays["obs"][g_idx, p] = next_obs[p_name]["obs"]
                        arrays["rewards"][g_idx, p] = rewards.get(p_name, 0.0)
                        done_flag = terminations.get(p_name, False) or truncations.get(p_name, False)
                        arrays["dones"][g_idx, p] = 1.0 if done_flag else 0.0
                        for k in ACTION_KEYS:
                            arrays[f"mask_{k}"][g_idx, p] = next_obs[p_name]["action_mask"][k]

                remote.send((episodes_done, trades_prop, trades_acc, houses_built, bank_houses_list))

            elif cmd == "set_allow_trade":
                allow = remote.recv()
                for env in envs:
                    env.set_allow_trade(allow)
                remote.send(1)

            elif cmd == "get_live_stats":
                stats = [
                    {
                        "trades_proposed": env.core.trades_proposed_count,
                        "trades_accepted": env.core.trades_accepted_count,
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


class SharedMemoryMonopolyVecEnv:
    """Zero-Copy Shared-Memory Vectorized Environment for Multi-Agent Monopoly.

    Allocates contiguous memory arrays mapped across W worker processes.
    Master thread receives PyTorch tensors directly wrapping shared memory without
    dictionary serialization or inter-process IPC copying overhead.
    """

    def __init__(self, env_configs: List[Dict[str, Any]], num_workers: int):
        self.num_envs = len(env_configs)
        self.num_workers = min(max(1, num_workers), self.num_envs)

        probe = MonopolyParallelEnv(**env_configs[0])
        self.obs_dim = probe.obs_dim
        self.state_dim = probe.state_dim
        del probe

        self.action_keys = list(ACTION_KEYS)
        self.mask_dims = dict(MASK_DIMS)

        # Buffer specifications
        self.shapes = {
            "obs": (self.num_envs, 4, self.obs_dim),
            "state": (self.num_envs, self.state_dim),
            "rewards": (self.num_envs, 4),
            "dones": (self.num_envs, 4),
            "active_player": (self.num_envs,),
        }
        self.dtypes = {
            "obs": np.float32,
            "state": np.float32,
            "rewards": np.float32,
            "dones": np.float32,
            "active_player": np.int64,
        }
        for k in self.action_keys:
            self.shapes[f"mask_{k}"] = (self.num_envs, 4, self.mask_dims[k])
            self.dtypes[f"mask_{k}"] = np.int8
            self.shapes[f"act_{k}"] = (self.num_envs, 4)
            self.dtypes[f"act_{k}"] = np.int64

        # Allocate shared memory blocks
        self.shms: Dict[str, shared_memory.SharedMemory] = {}
        self.arrays: Dict[str, np.ndarray] = {}
        self.shm_names: Dict[str, str] = {}
        for k in self.shapes:
            nbytes = int(np.prod(self.shapes[k]) * np.dtype(self.dtypes[k]).itemsize)
            shm = shared_memory.SharedMemory(create=True, size=nbytes)
            self.shms[k] = shm
            self.shm_names[k] = shm.name
            arr = np.ndarray(self.shapes[k], dtype=self.dtypes[k], buffer=shm.buf)
            arr.fill(0)
            self.arrays[k] = arr

        # Convenient attribute pointers
        self.obs_buf = self.arrays["obs"]
        self.state_buf = self.arrays["state"]
        self.rewards_buf = self.arrays["rewards"]
        self.dones_buf = self.arrays["dones"]
        self.active_player_buf = self.arrays["active_player"]
        self.mask_bufs = {k: self.arrays[f"mask_{k}"] for k in self.action_keys}
        self.act_bufs = {k: self.arrays[f"act_{k}"] for k in self.action_keys}

        # Partition environments across workers
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
                ),
                daemon=True,
            )
            p.start()
            self.work_remotes[w_idx].close()
            self.processes.append(p)

        self._closed = False
        atexit.register(self.close)

    def reset(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """Resets all environments in parallel and populates shared buffers."""
        for r in self.remotes:
            r.send("reset")
        for r in self.remotes:
            r.recv()
        return self.obs_buf, self.state_buf, self.mask_bufs, self.active_player_buf

    def step(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Steps all environments in parallel reading actions from shared memory act_bufs."""
        for r in self.remotes:
            r.send("step")

        episodes = 0
        trades_prop = 0
        trades_acc = 0
        houses_built = 0
        bank_houses = []

        for r in self.remotes:
            ep_done, t_p, t_a, h_b, bh_list = r.recv()
            episodes += ep_done
            trades_prop += t_p
            trades_acc += t_a
            houses_built += h_b
            bank_houses.extend(bh_list)

        stats = {
            "episodes": episodes,
            "trades_proposed": trades_prop,
            "trades_accepted": trades_acc,
            "houses_built": houses_built,
            "bank_houses": bank_houses,
        }
        return (
            self.obs_buf,
            self.state_buf,
            self.mask_bufs,
            self.rewards_buf,
            self.dones_buf,
            self.active_player_buf,
            stats,
        )

    def set_allow_trade(self, allow: bool):
        """Sets trading curriculum flag across all workers."""
        for r in self.remotes:
            r.send("set_allow_trade")
            r.send(allow)
        for r in self.remotes:
            r.recv()

    def get_live_stats(self) -> List[Dict[str, Any]]:
        for r in self.remotes:
            r.send("get_live_stats")
        stats = []
        for r in self.remotes:
            stats.extend(r.recv())
        return stats

    def close(self):
        """Gracefully shuts down worker processes and unlinks shared memory blocks."""
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

        for s in self.shms.values():
            try:
                s.close()
                s.unlink()
            except Exception:
                pass


def _worker_loop(remote, parent_remote, env_configs: List[Dict[str, Any]]):
    """Worker process holding a shard of MonopolyParallelEnv instances for SubprocVecEnv."""
    parent_remote.close()
    envs = [MonopolyParallelEnv(**cfg) for cfg in env_configs]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                results = [env.reset() for env in envs]
                remote.send(results)
            elif cmd == "step":
                results = []
                for env, actions in zip(envs, data):
                    next_obs, rewards, terminations, truncations, infos = env.step(actions)
                    ep_stats = None
                    if any(truncations.values()) or all(terminations.values()):
                        ep_stats = {
                            "trades_proposed": env.core.trades_proposed_count,
                            "trades_accepted": env.core.trades_accepted_count,
                            "houses_built": env.core.houses_built_count,
                            "bank_houses": env.core.bank_houses,
                        }
                        next_obs, infos = env.reset()
                    results.append((next_obs, rewards, terminations, truncations, infos, ep_stats))
                remote.send(results)
            elif cmd == "set_allow_trade":
                for env in envs:
                    env.set_allow_trade(data)
                remote.send(True)
            elif cmd == "get_live_stats":
                stats = [
                    {
                        "trades_proposed": env.core.trades_proposed_count,
                        "trades_accepted": env.core.trades_accepted_count,
                        "houses_built": env.core.houses_built_count,
                        "bank_houses": env.core.bank_houses,
                    }
                    for env in envs
                ]
                remote.send(stats)
            elif cmd == "close":
                remote.close()
                break
    except EOFError:
        pass


class SubprocMonopolyVecEnv:
    """Vectorized environment distributing N environments across W worker processes (Pipe IPC fallback)."""

    def __init__(self, env_configs: List[Dict[str, Any]], num_workers: int):
        self.num_envs = len(env_configs)
        self.num_workers = min(max(1, num_workers), self.num_envs)

        probe = MonopolyParallelEnv(**env_configs[0])
        self.obs_dim = probe.obs_dim
        self.state_dim = probe.state_dim
        del probe

        shards: List[List[Dict[str, Any]]] = [[] for _ in range(self.num_workers)]
        for i, cfg in enumerate(env_configs):
            shards[i % self.num_workers].append(cfg)
        self.shard_indices: List[List[int]] = [[] for _ in range(self.num_workers)]
        for i in range(self.num_envs):
            self.shard_indices[i % self.num_workers].append(i)

        ctx = mp.get_context("spawn")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_workers)])
        self.processes = []
        for work_r, remote, shard in zip(self.work_remotes, self.remotes, shards):
            p = ctx.Process(target=_worker_loop, args=(work_r, remote, shard), daemon=True)
            p.start()
            work_r.close()
            self.processes.append(p)

    def reset(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        for r in self.remotes:
            r.send(("reset", None))
        obs_list = [None] * self.num_envs
        infos_list = [None] * self.num_envs
        for w_idx, r in enumerate(self.remotes):
            res = r.recv()
            for local_i, (o, info) in enumerate(res):
                g_idx = self.shard_indices[w_idx][local_i]
                obs_list[g_idx] = o
                infos_list[g_idx] = info
        return obs_list, infos_list

    def step(self, all_actions: List[Dict[str, Any]]):
        for w_idx, r in enumerate(self.remotes):
            shard_acts = [all_actions[g_idx] for g_idx in self.shard_indices[w_idx]]
            r.send(("step", shard_acts))

        ordered = [None] * self.num_envs
        for w_idx, r in enumerate(self.remotes):
            res = r.recv()
            for local_i, item in enumerate(res):
                g_idx = self.shard_indices[w_idx][local_i]
                ordered[g_idx] = item
        return ordered

    def set_allow_trade(self, allow: bool):
        for r in self.remotes:
            r.send(("set_allow_trade", allow))
        for r in self.remotes:
            r.recv()

    def get_live_stats(self) -> List[Dict[str, Any]]:
        for r in self.remotes:
            r.send(("get_live_stats", None))
        stats = []
        for r in self.remotes:
            stats.extend(r.recv())
        return stats

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        for p in self.processes:
            p.join(timeout=2.0)
