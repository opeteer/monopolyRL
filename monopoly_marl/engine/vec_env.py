"""Multiprocessing Vectorized Environment Wrapper for Monopoly MARL.

Distributes N MonopolyParallelEnv instances across W CPU worker processes
using Pipe IPC so rollout stepping executes in parallel across physical CPU cores.
"""

import multiprocessing as mp
from typing import List, Dict, Any, Tuple
import numpy as np

from monopoly_marl.engine.pettingzoo_wrapper import MonopolyParallelEnv


def _worker_loop(remote, parent_remote, env_configs: List[Dict[str, Any]]):
    """Worker process holding a shard of MonopolyParallelEnv instances."""
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
    """Vectorized environment distributing N environments across W worker processes."""

    def __init__(self, env_configs: List[Dict[str, Any]], num_workers: int):
        self.num_envs = len(env_configs)
        self.num_workers = min(max(1, num_workers), self.num_envs)

        # Probe dims from single temporary env
        probe = MonopolyParallelEnv(**env_configs[0])
        self.obs_dim = probe.obs_dim
        self.state_dim = probe.state_dim
        del probe

        # Partition env configs evenly across workers
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
