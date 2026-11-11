"""PettingZoo ParallelEnv Wrapper for Tournament MARL Monopoly Engine.

Conforms to standard PettingZoo multi-agent ParallelEnv interface,
providing vectorized actor observations, factorized action spaces with strict masks,
and centralized global state for CTDE Centralized Critic.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv

from monopoly_marl.engine.monopoly_core import MonopolyCore, TRADE_CASH_BRACKETS


class MonopolyParallelEnv(ParallelEnv):
    """PettingZoo Parallel Environment for 4-agent Monopoly with CTDE support."""

    metadata = {"render_modes": ["human", "rgb_array"], "name": "monopoly_marl_v1"}

    def __init__(self, max_rounds: int = 150, gamma: float = 0.99, seed: Optional[int] = None):
        super().__init__()
        self.possible_agents = [f"player_{i}" for i in range(4)]
        self.agents = list(self.possible_agents)
        self.core = MonopolyCore(seed=seed, max_rounds=max_rounds, gamma=gamma)
        self.seed_val = seed

        # Observation dimension: 327
        self.obs_dim = 327
        # Centralized Critic Global State dimension: 1340
        self.state_dim = 1340

        # Define Observation Spaces for each agent
        # Dict with 'obs' Box and 'action_mask' Dict
        self.observation_spaces = {
            agent: spaces.Dict({
                "obs": spaces.Box(low=-10.0, high=100.0, shape=(self.obs_dim,), dtype=np.float32),
                "action_mask": spaces.Dict({
                    "core": spaces.Box(low=0, high=1, shape=(6,), dtype=np.int8),
                    "auction": spaces.Box(low=0, high=1, shape=(5,), dtype=np.int8),
                    "property": spaces.Box(low=0, high=1, shape=(160,), dtype=np.int8),
                    "trade_target": spaces.Box(low=0, high=1, shape=(3,), dtype=np.int8),
                    "trade_offer_prop": spaces.Box(low=0, high=1, shape=(41,), dtype=np.int8),
                    "trade_offer_cash": spaces.Box(low=0, high=1, shape=(10,), dtype=np.int8),
                    "trade_req_prop": spaces.Box(low=0, high=1, shape=(41,), dtype=np.int8),
                    "trade_req_cash": spaces.Box(low=0, high=1, shape=(10,), dtype=np.int8),
                    "trade_response": spaces.Box(low=0, high=1, shape=(2,), dtype=np.int8),
                })
            })
            for agent in self.possible_agents
        }

        # Define Factorized Multi-Head Action Spaces for each agent
        self.action_spaces = {
            agent: spaces.Dict({
                "core": spaces.Discrete(6),
                "auction": spaces.Discrete(5),
                "property": spaces.Discrete(160),
                "trade_target": spaces.Discrete(3),
                "trade_offer_prop": spaces.Discrete(41),
                "trade_offer_cash": spaces.Discrete(10),
                "trade_req_prop": spaces.Discrete(41),
                "trade_req_cash": spaces.Discrete(10),
                "trade_response": spaces.Discrete(2),
            })
            for agent in self.possible_agents
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Resets the environment and returns (observations, infos)."""
        if seed is not None:
            self.seed_val = seed
        self.agents = list(self.possible_agents)
        core_obs = self.core.reset(self.seed_val)

        observations = {}
        infos = {}
        for agent in self.agents:
            observations[agent] = {
                "obs": core_obs["obs"][agent],
                "action_mask": core_obs["action_mask"][agent],
            }
            infos[agent] = {
                "action_mask": core_obs["action_mask"][agent],
                "active_player": core_obs["active_player"],
                "phase": core_obs["phase"],
                "global_state": core_obs["global_state"],
            }

        return observations, infos

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """Executes actions dictionary for all agents."""
        core_obs, rewards, dones, core_infos = self.core.step(actions)

        observations = {}
        terminations = {}
        truncations = {}
        infos = {}

        is_game_over = dones.get("__all__", False)

        for agent in self.possible_agents:
            observations[agent] = {
                "obs": core_obs["obs"][agent],
                "action_mask": core_obs["action_mask"][agent],
            }
            terminations[agent] = dones.get(agent, False)
            truncations[agent] = is_game_over and not terminations[agent]
            infos[agent] = {
                **core_infos,
                "action_mask": core_obs["action_mask"][agent],
                "active_player": core_obs["active_player"],
                "phase": core_obs["phase"],
                "global_state": core_obs["global_state"],
            }

        # Update remaining active agents
        self.agents = [a for a in self.possible_agents if not terminations[a]]

        return observations, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        """Returns Centralized Critic Global State S_t^{global} for CTDE."""
        return self.core.get_global_state()

    def render(self):
        """Headless text render."""
        print(f"--- Round {self.core.round_count} (Active: Player {self.core.active_decision_player}, Phase {self.core.phase}) ---")
        for i in range(4):
            print(f"Player {i}: Cash=${self.core.cash[i]} Pos={self.core.position[i]} Jail={self.core.in_jail[i]} Bankrupt={self.core.bankrupt[i]}")
        print(f"Bank: Houses={self.core.bank_houses}/32, Hotels={self.core.bank_hotels}/12")
        print("Recent logs:", self.core.event_log[-3:])
