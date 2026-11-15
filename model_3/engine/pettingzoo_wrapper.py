"""PettingZoo ParallelEnv Wrapper for Model-3 Monopoly Engine.

Exposes 3-channel structured actor observations (tile_state, scalar_obs, event_ring),
unconstrained 42-bit trade bundle action/mask shapes, and compact 592-dim global state.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from model_2.engine.monopoly_core import DEFAULT_PLAYER_NAMES
from model_3.engine.monopoly_core import (
    MonopolyCore,
    TILE_FEAT_DIM,
    SCALAR_OBS_DIM,
    EVENT_RING_LEN,
    EVENT_FEAT_DIM,
    GLOBAL_STATE_DIM,
    BUNDLE_DIM,
    MASK_SHAPES,
    TRADE_CASH_DIM,
)


class MonopolyParallelEnvV3(ParallelEnv):
    """PettingZoo Parallel Environment for 4-agent Model-3 Monopoly."""

    metadata = {"render_modes": ["human", "rgb_array"], "name": "model_3_v1"}

    def __init__(
        self,
        max_rounds: Optional[int] = 80,
        gamma: float = 0.997,
        seed: Optional[int] = None,
        w1: float = 0.025,
        w2: float = 0.0034,
        w3: float = 0.014,
        w4: float = 0.0006,
        w_syn: float = 0.030,
        beta_starve: float = 4.0,
        w_house: float = 0.0145,
        w_cash: float = 0.0034,
        w_opp_mono: float = 0.85,
        w_lead: float = 0.80,
        w_income: float = 0.0005,
        w_house_lock: float = 0.35,
        pbrs_scale: float = 1.0,
        allow_trade: bool = True,
        player_names: Optional[List[str]] = None,
        house_rules: Optional[Any] = None,
        randomize_first_player: bool = False,
        fast_search_mode: bool = False,
    ):
        super().__init__()
        self.possible_agents = [f"player_{i}" for i in range(4)]
        self.agents = list(self.possible_agents)
        self.player_names = list(player_names) if player_names is not None else list(DEFAULT_PLAYER_NAMES)
        self.agent_to_name = {f"player_{i}": self.player_names[i] for i in range(len(self.player_names))}

        self.core = MonopolyCore(
            seed=seed,
            max_rounds=max_rounds,
            gamma=gamma,
            w1=w1,
            w2=w2,
            w3=w3,
            w4=w4,
            w_syn=w_syn,
            beta_starve=beta_starve,
            w_house=w_house,
            w_cash=w_cash,
            w_opp_mono=w_opp_mono,
            w_lead=w_lead,
            w_income=w_income,
            w_house_lock=w_house_lock,
            pbrs_scale=pbrs_scale,
            allow_trade=allow_trade,
            player_names=self.player_names,
            house_rules=house_rules,
            randomize_first_player=randomize_first_player,
            fast_search_mode=fast_search_mode,
        )
        self.seed_val = seed

        self.tile_shape = (40, TILE_FEAT_DIM)
        self.scalar_dim = SCALAR_OBS_DIM
        self.event_shape = (EVENT_RING_LEN, EVENT_FEAT_DIM)
        self.state_dim = GLOBAL_STATE_DIM

        self.observation_spaces = {
            agent: spaces.Dict({
                "tile_state": spaces.Box(low=-5.0, high=20.0, shape=self.tile_shape, dtype=np.float32),
                "scalar_obs": spaces.Box(low=-10.0, high=100.0, shape=(self.scalar_dim,), dtype=np.float32),
                "event_ring": spaces.Box(low=0, high=64, shape=self.event_shape, dtype=np.int16),
                "action_mask": spaces.Dict({
                    k: spaces.Box(low=0, high=1, shape=shape, dtype=np.int8)
                    for k, shape in MASK_SHAPES.items()
                }),
            })
            for agent in self.possible_agents
        }

        self.action_spaces = {
            agent: spaces.Dict({
                "core": spaces.Discrete(6),
                "auction": spaces.Discrete(7),
                "property": spaces.Discrete(162),
                "trade_target": spaces.Discrete(3),
                "trade_req_bundle": spaces.MultiBinary(BUNDLE_DIM),
                "trade_offer_bundle": spaces.MultiBinary(BUNDLE_DIM),
                "trade_offer_cash": spaces.Discrete(TRADE_CASH_DIM),
                "trade_req_cash": spaces.Discrete(TRADE_CASH_DIM),
                "trade_response": spaces.Discrete(3),
            })
            for agent in self.possible_agents
        }

    def set_allow_trade(self, allow: bool):
        self.core.set_allow_trade(allow)

    def set_pbrs_scale(self, scale: float):
        self.core.set_pbrs_scale(scale)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            self.seed_val = seed
        self.agents = list(self.possible_agents)
        core_obs = self.core.reset(seed)

        observations = {}
        infos = {}
        for agent in self.agents:
            observations[agent] = {
                "tile_state": core_obs["tile_state"][agent],
                "scalar_obs": core_obs["scalar_obs"][agent],
                "event_ring": core_obs["event_ring"][agent],
                "action_mask": core_obs["action_mask"][agent],
            }
            infos[agent] = {
                "action_mask": core_obs["action_mask"][agent],
                "active_player": core_obs["active_player"],
                "phase": core_obs["phase"],
                "global_state": core_obs["global_state"],
            }
        return observations, infos

    def step(
        self, actions: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        core_obs, rewards, dones, core_infos = self.core.step(actions)
        is_game_over = dones.get("__all__", False)

        observations = {}
        terminations = {}
        truncations = {}
        infos = {}
        for agent in self.possible_agents:
            observations[agent] = {
                "tile_state": core_obs["tile_state"][agent],
                "scalar_obs": core_obs["scalar_obs"][agent],
                "event_ring": core_obs["event_ring"][agent],
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

        self.agents = [a for a in self.possible_agents if not terminations[a]]
        return observations, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        return self.core.get_global_state()
