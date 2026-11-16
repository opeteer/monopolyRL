"""Training Configuration Presets for Model-3 CPU-Only 24/7 Server & Local Dev."""

from dataclasses import dataclass, asdict
from typing import Dict, Any
import json


@dataclass
class TrainingConfigV3:
    preset_name: str = "custom"
    description: str = "Model-3 CPU-only training configuration"

    # Compute & Parallelism
    total_steps: int = 2_000_000
    rollout_steps: int = 256
    num_envs: int = 64
    num_workers: int = 8
    num_threads: int = 8
    device: str = "cpu"

    # Architecture (TileMixer default = ~10.3x faster than Heavy Tx on CPU)
    arch: str = "tile_mixer"  # "tile_mixer" or "light_tx"
    d_model: int = 64
    hidden_dim: int = 384
    critic_hidden_dim: int = 512
    num_quantiles: int = 32

    # Selective Fast-Snapshot Chance Lookahead + Search Distillation
    search_worker_ratio: float = 0.125
    search_top_k: int = 4
    search_temperature: float = 0.25
    search_distill_coef: float = 0.05

    # Risk-Sensitive Quantile RL & PPO Hyperparameters
    gamma: float = 0.997
    gae_lambda: float = 0.955
    cvar_alpha: float = 0.25
    cvar_weight: float = 0.30
    quantile_huber_kappa: float = 1.0
    lr_actor: float = 2.0e-4
    lr_actor_min: float = 2.5e-5
    lr_critic: float = 4.0e-4
    lr_critic_min: float = 5.0e-5
    no_lr_anneal: bool = False
    clip_param: float = 0.15
    ppo_epochs: int = 4
    num_mini_batches: int = 8
    target_kl: float = 0.018
    value_loss_coef: float = 0.50
    entropy_coef: float = 0.0035
    entropy_min: float = 0.0005
    uniform_kl_coef: float = 0.020
    uniform_kl_min: float = 0.004
    logit_l2_coef: float = 0.0015
    no_entropy_anneal: bool = False

    # Game Rules, Curriculum & PBRS v3
    max_rounds: int = 80
    randomize_first_player: bool = True
    curriculum: bool = True
    curriculum_trade_frac: float = 0.08
    pbrs_anneal: bool = True
    pbrs_min_scale: float = 0.15
    w_income: float = 0.015
    w_house_lock: float = 0.12

    # League & 24/7 Checkpointing
    sparring_prob: float = 0.20
    historical_save_interval: int = 15
    checkpoint_interval: int = 10
    eval_interval: int = 10
    eval_games: int = 12
    gauntlet_max_rounds: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "TrainingConfigV3":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


PRESETS: Dict[str, TrainingConfigV3] = {
    "smoke_test": TrainingConfigV3(
        preset_name="smoke_test",
        description="Quick end-to-end Model-3 verification preset (~5s CPU)",
        total_steps=2_560,
        rollout_steps=40,
        num_envs=16,
        num_workers=2,
        num_threads=4,
        arch="tile_mixer",
        d_model=64,
        hidden_dim=384,
        search_worker_ratio=0.125,
        ppo_epochs=2,
        num_mini_batches=4,
        checkpoint_interval=2,
        historical_save_interval=2,
        eval_interval=2,
        eval_games=4,
        max_rounds=50,
        gauntlet_max_rounds=50,
    ),
    "local_dev": TrainingConfigV3(
        preset_name="local_dev",
        description="Local development 200k steps CPU run",
        total_steps=200_000,
        rollout_steps=128,
        num_envs=32,
        num_workers=4,
        num_threads=4,
        arch="tile_mixer",
        d_model=64,
        hidden_dim=384,
        search_worker_ratio=0.125,
        checkpoint_interval=10,
        eval_interval=5,
        eval_games=8,
    ),
    "server_cpu_standard": TrainingConfigV3(
        preset_name="server_cpu_standard",
        description="24/7 CPU server standard 5M steps (8 cores, 8 workers)",
        total_steps=5_000_000,
        rollout_steps=256,
        num_envs=64,
        num_workers=8,
        num_threads=8,
        arch="tile_mixer",
        d_model=64,
        hidden_dim=384,
        search_worker_ratio=0.125,
        checkpoint_interval=20,
        historical_save_interval=20,
        eval_interval=15,
        eval_games=16,
    ),
    "server_cpu_highcore": TrainingConfigV3(
        preset_name="server_cpu_highcore",
        description="High-core 24/7 CPU server (16 workers, d_model=96, 20M steps)",
        total_steps=20_000_000,
        rollout_steps=256,
        num_envs=128,
        num_workers=16,
        num_threads=12,
        arch="tile_mixer",
        d_model=96,
        hidden_dim=512,
        search_worker_ratio=0.15,
        checkpoint_interval=25,
        historical_save_interval=25,
        eval_interval=20,
        eval_games=20,
    ),
    "server_cpu_24h": TrainingConfigV3(
        preset_name="server_cpu_24h",
        description="Long-horizon 24/7 CPU server marathon (100M steps, auto-checkpoint + PBRS anneal)",
        total_steps=100_000_000,
        rollout_steps=512,
        num_envs=100,
        num_workers=10,
        num_threads=6,
        arch="tile_mixer",
        d_model=64,
        hidden_dim=384,
        critic_hidden_dim=512,
        num_quantiles=32,
        search_worker_ratio=0.0,
        ppo_epochs=2,
        num_mini_batches=4,
        checkpoint_interval=20,
        historical_save_interval=20,
        eval_interval=15,
        eval_games=16,
    ),
    "dragonfly_100m": TrainingConfigV3(
        preset_name="dragonfly_100m",
        description="Model-3 Dragonfly 100M steps server marathon (10 workers, 100 envs, TileMixer d=64, peak ~1750 SPS)",
        total_steps=100_000_000,
        rollout_steps=512,
        num_envs=100,
        num_workers=10,
        num_threads=6,
        arch="tile_mixer",
        d_model=64,
        hidden_dim=384,
        critic_hidden_dim=512,
        num_quantiles=32,
        search_worker_ratio=0.0,
        ppo_epochs=2,
        num_mini_batches=4,
        checkpoint_interval=20,
        historical_save_interval=20,
        eval_interval=15,
        eval_games=16,
    ),
}


def get_preset(name: str) -> TrainingConfigV3:
    if name not in PRESETS:
        raise ValueError(f"Unknown Model-3 preset '{name}'. Available: {list(PRESETS.keys())}")
    # Return a fresh copy so callers can override CLI flags safely
    return TrainingConfigV3(**PRESETS[name].to_dict())
