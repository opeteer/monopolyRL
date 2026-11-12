"""Training Configuration Presets and Template Manager for Monopoly RL."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import json


@dataclass
class TrainingConfig:
    preset_name: str = "custom"
    description: str = "Custom training configuration"
    total_steps: int = 1_000_000
    rollout_steps: int = 128
    num_envs: int = 80
    num_workers: int = 10
    num_threads: int = 6
    checkpoint_interval: int = 25
    eval_interval: int = 10
    entropy_coef: float = 0.01
    entropy_min: float = 0.001
    no_entropy_anneal: bool = False
    league_rollout_fraction: float = 0.25
    curriculum: bool = True
    curriculum_phase_2_frac: float = 0.15
    gamma: float = 0.997
    gae_lambda: float = 0.97
    lr_actor: float = 3e-4
    lr_critic: float = 5e-4
    clip_param: float = 0.2
    ppo_epochs: int = 4
    num_mini_batches: int = 4
    w1: float = 0.050
    w2: float = 0.005
    w3: float = 0.008
    w4: float = 0.0005
    w_syn: float = 0.020
    beta_starve: float = 3.5
    w_house: float = 0.015
    use_shm: bool = True
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "TrainingConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


PRESETS: Dict[str, TrainingConfig] = {
    "smoke_test": TrainingConfig(
        preset_name="smoke_test",
        description="Quick 10k steps smoke test to verify training pipeline integrity (<3 seconds)",
        total_steps=10_000,
        rollout_steps=64,
        num_envs=32,
        num_workers=4,
        num_threads=4,
        checkpoint_interval=2,
        eval_interval=2,
        curriculum_phase_2_frac=0.10,
    ),
    "fast_dev": TrainingConfig(
        preset_name="fast_dev",
        description="Fast 250k steps prototyping session (~30-45 seconds)",
        total_steps=250_000,
        rollout_steps=128,
        num_envs=60,
        num_workers=10,
        num_threads=6,
        checkpoint_interval=10,
        eval_interval=5,
        curriculum_phase_2_frac=0.15,
    ),
    "standard_1m": TrainingConfig(
        preset_name="standard_1m",
        description="Standard 1M steps training session (~2-3 minutes)",
        total_steps=1_000_000,
        rollout_steps=128,
        num_envs=80,
        num_workers=10,
        num_threads=6,
        checkpoint_interval=25,
        eval_interval=10,
        curriculum_phase_2_frac=0.15,
    ),
    "tournament_5m": TrainingConfig(
        preset_name="tournament_5m",
        description="Full 5M steps tournament competitive training (~10-15 minutes)",
        total_steps=5_000_000,
        rollout_steps=128,
        num_envs=80,
        num_workers=10,
        num_threads=6,
        checkpoint_interval=25,
        eval_interval=10,
        curriculum_phase_2_frac=0.15,
    ),
    "overnight_10m": TrainingConfig(
        preset_name="overnight_10m",
        description="Deep 10M steps tournament master league training (~25-35 minutes)",
        total_steps=10_000_000,
        rollout_steps=256,
        num_envs=80,
        num_workers=10,
        num_threads=6,
        checkpoint_interval=25,
        eval_interval=10,
        curriculum_phase_2_frac=0.15,
    ),
    "deep_league_25m": TrainingConfig(
        preset_name="deep_league_25m",
        description="Extended 25M steps Grandmaster training session (~1-1.5 hours)",
        total_steps=25_000_000,
        rollout_steps=256,
        num_envs=80,
        num_workers=10,
        num_threads=6,
        checkpoint_interval=50,
        eval_interval=20,
        curriculum_phase_2_frac=0.12,
    ),
    "grandmaster_50m": TrainingConfig(
        preset_name="grandmaster_50m",
        description="Ultra-deep 50M steps Grandmaster league tournament session (~2.5 hours)",
        total_steps=50_000_000,
        rollout_steps=256,
        num_envs=80,
        num_workers=10,
        num_threads=6,
        checkpoint_interval=50,
        eval_interval=25,
        curriculum_phase_2_frac=0.10,
    ),
}


def get_preset(name: str) -> Optional[TrainingConfig]:
    """Retrieves a configuration preset by name."""
    return PRESETS.get(name)


def list_presets() -> Dict[str, str]:
    """Returns mapping of preset name to description."""
    return {name: cfg.description for name, cfg in PRESETS.items()}
