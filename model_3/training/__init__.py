"""Model-3 CTDE MAPPO + Quantile Huber + Search Distillation Trainer and PSRO League."""

from model_3.training.mappo_trainer import MAPPOTrainerV3
from model_3.training.psro_league import PSROLeagueManager, PSROAgentRecord

__all__ = [
    "MAPPOTrainerV3",
    "PSROLeagueManager",
    "PSROAgentRecord",
]
