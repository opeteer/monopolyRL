"""Training package for Monopoly MARL."""

from model_2.training.mappo_trainer import MAPPOTrainer
from model_2.training.league_manager import LeagueManager, LeagueAgentRecord

__all__ = ["MAPPOTrainer", "LeagueManager", "LeagueAgentRecord"]
