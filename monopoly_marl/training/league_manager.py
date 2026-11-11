"""League Training Manager for Multi-Agent Monopoly.

Orchestrates Main Agent, Main Exploiter, and Historical Checkpoint Pool.
Computes multi-player Elo ratings, samples league matches, and preserves
population diversity against strategy cycles.
"""

from typing import Dict, List, Optional, Tuple, Any
import os
import copy
import json
import torch
import numpy as np

from monopoly_marl.models.actor_network import ActorNetwork


class LeagueAgentRecord:
    """Represents a member of the tournament league."""

    def __init__(self, name: str, role: str, elo: float = 1200.0, checkpoint_path: Optional[str] = None):
        self.name = name
        self.role = role  # "main", "exploiter", "historical"
        self.elo = elo
        self.checkpoint_path = checkpoint_path
        self.matches_played = 0
        self.wins = 0
        self.bankruptcies = 0
        self.total_net_worth = 0.0

    @property
    def win_rate(self) -> float:
        return (self.wins / self.matches_played) if self.matches_played > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "elo": round(self.elo, 1),
            "matches_played": self.matches_played,
            "wins": self.wins,
            "win_rate": round(self.win_rate, 3),
            "checkpoint_path": self.checkpoint_path,
        }


class LeagueManager:
    """Orchestrates MARL League Training, Checkpointing, and Matchmaking."""

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        k_factor: float = 32.0,
        max_historical_pool: int = 15,
        device: torch.device = torch.device("cpu"),
    ):
        self.checkpoint_dir = checkpoint_dir
        self.k_factor = k_factor
        self.max_historical_pool = max_historical_pool
        self.device = device

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Active participants
        self.main_agent_record = LeagueAgentRecord("MainAgent", "main", elo=1200.0)
        self.exploiter_record = LeagueAgentRecord("MainExploiter", "exploiter", elo=1200.0)
        self.historical_records: List[LeagueAgentRecord] = []

        # Cached policy instances
        self.cached_historical_policies: Dict[str, ActorNetwork] = {}

    def save_checkpoint(
        self,
        policy: ActorNetwork,
        iteration: int,
        role: str = "main",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Saves a policy checkpoint and registers it into historical league pool."""
        filename = f"{role}_iter_{iteration:05d}.pt"
        path = os.path.join(self.checkpoint_dir, filename)

        torch.save({
            "iteration": iteration,
            "role": role,
            "state_dict": policy.state_dict(),
            "metadata": metadata or {},
        }, path)

        if role == "main":
            # Add to historical pool
            hist_name = f"Historical_{iteration:05d}"
            record = LeagueAgentRecord(hist_name, "historical", elo=self.main_agent_record.elo, checkpoint_path=path)
            self.historical_records.append(record)

            # Evict oldest if beyond capacity
            if len(self.historical_records) > self.max_historical_pool:
                removed = self.historical_records.pop(0)
                if removed.name in self.cached_historical_policies:
                    del self.cached_historical_policies[removed.name]

        self.save_league_summary()
        return path

    def sample_match_lineup(
        self,
        main_policy: ActorNetwork,
        exploiter_policy: ActorNetwork,
        mode: str = "league",
    ) -> Tuple[List[ActorNetwork], List[LeagueAgentRecord]]:
        """Samples a 4-agent lineup based on league matchmaking mode:

        - 'self_play': 4 Main Agents
        - 'league': 2 Main Agents + 1 Exploiter + 1 Historical Checkpoint
        - 'exploiter': 1 Exploiter + 3 Main Agents
        """
        lineup_policies: List[ActorNetwork] = []
        lineup_records: List[LeagueAgentRecord] = []

        if mode == "self_play" or not self.historical_records:
            for _ in range(4):
                lineup_policies.append(main_policy)
                lineup_records.append(self.main_agent_record)

        elif mode == "exploiter":
            lineup_policies.append(exploiter_policy)
            lineup_records.append(self.exploiter_record)
            for _ in range(3):
                lineup_policies.append(main_policy)
                lineup_records.append(self.main_agent_record)

        else:  # 'league'
            # 2 Main Agents
            lineup_policies.append(main_policy)
            lineup_records.append(self.main_agent_record)
            lineup_policies.append(main_policy)
            lineup_records.append(self.main_agent_record)

            # 1 Exploiter
            lineup_policies.append(exploiter_policy)
            lineup_records.append(self.exploiter_record)

            # 1 Historical Checkpoint
            hist_rec = np.random.choice(self.historical_records)
            hist_policy = self._get_cached_policy(hist_rec)
            lineup_policies.append(hist_policy)
            lineup_records.append(hist_rec)

        return lineup_policies, lineup_records

    def _get_cached_policy(self, record: LeagueAgentRecord) -> ActorNetwork:
        """Loads and caches a historical checkpoint policy."""
        if record.name not in self.cached_historical_policies:
            policy = ActorNetwork(obs_dim=327).to(self.device)
            ckpt = torch.load(record.checkpoint_path, map_location=self.device)
            policy.load_state_dict(ckpt["state_dict"])
            policy.eval()
            self.cached_historical_policies[record.name] = policy
        return self.cached_historical_policies[record.name]

    def update_match_results(
        self,
        records: List[LeagueAgentRecord],
        ranks: List[int],  # 1 to 4 (1 = winner, 4 = lowest)
        net_worths: List[int],
    ):
        """Updates multi-player Elo ratings and win/match statistics."""
        n = len(records)
        elo_deltas = [0.0] * n

        for i in range(n):
            records[i].matches_played += 1
            records[i].total_net_worth += net_worths[i]
            if ranks[i] == 1:
                records[i].wins += 1

            # Pairwise Elo updates against all other players in match
            for j in range(n):
                if i == j:
                    continue
                # Expected score E_ij
                e_ij = 1.0 / (1.0 + 10.0 ** ((records[j].elo - records[i].elo) / 400.0))
                # Actual score S_ij
                if ranks[i] < ranks[j]:
                    s_ij = 1.0
                elif ranks[i] > ranks[j]:
                    s_ij = 0.0
                else:
                    s_ij = 0.5

                elo_deltas[i] += (self.k_factor / (n - 1)) * (s_ij - e_ij)

        for i in range(n):
            records[i].elo = max(100.0, records[i].elo + elo_deltas[i])

    def save_league_summary(self):
        """Saves league standings and ratings to JSON file."""
        summary_path = os.path.join(self.checkpoint_dir, "league_summary.json")
        data = {
            "main_agent": self.main_agent_record.to_dict(),
            "exploiter": self.exploiter_record.to_dict(),
            "historical_pool": [rec.to_dict() for rec in self.historical_records],
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
