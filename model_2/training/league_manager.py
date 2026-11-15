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

from model_2.models.actor_network import ActorNetwork


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
        checkpoint_dir: str = "checkpoints_model_2",
        k_factor: float = 32.0,
        max_historical_pool: int = 15,
        anchor_elo: float = 1200.0,
        win_weight: float = 0.70,
        freeze_historical_elo: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        self.checkpoint_dir = checkpoint_dir
        self.k_factor = k_factor
        self.max_historical_pool = max_historical_pool
        self.anchor_elo = float(anchor_elo)
        self.win_weight = float(win_weight)
        self.freeze_historical_elo = bool(freeze_historical_elo)
        self.device = device

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Active participants
        self.main_agent_record = LeagueAgentRecord("MainAgent", "main", elo=self.anchor_elo)
        self.exploiter_record = LeagueAgentRecord("MainExploiter", "exploiter", elo=self.anchor_elo)
        self.historical_records: List[LeagueAgentRecord] = []
        self.pinned_anchor_name: Optional[str] = None

        # Load existing league state if present
        summary_path = os.path.join(checkpoint_dir, "league_summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "main_agent" in data:
                    self.main_agent_record.elo = data["main_agent"].get("elo", 1200.0)
                    self.main_agent_record.matches_played = data["main_agent"].get("matches_played", 0)
                    self.main_agent_record.wins = data["main_agent"].get("wins", 0)
                if "exploiter" in data:
                    self.exploiter_record.elo = data["exploiter"].get("elo", 1200.0)
                    self.exploiter_record.matches_played = data["exploiter"].get("matches_played", 0)
                    self.exploiter_record.wins = data["exploiter"].get("wins", 0)
                if "historical_pool" in data:
                    for h_data in data["historical_pool"]:
                        ckpt_path = h_data.get("checkpoint_path")
                        if ckpt_path and os.path.exists(ckpt_path):
                            rec = LeagueAgentRecord(
                                name=h_data.get("name", "Historical"),
                                role="historical",
                                elo=h_data.get("elo", 1200.0),
                                checkpoint_path=ckpt_path,
                            )
                            rec.matches_played = h_data.get("matches_played", 0)
                            rec.wins = h_data.get("wins", 0)
                            self.historical_records.append(rec)
                    if len(self.historical_records) > self.max_historical_pool:
                        self.historical_records = self.historical_records[-self.max_historical_pool:]
            except Exception as e:
                print(f"[LeagueManager] Notice: failed to load existing summary ({e}); starting fresh league state.")

        # Cached policy instances
        self.cached_historical_policies: Dict[str, ActorNetwork] = {}

    def reset_for_finetune(
        self,
        anchor_policy: ActorNetwork,
        critic: Optional[torch.nn.Module] = None,
        exploiter: Optional[ActorNetwork] = None,
    ) -> str:
        """Resets league standings for fine-tuning and pins the pre-fine-tune model as Historical_00000."""
        for rec in (self.main_agent_record, self.exploiter_record):
            rec.elo = self.anchor_elo
            rec.matches_played = 0
            rec.wins = 0
            rec.bankruptcies = 0
            rec.total_net_worth = 0.0
        self.historical_records.clear()
        self.cached_historical_policies.clear()
        self.pinned_anchor_name = "Historical_00000"
        return self.save_checkpoint(
            anchor_policy,
            iteration=0,
            role="main",
            metadata={"fine_tune_anchor": True, "iteration": 0},
            critic=critic,
            exploiter=exploiter,
        )

    def save_checkpoint(
        self,
        policy: ActorNetwork,
        iteration: int,
        role: str = "main",
        metadata: Optional[Dict[str, Any]] = None,
        critic: Optional[torch.nn.Module] = None,
        exploiter: Optional[ActorNetwork] = None,
    ) -> str:
        """Saves a policy checkpoint (and optional critic/exploiter states) and updates historical league pool."""
        filename = f"{role}_iter_{iteration:05d}.pt"
        path = os.path.join(self.checkpoint_dir, filename)

        payload: Dict[str, Any] = {
            "iteration": iteration,
            "role": role,
            "state_dict": policy.state_dict(),
            "metadata": metadata or {},
        }
        if critic is not None:
            payload["critic_state_dict"] = critic.state_dict()
        if exploiter is not None:
            payload["exploiter_state_dict"] = exploiter.state_dict()

        torch.save(payload, path)

        if role == "main":
            # Add to historical pool
            hist_name = f"Historical_{iteration:05d}"
            record = LeagueAgentRecord(hist_name, "historical", elo=self.main_agent_record.elo, checkpoint_path=path)
            self.historical_records.append(record)

            # Evict oldest non-pinned snapshot if beyond capacity
            if len(self.historical_records) > self.max_historical_pool:
                evict_idx = 1 if (
                    self.pinned_anchor_name
                    and len(self.historical_records) > 1
                    and self.historical_records[0].name == self.pinned_anchor_name
                ) else 0
                removed = self.historical_records.pop(evict_idx)
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

            # 1 Historical Checkpoint (PFSP Elo-weighted prioritization)
            hist_rec = self._select_historical_record(pfsp=True)
            hist_policy = self._get_cached_policy(hist_rec)
            lineup_policies.append(hist_policy)
            lineup_records.append(hist_rec)

        return lineup_policies, lineup_records

    def _select_historical_record(self, pfsp: bool = True) -> LeagueAgentRecord:
        """Selects a historical checkpoint using Prioritized Fictitious Self-Play (PFSP) Elo weights."""
        if len(self.historical_records) == 1 or not pfsp:
            return np.random.choice(self.historical_records)
        main_elo = float(self.main_agent_record.elo)
        elos = np.array([float(rec.elo) for rec in self.historical_records], dtype=np.float64)
        diffs = np.clip((elos - main_elo) / 200.0, -10.0, 10.0)
        expected_opp_score = 1.0 / (1.0 + np.exp(-diffs))
        weights = (expected_opp_score ** 1.5) + 0.10
        probs = weights / weights.sum()
        idx = int(np.random.choice(len(self.historical_records), p=probs))
        return self.historical_records[idx]

    def get_random_historical_policy(self, pfsp: bool = True) -> Optional[ActorNetwork]:
        """Returns a cached policy sampled from the historical pool (PFSP-prioritized by default)."""
        if not self.historical_records:
            return None
        rec = self._select_historical_record(pfsp=pfsp)
        return self._get_cached_policy(rec)

    def _get_cached_policy(self, record: LeagueAgentRecord) -> ActorNetwork:
        """Loads and caches a historical checkpoint policy with seamless head adaptation."""
        if record.name not in self.cached_historical_policies:
            ckpt = torch.load(record.checkpoint_path, map_location=self.device)
            state_dict = ckpt["state_dict"]
            ckpt_obs_dim = state_dict["trunk.0.weight"].shape[1]
            ckpt_hidden_dim = state_dict["trunk.0.weight"].shape[0]
            policy = ActorNetwork(obs_dim=ckpt_obs_dim, hidden_dim=ckpt_hidden_dim).to(self.device)
            new_sd = policy.state_dict()
            old_auc_dim = state_dict["head_auction.weight"].shape[0] if "head_auction.weight" in state_dict else 5
            curr_auc_dim = new_sd["head_auction.weight"].shape[0]

            for k in state_dict:
                if k == "head_auction.weight" and old_auc_dim != curr_auc_dim:
                    new_sd[k][:old_auc_dim, :] = state_dict[k]
                    new_sd[k][old_auc_dim:, :] = 0.0
                elif k == "head_auction.bias" and old_auc_dim != curr_auc_dim:
                    new_sd[k][:old_auc_dim] = state_dict[k]
                    new_sd[k][old_auc_dim:] = -0.1
                elif k in new_sd and new_sd[k].shape == state_dict[k].shape:
                    new_sd[k] = state_dict[k]

            policy.load_state_dict(new_sd)
            policy.eval()
            self.cached_historical_policies[record.name] = policy
        return self.cached_historical_policies[record.name]

    def sync_exploiter_elo(self):
        """Aligns MainExploiter Elo with MainAgent Elo when exploiter weights are reset to main_actor."""
        self.exploiter_record.elo = float(self.main_agent_record.elo)

    @staticmethod
    def compute_composite_score(rank_i: int, rank_j: int, win_weight: float = 0.70) -> float:
        """Returns Monopoly-calibrated pairwise outcome score S_ij in [0, 1] (symmetric: S_ij + S_ji = 1).

        Weights sole-survivor / 1st-place victory primarily (win_weight), while preserving
        smooth placement signal among eliminated seats (1 - win_weight).
        """
        if rank_i == 1 and rank_j != 1:
            win_share = 1.0
        elif rank_j == 1 and rank_i != 1:
            win_share = 0.0
        else:
            win_share = 0.5

        if rank_i < rank_j:
            place_share = 1.0
        elif rank_i > rank_j:
            place_share = 0.0
        else:
            place_share = 0.5

        return win_weight * win_share + (1.0 - win_weight) * place_share

    def update_match_results(
        self,
        records: List[LeagueAgentRecord],
        ranks: List[int],  # 1 to 4 (1 = winner, 4 = lowest)
        net_worths: List[int],
    ):
        """Updates multi-player Elo ratings without self-comparison noise or historical eviction drain."""
        n = len(records)
        if n <= 1:
            return
        elo_deltas = [0.0] * n

        for i in range(n):
            records[i].matches_played += 1
            records[i].total_net_worth += net_worths[i]
            if ranks[i] == 1:
                records[i].wins += 1

            # Skip updating frozen historical milestone snapshots (prevents eviction Elo drain)
            if self.freeze_historical_elo and records[i].role == "historical":
                continue

            # Pairwise Elo updates strictly against distinct opponent policy instances
            distinct_opps = [j for j in range(n) if j != i and records[j] is not records[i]]
            if not distinct_opps:
                continue

            norm = max(1, n - 1)
            for j in distinct_opps:
                e_ij = 1.0 / (1.0 + 10.0 ** ((records[j].elo - records[i].elo) / 400.0))
                s_ij = self.compute_composite_score(ranks[i], ranks[j], win_weight=self.win_weight)
                elo_deltas[i] += (self.k_factor / norm) * (s_ij - e_ij)

        for i in range(n):
            if self.freeze_historical_elo and records[i].role == "historical":
                continue
            records[i].elo = max(100.0, records[i].elo + elo_deltas[i])

    @classmethod
    def compute_mle_elos(
        cls,
        matches: List[Tuple[List[str], List[int]]],
        anchor_name: str,
        anchor_elo: float = 1200.0,
        win_weight: float = 0.70,
        max_iters: int = 400,
    ) -> Dict[str, float]:
        """Solves stationary Maximum-Likelihood Bradley-Terry Elo ratings across multi-player matches.

        Each match is (seat_model_names, seat_ranks). Uses exact pairwise composite outcomes
        across distinct opponents and anchors `anchor_name` to `anchor_elo`.
        """
        models = sorted({m for names, _ in matches for m in names})
        if not models:
            return {}
        ratings = {m: float(anchor_elo) for m in models}
        if anchor_name not in ratings:
            anchor_name = models[0]

        # Accumulate pairwise observed scores and match counts across distinct-model pairs
        pair_scores: Dict[Tuple[str, str], float] = {}
        pair_counts: Dict[Tuple[str, str], int] = {}
        for names, ranks in matches:
            n = len(names)
            for i in range(n):
                for j in range(n):
                    if i == j or names[i] == names[j]:
                        continue
                    key = (names[i], names[j])
                    pair_scores[key] = pair_scores.get(key, 0.0) + cls.compute_composite_score(
                        ranks[i], ranks[j], win_weight=win_weight
                    )
                    pair_counts[key] = pair_counts.get(key, 0) + 1

        # Iterative MM / Newton-style Bradley-Terry Elo convergence
        ln10_400 = float(np.log(10.0) / 400.0)
        for _ in range(max_iters):
            max_step = 0.0
            for m in models:
                if m == anchor_name:
                    continue
                grad = 0.0
                hess = 0.0
                for (mi, mj), cnt in pair_counts.items():
                    if mi != m:
                        continue
                    e_ij = 1.0 / (1.0 + 10.0 ** ((ratings[mj] - ratings[mi]) / 400.0))
                    obs_s = pair_scores[(mi, mj)]
                    grad += (obs_s - cnt * e_ij) * ln10_400
                    hess += cnt * e_ij * (1.0 - e_ij) * (ln10_400 ** 2)
                if hess > 1e-12:
                    step = float(np.clip(grad / hess, -100.0, 100.0))
                    ratings[m] += step
                    max_step = max(max_step, abs(step))
            # Keep anchor pinned cleanly
            shift = anchor_elo - ratings[anchor_name]
            for m in models:
                ratings[m] += shift
            if max_step < 1e-5:
                break

        return {m: round(r, 1) for m, r in ratings.items()}

    @staticmethod
    def compute_plackett_luce_elos(
        matches: List[Tuple[List[str], List[int]]],
        anchor_name: str,
        anchor_elo: float = 1200.0,
        prior_pseudo_wins: float = 0.25,
        max_iters: int = 600,
    ) -> Dict[str, float]:
        """Solves stationary Plackett-Luce (Luce Choice) Multiplayer Win-Likelihood Elo ratings.

        Models P(seat i wins) = exp(r_i) / sum_k exp(r_k), avoiding bankrupt-tie compression
        when 3 eliminated players finish with 0 Net Worth in Sole-Survivor Monopoly.
        Uses a small Bayesian pseudo-win prior to keep zero-win baselines bounded.
        """
        import math

        models = sorted({m for names, _ in matches for m in names})
        if not models:
            return {}
        if anchor_name not in models:
            anchor_name = models[0]

        r = {m: 0.0 for m in models}
        for _ in range(max_iters):
            mean_exp = sum(math.exp(r[k]) for k in models) / len(models)
            wins_cnt = {m: float(prior_pseudo_wins) for m in models}
            denom = {m: float(prior_pseudo_wins) * (math.exp(r[m]) / max(1e-12, mean_exp)) for m in models}

            for names, ranks in matches:
                winners = [i for i, rk in enumerate(ranks) if rk == 1]
                if not winners:
                    continue
                for i in winners:
                    wins_cnt[names[i]] += 1.0 / len(winners)
                exp_sum = sum(math.exp(r[s]) for s in names)
                for s in names:
                    denom[s] += math.exp(r[s]) / max(1e-12, exp_sum)

            max_step = 0.0
            for m in models:
                step = 0.5 * math.log(max(1e-12, wins_cnt[m]) / max(1e-12, denom[m]))
                r[m] += step
                max_step = max(max_step, abs(step))

            shift = r[anchor_name]
            for m in models:
                r[m] -= shift
            if max_step < 1e-6:
                break

        scale = 400.0 / math.log(10.0)
        return {m: round(float(anchor_elo) + r[m] * scale, 1) for m in models}

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
