"""PSRO League Training Manager & 24/7 Checkpoint Orchestrator for Model-3.

Combines:
- MainAgent + MainExploiter + Historical Snapshot Pool (PFSP matchmaking)
- Permanent External Tactical Benchmarks & Adversaries (ProHeuristicBot, OverbidderBot)
- Atomic checkpoint saving (checkpoint_latest.pt + historical milestones) for crash-proof 24/7 runs
- Pairwise composite + stationary MLE Elo rating anchored at Initial/Heuristic baselines
"""

from typing import Dict, List, Optional, Tuple, Any
import os
import json
import torch
import numpy as np

from model_2.training.league_manager import LeagueManager
from model_3.models.actor_network import ActorNetworkV3

SEAT_KIND_RL = 0
SEAT_KIND_PRO_BOT = 1
SEAT_KIND_OVERBIDDER_BOT = 2


class PSROAgentRecord:
    """Represents a Model-3 PSRO league participant (neural or external heuristic bot)."""

    def __init__(
        self,
        name: str,
        role: str,
        elo: float = 1200.0,
        checkpoint_path: Optional[str] = None,
        bot_kind: int = SEAT_KIND_RL,
    ):
        self.name = name
        self.role = role  # "main", "exploiter", "historical", "heuristic_pro", "heuristic_overbidder"
        self.elo = float(elo)
        self.checkpoint_path = checkpoint_path
        self.bot_kind = int(bot_kind)
        self.matches_played = 0
        self.wins = 0
        self.bankruptcies = 0
        self.total_net_worth = 0.0
        # Pairwise record vs MainAgent: [games, score_sum] for PFSP difficulty calibration
        self.vs_main_games = 0
        self.vs_main_loss_sum = 0.0

    @property
    def win_rate(self) -> float:
        return (self.wins / self.matches_played) if self.matches_played > 0 else 0.0

    @property
    def pfsp_weight(self) -> float:
        """Prioritized Fictitious Self-Play difficulty weight favoring opponents Main struggles against."""
        if self.vs_main_games <= 0:
            return 1.0
        avg_main_loss = self.vs_main_loss_sum / float(self.vs_main_games)
        return float(max(0.15, min(1.5, 0.25 + 1.25 * avg_main_loss)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "elo": round(self.elo, 1),
            "matches_played": self.matches_played,
            "wins": self.wins,
            "win_rate": round(self.win_rate, 3),
            "checkpoint_path": self.checkpoint_path,
            "bot_kind": self.bot_kind,
            "vs_main_games": self.vs_main_games,
            "vs_main_loss_sum": round(self.vs_main_loss_sum, 3),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PSROAgentRecord":
        rec = cls(
            name=d["name"],
            role=d["role"],
            elo=d.get("elo", 1200.0),
            checkpoint_path=d.get("checkpoint_path"),
            bot_kind=d.get("bot_kind", SEAT_KIND_RL),
        )
        rec.matches_played = int(d.get("matches_played", 0))
        rec.wins = int(d.get("wins", 0))
        rec.vs_main_games = int(d.get("vs_main_games", 0))
        rec.vs_main_loss_sum = float(d.get("vs_main_loss_sum", 0.0))
        return rec


class PSROLeagueManager:
    """Orchestrates Model-3 PSRO matchmaking, atomic checkpoints, and Elo tracking."""

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints_model_3",
        k_factor: float = 32.0,
        max_historical_pool: int = 16,
        anchor_elo: float = 1200.0,
        win_weight: float = 0.70,
        freeze_historical_elo: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        self.checkpoint_dir = checkpoint_dir
        self.k_factor = float(k_factor)
        self.max_historical_pool = int(max_historical_pool)
        self.anchor_elo = float(anchor_elo)
        self.win_weight = float(win_weight)
        self.freeze_historical_elo = bool(freeze_historical_elo)
        self.device = device

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.main_record = PSROAgentRecord("MainAgent", "main", elo=self.anchor_elo)
        self.exploiter_record = PSROAgentRecord("MainExploiter", "exploiter", elo=self.anchor_elo)
        self.pro_bot_record = PSROAgentRecord(
            "ProHeuristicBot", "heuristic_pro", elo=self.anchor_elo, bot_kind=SEAT_KIND_PRO_BOT
        )
        self.overbidder_bot_record = PSROAgentRecord(
            "OverbidderBot", "heuristic_overbidder", elo=self.anchor_elo, bot_kind=SEAT_KIND_OVERBIDDER_BOT
        )
        self.historical_records: List[PSROAgentRecord] = []
        self.cached_historical_policies: Dict[str, ActorNetworkV3] = {}
        self.pinned_anchor_name: Optional[str] = "ProHeuristicBot"
        self.gauntlet_matches: List[Tuple[List[str], List[int]]] = []

        self._load_summary_if_exists()

    def _summary_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "league_summary.json")

    def _load_summary_if_exists(self):
        path = self._summary_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "main_agent" in data:
                self.main_record = PSROAgentRecord.from_dict(data["main_agent"])
            if "exploiter" in data:
                self.exploiter_record = PSROAgentRecord.from_dict(data["exploiter"])
            if "pro_bot" in data:
                self.pro_bot_record = PSROAgentRecord.from_dict(data["pro_bot"])
            if "overbidder_bot" in data:
                self.overbidder_bot_record = PSROAgentRecord.from_dict(data["overbidder_bot"])
            if "historical_pool" in data:
                self.historical_records = []
                for hd in data["historical_pool"]:
                    cp = hd.get("checkpoint_path")
                    if cp and os.path.exists(cp):
                        self.historical_records.append(PSROAgentRecord.from_dict(hd))
        except Exception as e:
            print(f"[PSROLeague] Failed to load summary: {e}")

    def _save_summary(self):
        summary = {
            "main_agent": self.main_record.to_dict(),
            "exploiter": self.exploiter_record.to_dict(),
            "pro_bot": self.pro_bot_record.to_dict(),
            "overbidder_bot": self.overbidder_bot_record.to_dict(),
            "historical_pool": [r.to_dict() for r in self.historical_records],
        }
        target = self._summary_path()
        tmp = target + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        os.replace(tmp, target)

    def load_historical_policy(self, record: PSROAgentRecord) -> ActorNetworkV3:
        if record.name not in self.cached_historical_policies:
            ckpt = torch.load(record.checkpoint_path, map_location=self.device, weights_only=False)
            cfg = ckpt.get("model_config", {})
            policy = ActorNetworkV3(
                d_model=int(cfg.get("d_model", 64)),
                hidden_dim=int(cfg.get("hidden_dim", 384)),
                arch=str(cfg.get("arch", "tile_mixer")),
            ).to(self.device)
            policy.load_state_dict(ckpt["state_dict"], strict=True)
            policy.eval()
            self.cached_historical_policies[record.name] = policy
        return self.cached_historical_policies[record.name]

    def save_historical_snapshot(self, actor: ActorNetworkV3, name: str, iteration: int) -> str:
        path = os.path.join(self.checkpoint_dir, f"{name}.pt")
        tmp = path + ".tmp"
        payload = {
            "name": name,
            "iteration": int(iteration),
            "elo": float(self.main_record.elo),
            "model_config": {
                "arch": actor.arch,
                "d_model": actor.d_model,
                "hidden_dim": actor.hidden_dim,
            },
            "state_dict": {k: v.detach().cpu() for k, v in actor.state_dict().items()},
        }
        torch.save(payload, tmp)
        os.replace(tmp, path)

        existing = next((r for r in self.historical_records if r.name == name), None)
        if existing is None:
            rec = PSROAgentRecord(
                name=name,
                role="historical",
                elo=self.main_record.elo,
                checkpoint_path=path,
                bot_kind=SEAT_KIND_RL,
            )
            self.historical_records.append(rec)
            if len(self.historical_records) > self.max_historical_pool:
                # Preserve earliest milestone + Pro/anchor coverage; evict second-oldest
                evict_idx = 1 if len(self.historical_records) > 1 else 0
                removed = self.historical_records.pop(evict_idx)
                self.cached_historical_policies.pop(removed.name, None)
        else:
            existing.elo = self.main_record.elo
            self.cached_historical_policies.pop(name, None)

        self._save_summary()
        return path

    def save_training_checkpoint(
        self,
        path: str,
        iteration: int,
        total_steps: int,
        actor: ActorNetworkV3,
        critic: torch.nn.Module,
        exploiter_actor: ActorNetworkV3,
        exploiter_optimizer: torch.optim.Optimizer,
        trainer_state: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        payload = {
            "iteration": int(iteration),
            "total_steps": int(total_steps),
            "model_config": {
                "arch": actor.arch,
                "d_model": actor.d_model,
                "hidden_dim": actor.hidden_dim,
                "critic_hidden_dim": getattr(critic, "hidden_dim", 512),
                "num_quantiles": getattr(critic, "num_quantiles", 32),
            },
            "actor_state_dict": {k: v.detach().cpu() for k, v in actor.state_dict().items()},
            "critic_state_dict": {k: v.detach().cpu() for k, v in critic.state_dict().items()},
            "exploiter_state_dict": {k: v.detach().cpu() for k, v in exploiter_actor.state_dict().items()},
            "exploiter_optimizer": exploiter_optimizer.state_dict(),
            "trainer_state": trainer_state,
            "extra": dict(extra or {}),
        }
        torch.save(payload, tmp)
        os.replace(tmp, path)
        self._save_summary()

    @staticmethod
    def load_training_checkpoint(path: str, map_location: torch.device = torch.device("cpu")) -> Dict[str, Any]:
        return torch.load(path, map_location=map_location, weights_only=False)

    def _pfsp_sample_opponent(self, rng: np.random.Generator) -> PSROAgentRecord:
        pool: List[PSROAgentRecord] = list(self.historical_records) + [
            self.pro_bot_record,
            self.overbidder_bot_record,
        ]
        weights = np.array([r.pfsp_weight for r in pool], dtype=np.float64)
        probs = weights / weights.sum()
        idx = int(rng.choice(len(pool), p=probs))
        return pool[idx]

    def sample_matchups(
        self,
        num_envs: int,
        search_env_set: Optional[set] = None,
        rng: Optional[np.random.Generator] = None,
        sparring_prob: float = 0.20,
    ) -> Tuple[List[List[PSROAgentRecord]], np.ndarray]:
        """Assigns 4 seats per env and returns (seat_records_per_env, bot_kind_matrix [E, 4]).

        Search envs always run 4-way MainAgent (cleanest self-play trajectories for lookahead).
        """
        if rng is None:
            rng = np.random.default_rng()
        s_set = search_env_set or set()
        assignments: List[List[PSROAgentRecord]] = []
        bot_kinds = np.zeros((num_envs, 4), dtype=np.int8)

        for e in range(num_envs):
            if e in s_set:
                seats = [self.main_record] * 4
                assignments.append(seats)
                continue

            u = float(rng.random())
            if u < 0.35:
                # Self-play mirror
                seats = [self.main_record] * 4
            elif u < 0.35 + sparring_prob:
                # Tactical Sparring vs ProHeuristicBot + OverbidderBot
                seats = [
                    self.main_record,
                    self.pro_bot_record,
                    self.pro_bot_record if rng.random() < 0.5 else self.main_record,
                    self.overbidder_bot_record,
                ]
            elif u < 0.72 and self.historical_records:
                # PFSP Historical + Bot Mix
                seats = [
                    self.main_record,
                    self._pfsp_sample_opponent(rng),
                    self._pfsp_sample_opponent(rng),
                    self._pfsp_sample_opponent(rng),
                ]
            elif u < 0.87:
                # Exploiter Match: 1 MainExploiter vs 3 Main (or PFSP)
                seats = [
                    self.exploiter_record,
                    self.main_record,
                    self.main_record,
                    self.main_record if not self.historical_records else self._pfsp_sample_opponent(rng),
                ]
            else:
                # Main vs Mixed League & Exploiters
                seats = [
                    self.main_record,
                    self.exploiter_record,
                    self._pfsp_sample_opponent(rng),
                    self.pro_bot_record if rng.random() < 0.5 else self._pfsp_sample_opponent(rng),
                ]

            perm = rng.permutation(4)
            shuffled = [seats[int(i)] for i in perm]
            assignments.append(shuffled)
            for p_idx, rec in enumerate(shuffled):
                bot_kinds[e, p_idx] = rec.bot_kind

        return assignments, bot_kinds

    def record_match_result(
        self,
        seats: List[PSROAgentRecord],
        ranks: List[int],
        net_worths: List[int],
        record_for_mle: bool = False,
    ):
        """Updates pairwise Elos and PFSP loss statistics."""
        n = len(seats)
        if n <= 1:
            return
        elo_deltas = [0.0] * n

        # Track Main's vs-opponent outcomes for PFSP weighting
        for i in range(n):
            rec = seats[i]
            rec.matches_played += 1
            rec.total_net_worth += net_worths[i]
            if ranks[i] == 1:
                rec.wins += 1

            if rec.role == "main":
                for j in range(n):
                    if j != i and seats[j] is not rec:
                        s_ij = LeagueManager.compute_composite_score(ranks[i], ranks[j], win_weight=self.win_weight)
                        seats[j].vs_main_games += 1
                        seats[j].vs_main_loss_sum += (1.0 - s_ij)

            if self.freeze_historical_elo and rec.role in ("historical", "heuristic_pro", "heuristic_overbidder"):
                continue

            distinct_opps = [j for j in range(n) if j != i and seats[j] is not rec]
            if not distinct_opps:
                continue
            norm = max(1, n - 1)
            for j in distinct_opps:
                e_ij = 1.0 / (1.0 + 10.0 ** ((seats[j].elo - rec.elo) / 400.0))
                s_ij = LeagueManager.compute_composite_score(ranks[i], ranks[j], win_weight=self.win_weight)
                elo_deltas[i] += (self.k_factor / norm) * (s_ij - e_ij)

        for i in range(n):
            if self.freeze_historical_elo and seats[i].role in ("historical", "heuristic_pro", "heuristic_overbidder"):
                continue
            seats[i].elo = max(100.0, seats[i].elo + elo_deltas[i])

        if record_for_mle:
            self.gauntlet_matches.append(([s.name for s in seats], list(ranks)))
            if len(self.gauntlet_matches) > 600:
                self.gauntlet_matches = self.gauntlet_matches[-600:]

    def calibrate_mle_elos(self) -> Dict[str, float]:
        if not self.gauntlet_matches:
            return {}
        ratings = LeagueManager.compute_mle_elos(
            self.gauntlet_matches,
            anchor_name=self.pinned_anchor_name or "ProHeuristicBot",
            anchor_elo=self.anchor_elo,
            win_weight=self.win_weight,
        )
        for rec in [self.main_record, self.exploiter_record, self.pro_bot_record, self.overbidder_bot_record] + self.historical_records:
            if rec.name in ratings:
                if self.freeze_historical_elo and rec.role in ("historical", "heuristic_pro", "heuristic_overbidder") and rec.name == self.pinned_anchor_name:
                    continue
                rec.elo = float(ratings[rec.name])
        self._save_summary()
        return ratings
