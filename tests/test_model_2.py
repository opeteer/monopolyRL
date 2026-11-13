"""Unit & Integration Tests for Model-2 MARL Monopoly Architecture & Algorithms.

Validates that Model-2 uses the EXACT universal guard masks (100% parity with Model-1)
while addressing the 4 weaknesses purely via RL/PBRS, reward shaping, architecture,
and MAPPO hyperparameters:
1. Eliminates $500 singleton trade arbitrage (PBRS w_cash == w2, removes forced $500 snap,
   applies -0.08/-0.04 churn shaping on non-advancing trades).
2. Prevents nominal-price bias & leader blunders (PBRS Term 7 opponent monopoly threat,
   Term 8 net-worth lead preservation, one-sided anti-kingmaking penalties -1.25 / -1.75).
3. Discourages pointless 10% mortgage/unmortgage churn via PBRS 1.25x mortgage valuation
   (negative potential delta on unnecessary mortgage, positive on unmortgage) while keeping
   universal same-turn & liquidity masks identical to Model-1.
4. Reduces stochastic decision noise (Residual Actor hidden_dim=384 with SiLU+LayerNorm,
   trade_resp_branch, temperature scaling, AdamW, target_kl early stopping, cosine LR).
"""

import torch
import numpy as np

from model_2.engine.monopoly_core import (
    MonopolyCore,
    PHASE_PROPERTY_MANAGE,
    PHASE_TRADE_RESPONSE,
    ACT_CORE_END_MANAGE,
    TRADE_CASH_BRACKETS,
)
from model_2.engine.pettingzoo_wrapper import MonopolyParallelEnv
from model_2.models.actor_network import ActorNetwork
from model_2.models.critic_network import CentralizedCritic
from model_2.models.storage import RolloutBuffer
from model_2.training.mappo_trainer import MAPPOTrainer
from model_2.training.league_manager import LeagueManager
from model_2.system.config_presets import get_preset, TrainingConfig
from monopoly_marl.engine.monopoly_core import MonopolyCore as Model1Core
from monopoly_marl.models.actor_network import ActorNetwork as Model1ActorNetwork


def test_model_2_universal_guard_mask_parity_with_model_1():
    """Verifies Model-2 preserves 100% universal action masks matching Model-1."""
    c1 = Model1Core(seed=7)
    c2 = MonopolyCore(seed=7)

    for c in (c1, c2):
        c.phase = PHASE_PROPERTY_MANAGE
        c.active_decision_player = 0
        c.current_player = 0
        c.allow_trade = True
        c.cash = [5000, 1500, 1500, 1500]
        # Give P0 Brown monopoly + Reading RR, P1 gets Penn RR
        c.property_owner[1] = 0
        c.property_owner[3] = 0
        c.property_owner[5] = 0
        c.property_owner[15] = 1

    m1 = c1.get_action_masks(0)
    m2 = c2.get_action_masks(0)
    for head_key in m1:
        assert np.array_equal(m1[head_key], m2[head_key]), f"Mask mismatch on head '{head_key}'"

    # Confirm universal cooldown attrs do NOT include Model-2-only mask locks
    assert not hasattr(c2, "accepted_prop_hold_cooldown")
    assert not hasattr(c2, "prop_mortgage_cooldown")


def test_model_2_pbrs_mortgage_burn_and_anti_kingmaking_terms():
    """Verifies PBRS discourages flush mortgaging, kills $500 arbitrage, and activates Term 7 & 8."""
    core = MonopolyCore(seed=42)
    assert abs(core.w_cash - core.w2) < 1e-9
    assert core.w_opp_mono == 0.90
    assert core.w_lead == 0.60

    # Mortgaging Penn RR ($100 mortgage) when already holding $5,000 must drop PBRS potential
    core.property_owner[15] = 0
    core.cash = [5000, 1500, 1500, 1500]
    phi_before_mtg = core._compute_potential(0)
    core.property_mortgaged[15] = True
    core.cash[0] += 100
    phi_after_mtg = core._compute_potential(0)
    assert phi_after_mtg < phi_before_mtg

    # Unmortgaging when liquid ($5,100 -> $4,990) raises PBRS potential (positive incentive to redeem)
    phi_after_unmtg_before_net = phi_after_mtg
    core.property_mortgaged[15] = False
    core.cash[0] -= 110
    phi_after_unmtg = core._compute_potential(0)
    assert phi_after_unmtg > phi_after_unmtg_before_net
    # Round-trip mortgage -> unmortgage permanently lowers potential due to 10% interest burn
    assert phi_after_unmtg < phi_before_mtg

    # Term 8: net-worth lead preservation increases potential for leader
    core.cash = [2500, 1500, 1500, 1500]
    phi_lead_p0 = core._compute_potential(0)
    core.cash = [1500, 1500, 1500, 1500]
    phi_even_p0 = core._compute_potential(0)
    assert phi_lead_p0 > phi_even_p0

    # Term 7: rival completing a street monopoly penalizes other players' potential
    core.property_owner[16] = 1  # St. James
    core.property_owner[18] = 1  # Tennessee
    phi_before_opp_mono = core._compute_potential(0)
    core.property_owner[19] = 1  # New York -> P1 completes Orange monopoly
    phi_after_opp_mono = core._compute_potential(0)
    assert phi_after_opp_mono < phi_before_opp_mono - 0.5


def test_model_2_trade_no_forced_500_snap_and_churn_penalty():
    """Verifies minimum-mortgage cash snap (no $500 inflation), churn reward penalty, and universal 8-round pair cooldown."""
    core = MonopolyCore(seed=11)
    core.allow_trade = True
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0
    core.current_player = 0

    # P1 owns Reading RR (tile 5, mortgage $100), P0 has plenty of cash ($1500)
    core.property_owner[5] = 1
    core.cash = [1500, 1500, 1500, 1500]

    # P0 requests Reading RR with pure cash bracket 0 ($0): snaps to minimum legal $100, NOT forced $500
    propose_act = {
        "core": ACT_CORE_END_MANAGE,
        "auction": 0,
        "property": 160,
        "trade_target": 0,
        "trade_offer_prop": 40,
        "trade_offer_cash": 0,
        "trade_req_prop": 5,
        "trade_req_cash": 0,
        "trade_response": 0,
    }
    core.step({"player_0": propose_act})
    assert core.phase == PHASE_TRADE_RESPONSE
    assert core.pending_trade is not None
    assert core.pending_trade["offer_cash"] == 100
    assert 100 in TRADE_CASH_BRACKETS

    # P1 accepts the non-advancing, non-emergency trade -> incurs churn penalty (-0.08 P0, -0.04 P1)
    env_rewards = [0.0, 0.0, 0.0, 0.0]
    core._handle_trade_response(1, trade_response=1, env_rewards=env_rewards)
    assert core.property_owner[5] == 0
    assert np.isclose(env_rewards[0], -0.08)
    assert np.isclose(env_rewards[1], -0.04)

    # Universal pair anti-churn cooldown (8 rounds between P0 and P1) stays active;
    # P2 (a 3rd party) is NOT blocked by extra hold cooldowns (universal mask behavior)
    assert core.rejected_req_prop_cooldown[(1, 0, 5)] == core.round_count + 8
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 2
    core.trade_proposals_this_turn = 0
    p2_masks = core.get_action_masks(2)
    assert p2_masks["trade_req_prop"][5] == 1


def test_model_2_anti_kingmaking_penalties_4p_and_1v1():
    """Donating a monopoly one-sidedly (including 1v1 leader blunder) triggers strict donor penalties."""
    core = MonopolyCore(seed=19)
    core.allow_trade = True
    # P0 owns St. James (16), P1 owns Tennessee (18), New York (19) and Pacific (31)
    core.property_owner[16] = 0
    core.property_owner[18] = 1
    core.property_owner[19] = 1
    core.property_owner[31] = 1
    core.pending_trade = {
        "initiator": 0,
        "target": 1,
        "offer_prop": 16,
        "offer_cash": 25,
        "req_prop": 31,
        "req_cash": 0,
    }
    rewards_4p = [0.0, 0.0, 0.0, 0.0]
    core._handle_trade_response(1, trade_response=1, env_rewards=rewards_4p)
    # P1 gained Orange monopoly; P0 received neither monopoly nor >= $500 cash
    assert np.isclose(rewards_4p[1], 1.5)
    assert np.isclose(rewards_4p[0], -1.25)

    # 1v1 dominant leader (P1 already owns Brown + Light Blue monopolies) donates Kentucky (21) to P0 (owns 23, 24)
    core2 = MonopolyCore(seed=23)
    core2.allow_trade = True
    core2.bankrupt = [False, False, True, True]
    for t in [1, 3, 6, 8, 9, 21]:
        core2.property_owner[t] = 1
    core2.property_owner[23] = 0
    core2.property_owner[24] = 0
    core2.pending_trade = {
        "initiator": 0,
        "target": 1,
        "offer_prop": 40,
        "offer_cash": 500,
        "req_prop": 21,
        "req_cash": 0,
    }
    rewards_1v1 = [0.0, 0.0, 0.0, 0.0]
    core2._handle_trade_response(1, trade_response=1, env_rewards=rewards_1v1)
    assert np.isclose(rewards_1v1[0], 1.5)
    assert np.isclose(rewards_1v1[1], -1.75)


def test_model_2_networks_trainer_and_league_compatibility():
    """Verifies Residual Actor (hidden_dim=384, temperature), Critic (2032, 640), MAPPOTrainer, and LeagueManager."""
    env = MonopolyParallelEnv(max_rounds=20, seed=5)
    assert env.obs_dim == 500
    assert env.state_dim == 2032

    actor = ActorNetwork(obs_dim=500, hidden_dim=384)
    critic = CentralizedCritic(state_dim=2032, hidden_dim=640)
    assert hasattr(actor, "res_block")
    assert hasattr(actor, "trade_resp_branch")

    obs_dict, _ = env.reset()
    obs_t = torch.tensor(obs_dict["player_0"]["obs"], dtype=torch.float32).unsqueeze(0)
    masks_t = {
        k: torch.tensor(v, dtype=torch.int8).unsqueeze(0)
        for k, v in obs_dict["player_0"]["action_mask"].items()
    }
    state_t = torch.tensor(env.core.get_global_state(), dtype=torch.float32).unsqueeze(0)

    actions, log_prob, entropy = actor(obs_t, masks_t, deterministic=False, temperature=0.65)
    assert set(actions.keys()) == {
        "core", "auction", "property", "trade_target",
        "trade_offer_prop", "trade_offer_cash", "trade_req_prop",
        "trade_req_cash", "trade_response",
    }
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()

    value = critic(state_t)
    assert value.shape == (1, 4)
    assert torch.isfinite(value).all()

    trainer = MAPPOTrainer(actor=actor, critic=critic, num_mini_batches=2, target_kl=0.020)
    assert isinstance(trainer.optimizer_actor, torch.optim.AdamW)
    trainer.set_lr(1.0e-4, 2.0e-4)
    assert abs(trainer.optimizer_actor.param_groups[0]["lr"] - 1.0e-4) < 1e-12
    assert abs(trainer.optimizer_critic.param_groups[0]["lr"] - 2.0e-4) < 1e-12

    # Tiny rollout buffer update validates approx_kl metric & AdamW step
    buf = RolloutBuffer(buffer_size=8, num_envs=1, obs_dim=500, state_dim=2032, gamma=0.9985, gae_lambda=0.98)
    for _ in range(8):
        buf.insert(
            obs=obs_t.repeat(1, 4, 1),
            global_state=state_t,
            actions={k: actions[k].view(1, 1).repeat(1, 4) for k in buf.action_keys},
            action_masks={k: masks_t[k].unsqueeze(1).repeat(1, 4, 1) for k in buf.action_keys},
            rewards=torch.zeros((1, 4)),
            dones=torch.zeros((1, 4)),
            values=value.detach(),
            log_probs=log_prob.detach().view(1, 1).repeat(1, 4),
            active_masks=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )
    buf.compute_gae(torch.zeros((1, 4)), torch.zeros((1, 4)))
    metrics = trainer.train_step(buf)
    assert "approx_kl" in metrics
    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["value_loss"])

    # Presets sanity
    cfg_default = TrainingConfig()
    assert cfg_default.gamma == 0.9985
    assert cfg_default.num_mini_batches == 8
    cfg_100m = get_preset("grandmaster_100m")
    assert cfg_100m is not None and cfg_100m.total_steps == 100_000_000
    cfg_150m = get_preset("grandmaster_150m")
    assert cfg_150m is not None and cfg_150m.total_steps == 150_000_000

    # Cross-model checkpoint auto-detection helper check (Model-1 hidden=256 vs Model-2 hidden=384)
    m1_actor = Model1ActorNetwork(obs_dim=500, hidden_dim=256)
    assert "res_block.0.weight" not in m1_actor.state_dict()
    assert "res_block.0.weight" in actor.state_dict()
    lm = LeagueManager(checkpoint_dir="/tmp/m2_test_ckpts_unused")
    assert lm.checkpoint_dir == "/tmp/m2_test_ckpts_unused"
