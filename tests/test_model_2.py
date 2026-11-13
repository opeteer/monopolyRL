"""Unit & Integration Tests for Model-2 MARL Monopoly Architecture & Algorithms.

Validates that Model-2 enforces only official Monopoly rules and minimal physical/halting guards
without Model-1 hardcoded strategic heuristics ("training wheels"):
1. Unrestricted liquidity mortgaging (any unbuilt, unmortgaged property at any cash level)
   guided by policy-invariant PBRS (1.25x mortgage equity burn on wasteful mortgaging).
2. Pure public-information auction squeeze bid (`max_opp_cash + 1`) with zero hardcoded
   compute_bidding_ceiling valuation/price caps.
3. Free trade pricing across $100-$2,000 ($100 step, 21 brackets) with no mortgage-value floor,
   no silent action-mutation snap, no cross-round trade cooldowns, and up to 12 manage ops/turn.
4. Pure Markov-game outcome rewards + policy-invariant PBRS Terms 1-8 (no non-PBRS ad-hoc trade
   bonuses/penalties), plus causal trade ActorNetwork, uncoupled Huber Critic, and League/Exploiter.
"""

import torch
import numpy as np

from model_2.engine.monopoly_core import (
    MonopolyCore,
    PHASE_AUCTION,
    PHASE_PROPERTY_MANAGE,
    PHASE_TRADE_RESPONSE,
    ACT_CORE_END_MANAGE,
    ACT_AUC_JUMP_SQUEEZE,
    TRADE_CASH_BRACKETS,
    TRADE_CASH_INTERVAL,
    TRADE_CASH_MAX,
    TRADE_CASH_DIM,
)
from model_2.engine.vec_env import MASK_DIMS
from model_2.engine.pettingzoo_wrapper import MonopolyParallelEnv
from model_2.models.actor_network import ActorNetwork
from model_2.models.critic_network import CentralizedCritic
from model_2.models.storage import RolloutBuffer
from model_2.training.mappo_trainer import MAPPOTrainer
from model_2.training.league_manager import LeagueManager
from model_2.system.config_presets import get_preset, TrainingConfig
from monopoly_marl.engine.monopoly_core import MonopolyCore as Model1Core
from monopoly_marl.models.actor_network import ActorNetwork as Model1ActorNetwork


def test_model_2_unconstrained_official_masks_vs_model_1():
    """Verifies Model-2 unlocks Model-1 mortgage & auction ceiling restrictions while keeping official rules."""
    c1 = Model1Core(seed=7)
    c2 = MonopolyCore(seed=7)

    for c in (c1, c2):
        c.phase = PHASE_PROPERTY_MANAGE
        c.active_decision_player = 0
        c.current_player = 0
        c.allow_trade = True
        c.cash = [5000, 1500, 1500, 1500]
        # Give P0 maxed Brown monopoly (houses=5 -> official rule forbids mortgaging Brown) + singleton Reading RR (#5)
        c.property_owner[1] = 0
        c.property_owner[3] = 0
        c.property_houses[1] = 5
        c.property_houses[3] = 5
        c.property_owner[5] = 0
        c.property_owner[15] = 1

    m1 = c1.get_action_masks(0)
    m2 = c2.get_action_masks(0)
    # Model-1 blocks mortgaging #5 when cash >= 250 & Brown monopoly is maxed; Model-2 allows it freely
    assert m1["property"][80 + 5] == 0
    assert m2["property"][80 + 5] == 1
    # Official Monopoly rule remains enforced in Model-2: cannot mortgage Brown streets while houses exist
    assert m2["property"][80 + 1] == 0
    assert m2["property"][80 + 3] == 0
    assert m2["trade_offer_cash"].shape == (TRADE_CASH_DIM,)
    assert np.all(m2["trade_offer_cash"] == 1)

    # Confirm Model-2 removed hardcoded heuristic ceiling & cooldown locks
    assert not hasattr(c2, "compute_bidding_ceiling")
    assert not hasattr(c2, "rejected_req_prop_cooldown")
    assert not hasattr(c2, "rejected_offer_prop_cooldown")
    assert not hasattr(c2, "accepted_prop_hold_cooldown")
    assert not hasattr(c2, "prop_mortgage_cooldown")

    # Auction squeeze comparison: Baltic Ave ($60, #3); P0 has $1500, richest opponent P1 has $350
    c1_auc = Model1Core(seed=8)
    c2_auc = MonopolyCore(seed=8)
    for c in (c1_auc, c2_auc):
        c.phase = PHASE_AUCTION
        c.active_decision_player = 0
        c.current_player = 0
        c.auction_property = 3
        c.auction_highest_bid = 60
        c.auction_highest_bidder = 1
        c.auction_active_bidders = [0, 1]
        c.cash = [1500, 350, 0, 0]

    # Model-1 caps squeeze at $60 ceiling <= current bid $60 (blocked); Model-2 allows unconstrained $351 squeeze!
    assert c1_auc.get_action_masks(0)["auction"][ACT_AUC_JUMP_SQUEEZE] == 0
    assert c2_auc.get_action_masks(0)["auction"][ACT_AUC_JUMP_SQUEEZE] == 1
    c2_auc.step({
        "player_0": {
            "core": 0, "auction": ACT_AUC_JUMP_SQUEEZE, "property": 160,
            "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
            "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 0,
        }
    })
    assert c2_auc.auction_highest_bid == 351 or (c2_auc.phase == PHASE_PROPERTY_MANAGE and c2_auc.property_owner[3] == 0)


def test_model_2_trade_cash_interval_0_to_2000_step_100():
    """Verifies $100 interval trade cash brackets ($0..$2,000, 21 options) across engine, wrapper, actor, and buffer."""
    assert TRADE_CASH_INTERVAL == 100
    assert TRADE_CASH_MAX == 2000
    assert TRADE_CASH_DIM == 21
    assert TRADE_CASH_BRACKETS == list(range(0, 2001, 100))
    assert MASK_DIMS["trade_offer_cash"] == 21
    assert MASK_DIMS["trade_req_cash"] == 21

    core = MonopolyCore(seed=13)
    core.allow_trade = True
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0
    core.current_player = 0
    core.cash = [1500, 1500, 1500, 1500]
    core.property_owner[34] = 1  # P1 owns Pennsylvania Ave

    masks = core.get_action_masks(0)
    # With $1,500 cash: indices 0..15 ($0..$1,500) valid, indices 16..20 ($1,600..$2,000) masked out
    assert np.all(masks["trade_offer_cash"][:16] == 1)
    assert np.all(masks["trade_offer_cash"][16:] == 0)
    assert np.all(masks["trade_req_cash"] == 1)

    # Propose $1,500 (index 15) for Pennsylvania Ave — previously impossible with $500 cap
    act = {
        "core": ACT_CORE_END_MANAGE,
        "auction": 0,
        "property": 160,
        "trade_target": 0,
        "trade_offer_prop": 40,
        "trade_offer_cash": 15,
        "trade_req_prop": 34,
        "trade_req_cash": 0,
        "trade_response": 0,
    }
    core.step({"player_0": act})
    assert core.phase == PHASE_TRADE_RESPONSE
    assert core.pending_trade["offer_cash"] == 1500

    # Verify $2,000 (index 20) becomes affordable when cash >= 2000
    core2 = MonopolyCore(seed=14)
    core2.allow_trade = True
    core2.phase = PHASE_PROPERTY_MANAGE
    core2.active_decision_player = 0
    core2.current_player = 0
    core2.cash = [2500, 1500, 1500, 1500]
    core2.property_owner[39] = 1  # Boardwalk
    masks2 = core2.get_action_masks(0)
    assert np.all(masks2["trade_offer_cash"] == 1)
    act2 = dict(act, trade_offer_cash=20, trade_req_prop=39)
    core2.step({"player_0": act2})
    assert core2.pending_trade["offer_cash"] == 2000


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


def test_model_2_unconstrained_trade_pricing_and_no_silent_mutation():
    """Verifies free <mortgage pricing, zero silent action snap, empty-beg rejection, and zero cross-round locks."""
    core = MonopolyCore(seed=11)
    core.allow_trade = True
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0
    core.current_player = 0

    # P1 owns Boardwalk (#39, mortgage $200); P0 has $1,500 and bids $100 (below mortgage)
    core.property_owner[39] = 1
    core.cash = [1500, 1500, 1500, 1500]

    # Empty 0-for-0 beg (offer_cash=0, offer_prop=None, req_prop=39) is rejected without silent mutation
    assert core._propose_trade(0, 0, 40, 0, 39, 0) is False

    # Pure cash bid at bracket 1 ($100 < $200 Boardwalk mortgage) executes faithfully at $100
    propose_act = {
        "core": ACT_CORE_END_MANAGE,
        "auction": 0,
        "property": 160,
        "trade_target": 0,
        "trade_offer_prop": 40,
        "trade_offer_cash": 1,
        "trade_req_prop": 39,
        "trade_req_cash": 0,
        "trade_response": 0,
    }
    core.step({"player_0": propose_act})
    assert core.phase == PHASE_TRADE_RESPONSE
    assert core.pending_trade is not None
    assert core.pending_trade["offer_cash"] == 100

    # Rejecting the trade imposes ZERO hardcoded -0.08 penalty and ZERO cross-round cooldown lock
    env_rewards = [0.0, 0.0, 0.0, 0.0]
    core._handle_trade_response(1, trade_response=0, env_rewards=env_rewards)
    assert np.allclose(env_rewards, 0.0)
    assert core.property_owner[39] == 1

    # On a subsequent turn (trade_proposals_this_turn reset), P0 can immediately re-bid on Boardwalk (#39)
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0
    core.trade_proposals_this_turn = 0
    re_masks = core.get_action_masks(0)
    assert re_masks["trade_req_prop"][39] == 1

    # Same-turn halting guard: after 1 proposal this turn, cannot propose a second trade until turn ends
    core.trade_proposals_this_turn = 1
    same_turn_masks = core.get_action_masks(0)
    assert same_turn_masks["trade_req_prop"][39] == 0

    # Selling own property (#39) for $100 (< mortgage $200) also executes faithfully without silent up-snap
    core.property_owner[39] = 0
    core.trade_proposals_this_turn = 0
    assert core._propose_trade(0, 0, 39, 0, 40, 1) is True
    assert core.pending_trade["req_cash"] == 100


def test_model_2_pure_pbrs_replaces_adhoc_trade_rewards_and_manage_cap_12():
    """Verifies anti-kingmaking emerges from policy-invariant PBRS (0 ad-hoc env_rewards) & manage limit is 12."""
    # Scenario A: P1 donates Dark Blue monopoly (Boardwalk #39) to P0 (#37) for $500
    core_low = MonopolyCore(seed=19)
    core_low.allow_trade = True
    core_low.cash = [2000, 2000, 2000, 2000]
    core_low.property_owner[37] = 0
    core_low.property_owner[39] = 1
    core_low.prev_potentials = [core_low._compute_potential(p) for p in range(4)]
    core_low.phase = PHASE_TRADE_RESPONSE
    core_low.active_decision_player = 1
    core_low.current_player = 0
    core_low.pending_trade = {
        "initiator": 0, "target": 1, "offer_prop": 40, "offer_cash": 500, "req_prop": 39, "req_cash": 0,
    }
    direct_rew = [0.0, 0.0, 0.0, 0.0]
    core_low._handle_trade_response(1, trade_response=1, env_rewards=direct_rew)
    assert np.allclose(direct_rew, 0.0), "Trade outcome must use pure PBRS, not hardcoded bonuses/penalties"

    # Full step() check: selling Dark Blue monopoly for $500 yields steep negative PBRS delta for P1,
    # while selling for $1,200 yields positive PBRS delta for P1
    core_step_low = MonopolyCore(seed=20)
    core_step_low.cash = [2000, 2000, 2000, 2000]
    core_step_low.property_owner[37] = 0
    core_step_low.property_owner[39] = 1
    core_step_low.prev_potentials = [core_step_low._compute_potential(p) for p in range(4)]
    core_step_low.phase = PHASE_TRADE_RESPONSE
    core_step_low.active_decision_player = 1
    core_step_low.current_player = 0
    core_step_low.pending_trade = {
        "initiator": 0, "target": 1, "offer_prop": 40, "offer_cash": 500, "req_prop": 39, "req_cash": 0,
    }
    accept_act = {
        "core": 0, "auction": 0, "property": 160,
        "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
        "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 1,
    }
    _, rew_low, _, _ = core_step_low.step({"player_1": accept_act})
    assert rew_low["player_1"] < -0.3

    core_step_fair = MonopolyCore(seed=21)
    core_step_fair.cash = [2000, 2000, 2000, 2000]
    core_step_fair.property_owner[37] = 0
    core_step_fair.property_owner[39] = 1
    core_step_fair.prev_potentials = [core_step_fair._compute_potential(p) for p in range(4)]
    core_step_fair.phase = PHASE_TRADE_RESPONSE
    core_step_fair.active_decision_player = 1
    core_step_fair.current_player = 0
    core_step_fair.pending_trade = {
        "initiator": 0, "target": 1, "offer_prop": 40, "offer_cash": 1200, "req_prop": 39, "req_cash": 0,
    }
    _, rew_fair, _, _ = core_step_fair.step({"player_1": accept_act})
    assert rew_fair["player_1"] > rew_low["player_1"] + 2.0

    # Manage cap raised from 5 to 12: P0 can mortgage 6 properties in a single turn without premature turn end
    core_mtg = MonopolyCore(seed=22)
    core_mtg.cash = [1500, 1500, 1500, 1500]
    singletons = [1, 5, 11, 15, 16, 21]
    for tid in singletons:
        core_mtg.property_owner[tid] = 0
    core_mtg.phase = PHASE_PROPERTY_MANAGE
    core_mtg.active_decision_player = 0
    core_mtg.current_player = 0
    for tid in singletons:
        assert core_mtg.phase == PHASE_PROPERTY_MANAGE
        core_mtg.step({
            "player_0": {
                "core": ACT_CORE_END_MANAGE, "auction": 0, "property": 80 + tid,
                "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
                "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 0,
            }
        })
    assert core_mtg.manage_actions_this_turn == 6
    assert core_mtg.phase == PHASE_PROPERTY_MANAGE
    assert all(core_mtg.property_mortgaged[tid] for tid in singletons)


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


def test_model_2_mortgaged_trade_equity_and_street_only_advancement():
    """Verifies get_trade_features deducts 110% mortgage on mortgaged tiles and ignores RR/Utility for street advancement."""
    core = MonopolyCore(seed=101, randomize_first_player=False)
    core.cash = [1500, 1500, 1500, 1500]
    # P0 owns unmortgaged Illinois Ave (#24, $240); P1 owns mortgaged St. James (#16, $180, mortgage $90)
    core.property_owner[24] = 0
    core.property_owner[16] = 1
    core.property_mortgaged[16] = True
    core.phase = PHASE_TRADE_RESPONSE
    core.pending_trade = {
        "initiator": 1,
        "target": 0,
        "offer_prop": 16,
        "offer_cash": 0,
        "req_prop": 24,
        "req_cash": 0,
    }
    feat = core.get_trade_features(0)
    # Effective value of mortgaged St. James = 180 - 1.1 * 90 = 81; give Illinois = 240 -> net_equity_gain = (81 - 240)/1000
    assert np.isclose(feat[-1], (81.0 - 240.0) / 1000.0, atol=1e-4)

    # Owning Penn RR (#15) and acquiring Reading RR (#5) must NOT count as street monopoly advancement
    core2 = MonopolyCore(seed=102, randomize_first_player=False)
    core2.cash = [1500, 1500, 1500, 1500]
    core2.property_owner[15] = 0
    core2.property_owner[5] = 1
    core2.phase = PHASE_TRADE_RESPONSE
    core2.pending_trade = {
        "initiator": 0,
        "target": 1,
        "offer_prop": 40,
        "offer_cash": 200,
        "req_prop": 5,
        "req_cash": 0,
    }
    feat_rr = core2.get_trade_features(0)
    # strategic_impact[2] is advances_my_monopoly at index -6
    assert feat_rr[-6] == 0.0
    rewards = [0.0, 0.0, 0.0, 0.0]
    core2._handle_trade_response(1, trade_response=1, env_rewards=rewards)
    assert np.allclose(rewards, 0.0)


def test_model_2_unrestricted_mortgage_low_cash_request_and_railroad_card_salary():
    """Verifies unrestricted singleton mortgaging with houses in other groups, $100 cash-buy floor, and Pass GO RR card."""
    core = MonopolyCore(seed=106, randomize_first_player=False)
    core.cash = [1500, 1500, 1500, 1500]
    # Maxed Brown monopoly (hotels=5 on #1 and #3) + unbuilt singleton Illinois (#24, houses=0)
    core.property_owner[1] = 0
    core.property_owner[3] = 0
    core.property_houses[1] = 5
    core.property_houses[3] = 5
    core.property_owner[24] = 0
    core.phase = PHASE_PROPERTY_MANAGE
    core.active_decision_player = 0
    masks = core.get_action_masks(0)
    # Official Monopoly rule: #1 and #3 cannot be mortgaged because Brown group has hotels,
    # whereas unbuilt Illinois (#24) can be mortgaged freely regardless of $1,500 cash or maxed Brown monopoly!
    assert masks["property"][80 + 1] == 0
    assert masks["property"][80 + 3] == 0
    assert masks["property"][80 + 24] == 1

    # Low-cash zero-property player with $100 (smallest non-zero bracket) CAN request Boardwalk (#39) without mortgage floor,
    # while player with $50 (< $100 smallest non-zero bracket) and 0 properties cannot make any non-zero offer.
    core_100 = MonopolyCore(seed=107, randomize_first_player=False)
    core_100.cash = [100, 1500, 1500, 1500]
    core_100.property_owner[39] = 1
    core_100.phase = PHASE_PROPERTY_MANAGE
    core_100.active_decision_player = 0
    assert core_100.get_action_masks(0)["trade_req_prop"][39] == 1

    core_50 = MonopolyCore(seed=109, randomize_first_player=False)
    core_50.cash = [50, 1500, 1500, 1500]
    core_50.property_owner[39] = 1
    core_50.phase = PHASE_PROPERTY_MANAGE
    core_50.active_decision_player = 0
    assert core_50.get_action_masks(0)["trade_req_prop"][39] == 0

    # Advance nearest railroad routes Pass GO bonus through house_rules.apply_pass_go_salary (not hardcoded +200)
    core_hr = MonopolyCore(seed=108, randomize_first_player=False)
    calls = []
    core_hr.house_rules.apply_pass_go_salary = lambda pid, sal=200: (calls.append((pid, sal)) or 333)
    core_hr.position[0] = 36  # Chance near Boardwalk -> wraps past GO to Reading RR (#5)
    cash_before = core_hr.cash[0]
    core_hr._execute_card_action(0, {"action": "advance_nearest_railroad"}, is_double=False)
    assert core_hr.position[0] == 5
    assert calls == [(0, 200)]
    assert core_hr.cash[0] == cash_before + 333


def test_model_2_autoregressive_trade_conditioning_and_critic_huber_update():
    """Verifies trade_target & offer_cash conditioning gradients plus decoupled critic + exploiter PPO step."""
    actor = ActorNetwork(obs_dim=500, hidden_dim=384)
    critic = CentralizedCritic(state_dim=2032)

    # 1. Verify trade_target shifts head_trade_req_cash logits on pure property sales (req_prop == 40)
    obs = torch.zeros((1, 500), dtype=torch.float32)
    obs[0, 307 + 4] = 1.0  # PHASE_PROPERTY_MANAGE
    masks = {
        "core": torch.ones((1, 6), dtype=torch.int8),
        "auction": torch.ones((1, 7), dtype=torch.int8),
        "property": torch.zeros((1, 161), dtype=torch.int8),
        "trade_target": torch.ones((1, 3), dtype=torch.int8),
        "trade_offer_prop": torch.zeros((1, 41), dtype=torch.int8),
        "trade_offer_cash": torch.ones((1, 21), dtype=torch.int8),
        "trade_req_prop": torch.zeros((1, 41), dtype=torch.int8),
        "trade_req_cash": torch.ones((1, 21), dtype=torch.int8),
        "trade_response": torch.ones((1, 2), dtype=torch.int8),
    }
    masks["property"][0, 160] = 1
    masks["trade_offer_prop"][0, 14] = 1
    masks["trade_offer_prop"][0, 40] = 1
    masks["trade_req_prop"][0, 40] = 1
    masks["trade_req_prop"][0, 16] = 1

    # Give emb_trade_target_for_cash distinct weights on orthogonal head and confirm logits differ across targets 0 vs 1
    with torch.no_grad():
        actor.emb_trade_target_for_cash.weight[0].fill_(0.05)
        actor.emb_trade_target_for_cash.weight[1].fill_(-0.05)
    acts_t0 = {
        "core": torch.tensor([0]), "auction": torch.tensor([0]), "property": torch.tensor([160]),
        "trade_target": torch.tensor([0]), "trade_offer_prop": torch.tensor([14]),
        "trade_offer_cash": torch.tensor([0]), "trade_req_prop": torch.tensor([40]),
        "trade_req_cash": torch.tensor([10]), "trade_response": torch.tensor([0]),
    }
    acts_t1 = {**acts_t0, "trade_target": torch.tensor([1])}
    lp0, _ = actor.evaluate_actions(obs, masks, acts_t0)
    lp1, _ = actor.evaluate_actions(obs, masks, acts_t1)
    assert lp0.requires_grad and torch.isfinite(lp0) and torch.isfinite(lp1)
    assert not torch.allclose(lp0, lp1), "head_trade_req_cash must condition on trade_target when req_prop == 40"

    # Confirm gradient reaches new conditioning embeddings across property-sale + swap paths
    actor.zero_grad()
    acts_swap = {
        "core": torch.tensor([0]), "auction": torch.tensor([0]), "property": torch.tensor([160]),
        "trade_target": torch.tensor([0]), "trade_offer_prop": torch.tensor([14]),
        "trade_offer_cash": torch.tensor([0]), "trade_req_prop": torch.tensor([16]),
        "trade_req_cash": torch.tensor([5]), "trade_response": torch.tensor([0]),
    }
    lp_sale, _ = actor.evaluate_actions(obs, masks, acts_t0)
    lp_swap, _ = actor.evaluate_actions(obs, masks, acts_swap)
    (lp_sale + lp_swap).sum().backward()
    assert actor.emb_trade_req_for_target.weight.grad.abs().sum() > 0.0
    assert actor.emb_trade_target_for_offer.weight.grad.abs().sum() > 0.0
    assert actor.emb_trade_target_for_cash.weight.grad.abs().sum() > 0.0
    assert actor.emb_trade_offer_cash_for_req.weight.grad.abs().sum() > 0.0

    # 2. Verify Critic Huber + decoupled value_clip_param=2.0 + Exploiter actor PPO update
    exploiter = ActorNetwork(obs_dim=500, hidden_dim=384)
    exp_opt = torch.optim.AdamW(exploiter.parameters(), lr=2e-4)
    trainer = MAPPOTrainer(actor=actor, critic=critic, clip_param=0.12, value_clip_param=2.0, num_mini_batches=2)

    buf = RolloutBuffer(buffer_size=4, num_envs=2, obs_dim=500, state_dim=2032, gamma=0.9985, gae_lambda=0.98)
    state_t = torch.randn((2, 2032), dtype=torch.float32) * 0.25
    rewards_t = torch.tensor([[10.0, -2.0, -5.0, 0.0], [10.0, -2.0, -5.0, 0.0]], dtype=torch.float32)
    for _ in range(4):
        buf.insert(
            obs=obs.unsqueeze(1).repeat(2, 4, 1),
            global_state=state_t,
            actions={k: acts_t0[k].view(1, 1).repeat(2, 4) for k in buf.action_keys},
            action_masks={k: masks[k].unsqueeze(1).repeat(2, 4, 1) for k in buf.action_keys},
            rewards=rewards_t,
            dones=torch.zeros((2, 4)),
            values=torch.zeros((2, 4)),
            log_probs=lp0.detach().view(1, 1).repeat(2, 4),
            active_masks=torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
            exploiter_masks=torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
        )
    buf.compute_gae(torch.zeros((2, 4)), torch.zeros((2, 4)))
    exp_before = exploiter.trunk[0].weight.detach().clone()
    critic_before = critic.in_proj[0].weight.detach().clone()
    metrics = trainer.train_step(buf, exploiter_actor=exploiter, exploiter_optimizer=exp_opt)
    assert np.isfinite(metrics["value_loss"]) and metrics["value_loss"] > 0.0
    assert not torch.allclose(critic.in_proj[0].weight, critic_before)
    assert not torch.allclose(exploiter.trunk[0].weight, exp_before)


def test_model_2_anti_stalemate_timeout_payoff_and_monopoly_dominance_over_cash():
    """Verifies monopoly + houses strictly beat cash hoarding and max_rounds stalemate is penalized."""
    # 1. Equal total capital ($1,500): Orange monopoly ($560) + $940 cash strictly beats $1,500 pure cash
    c_cash = MonopolyCore(seed=201, randomize_first_player=False)
    c_cash.cash = [1500, 1500, 1500, 1500]
    c_mono = MonopolyCore(seed=201, randomize_first_player=False)
    c_mono.cash = [940, 1500, 1500, 1500]
    for tid in (16, 18, 19):
        c_mono.property_owner[tid] = 0
    phi_cash_r0 = c_cash._compute_potential(0)
    phi_mono_r0 = c_mono._compute_potential(0)
    assert phi_mono_r0 > phi_cash_r0 + 8.0

    # Building 9 houses ($900) on Orange strictly outperforms holding that $900 as idle cash
    c_built = MonopolyCore(seed=201, randomize_first_player=False)
    c_built.cash = [40, 1500, 1500, 1500]
    for tid in (16, 18, 19):
        c_built.property_owner[tid] = 0
        c_built.property_houses[tid] = 3
    assert c_built._compute_potential(0) > phi_mono_r0 + 8.0

    # 2. Term 6 keeps deepening past round 40 up to round 80 (outpacing +$200 Pass GO cash), then caps cleanly
    c_hoard = MonopolyCore(seed=202, randomize_first_player=False)
    c_hoard.cash = [1500, 1500, 1500, 1500]
    c_hoard.round_count = 40
    phi_r40 = c_hoard._compute_potential(0)
    c_hoard.round_count = 80
    c_hoard.cash[0] = 2700  # Even +$1,200 Pass GO cash (+5.4 Term 1) cannot offset -10.5 extra starvation
    phi_r80 = c_hoard._compute_potential(0)
    assert phi_r80 < phi_r40 - 4.0
    c_hoard.round_count = 120
    assert np.isclose(c_hoard._compute_potential(0), phi_r80)

    # 3. Anti-stalemate timeout: sole Net Worth leader gets +1.0 (not +4.0/+10.0), others get -10.0
    c_to = MonopolyCore(seed=203, max_rounds=10, randomize_first_player=False)
    c_to.round_count = 10
    c_to.phase = PHASE_PROPERTY_MANAGE
    c_to.active_decision_player = 0
    c_to.current_player = 0
    c_to.cash = [2000, 1500, 1500, 1500]
    c_to.prev_potentials = [c_to._compute_potential(p) for p in range(4)]
    _, rew_to, dones_to, _ = c_to.step({
        "player_0": {
            "core": ACT_CORE_END_MANAGE, "auction": 0, "property": 160,
            "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
            "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 0,
        }
    })
    assert dones_to["__all__"] is True
    # P0 expected env_reward +1.0 (minus tiny (1-gamma)*Phi); P1..P3 receive -10.0 penalty
    assert 0.5 < rew_to["player_0"] < 1.5
    for loser in ("player_1", "player_2", "player_3"):
        assert rew_to[loser] < -9.5

    # 4. Exact Net Worth tie at timeout penalizes ALL 4 passive players (-10.0)
    c_tie = MonopolyCore(seed=204, max_rounds=10, randomize_first_player=False)
    c_tie.round_count = 10
    c_tie.phase = PHASE_PROPERTY_MANAGE
    c_tie.active_decision_player = 0
    c_tie.current_player = 0
    c_tie.cash = [1500, 1500, 1500, 1500]
    c_tie.prev_potentials = [c_tie._compute_potential(p) for p in range(4)]
    _, rew_tie, dones_tie, _ = c_tie.step({
        "player_0": {
            "core": ACT_CORE_END_MANAGE, "auction": 0, "property": 160,
            "trade_target": 0, "trade_offer_prop": 40, "trade_offer_cash": 0,
            "trade_req_prop": 40, "trade_req_cash": 0, "trade_response": 0,
        }
    })
    assert dones_tie["__all__"] is True
    for pl_name in ("player_0", "player_1", "player_2", "player_3"):
        assert rew_tie[pl_name] < -9.5

    # 5. Default training horizon matches external 4p Monopoly bot benchmark buffer (80 rounds)
    assert TrainingConfig().max_rounds == 80
    assert get_preset("grandmaster_150m").max_rounds == 80


