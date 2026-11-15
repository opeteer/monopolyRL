"""Comprehensive Unit & Integration Tests for Model-3 Architecture, Search, Bots, and Training."""

import tempfile
import numpy as np
import torch

from model_2.engine.board_data import PURCHASABLE_TILES, COLOR_GROUPS
from model_3.engine.monopoly_core import (
    MonopolyCore,
    PHASE_PROPERTY_MANAGE,
    PHASE_TRADE_RESPONSE,
    PHASE_TRADE_COUNTER,
    PHASE_GAME_OVER,
    ACT_PROP_END_MANAGE,
    ACT_PROP_PROPOSE_TRADE,
    BUNDLE_DIM,
    GOOJF_SLOT_1,
    TILE_FEAT_DIM,
    SCALAR_OBS_DIM,
    EVENT_RING_LEN,
    EVENT_FEAT_DIM,
    GLOBAL_STATE_DIM,
    ACTION_KEYS,
    MASK_SHAPES,
    ACT_SHAPES,
    TRADE_CASH_BRACKETS,
)
from model_3.engine.vec_env import SharedMemoryMonopolyVecEnvV3
from model_3.models.actor_network import ActorNetworkV3
from model_3.models.critic_network import QuantileCritic
from model_3.models.storage import RolloutBufferV3
from model_3.agents.pro_heuristic_bot import ProHeuristicBot, OverbidderBot, make_noop_action_v3
from model_3.search.chance_lookahead import BoundedChanceLookahead
from model_3.training.mappo_trainer import MAPPOTrainerV3
from model_3.training.psro_league import PSROLeagueManager


def test_fast_snapshot_equivalence():
    """Verifies fast save_snapshot()/restore_snapshot() produces bit-exact rollouts vs un-snapshotted play."""
    c1 = MonopolyCore(seed=1234, max_rounds=60)
    c1.reset(seed=1234)
    bots = [ProHeuristicBot(seed=i) for i in range(4)]

    # Advance 40 varied steps to populate board ownership, cash, decks, event ring
    for _ in range(40):
        if c1.phase == PHASE_GAME_OVER:
            break
        ap = c1.active_decision_player
        c1.step({ap: bots[ap].act(c1, ap)}, skip_obs=True)

    # Inject a GOOJF + house state for richer snapshot test
    c1.property_owner[16] = 0
    c1.property_owner[18] = 0
    c1.property_owner[19] = 0
    c1.property_houses[16] = 2
    c1.bank_houses -= 2
    c1.goojf_cards[1] = 1
    c1.goojf_sources[1] = ["chance"]

    snap = c1.save_snapshot()
    obs_ref = c1.get_observations()
    gs_ref = c1.get_global_state()

    # Record a 15-step reference rollout directly from state
    ref_trace = []
    for _ in range(15):
        if c1.phase == PHASE_GAME_OVER:
            break
        ap = c1.active_decision_player
        act = bots[ap].act(c1, ap)
        _, rew, done, _ = c1.step({ap: act}, skip_obs=False)
        ref_trace.append((c1.phase, c1.active_decision_player, tuple(c1.cash), tuple(c1.position), tuple(c1.property_owner), rew))

    # Restore exact snapshot, mutate state aggressively first, then restore again and replay
    c1.cash[0] = 99999
    c1.property_houses[16] = 5
    c1.restore_snapshot(snap)

    obs_restored = c1.get_observations()
    gs_restored = c1.get_global_state()
    assert np.array_equal(gs_ref, gs_restored)
    assert np.array_equal(obs_ref["tile_state"]["player_0"], obs_restored["tile_state"]["player_0"])
    assert np.array_equal(obs_ref["scalar_obs"]["player_0"], obs_restored["scalar_obs"]["player_0"])
    assert np.array_equal(obs_ref["event_ring"]["player_0"], obs_restored["event_ring"]["player_0"])

    bots2 = [ProHeuristicBot(seed=i) for i in range(4)]
    replay_trace = []
    for _ in range(len(ref_trace)):
        ap = c1.active_decision_player
        act = bots2[ap].act(c1, ap)
        _, rew, done, _ = c1.step({ap: act}, skip_obs=False)
        replay_trace.append((c1.phase, c1.active_decision_player, tuple(c1.cash), tuple(c1.position), tuple(c1.property_owner), rew))

    assert ref_trace == replay_trace


def test_bundle_trade_unconstrained_and_counter_masks():
    """Verifies 5-for-3+GOOJF bundles, $25 cash precision, 1x counter-offer, and anti-paradox masks."""
    core = MonopolyCore(seed=77)
    core.reset(seed=77)
    core.phase = PHASE_PROPERTY_MANAGE
    core.current_player = 0
    core.active_decision_player = 0
    core.allow_trade = True
    core.trade_proposals_this_turn = 0

    # P0 owns 5 properties; P1 (rel_idx=0) owns 3 properties + 1 GOOJF
    p0_tiles = [1, 3, 6, 8, 9]
    p1_tiles = [11, 13, 14]
    for t in p0_tiles:
        core.property_owner[t] = 0
    for t in p1_tiles:
        core.property_owner[t] = 1
    core.goojf_cards[1] = 1
    core.goojf_sources[1] = ["chance"]
    core.cash[0] = 600
    core.cash[1] = 500

    masks0 = core.get_action_masks(0)
    assert masks0["property"][ACT_PROP_PROPOSE_TRADE] == 1
    assert masks0["property"][ACT_PROP_END_MANAGE] == 1
    assert TRADE_CASH_BRACKETS[1] == 25

    # P0 proposes 5 properties + $25 cash for P1's 3 properties + GOOJF
    req_b = np.zeros(BUNDLE_DIM, dtype=np.int8)
    for t in p1_tiles:
        req_b[t] = 1
    req_b[GOOJF_SLOT_1] = 1

    off_b = np.zeros(BUNDLE_DIM, dtype=np.int8)
    for t in p0_tiles:
        off_b[t] = 1

    act_prop = make_noop_action_v3()
    act_prop["property"] = ACT_PROP_PROPOSE_TRADE
    act_prop["trade_target"] = 0  # (0 + 1 + 0) % 4 = Player 1
    act_prop["trade_req_bundle"] = req_b
    act_prop["trade_offer_bundle"] = off_b
    act_prop["trade_offer_cash"] = 1  # $25 bracket
    act_prop["trade_req_cash"] = 0

    core.step({0: act_prop}, skip_obs=True)
    assert core.phase == PHASE_TRADE_RESPONSE
    assert core.active_decision_player == 1
    assert core.pending_trade is not None
    assert len(core.pending_trade["offer_tiles"]) == 5
    assert len(core.pending_trade["req_tiles"]) == 3
    assert core.pending_trade["req_goojf"] == 1
    assert core.pending_trade["offer_cash"] == 25

    # Verify target has Reject(0), Accept(1), AND 1x Counter(2) unmasked
    m1_resp = core.get_action_masks(1)["trade_response"]
    assert list(m1_resp) == [1, 1, 1]

    # Empty/1-way giveaway anti-paradox check on same state via direct _propose_bundle_trade
    snap_resp = core.save_snapshot()
    act_counter_go = make_noop_action_v3()
    act_counter_go["trade_response"] = 2
    core.step({1: act_counter_go}, skip_obs=True)
    assert core.phase == PHASE_TRADE_COUNTER
    assert core.active_decision_player == 1

    # In counter phase, target mask MUST lock exclusively onto orig_initiator=0 -> rel_idx = (0 - 1 - 1) % 4 = 2
    mc = core.get_action_masks(1)
    assert int(mc["trade_target"].sum()) == 1
    assert mc["trade_target"][2] == 1

    # Reject empty-beg counter: req 0 assets + 0 cash -> cancels cleanly back to P0 manage
    snap_counter = core.save_snapshot()
    bad_counter = make_noop_action_v3()
    bad_counter["trade_target"] = 2
    core.step({1: bad_counter}, skip_obs=True)
    assert core.phase == PHASE_PROPERTY_MANAGE and core.active_decision_player == 0

    # Replay valid 1x counter-offer: P1 counters for P0's 5 props + $50 cash -> P0 sees mask[2] == 0 (no 2nd counter!)
    core.restore_snapshot(snap_counter)
    good_counter = make_noop_action_v3()
    good_counter["trade_target"] = 2
    # In PHASE_TRADE_COUNTER as P1, P1 offers its 3 props + GOOJF and requests P0's 5 props + $50 cash
    c_req = np.zeros(BUNDLE_DIM, dtype=np.int8)
    for t in p0_tiles:
        c_req[t] = 1
    c_off = np.zeros(BUNDLE_DIM, dtype=np.int8)
    for t in p1_tiles:
        c_off[t] = 1
    c_off[GOOJF_SLOT_1] = 1
    good_counter["trade_req_bundle"] = c_req
    good_counter["trade_offer_bundle"] = c_off
    good_counter["trade_req_cash"] = 2  # $50 bracket

    core.step({1: good_counter}, skip_obs=True)
    assert core.phase == PHASE_TRADE_RESPONSE
    assert core.active_decision_player == 0
    m0_counter_resp = core.get_action_masks(0)["trade_response"]
    assert m0_counter_resp[0] == 1 and m0_counter_resp[1] == 1
    assert m0_counter_resp[2] == 0, "2nd counter-offer must be strictly prohibited!"

    # P0 accepts the counter-offer -> all 8 properties + GOOJF + $50 swap atomically!
    act_acc = make_noop_action_v3()
    act_acc["trade_response"] = 1
    core.step({0: act_acc}, skip_obs=True)
    assert core.phase == PHASE_PROPERTY_MANAGE
    assert core.active_decision_player == 0
    for t in p1_tiles:
        assert core.property_owner[t] == 0
    for t in p0_tiles:
        assert core.property_owner[t] == 1
    assert core.goojf_cards[0] == 1 and core.goojf_cards[1] == 0
    assert core.cash[0] == 550 and core.cash[1] == 550


def test_conditioned_logit_l2_stability():
    """Tests both TileMixer and LightTx actors for set pooling sensitivity, exact logit_l2 masking, and safe grads."""
    B = 8
    for arch in ("tile_mixer", "light_tx"):
        actor = ActorNetworkV3(arch=arch, d_model=64, hidden_dim=384)
        tile = torch.randn(B, 40, TILE_FEAT_DIM)
        scalar = torch.randn(B, SCALAR_OBS_DIM)
        scalar[:, 40:48] = 0.0
        scalar[:, 40 + PHASE_PROPERTY_MANAGE] = 1.0
        ev = torch.zeros(B, EVENT_RING_LEN, EVENT_FEAT_DIM, dtype=torch.int64)
        masks = {k: torch.zeros((B,) + MASK_SHAPES[k], dtype=torch.float32) for k in ACTION_KEYS}
        masks["property"][:, ACT_PROP_END_MANAGE] = 1.0
        masks["property"][:, ACT_PROP_PROPOSE_TRADE] = 1.0
        masks["trade_target"][:, 0] = 1.0
        masks["trade_req_bundle"][:, 0, :6] = 1.0
        masks["trade_offer_bundle"][:, 10:16] = 1.0
        masks["trade_offer_cash"][:, :] = 1.0
        masks["trade_req_cash"][:, :] = 1.0

        acts, lp, ent = actor(tile, scalar, ev, masks)
        lp2, ent2, ukl, l2, ldict = actor.evaluate_actions_with_reg(tile, scalar, ev, masks, acts)
        assert torch.allclose(lp, lp2, atol=1e-5)
        assert torch.isfinite(ukl).all() and torch.isfinite(l2).all()
        assert (l2 >= 0).all()

        # Backward check: gradients finite and non-exploding despite -1e8 masked entries
        loss = (-lp2 - 0.01 * ent2 + 0.02 * ukl + 0.0015 * l2).mean()
        actor.zero_grad()
        loss.backward()
        grad_norm = sum(p.grad.norm().item() ** 2 for p in actor.parameters() if p.grad is not None) ** 0.5
        assert np.isfinite(grad_norm) and grad_norm < 50.0

        # Inactive check: when phase & masks are zeroed, logit_l2 MUST be strictly 0
        scalar_zero = scalar.clone()
        scalar_zero[:, 40:48] = 0.0
        masks_zero = {k: torch.zeros_like(v) for k, v in masks.items()}
        _, _, ukl_z, l2_z, _ = actor.evaluate_actions_with_reg(tile, scalar_zero, ev, masks_zero, acts)
        assert (l2_z == 0.0).all() and (ukl_z == 0.0).all()

    # Check conditioning sensitivity: changing trade_offer_bundle MUST change cash head logits
    actor = ActorNetworkV3(arch="tile_mixer")
    tile1 = torch.randn(2, 40, TILE_FEAT_DIM)
    scalar1 = torch.zeros(2, SCALAR_OBS_DIM)
    scalar1[:, 40 + PHASE_PROPERTY_MANAGE] = 1.0
    ev1 = torch.zeros(2, EVENT_RING_LEN, EVENT_FEAT_DIM, dtype=torch.int64)
    masks1 = {k: torch.ones((2,) + MASK_SHAPES[k], dtype=torch.float32) for k in ACTION_KEYS}

    acts_a = {k: torch.zeros((2,) + ACT_SHAPES[k], dtype=torch.int8 if len(ACT_SHAPES[k]) > 0 else torch.int64) for k in ACTION_KEYS}
    acts_a["property"][:] = ACT_PROP_PROPOSE_TRADE
    acts_a["trade_req_bundle"][:, 1] = 1
    acts_b = {k: (v.clone()) for k, v in acts_a.items()}
    acts_b["trade_offer_bundle"][:, 16] = 1  # Add high-value property to offer bundle!
    acts_b["trade_offer_bundle"][:, 18] = 1

    with torch.no_grad():
        _, _, _, _, ldict_a = actor.evaluate_actions_with_reg(tile1, scalar1, ev1, masks1, acts_a)
        _, _, _, _, ldict_b = actor.evaluate_actions_with_reg(tile1, scalar1, ev1, masks1, acts_b)
    cash_diff = (ldict_a["trade_offer_cash"] - ldict_b["trade_offer_cash"]).abs().max().item()
    assert cash_diff > 1e-4, f"Cash head did not condition on bundle bits (diff={cash_diff})!"


def test_qr_critic_cvar_and_search_ev():
    """Tests QuantileCritic bimodal Huber regression + CVaR < Mean, Pro bots, and Lookahead 4-house lock."""
    # 1. Bimodal Huber distribution test: -10 (p=0.25) and +5 (p=0.75) -> Mean = +1.25, CVaR_0.25 = -10.0
    critic = QuantileCritic(state_dim=GLOBAL_STATE_DIM, hidden_dim=128, num_quantiles=32)
    opt = torch.optim.Adam(critic.parameters(), lr=3e-3)
    gs_fixed = torch.zeros(4, GLOBAL_STATE_DIM)
    rng = np.random.default_rng(123)
    alive = torch.ones(4, 4)

    for _ in range(180):
        # Sample bimodal Bellman targets in [4, 4, 32]
        samples = np.where(rng.random((4, 4, 32)) < 0.25, -10.0, 5.0).astype(np.float32)
        targ = torch.from_numpy(samples)
        pred = critic(gs_fixed)
        loss = MAPPOTrainerV3.quantile_huber_loss(pred, targ, critic.taus, scale=5.0, alive_mask=alive, kappa=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        rv, qs = critic.get_risk_value(gs_fixed[:1], cvar_alpha=0.25, cvar_weight=0.50)
        mean_est = qs[0, 0].mean().item()
        low_q = torch.sort(qs[0, 0])[0][:8].mean().item()
        assert low_q < -4.0 and mean_est > -0.5, f"QR failed bimodal separation: low_q={low_q}, mean={mean_est}"
        assert rv[0, 0].item() < mean_est, f"CVaR risk value ({rv[0,0].item()}) should be below mean ({mean_est})!"

    # 2. BoundedChanceLookahead avoids Hotel upgrade when bank_houses <= 16 and leaf evaluator penalizes relieving lock
    core = MonopolyCore(seed=55)
    core.reset(seed=55)
    core.phase = PHASE_PROPERTY_MANAGE
    core.current_player = 0
    core.active_decision_player = 0
    core.cash = [1200, 1200, 1200, 1200]
    for t in COLOR_GROUPS["ORANGE"]:  # 16, 18, 19
        core.property_owner[t] = 0
        core.property_houses[t] = 4
    for t in COLOR_GROUPS["LIGHT_BLUE"]:  # 6, 8, 9
        core.property_owner[t] = 1
        core.property_houses[t] = 2
    core.bank_houses = 32 - 12 - 6  # 14 houses left (<= 16!)

    def shortage_aware_eval(gs_batch: np.ndarray, p_id: int) -> np.ndarray:
        # gs_batch[:, 416] is bank_houses / 32.0: reward keeping bank_houses low when opponent needs houses!
        bh_norm = gs_batch[:, 416]
        return (1.0 - bh_norm) * 10.0

    lookahead = BoundedChanceLookahead(top_k=4, branch_dice=True)
    snap_before = core.save_snapshot()
    logits_hint = {"property": torch.zeros(162, dtype=torch.float32)}
    logits_hint["property"][16] = 2.0  # Prioritize considering hotel on St. James (16) vs END_MANAGE (160)
    best_act, s_valid, _, s_probs = lookahead.select_action(
        core,
        player_id=0,
        sampled_actions=make_noop_action_v3(),
        actor_logits=logits_hint,
        leaf_value_fn=shortage_aware_eval,
    )
    assert s_valid and abs(float(s_probs.sum()) - 1.0) < 1e-4
    assert core.save_snapshot() == snap_before, "Lookahead mutated core state!"
    assert best_act["property"] == ACT_PROP_END_MANAGE, (
        f"Expected 4-house shortage lock (END_MANAGE=160), got prop={best_act['property']}"
    )

    # 3. Verify ProHeuristicBot ALSO enforces 4-house shortage lock here
    pro_bot = ProHeuristicBot(seed=9)
    bot_act = pro_bot.act(core, 0)
    assert bot_act["property"] == ACT_PROP_END_MANAGE

    # 4. Run a full 4-bot match to completion / max_rounds verifying 0 illegal states
    core_game = MonopolyCore(seed=88, max_rounds=45)
    core_game.reset(seed=88)
    mixed_bots = [ProHeuristicBot(seed=0), ProHeuristicBot(seed=1), OverbidderBot(seed=2), ProHeuristicBot(seed=3)]
    for _ in range(1500):
        if core_game.phase == PHASE_GAME_OVER:
            break
        ap = core_game.active_decision_player
        _, _, dones, _ = core_game.step({ap: mixed_bots[ap].act(core_game, ap)}, skip_obs=True)
        if all(dones.values()):
            break


def test_vec_env_and_mappo_smoke_iteration():
    """End-to-end test: SharedMemoryMonopolyVecEnvV3 (with search & bots) -> RolloutBufferV3 -> MAPPOTrainerV3 + PSRO save/load."""
    with tempfile.TemporaryDirectory() as tmpd:
        actor = ActorNetworkV3(arch="tile_mixer", d_model=64, hidden_dim=384)
        critic = QuantileCritic(hidden_dim=256, num_quantiles=32)
        trainer = MAPPOTrainerV3(actor, critic, ppo_epochs=2, num_mini_batches=2, search_distill_coef=0.05)
        league = PSROLeagueManager(checkpoint_dir=tmpd)

        num_envs = 4
        T = 8
        vec = SharedMemoryMonopolyVecEnvV3(
            env_configs=[{"max_rounds": 40, "seed": 300 + i} for i in range(num_envs)],
            num_workers=2,
            search_env_indices=[0],
            search_cfg={"critic_hidden_dim": 256, "num_quantiles": 32, "top_k": 4},
        )
        try:
            vec.sync_search_critic(critic.state_dict())
            vec.bot_seat_types_buf[1, 1] = 1
            vec.bot_seat_types_buf[2, 2] = 2

            tile, scalar, ev, gs, masks, ap = vec.reset()
            buf = RolloutBufferV3(buffer_size=T, num_envs=num_envs, num_quantiles=32)

            for _ in range(T):
                snap_t, snap_s, snap_e, snap_g = tile.copy(), scalar.copy(), ev.copy(), gs.copy()
                snap_masks = {k: masks[k].copy() for k in ACTION_KEYS}

                with torch.no_grad():
                    acts, lps, _ = actor(
                        torch.from_numpy(snap_t.reshape(-1, 40, TILE_FEAT_DIM)),
                        torch.from_numpy(snap_s.reshape(-1, SCALAR_OBS_DIM)),
                        torch.from_numpy(snap_e.reshape(-1, EVENT_RING_LEN, EVENT_FEAT_DIM).astype(np.int64)),
                        {k: torch.from_numpy(snap_masks[k].reshape((-1,) + MASK_SHAPES[k]).astype(np.float32)) for k in ACTION_KEYS},
                    )
                    rv, qs = critic.get_risk_value(torch.from_numpy(snap_g))

                for k in ACTION_KEYS:
                    vec.act_bufs[k][:] = acts[k].numpy().reshape((num_envs, 4) + ACT_SHAPES[k])
                acts_np = {k: vec.act_bufs[k].copy() for k in ACTION_KEYS}
                active_m = np.zeros((num_envs, 4), dtype=np.float32)
                for e_i in range(num_envs):
                    a_i = int(ap[e_i])
                    if vec.bot_seat_types_buf[e_i, a_i] == 0:
                        active_m[e_i, a_i] = 1.0

                tile, scalar, ev, gs, masks, rews, dones, ap, _ = vec.step(use_search=True)
                overridden = vec.search_overridden_buf
                if np.any(overridden > 0.5):
                    active_m[overridden > 0.5] = 0.0

                buf.insert(
                    snap_t, snap_s, snap_e, snap_g, acts_np, snap_masks,
                    rews.copy(), dones.copy(), rv.numpy(), qs.numpy(), lps.numpy().reshape(num_envs, 4),
                    active_masks=active_m,
                    search_valid=vec.search_valid_buf.copy(),
                    search_head=vec.search_head_buf.copy(),
                    search_probs=vec.search_probs_buf.copy(),
                )

            last_gs = gs.copy()
            last_d = dones.copy()
        finally:
            vec.close()

        with torch.no_grad():
            _, last_qs = critic.get_risk_value(torch.from_numpy(last_gs))
        buf.compute_gae_and_quantile_targets(last_qs.numpy(), last_d)
        metrics = trainer.train_step(buf)
        assert all(np.isfinite(v) for v in metrics.values()), metrics

        # Verify PSRO checkpoint save & reload preserves weights and trainer state
        exp_actor = ActorNetworkV3(arch="tile_mixer", d_model=64, hidden_dim=384)
        exp_opt = torch.optim.AdamW(exp_actor.parameters(), lr=2e-4)
        ckpt_path = f"{tmpd}/checkpoint_latest.pt"
        league.save_training_checkpoint(
            ckpt_path, iteration=3, total_steps=1024,
            actor=actor, critic=critic, exploiter_actor=exp_actor,
            exploiter_optimizer=exp_opt, trainer_state=trainer.state_dict(),
        )
        loaded = PSROLeagueManager.load_training_checkpoint(ckpt_path)
        assert loaded["iteration"] == 3 and loaded["total_steps"] == 1024
        assert loaded["model_config"]["arch"] == "tile_mixer"
