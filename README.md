# Tournament-Grade MARL Monopoly Engine with CTDE, Advanced Trading, League Training & Pygame GUI

An autonomous Multi-Agent Reinforcement Learning (MARL) tournament Monopoly engine implementing the full official tournament rules, Centralized Training with Decentralized Execution (CTDE) using MAPPO, Strict Action Masking, Potential-Based Reward Shaping (PBRS), League Training (Main Agent vs. Main Exploiter vs. Historical Checkpoint Pool), and a real-time interactive Pygame GUI.

---

## 1. Architectural Principles & Strict Constraints

1. **Identical Neural Blueprint**:
   - All 4 agents share a single, 100% identical neural network architecture (`ActorNetwork`).
   - No hand-crafted heuristics or personality biases (no predefined "Aggressive", "Hoarder", or "Trader" bots).
   - Uniform Orthogonal Initialization across all layers with identical gain ($\sqrt{2}$ for hidden layers, $0.01$ for factorized action heads).
   - All strategies (house scarcity locking, endgame jail sheltering, bidding thresholds, trade valuations) emerge purely through gradient optimization.

2. **Full Official Tournament Rules**:
   - **Physical Bank Scarcity**: Exactly 32 houses ($B_h = 32$) and 12 hotels ($B_{ht} = 12$). When houses run out ($B_h = 0$), building is strictly frozen.
   - **Mandatory Open Auctions**: When an active player declines purchasing an unowned landed property, the bank immediately triggers an open ascending auction across all 4 players with no reserve ($1 bid).
   - **Speeding Penalty**: Throwing 3 consecutive doubles in one turn sends the player directly to Jail ($x_i \leftarrow 10, j_i \leftarrow 1$) and immediately terminates the turn.
   - **Jail Mechanics**: Maximum 3 turns. Options: roll for doubles, pay $50 fine, or use Get Out of Jail Free (GOOJF) card. On turn 3, failing doubles forces $50 payment followed by movement.
   - **Even Building Rule**: Houses and hotels must be built and sold evenly across all properties within a color group (difference $\le 1$).
   - **Mortgages & 10% Interest**: Unimproved properties mortgage for 50% purchase price. Unmortgaging requires paying principal + 10% interest. Mortgaged properties collect no rent.
   - **Dynamic FIFO Card Decks**: 16 Chance and 16 Community Chest cards drawn FIFO from top and returned to bottom, with GOOJF retained until redeemed.

---

## 2. Markov Game Formulation

$$\mathcal{M} = \big\langle \mathcal{N}, \mathcal{S}, \{\mathcal{A}^i\}_{i \in \mathcal{N}}, \mathcal{P}, \{\mathcal{R}^i\}_{i \in \mathcal{N}}, \gamma \big\rangle$$

### Observation Space
- **Decentralized Actor Observation ($s_t^i \in \mathbb{R}^{327}$)**:
  $$s_t^i = \Big[ \mathbf{X}, \; \mathbf{C}, \; \mathbf{J}, \; \mathbf{K}, \; \mathbf{O}, \; \mathbf{H}, \; \mathbf{M}, \; B_h, \; B_{ht}, \; \tau, \; \mathbf{u}, \; \text{OneHot}(i) \Big]$$
- **Centralized Critic Global State ($S_t^{\text{global}} \in \mathbb{R}^{1340}$)**:
  $$S_t^{\text{global}} = \Big[ s_t^1, \; s_t^2, \; s_t^3, \; s_t^4, \; \mathbf{Deck}_{\text{chance}}, \; \mathbf{Deck}_{\text{chest}} \Big]$$

### Factorized Multi-Head Action Space ($\mathcal{A}^i$)
$$\mathcal{A}^i = \mathcal{A}_{\text{core}} \times \mathcal{A}_{\text{auction}} \times \mathcal{A}_{\text{property}} \times \mathcal{A}_{\text{trade}}$$
1. $\mathcal{A}_{\text{core}}$ (6 dims): Roll/Pass, Buy Direct, Decline Buy (Trigger Auction), Pay $50 Jail Fine, Use GOOJF, End Management.
2. $\mathcal{A}_{\text{auction}}$ (5 dims): Fold, Bid +$1, Bid +$10, Bid +$50, Bid +$100.
3. $\mathcal{A}_{\text{property}}$ (160 dims): Build House ($p$), Sell House ($p$), Mortgage ($p$), Unmortgage ($p$) for $p \in \{0, \dots, 39\}$.
4. $\mathcal{A}_{\text{trade}}$: Bilateral negotiation protocol (Target Player, Offer Property, Offer Cash, Request Property, Request Cash, Response).

### Strict Action Masking
$$\pi_{\theta, k}(a_k \mid s_t^i) = \frac{\exp\big(z_k(s_t^i)\big) \cdot M_k^i(s_t^i)}{\sum_{j} \exp\big(z_j(s_t^i)\big) \cdot M_j^i(s_t^i)}$$
Invalid logits receive $-10^8$ prior to softmax.

### Potential-Based Reward Shaping (PBRS)
$$\mathcal{R}'^i(s_t, a_t, s_{t+1}) = \mathcal{R}_{\text{env}}^i + \gamma \Phi^i(s_{t+1}) - \Phi^i(s_t)$$
$$\Phi^i(s) = w_1 \ln(c_i + 1) + w_2 \sum_{p \in \mathcal{O}_i} \Big( V_{\text{base}}(p) + \sum_{l=1}^{h_p} V_{\text{bld}}(p, l) - m_p V_{\text{mtg}}(p) \Big) + w_3 \sum_{g \in \mathcal{G}} \mathbb{I}(g \subseteq \mathcal{O}_i) U_g - w_4 \sum_{j \neq i} \sum_{d=2}^{12} P(D=d) \cdot \text{Rent}_{j \to i}((x_i + d) \bmod 40)$$

---

## 3. Directory Structure

```text
monopoly_marl/
├── engine/
│   ├── __init__.py
│   ├── board_data.py          # 40 tiles, tiered rents, color groups, dice distributions
│   ├── cards_data.py          # 16 Chance & 16 Community Chest cards + FIFO queue manager
│   ├── monopoly_core.py       # Full tournament rules state machine + PBRS
│   └── pettingzoo_wrapper.py  # PettingZoo ParallelEnv wrapper + action mask generator
├── models/
│   ├── __init__.py
│   ├── actor_network.py       # Decentralized Multi-Head Actor Net (Identical Blueprint)
│   ├── critic_network.py      # Centralized Critic Net (Evaluates S_t^global)
│   └── storage.py             # Multi-Agent GAE Rollout Buffer (CTDE)
├── training/
│   ├── __init__.py
│   ├── mappo_trainer.py       # MAPPO policy & centralized value gradient updates
│   └── league_manager.py      # Main Agent vs Exploiter vs Historical Checkpoint Pool
├── gui/
│   ├── __init__.py
│   ├── renderer.py            # Pygame 11x11 board, tokens, houses, hotels, dice
│   └── dashboard.py           # Real-time agent statistics, Net Worth, event log, speed slider
├── main_train.py              # Headless multi-environment training execution
└── main_play_gui.py           # Real-time interactive Pygame GUI
```

---

## 4. Quickstart & Execution

### 1. Run Headless Training
```bash
python main_train.py --num_iterations 50 --rollout_steps 128 --num_envs 4 --checkpoint_interval 10 --device cuda
```

### 2. Launch Real-Time Pygame GUI
```bash
python main_play_gui.py --checkpoint checkpoints/main_iter_00010.pt
```

### 3. Run Invariant & Unit Tests
```bash
python -m pytest tests/test_monopoly.py -v
```

### GUI Interactive Hotkeys:
- `Space`: Pause / Play simulation
- `S`: Single step forward
- `R`: Reset match
- `1 - 5`: Speed presets (1x, 2x, 5x, 20x, 100x)
- `C`: Capture screenshot to `screenshots/`
- `Esc`: Quit
