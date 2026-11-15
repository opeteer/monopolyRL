# Rancangan Arsitektur & Peningkatan Model-3 Monopoly MARL (Menuju Level Pro-Turnamen & Superhuman)

Dokumen ini merangkum seluruh rancangan peningkatan (**Model-3**) yang disusun berdasarkan temuan empiris pengujian **Model-2 Barracuda (`180M`, `1290.3` Elo)**—termasuk hasil **152 pertandingan benchmark CUDA**, kalibrasi **300 game Plackett-Luce Elo**, audit bobot jaringan, serta analisis log *4x mirror* ([`barracuda_180m_4way_seed9002.log`](file:///home/opeteer/.gemini/antigravity/brain/090c70b9-cff1-4d78-aa8e-2dbad6f4ffb6/scratch/barracuda_180m_4way_seed9002.log)).

---

## 1. Diagnosis Kekurangan Model-2 (Akar Masalah Menuju Level Pro)

Berdasarkan inspeksi langsung atas [`model_2/`](file:///home/opeteer/monopolyRL/model_2) dan perilaku aktual `Model-2_Barracuda.pt`:

1. **Kebiasaan *Instant-Mortgage* saat Kas Melimpah**:
   - Di log `seed 9002`, agen rutin membeli properti langsung lalu **detik yang sama menggadaikannya** meski kas `>$1,000` (bahkan menggadai `North Carolina` lalu langsung menebus `Ventnor` dengan penalti bunga `10%` pada giliran yang sama).
   - **Akar Masalah di Kode**: Pada [`MonopolyCore._compute_potential`](file:///home/opeteer/monopolyRL/model_2/engine/monopoly_core.py#L346-L430), `term4` hanya menghitung risiko sewa yang **dibayar ke lawan** (`owner != player_id`), tetapi **tidak memodelkan potensi pendapatan sewa yang hilang** ketika properti milik sendiri digadai (`0` sewa). Di saat yang sama, kas hasil gadai meningkatkan `term1` (`w1 * ln(c+1) + w_cash * c`), sehingga *instant-mortgage* terlihat menguntungkan di jangka pendek.
2. **Trade Terkunci pada Format *1-for-1 Single Property***:
   - Pada [`ActorNetwork`](file:///home/opeteer/monopolyRL/model_2/models/actor_network.py#L71-L74) dan [`monopoly_core.py`](file:///home/opeteer/monopolyRL/model_2/engine/monopoly_core.py#L58-L62), head trade hanya mendukung **1 properti ditawarkan** (`41` dim) untuk **1 properti diminta** (`41` dim) + kas kelipatan `$100` (`$0..$2,000`).
   - Pemain pro hampir selalu menegosiasikan **paket asimetris** (*2 properti + kas* untuk *1 monopoli*, barter kartu *GOOJF*, atau eskalasi negosiasi bertahap). Batasan ini menyebabkan fenomena *underpricing* properti pengunci monopoli (seperti penjualan tunai `$200`) atau *ping-pong* properti 1-lawan-1.
3. **Arsitektur *Memoryless* (MLP `obs_dim=500`) Tanpa Pemodelan Lawan**:
   - [`ActorNetwork`](file:///home/opeteer/monopolyRL/model_2/models/actor_network.py#L30-L53) hanya memproses vektor keadaan sesaat `s_t^i` (`500` dim) tanpa memori historis.
   - Agen tidak dapat mengenali reputasi atau pola spesifik tiap lawan di meja—terbukti di `Suite D`, ketika `Barracuda_180M` menaikkan `BuyRate` menjadi `82.1%` menghadapi `3x M1-100M` (yang memiliki bias *overbid* `2.1x–2.2x` harga daftar) alih-alih memancing lawan membangkrutkan diri di lelang (`66.7%` win rate pada `150M` vs `25.0%` pada `180M`).
4. **Kebijakan *Model-Free* Reaktif Tanpa *Lookahead Search***:
   - Tidak ada simulasi probabilistik (`DICE_PROBABILITIES`) maupun pohon keputusan untuk menghitung taktik pro berdaya ungkit tinggi:
     - ***32-House Housing Shortage Lock***: sengaja berhenti membangun di **4 rumah** (menolak upgrade ke hotel) agar stok `bank_houses = 32` habis dan lawan tidak bisa membangun rumah sama sekali.
     - ***Late-Game Strategic Jail Stalling***: menghitung probabilitas risiko kematian 2–3 giliran ke depan untuk memilih bertahan di penjara ketimbang membayar `$50`.
     - **Valuasi EV Trade**: menghitung selisih probabilitas menang (*win-probability delta*) sebelum menerima atau mengajukan trade.
5. **Homogenitas Liga *Self-Play***:
   - *Fine-tuning* hanya melawan varian `Barracuda` dan satu `MainExploiter` memicu *self-play specialization*—unggul `18–14` (`+28.4%` NW) melawan `150M`, tetapi rentan saat bertemu strategi *out-of-distribution*.

---

## 2. Matriks Perubahan Lengkap: Model-2 $\rightarrow$ Model-3

| Komponen | Model-2 (Saat Ini) | Upgrade pada Model-3 | Tujuan & Dampak Strategis |
|---|---|---|---|
| **Engine & Trade Protocol** | `1-for-1` properti (`41 x 41`), kas kelipatan `$100` (`$0..$2,000`), langsung terima/tolak | **Multi-Property Bundle Trade** (hingga `3-for-3` + kartu `GOOJF`), kas `$25/$50` s.d. `$2,500`, + opsi **1x Counter-Offer** | Membuka negosiasi pro (*2-for-1 + kas*), mencegah penjualan properti monopoli murah |
| **Observasi Actor** | Flat `500` dim *stateless* (`prefix_314 + id + suffix_9 + syn_64 + auc + trade_101`) | **Structured Board + Opponent Telemetry + Bank Lock Features** (`~640` dim) + **Riwayat 24 Event Terakhir** | Sadar stok `32 rumah`, zona bahaya lawan 2–12 petak, dan kecenderungan bid/trade tiap kursi |
| **Arsitektur Actor** | MLP `500 -> 384` + `1 ResBlock` + `trade_resp_branch` (`SiLU`) | **Board-Entity Attention + Opponent Transformer/GRU Encoder + Factorized Bundle Heads** | Invarian permutasi warna/kursi, memori negosiasi, dan adaptasi lawan *real-time* |
| **Centralized Critic** | Scalar Value `V^i(S_global)` (`2032 -> 640 -> 4`, Huber Loss) | **Distributional Quantile Critic (`QR-Critic`, `32` kuantil $\times 4$ agen)** | Mengukur *tail-risk* (CVaR) kebangkrutan akibat sewa besar, bukan sekadar rata-rata |
| **PBRS & Reward Shaping** | `8` term (`w1=0.025, w2=0.0034, w_cash=0.0034, w4=0.0006`, dll.) tanpa sewa masuk | Tambah **Expected Rent Income** (hanya properti tak digadai) + **House-Lock Scarcity**; *annealing* bobot shaping di fase lanjut | Menghilangkan *instant-mortgage* saat kas aman & menjaga fokus kemenangan murni (`+10/-10`) |
| **Inference & Decision** | Pure *1-pass* `Categorical` sampling ($\tau = 0.65$) | **Hybrid Neural + Bounded Chance-Node Search (Expectiminimax/MCTS ringan)** pada `Buy/Auction`, `Manage`, `TradeResp` | Menguasai *4-house lock*, *jail stalling*, bid batas EV tepat, dan menolak trade merugikan |
| **Algoritma Training** | MAPPO + Uniform-KL + `conditioned logit_l2` + `ret_std` warm-start | **Risk-Sensitive PPO + Search Policy Distillation (`KL` ke target Search) + Quantile Huber Loss** | Menggabungkan stabilitas gradien Model-2 dengan akurasi taktis hasil *lookahead search* |
| **Liga & Evaluasi** | `Main + MainExploiter + Historical Pool` (homogen di *fine-tune*) | **PSRO Multi-Arketipe Liga** (Self-Play + Pro-Heuristic Bot + Auction-Bait + House-Locker + Search Bot) | Tahan terhadap *overbidder*, pemain pasif, maupun pemain pro (*anti-overfitting*) |

---

## 3. Rancangan Detail Upgrade Model-3

### 3.1. Engine & Action Space Pro (`MonopolyCore v3`)
Tetap mempertahankan aturan turnamen resmi 100% tanpa aturan *hardcoded* (sejalan dengan prinsip Model-2), namun memperkaya protokol transaksi:
1. **Multi-Property Bundle Trade (Sequential Set Action Space)**:
   - Di fase `PHASE_PROPERTY_MANAGE`, bila agen memilih membuat proposal trade:
     - `trade_target`: 3 opsi lawan aktif.
     - `offer_bundle`: pemilihan sekuensial **0 hingga 3 properti** milik sendiri + flag **0–2 kartu GOOJF**, diakhiri token `DONE` (maks 4 langkah internal mikro tanpa membebani langkah dadu).
     - `req_bundle`: pemilihan sekuensial **0 hingga 3 properti** milik target + flag **0–2 kartu GOOJF**.
     - `offer_cash` & `req_cash`: tetap menjaga **aturan mutual-exclusion dari Model-2** (nol dual-cash, nol giveaway sepihak) dengan resolusi lebih rapat: `[0, 25, 50, 75, 100, 150, 200, ..., 2500]` (`~32` bracket).
   - **Protokol Counter-Offer (Maks 1 Balasan)**: Target dapat memilih `Reject (0)`, `Accept (1)`, atau `Counter (2)` (terbatasi 1 balasan per giliran agar bebas *infinite loop*). Ini mereplikasi tawar-menawar pro yang sesungguhnya.

### 3.2. Fitur Observasi Baru & *Opponent Telemetry*
Memperluas vektor observasi [`_build_shared_obs_parts`](file:///home/opeteer/monopolyRL/model_2/engine/monopoly_core.py#L1033-L1069) + [`_assemble_actor_obs`](file:///home/opeteer/monopolyRL/model_2/engine/monopoly_core.py#L1071-L1082):
1. **Bank & *Housing Shortage* Awareness (`+24` dim)**:
   - Rasio rumah tersisa terhadap total kebutuhan upgrade semua monopoli yang ada (`bank_houses / max(1, total_needed_for_4_houses)`).
   - Indikator per grup warna: apakah upgrade langsung ke hotel akan **membuka kembali rumah untuk lawan** (kunci utama *4-house lock*).
2. **Expected Rent Income & *Danger Zone* (`+32` dim)**:
   - Ekspektasi pendapatan sewa per giliran dari seluruh lawan aktif berdasarkan posisi mereka + distribusi dadu `2..12` (*next-roll expected inflow*).
   - Jumlah kas likuid aman (*cash buffer*) dibanding sewa tertinggi di papan saat ini (`cash_i / max(100, max_opp_rent)`).
3. **Per-Opponent Online Behavioral Summary (`3 lawan x 10 fitur = +30` dim)**:
   - Rata-rata rasio `bid / list_price` tiap lawan sepanjang game berjalan (mendeteksi *overbidder* seperti `M1-100M` vs *value sniper* seperti `Barracuda`).
   - Tingkat penolakan trade, frekuensi *cash-reserve* rata-rata, dan frekuensi *jail-stay*.
4. **Ring-Buffer Riwayat Event Episodik (`L = 24` token)**:
   - Setiap event penting (pembelian, pemenang & harga lelang, proposal trade + hasil `Accepted/Rejected`, pembangunan rumah, masuk penjara) di-encode ke token kompak (`event_type, actor_id, tile_or_group, cash_ratio`) untuk diproses oleh encoder sekuensial.

### 3.3. Arsitektur Jaringan Neural (`ActorNetwork v3` & `QR-Critic v3`)
1. **Board & Entity Transformer + History GRU**:
   - Memecah input `40` petak menjadi token petak (*tile embeddings* memuat jenis, pemilik, jumlah rumah, status gadai, posisi pemain di petak) yang diproses oleh **2-layer Self-Attention (Pre-LN, 4 heads)** sehingga jaringan memahami korelasi spasial dadu (`+6/+7/+8` dari penjara) dan sinergi grup warna secara alami.
   - Memadukan *board representation* dengan **GRU (`hidden=192`)** atau **Causal Attention** atas 24 event terakhir (membawa state memori di `RolloutBuffer` dengan truncated BPTT).
2. **Autoregressive Bundle Trade Heads**:
   - Mempertahankan pola *conditioned masking* & `emb_trade_*` dari [`ActorNetwork`](file:///home/opeteer/monopolyRL/model_2/models/actor_network.py#L77-L85), diperluas dengan **Pointer / Mask-Filtered Sequential Head** (`step 1..3 + STOP`) untuk `offer_bundle` dan `req_bundle` sehingga tidak pernah memilih properti yang sama dua kali atau memicu *dual-cash*.
   - **Wajib mempertahankan perbaikan Model-2**: [`_valid_logit_l2`](file:///home/opeteer/monopolyRL/model_2/models/actor_network.py) yang memfilter `logits > -1e4` dan mengevaluasi mask kondisional aktual agar gradien tetap terkendali (`~1e-3`).
3. **Distributional Quantile Critic (`QR-Critic`)**:
   - Mengganti `value_head(features) -> (B, 4)` pada [`CentralizedCritic`](file:///home/opeteer/monopolyRL/model_2/models/critic_network.py#L45) menjadi `quantile_head(features) -> (B, 4, N_quantiles)` (misal $N = 32$ kuantil dengan **Quantile Huber Loss**).
   - **Keunggulan Pro**: Menangkap distribusi bimodal Monopoly (menang besar vs bangkrut mendadak karena salah langkah di *Boardwalk Hotel*). Advantage dihitung memakai kombinasi **Mean + CVaR ($\alpha=0.25$)** agar agen paham bahwa menjaga bantalan kas saat bahaya tinggi jauh lebih bernilai daripada selisih EV kecil.

### 3.4. *Hybrid Stochastic Lookahead Search* (Menembus Batas Manusia)
Agar benar-benar melampaui kalkulasi manusia di papan yang kompleks:
1. **Kapan Search Dijalankan**: Hanya pada keputusan bernilai tinggi (*low-frequency, high-leverage*):
   - `PHASE_BUY_OR_AUCTION` (Beli vs Lelang — terutama jika kas menipis atau lawan punya *overbid* tinggi).
   - `PHASE_AUCTION` (menentukan *reservation price* maksimum berdasarkan EV jika properti jatuh ke diri sendiri vs ke lawan tertinggi).
   - `PHASE_PROPERTY_MANAGE` (simulasi 2–3 giliran ke depan: memilih *build 4 houses vs hotel* saat `bank_houses <= 12`, dan memilih *tebus gadai* vs simpan kas).
   - `PHASE_TRADE_RESPONSE` & `PHASE_JAIL_DECISION` (bayar `$50` vs tunggu *doubles*).
2. **Mekanisme Search (*Fast Leaf-Evaluated Chance Rollout*)**:
   - Untuk maksimal `K` kandidat aksi teratas dari *prior policy* $\pi_\theta(a|s)$ (`K = 3..5`, menggunakan `MonopolyCore` ringan tanpa logging), lakukan **Chance-Node Expectiminimax / PUCT kedalaman `2–4` giliran** dengan distribusi `DICE_PROBABILITIES` dan mengevaluasi *leaf state* memakai `QR-Critic` / *Policy-Value rollout*.
   - **Saat Training (*Search Distillation*)**: Jalankan *search* di sebagian worker/rollout (`20–30%` batch atau *dagger-style refinement*), lalu latih actor agar distribusi $\pi_\theta$ meniru kebijakan hasil *search* $\pi_{\text{search}}$ lewat tambahan loss $\lambda_{\text{distill}} \cdot D_{\text{KL}}(\pi_{\text{search}} \,\|\, \pi_\theta)$ di samping PPO. Dengan cara ini, **bahkan tanpa *search* penuh saat mode cepat, jaringan sarafnya sudah terdistilasi oleh taktik lookahead**!

### 3.5. Pembaruan PBRS & Algoritma Pembelajaran
1. **Perbaikan Fungsi Potensial $\Phi^i(s)$ ([`monopoly_core.py`](file:///home/opeteer/monopolyRL/model_2/engine/monopoly_core.py#L346-L430))**:
   - **Tambah `Term 9 — Expected Rent Income Receivable`**:
     $$\text{Income}_i = \sum_{j \neq i} \sum_{d=2}^{12} P(D=d) \cdot \text{Rent}_{i \to j}((x_j + d)\bmod 40)$$
     Dengan bobot `w_income > 0` yang proporsional terhadap `w4`. Karena properti yang digadai menghasilkan `Rent = 0`, menggadai properti produktif ketika kas sudah cukup aman **langsung menurunkan $\Phi^i(s)$**, sehingga menghentikan kebiasaan *instant-mortgage*!
   - **Tambah `Term 10 — Bank House Scarcity Control`**:
     Saat `bank_houses <= 12`, berikan penghargaan kecil berbasis selisih rumah terpasang milik sendiri (`my_houses / 32.0`) bila lawan memiliki monopoli yang belum terbangun (*incentivizes 4-house shortage lock*).
   - **PBRS Horizon Annealing**: Secara bertahap turunkan skala koefisien PBRS pada tahap akhir pelatihan (misal dari `1.0x` ke `0.35x`) dan gunakan $\gamma = 0.997$ agar terminal *sole-survivor win* (`+10.0 / -10.0`) menjadi sinyal utama di *late-game*.
2. **Stabilitas Optimasi dari Model-2 yang Wajib Dipertahankan**:
   - Pertahankan **`ret_std` warm-start** di [`MAPPOTrainer`](file:///home/opeteer/monopolyRL/model_2/training/mappo_trainer.py#L55-L216), *conditioned* `logit_l2`, `clip_param = 0.12–0.15`, *decision-only KL stopping*, dan *pinned `Historical_00000` anchor* saat *fine-tuning*.

### 3.6. Liga Pelatihan PSRO Heterogen
Untuk mengatasi masalah *overfitting* terhadap gaya *self-play* sendiri:
1. **Campuran Lawan Selama Training (`PSRO Mix`)**:
   - **55%** *Self-Play & Historical Snapshots* (dengan PFSP terkalibrasi).
   - **20%** *Specialized Exploiters* (1 agen *Auction Shark* yang memanfaatkan lawan pasif, 1 agen *House-Locker* agresif).
   - **15%** *Baseline Diversifier Agents* (termasuk *Oracle/Pro-Heuristic Bot* yang mempraktikkan *zero-unnecessary-mortgage*, *3-house EV cutoff*, & *4-house lock*, serta snapshot Model-1 / `Barracuda` lama agar agen tetap hafal cara mengeksploitasi *overbidder*).
   - **10%** *Search-Augmented Champion* (versi guru dengan MCTS aktif).
2. **Protokol Evaluasi Anti-Bias**:
   - Kenaikan checkpoint utama (`Champion Gate`) wajib lulus **dua gerbang** menggunakan Plackett-Luce Elo:
     1. Menang signifikan melawan **Champion sebelumnya** (`+20` Elo di >= 48 game *head-to-head*).
     2. Tidak turun melawan **Gauntlet Heterogen** (*Pro-Heuristic Bot* + `M2-Barracuda` + `M2-Impala` + `M1-100M` + `M1-50M`, minimal 100 game).

---

## 4. Roadmap Implementasi Model-3 (Bertahap & Terukur)

| Fase | Lingkup Pekerjaan | Kriteria Keberhasilan (Exit Gate) |
|---|---|---|
| **Fase 1: PBRS v3 & Telemetry (Low-Risk Upgrade)** | Tambah `Expected Rent Income`, `Housing Scarcity Control`, `Opponent Telemetry (+30 dim)`, dan *GRU History* di atas engine Model-2 | *Instant-mortgage rate* saat kas `>$600` turun `<5%`; menang >= 60% melawan `Barracuda 180M` & >= 70% vs `3x M1-100M` |
| **Fase 2: Bundle Trade & QR-Critic** | Implementasi *multi-property bundle trade (up to 3-for-3 + GOOJF)*, *1x counter-offer*, dan `QR-Critic (32 quantiles)` dengan CVaR advantage | Trade *2-for-1* muncul alami tanpa *1-way giveaway*; `ForcedLiq` terkendali; stabilitas pelatihan tanpa *policy collapse* |
| **Fase 3: Search Distillation & PSRO League** | Tambahkan *Chance-Node Expectiminimax/PUCT Search* untuk fase kritis + *Search Distillation* + Liga PSRO dengan *Pro-Heuristic Bot* | Agen konsisten mengeksekusi *4-house lock* saat `bank_houses` tipis & *late-game jail stalling*; kalahkan *Pro-Heuristic Bot* `>= 65%` |
| **Fase 4: Uji Turnamen vs Pemain Pro Manusia** | *Blind human-vs-AI match* dengan UI Model-3 (log transaksi lengkap, *win-prob telemetry*, audit Plackett-Luce Elo) | Plackett-Luce Elo menembus **`>= 1450+`** (anchor `M1-50M=1200`), rasio kemenangan `>35–40%` di meja 4-pemain melawan pemain turnamen pro |
