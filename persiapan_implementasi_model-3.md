# Dokumen Persiapan Teknis & Riset Implementasi Model-3

**Referensi Utama**: [`rancangan_model-3.md`](file:///home/opeteer/monopolyRL/rancangan_model-3.md)  
**Baseline Codebase**: [`model_2/`](file:///home/opeteer/monopolyRL/model_2), [`main_train_model_2.py`](file:///home/opeteer/monopolyRL/main_train_model_2.py)  
**Tujuan**: Menjembatani rancangan konseptual Model-3 menjadi spesifikasi teknik yang siap dikodekan, telah diukur terhadap keterbatasan CPU/RAM/VRAM mesin target, serta mencegah bug pelatihan (seperti *policy collapse*, ledakan gradien, atau *hidden-state drift*).

---

## 1. Hasil Audit & *Micro-Benchmark* Pengembangan (Target Server CPU-Only 24/7)

Karena training produksi dijalankan di **server 24 jam CPU-only (tanpa GPU) dengan CPU *multi-core* kuat dan RAM besar**, dilakukan *benchmark* langsung **pada CPU** (`torch.set_num_threads(8)`) untuk menghindari arsitektur yang cepat di GPU tetapi tercekik di CPU:

| Tolok Ukur (Diukur Langsung) | Hasil Ukur CPU (`B=512` train / `B=128` inf) | Implikasi Desain untuk **Server CPU-Only 24/7** |
|---|---|---|
| `MonopolyCore.step` (termasuk `obs` + `masks`) | **`0.241 ms/step`** (`4,152 steps/detik`/core) | Di server CPU *many-core*, `128–256` worker paralel dapat menghasilkan puluhan hingga ratusan ribu *steps/detik*; tetap pakai mode *fast step* tanpa assembly observasi di simpul Search |
| `copy.deepcopy(MonopolyCore)` | **`0.480 ms/clone`** (`2,084 clones/detik`) | Tetap **tidak layak untuk Search** karena alokasi memori & GC menumpuk di puluhan worker paralel |
| **`fast_snapshot(tuple)` (prototipe)** | **`0.0134 ms/clone`** (`74,600 clones/detik`) | **`35.8x` lebih cepat** dari `deepcopy`; tiap worker CPU sanggup menjalankan **Expectiminimax 11 cabang dadu penuh (`2..12`)** tanpa menghambat rollout |
| `Heavy 2L-Tx(d=128) + GRU(24)` (`687k` params) | `train_b512 = 694.28 ms` \| `inf_b128 = 91.14 ms` | **TERLALU LAMBAT di CPU-only** (~6 menit per update PPO, lambat untuk inference rollout & Search) |
| `Light 1L-Tx(d=64) + Conv1d(24)` (`331k` params) | `train_b512 = 150.12 ms` \| `inf_b128 = 20.33 ms` | `4.6x` lebih cepat dari `Heavy Tx+GRU`, masih punya overhead softmax atensi per langkah |
| **`TileMixer(2L, d=64) + EventMixer(24)` (`412k` params)** | **`train_b512 = 67.34 ms` \| `inf_b128 = 6.16 ms`** | **`10.3x` lebih cepat train & `14.8x` lebih cepat inferensi di CPU!** Memanfaatkan operasi GEMM BLAS padat yang cocok dengan topologi tetap 40 petak Monopoly |
| **`TileMixer(2L, d=96) + EventMixer(24)` (`527k` params)** | **`train_b512 = 103.46 ms` \| `inf_b128 = 9.92 ms`** | Kapasitas lebih kaya (`527k` params), tetap **`6.7x` lebih cepat train & `9.2x` lebih cepat inferensi** di CPU |

---

## 2. Desain Teknis & Cara Merealisasikan 6 Pilar Model-3

### 2.1. *Fast State Cloning* & *Skip-Obs Rollout* (Fondasi Search)
Agar *Stochastic Lookahead Search* dapat berjalan pada laju tinggi tanpa `copy.deepcopy`:
1. **`MonopolyCore.fast_clone()` / `snapshot()`**:
   - Referensi statis (`tile_prices`, `tile_house_costs`, `tile_mortgages`, `house_rules`) dibagikan (*shared immutable*).
   - Hanya salin `26` atribut mutabel (`cash`, `position`, `in_jail`, `jail_turns`, `goojf_cards`, `bankrupt`, `elimination_order`, `property_owner`, `property_houses`, `property_mortgaged`, `deck_manager.chance_queue`, `deck_manager.chest_queue`, `pending_trade`, `rng.getstate()`, dan flag fase).
2. **Mode `fast_search_mode=True`**:
   - Menonaktifkan penulisan string ke `self.event_log` serta melewati perhitungan PBRS & penyusunan `500+` dim observasi di simpul non-daun.
   - Estimasi kecepatan setelah dioptimasi: `~10,000–15,000` simpul pohon/detik/core, sehingga evaluasi `4` kandidat aksi $\times$ kedalaman `2–3` giliran selesai dalam `< 10–15 ms`.

---

### 2.2. *Unconstrained Multi-Property Bundle Trade* + *1x Counter-Offer* Tanpa Meledakkan Action Space
Mengubah trade dari `1-for-1` ([`actor_network.py`](file:///home/opeteer/monopolyRL/model_2/models/actor_network.py#L71-L74)) ke *multi-property bundle* **tanpa batasan maksimum 3 properti** harus menghindari kombinasi eksponensial ($2^{40}$) maupun ledakan permutasi urutan ($k!$), sekaligus menjaga tensor berukuran statis di [`SharedMemoryMonopolyVecEnv`](file:///home/opeteer/monopolyRL/model_2/engine/vec_env.py#L138-L200):

1. **Representasi *Target-Conditioned Multi-Bernoulli Set* (`42`-bit Tetap, Bebas Jumlah Properti)**:
   - Di Monopoly hanya terdapat `28` petak properti (diindeks langsung sesuai papan `0..39`) + maksimal `2` kartu `GOOJF` (`40`, `41`). Maka seluruh himpunan aset berapa pun ukurannya (**`0` hingga `28` properti + `2` `GOOJF`**) cukup direpresentasikan sebagai vektor biner statis `{0, 1}^{42}` bertipe `int8`—bahkan lebih kecil dibanding 3 slot kategorikal!
   - Field baru pada SHM & `RolloutBuffer`:
     - `trade_target`: `3` opsi lawan aktif
     - `trade_req_bundle` & `mask_trade_req_bundle`: `(42,)` `int8` (`1` = sertakan aset target, `0` = abaikan)
     - `trade_offer_bundle` & `mask_trade_offer_bundle`: `(42,)` `int8` (`1` = tawarkan aset milik sendiri, `0` = abaikan)
     - `trade_offer_cash` & `trade_req_cash`: `32` bracket (`[0, 25, 50, 75, 100, 150, 200, ..., 2500]`)
     - `trade_response`: `3` aksi (`0: Reject`, `1: Accept`, `2: Counter`)
2. **Alur Autoregresif Bilateral & *Permutation-Invariant Set Pooling***:
   - Urutan generasi 1-langkah mikro:
     `trade_target -> req_bundle -> offer_bundle -> offer_cash -> req_cash`
   - **Langkah A (`trade_target`)**: memilih 1 lawan aktif yang memiliki minimal 1 aset legal yang bisa diperdagangkan (`unimproved property` atau `GOOJF`). Karena Monopoly bersifat bilateral, mengunci target lebih dulu menjamin seluruh isi `req_bundle` pasti dimiliki oleh target yang sama.
   - **Langkah B (`req_bundle` — *Masked Multi-Bernoulli*)**:
     - `mask_req[i] = 1` hanya untuk aset legal milik `trade_target` (slot `goojf_2` terbuka hanya jika target memegang `2` kartu).
     - Logit tiap item $\ell^{\text{req}}_i$ dihitung oleh *Per-Tile Scoring Head* dari output *Board `TileMixer`* `H_tiles[:, i, :]` yang dikondisikan pada `[trunk_feat, emb_target]`.
     - Probabilitas bersama himpunan $b^{\text{req}} \in \{0,1\}^{42}$:
       $$\log \pi(b^{\text{req}} \mid s, t) = \sum_{i : \text{mask}_i = 1} \Big[ b^{\text{req}}_i \log \sigma(\ell^{\text{req}}_i) + (1 - b^{\text{req}}_i)\log(1 - \sigma(\ell^{\text{req}}_i)) \Big]$$
     - Model bebas memilih `0`, `1`, `4`, `10`, hingga seluruh aset legal target dengan **nol ambiguitas urutan** (`{A, B}` dan `{B, A}` menjadi satu representasi biner unik).
   - **Langkah C (`offer_bundle` — Dikondisikan pada *Set Pooling* `req_bundle`)**:
     - Hitung representasi himpunan yang diminta:
       $$z_{\text{req}} = \sum_{i=0}^{41} b^{\text{req}}_i \cdot E_{\text{item}}(i)$$
     - `head_offer_bundle` membaca `[trunk_feat, emb_target, z_req]` + `H_tiles` dan memberi skor Bernoulli untuk tiap aset legal milik inisiator (`mask_offer[i] = 1`). Dengan melihat $z_{\text{req}}$, model mengetahui total bobot paket yang diminta sebelum memutuskan berapa banyak properti miliknya (`0..semua`) yang layak ditawarkan balik.
   - **Langkah D (`offer_cash` & `req_cash` — Anti-Giveaway & Anti-Dual-Cash Eksak)**:
     - Hitung $z_{\text{offer}} = \sum_{i=0}^{41} b^{\text{offer}}_i \cdot E_{\text{item}}(i)$, lalu kondisikan `offer_cash` dan `req_cash` pada `[trunk_feat, emb_target, z_req, z_offer]`.
     - Misal `has_req_prop = (b_req.sum() > 0)` dan `has_offer_prop = (b_offer.sum() > 0)`:
       - Jika `!has_offer_prop and has_req_prop` $\rightarrow$ wajib `offer_cash > 0` (`mask[0] = 0`).
       - Jika `has_offer_prop and !has_req_prop` $\rightarrow$ wajib `req_cash > 0` (`mask[0] = 0`).
       - Jika `offer_cash > 0` $\rightarrow$ `req_cash` dipaksa `0` (nol *dual-cash*).
       - Jika `!has_offer_prop and !has_req_prop` (kedua bundel kosong) $\rightarrow$ `offer_cash = 0` dan `req_cash = 0`, yang ditangani bersih oleh `MonopolyCore` sebagai *Cancel/No-Op Trade* tanpa mendistorsi *log-prob* Bernoulli di `PPO`.
   - **Kewajiban Anti-Collapse (Pelajaran dari Barracuda Fine)**:
     - Hitung entropi Bernoulli, `KL-to-Uniform(p=0.5)`, dan `_valid_logit_l2` **hanya pada indeks bermask `mask[i] > 0`** saat `is_trade_detail == True` (hindari penalti atas petak invalid/tidak dimiliki agar bobot tidak runtuh atau terdistorsi).
   - *(Opsional saat Search / Evaluasi Lanjut)*: Untuk menangkap sinergi tajam antar-aset dalam satu sisi bundel (mis. komplemen grup warna), `ChanceLookahead` dapat mengevaluasi kombinasi kandidat teratas ($K=4$ bundel hasil *multi-sample* Bernoulli $\pi_\theta$) memakai `QR-Critic` secara penuh.
3. **Protokol *1x Counter-Offer***:
   - Tambah flag `pending_trade["is_counter"]` di `MonopolyCore`.
   - Jika `is_counter == False`, `mask_trade_response = [1, accept_ok, 1]` (target boleh `Reject`, `Accept`, atau `Counter`).
   - Jika target memilih `Counter (2)`, `MonopolyCore` masuk sub-fase `PHASE_TRADE_COUNTER` dengan `trade_target` terkunci ke inisiator awal dan `is_counter = True`. Target bebas menyusun paket balasan dengan **kebebasan jumlah properti yang sama (`{0, 1}^{42}`)**, lalu putaran balasan selanjutnya hanya mengizinkan `Reject (0)` atau `Accept (1)`, sehingga dijamin bebas *infinite loop*.

---

### 2.3. *Board Spatial `TileMixer`*, *Opponent Telemetry*, & Memori Episodik (Ramah CPU & Shuffle PPO)
Dua hambatan utama pada rencana lama (`2L Transformer + 24-step GRU`) di server **CPU-only** adalah: (1) *loop* sekuensial 24 langkah GRU & softmax atensi memakan `694 ms/batch` saat PPO update, dan (2) pengacakan *mini-batch* di [`RolloutBuffer.get_generator`](file:///home/opeteer/monopolyRL/model_2/models/storage.py#L153-L197) memutus kesinambungan *hidden state* $h_t$. Solusi *CPU-native* untuk Model-3:

1. **Observasi Terstruktur Tiga Kanal (Pengganti Flat `500`-dim)**:
   - **Kanal A — `tile_state` `(B, 40, 14)`**: fitur tiap petak relatif terhadap aktor aktif (`owner_rel_onehot[5]`, `houses/5`, `is_mortgaged`, `price/400`, `current_rent/2000`, `players_on_tile_rel[4]`, `completes_group_flag`).
   - **Kanal B — `scalar_telemetry` `(B, ~192)`**: status 4 pemain (relatif terhadap diri sendiri), `bank_houses/32`, `bank_hotels/12`, **rasio `housing_shortage`**, **expected rent inflow vs outflow**, konteks lelang/bundle trade, dan **`3 x 10 = 30` dim Opponent Behavioral Telemetry** (EMA rasio `bid/list`, tingkat penolakan trade per kursi, rata-rata cadangan kas, frekuensi bertahan di penjara).
   - **Kanal C — `event_ring` `(B, 24, 5)` bertipe `int16`**: *sliding window* 24 kejadian terakhir `(event_type, actor_rel, target_rel, tile_or_group, amount_bucket)`.
2. **Arsitektur Neural *BLAS-Friendly & Self-Contained* (`TileMixer + EventMixer`)**:
   - `tile_state` `(B, 40, 14)` $\rightarrow$ Linear(`14 -> d`, default `d=64` atau `96`) + *Tile/Group Positional Embedding* $\rightarrow$ **`2-layer TileMixer`** (*Token-Mixing* `40 -> 40` melintasi petak papan agar jaringan otomatis mengenali jarak dadu `+2..+12` & grup warna + *Channel-Mixing* `d -> 2d -> d`) $\rightarrow$ keluaran per-petak `H_tiles (B, 40, d)` (dipakai langsung oleh *head* `req_bundle` / `offer_bundle`) + `[mean(H), max(H)]`.
   - `event_ring` `(B, 24, 5)` $\rightarrow$ Embedding terstruktur $\rightarrow$ **`EventMixer` / 1D Temporal Mixing** melintasi sumbu `24` langkah $\rightarrow$ ringkasan konteks negosiasi & lelang tanpa *loop* sekuensial GRU.
   - Gabungan `[tile_pool, event_ctx, scalar_enc]` $\rightarrow$ Residual Fusion Trunk (`384` dim) menuju multi-head action heads.
   - **Keuntungan ganda**: **`6.7x–10.3x` lebih cepat training dan `9.2x–14.8x` lebih cepat inferensi di CPU** (`6–10 ms` per *batch* `128`), sekaligus karena `event_ring` tersimpan utuh di setiap observasi (`int16`), [`RolloutBuffer.get_generator`](file:///home/opeteer/monopolyRL/model_2/models/storage.py#L153) dapat tetap melakukan *full random mini-batch shuffling* tanpa *burn-in* GRU atau kebocoran `h_t`!

---

### 2.4. *Distributional Quantile Critic (`QR-Critic`)* + *CVaR Risk-Sensitive MAPPO*
Mengganti scalar `value_head` ([`CentralizedCritic`](file:///home/opeteer/monopolyRL/model_2/models/critic_network.py#L45)) dengan **Quantile Regression Critic** untuk memahami *tail-risk* kebangkrutan mendadak:

1. **Output & Target Kuantil ($N = 32$)**:
   - Output critic: $Z_\phi(S_t) \in \mathbb{R}^{B \times 4 \times 32}$, merepresentasikan nilai pada titik tengah kuantil $\hat{\tau}_i = \frac{i + 0.5}{32}$ ($i = 0 \dots 31$).
   - **1-Step Distributional Bellman Target** untuk setiap transisi $(S_t, r_t, d_t, S_{t+1})$:
     $$y_{j} = r_t + \gamma (1 - d_t) \cdot Z_{\phi^-}(S_{t+1})_j, \quad j \in \{0 \dots 31\}$$
     *(menggunakan soft-target / frozen-epoch critic $Z_{\phi^-}$ dalam mini-batch update agar target stabil).*
2. **Scaled Quantile Huber Loss ($\kappa = 1.0$)**:
   - Mempertahankan normalisasi `scale = max(1.0, self.ret_std)` dari Model-2 (`_ret_std_initialized` warm-start):
     $$\delta_{ij} = \frac{y_j - Z_\phi(S_t)_i}{\text{scale}}$$
     $$\mathcal{L}_{\text{QR}} = \frac{1}{N^2} \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} \left|\hat{\tau}_i - \mathbb{I}(\delta_{ij} < 0)\right| \cdot \text{SmoothL1}_{\kappa=1.0}(\delta_{ij})$$
3. **Risk-Sensitive Value & GAE untuk Actor**:
   - Hitung *risk-adjusted scalar state value* dari 32 kuantil terurut:
     $$V_{\text{risk}}(S_t) = (1 - \lambda_{\text{cvar}}) \cdot \frac{1}{N}\sum_{i=0}^{N-1} Z_{(i)}(S_t) + \lambda_{\text{cvar}} \cdot \frac{1}{K}\sum_{i=0}^{K-1} Z_{(i)}(S_t)$$
     *(dengan $\alpha = 0.25 \Rightarrow K = 8$ kuantil terbawah, dan $\lambda_{\text{cvar}} \approx 0.25\text{–}0.35$).*
   - Nilai $V_{\text{risk}}(S_t)$ dimasukkan ke GAE standar (`gamma=0.997, gae_lambda=0.955`) sehingga kebijakan PPO secara alami menghindari langkah berisiko bangkrut (*over-leveraging* kas di *late game*).

---

### 2.5. *Hybrid Chance-Node Lookahead Search* + *Search Policy Distillation*
Agar melampaui kalkulasi taktis pemain pro tanpa mengorbankan kecepatan latih:

1. **Titik Aktivasi Selektif (`Phase-Gated Search`)**:
   - Hanya dijalankan pada 4 situasi bernilai tinggi (`~10–15%` dari total step):
     1. `PHASE_BUY_OR_AUCTION`
     2. `PHASE_AUCTION` (ketika harga saat ini $\ge 0.5\times$ list atau melengkapi grup/kas ketat)
     3. `PHASE_PROPERTY_MANAGE` (ketika memiliki monopoli penuh, `bank_houses <= 16`, atau kas di bawah ambang bahaya sewa)
     4. `PHASE_TRADE_RESPONSE` & `PHASE_JAIL_DECISION` (ketika lawan punya monopoli berumah $\ge 2$)
2. **Algoritma *Bounded Chance-Node Expectiminimax* (`Top-K = 4`, Horizon `2` giliran penuh)**:
   - Ambil hingga $K = 4$ aksi legal teratas menurut *prior policy* $\pi_\theta(a|s)$.
   - Untuk tiap kandidat $a_k$, lakukan `fast_snapshot` $\rightarrow$ eksekusi $a_k$ $\rightarrow$ cabang dadu terstratifikasi (*11 hasil dadu `2..12` dibobot `DICE_PROBABILITIES`* atau `6` sampel representatif untuk menjaga kecepatan), jalankan aksi lawan secara *greedy/batched* memakai $\pi_\theta$, dan evaluasi simpul daun menggunakan $V_{\text{risk}}(S_{\text{leaf}})$ dari `QR-Critic`.
   - Hasilkan $Q_{\text{search}}(s, a_k)$ dan distribusi target $\pi_{\text{search}}(a|s) = \text{softmax}(Q_{\text{search}}(s, a) / \tau_{\text{search}})$.
3. **Integrasi Training (*Worker Distillation Loss*)**:
   - Karena target eksekusi adalah **server ber-CPU kuat dengan RAM besar**, rasio environment bertag `search_guided` dapat dinaikkan dari `15–20%` menjadi **`30–50%` worker** dengan **11 cabang dadu eksak (`2..12`)** (`Top-K = 4..6`, kedalaman `2–3` giliran).
   - Worker `search_guided` melakukan `fast_snapshot` di proses CPU paralel masing-masing + *batched leaf evaluation* (`QR-Critic`) melalui SharedMemory / *async inference queue*:
     - Simpan $\pi_{\text{search}}$ (Top-$K$ sparse vector `float32`) ke buffer.
     - Tambah objektif distilasi pada [`MAPPOTrainer`](file:///home/opeteer/monopolyRL/model_2/training/mappo_trainer.py#L120-L125):
       $$\mathcal{L}_{\text{actor}} = \mathcal{L}_{\text{PPO}} - c_{\text{ent}} H(\pi_\theta) + c_{\text{kl}} \text{KL}_{\text{unif}} + c_{\text{l2}} \|\text{logits}\|^2 + \lambda_{\text{distill}} \cdot D_{\text{KL}}(\pi_{\text{search}} \,\|\, \pi_\theta)$$
   - **Manfaat ganda**: Saat *inference* cepat (`no search`), agen sudah mendistilasi pola *4-house lock* & *jail stalling*; saat *tournament/pro mode* (`--use_search`), agen bisa menyalakan Search langsung untuk presisi maksimal.

---

### 2.6. Perbaikan PBRS v3 & Liga PSRO Heterogen
1. **Tambah Dua Suku Potensial PBRS di `MonopolyCore` (Anti-*Instant-Mortgage* & Pro-*House-Lock*)**:
   - **`Term 9 — Expected Rent Receivable`**:
     $$\Phi_{\text{income}}^i(s) = w_{\text{income}} \sum_{j \neq i, \neg \text{jail}_j} \sum_{d=2}^{12} P(D=d) \cdot \text{Rent}_{i \to j}((x_j + d) \bmod 40)$$
     Karena properti yang digadai memiliki `Rent = 0`, menggadai properti produktif saat kas masih aman (`c_i >= $400`) langsung menurunkan potensi bersih lebih besar daripada keuntungan log kas `Term 1`.
   - **`Term 10 — Housing Shortage Lock Control`**:
     Saat `bank_houses <= 16` dan minimal satu lawan memiliki monopoli street dengan rumah $< 4$, berikan bonus kecil proporsional ke rumah non-hotel terpasang (`my_street_houses_under_hotel / 32.0`) agar agen mempelajari taktik menahan upgrade ke hotel demi memblokir pembangunan lawan.
   - **PBRS Horizon Annealing**: Skalakan bobot shaping total `pbrs_scale` dari `1.0` $\rightarrow$ `0.30` seiring kemajuan iterasi.
2. **`ProHeuristicBot` (Sparring & Benchmark Pro Tanpa Hardcoding ke Agen RL)**:
   - Membuat modul terpisah `model_3/agents/pro_heuristic_bot.py` yang mengeksekusi aturan standar turnamen manusia (cadangan kas dinamis berdasarkan sewa puncak papan, *zero unnecessary mortgage*, *4-house lock* saat `bank_houses` menipis, *late-game jail stalling*, dan valuasi trade berbasis monopoli).
   - Dipakai dalam komposisi **Liga PSRO**: `55%` Self-Play/Historical + `20%` Diverse Exploiters + `15%` ProHeuristicBot / Baseline Diversifier (termasuk *Auction Overbidder* agar agen ingat cara mengeksploitasi lawan *overbid*) + `10%` Search-Guided Champion.

---

## 3. Arsitektur Eksekusi Server 24/7 **CPU-Only** (RAM Besar, Multi-Core Tinggi)

Karena server training bersifat **CPU-only tanpa GPU**, seluruh desain eksekusi dioptimalkan untuk throughput CPU & RAM sistem besar tanpa ketergantungan CUDA:

### 3.1. Profil Skala & Disiplin Threading CPU
- **Aturan Emas Anti-Thread-Oversubscription (Wajib di Server CPU Many-Core)**:
  - Proses worker `SharedMemoryMonopolyVecEnv` wajib mengeset `torch.set_num_threads(1)`, `OMP_NUM_THREADS=1`, dan `MKL_NUM_THREADS=1` agar `64–128` worker tidak saling berebut *thread pool* BLAS.
  - Proses utama (`MAPPOTrainer` untuk inference batch & backward PPO) memakai `torch.set_num_threads(physical_cores)` dan `torch.inference_mode()` penuh saat *rollout* serta *search leaf evaluation*.
- **Profil Server Rekomendasi (`num_envs=64..128`, `rollout_steps=512..1024` $\Rightarrow$ `32k–131k` langkah/iterasi)**:
  - **SharedMemory RAM**: `~2–4 MB` (`tile_state`, `scalar_telemetry`, `event_ring`, `masks`, dan antrean evaluasi *leaf search*).
  - **`RolloutBuffer` di CPU RAM** (nol biaya transfer CPU $\leftrightarrow$ GPU!):
    - Untuk `65,536` transisi (`num_envs=128, rollout_steps=512`): `tile_state` (`70 MB`) + `scalar_telemetry` (`50 MB`) + `event_ring` (`16 MB`) + `global_state` (`~240 MB`) + `quantiles/actions/masks/distill` (`~180 MB`) = hanya **`~556 MB` RAM** (atau `~1.1–2.2 GB` pada `rollout_steps=1024`, sangat ringan di server ber-RAM besar).
  - **Tuning PPO Ramah CPU-Only**:
    - Gunakan arsitektur default **`TileMixer(2L, d=64..96) + EventMixer(24)`** (`67–103 ms` per *batch* `B=512`, `6–10 ms` per *inference batch* `B=128`) alih-alih Transformer+GRU yang `~7–10x` lebih lambat di CPU.
    - Mini-batch optimal **`B=512..1024`** dengan **`ppo_epoch=3`** dan *early KL stopping*, sehingga satu iterasi PPO update (`65k` steps $\rightarrow$ `~190–380` update batch) selesai dalam **~15–35 detik** pada CPU modern—cocok untuk dijalankan 24 jam nonstop!

### 3.2. Fitur Keandalan Operasi Server 24 Jam (*Unattended Training*)
1. **Auto-Resume & Rolling Checkpoints**: simpan `latest_checkpoint.pt` + snapshot periodik tiap `N` iterasi beserta state lengkap optimizer, scheduler, `ret_std`, `pbrs_scale`, serta komposisi liga PSRO; tangani `SIGINT`/`SIGTERM` dengan penyimpanan darurat agar progres berjam-jam tidak pernah hilang.
2. **Evaluator Tournament Async / Berkala**: tiap `K` juta steps, jalankan *benchmark gate* otomatis melawan `Model-2_Barracuda.pt` dan `ProHeuristicBot`, lalu log metrik Elo Plackett-Luce, `BuyRate`, *mortgage-ratio*, dan `4-house lock rate` ke CSV/log persisten.
3. **Konfigurasi Fleksibel (`--num_envs`, `--search_worker_ratio`, `--num_threads`, `--arch tile_mixer|light_tx`)**: satu *codebase* `model_3/` yang ringan diuji di laptop (`num_envs=8..16`) lalu langsung di-scale di server 24 jam.

---

## 4. Urutan Eksekusi Implementasi & Checklist Kesiapan

1. **Langkah 1 — Paket `model_3/` Terpisah (Non-Destruktif terhadap `model_2/`)**:
   - Biarkan `model_2/` & `checkpoints_model_2/Model-2_Barracuda.pt` tetap utuh sebagai baseline & sparring benchmark.
   - Struktur folder baru:
     - `model_3/engine/` (`monopoly_core.py` v3 dengan `fast_snapshot`/`restore_snapshot`, *bundle trade*, *counter-offer*, `event_ring`, `telemetry`, & PBRS v3; `vec_env.py` dengan multi-channel SHM)
     - `model_3/models/` (`actor_network.py` `TileMixer`+`EventMixer`+`BundleHeads`; `critic_network.py` `QR-Critic`; `storage.py`)
     - `model_3/search/` (`chance_lookahead.py` Expectiminimax / Chance-Node Search + `ProHeuristicBot` di `model_3/agents/`)
     - `model_3/training/` (`mappo_trainer.py` QR-Huber + CVaR-GAE + KL Search Distillation; `league_manager.py` PSRO Mix)
2. **Langkah 2 — Unit Test & Smoke Test Wajib Sebelum Training Besar**:
   - `test_fast_snapshot_equivalence`: memverifikasi `fast_snapshot` + `restore_snapshot` + `step()` menghasilkan state & event yang **100% deterministik identik** dengan eksekusi tanpa snapshot.
   - `test_bundle_trade_and_counter_masks`: memverifikasi target selalu tunggal & sah, `req_bundle`/`offer_bundle` dapat memilih `0..seluruh` properti legal milik masing-masing pihak (`{0,1}^{42}`), nol *dual-cash*, nol giveaway sepihak, bundel kosong ditangani bersih sebagai no-op, serta *1x counter-offer* tidak pernah masuk infinite loop.
   - `test_conditioned_logit_l2_stability`: memverifikasi `logit_l2` dan norm gradien seluruh bundle heads (`< 0.05`) bebas ledakan `-1e8`.
   - `test_qr_critic_cvar_and_search_ev`: memverifikasi *Quantile Huber Loss* terkonvergensi dan `ChanceLookahead` memilih *4-house lock* pada skenario sintetis kekurangan rumah bank.
3. **Langkah 3 — Pelatihan Bertahap & Gerbang Evaluasi**:
   - Jalankan fase awal 25M–50M steps dengan `PBRS v3 + Telemetry + Bundle Trade + QR-Critic`, bandingkan langsung dengan [`Model-2_Barracuda.pt`](file:///home/opeteer/monopolyRL/checkpoints_model_2/Model-2_Barracuda.pt) dan `ProHeuristicBot` di suite uji Plackett-Luce Elo sebelum mengaktifkan *Search Distillation* & PSRO skala penuh.
