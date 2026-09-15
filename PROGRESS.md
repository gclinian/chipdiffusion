# Progress Tracker

## Step 12: 新方法/架構 survey → 實驗 (2026-09-13)
- [x] 12.0 環境修復：`chipdiff-b` = torch 2.11.0+cu128 / torchvision 0.26 / PyG 2.8 / numpy 1.23.5，
  sm_120 kernel 可跑，smoke eval 23 s 通過，training path (autocast+GradScaler) 通過。`chipdiff` 未動。
  用法：`PYTHONPATH=. /ibmnas/427/r115/gclin/miniforge3/envs/chipdiff-b/bin/python diffusion/eval.py ...`
  **⚠ 漂移訊號**：adaptec1 baseline 新 stack = 9.176 ± 0.018 (n=3) vs 舊紀錄 10.22 (n=1, 原始檔已遺失)。
  新值貼近 paper 的 9.19；舊值才是離群。**整張 48.691 baseline 表要在新 stack 重量。**
  另：eval 在固定 seed 下不完全 deterministic（GPU scatter atomics），full settings 下 σ≈0.02 可忽略，
  但縮減步數的 smoke config 下膨脹到 ~1.8% — 不要用縮減設定做比較。
- [x] 12.1 Survey → `docs/survey/method_survey_1.md`（6 主題 × Opus web search + skeptic pass）。
  結論：無新生成範式是好賭注；drifting = 一步生成器（Kaiming He 2602.04770），size-OOD 更糟，不做；
  FM 報告的「no-guidance 就輸」背後零個 run，要補對照。領域 SOTA = OrderPlace (ICML'26) 29.98 vs paper 31.27。
- [ ] 12.2 升級後驗證 — **In progress**
  - [x] paper baseline 7 circuits seed 300（`base_cu128_s300_part{1,2}`）= **45.987**，舊紀錄 48.691，
    **paper 46.89** → paper 自己的 checkpoint 在我們環境現在**贏 paper 1.9%**。
    逐 circuit：adaptec1 9.03 / adaptec2 30.75 (舊 39.06!) / adaptec3 55.98 / adaptec4 57.40 /
    bigblue1 2.63 / bigblue3 36.59 / bigblue4 129.53。gen time 25 min（舊卡 63 min；5090 快 2.5×）。
    判讀：舊 3 月 baseline 是壞掉的 run（目錄已被碰撞覆蓋、無法驗 config、runtime 4.5× 慢、
    adaptec2 legality 0.93），不是 stack 漂移 — 其他舊 stack run 在同 checkpoint 上的 adaptec1
    落在 8.8–9.4，與新值 9.03 一致。
    **後果**：所有「−X% vs 48.691」的說法都灌水了。SVDD 44.845 vs 45.987 = Δ1.14 < 1.3（2σ）
    → SVDD 相對 paper 方法在同環境的優勢**尚未建立**。等 Run F 判定舊 stack 數字是否可信。
  - [x] Run F 新 stack seed 300 = **45.700**（舊 44.321，Δ=+1.379 ≥ 1.3 → **預登記 gate 觸發：舊 stack
    數字不可與新 stack 混比**）。位移幾乎全在 bigblue3 +8.85 / bigblue4 +7.52（σ 最大的兩個 circuit）
    → 換 GPU/torch 等於換 random stream，是 seed 量級的變動，不是系統性偏移。
    **⚠ 同 seed 同 stack：微調 45.700 vs paper ckpt 45.987 = Δ−0.29，在雜訊帶內。** 專案標題
    「微調贏 9.6%」是對壞掉的 baseline 量的。n=1，等 seed 301/302 再下結論。
  - [x] 0c baseline 3 seeds：avg6 **31.212 ± 0.813**（2σ = 1.63；adaptec4/bigblue3 per-circuit sd ~5.5%）
  - [x] **re-anchor 完成（paired, n=3, avg6）：微調 Δ=+0.002 (t=0.00)、SVDD Δ=−0.175 (t=−0.24)。
    專案兩個標題主張（微調 −6~9%、SVDD −4.4%）都不成立 — 全是壞 baseline 48.691 的假象，加上
    paper ckpt 在我們 stack 本來就 45.99（贏已發表值 2%）。Power：排除 ≥1.5 單位的效應。**
    詳 `docs/report/sampler_report_1.md`。CoDe/TDS 推定同 null，不重跑。
  - [x] **1B 完成**：unguided bigblue4 η=0 (DDIM) 439.2 vs η=1 269.6 = **1.63 ≥ 1.30 gate** → 確定性
    sampler 在 size-OOD 崩潰（in-distribution 反而略好）；few-step/drifting 家族 evidential 關閉
    （seed 301/302 已排）。有 guidance 時 η=0 bb4 −4.9% / a1 +3.4%。**T=100 bb4 +7~14%，只省 43% 時間
    → Phase 2 不用 T=100 draft。**
  - [x] **1C 完成**：t-shift up avg7 51.71（+5.72，bb4 +35%）、inv 45.66（−0.33）→ 兩方向關閉；
    符號與 SD3 影像直覺相反（往低噪聲對大 circuit 略好但重傷小 circuit）。
  - [x] **2a 完成**：pre-leg vs post-leg 候選排序 ρ=+0.40（a1 與 bb4 皆然，argmin 都不同）→ 便宜 draft
    選擇死亡；全 legalize 的 best-of-4 在 bigblue4 **−6.2%**（121.53 vs 129.53，n=1）。
  - [x] **2b 作廢（我的預登記錯誤）**：legality floor 0.97 是 post-leg 門檻，套在 pre-leg 值（0.80–0.95）上
    永遠過不了 → 每列都 fallback 到「取最高 legality」，**從未以 HPWL 選**。policies.py 已加 warning +
    純 HPWL fallback。
  - [x] **2c 完成**：best-of-4 全 legalize，paired Δavg6 **−1.02 ± 1.22 (t=−1.44, n=3)** → 未達 −1.3；
    方向對、power 不足（需 n≈6）。bb4 −6.2%（n=1）。
  - [x] **3E 關閉**（deep-K 最佳 −2.3% n=1；K=150 +7~15%）；**3F 關閉**（post-hoc 平均兩 window 皆更差）；
    **3H 關閉**（frame averaging −0.4% / +2.4%）。
  - [x] **1A / 1B evidential (n=3)**：FM/DDPM 2.22 ± 0.20；η=0/η=1 1.61 ± 0.02。
  - [x] **4X Run X seeds 301/302：paired Δavg6 +0.36 ± 1.35 (t=0.46)，同號否 → 44.649 是 n=1 假警報，關閉。**
    paper / 微調 / from-scratch / SVDD 四種來源全落在同一 avg6 帶（~31.2 ± 0.8）。
  - [x] **4C DataAug Run C：avg7 45.171 vs Run X 44.649（+0.52）→ 關閉。** 兩個月前的孤兒 checkpoint 終於 eval。
- [x] 12.3 `docs/plan/sampler_plan_1.md` **全部執行完畢**（2026-09-15 04:04，43 cells，零失敗）→
  `docs/report/sampler_report_1.md`、`docs/next/sampler_next_1.md`。
- [x] 12.4 Phase 4 完成（09-15 13:26）：4X Run X 關閉（+0.36 ± 1.35）、4C Run C 關閉（+0.52）、
  **4B best-of-4 n=6 關閉（−0.70 ± 1.10，CI 含 0）**。**Step 12 全部結束：乾淨 stack 上零個正向方法結果。**
- Status: **Done**
  - [x] 程式：`eta_scale` / `t_shift` sampler knobs（938e10e，39 行，defaults bit-identical）；
    best-of-N（e3ff934：`open_loop_multi` max_score bug 修正、`hpwl_pre_legalization` 永遠記錄、
    `+num_candidates=N +candidate_legality_floor=0.97 [+legalize_all_candidates=true]`）。
    兩者由 Opus 在 worktree 實作、CPU 測試、Fable review 後合併。
  - [x] 程式：`frame_average` D4 test-time symmetrization（branch `frame-average`，53 行，default off
    bit-identical）。`+model.frame_average=true`，四個 sampler（plain / SVDD / CoDe / TDS）全部走
    `_eps()` helper。CPU 測試：inverse round-trip 0.0、eps_bar 對 8 個 frame 的 equivariance
    rel 5.0e-8（未平均者 2.8e-1）、default 路徑與改動前 bit-identical、cost 8.0–8.4×/step。
    **尚未在 GPU 上量過 HPWL。**
  - [ ] GPU 佇列 `scripts/run_sampler_plan_1.sh`（log: `logs/eval_logs/_queue.log`）：
    1A unguided control (4) → 0c baseline s301/302 (4) → **re-anchor** Run F s301/302 + SVDD s300–302 (10)
    → 1B η×T (12) → 1C t_shift (4) → 2a legalize-all (2) → 2b BoN 協定 (7)。skip-if-done，單一失敗不中斷。
    教訓：三個 ad-hoc waiter 用 `pgrep -f <pattern>` 等前一批，pattern 出現在自己的 cmdline 裡 →
    一個死鎖、兩個提早觸發並同時搶 GPU。**佇列一律寫成檔案、用 PID 等待。**
  - [x] **1A unguided control 完成**（seed 300）：bigblue4 FM-none 660.84 vs DDPM-none 269.57，
    比值 **2.45 ≥ 1.5 gate** → FM 的失敗在 objective/sampler，不是 guidance 交互；**FM 維持關閉**，
    `flowmatch_report_1` §3.2 的說法首次有實驗支撐。pre-legalization HPWL：FM 654→1962（3×）。
    附帶發現：DDPM 無 guidance 的 bigblue4 (269.6) ≈ FM 有 guidance (267.8) → OOD 泛化靠的是
    DDPM **+ opt guidance**，raw model 兩者都弱。n=1；evidential 升級待 bb4 seed 301/302（排 Phase 2 後）。
  - [x] Survey skeptic pass 全部完成（28 候選）→ 只剩 5 個零訓練測試值得跑，無新範式/backbone。
    關鍵反證：ledger 全資料回歸顯示 **quality 不隨 macro 數退化**（bigblue4 hpwl_ratio 最好），
    劣勢在 543–1,329 macro 的小 circuit → hierarchical 關閉；AR-hybrid 的 mask 通道是死碼。
  - [ ] Phase 3（預登記於 plan）：3E deep-K（config，已排在 1A follow-up 後）；3F post-hoc 權重平均
    （Opus 產 `avg_250k_500k_uniform.ckpt` 中；Run X 同 stack 重量已排）；3H frame averaging
    （`+model.frame_average=true` 已合併 07b1784，D4 等變性驗證到 6e-8，8× 推論成本；cells 已排在 3F 後）— **In progress**
- 模型分工：survey 資料蒐集 / 環境 / 執行 → Opus；分析、排序、計畫、判讀 → Fable

## Step 10: 回歸回顧 (2026-09-06, 距上次動工 2 個月)
- [x] 全 repo 交叉回顧（8 條實驗線 + 程式狀態 + 磁碟結果稽核）→ `docs/next/comeback_next_1.md`
- **阻斷發現**: GPU 已換成 RTX 5090 (sm_120, 32.6 GB)，`chipdiff` env 的 torch 2.2.1+cu121
  只支援到 sm_90 → 任何 CUDA kernel 都會失敗（`torch.cuda.is_available()` 仍回 True，
  會在第一次 kernel launch 才炸）。修好前無法執行任何 GPU 實驗。
- **風險發現**: HEAD 仍是 40a396d (2026-04-15) 且 == origin/main。SVDD/CoDe/TDS/FlowMatching/
  from-scratch/dataaug 全部程式與文件（+753 行已追蹤 / ~7,680 行未追蹤，含 4 個
  guidance YAML）**從未 commit**，只存在於這個 NAS 路徑。
- **結論修正**: 關閉「fine-tuning × inference-search 組合」方向的 headroom 禁令
  (`svdd_next_3.md` §5.1) 建立於單一 seed 的 Δ=0.164，而該方法自身 seed std=0.654（4 倍）。
  建議以 3 seeds 重測。
- **未收割**: DataAug Run C 已訓練完成 (2026-07-08) 但從未 eval；
  FromScratch Stage 2 = 46.441 (2026-06-12 已 eval) 在所有文件與 CSV 中皆無記錄。
- [x] **P0.1 commit + push 完成**（2026-09-07）：4 個主題 commit，HEAD 40a396d → 300eb49，
  五個月工作的單一副本風險解除。
- [ ] P0.2 升級 torch ≥2.7/cu128（開新 env，保留 chipdiff）— **未開始，阻斷所有 GPU 工作**
- [ ] P0.3 升級後重跑 Run F（預期 44.32）驗證 — **未開始**

## Step 11: Repo 整理與 workflow 控管 (2026-09-07)
- [x] 補記 4 筆磁碟上未記錄的結果進兩個 CSV（FromScratch_Stage2 46.441 + 3 個 flow-matching）
- [x] 修 CSV 資料陷阱：56 列因「逗號後空格 + 引號」在天真 parser 下**無聲**變成 10 欄。
  現已改為無引號格式，兩種 parser 讀出相同結果。
- [x] `svdd_report_3.md` / `svdd_next_2.md` 加更正 banner（原文保留）
- [x] 補寫 `docs/next/from_scratch_next_1.md`（Stage 2 回歸 45.05 → 46.44），
  並修正 `from_scratch_plan_1.md` / `report_1.md` 的過期 checkbox
- [x] 重寫 `docs/context.txt`（原本停在 2026-05-18，少了六條完成的實驗線）
- [x] eval 目錄碰撞防護：`eval.py` / `train_graph.py` 寫入前比對既有 `config.yaml`
  的 `from_checkpoint`，不同則中止（`+allow_overwrite=true` 可強制）
- [x] 建立 `scripts/ledger.py` + 自動產生的 `STATUS.md`：無條件 in-process 記錄、
  逐筆 metrics.csv 存進 `docs/ledger/results/`、三個工作佇列（中斷 / 未收割 / 未記錄）
- [x] 停用 repo 內 `memory/`（工具讀不到且已過期四個月），改寫 harness memory
- [x] 重寫 `CLAUDE.md`：環境阻斷警告、六條判讀規則、ledger 用法、移除過期 framing
- Status: **Done**

## Step 9: ISPD2005 limit analysis (docs/plan/ispd_limit_plan_1.md)
- [x] Part A oracle + Part C literature — wrote `docs/report/ispd_limit_report_1.md` (2026-07-06).
  **Oracle (best-per-circuit over 30 results) = 42.09**, i.e. 4.4% below Ablation 10k → ISPD2005 NOT saturated;
  seed variance (5-12%/circuit) is the dominant unexploited resource. Best-of-N inference could reach ~42.
- [ ] Part B (pure-optimization lower bound on adaptec1/bigblue1) — pending GPU availability.

## Step 8: Flow Matching variant (docs/plan/flowmatch_plan_1.md)
- [x] Phase A done (2026-07-06): `FlowMatchingModel` added to `diffusion/models.py`
  (subclass of ContinuousDiffusionModel; CFM loss, Euler ODE sampler), registered as
  `family=flow_matching` in `train_graph.py` and `eval.py`.
  - Pilot 1k steps (`v1.61-fs.61.fm_p1_pilot.61`): loss 2.61 → 0.58, no NaN; eval report OK (hpwl_ratio 1.22).
  - Sampler sanity: num_timesteps=50 and 1000 both finite, range ~[-1.14, 1.08].
- [x] Phase B done: 500k training completed in ~8.3 hr (55 ms/step, train/loss 0.17).
- [x] Phase C done (2026-07-07): ported opt guidance to ODE sampler (exact linear-path update),
  ran 7-circuit eval at 10/50/1000 ODE steps + step sweep.
- Status: **CLOSED — flow matching loses.** Best setting (50 steps) avg = **70.03** vs DDPM 45.05 / paper 46.89.
  Root cause: deterministic velocity field generalizes far worse OOD (V>700 degrades 17-154%, bigblue4 2.1× worse);
  small circuits DO reach near-DDPM quality with only 10-50 steps (2× faster gen). Also found inverted
  quality-vs-steps (more ODE steps = worse, over-guidance). See `docs/report/flowmatch_report_1.md`,
  `docs/next/flowmatch_next_1.md`. Verdict per plan §4: > 46.89 → abandon, return to DDPM track.
  Drifting model NOT attempted (same few-step deterministic family, same expected OOD weakness).

## Current Goal: Verify pre-trained model performance matches paper

### Step 1: Generate v2 validation data (eval only)
- [x] **In progress** — Generate v2 val data: `PYTHONPATH=. python data-gen/generate_parallel.py versions@_global_=v2 num_train_samples=0 num_val_samples=200`
- Status: **In progress** (restarted 2026-03-18, previous run interrupted before generating any pkl files)

### Step 2: Run evaluation with pre-trained checkpoint
- [ ] Eval: `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python diffusion/eval.py task=v2.61 method=eval from_checkpoint=logs/public-models/large-v2/large-v2.ckpt num_output_samples=128 logger.wandb=False`
- Status: **Not started**

### Step 3: Compare metrics with paper
- [x] Compare HPWL, legality, etc. against paper's reported numbers
- Status: **Done** (2026-04-01). ISPD2005 macro-only eval completed for 7/8 circuits. Results close to paper; bigblue3/4 slightly better, adaptec1-4 slightly worse (seed difference). See `logs/diffusion_debug/ispd2005-s0.eval_macro_only.300/comparison_with_paper.csv`.

### Step 4: Fine-tune experiments — AddLoss v2 (timestep weighting)
- [x] Train AddLoss v2 (10k steps, `use_timestep_weighting=True`, hpwl_weight=1e-3, legality_weight=1e-2)
- [x] Eval AddLoss v2 on ISPD2005 (8 samples, macros_only)
- [x] Write `docs/report/addloss_report_2.md`
- Status: **Done** (2026-04-18). Eval actually completed on Apr 15 20:50 despite earlier belief it was stuck. 7-circuit avg HPWL **45.24** (v1: 45.39, Ablation 10k still best at 44.01). Mixed per-circuit results: bigblue2/4/adaptec4 improved as hypothesized, but bigblue3 regressed +15%. Report recommends pivoting to Best-of-N self-improve or bigblue2 cluster-aware guidance.

### Step 5: New guidance direction — SVDD (inference-time RL-guided)
Triggered by professor's 2026-05-08 suggestion to try other guidance methods (e.g. RL-guided) — see `docs/meet/meet_0508.md`.
- [x] Survey: write `docs/survey/guidance_survey_1.md` covering SVDD / CoDe / TDS / DRaFT / FreeDoM / Reflected / Mirror diffusion
- [x] Plan: write `docs/plan/svdd_plan_1.md` (SVDD-PM Phase 1 + Phase 2)
- [x] **Step 5a**: implemented `_reverse_samples_svdd()` in `models.py` + `configs/guidance/svdd.yaml` (K, α_temp, λ_legality, every_n_steps, start_step_frac)
- [x] Step 5b: Phase 1 sweep on adaptec1 (3 seeds × {none, K4L0, K4L1, K8L0, every_n=1, start_step=0.3}) + bigblue1 (2 seeds × default config). All in `logs/svdd_p1/`.
- [x] Step 5d: wrote `docs/report/svdd_report_1.md`
- [x] Step 5e: wrote `docs/next/svdd_next_1.md` (Phase 1 retrospective)
- [x] Step 5f: wrote `docs/plan/svdd_plan_2.md` (Phase 2 plan: ablation_10k + opt-adam + 7 circuits)
- [x] Step 5g: ran Phase 2 Run A (baseline, guidance=none) + Run C (SVDD every_n=1) on 7 ISPD circuits, skip bb2
  - Run A part2 (bigblue3, bigblue4) crashed on GPU 0 with cudaErrorIllegalAddress (other user using 10 GB); succeeded after switching to CUDA_VISIBLE_DEVICES=1
- [x] Step 5h: wrote `docs/report/svdd_report_2.md`
- [x] Step 5i: wrote `docs/next/svdd_next_2.md` (Phase 2 retrospective)
- [x] Step 5j: implemented `svdd_layer_opt` flag in `_reverse_samples_svdd` + `configs/guidance/svdd_layered.yaml`
- [x] Step 5k: wrote `docs/plan/svdd_plan_3.md` (Phase 3 plan: SVDD+opt layered vs paper baseline)
- [x] Step 5l: ran Phase 3 Run F (paper baseline reproduce) + Run E (SVDD+opt layered), 7 ISPD circuits each
- [x] Step 5m: wrote `docs/report/svdd_report_3.md`
- Status (Phase 3, since superseded): SVDD seemed dead-end based on ablation_10k comparison.
- [x] Step 5n: **User corrected framing error** — Run F (ablation_10k + opt = 44.32) is NOT paper baseline; paper baseline = large-v2 + opt + leg = **46.89 (paper) / 48.69 (our env)**
- [x] Step 5o: ran Phase 4 on **large-v2 checkpoint** (paper's actual ckpt). Run G (svdd only) = **66.53** (loses), Run H (svdd_layered) = **45.10** = **−3.8% vs paper 46.89, −7.4% vs our reproduction 48.69**
- [x] Step 5p: wrote `docs/report/svdd_report_4.md` with corrected framing
- Status (Phase 4, single-seed): SVDD layered on paper opt beats paper 45.10 vs 46.89 (-3.8%). Pending multi-seed validation.
- [x] Step 5q: Phase 5 multi-seed validation (seed=301, 302) on large-v2 + svdd_layered + opt-adam, 7 circuits each
- [x] Step 5r: wrote `docs/report/svdd_report_5.md` — **3-seed mean 44.85 ± 0.65 vs paper 46.89 = −4.36%**, all 3 seeds individually win paper. bigblue3 single-seed +14% turned out to be bad luck — 3-seed mean −3.0% (also wins). bigblue4 biggest win at -10.8%.
- Status: **SVDD-PM layered on paper opt CONFIRMED beats paper baseline** (3-seed mean 44.85 ± 0.65 vs paper 46.89). Leaderboard rank 5 — only non-fine-tuned method to beat paper. Phase 3 "no marginal value" only holds on fine-tuned ckpt (headroom theory).
- [x] Step 5s: wrote `docs/next/svdd_next_3.md` (Phase 3-5 retrospective)

### Step 6: CoDe (blockwise best-of-N) — SVDD ablation
Started 2026-05-24. Goal: test if hard argmax at block-level matches SVDD's soft-resample (per guidance_survey_1.md §1.2).
- [x] Step 6a: wrote `docs/plan/code_plan_1.md`
- [x] Step 6b: implemented `_reverse_samples_code` in `diffusion/models.py` + `configs/guidance/code_layered.yaml`
- [x] Step 6c: ran CoDe Phase 1 (3 seeds × 7 circuits) on large-v2 + code_layered + opt-adam
- [x] Step 6d: wrote `docs/report/code_report_1.md` — **CoDe 3-seed mean 45.22 ± 0.47, beats paper -3.57% (3/3 seeds), vs SVDD 44.85 = +0.37 (within noise, t-stat 0.80)**. Confirmed inference-time K-particle search is the core mechanism; soft/hard selection doesn't matter much.
- [x] Step 6e: wrote `docs/next/code_next_1.md`
- Status: **CoDe ≈ SVDD on 7-circuit avg, but bigblue4 SVDD wins by +4.42** (only meaningful per-circuit difference). CoDe is simpler/faster/lower-variance → suggested paper main method, SVDD as ablation. Next options: (a) CoDe block_size sweep to close bigblue4 gap, (b) FreeDoM time-travel, (c) TDS principled SMC, (d) write paper, (e) professor suggestion (1) from-scratch train.

### Step 7: TDS (Twisted Diffusion Sampler / SMC) — 3rd inference-time search method
Started 2026-05-25. User picked TDS as next guidance (per guidance_survey_1.md §1.3).
- [x] Step 7a: wrote `docs/plan/tds_plan_1.md`
- [x] Step 7b: implemented `_reverse_samples_tds` (N parallel particles + delta-based log-weight + ESS resample + best-of-N output) + `configs/guidance/tds_layered.yaml`
- [x] Step 7c: **fixed latent bug** in `reverse_guidance_opt_force` — `alpha_cost.backward()` only worked for B=1 (PyTorch auto-scalarizes `(1,)` shape). Changed to `.mean()` for N>1 support, identity for B=1.
- [x] Step 7d: ran TDS Phase 1 (3 seeds × 7 circuits) on large-v2 + tds_layered + opt-adam
- [x] Step 7e: wrote `docs/report/tds_report_1.md` — **TDS 3-seed mean 45.08 ± 0.38, beats paper -3.86% (3/3 seeds)**. Statistically equivalent to SVDD (44.85) and CoDe (45.22). **Unique value: lowest std (bigblue3 std 5.34 → 0.16, −97%) and best worst-case circuit (adaptec2 tied paper)**.
- [x] Step 7f: wrote `docs/next/tds_next_1.md`
- Status: **3 inference-time search methods (SVDD/CoDe/TDS) all beat paper -3.6~-4.4%, all statistically equivalent**. Method ceiling reached at ~-4%. Next: (a) FreeDoM time-travel layered on TDS (orthogonal, +5-10 lines, expected -0.5~-1% more), (b) write paper draft (full ablation table ready), (c) professor suggestion (1) from-scratch train.

---

## Full Data Generation (if needed later)

### v0 dataset
- [ ] Generate v0 dataset
- Status: **Not started**

### v1 dataset
- [ ] Generate v1 dataset
- Status: **Not started** (config.yaml created in `data-gen/outputs/v1.61/` but no .pkl files)

### v2 dataset (full)
- [ ] Generate v2 dataset (5000 train + 2500 val)
- Status: **Not started**

## Training / Fine-tuning (if needed later)
- [ ] Train large model on v1
- [ ] Fine-tune on v2
- Pre-trained checkpoint available: `logs/public-models/large-v2/large-v2.ckpt`
