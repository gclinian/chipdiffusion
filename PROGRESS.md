# Progress Tracker

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
