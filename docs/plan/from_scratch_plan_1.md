# 從頭訓練 Chip Placement Diffusion — 第一次實驗計畫（Strategy A: Full Paper Reproduce）

> 動機：`docs/meet/meet_0508.md` §1 教授建議「從頭 train，不要 fine-tuning」
> 使用者選擇（2026-05-25）：**Strategy A — 完整重現 paper 兩階段訓練流程**，apples-to-apples 跟 paper 46.89 比
> 現有最佳對照：Paper baseline 46.89；Ablation 10k 44.01（large-v2 + 10k fine-tune）；SVDD layered 44.85（inference-time）

## 1. 重大發現（影響整個 plan）

從 `logs/public-models/large-v2/config.yaml` 反推：**paper 的 `large-v2.ckpt` 是兩階段訓練產物**。

| Stage | Dataset | Mode | Steps | Batch | LR | from_checkpoint |
|-------|---------|------|-------|-------|----|----|
| **Stage 1**（from-scratch） | `v1.61`（max_instance=400） | train | **3,000,000** | 64 | 3e-4 | none |
| **Stage 2**（fine-tune） | `v2.61`（max_instance=1600） | finetune | **500,000** | 32 | 1e-4 | stage 1 ckpt |
| Result | — | — | — | — | — | `large-v2.ckpt` |

我們手邊的 dataset 狀況：
| Dataset | 我們有 | gen_params vs paper |
|---|---|---|
| v1.61 (paper stage 1) | 沒 train data on disk | — |
| v1.61-ddpo | ✅ 1600 train + 400 val | **gen_params 完全一樣** → 可直接代替 paper 的 v1.61 |
| v2.61 (paper stage 2) | 只有 200 val | 需要 regen train data（~4-6 hr）|

Model `size=large` (`configs/model/size/large.yaml`) 跟 paper config 完全一致（hidden 256, 3 blocks of [256,256,256]）。

## 2. 目標（重述）

### 2.1 兩個 variant 對照

| Variant | Stage 1 objective | Stage 2 objective | 目的 |
|---|---|---|---|
| **Run X**（paper exact） | Pure denoising MSE | Pure denoising MSE | **直接 reproduce paper**，給出我們 env 下的 paper baseline |
| **Run Y**（our integrated obj） | Denoising MSE + HPWL aux + legality aux | 同（continue）| 測 from step 0 整合 reward objective vs paper-style pure |

### 2.2 預期結果結構

| 比較 | 預期 / 意義 |
|---|---|
| Run X final（stage 1+2） vs CLAUDE.md large-v2 (paper) eval = 46.89 | 應該接近（差 < 2%）→ 確認 reproduce 成功 |
| Run X final vs Ablation 10k (44.01) | Run X 應該 ≈ paper ≈ 46.89, 輸 ablation 10k 一截（因為 ablation_10k 比 paper baseline 強 6.1%） |
| Run Y final vs Run X final | **核心問題：integrated obj 帶來什麼差？** |
| Run Y final vs 44.01 | **新 SOTA 候選**：若 < 44.01，from-scratch + integrated obj 是新 best |
| Run Y stage 1 vs Run X stage 1 | 中間 milestone，看 aux 在 stage 1 是否就有差 |

---

## 3. Approach Choices

### 3.1 Dataset

- **Stage 1**：`v1.61-ddpo`（gen_params == paper v1.61）
- **Stage 2**：`v2.61` ── **必須先 regen train data**（plan §6.1 為獨立 step）

### 3.2 Architecture

完全照 paper：
- `family=continuous_diffusion`
- `backbone=att_gnn`
- `size=large`（hidden 256, [256,256,256] blocks）
- `noise_schedule=cosine`, max_diffusion_steps=1000, t_encoding=sinusoid/32

### 3.3 Training Schedule（照 paper）

| Stage | 我們 (Run X / Run Y) |
|---|---|
| Stage 1 | 3,000,000 steps, batch=64, lr=3e-4 |
| Stage 2 | 500,000 steps, batch=32, lr=1e-4 |

### 3.4 Run Y 的 integrated objective（差別 vs Run X）

Run Y 在 `ContinuousDiffusionModel.loss()` 開：
- `hpwl_weight=1e-3`（同 AddLoss v2，已驗證 stable）
- `legality_weight=1e-2`（同 AddLoss v2）
- `use_timestep_weighting=True`（α_t² 壓制 early-t 噪音 predicted_x0）

從 stage 1 step 0 開始，stage 2 continue。

### 3.5 Eval

#### 3.5.1 Intermediate Checkpoints

Stage 1 中每 500k step save checkpoint + ISPD subset eval（adaptec1+bigblue1, ~15 min）：
- 500k, 1M, 1.5M, 2M, 2.5M, 3M
- 看 train loss 跟 ISPD HPWL 的對應，**如果 1.5M 還沒到 paper 水準就要 abort 改 plan**

#### 3.5.2 Final Evals

Run X 跟 Run Y 各跑 full 7-circuit ISPD eval (seed=300 single seed)：
1. Stage 1 final ckpt + opt + opt-adam（vs paper stage 1）
2. Stage 2 final ckpt + opt + opt-adam（vs paper large-v2 = 46.89）
3. Stage 2 final ckpt + svdd_layered + opt-adam（vs SVDD on large-v2 = 44.85）

---

## 4. 預估成本

| 工作項 | 估時 | 累積 |
|---|---|---|
| Stage 1 pilot Run X（1k step）| 5 min | 5 min |
| Stage 1 pilot Run Y（1k step） | 5 min | 10 min |
| Stage 1 Run X 3M step | **~3.5 天** | ~3.5 天 |
| Intermediate eval (5 milestones × ~15 min) | ~75 min | +1.25 hr |
| Stage 1 Run Y 3M step | **~3.5 天** | ~7 天 |
| Intermediate eval Run Y | ~75 min | ~7 天 |
| Regen v2.61 train data | **~4-6 hr** | ~7.25 天 |
| Stage 2 Run X 500k step | **~14 hr** | ~7.85 天 |
| Stage 2 Run Y 500k step | **~14 hr** | ~8.45 天 |
| Final ISPD eval 4 sets (Run X stage 1, Run X stage 2, Run Y stage 1, Run Y stage 2) × 1 hr | ~4 hr | ~8.6 天 |
| **Phase 1 總計** | **~8.5-9 days continuous GPU 1 time** | |

> ⚠️ **重要**：這是 sequential，GPU 1 single。若 GPU 0 在這 9 天內變空可以 parallel 兩 variant 把時間砍半（~4-5 天）。

---

## 5. 階段化執行（風險控制）

考慮到 9 天投入的 commitment，分階段並設 abort gate：

### Phase 1a — Pipeline 驗證（~10 min）
- Run X 1k pilot：確認 `mode=train, from_checkpoint=none, task=v1.61-ddpo, size=large, train_steps=1000` 跑得通
- Run Y 1k pilot：確認 `mode=train, ... addloss.hpwl_weight=1e-3, addloss.legality_weight=1e-2, addloss.use_timestep_weighting=True` 跑得通
- **Gate**: 兩個 pilot 都跑得通且 train loss 下降 → 進 Phase 1b

### Phase 1b — Stage 1 Run X (paper exact, ~3.5 天)
- 3M steps, eval at {500k, 1M, 1.5M, 2M, 2.5M, 3M}
- **Gate**: 1.5M intermediate eval 必須在 60 之內（合理收斂趨勢），不然 abort

### Phase 1c — Stage 1 Run Y (~3.5 天)
- 同 Phase 1b setup + addloss params
- **Gate**: 1.5M Run Y 跟 Run X 趨勢有 meaningful 差別

### Phase 1d — V2.61 train data regen (~4-6 hr)
- `PYTHONPATH=. python data-gen/generate_parallel.py versions@_global_=v2 num_train_samples=1600 num_val_samples=400`
- 確認 generate 出 1600 train pickle

### Phase 1e — Stage 2 Run X + Run Y (各 ~14 hr)
- finetune mode, 500k step, batch=32, lr=1e-4
- from_checkpoint=各自 stage 1 final

### Phase 1f — Final ISPD eval + report
- 4 個 final checkpoints × 7 circuits + opt + opt-adam, seed=300
- 寫 `docs/report/from_scratch_report_1.md`

---

## 6. 具體執行命令

### 6.1 Phase 1a pilots

**Run X pilot:**
```bash
PYTHONPATH=. python diffusion/train_graph.py \
  mode@_global_=train \
  task=v1.61-ddpo \
  method=fs_p1_X_pilot \
  train_steps=1000 \
  print_every=100 \
  eval_every=1000 \
  batch_size=64 \
  lr=3e-4 \
  logger.wandb=false
```

**Run Y pilot:**
```bash
PYTHONPATH=. python diffusion/train_graph.py \
  mode@_global_=train \
  task=v1.61-ddpo \
  method=fs_p1_Y_pilot \
  train_steps=1000 \
  print_every=100 \
  eval_every=1000 \
  batch_size=64 \
  lr=3e-4 \
  +addloss.hpwl_weight=1e-3 \
  +addloss.legality_weight=1e-2 \
  +addloss.use_timestep_weighting=True \
  logger.wandb=false
```

> **Open question**：`mode=train` 是否走 addloss path？已知 `mode=finetune` 走 addloss path（`train_graph.py` 內邏輯）。若 `mode=train` 不接 addloss，需要：(a) 改 `mode=finetune` + `from_checkpoint=none`，或 (b) 改 `train_graph.py` 讓 train mode 也讀 addloss config。**Phase 1a pilot 就是驗證這個**。

### 6.2 Phase 1b — Stage 1 Run X (3M steps)

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/train_graph.py \
  mode@_global_=train \
  task=v1.61-ddpo \
  method=fs_p1_X_stage1 \
  train_steps=3000000 \
  print_every=10000 \
  eval_every=500000 \
  batch_size=64 \
  lr=3e-4 \
  logger.wandb=false
```

### 6.3 Phase 1d — Regen v2.61 train data

```bash
PYTHONPATH=. python data-gen/generate_parallel.py \
  versions@_global_=v2 \
  num_train_samples=1600 \
  num_val_samples=400 \
  num_workers=4   # ⚠ 不用 default 64，避免 OOM 殺 SSH（CLAUDE.md gotcha）
```

### 6.4 Phase 1e — Stage 2 Run X (continue from stage 1 final)

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python diffusion/train_graph.py \
  mode@_global_=finetune \
  task=v2.61 \
  method=fs_p1_X_stage2 \
  from_checkpoint=v1.61-ddpo.fs_p1_X_stage1.61/latest.ckpt \
  train_steps=500000 \
  batch_size=32 \
  lr=1e-4 \
  logger.wandb=false
```

### 6.5 Phase 1f — Final eval (template)

```bash
PYTHONPATH=. python diffusion/eval.py \
  method=fs_p1_X_stage2_eval_opt \
  task=ispd2005-s0 \
  from_checkpoint=v2.61.fs_p1_X_stage2.61/latest.ckpt \
  legalizer@_global_=opt-adam \
  guidance@_global_=opt \
  num_output_samples=8 \
  +start_sample=0 \
  +skip_guidance_threshold=10000 \
  model.grad_descent_steps=20 \
  model.hpwl_guidance_weight=16e-4 \
  legalization.alpha_lr=8e-3 \
  legalization.hpwl_weight=12e-5 \
  legalization.legality_potential_target=0 \
  legalization.grad_descent_steps=20000 \
  macros_only=True \
  seed=300 \
  logger.wandb=false
```

---

## 7. 判定基準

### 7.1 Phase 1f final（stage 2 完成後）

| Run X (paper exact) avg HPWL | 意義 |
|---|---|
| 45 ~ 49 | ✅ Reproduce 成功，跟 paper 46.89 一致 |
| < 45 | ⚠️ 比 paper 好太多，可能 seed lucky 或環境差異 |
| > 49 | ❌ Reproduce 失敗（可能 dataset/code 環境問題）|

| Run Y (integrated obj) avg HPWL | 意義 | 行動 |
|---|---|---|
| **< 44.01** | **新 SOTA**：from-scratch + integrated obj 贏 ablation_10k | 寫 paper 重點章節；plan_2 + svdd_layered stack |
| 44.01 ~ Run X | 贏 paper 但不贏 ablation_10k | integrated obj 有效，但比 fine-tune-with-aux 弱 |
| Run X ± 1 | aux 沒幫助 | 結論「from step 0 加 aux 不比 paper-style 好」 |
| > Run X + 2 | aux 反而傷害 | 結論「aux 在 from-scratch 干擾學習」 |

### 7.2 Intermediate gates（每 500k step）

| Milestone | Run X 預期 HPWL（adaptec1+bigblue1 subset）| 觸發 abort 條件 |
|---|---|---|
| 500k | < 80 (still high) | > 100 → loss 沒下降，pipeline 有問題 |
| 1.5M | < 60 | > 80 → 收斂太慢，3M 也救不了 |
| 3M | < 55 | > 55 → stage 1 沒到位，stage 2 也救不回 paper 水準 |

---

## 8. 風險與對策（更新）

| 風險 | 對策 |
|---|---|
| **9 天 GPU 時間是大投入，中途 SSH/GPU 失聯損失** | 每 100k step save checkpoint；用 `nohup` + 寫 PID file；定期人工 check `nvidia-smi` |
| **GPU 0 contention 期間若搶到** | 預設 `CUDA_VISIBLE_DEVICES=1` 全程；GPU 0 留給其他人 |
| **v2.61 regen 失敗 / OOM** | 用 `num_workers=4` 而非 default 64（CLAUDE.md gotcha）|
| **`mode=train` 沒接 addloss** | Phase 1a pilot 驗證；若不接，改 `mode=finetune + from_checkpoint=none` |
| **3M step 訓練中 GPU OOM** | batch=64 在 V<400 應該 OK；OOM 改 batch=32 (lr 也跟著調) |
| **Run X 不能 reproduce paper 46.89** | check stage 1 ckpt loss curve；對照 `logs/public-models/large-v2/config.yaml` 找差別 |
| **Run X + Run Y 兩個各跑 3M 序列 9 天，期間需要 GPU 跑其他事情** | 跟 user 確認這 9 天 GPU 1 鎖給此實驗 |
| **Intermediate eval 中斷主訓練** | eval 用獨立 python process，跑完就退；train continue 不受影響 |

---

## 9. Open questions（在 Phase 1a 解答）

1. `mode=train` 跟 `mode=finetune` + `from_checkpoint=none` 行為差異？（pilot 跑兩個）
2. `addloss.hpwl_weight` 等等是否在 train mode 下 by default 是 0？需要 `+` prefix 加上？
3. Train loss 第一個 100 step 範圍 ~多少？（看 pilot 確認 reasonable）
4. 1k step VRAM 使用？（決定 3M run 會不會 OOM）

---

## 10. 行動清單

- [ ] **Step 1**：Phase 1a — Run X + Run Y 各 1k pilot, check pipeline + VRAM + loss decrease
- [ ] **Step 2**：跟 user 確認 commit 8-9 天 GPU 1，再進 Phase 1b
- [ ] **Step 3**：Phase 1b — Stage 1 Run X 3M (long-running background, GPU 1)
- [ ] **Step 4**：1.5M intermediate eval, decide continue/abort
- [ ] **Step 5**：Phase 1c — Stage 1 Run Y 3M
- [ ] **Step 6**：Phase 1d — regen v2.61 train data
- [ ] **Step 7**：Phase 1e — Stage 2 Run X + Run Y (500k each)
- [ ] **Step 8**：Phase 1f — final 7-circuit ISPD eval × 4 setups
- [ ] **Step 9**：寫 `docs/report/from_scratch_report_1.md` + `docs/next/from_scratch_next_1.md`

---

## 11. 跟其他 track 的關係

- **SVDD/CoDe/TDS（inference-time）軌道已完成 Phase 5/1**：3 個方法都 saturate 在 44.85 ± 0.4 區間
- 若 Run Y stage 2 < 44.01 → 從頭 train + aux 是新 SOTA → 加上 svdd_layered 可能再 -0.5%
- 若 Run Y = Run X = ~paper → from-scratch with v1.61-ddpo dataset 沒突破，需 plan_2 改 v2.61-only 訓練或更大 dataset

---

## 12. 開始前最後一個決策

**這是 8-9 天 GPU 1 commitment**。確認以下三件事：
1. 此期間 GPU 1 鎖給此實驗，不臨時插隊？
2. Phase 1a pilots（~10 min）跑完確認 pipeline，是否需要再 user confirm 才進 3M run？
3. 中途若 1.5M intermediate eval 觸發 abort gate（HPWL 太高），自動切到 plan_2 還是先停下來討論？

**我的建議**：先跑 Phase 1a 兩個 pilot（10 min），把結果給 user，再 user confirm 後啟動 3M。
