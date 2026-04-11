# DDPO Fine-tuning 執行計畫

## 概述

使用 Denoising Diffusion Policy Optimization (DDPO) 對 pre-trained `large-v2.ckpt` 進行強化學習微調，以 HPWL 和 legality 作為 reward signal，目標是在 ISPD2005 benchmark 上超越原始模型的 placement 品質。

---

## 1. DDPO 原理

DDPO 將 diffusion 的 reverse process 視為一條 RL trajectory：
- **State**：每個 timestep 的 noisy placement
- **Action**：denoising 預測的 noise
- **Reward**：最終 placement 的品質（HPWL + legality）

Loss 計算：
```
advantage = (reward - EMA_mean) / EMA_std
loss = -mean(trajectory_log_prob * advantage)
```

與 supervised training 不同，DDPO 不需要 ground truth placement，只需要一個可計算的 reward function。

---

## 2. 現有程式碼分析

### 已有的部分
| 檔案 | 內容 | 狀態 |
|------|------|------|
| `diffusion/ddpo.py` | DDPO class + reward functions | 已完成 |
| `diffusion/train_graph.py` | 訓練迴圈，支援 `mode=ddpo` | 已完成 |
| `diffusion/configs/mode/ddpo.yaml` | DDPO 超參數 config | 已完成 |
| `diffusion/guidance.py` | `legality_guidance_potential()`, `hpwl_guidance_potential()` | 已完成 |

### 需要修改的部分（關鍵）

**`ContinuousDiffusionModel.reverse_samples()` 不支援 `output_log_prob`**

`large-v2.ckpt` 使用的 model family 是 `continuous_diffusion`（`ContinuousDiffusionModel`）。但 DDPO 需要呼叫 `model.reverse_samples(..., output_log_prob=True)` 來取得 trajectory log probability。

目前只有 `CondDiffusionModel`（line 257）和 `GuidedDiffusionModel`（line 407）支援 `output_log_prob`，`ContinuousDiffusionModel`（line 1400）不支援。

**必須修改 `models.py`**，在 `ContinuousDiffusionModel.reverse_samples()` 中加入 `output_log_prob` 支援。參考 `CondDiffusionModel` 的實作：
- 初始化 `log_probs = torch.zeros((T, B), device=x.device)`
- 每個 timestep 計算 `pi_log_prob(x, mu, sigma)`
- 回傳時多回傳 `log_probs`

### Reward function 的 V×V 問題

`legality_reward()` 呼叫的是 `legality_guidance_potential()`（非 tiled 版本），會建立 V×V 矩陣。在 DDPO 中，reward 是對每個 batch sample 獨立計算的，所以：

- **Synthetic data（V~200-300）**：V×V 很小，沒問題
- **ISPD2005（V=500-8000）**：直接用 DDPO 的 reward function 會 OOM

如果要在 ISPD2005 上做 DDPO，需要把 `legality_reward()` 改為使用 `legality_guidance_potential_tiled()` 版本。但建議先在 synthetic data 上訓練。

---

## 3. 訓練資料

DDPO 不需要 ground truth，但需要 graph 結構（netlist）作為 conditioning input。

### 選項 A：用 synthetic data 微調（建議先做）
- 資料位置：`data-gen/outputs/v1.61/`（2000 val samples，每個 ~261 nodes）
- 優點：node 數少，VRAM 需求低，訓練快
- 缺點：distribution 與 ISPD2005 不同，需要驗證 transfer 效果
- 需要生成 train split（目前只有 val）

### 選項 B：用 ISPD2005 circuit 微調
- 資料位置：`datasets/graph/ispd2005-s0/`（7 個可用 circuit）
- 優點：直接在目標 domain 上訓練
- 缺點：只有 7 個 sample，容易 overfit；大 circuit 的 reward 計算會 OOM
- 需要修改 reward function 使用 tiled 版本

### 建議策略
1. **Phase 1**：先在 synthetic data (v1) 上做 DDPO，驗證 training pipeline 可行
2. **Phase 2**：在小型 ISPD circuit（adaptec1, bigblue1, ~500 nodes）上 fine-tune

---

## 4. 具體執行步驟

### Step 0：修改程式碼（必要）

在 `diffusion/models.py` 的 `ContinuousDiffusionModel.reverse_samples()` 中加入 `output_log_prob` 支援。需要：

1. 函式簽名加入 `output_log_prob=False` 參數
2. 在 reverse loop 中計算每步的 log probability
3. 需要確認 `ContinuousDiffusionModel` 的 noise scheduler step 如何對應到 `mu` 和 `sigma`，以便計算 `pi_log_prob`

這是最大的工程量，預估需要仔細對照 `CondDiffusionModel` 的實作。

### Step 1：驗證 DDPO pipeline（在 synthetic data 上）

```bash
PYTHONPATH=. python diffusion/train_graph.py \
  mode@_global_=ddpo \
  task=v1.61 \
  family=continuous_diffusion \
  from_checkpoint=public-models/large-v2/large-v2.ckpt \
  method=ddpo_v1_test \
  ddpo.legality_weight=0.0 \
  ddpo.hpwl_weight=1.0 \
  batch_size=4 \
  train_steps=1000 \
  print_every=100 \
  eval_every=500 \
  lr=1e-4 \
  logger.wandb=false
```

> 注意：上述指令可能需要根據實際 Hydra config 結構調整。先跑少量 steps 確認 loss 有在下降。

### Step 2：完整 DDPO 訓練

確認 pipeline 可行後，拉長訓練：

```bash
PYTHONPATH=. python diffusion/train_graph.py \
  mode@_global_=ddpo \
  task=v1.61 \
  family=continuous_diffusion \
  from_checkpoint=public-models/large-v2/large-v2.ckpt \
  method=ddpo_v1_full \
  ddpo.legality_weight=0.5 \
  ddpo.hpwl_weight=1.0 \
  batch_size=4 \
  train_steps=50000 \
  print_every=200 \
  eval_every=5000 \
  lr=1e-5 \
  logger.wandb=false
```

建議調整：
- `lr=1e-5`（比 config 預設的 1e-4 更保守，避免 catastrophic forgetting）
- 加入 `legality_weight=0.5`，避免只追求 HPWL 而犧牲 legality
- `batch_size=4`（受限於 VRAM）

### Step 3：在 ISPD2005 上評估

```bash
PYTHONPATH=. python diffusion/eval.py \
  method=eval_macro_only \
  task=ispd2005-s0 \
  from_checkpoint=<ddpo_checkpoint_path> \
  legalizer@_global_=opt-adam \
  guidance@_global_=opt \
  num_output_samples=5 \
  +start_sample=0 \
  +skip_guidance_threshold=10000 \
  model.grad_descent_steps=20 \
  model.hpwl_guidance_weight=16e-4 \
  legalization.alpha_lr=8e-3 \
  legalization.hpwl_weight=12e-5 \
  legalization.legality_potential_target=0 \
  legalization.grad_descent_steps=20000 \
  macros_only=True \
  logger.wandb=false
```

### Step 4（可選）：在 ISPD2005 小 circuit 上直接 DDPO

如果 Step 3 效果不佳，可以嘗試直接在 ISPD circuit 上做 DDPO。需要：
- 修改 `legality_reward()` 使用 tiled 版本
- 只用小 circuit（adaptec1, bigblue1）
- 非常小的 batch size（1-2）

---

## 5. 資源需求

### VRAM 估算

| 項目 | 估算 |
|------|------|
| 模型權重（large AttGNN） | ~0.5 GB |
| Forward pass（B=4, V=261） | ~1 GB |
| Reverse sampling（1000 steps, 需保留 log_prob 梯度） | ~4-6 GB |
| Reward 計算（legality V×V, V=261） | ~0.3 GB |
| Backward pass + optimizer states | ~4-6 GB |
| **Total（synthetic data, B=4）** | **~10-14 GB** |

> 注意：DDPO 的 reverse sampling 需要對整個 trajectory 保留梯度（log_prob 需要 grad），比普通 inference 吃更多 VRAM。

| 場景 | 預估 VRAM |
|------|-----------|
| Synthetic data, B=4, V~261 | ~10-14 GB ✅ 可行（24GB GPU） |
| Synthetic data, B=8 | ~18-22 GB ⚠️ 可能剛好 |
| ISPD adaptec1, B=1, V=543 | ~12-16 GB ✅ 可行 |
| ISPD adaptec4, B=1, V=1329 | ~18-24 GB ⚠️ 勉強 |
| ISPD bigblue4, B=1, V=8170 | >24 GB ❌ OOM |

### 訓練時間估算

| 階段 | Steps | 預估時間 |
|------|-------|---------|
| Step 1：Pipeline 驗證 | 1,000 | ~1-2 小時 |
| Step 2：完整訓練 | 50,000 | ~2-4 天 |
| Step 3：ISPD 評估 | — | ~2-3 小時（7 circuits） |

> 每個 DDPO step 需要完整的 reverse sampling（1000 denoising steps）+ reward 計算 + backward，比普通 supervised training 慢約 10-20 倍。

### 磁碟空間
- Checkpoint：~1 GB（每次儲存）
- 定期存 checkpoint（每 5000 steps）：~10 GB
- 建議預留 20 GB

---

## 6. 超參數建議

| 參數 | 建議值 | 說明 |
|------|--------|------|
| `lr` | 1e-5 | 保守 LR，避免 catastrophic forgetting |
| `batch_size` | 4 | 受限於 VRAM（reverse sampling 很吃記憶體） |
| `ddpo.hpwl_weight` | 1.0 | 主要目標 |
| `ddpo.legality_weight` | **≥ 0.5（必須 > 0）** | 第一次實驗用 0.0 導致 adaptec1 legality 從 0.994 暴跌到 0.919。**不能省略 legality reward** |
| `ddpo.ema_factor` | 0.999 | Reward baseline 的 EMA 衰減 |
| `train_steps` | 50,000 | 先跑 50k 看趨勢 |
| `eval_every` | 5,000 | 定期評估 |

---

## 7. 風險與對策

| 風險 | 說明 | 對策 |
|------|------|------|
| Catastrophic forgetting | RL fine-tuning 可能破壞 pre-trained 的品質 | 用小 LR (1e-5)；定期在 ISPD 上 eval；保留 baseline checkpoint |
| Reward hacking | 模型找到高 reward 但實際品質差的 shortcut | 同時監控多個指標（HPWL, legality, visual quality） |
| 訓練不穩定 | RL 的 variance 高，loss 可能震盪 | EMA baseline 已內建；可考慮加 gradient clipping |
| V×V OOM | ISPD circuit 上跑 DDPO reward 會 OOM | Phase 1 先用 synthetic data；Phase 2 只用小 circuit |
| `output_log_prob` 未實作 | `ContinuousDiffusionModel` 不支援 | 必須先修改 `models.py`（Step 0） |
| Domain gap | Synthetic data 上的改進不一定 transfer 到 ISPD | 在 ISPD 上 eval 驗證；必要時直接在小 ISPD circuit 上 DDPO |

---

## 8. 成功指標

| 指標 | Baseline（目前結果） | 目標 |
|------|---------------------|------|
| adaptec1 HPWL (x10^5) | 10.22 | < 9.19（達到或超越 paper） |
| bigblue1 HPWL (x10^5) | 2.69 | < 2.64（超越 paper） |
| bigblue3 HPWL (x10^5) | 34.26 | < 34（維持優勢） |
| 平均 legality | >0.99 | >0.99（不能犧牲） |

---

## 9. 行動清單

- [ ] **Step 0**：修改 `ContinuousDiffusionModel.reverse_samples()` 支援 `output_log_prob`
- [ ] **Step 0b**：驗證修改後的 log_prob 計算正確（用小 batch 測試）
- [ ] **Step 1**：在 synthetic data 上跑 1000 steps DDPO，確認 loss 下降
- [ ] **Step 2**：完整 DDPO 訓練 50k steps
- [ ] **Step 3**：用 DDPO checkpoint 在 ISPD2005 上 eval，比較結果
- [ ] **Step 4**（可選）：在 ISPD 小 circuit 上直接 DDPO fine-tune
