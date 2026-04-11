# DDPO Fine-tuning 實驗紀錄

## 實驗概述

對 ChipDiffusion pre-trained model (`large-v2.ckpt`) 進行 DDPO (Denoising Diffusion Policy Optimization) 強化學習微調，目標是透過 HPWL reward signal 改善 placement 品質。

---

## 程式碼修改

### 1. `diffusion/models.py` — ContinuousDiffusionModel.reverse_samples()
- 新增 `output_log_prob` 參數支援
- 在 reverse loop 中計算每步的 `pi_log_prob(x, mu, eta)`，其中 `mu` 和 `eta` 從 CosineScheduler 的參數推導
- 驗證通過：log_probs 有正確的 grad_fn，207/219 個模型參數有非零梯度

### 2. `diffusion/ddpo.py` — DDPO class
- 新增 `num_timesteps` 參數，控制 reverse sampling 的步數（預設 -1 = 使用模型預設值）
- 修改 `loss()` 方法，透過 `**extra_kwargs` 傳遞 `num_timesteps`，保持與不同 model family 的相容性

### 3. `diffusion/train_graph.py`
- 修改 DDPO 初始化，從 config 讀取 `ddpo.num_timesteps`

### 4. `diffusion/configs/mode/ddpo.yaml`
- 新增 `num_timesteps: 50`

### 5. 資料集
- 建立 `data-gen/outputs/v1.61-ddpo/`：將原本 v1.61 的 2000 val samples 重新分配為 1600 train + 400 val

---

## 實驗記錄

### Run 1：Pipeline 驗證（lr=1e-4, 200 steps）

| 項目 | 設定 |
|------|------|
| method | ddpo_v1_test |
| batch_size | 4 → OOM, 降為 2 |
| num_timesteps | 50 |
| lr | 1e-4 |
| train_steps | 200 |
| reward | hpwl_weight=1.0, legality_weight=0.0 |

**結果**：Pipeline 可跑，但 val/loss 劇烈震盪（59 → 266 → 2 → 320），模型 denoising 能力被 RL 快速破壞。lr=1e-4 太大。

**OOM 問題**：
- batch_size=4, num_timesteps=1000 → OOM（1000 步 reverse sampling 的計算圖太大）
- batch_size=2, num_timesteps=1000 → OOM
- batch_size=2, num_timesteps=50 → 可行（~10-14 GB VRAM）

### Run 2：完整訓練（lr=1e-5, 5000 steps）— 完成

| 項目 | 設定 |
|------|------|
| method | ddpo_v1_lr1e5 |
| batch_size | 2 |
| num_timesteps | 50 |
| lr | 1e-5 |
| train_steps | 5000 |
| reward | hpwl_weight=1.0, legality_weight=0.0 |
| checkpoint | logs/diffusion_debug/v1.61-ddpo.ddpo_v1_lr1e5.61/ |
| 總訓練時間 | ~61 分鐘（3683 秒） |

**完整訓練曲線**：

| Step | Reward (HPWL) | Reward EMA Mean | Val Loss | 備註 |
|------|--------------|-----------------|----------|------|
| 100  | -241.7       | -222.2          | 0.17     | 剛開始，模型完好 |
| 500  | -248.0       | -243.7          | 0.10     | 穩定 |
| 1000 | -271.9       | -248.4          | 2.03     | val loss 開始上升 |
| 1500 | -353.0       | -261.4          | 4804     | val loss 爆炸 |
| 2000 | -234.6       | -266.4          | 6057     | reward 回升但 val loss 持續惡化 |
| 3000 | -172.3       | -216.8          | 1.6e4    | reward 改善，模型嚴重偏移 |
| 4000 | -148.9       | -185.4          | 1.3e5    | val loss 爆到十萬 |
| 5000 | -102.4       | -144.5          | 264.2    | reward 持續改善，val loss 仍遠高於初始 |

**觀察**：
- **Reward 持續改善**：從 -241.7 → -102.4（HPWL 降低 58%）
- **Val loss 嚴重惡化**：從 0.17 → 高峰 1.3e5，最終回到 264（仍比初始高 1500 倍）
- **Catastrophic forgetting 確認發生**：模型為了最佳化 HPWL reward 而犧牲了 denoising 品質
- 即使 lr=1e-5，5000 steps 仍然出現嚴重 forgetting

### Run 2 ISPD2005 Eval — 完成

使用 Run 2 的 `latest.ckpt`（step 5000）在 ISPD2005 samples 0-4 上 eval。

**DDPO vs Baseline vs Paper — 完整數據**：

HPWL（x10⁵，越低越好）：

| Circuit | Baseline HPWL | DDPO HPWL | Paper HPWL | DDPO vs Baseline |
|---------|--------------|-----------|------------|-----------------|
| adaptec1 | 10.22 | 10.57 | 9.19 | +3.4% 變差 |
| adaptec2 | 39.06 | 38.81 | 31.0 | -0.6% 略好 |
| adaptec3 | 62.14 | 58.48 | 54.4 | **-5.9% 改善** |
| adaptec4 | 60.51 | 57.99 | 54.5 | **-4.2% 改善** |
| bigblue1 | 2.69 | 2.87 | 2.64 | +6.7% 變差 |

Legality（越高越好，1.0 = 完全合法）：

| Circuit | Baseline Legality | DDPO Legality | 變化 |
|---------|------------------|---------------|------|
| adaptec1 | 0.9942 | 0.9187 | **大幅下降 -7.6%** |
| adaptec2 | 0.9305 | 0.9322 | 持平 |
| adaptec3 | 0.9943 | 0.9944 | 持平 |
| adaptec4 | 0.9965 | 0.9930 | 微降 -0.4% |
| bigblue1 | 0.9963 | 0.9964 | 持平 |

HPWL Ratio（越低越好，< 1.0 表示優於 initial placement）：

| Circuit | Baseline Ratio | DDPO Ratio | 變化 |
|---------|---------------|------------|------|
| adaptec1 | 0.718 | 0.743 | 變差 |
| adaptec2 | 1.056 | 1.050 | 持平 |
| adaptec3 | 0.798 | 0.751 | **改善** |
| adaptec4 | 0.679 | 0.650 | **改善** |
| bigblue1 | 0.822 | 0.878 | 變差 |

Generation Time（秒）：

| Circuit | Baseline Time | DDPO Time | 變化 |
|---------|-------------|-----------|------|
| adaptec1 | 419.6 | 72.0 | **5.8x 加速** |
| adaptec2 | 422.8 | 71.8 | **5.9x 加速** |
| adaptec3 | 440.1 | 72.7 | **6.1x 加速** |
| adaptec4 | 406.3 | 77.4 | **5.3x 加速** |
| bigblue1 | 90.7 | 70.6 | 1.3x 加速 |

**Eval 觀察**：
- **HPWL**：adaptec3/4 有明顯改善（-4~6%），但 adaptec1 和 bigblue1 變差。整體好壞參半，仍未超越 paper
- **Legality**：adaptec1 嚴重下降（0.994 → 0.919），其餘持平。因為 DDPO 訓練時 `legality_weight=0.0`，模型不在意合法性
- **速度異常加快**：generation time 從 ~400s 降到 ~72s（5-6 倍），這不是好事——推測是 DDPO 破壞了正常的 denoising 過程，模型在 guidance 階段快速收斂到固定 pattern 而非正常迭代優化
- **結論**：DDPO 在 catastrophic forgetting 嚴重的情況下未能穩定改善所有 circuit

Log 位置：`logs/ddpo_v1_lr1e5_eval.log`
Metrics 位置：`logs/diffusion_debug/ispd2005-s0.eval_macro_only.300/metrics.csv`（已被覆寫為 DDPO 結果）

---

## 問題分析

### 1. Catastrophic Forgetting（最嚴重）
DDPO 的 reward signal 只看最終 placement 的 HPWL，不監督中間的 denoising 品質。模型學會「跳過正常 denoising，直接輸出低 HPWL 的固定 pattern」，導致 val loss 從 0.17 爆到 1.3e5。ISPD eval 結果也反映了這個問題——部分 circuit 變差，legality 下降。

### 2. num_timesteps=50 vs 1000
受限於 VRAM（24GB），DDPO 只能用 50 步 reverse sampling，但 eval 時用 1000 步。50 步 fine-tune 的效果不一定能 transfer 到 1000 步 eval。若有 A100 80GB，可以用 100-200 步，更接近 eval 設定。

### 3. 單一 Reward（只有 HPWL）
目前 legality_weight=0.0，模型可能產出 HPWL 很低但完全不合法的 placement。adaptec1 的 legality 下降（0.994 → 0.919）證實了這個問題。

### 4. VRAM 限制
batch_size=2 導致 policy gradient 的 variance 很高，訓練不穩定。理想設定需要 batch=4-8，但需要 40-80GB VRAM。

---

## GPU 資源需求

若要認真做 DDPO，建議申請更大的 GPU：

| GPU | VRAM | 可跑設定 | 預期改善 |
|-----|------|---------|---------|
| 目前 RTX 24GB | 24 GB | batch=2, ts=50 | 已驗證，效果有限 |
| A100 40GB | 40 GB | batch=4, ts=50 | batch 翻倍，訓練更穩定 |
| **A100 80GB（建議）** | **80 GB** | **batch=4, ts=100-200** | 更多 timesteps + 更大 batch + 可跑 bigblue2 |

---

## 結論與後續建議

DDPO 在目前的設定下（24GB GPU, batch=2, ts=50）效果有限，主要受 catastrophic forgetting 影響。

**如果繼續 DDPO 方向**：
1. 混合 Loss：`total_loss = supervised_loss + alpha * ddpo_loss`，防止 forgetting
2. 加入 legality reward（legality_weight > 0）
3. 申請 A100 80GB 以使用更理想的超參數
4. Early stopping：在 val loss 開始惡化前停止（~step 500）

**替代方向（不需要額外 GPU）**：
1. Multi-sample best-of-N：零改動，生成多個取最好的
2. Guidance 超參數搜索：調整現有 guidance 參數
3. Iterative clustering：已有實作，對大 circuit 有潛力
4. 詳見 `docs/plan/plan_4_3.md`
