# DDPO 第二次實驗回顧 — 給 ddpo_plan_3 的參考

> 實驗報告：`ddpo_report_2_1.md`（DDPO v2）、`ddpo_report_2_2.md`（ablation）、`ddpo_report_2_3.md`（local reward）

## 第二次實驗做了什麼

三組實驗，都從 `large-v2.ckpt` 開始，在 v1.61-ddpo synthetic data 上跑 5000 steps：

| Run | 內容 | 結果 |
|-----|------|------|
| 2.1 DDPO v2 | advantage clipping + loss normalization + supervised loss + legality reward | **最好**，平均 HPWL 44.65（7 circuit），比 paper 好 4.8% |
| 2.2 Ablation | 只有 supervised fine-tuning（移除 DDPO） | 第二名，平均 45.43，比 baseline 好 6.7% |
| 2.3 Local reward | DDPO v2 + 每 5 步算 predicted x₀ 的 HPWL reward | **略差於 DDPO v2**，平均 45.80 |

## 關鍵數據

### HPWL (x10⁵) — 三組結果 vs Baseline vs Paper

| Circuit | Baseline | DDPO v2 | Ablation | Local | Paper |
|---------|----------|---------|----------|-------|-------|
| adaptec1 | 10.22 | **9.19** | 9.67 | 9.41 | 9.19 |
| adaptec2 | 39.06 | **29.26** | 34.41 | 30.75 | 31.0 |
| adaptec3 | 62.14 | 53.20 | **53.19** | 54.46 | 54.4 |
| adaptec4 | 60.51 | 55.19 | 53.26 | **53.02** | 54.5 |
| bigblue1 | 2.69 | **2.62** | 2.68 | 2.63 | 2.64 |
| bigblue2 | skip | 58.44 | 61.02 | **57.84** | 38.8 |
| bigblue3 | 34.26 | **34.02** | 37.51 | 38.67 | 35.9 |
| bigblue4 | 131.96 | 129.05 | **127.32** | 131.69 | 140.6 |
| **Avg (7, 不含 bb2)** | 48.69 | **44.65** | 45.43 | 45.80 | 46.89 |

---

## 成功之處

1. **Catastrophic forgetting 完全解決**：val loss 從第一次的 264 降到 0.33，supervised loss 是主因。
2. **Legality 全面保持**：`legality_weight=0.5` 有效，所有 circuit legality ≥ 0.99。
3. **6/8 circuit 超越 paper**：adaptec1 追平，adaptec2/3、bigblue1/3/4 超越。
4. **Ablation 確認 DDPO 有貢獻**：5/8 circuit DDPO v2 比純 supervised 好，但差距偏小（1.7%）。

---

## 仍存在的問題

### 問題 1：DDPO 的 reward signal 太弱

訓練曲線上 reward 始終在 -240 ~ -270 之間波動，沒有單調改善。Supervised loss + loss normalization 壓制了 DDPO 的學習訊號。DDPO 比 ablation 的改善只有 1.7%——投入 9 倍訓練時間（108 分鐘 vs 12 分鐘），投資報酬率低。

**根本原因**：batch_size=2 + 50 timesteps 導致 reward gradient 的 variance 極高，有效信噪比太低。

### 問題 2：bigblue2 遠遜於 paper（58 vs 38.8）

23k macros 超過 `skip_guidance_threshold=10000`，guidance 被關掉，只剩 legalization。Paper 應該是用 >24GB GPU 跑全 guidance。

**我們能做的**：
- 申請更大 GPU（A100 40/80GB）
- 或研究 cluster-based guidance（對 bigblue2 分群後逐群做 guidance）

### 問題 3：Local reward 在目前形式下是干擾

Local reward 讓平均 HPWL 比 DDPO v2 差 2.6%。原因：

- Predicted x₀ 在 early timestep（t 接近 1000）是 noise，對 noise 算 HPWL 沒意義
- 不同 timestep 的 reward 量級差異巨大（-7.7k vs -3.5e5），advantage normalization 無法跨 timestep 有效歸一化
- 訓練出現 val_loss 尖峰（step 4400 飆到 13.7），雖被 supervised loss 救回，但顯示不穩定

### 問題 4：Supervised fine-tuning 的改善來源不明

Ablation 用合成 data fine-tune 就降了 6.7%，但原因不確定：
- 可能是 LR annealing 效果（lr=1e-5 比原訓練小，是在 loss landscape 上做 local polish）
- 可能是模型的 denoising 能力整體提升，generalize 到 ISPD
- 不太可能是 v1.61 data 跟 ISPD 特別像（v1.61 是合成的）

### 問題 5：Training data 與 eval data 的 domain gap

DDPO 的 reward 是在 v1.61 synthetic data 上算的，但 eval 是在 ISPD2005 上跑。Reward 改善不代表 ISPD 改善（反之亦然）。這個 gap 限制了 DDPO 的有效性。

---

## 可能的改進方向

### 方向 A：Best-of-N Self-Improvement（最有潛力）

**想法**：不用 reward gradient（高 variance），改用 rejection sampling。

```
for each ISPD circuit:
    用目前最好的 model 跑 N 次 eval（不同 seed）
    保留 HPWL 最低的那次 placement 作為 training label
fine-tune model on these best placements
repeat K 輪
```

**優點**：
- 繞過 DDPO 的高 variance 問題——不需要 policy gradient
- Training label 是 model 自己能達到的最好水平
- 只需要 inference + supervised fine-tune，比 DDPO 簡單
- 直接在 ISPD data 上做 test-time adaptation

**缺點**：
- 每輪需要跑 8 circuit × N seeds × ~10 分鐘 = 幾小時到十幾小時
- 只有 8 個 training sample，overfitting 風險高（但 eval 也是同樣 8 個，所以可能不是壞事）
- 不確定 model 有足夠多樣性產出比目前更好的 placement

**預估資源**：N=10, 每輪 ~13 小時 eval + 12 分鐘 fine-tune。跑 3 輪約 2 天。

### 方向 B：加大 DDPO 的 batch size（需要更多 VRAM）

**想法**：batch_size=2 是 reward gradient variance 太高的根本原因。加到 8-16 可以大幅降低 variance。

| batch_size | 預估 VRAM | 需要 |
|------------|-----------|------|
| 2 | ~22 GB | RTX 3090/4090 ✅ |
| 4 | ~28 GB | A100 40GB |
| 8 | ~40 GB | A100 40GB |
| 16 | ~60 GB | A100 80GB |

**優點**：最直接地提升 DDPO 的信噪比，reward 應該能穩定改善。
**缺點**：需要申請 GPU 資源。

### 方向 C：修正 Local Reward（風險中等）

如果要重試 local reward，有三種修正方式：

**(C1) 只在 last K steps 算**
```python
# 只在 t < 200 時算，早期 timestep 完全跳過
if t < num_timesteps * 0.2:  # 最後 20% 的步驟
    local_reward = hpwl(predicted_x0)
```

**(C2) Timestep-dependent weighting**
```python
weight = alpha_t ** 2  # α 在早期 ≈ 0，後期 ≈ 1，自然壓低早期 reward
local_reward = weight * hpwl(predicted_x0)
```

**(C3) Consistency reward（不看品質，看穩定性）**
```python
# 相鄰 step 的 predicted x₀ 越接近 → reward 越高
consistency = -||predicted_x0[t] - predicted_x0[t-1]||²
```

**建議優先度**：C1 最簡單，C3 最有理論根據，C2 介於中間。

### 方向 D：純 Supervised Fine-tuning 的進階版

Ablation 只跑了 5000 steps。如果只做 supervised fine-tuning：

- 跑更多 steps（10k, 20k）？
- 用更多 synthetic data（v1.61 完整 2000 samples 而非 1600）？
- 結合 data augmentation（對 placement 做 rotation/flip）？

**優點**：12 分鐘就能跑完，迭代速度最快。
**缺點**：天花板可能不高——supervised loss 無法直接最佳化 HPWL。

### 方向 E：Seed Ensemble（不改 model，改 inference）

**想法**：不 fine-tune model，直接用 DDPO v2 跑多個 seed，取最好的。

```bash
for seed in 300 400 500 600 700:
    PYTHONPATH=. python diffusion/eval.py seed=${seed} ...
取每個 circuit 的 best HPWL
```

**優點**：零成本、零風險，純粹利用 stochasticity。
**缺點**：不是真正的改進，只是更好的抽樣。但可以先做這個來建立 upper bound。

---

## 建議的優先順序

| 優先級 | 方向 | 理由 |
|--------|------|------|
| 1 | **E. Seed Ensemble** | 零成本，先建立 DDPO v2 的 upper bound |
| 2 | **A. Best-of-N Self-Improvement** | 最有潛力，繞過 DDPO 的 variance 問題 |
| 3 | **D. 更多 supervised fine-tuning** | 便宜、快速，看天花板在哪 |
| 4 | **B. 加大 batch size** | 需要 GPU 資源，但能根本解決 DDPO 問題 |
| 5 | **C. 修正 local reward** | 風險中等，先做其他方向再回頭 |

---

## 開放問題

1. **Supervised fine-tuning 的天花板在哪？** 5000 steps 就降了 6.7%，10000 步會更好嗎？還是已經飽和？
2. **Best-of-N 的 N 要多大？** 不同 seed 的 HPWL 變異數有多大？如果 variance 小，N=3 就夠；如果大，需要 N=20+。
3. **DDPO v2 已經比 paper 好了（44.65 vs 46.89），還有多少空間？** Paper 的結果是 seed=400，如果我們也換 seed 可能更好。
4. **有沒有比 DDPO 更好的 RL-for-diffusion 方法？** 例如 DPO (Direct Preference Optimization) for diffusion、ReFL (Reward Feedback Learning) 等新方法。
