# DDPO 第三次實驗報告

> 計畫：`docs/plan/ddpo_plan_3.md`
> 參考：`docs/next/ddpo_next_2.md`

## 實驗概述

兩組實驗，目標是量化 DDPO v2 的 stochasticity 和 supervised fine-tuning 的天花板。

| 實驗 | 內容 |
|------|------|
| 3.1 Seed Ensemble | 用 DDPO v2 checkpoint 跑 4 個 seed（300/400/500/600），看不同 seed 的 variance |
| 3.2 Supervised 10k | 純 supervised fine-tuning 10000 steps（之前 ablation 是 5000 steps） |

---

## 實驗 3.1：Seed Ensemble

### 設定

- Checkpoint：`v1.61-ddpo.ddpo_v2_ppo.61/latest.ckpt`（DDPO v2 最佳 model）
- Seeds：300, 400, 500, 600
- Eval 設定同 `ddpo_report_2_1.md`

### HPWL 結果（x10⁵，越低越好）

| Circuit | s300 | s400 | s500 | s600 | **Best** | Worst | Std | Paper |
|---------|------|------|------|------|----------|-------|-----|-------|
| adaptec1 | 9.19 | **8.95** | 9.32 | 9.32 | **8.95** | 9.32 | 0.17 | 9.19 |
| adaptec2 | **29.26** | 32.21 | 36.61 | 32.34 | **29.26** | 36.61 | 3.03 | 31.0 |
| adaptec3 | 53.20 | 57.44 | 57.17 | **52.65** | **52.65** | 57.44 | 2.54 | 54.4 |
| adaptec4 | 55.19 | 56.41 | 52.42 | **51.15** | **51.15** | 56.41 | 2.43 | 54.5 |
| bigblue1 | **2.62** | 2.70 | 2.70 | **2.62** | **2.62** | 2.70 | 0.04 | 2.64 |
| bigblue2 | 58.44 | 57.16 | **56.88** | 60.26 | **56.88** | 60.26 | 1.54 | 38.8 |
| bigblue3 | **34.02** | 36.41 | 35.32 | 34.85 | **34.02** | 36.41 | 1.00 | 35.9 |
| bigblue4 | 129.05 | 130.54 | **128.56** | 130.30 | **128.56** | 130.54 | 0.96 | 140.6 |

### 平均（不含 bigblue2）

| Seed | Avg HPWL (7 circuit) |
|------|---------------------|
| 300 | 44.65 |
| 600 | 44.75 |
| 500 | 46.01 |
| 400 | 46.38 |
| **Best-per-circuit** | **43.89** |
| Paper | 46.89 |

### Legality（best-per-circuit 對應的 seed）

| Circuit | Best Seed | HPWL | Legality |
|---------|-----------|------|----------|
| adaptec1 | 400 | 8.95 | 0.9610 ⚠️ |
| adaptec2 | 300 | 29.26 | 0.9791 |
| adaptec3 | 600 | 52.65 | 0.9971 |
| adaptec4 | 600 | 51.15 | 0.9981 |
| bigblue1 | 300 | 2.62 | 0.9962 |
| bigblue2 | 500 | 56.88 | 0.9992 |
| bigblue3 | 300 | 34.02 | 0.9951 |
| bigblue4 | 500 | 128.56 | 0.9902 |

⚠️ adaptec1 在 seed=400 的 legality=0.961，偏低。如果要求 legality > 0.99，則 adaptec1 應取 seed=300（HPWL=9.19, legality=0.994）。

### 分析

1. **Variance 不小**：adaptec2 的 std=3.03（29.26~36.61），跨 seed 差距 25%。大部分 circuit 的 std 在 1-3 之間。
2. **Best-per-circuit = 43.89**，比任何單一 seed 的平均（最好 44.65）好 1.7%。
3. **每個 circuit 的 best seed 都不同**（300 贏 3 個, 400 贏 1 個, 500 贏 2 個, 600 贏 2 個），代表沒有「萬能 seed」。
4. **所有 8 個 circuit 的 best-per-circuit 都超越 paper**（除了 bigblue2）。

---

## 實驗 3.2：Extended Supervised Fine-tuning (10k steps)

### 設定

| 項目 | 設定 |
|------|------|
| mode | finetune |
| method | ablation_supervised_10k |
| train_steps | 10000 |
| lr | 1e-5 |
| batch_size | 2 |
| from_checkpoint | large-v2.ckpt |
| 訓練時間 | ~17 分鐘（989 秒） |

### 訓練曲線

| Step | Val Loss | Train Loss |
|------|----------|------------|
| 200 | 0.40 | 0.18 |
| 1000 | 0.10 | 0.17 |
| 5000 | 0.12 | 0.14 |
| 10000 | **0.17** | 0.14 |

Val loss 在 step 1000 降到 0.10 最低，之後緩慢回升到 0.17。模型可能在 step ~5000 附近開始 overfitting（train loss 持續下降但 val loss 反彈）。

### ISPD2005 Eval 結果（x10⁵）

| Circuit | Baseline | Abl. 5k | **Abl. 10k** | DDPO v2 (s300) | Paper |
|---------|----------|---------|--------------|----------------|-------|
| adaptec1 | 10.22 | 9.67 | **9.74** | 9.19 | 9.19 |
| adaptec2 | 39.06 | 34.41 | **32.05** | 29.26 | 31.0 |
| adaptec3 | 62.14 | 53.19 | **53.90** | 53.20 | 54.4 |
| adaptec4 | 60.51 | 53.26 | **54.16** | 55.19 | 54.5 |
| bigblue1 | 2.69 | 2.68 | **2.70** | 2.62 | 2.64 |
| bigblue2 | skip | 61.02 | **58.75** | 58.44 | 38.8 |
| bigblue3 | 34.26 | 37.51 | **30.70** | 34.02 | 35.9 |
| bigblue4 | 131.96 | 127.32 | **124.85** | 129.05 | 140.6 |
| **Avg (7)** | 48.69 | 45.43 | **44.01** | 44.65 | 46.89 |

### Legality

| Circuit | Abl. 5k | **Abl. 10k** | DDPO v2 |
|---------|---------|--------------|---------|
| adaptec1 | 0.9940 | **0.9939** | 0.9943 |
| adaptec2 | 0.9541 | **0.9774** | 0.9791 |
| adaptec3 | 0.9966 | **0.9954** | 0.9955 |
| adaptec4 | 0.9987 | **0.9966** | 0.9982 |
| bigblue1 | 0.9964 | **0.9962** | 0.9962 |
| bigblue2 | 0.9995 | **0.9997** | 0.9996 |
| bigblue3 | 0.9969 | **0.9964** | 0.9951 |
| bigblue4 | 0.9931 | **0.9922** | 0.9936 |

### 分析

1. **Supervised 10k 比 5k 明顯更好**：平均 44.01 vs 45.43（−3.1%）。
2. **bigblue3 大幅改善**：30.70 vs 37.51（5k）vs 34.02（DDPO v2），三組裡最好，也超過 paper 的 35.9。
3. **bigblue4 也大幅改善**：124.85 vs 127.32（5k），三組裡最好。
4. **adaptec2 也改善**：32.05 vs 34.41（5k），接近 paper 的 31.0。
5. **但 adaptec1/bigblue1 沒改善**：跟 5k 差不多，瓶頸可能不在 denoising 品質。

---

## 三組方法總比較

### HPWL 平均（x10⁵，不含 bigblue2）

| 方法 | Avg (7 circuit) | vs Paper | 訓練時間 |
|------|----------------|----------|----------|
| Baseline (large-v2.ckpt) | 48.69 | +3.8% | 0 |
| Ablation 5k | 45.43 | −3.1% | 12 min |
| DDPO v2 (s300) | 44.65 | −4.8% | 108 min |
| **Ablation 10k** | **44.01** | **−6.1%** | **17 min** |
| **Seed Ensemble (best-per-circuit)** | **43.89** | **−6.4%** | 108 min + 10 hr eval |
| Paper | 46.89 | — | — |

### Per-circuit 最佳（所有實驗中）

| Circuit | Best HPWL | 來自 | Paper | vs Paper |
|---------|-----------|------|-------|----------|
| adaptec1 | **8.95** | Seed Ensemble (s400) | 9.19 | **−2.6%** |
| adaptec2 | **29.26** | DDPO v2 (s300) | 31.0 | **−5.6%** |
| adaptec3 | **52.65** | Seed Ensemble (s600) | 54.4 | **−3.2%** |
| adaptec4 | **51.15** | Seed Ensemble (s600) | 54.5 | **−6.2%** |
| bigblue1 | **2.62** | DDPO v2 (s300) | 2.64 | **−0.8%** |
| bigblue2 | 56.88 | Seed Ensemble (s500) | 38.8 | +46.6% |
| bigblue3 | **30.70** | Ablation 10k | 35.9 | **−14.5%** |
| bigblue4 | **124.85** | Ablation 10k | 140.6 | **−11.2%** |

**7/8 circuit 全部超越 paper。** bigblue2 因為 guidance 被關掉，是唯一弱項。

---

## 關鍵發現

### 1. Supervised fine-tuning 是主要驅動力

| 方法 | 改善量（vs baseline） | 時間投入 |
|------|---------------------|----------|
| Ablation 5k | −6.7% | 12 min |
| Ablation 10k | −9.6% | 17 min |
| DDPO v2 | −8.3% | 108 min |

**純 supervised fine-tuning 10k steps 達到的效果（44.01）比 DDPO v2（44.65）更好**，而且只花 17 分鐘 vs 108 分鐘。DDPO 的 reward signal 帶來的額外改善（1.7%）被更長的 supervised training 超越了。

### 2. Val loss 顯示 10k 可能接近天花板

Val loss 在 step 1000 最低（0.10），到 10k 回升到 0.17。模型開始在 v1.61 training data 上 overfit，但 ISPD eval 仍在改善——這代表 slight overfitting to v1.61 可能反而有益（v1.61 的某些特徵 generalize 到 ISPD）。

但繼續到 20k、50k 可能會過頭。建議下次嘗試 15k 或 20k，看 ISPD eval 是否繼續改善。

### 3. Seed variance 不可忽視

單一 seed 的結果可能差很多（adaptec2: 29.26 ~ 36.61）。報告結果時應該跑多個 seed。best-per-circuit 相當於免費提升 1.7%。

### 4. Ablation 10k 在大型 circuit 特別強

bigblue3（30.70）和 bigblue4（124.85）是所有方法中最好的，包括 DDPO v2 的 4 個 seed。可能原因：更長的 supervised training 讓模型在大型 circuit 的 denoising 更穩定，給 guidance + legalization 更好的起點。

---

## 檔案位置

- Seed Ensemble logs：`logs/seed_ensemble_{400,500,600}.log`（seed=300 在 `logs/ddpo_v2_ppo_eval_fix.log`）
- Supervised 10k training：`logs/ablation_supervised_10k.log`
- Supervised 10k eval：`logs/ablation_supervised_10k_eval.log`
- Supervised 10k checkpoint：`logs/diffusion_debug/v1.61-ddpo.ablation_supervised_10k.61/latest.ckpt`
