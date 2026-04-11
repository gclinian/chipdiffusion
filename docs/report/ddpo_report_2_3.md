# DDPO + Local Reward 實驗報告 (Run 2.3)

> 計畫：`docs/plan/ddpo_plan_2.md`（改進 4：Local Reward）
> 教授建議：`docs/meet/meet_0403.md`

## 實驗概述

在 DDPO v2（`ddpo_report_2_1.md`）的基礎上，加入 **local reward**：在 reverse sampling 的中間步驟，用 predicted x₀ 算 HPWL reward，提供 per-step 學習訊號。動機是 global reward 只在最後一步有訊號，1000 個 timestep 中哪幾步做對／做錯，模型無從得知。

---

## Local Reward 實作

### 核心想法

每一步 reverse sampling 都會算出 `predicted_x0 = (x_t - σ_t * ε_θ(x_t)) / α_t`，這就是模型對「乾淨樣本」的當前最佳估計。對這個 predicted x₀ 計算 HPWL，就能得到該步的 local reward。

### 修改點

**`diffusion/models.py` — `ContinuousDiffusionModel.reverse_samples()`**
- 在 `output_log_prob=True` 路徑中，每步收集 `predicted_x0.detach()` 進 `predicted_x0_list`
- Return signature 改為 `(x, intermediates, log_probs, predicted_x0_list)`

**`diffusion/ddpo.py` — `DDPO.loss()`**
- 新增 `local_reward_weight` 和 `local_reward_every` 參數
- `local_reward_fn = get_reward_fn(0.0, 1.0)`（只算 HPWL，不算 legality 以避免 V×V 矩陣 OOM）
- 每 N 個 timestep 算一次 local reward，最後取平均
- 對 local reward 做 advantage normalization（同 global）
- 最終 loss = `global_loss + local_reward_weight * local_loss`

### 為何 local reward 不算 legality？

`legality_guidance_potential()` 會建立 `(B, V, V, D)` 的距離矩陣，對於 v1.61 的 ~150 macros 已經佔用大量 VRAM。若每 5 步算一次，VRAM 會爆。所以 local reward **只用 HPWL**，legality 仍由 global reward 負責。

---

## 實驗設定

| 項目 | 設定 | 與 DDPO v2 的差異 |
|------|------|-----------------|
| method | ddpo_v2_local | — |
| batch_size | 2 | 同 |
| num_timesteps | 50 | 同 |
| lr | 1e-5 | 同 |
| train_steps | 5000 | 同 |
| clip_epsilon | 0.2 | 同 |
| supervised_weight | 1.0 | 同 |
| legality_weight | 0.5 | 同（global only） |
| hpwl_weight | 1.0 | 同 |
| **local_reward_weight** | **0.5** | **新增** |
| **local_reward_every** | **5** | **新增（每 5 步算一次）** |
| from_checkpoint | large-v2.ckpt | 同 |
| 總訓練時間 | ~66 分鐘（3942 秒） | DDPO v2 是 108 分鐘 |
| checkpoint | logs/diffusion_debug/v1.61-ddpo.ddpo_v2_local.61/ | — |

> 訓練時間反而比 DDPO v2 短，因為這次跑的是 lighter 版本（沒有兩次 reverse sampling）。

---

## 訓練曲線

| Step | Reward | Local Reward | Val Loss | Supervised Loss | Total Loss |
|------|--------|--------------|----------|-----------------|------------|
| 200  | -239.4 | -7.7e3       | —        | —               | 0.07       |
| 1000 | -273.8 | -1.2e5       | —        | —               | -0.09      |
| 2500 | -221.0 | -1.8e5       | —        | —               | 1.61       |
| 4400 | -255.7 | -2.0e5       | **13.7** | 0.23            | 0.19       |
| 5000 | -266.7 | -3.5e5       | **0.32** | 0.75            | 0.50       |

**觀察**：
- Val loss 在 step 4400 一度爆到 13.7，但隨後又恢復到 0.32（step 5000），代表訓練不穩定但 supervised loss 有把模型拉回來。
- Local reward 持續變得更負（−7.7k → −3.5e5），這個趨勢有兩個可能解讀：
  1. 模型在中間步驟的 predicted x₀ 變差了
  2. Reward 計算的縮放在不同 noise level 下變化大（值本身意義有限，看相對差才有意義）
- Global reward 沒有明顯改善（與 DDPO v2 一樣停在 -250 左右）

---

## ISPD2005 Eval 結果

> Eval log: `logs/ddpo_v2_local_eval_fix.log`
> 設定：seed=300, num_output_samples=8, skip_guidance_threshold=10000, macros_only=True

### HPWL（x10⁵，越低越好）

| idx | Circuit | Baseline | DDPO v2 | **Local (v2.3)** | Paper | Local vs DDPO v2 |
|-----|---------|----------|---------|-------------------|-------|------------------|
| 0 | adaptec1 | 10.22 | **9.19** | 9.41 | 9.19 | +2.4% 略差 |
| 1 | adaptec2 | 39.06 | **29.26** | 30.75 | 31.0 | +5.1% 略差 |
| 2 | adaptec3 | 62.14 | **53.20** | 54.46 | 54.4 | +2.4% 略差 |
| 3 | adaptec4 | 60.51 | 55.19 | **53.02** | 54.5 | **−3.9% 改善** |
| 4 | bigblue1 | 2.69 | **2.62** | 2.63 | 2.64 | 持平 |
| 5 | bigblue2 | skip | 58.44 | **57.84** | 38.8 | **−1.0% 改善** |
| 6 | bigblue3 | 34.26 | **34.02** | 38.67 | 35.9 | +13.7% 變差 |
| 7 | bigblue4 | 131.96 | **129.05** | 131.69 | 140.6 | +2.0% 略差 |
| **Avg (8)** | — | **46.37** | 47.31 | 45.9 | +2.0% |
| **Avg (7, 不含 bigblue2)** | 48.69 | **44.65** | 45.80 | 46.89 | +2.6% |

### Legality

| idx | Circuit | Baseline | DDPO v2 | **Local (v2.3)** |
|-----|---------|----------|---------|-------------------|
| 0 | adaptec1 | 0.9942 | **0.9943** | 0.9939 |
| 1 | adaptec2 | 0.9305 | **0.9791** | 0.9761 |
| 2 | adaptec3 | 0.9943 | 0.9955 | **0.9964** |
| 3 | adaptec4 | 0.9965 | 0.9982 | 0.9978 |
| 4 | bigblue1 | 0.9963 | 0.9962 | **0.9965** |
| 5 | bigblue2 | — | 0.9996 | **0.9997** |
| 6 | bigblue3 | 0.9951 | 0.9951 | **0.9968** |
| 7 | bigblue4 | 0.9914 | **0.9936** | 0.9906 |

Legality 大致持平，只有 bigblue4 略差（0.9906 vs 0.9936）。

### HPWL Ratio

| idx | Circuit | Baseline | DDPO v2 | **Local (v2.3)** |
|-----|---------|----------|---------|-------------------|
| 0 | adaptec1 | 0.718 | **0.646** | 0.661 |
| 1 | adaptec2 | 1.056 | **0.791** | 0.831 |
| 2 | adaptec3 | 0.798 | **0.683** | 0.700 |
| 3 | adaptec4 | 0.679 | 0.619 | **0.595** |
| 4 | bigblue1 | 0.822 | **0.799** | 0.803 |
| 5 | bigblue2 | — | 0.693 | **0.686** |
| 6 | bigblue3 | 0.598 | **0.594** | 0.675 |
| 7 | bigblue4 | 0.491 | **0.481** | 0.491 |

---

## 三組實驗總比較

### HPWL 平均（x10⁵）

| 平均 | Baseline | Ablation | DDPO v2 | **Local (v2.3)** | Paper |
|------|----------|----------|---------|-------------------|-------|
| 8 circuit | — | 47.38 | **46.37** | 47.31 | 45.9 |
| 7 circuit (不含 bigblue2) | 48.69 | 45.43 | **44.65** | 45.80 | 46.89 |

### 排名（每個 circuit 的最佳）

| Circuit | 第一 | 第二 | 第三 |
|---------|------|------|------|
| adaptec1 | DDPO v2 (9.19) | Local (9.41) | Ablation (9.67) |
| adaptec2 | DDPO v2 (29.26) | Local (30.75) | Ablation (34.41) |
| adaptec3 | Ablation (53.19) | DDPO v2 (53.20) | Local (54.46) |
| adaptec4 | **Local (53.02)** | Ablation (53.26) | DDPO v2 (55.19) |
| bigblue1 | DDPO v2 (2.62) | Local (2.63) | Ablation (2.68) |
| bigblue2 | **Local (57.84)** | DDPO v2 (58.44) | Ablation (61.02) |
| bigblue3 | DDPO v2 (34.02) | Ablation (37.51) | Local (38.67) |
| bigblue4 | Ablation (127.32) | DDPO v2 (129.05) | Local (131.69) |
| **第一名次數** | DDPO v2: 4 | Local: 2 | Ablation: 2 |

---

## 分析

### Local reward 的效果：弱，且不穩定

**整體 HPWL 平均反而比 DDPO v2 差（45.80 vs 44.65）**，雖然差距不大（2.6%）。Local reward 並沒有預期中「per-step 訊號」帶來的改善。

#### 為什麼 local reward 沒效？

1. **Predicted x₀ 在早期 timestep 是雜訊**
   - Reverse sampling 早期（t 接近 1000）時，`predicted_x0 = (x - σ * ε) / α` 中 σ 很大，predicted x₀ 幾乎是 random noise scaled，HPWL reward 沒有意義。
   - 對這些 step 給 reward 等於對 noise 給 reward，純 noise injection。

2. **Reward 量級在不同 t 差異巨大**
   - Step 200 的 local reward 是 -7.7k，step 5000 是 -3.5e5，差 45 倍。
   - 我們對 local reward 做了 batch-level normalization，但跨 timestep 的 scale 還是混在一起，advantage 不穩定。

3. **Local reward 跟 global reward 衝突**
   - Global reward 想優化最終 placement
   - Local reward 想優化每一步的 predicted x₀
   - 兩者在中間步驟可能拉相反方向（global 希望保留 noise 給後面細修，local 希望立刻收斂）

#### Local reward 的兩個亮點

- **adaptec4**：53.02，三組裡最低。Local reward 在這個 circuit 幫到了 DDPO v2 拿不到的提升。
- **bigblue2**：57.84，三組裡最低。Bigblue2 沒有 guidance（被 skip），完全靠模型本身的輸出，local reward 對「模型直接輸出」特別有幫助。

### 訓練不穩定

Step 4400 出現 val_loss 飆到 13.7 的尖峰，雖然之後恢復，但顯示加上 local reward 後訓練 dynamics 比 DDPO v2 不穩定。Supervised loss 把模型拉回來了，但這是 fragile 的恢復。

### 與教授建議的對應

教授在 `meet_0403.md` 建議「每個 reverse step 都要有 reward」。實作後發現：
- 想法本身有道理，但 reward 計算的位置（predicted x₀）在 noisy 階段意義不明
- 改善方向：只在後期 timestep（t < 200）算 local reward，那時 predicted x₀ 已經接近真實 placement
- 或者改為：對 final x₀ 用 trajectory 內的時間平均代替單點，但這也不是真正的「per-step」

---

## 結論

| 結論 | 說明 |
|------|------|
| Local reward 沒有帶來 net improvement | 平均 HPWL 比 DDPO v2 差 2.6% |
| 但有 circuit-specific 效益 | adaptec4, bigblue2 取得三組最低 HPWL |
| 訓練不穩定 | val_loss 出現尖峰，雖被 supervised loss 救回 |
| 實作正確但設計有缺陷 | predicted x₀ 在早期 timestep 不適合算 reward |

**建議**：放棄目前形式的 local reward。下次改進方向：
1. 只在 last K steps 算 local reward（K=10）
2. 或改成 step-decoupled reward（每 step 用獨立的 reward function）
3. 或完全放棄 local reward，把資源花在其他改進（cluster-aware guidance for bigblue2）

---

## 檔案位置

- Training log：`logs/ddpo_v2_local.log`
- Eval log（修正後）：`logs/ddpo_v2_local_eval_fix.log`
- Checkpoint：`logs/diffusion_debug/v1.61-ddpo.ddpo_v2_local.61/latest.ckpt`
- 程式碼變更：
  - `diffusion/models.py`: `reverse_samples` 收集 `predicted_x0_list`
  - `diffusion/ddpo.py`: `loss()` 加 local reward 計算
  - `diffusion/configs/mode/ddpo.yaml`: `local_reward_weight`, `local_reward_every`
