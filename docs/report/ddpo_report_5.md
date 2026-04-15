# DDPO 第五次實驗報告 (Run 2.5)

> 計畫：`docs/plan/ddpo_plan_5.md`
> 對照：`ddpo_report_4.md`（v2.4，legality 在 all-timestep 爆炸）

## 實驗概述

用 **last-K 策略**把 local reward 限制在 reverse sampling 的最後 20 個 timestep（後 40%），避開早期 noisy predicted x₀ 的訊號干擾。

**結果：**
- ✅ 成功修正 v2.4 的 local reward 爆炸問題（量級從 1e10+ 降回 ~200）
- ❌ 但整體 HPWL 沒有改善（45.86），仍輸給 DDPO v2（44.65）和 Supervised 10k（44.01）

---

## 實驗設定

| 項目 | 設定 | vs v2.4 差異 |
|------|------|-------------|
| method | ddpo_v2_5_last_k | — |
| **local_reward_last_k** | **20** | **新增：只算 last 40%（index 30-49）** |
| local_reward_every | 5 | 同 → 實際計算 4 次 |
| local_reward composition | HPWL + 0.5×legality | 同 v2.4 |
| local_reward_weight | 0.5 | 同 |
| 其餘 | 同 DDPO v2 系列 | |
| 訓練時間 | ~60 分鐘（3604 秒） | |
| checkpoint | `v1.61-ddpo.ddpo_v2_5_last_k.61/latest.ckpt` | |

### 程式碼變更

`diffusion/ddpo.py`：
```python
# 新增參數 local_reward_last_k
start_idx = max(0, T - self.local_reward_last_k) if self.local_reward_last_k > 0 else 0
for t_idx in range(start_idx, T, self.local_reward_every):
    ...
```

`diffusion/train_graph.py`：傳遞 `local_reward_last_k` 到 DDPO constructor。

---

## 訓練曲線

| Step | Reward | Local Reward | Val Loss | Supervised Loss |
|------|--------|--------------|----------|-----------------|
| 200 | -238.4 | **-220.9** | 0.38 | 0.22 |
| 1000 | -266.0 | **-247.8** | 0.10 | 0.38 |
| 2500 | — | — | **1.51** | 1.15 |
| 5000 | -255.6 | **-235.7** | — | — |

### Last-K 成功點：Local Reward 量級穩定

| 實驗 | Local reward 量級（step 200 → step 5000） |
|------|-------------------------------------------|
| v2.3 (HPWL only, all T) | -7.7e3 → -3.5e5（45× 成長） |
| v2.4 (HPWL+leg, all T) | -1.2e6 → **-7.8e10**（65000× 成長，爆炸） |
| **v2.5 (HPWL+leg, last K=20)** | **-220.9 → -235.7（穩定）** |

Last-K 策略**徹底解決了 v2.4 的 reward 爆炸問題**。Local reward 穩定在 O(200) 量級，跟 global reward 同數量級。

### 但 Val Loss 仍然不穩

Val loss 在 step 2500 飆到 1.51，代表 catastrophic forgetting 仍在發生，只是比 v2.4 的 1.77 好一點。供 supervised loss 的 regularization 沒完全壓制 DDPO 的干擾。

---

## ISPD2005 Eval 結果（x10⁵）

### HPWL

| idx | Circuit | Baseline | DDPO v2 | v2.3 | v2.4 | **v2.5** | Sup 10k | Paper |
|-----|---------|----------|---------|------|------|----------|---------|-------|
| 0 | adaptec1 | 10.22 | **9.19** | 9.41 | 9.04 | 9.42 | 9.74 | 9.19 |
| 1 | adaptec2 | 39.06 | **29.26** | 30.75 | 34.17 | 33.31 | 32.05 | 31.0 |
| 2 | adaptec3 | 62.14 | 53.20 | 54.46 | 56.41 | 56.85 | **53.90** | 54.4 |
| 3 | adaptec4 | 60.51 | 55.19 | 53.02 | 55.57 | 54.84 | **54.16** | 54.5 |
| 4 | bigblue1 | 2.69 | **2.62** | 2.63 | 2.74 | 2.74 | 2.70 | 2.64 |
| 5 | bigblue2 | skip | 58.44 | 57.84 | **57.13** | 58.68 | 58.75 | 38.8 |
| 6 | bigblue3 | 34.26 | 34.02 | 38.67 | 32.47 | 34.95 | **30.70** | 35.9 |
| 7 | bigblue4 | 131.96 | 129.05 | 131.69 | 135.12 | **128.89** | 124.85 | 140.6 |
| **Avg (7)** | 48.69 | **44.65** | 45.80 | 46.50 | **45.86** | **44.01** | 46.89 |

### Legality

| idx | Circuit | v2.3 | v2.4 | **v2.5** |
|-----|---------|------|------|----------|
| 0 | adaptec1 | 0.9939 | 0.9941 | **0.9660** ⚠️ |
| 1 | adaptec2 | 0.9761 | 0.9707 | **0.9839** |
| 2 | adaptec3 | 0.9964 | 0.9968 | 0.9951 |
| 3 | adaptec4 | 0.9978 | 0.9985 | **0.9987** |
| 4 | bigblue1 | 0.9965 | 0.9963 | 0.9962 |
| 5 | bigblue2 | 0.9997 | 0.9996 | 0.9996 |
| 6 | bigblue3 | 0.9968 | 0.9958 | **0.9970** |
| 7 | bigblue4 | 0.9906 | 0.9907 | 0.9874 |

⚠️ **adaptec1 legality 掉到 0.9660**，比 baseline（0.9942）差很多。這是過往所有 DDPO 版本裡最低的。

---

## 分析

### Last-K 做對了什麼

1. **徹底解決 reward 爆炸**：local reward 從 v2.4 的 7.8e10 降到 v2.5 的 235。advantage normalization 能正常工作。
2. **訓練比 v2.4 稍穩**：val loss 峰值 1.51 vs v2.4 的 1.77。

### 但為什麼 HPWL 還是沒改善？

**Local reward 在「後期 timestep」的 signal 跟 global reward 實質上重複。**

- Global reward 是 final x₀（也就是 `predicted_x0_list[-1]` 或幾乎等於）的 HPWL+legality
- Local reward 在 last K=20 是 `predicted_x0_list[30..49]` 的 HPWL+legality
- 後期 `predicted_x0_list[45]` 跟 final x₀ 差異不大

這代表 v2.5 等於對「幾乎相同的 reward 」加權重 0.5 再算一次 advantage，等於把 global reward 的效果稀釋了，而不是補充新資訊。

### 為什麼 adaptec1 legality 崩了

Local reward 的 advantage 在 batch 間變化大（batch_size=2），每個 batch 的 std 很小時，advantage 被 clipping bound 放大。adaptec1 在某些 seed 下可能就會觸發這個不穩定。

### 最終結論：Local Reward 方向失敗

經過 v2.3, v2.4, v2.5 三次實驗，三種變體都**無法贏過沒有 local reward 的 DDPO v2**：

| 實驗 | 做法 | Avg HPWL | vs v2 |
|------|------|----------|-------|
| v2 | 無 local reward | **44.65** | — |
| v2.3 | HPWL only, all timestep | 45.80 | +2.6% |
| v2.4 | HPWL+leg, all timestep | 46.50 | +4.1% |
| **v2.5** | **HPWL+leg, last K=20** | **45.86** | **+2.7%** |

Local reward 對 DDPO 這個 setup **根本無益**。預期的理論收益（per-step signal）在實務上無法兌現，原因包括：
1. 早期 predicted x₀ 是 ill-defined → reward meaningless
2. 後期 predicted x₀ 跟 final x₀ 太接近 → 重複 global reward
3. 不管怎麼選 timestep，都會落在上述兩個問題之一

**決定：正式放棄 local reward 方向。**

---

## 三組 local reward 變體的完整對比

| 實驗 | local_reward_every | last_k | reward | Avg HPWL | Val loss 穩定性 |
|------|-------------------|--------|--------|----------|----------------|
| v2 | — | — | — | 44.65 | 穩定 (0.33) |
| v2.3 | 5 | 0 (all) | HPWL only | 45.80 | 尖峰 13.7 |
| v2.4 | 5 | 0 (all) | HPWL+leg | 46.50 | 爛 (1.77) |
| v2.5 | 5 | 20 | HPWL+leg | 45.86 | 中 (1.51) |

---

## 整體方法排名（Avg HPWL, 7 circuit, 不含 bigblue2）

| 排名 | 方法 | Avg HPWL | 訓練時間 |
|------|------|----------|---------|
| 1 | **Ablation Supervised 10k** | **44.01** | 17 min |
| 2 | Seed Ensemble best-per-circuit | 43.89（upper bound） | 10+ hr eval |
| 3 | DDPO v2 | 44.65 | 108 min |
| 4 | Ablation 5k | 45.43 | 12 min |
| 5 | DDPO v2.3 (local HPWL) | 45.80 | 66 min |
| 6 | **DDPO v2.5 (local HPWL+leg, last K)** | **45.86** | 60 min |
| 7 | DDPO v2.4 (local HPWL+leg, all) | 46.50 | 60 min |
| — | Paper | 46.89 | — |
| — | Baseline | 48.69 | 0 |

---

## 下一步方向

Local reward 三次實驗都沒有成功。下一步應該：
1. **放棄 local reward**
2. **探索 supervised fine-tuning 的天花板**（15k, 20k, 30k）
3. **試試 Best-of-N self-improvement**（用 seed ensemble 的 best placement 當 label）
4. **從 Ablation 10k 當 init 重跑 DDPO**（起點更好，reward signal 可能 更 productive）
5. **Cluster-aware guidance for bigblue2**（解決唯一輸給 paper 的 circuit）

完整分析見 `docs/next/ddpo_next_3.md`。

---

## 檔案位置

- Training log：`logs/ddpo_v2_5_last_k.log`
- Eval log：`logs/ddpo_v2_5_last_k_eval.log`
- Checkpoint：`logs/diffusion_debug/v1.61-ddpo.ddpo_v2_5_last_k.61/latest.ckpt`
- 程式碼變更：
  - `diffusion/ddpo.py`：新增 `local_reward_last_k` 參數
  - `diffusion/train_graph.py`：從 cfg 傳遞 `local_reward_last_k`
