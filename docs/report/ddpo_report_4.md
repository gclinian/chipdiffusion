# DDPO 第四次實驗報告 (Run 2.4)

> 計畫：`docs/plan/ddpo_plan_4.md`
> 對照：`ddpo_report_2_3.md`（v2.3, HPWL-only local reward）

## 實驗概述

把 local reward 從「只有 HPWL」改為「HPWL + 0.5×legality」（與 global reward 同組成），測試加入 legality 是否能修正 v2.3 的失敗。

**結果：不但沒變好，反而變更差。**

---

## 實驗設定

| 項目 | 設定 |
|------|------|
| method | ddpo_v2_4_local_legality |
| 唯一變動 | `local_reward_fn = reward_fn`（原本 `get_reward_fn(0.0, 1.0)`） |
| 其餘 | 同 DDPO v2.3 |
| 總訓練時間 | ~60 分鐘（3635 秒） |
| checkpoint | `logs/diffusion_debug/v1.61-ddpo.ddpo_v2_4_local_legality.61/latest.ckpt` |

**VRAM**：訓練中實測峰值 ~10 GB，遠低於 24 GB 上限。先前的 OOM 擔憂完全不成立。

---

## 訓練曲線

| Step | Reward | Local Reward | Val Loss | Supervised Loss |
|------|--------|--------------|----------|-----------------|
| 200 | -238.4 | **-1.2e6** | 0.39 | 0.22 |
| 1000 | -266.3 | **-3.4e9** | — | — |
| 5000 | -256.1 | **-7.8e10** | **1.77** | 1.25 |

### 觀察

1. **Local reward 量級爆炸**：從 step 200 的 -1.2e6 漲到 step 5000 的 -7.8e10，**五個數量級**的增長。
2. **Val loss 嚴重退化**：1.77（step 5000），比 v2 的 0.33、v2.3 的 0.32、supervised 10k 的 0.17 都糟。
3. **Supervised loss 也偏高**：1.25（v2.3 是 0.75），代表 DDPO 對 supervised 方向造成干擾。

---

## ISPD2005 Eval 結果（x10⁵）

| idx | Circuit | Baseline | DDPO v2 | v2.3 | **v2.4** | Sup 10k | Paper |
|-----|---------|----------|---------|------|----------|---------|-------|
| 0 | adaptec1 | 10.22 | 9.19 | 9.41 | **9.04** | 9.74 | 9.19 |
| 1 | adaptec2 | 39.06 | **29.26** | 30.75 | 34.17 | 32.05 | 31.0 |
| 2 | adaptec3 | 62.14 | 53.20 | 54.46 | 56.41 | **53.90** | 54.4 |
| 3 | adaptec4 | 60.51 | 55.19 | 53.02 | 55.57 | **54.16** | 54.5 |
| 4 | bigblue1 | 2.69 | **2.62** | 2.63 | 2.74 | 2.70 | 2.64 |
| 5 | bigblue2 | skip | 58.44 | 57.84 | **57.13** | 58.75 | 38.8 |
| 6 | bigblue3 | 34.26 | 34.02 | 38.67 | 32.47 | **30.70** | 35.9 |
| 7 | bigblue4 | 131.96 | 129.05 | 131.69 | 135.12 | **124.85** | 140.6 |
| **Avg (7)** | 48.69 | 44.65 | 45.80 | **46.50** | **44.01** | 46.89 |

### Legality

| idx | Circuit | v2.3 | **v2.4** |
|-----|---------|------|----------|
| 0 | adaptec1 | 0.9939 | **0.9941** |
| 1 | adaptec2 | 0.9761 | 0.9707 |
| 2 | adaptec3 | 0.9964 | **0.9968** |
| 3 | adaptec4 | 0.9978 | **0.9985** |
| 4 | bigblue1 | **0.9965** | 0.9963 |
| 5 | bigblue2 | 0.9997 | 0.9996 |
| 6 | bigblue3 | 0.9968 | 0.9958 |
| 7 | bigblue4 | 0.9906 | **0.9907** |

Legality 大致持平（沒有顯著改善也沒顯著惡化），代表 legality reward 的訊號有到，但沒有 translate 到好的整體 placement。

---

## 為什麼反而變差？

### 我的假設（錯的）

「Local reward 缺 legality 導致 HPWL-only 在早期 timestep 誤導模型擠壓 macros」

### 實際 failure mode

加入 legality 後，**legality potential 在早期 timestep 的絕對值非常大**：
- Legality_potential 是 `(relu(-l))² / 4` 對所有 pair 求和
- 早期 predicted x₀ 是 noisy 分散狀態，很多 pair 有 overlap → potential 巨大
- 所以 local_reward 量級從 1.2e6 → 3.4e9 → 7.8e10

即使我們對 advantage 做了 normalization 和 clipping，這個 reward signal 的**時間梯度過陡**（每個 step 差幾個數量級），讓 advantage 計算變得不穩定：
- batch 內每個 timestep 的 reward 分布是 heavy-tailed
- `local_reward_mean.std()` 被極端值 dominate
- advantage 雖然被 clip 到 [-5, 5]，但方向是被 outlier 決定的

### 更深層的問題

**Legality potential on noisy predictions is meaningless**。就像算 HPWL-only on noisy predictions 沒意義一樣，legality 也一樣沒意義——早期 predicted x₀ 本來就應該有高 legality potential（因為還沒 denoise 完），這不是 model 的錯，是 diffusion process 的必然階段。

我原本以為「legality 在早期有意義，HPWL 早期沒意義」，但實際上**兩者在早期都沒意義**。Predicted x₀ 本身在早期 t 就是 ill-defined 的量，不適合直接用來算 reward。

---

## 與其他方法的總比較

### 平均 HPWL（7 circuit，不含 bigblue2）

| 方法 | Avg HPWL | 訓練時間 | 備註 |
|------|----------|---------|------|
| Paper | 46.89 | — | |
| Baseline | 48.69 | 0 | |
| Ablation 5k | 45.43 | 12 min | supervised only |
| **Ablation 10k** | **44.01** | **17 min** | **最佳方法** |
| DDPO v2 | 44.65 | 108 min | PPO clipping + supervised |
| DDPO v2.3 | 45.80 | 66 min | + local HPWL |
| **DDPO v2.4** | **46.50** | **60 min** | **+ local legality, 最差 DDPO 版本** |

### 觀察

1. **Local reward 這個方向越來越糟**：無 local → 加 HPWL-only → 加 HPWL+legality，HPWL 一路從 44.65 升到 46.50。
2. **Supervised 10k 仍然稱王**：最簡單、最便宜、最好。
3. **bigblue2 全部 DDPO 版本表現接近**（57-58），都顯著差於 paper 的 38.8。瓶頸在 guidance 被關掉，不是 model 問題。

---

## 結論

### 這次學到什麼

1. **「local reward 缺 legality」不是 v2.3 失敗的真正原因**。加上 legality 後變更糟，反證我之前的分析。
2. **根本問題是 local reward 這個 design 對 diffusion model 不適合**：predicted x₀ 在早期 timestep 本質上就是 noisy estimate，對它算任何 reward 都是 meaningless。
3. **Reward signal 的量級穩定性比內容更重要**。HPWL-only local reward 的量級是跨 timestep 單調變化（-7k → -3.5e5），還算好處理。加入 legality 後量級變 heavy-tailed（-1.2e6 → -7.8e10），advantage normalization 無法有效穩定。

### Local reward 方向應該放棄

v2.3 + v2.4 兩次實驗都證明 local reward 不會幫助。原因是根本性的（predicted x₀ 本質上不適合），不是技術細節可以修正的。除非改用非常不同的 reward 定義（例如 timestep-dependent weighting、只在 last K steps 算、或 consistency-based reward），否則應該停止嘗試。

### Supervised Fine-tuning 仍然是最有效的方向

本次再次驗證：**supervised 10k > 任何 DDPO 版本**。資源應該集中在：
- 找出 supervised 的最佳 step 數（15k, 20k, 30k）
- Best-of-N self-improvement（用 model 自己生成的 best placement 當 label）
- Cluster-aware guidance 解決 bigblue2

---

## 檔案位置

- Training log：`logs/ddpo_v2_4_local_legality.log`
- Eval log：`logs/ddpo_v2_4_local_legality_eval.log`
- Checkpoint：`logs/diffusion_debug/v1.61-ddpo.ddpo_v2_4_local_legality.61/latest.ckpt`
- 程式碼變更：`diffusion/ddpo.py:18` `self.local_reward_fn = reward_fn`
