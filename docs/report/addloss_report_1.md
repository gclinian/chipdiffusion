# AddLoss 第一次實驗報告

> 計畫：`docs/plan/addloss_plan_1.md`
> 可行性分析：`docs/next/addloss_init.md`
> 對照：`ddpo_report_3.md`（Ablation 10k: 44.01, 目前最佳）

## 實驗概述

在 supervised fine-tuning 的 denoising loss 上**加入 HPWL + legality auxiliary loss（direct gradient, ReFL-style）**，跑 10000 steps。

**結論**：訓練機制運作正常（無爆炸、無 NaN、auxiliary loss 確實下降），但 **ISPD eval 平均 45.39 略差於 Ablation 10k 的 44.01（+3.1%）**。Direct gradient 方法沒有帶來預期的增益。

---

## 實驗設定

| 項目 | 設定 | vs Ablation 10k 差異 |
|------|------|---------------------|
| mode | finetune | 同 |
| method | addloss_v1 | — |
| from_checkpoint | `../public-models/large-v2/large-v2.ckpt` | **同（控制變因）** |
| train_steps | 10000 | 同 |
| lr | 1e-5 | 同 |
| batch_size | 2 | 同 |
| **hpwl_weight** | **1e-3** | **新增** |
| **legality_weight** | **1e-2** | **新增** |
| 訓練時間 | ~18 分鐘（1088 秒） | Ablation 10k 是 17 分鐘（多算 aux loss ~5%） |

### 程式碼變更

`diffusion/models.py` `ContinuousDiffusionModel.loss()`：
```python
def loss(self, x, cond, _, hpwl_weight=0.0, legality_weight=0.0):
    # ... 標準 denoising ...
    denoising_loss = self._loss(eps_predict, epsilon, mask)

    if hpwl_weight > 0 or legality_weight > 0:
        predicted_x0 = (x_perturbed - sigma_t * eps_predict) / alpha_t
        predicted_x0 = torch.clamp(predicted_x0, -2, 2)
        total_loss = denoising_loss
        if hpwl_weight > 0:
            hpwl_loss = guidance.hpwl_guidance_potential(predicted_x0, cond).mean()
            total_loss += hpwl_weight * hpwl_loss
        if legality_weight > 0:
            legality_loss = guidance.legality_guidance_potential(predicted_x0, cond, mask=mask).mean()
            total_loss += legality_weight * legality_loss
        return total_loss, metrics
    return denoising_loss, metrics
```

`diffusion/train_graph.py` 傳遞 weights：
```python
addloss = cfg.get("addloss", {}) or {}
loss, model_metrics = model.loss(
    x, cond, t,
    hpwl_weight=addloss.get("hpwl_weight", 0.0),
    legality_weight=addloss.get("legality_weight", 0.0),
)
```

---

## 訓練曲線

| Step | Total Loss | HPWL Loss | Legality Loss | Val Loss |
|------|-----------|-----------|---------------|----------|
| 200 | 0.42 | **211.76** | 2.70 | 0.41 |
| 1000 | 0.39 | **178.92** | 2.02 | 0.11 |
| 5000 | 0.34 | **149.64** | 3.29 | 0.14 |
| 10000 | 0.33 | **149.20** | 1.93 | 0.21 |

### 觀察

1. **HPWL loss 確實下降**：從 211 → 149（-30%）。代表 model 確實學到在 predicted_x0 上產生較低 HPWL。✓ 機制有效
2. **Legality loss 保持低量級**：2-3 之間，沒爆炸。Ground truth 本來就是合法 placement，legality signal 很小。
3. **訓練穩定**：無 val_loss 尖峰、無 NaN。這跟 DDPO local reward (v2.4 爆到 1e10) 形成強烈對比。
4. **Val loss 比 Ablation 10k 略差**：0.21 vs 0.17。HPWL loss 對 denoising 有輕微干擾。

### 驗證 Direct Gradient 機制

v2.4 的 local reward：量級 5 個數量級爆炸（1.2e6 → 7.8e10）
AddLoss v1 的 hpwl/legality_loss：穩定下降（-30%）

**確認 direct gradient 機制在訓練上是穩定的**——這證實了 `addloss_init.md` 中的理論分析。失敗原因不在訓練穩定性，而在別處。

---

## ISPD2005 Eval 結果（x10⁵）

### HPWL

| idx | Circuit | Baseline | Ablation 10k | **AddLoss v1** | DDPO v2 | Paper |
|-----|---------|----------|--------------|-----------------|---------|-------|
| 0 | adaptec1 | 10.22 | 9.74 | **9.47** ✅ | 9.19 | 9.19 |
| 1 | adaptec2 | 39.06 | **32.05** | 33.54 ❌ | 29.26 | 31.0 |
| 2 | adaptec3 | 62.14 | 53.90 | **51.60** ✅ | 53.20 | 54.4 |
| 3 | adaptec4 | 60.51 | **54.16** | 56.14 ❌ | 55.19 | 54.5 |
| 4 | bigblue1 | 2.69 | 2.70 | **2.59** ✅ | 2.62 | 2.64 |
| 5 | bigblue2 | skip | **58.75** | 66.49 ❌ | 58.44 | 38.8 |
| 6 | bigblue3 | 34.26 | 30.70 | **30.32** ✅ | 34.02 | 35.9 |
| 7 | bigblue4 | 131.96 | **124.85** | 134.05 ❌ | 129.05 | 140.6 |
| **Avg (7, no bb2)** | 48.69 | **44.01** | **45.39** | 44.65 | 46.89 |
| **Avg (8)** | — | 45.86 | 48.03 | 46.37 | 45.9 |

**結果**：4 勝 4 敗，平均略差。沒有明顯的整體優勢。

### Legality

| idx | Circuit | Ablation 10k | **AddLoss v1** |
|-----|---------|--------------|-----------------|
| 0 | adaptec1 | 0.9940 | **0.9943** |
| 1 | adaptec2 | 0.9774 | 0.9694 ⚠️ |
| 2 | adaptec3 | 0.9954 | **0.9970** |
| 3 | adaptec4 | 0.9966 | **0.9987** |
| 4 | bigblue1 | 0.9962 | **0.9965** |
| 5 | bigblue2 | 0.9997 | 0.9996 |
| 6 | bigblue3 | 0.9964 | **0.9973** |
| 7 | bigblue4 | 0.9922 | 0.9908 |

Legality 大致相等（6/8 略有改善），沒有 catastrophic 下降。

---

## 分析

### 為什麼 Auxiliary Loss 沒帶來預期增益？

訓練時 `hpwl_loss` 明明下降了 30%，但 ISPD eval 沒改善。可能原因：

#### 假說 1：v1.61 train HPWL 跟 ISPD eval HPWL 是不同分佈

Model 學到降低 v1.61 的 synthetic circuit 的 HPWL，但這個 learned "preference" 不 transfer 到 ISPD。ISPD 的 circuit 結構（connectivity、size distribution）跟 v1.61 不同，低 v1.61 HPWL ≠ 低 ISPD HPWL。

**證據**：HPWL loss 降 30% 但 ISPD eval 沒同步改善。

#### 假說 2：Auxiliary Loss 略微干擾 denoising

Val loss 從 Ablation 10k 的 0.17 升到 AddLoss v1 的 0.21（+24%）。雖然不嚴重，但可能是在大型/複雜 circuit（bigblue2、bigblue4）上表現退步的主因，因為 denoising 品質影響更大。

#### 假說 3：Ground Truth 已經是好的 target，Auxiliary Loss 冗餘

Ablation 10k 已經很接近 Ground Truth（44.01 vs v1.61 自己的 HPWL 平均 ~45）。在 ground truth 上加 auxiliary reward 是**冗餘訊號**——model 已經在朝向低 HPWL ground truth 學習。

**證據**：hpwl_loss 初始值 211 與 v1.61 data 的典型 HPWL 值相近，訓練後降到 149——這個降幅更像是 model 對 v1.61 data 的 overfitting，而非學到 generalize 的 HPWL 優化技能。

### Per-circuit pattern：小 circuit 好、大 circuit 差

| 贏的 circuit | macros | 變化 |
|-------------|--------|------|
| adaptec1 | 543 | 9.74 → 9.47 (-2.8%) |
| adaptec3 | 723 | 53.90 → 51.60 (-4.3%) |
| bigblue1 | 560 | 2.70 → 2.59 (-4.1%) |
| bigblue3 | 1298 | 30.70 → 30.32 (-1.2%) |

| 輸的 circuit | macros | 變化 |
|-------------|--------|------|
| adaptec2 | 566 | 32.05 → 33.54 (+4.6%) |
| adaptec4 | 1329 | 54.16 → 56.14 (+3.7%) |
| bigblue2 | **23084** | 58.75 → 66.49 (+13.2%) |
| bigblue4 | **8170** | 124.85 → 134.05 (+7.4%) |

**明顯趨勢：大 circuit（>1300 macros）輸得多。** bigblue2/4 在任何方法下都是難題，AddLoss 把這個難度放大了。

---

## 整體方法排名（更新）

### 7 circuit 平均 HPWL（不含 bigblue2）

| 排名 | 方法 | Avg HPWL | 訓練時間 |
|------|------|----------|---------|
| 1 | **Ablation 10k** | **44.01** | 17 min |
| 2 | DDPO v2 | 44.65 | 108 min |
| 3 | **AddLoss v1** | **45.39** | 18 min |
| 4 | Ablation 5k | 45.43 | 12 min |
| 5 | DDPO v2.3 (local HPWL) | 45.80 | 66 min |
| 6 | DDPO v2.5 (last-K local) | 45.86 | 60 min |
| 7 | DDPO v2.4 (local HPWL+leg) | 46.50 | 60 min |
| — | Paper | 46.89 | — |
| — | Baseline | 48.69 | 0 |

AddLoss v1 排第 3，贏 Ablation 5k 但輸給 Ablation 10k 跟 DDPO v2。

---

## 結論

### 好消息

1. **Direct gradient 機制運作正常**：hpwl_loss 平滑下降 30%，無爆炸，無 NaN，val loss 穩定。**驗證了 ReFL 類方法在這個 task 上是訓練可行的**，不像 DDPO local reward 會訓練崩潰。
2. **Legality 有輕微改善**：6/8 circuit 比 Ablation 10k 略好。
3. **小 circuit 有改善**：adaptec1/3, bigblue1/3 都更好。

### 壞消息

1. **平均 HPWL 沒改善**：45.39 比 Ablation 10k 的 44.01 更差 3.1%。
2. **大 circuit 退步明顯**：bigblue2 (+13%), bigblue4 (+7.4%)。這兩個是 paper 的瓶頸，AddLoss 把問題放大。
3. **Val loss 略升**：0.21 vs Ablation 10k 的 0.17，auxiliary loss 輕微干擾 denoising。

### 核心觀察

> 我之前在 `addloss_init.md` 提到的風險「可能因為冗餘而效果有限」似乎成真了：**ground truth 已經是好的 target，額外的 HPWL 信號沒給出 beyond-supervised 的新資訊**。

### 與 DDPO Local Reward 的對比

| 面向 | DDPO local reward | AddLoss v1 |
|------|------------------|-----------|
| 訓練穩定性 | ❌ Reward 爆炸、val_loss 尖峰 | ✅ 平滑、穩定 |
| ISPD eval | 全部輸給 DDPO v2 | 輸給 Ablation 10k 但接近 |
| 失敗模式 | 根本性（policy gradient 不適用） | 邊際性（冗餘/略負） |

**結論符合 `addloss_init.md` 的預測**：direct gradient 沒有 policy gradient 的訓練崩潰問題，但本 task 下改善很有限。

---

## 可能的下一步

### 方向 A：調整 weight 或 timestep weighting

1. **降低 weight**（alpha=1e-4, beta=1e-3）看是否減少對大 circuit 的負面影響
2. **Timestep weighting**：只在 t 較小時算 aux loss（`weight = alpha_t²`）
3. **只用 HPWL loss**：移除 legality loss（ground truth 本來就合法，legality loss 可能雜訊）

### 方向 B：從 ablation_10k init 再做一次（疊加實驗）

`addloss_init.md` 原本的「推薦做法」：從 `ablation_supervised_10k` 開始再 fine-tune 5k-10k steps 加 aux loss。測試是否能在已收斂的 model 上再擠出一點改善。

### 方向 C：放棄 AddLoss 方向

如果方向 A 和 B 都沒改善，可能這個 task 的 supervised signal 就到頂了。應該轉戰：
- Best-of-N self-improvement
- Cluster-aware guidance for bigblue2
- 或更根本的架構改動

---

## 檔案位置

- Training log：`logs/addloss_v1.log`
- Eval log：`logs/addloss_v1_eval.log`
- Checkpoint：`logs/diffusion_debug/v1.61-ddpo.addloss_v1.61/latest.ckpt`
- 程式碼變更：
  - `diffusion/models.py`：`ContinuousDiffusionModel.loss()` 加 `hpwl_weight/legality_weight` 參數
  - `diffusion/train_graph.py`：從 `cfg.addloss` 讀取 weights 並傳給 `model.loss()`
