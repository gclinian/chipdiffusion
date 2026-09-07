# AddLoss 第二次實驗報告（Timestep Weighting）

> 計畫：`docs/plan/addloss_plan_2.md`
> 前次：`docs/report/addloss_report_1.md`（AddLoss v1, 45.39）
> 對照：`docs/report/ddpo_report_3.md`（Ablation 10k, 44.01, 目前最佳）

## 實驗概述

在 AddLoss v1 的 HPWL + legality auxiliary loss 基礎上，**加入 timestep weighting（`weight(t) = α_t²`）**，讓 aux loss 在早期 timestep（pure-noise phase）自動衰減，只在晚期（predicted_x0 接近 ground truth 時）滿權重參與。目標是修正 v1 在大 circuit（bigblue2 +13%, bigblue4 +7.4%, adaptec4 +3.7%）的退步。

**結論**：訓練指標全面改善（val_loss 0.21→0.18, hpwl_loss 149→103, legality_loss 1.93→0.18），**ISPD eval 平均 45.24，略優於 v1（45.39, -0.33%），但仍輸給 Ablation 10k（44.01, +2.8%）**。落在計畫判定的「比 v1 好但沒贏 Ablation 10k」區間（44.5–45.39），timestep weighting 只部分成功：**三個最大 circuit（bigblue2/bigblue4/adaptec4）全部改善，但中等 circuit（adaptec3, bigblue3）出現新退步**。

---

## 實驗設定

| 項目 | v1 | **v2** | 差異 |
|------|----|----|------|
| from_checkpoint | `../public-models/large-v2/large-v2.ckpt` | 同 | 同 |
| train_steps | 10000 | 同 | 同 |
| lr | 1e-5 | 同 | 同 |
| batch_size | 2 | 同 | 同 |
| hpwl_weight | 1e-3 | **1e-3** | **同**（為乾淨比較 timestep weighting 的效果）|
| legality_weight | 1e-2 | **1e-2** | **同** |
| **use_timestep_weighting** | False | **True** | **新加入** |
| 訓練時間 | ~18 min | ~20 min | +10%（多一次 `alpha(t)` call） |

### 程式碼變更

`diffusion/models.py` `ContinuousDiffusionModel.loss()`：
```python
if use_timestep_weighting:
    alpha_t_1d = self._noise_scheduler.alpha(t)  # (B,)
    weights = alpha_t_1d ** 2  # (B,), early t→0, late t→1
else:
    weights = torch.ones_like(t)
hpwl_loss = (weights * hpwl_per_sample).mean()
legality_loss = (weights * legality_per_sample).mean()
```

`diffusion/train_graph.py` 讀 `cfg.addloss.use_timestep_weighting` 並傳遞。

---

## 訓練曲線（vs v1）

| Metric | v1 (step 10000) | **v2 (step 10000)** | 變化 |
|--------|----------------|----------------------|------|
| `hpwl_loss` | 149.20 | **~103.62** | **-30.5%**（α² 平均 ≈ 0.5，衰減如預期）|
| `legality_loss` | 1.93 | **~0.18** | **-90%**（早期 noisy legality 被壓掉）|
| `val/loss` | 0.21 | **0.18** | **-14%**（denoising 干擾減少）|
| 訓練穩定性 | ✅ 穩定 | ✅ 穩定 | — |

### 訓練層面觀察

1. **Timestep weighting 按設計運作**：aux loss 的 effective scale ≈ v1 × 0.5，符合 `E[α²] ≈ 0.5` 的預測。
2. **Val loss 下降到 Ablation 10k 水準**（v2: 0.18, Ablation 10k: 0.17, v1: 0.21）——證實計畫中「aux loss 早期干擾 denoising」的假說。
3. **Legality loss 下降 90%** 是因為 ground truth 本來就合法，早期 noisy predicted_x0 的 legality potential 本來就很高（噪音放大），被 α² 壓下後平均值大幅下降。

---

## ISPD2005 Eval 結果（x10⁵）

Seed=300, 8 samples, macros_only=True, guidance/legalizer=opt-adam（與 v1、Ablation 10k 一致）。

### HPWL 對照表

| idx | Circuit  | Macros | Baseline | Ablation 10k | AddLoss v1 | **AddLoss v2** | DDPO v2 | Paper |
|-----|----------|-------:|---------:|-------------:|-----------:|---------------:|--------:|------:|
| 0 | adaptec1 | 543    | 10.22 | 9.74 | 9.47 | **9.15** ✅ | 9.19 | 9.19 |
| 1 | adaptec2 | 566    | 39.06 | 32.05 | 33.54 | **31.58** ✅ | 29.26 | 31.0 |
| 2 | adaptec3 | 723    | 62.14 | 53.90 | **51.60** | 53.67 ❌ | 53.20 | 54.4 |
| 3 | adaptec4 | 1,329  | 60.51 | 54.16 | 56.14 | **54.88** ✅ | 55.19 | 54.5 |
| 4 | bigblue1 | 560    | 2.69  | 2.70 | **2.59** | 2.65 ❌ | 2.62 | 2.64 |
| 5 | bigblue2 | 23,084 | skip  | 58.75 | 66.49 | **65.49** ✅ | 58.44 | 38.8 |
| 6 | bigblue3 | 1,298  | 34.26 | **30.70** | 30.32 | 34.91 ❌ | 34.02 | 35.9 |
| 7 | bigblue4 | 8,170  | 131.96 | 124.85 | 134.05 | **129.81** ✅ | 129.05 | 140.6 |
| **Avg (7, no bb2)** |    | 48.69 | **44.01** | 45.39 | **45.24** | 44.65 | 46.89 |
| **Avg (8)**         |    | —     | 45.86 | 48.03 | **47.77** | 46.37 | 45.9 |

- **v2 vs v1**: -0.15 (-0.33%) — 微幅改善
- **v2 vs Ablation 10k**: +1.23 (+2.8%) — 仍輸
- **v2 vs Paper**: -1.65 (-3.5%) — 贏 paper
- **v2 win/loss vs v1**: 5 勝 3 敗（adaptec1/2/4, bigblue2/4 改善；adaptec3, bigblue1, bigblue3 退步）

### Legality 對照表

| idx | Circuit | Ablation 10k | AddLoss v1 | **AddLoss v2** |
|-----|---------|-------------:|-----------:|---------------:|
| 0 | adaptec1 | 0.9940 | 0.9943 | 0.9910 |
| 1 | adaptec2 | 0.9774 | 0.9694 | **0.9843** ✅ |
| 2 | adaptec3 | 0.9954 | 0.9970 | 0.9971 |
| 3 | adaptec4 | 0.9966 | 0.9987 | 0.9984 |
| 4 | bigblue1 | 0.9962 | 0.9965 | 0.9964 |
| 5 | bigblue2 | 0.9997 | 0.9996 | 0.9997 |
| 6 | bigblue3 | 0.9964 | 0.9973 | 0.9965 |
| 7 | bigblue4 | 0.9922 | 0.9908 | 0.9901 |

Legality 大致持平，**adaptec2 顯著改善**（0.9694 → 0.9843, +1.5%），其餘差異 <0.5%。

---

## 核心分析：Timestep Weighting 是否修正了 v1 的退步？

### 計畫假設（addloss_plan_2.md）

> 大 circuit 上 predicted_x0 的 noise 影響更大，早期 timestep 的「假訊號」更有破壞力。用 `α_t²` 讓早期 timestep 自動衰減，應能修正大 circuit 退步。

### 驗證：v1 的 4 個「退步 circuit」在 v2 的結果

| Circuit (macros) | v1    | v2    | Δ       | 假設預測 | 結果 |
|------------------|-------|-------|---------|---------|------|
| bigblue2 (23k)   | 66.49 | 65.49 | **-1.5%** | 大幅改善 | ✅ 改善（但遠不到計畫期望的 58-60）|
| bigblue4 (8.2k)  | 134.05| 129.81| **-3.2%** | 應改善 | ✅ 改善 |
| adaptec4 (1.3k)  | 56.14 | 54.88 | **-2.2%** | 應改善 | ✅ 改善 |
| adaptec2 (566)   | 33.54 | 31.58 | **-5.8%** | 小 circuit 應持平 | ✅ 大幅改善（意外）|

**四個 v1 退步的 circuit 全部改善**——假設基本成立。但改善幅度小於計畫期望值。

### 新退步：v1 贏的 circuit 在 v2 的結果

| Circuit (macros) | v1    | v2    | Δ       | 原因推測 |
|------------------|-------|-------|---------|---------|
| adaptec3 (723)   | 51.60 | 53.67 | **+4.0%** | 中 circuit，aux 衰減過頭 |
| bigblue1 (560)   | 2.59  | 2.65  | **+2.3%** | 小 circuit，α² 過度衰減了有用訊號 |
| bigblue3 (1,298) | 30.32 | 34.91 | **+15.1%** | **最嚴重退步**——原因待調查 |

**三個 v1 贏的 circuit 都退步，其中 bigblue3 退步 15%**。這是 v2 沒預料到的副作用。

### 綜合：timestep weighting 的 trade-off

Timestep weighting 的效果不是單調的：
- **大 circuit 上**（bigblue2/4, adaptec4）：晚期的 HPWL signal 是正訊號，衰減早期 noise 讓 model 學到「乾淨版」的 HPWL 優化 → 改善。
- **中等 circuit 上**（adaptec3, bigblue3）：v1 可能本來就在一個剛好的平衡點，衰減早期 signal 反而破壞了這個平衡 → 退步。
- **bigblue3 15% 退步** 是最大異常，可能跟它的 connectivity 分佈特殊有關（macro-only eval 的結果對 pin distribution 很敏感）。

---

## 整體方法排名（更新後）

### 7 circuit 平均 HPWL（不含 bigblue2）

| 排名 | 方法 | Avg HPWL | 訓練時間 |
|-----:|------|---------:|---------:|
| 1 | **Ablation 10k** | **44.01** | 17 min |
| 2 | DDPO v2 | 44.65 | 108 min |
| 3 | **AddLoss v2** | **45.24** | 20 min |
| 4 | AddLoss v1 | 45.39 | 18 min |
| 5 | Ablation 5k | 45.43 | 12 min |
| 6 | DDPO v2.3 (local HPWL) | 45.80 | 66 min |
| 7 | DDPO v2.5 (last-K local) | 45.86 | 60 min |
| 8 | DDPO v2.4 (local HPWL+leg) | 46.50 | 60 min |
| — | Paper | 46.89 | — |
| — | Baseline | 48.69 | 0 |

AddLoss v2 從 v1 的第 4 名上升到第 3 名，但仍無法撼動 Ablation 10k 的領先。

---

## 計畫判定 → 結論

按 `addloss_plan_2.md` 的判定標準：

| 區間 | 結果判讀 | 建議 |
|------|---------|------|
| Avg < 44.01 | timestep weighting 是關鍵 | 調高 weight、改 init |
| 44.01–44.5 | 部分有效 | 試 hpwl_weight=2e-3 補償 |
| **44.5–45.39** | **比 v1 好但沒贏 Ablation 10k** | **調整方向繼續** |
| ≥ 45.39 | AddLoss 方向無效 | 徹底放棄 |

**我們落在 44.5–45.39 區間（45.24）**。依計畫建議「調整方向繼續」，但增益極小（-0.33%）——這比「marginal」還輕。需要判斷是否值得再投資。

### 好消息

1. **訓練假設完全驗證**：val_loss 0.18（v1: 0.21），證實「aux loss 干擾 denoising」為真，timestep weighting 修正了這點。
2. **大 circuit 退步修正有效**：bigblue2/4 + adaptec4 全部改善。
3. **Avg 排名進步**：從第 4 升到第 3，首次超越 Ablation 5k。

### 壞消息

1. **新退步出現**：bigblue3 (+15.1%), adaptec3 (+4.0%), bigblue1 (+2.3%) 。**timestep weighting 不是純粹的改善**，存在明顯 trade-off。
2. **仍輸 Ablation 10k 2.8%**：純 supervised 還是最強。AddLoss 的整個方向性貢獻（5-min of work beyond supervised）只有 -1.5%（44.01 → 45.24 - 0 = 相對化）。
3. **bigblue2 遠未達目標**：計畫預期 58-60，實際 65.49，只改善 1.5%。大型 circuit 的主要瓶頸不在 aux loss 而在 legalization 本身（V²記憶體限制 + 收斂困難）。

---

## 下一步建議（優先序）

### 方向 A：放棄 AddLoss，轉 Best-of-N / self-improve（推薦）
- AddLoss 兩版本（v1, v2）都沒贏 Ablation 10k。direct gradient on ground-truth target 基本確認是**冗餘訊號**。
- **Best-of-N self-improvement**：seed ensemble 選最佳 → 當新 supervised label → 再訓。已知 seed ensemble best-per-circuit 可達 43.89，理論上限比 44.01 還低 0.3%。

### 方向 B：AddLoss + Ablation 10k init（低優先）
- 從 Ablation 10k checkpoint 再 fine-tune 5k-10k 加 AddLoss（疊加）。
- 機會成本低（20 min），但既然 v2 增益 <1%，疊加也很可能只擠出 <0.5% → 不如方向 A。

### 方向 C：hpwl_weight=2e-3 補償 α² 衰減（不推薦）
- 計畫原本備案，但 v2 結果顯示「退步換進步」不是單純 weight 問題，而是 timestep 分佈的結構性 trade-off。加 weight 只會放大兩邊。

### 方向 D：bigblue2 cluster-aware（獨立方向）
- Paper 的 bigblue2 用 guidance + 大 GPU（我們 24GB OOM）。需要 cluster-level hierarchy，改動範圍大。獨立於 AddLoss，可與 A 並行。

### 最終判斷
**AddLoss 方向已經充分探索，實際收益很小（v1/v2 都輸 supervised-only）。建議正式結束 AddLoss，轉向 Best-of-N（方向 A）或 bigblue2 cluster-aware（方向 D）。**

---

## 檔案位置

- Training log：`logs/addloss_v2.log`（10k steps, val_loss 0.18）
- Eval log：`logs/addloss_v2_eval.log`（8/8 samples completed Apr 15 20:50）
- Checkpoint：`logs/diffusion_debug/v1.61-ddpo.addloss_v2.61/latest.ckpt`
- Eval output：`logs/diffusion_debug/ispd2005-s0.eval_macro_only.300/metrics.csv`
  - 注意：這個 dir 的 metrics.csv **已被 addloss_v2 eval 覆寫**（原 baseline metrics 保留在 `metrics0-4.csv` + `metrics6-7.csv`）
- 程式碼變更（commit `40a396d`）：
  - `diffusion/models.py`：`loss()` 加 `use_timestep_weighting` 參數
  - `diffusion/train_graph.py`：從 `cfg.addloss.use_timestep_weighting` 讀取並傳遞
