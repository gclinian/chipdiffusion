# DDPO Ablation 實驗報告 (Run 2.2)

> 目的：驗證 DDPO v2 (`ddpo_report_2_1.md`) 的改善是來自 DDPO reward signal，還是純粹來自 supervised fine-tuning 的 regularization 效果。
>
> ⚠️ **本報告為修正版**：先前版本的 ISPD eval 因 `eval.py` 的 checkpoint path bug，沒有實際載入 fine-tuned checkpoint。本版改用 `from_checkpoint=v1.61-ddpo.ablation_supervised_only.61/latest.ckpt`，eval log 已確認 `successfully loaded state dict for model`。

## 實驗設計

| | DDPO v2 (Run 2.1) | Ablation (Run 2.2) |
|---|---|---|
| Supervised loss | ✅ weight=1.0 | ✅ weight=1.0（唯一的 loss） |
| DDPO loss | ✅ (advantage clipping + normalization) | ❌ 移除 |
| Legality reward | ✅ weight=0.5 | ❌ 無 |
| mode | ddpo | finetune |
| 其餘設定 | lr=1e-5, batch=2, 5000 steps | 相同 |

兩者都從 `large-v2.ckpt` 開始，用 v1.61 synthetic data 跑 5000 steps。差異只在於有沒有 DDPO 的 reward signal。

---

## 訓練結果

| 指標 | DDPO v2 | Ablation |
|------|---------|----------|
| Val loss (Step 200) | 0.30 | 0.24 |
| Val loss (Step 1000) | 0.11 | 0.13 |
| Val loss (Step 2500) | 2.90 | 0.17 |
| Val loss (Step 5000) | **0.33** | **0.11** |
| Total loss (Step 5000) | 0.57 | 0.17 |
| 訓練時間 | 108 分鐘 | **12 分鐘**（9× faster） |

Ablation 的 val loss 更穩定（0.11 vs 0.33），訓練時間也快 9 倍——因為沒有 reverse sampling 的 50 個 timestep loop。

---

## ISPD2005 Eval 結果

> Eval log: `logs/ablation_supervised_eval_fix.log`
> 設定：seed=300, num_output_samples=8, skip_guidance_threshold=10000, macros_only=True

### HPWL（x10⁵，越低越好）

| idx | Circuit | Baseline | DDPO v2 | **Ablation** | Paper | DDPO 是否比 Ablation 好? |
|-----|---------|----------|---------|--------------|-------|------------------------|
| 0 | adaptec1 | 10.22 | **9.19** | 9.67 | 9.19 | **✅ DDPO −5.0%** |
| 1 | adaptec2 | 39.06 | **29.26** | 34.41 | 31.0 | **✅ DDPO −15.0%** |
| 2 | adaptec3 | 62.14 | 53.20 | **53.19** | 54.4 | 持平 |
| 3 | adaptec4 | 60.51 | 55.19 | **53.26** | 54.5 | ❌ Ablation 更好 |
| 4 | bigblue1 | 2.69 | **2.62** | 2.68 | 2.64 | **✅ DDPO −2.2%** |
| 5 | bigblue2 | skip | **58.44** | 61.02 | 38.8 | **✅ DDPO −4.2%** |
| 6 | bigblue3 | 34.26 | **34.02** | 37.51 | 35.9 | **✅ DDPO −9.3%** |
| 7 | bigblue4 | 131.96 | 129.05 | **127.32** | 140.6 | ❌ Ablation 更好 |
| **Avg (8)** | — | **46.37** | 47.38 | 45.9 | DDPO 平均 −2.1% |
| **Avg (7, 不含 bigblue2)** | 48.69 | **44.65** | 45.43 | 46.89 | DDPO −1.7% |

### Legality

| idx | Circuit | Baseline | DDPO v2 | Ablation |
|-----|---------|----------|---------|----------|
| 0 | adaptec1 | 0.9942 | **0.9943** | 0.9940 |
| 1 | adaptec2 | 0.9305 | **0.9791** | 0.9541 |
| 2 | adaptec3 | 0.9943 | 0.9955 | **0.9966** |
| 3 | adaptec4 | 0.9965 | 0.9982 | **0.9987** |
| 4 | bigblue1 | 0.9963 | 0.9962 | **0.9964** |
| 5 | bigblue2 | — | **0.9996** | 0.9995 |
| 6 | bigblue3 | 0.9951 | 0.9951 | **0.9969** |
| 7 | bigblue4 | 0.9914 | **0.9936** | 0.9931 |

DDPO v2 的 legality 在 4/8 比 Ablation 好，3/8 略差但差距小（< 0.005）。adaptec2 是最大差距：DDPO v2 = 0.9791 vs Ablation = 0.9541，這是 legality reward 起作用的明顯證據。

### HPWL Ratio

| idx | Circuit | Baseline | DDPO v2 | Ablation |
|-----|---------|----------|---------|----------|
| 0 | adaptec1 | 0.718 | **0.646** | 0.679 |
| 1 | adaptec2 | 1.056 | **0.791** | 0.931 |
| 2 | adaptec3 | 0.798 | 0.683 | **0.683** |
| 3 | adaptec4 | 0.679 | 0.619 | **0.598** |
| 4 | bigblue1 | 0.822 | **0.799** | 0.818 |
| 5 | bigblue2 | — | **0.693** | 0.724 |
| 6 | bigblue3 | 0.598 | **0.594** | 0.655 |
| 7 | bigblue4 | 0.491 | 0.481 | **0.474** |

---

## 分析

### DDPO 有貢獻嗎？

**有，而且貢獻明確。** 在 8 個 circuit 中：

- **DDPO v2 比 Ablation 好的有 5 個**：adaptec1, adaptec2, bigblue1, bigblue2, bigblue3
- **Ablation 略好的有 2 個**：adaptec4, bigblue4
- **持平 1 個**：adaptec3

**最具說服力的證據：adaptec2**
- DDPO v2 HPWL = 29.26，Ablation HPWL = 34.41（DDPO 好 15%）
- DDPO v2 Legality = 0.9791，Ablation Legality = 0.9541（DDPO 好 2.6%）
- 同時 HPWL 和 legality 都好，這是 DDPO 的 reward signal（HPWL + legality）的直接效果

### Ablation（純 supervised fine-tuning）的效果

純 supervised fine-tuning **本身也已經超越 baseline**，但比 DDPO v2 略遜：

| 指標 | Baseline | Ablation | 改善 |
|------|----------|----------|------|
| Avg HPWL (7 circuit) | 48.69 | 45.43 | **−6.7%** |

這代表 v1.61 synthetic data 含有 ISPD-like 的特徵，光是繼續 fine-tune 就有幫助。

### Ablation 比 DDPO v2 好的兩個 circuit

- **adaptec4**：Ablation 53.26 vs DDPO v2 55.19。Ablation 的 legality 也更高（0.9987 vs 0.9982）。
- **bigblue4**：Ablation 127.32 vs DDPO v2 129.05。

可能原因：DDPO 的 reward signal 來自 v1.61 sample 的 reverse sampling，這些 sample 的特徵與 adaptec4/bigblue4 有偏差，導致 DDPO 的 gradient 推離 optimal 方向。

### 結論

| 結論 | 說明 |
|------|------|
| DDPO reward 確實有貢獻 | 5/8 circuit 優於 ablation，平均 HPWL 好 1.7% |
| Supervised fine-tuning 是 baseline 改善的主因 | 即使沒有 DDPO，平均 HPWL 也降 6.7% |
| Legality reward 是 DDPO 的關鍵 | adaptec2 legality 從 0.954（ablation）→ 0.979（DDPO） |
| Ablation 的 cost 極低 | 12 分鐘 vs 108 分鐘，效益比很好 |

**Take-away**：DDPO 的學習訊號偏弱（相對於 supervised loss），但對 legality 有明顯影響。如果只追求 HPWL，pure supervised fine-tuning 已經 80% 達到 DDPO 的效果。

---

## 檔案位置

- Training log：`logs/ablation_supervised_only.log`
- Eval log（修正後）：`logs/ablation_supervised_eval_fix.log`
- Checkpoint：`logs/diffusion_debug/v1.61-ddpo.ablation_supervised_only.61/latest.ckpt`
