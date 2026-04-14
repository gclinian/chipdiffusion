# DDPO 第三次實驗計畫

> 參考：`docs/next/ddpo_next_2.md`（第二次實驗回顧）

## 目標

DDPO v2 已經在 7 個 ISPD2005 circuit 上平均超越 paper（44.65 vs 46.89）。本次實驗目標：
1. 用 Seed Ensemble 建立 DDPO v2 的 upper bound
2. 測試 extended supervised fine-tuning 的天花板
3. 綜合分析，決定下一步方向

---

## 實驗 3.1：Seed Ensemble

**動機**：Eval 的 reverse sampling 從 random noise 開始，不同 seed 會產生不同結果。目前只有 seed=300 的單次結果，variance 不清楚。跑多個 seed 取 per-circuit best，可以建立「不改 model、只靠抽樣」的 upper bound。

**做法**：用 DDPO v2 checkpoint 跑 3 個額外 seed（400, 500, 600），加上已有的 300，共 4 個 seed。

| 項目 | 設定 |
|------|------|
| checkpoint | `v1.61-ddpo.ddpo_v2_ppo.61/latest.ckpt` |
| seeds | 300（已有）, 400, 500, 600 |
| num_output_samples | 8（含 bigblue2） |
| 其餘設定 | 同 ddpo_report_2_1 的 eval |

**預估時間**：每個 seed ~2.5 小時（bigblue2 的 legalization 佔 ~100 分鐘），3 個新 seed 共 ~7.5 小時。

**產出**：
- Per-circuit best HPWL（seed ensemble 的 upper bound）
- Per-circuit HPWL 的 variance（判斷 stochasticity 有多大）

---

## 實驗 3.2：Extended Supervised Fine-tuning

**動機**：Ablation（5000 steps）就降了 6.7%，val loss 仍在下降（0.11）。如果增加到 10k steps，model 可能繼續改善。這是最 cost-effective 的改進。

**做法**：supervised-only fine-tuning，10000 steps。

| 項目 | 設定 |
|------|------|
| mode | finetune |
| method | ablation_supervised_10k |
| train_steps | 10000 |
| lr | 1e-5 |
| batch_size | 2 |
| from_checkpoint | `../public-models/large-v2/large-v2.ckpt` |
| data | v1.61-ddpo（1600 train / 400 val） |

**預估時間**：訓練 ~24 分鐘 + eval ~2.5 小時。

**觀察重點**：
- val loss 是否繼續下降？還是已經飽和？
- ISPD eval 是否比 5k ablation 更好？

---

## 執行順序

所有實驗串接在同一個 tmux session 中依序執行：

```
1. Seed Ensemble: seed=400 eval  (~2.5h)
2. Seed Ensemble: seed=500 eval  (~2.5h)
3. Seed Ensemble: seed=600 eval  (~2.5h)
4. Supervised 10k: training      (~24min)
5. Supervised 10k: eval          (~2.5h)
```

總計約 ~13 小時。

---

## 成功指標

| 指標 | 目標 |
|------|------|
| Seed Ensemble best avg (7 circuit) | < 44.0（比 DDPO v2 single-seed 的 44.65 好） |
| Supervised 10k avg (7 circuit) | < 45.0（比 5k ablation 的 45.43 好） |
| 每個 circuit 的 seed variance (std) | 了解 stochasticity 量級 |
