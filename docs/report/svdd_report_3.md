# SVDD-PM 第三次實驗 Report (Phase 3: SVDD layered on paper opt guidance)

> Plan: `docs/plan/svdd_plan_3.md`
> 前次回顧：`docs/next/svdd_next_2.md`
> Report_2：`docs/report/svdd_report_2.md`
> 動機 meeting: `docs/meet/meet_0508.md`
> 執行日期：2026-05-21
> Checkpoint：`v1.61-ddpo.ablation_supervised_10k.61/latest.ckpt`
> Legalizer：`opt-adam` with `grad_descent_steps=20000`（CLAUDE.md 標準）

---

## 1. 結論摘要

| 項目 | 結果 |
|------|------|
| **Run F（重現 paper baseline）7-circuit avg HPWL** | **44.32** vs CLAUDE.md 公告 **44.01**，**Δ = +0.7%（符合 ±0.5 重現容忍）** |
| **Run E（SVDD + opt layered）7-circuit avg HPWL** | **44.49** vs Run F **44.32**，**Δ = +0.37%（持平／微差）** |
| **plan_3 §5.2 判定基準** | −0.5 ≤ +0.37 ≤ +0.5 → **「持平 → inference-time search 路線正式放棄」** |
| **核心結論** | **SVDD-PM 在 paper opt guidance 強度下沒有 marginal value** |
| **下一步** | 寫 `svdd_next_3.md` 結案，**轉教授建議 (1)「從頭 train」**，開 `from_scratch_plan_1.md` |

---

## 2. 對 paper 的真正答案

在前次 report_2 我們發現「SVDD 單獨用輸 paper」，但 plan_2 沒測 layered，所以結論還沒拍板。**Phase 3 是回答這個 missing comparison**：

| 設定（全部用 ablation_10k init + opt-adam legalizer，seed=300） | 7-circuit avg HPWL | 相對 paper baseline (Run F = 44.32) |
|--------|---:|---:|
| **Paper 方法（Run F: opt guidance）** | **44.32** | — |
| **SVDD + opt layered (Run E)** | **44.49** | **+0.37%（沒贏）** |
| ~~SVDD 單獨（Phase 2 Run C）~~ | ~~71.94~~ | ~~+62%（顯著輸）~~ |
| ~~no guidance（Phase 2 Run A）~~ | ~~84.85~~ | ~~+91%（顯著輸）~~ |

→ **SVDD 不管是替代 opt 還是疊加 opt，都沒辦法贏 paper 方法**。

---

## 3. 完整 7-circuit 結果

| idx | Circuit | V | F (opt) HPWL | E (svdd+opt) HPWL | Δ% | F legality | E legality |
|----|---------|---|---:|---:|---:|---:|---:|
| 0 | adaptec1 | 543 | 9.44 | 9.67 | **+2.41%** | 0.988 | **0.957** ↓ |
| 1 | adaptec2 | 566 | 32.23 | 32.43 | +0.62% | 0.972 | 0.983 |
| 2 | adaptec3 | 723 | 54.02 | 53.53 | -0.90% | 0.995 | 0.997 |
| 3 | adaptec4 | 1329 | 55.93 | 52.97 | **-5.30%** | 0.999 | 0.999 |
| 4 | bigblue1 | 560 | 2.74 | 2.62 | **-4.16%** | 0.996 | 0.996 |
| 6 | bigblue3 | 1298 | 30.61 | 30.83 | +0.70% | 0.997 | 0.997 |
| 7 | bigblue4 | 8170 | 125.28 | 129.35 | **+3.25%** | 0.990 | 0.992 |
| **7-circuit avg** | | | **44.32** | **44.49** | **+0.37%** | 0.991 | 0.989 |

**個別 circuit 觀察**：
- SVDD 進步 3 個（adaptec3, adaptec4, bigblue1）— 共 -10.4%
- SVDD 退步 4 個（adaptec1, adaptec2, bigblue3, bigblue4）— 共 +6.98%
- **抵銷後平均 +0.37%（無增量）**
- adaptec1 legality 從 0.988 掉到 **0.957** — SVDD 跟 opt 的 Lagrangian alpha 互相干擾的微弱跡象

---

## 4. 對照 CLAUDE.md leaderboard

| Rank | 方法 | 7-circuit avg HPWL | 來源 |
|------|------|---:|------|
| 1 | Ablation 10k (paper opt + legalizer) | **44.01** | CLAUDE.md 公告 |
| **2** | **Run F (我們重現 paper)** | **44.32** | Phase 3 |
| 3 | DDPO v2 | 44.65 | CLAUDE.md |
| **4** | **Run E (SVDD + opt layered)** | **44.49** | Phase 3 |
| 5 | AddLoss v2 | 45.24 | CLAUDE.md |
| 6 | AddLoss v1 | 45.39 | CLAUDE.md |
| 7 | Ablation 5k | 45.43 | CLAUDE.md |
| 8 | DDPO v2.3 | 45.80 | CLAUDE.md |
| 9 | DDPO v2.5 | 45.86 | CLAUDE.md |
| 10 | DDPO v2.4 | 46.50 | CLAUDE.md |
| — | Paper | 46.89 | CLAUDE.md |
| — | Baseline (large-v2) | 48.69 | CLAUDE.md |

Run E **沒進 top 2，也沒進 leaderboard 前段**（落在 Run F 跟 DDPO v2 之間）。

---

## 5. 對照 plan_3 §5.2 的判定基準

| Δ = Run E − Run F | 意義 | 行動 | 是否觸發 |
|----|----|----|----|
| Δ < −0.5 | **SVDD 對 opt 有真增量** | 寫 paper 章節；plan_4 升級 SVDD-MC | ❌ |
| **−0.5 ≤ Δ ≤ +0.5** | **持平，SVDD 對 opt 無增量** | **inference-time search 路線正式放棄；轉教授建議 (1)「從頭 train」** | ✅ **(+0.37)** |
| Δ > +0.5 | SVDD 干擾 opt | 同上：放棄 | ❌ |

**觸發「持平」結論。inference-time search 路線（SVDD/CoDe/TDS 整族）正式判定無法在 paper opt guidance 之上產生增量價值。**

---

## 6. 為什麼 SVDD layered 沒贏 opt？

### 6.1 假說 1: opt 已經把預測拉到局部最優，K candidates 失去探索空間

`opt guidance` 每步跑 20 inner SGD（+ Lagrangian alpha）把 predicted_x0 拉到「reward-friendly」區域。然後 SVDD 抽 K=4 candidate 就只是在「opt 已經選好的 mu 附近 ±η」做小幅 jitter，re-sample 也只是抽到「opt's mu ± noise」── **SVDD 沒有 search 空間**。

證據：adaptec1 legality 從 0.988 掉到 0.957 → SVDD reward 對 opt 的 alpha-tuning 有干擾，但 HPWL 沒換回好處。

### 6.2 假說 2: ablation_10k 的 raw output 已經比 large-v2 接近 opt 的目標

Phase 2 看到「SVDD 對 raw output 越差的 circuit 越有用」。ablation_10k 對 7-circuit raw（Run A guidance=none = 84.85）已經比 large-v2 + no-guidance 接近 paper。**SVDD 受益的 headroom 變小**。

→ 如果換成 large-v2 init + SVDD + opt 也許還有救？但 large-v2 + opt 公告數字 = 48.69，比 ablation_10k + opt 44.01 差 ~11%。**用更差的 init 來證明 SVDD 有用沒實質意義**。

### 6.3 假說 3: SVDD reward 跟 opt reward 是 redundant

opt's HPWL guidance 跟 SVDD 的 HPWL reward 都在優化同一個東西，**信號重複**。沒有 SVDD 帶來新資訊。
- 教科書「policy gradient + value function」是有效的因為 value 提供 variance reduction
- 我們的 SVDD-PM value = `r(\hat x_0)` 跟 opt 的 gradient guidance 用的也是同一個 reward → no extra info

---

## 7. Phase 1 → 2 → 3 全 SVDD 系列總結

| Phase | 設定 | 7-circuit avg | 結論 |
|---|---|---:|---|
| 1 | SVDD raw, large-v2, no legalization, 2 circuits, multi-seed | adaptec1: -0.7%, bigblue1: -3.0% (within noise) | 機制 OK，single-circuit 訊號弱 |
| 2 | SVDD only, ablation_10k, opt-adam legalization, 7 circuits | 71.94 (vs no-guidance 84.85: **-15.2%**) | 贏 no-guidance，但輸 paper opt |
| **3** | **SVDD + opt layered, ablation_10k, opt-adam, 7 circuits** | **44.49 (vs paper opt 44.32: +0.37%)** | **持平，無增量** |

**結論**：SVDD-PM 在 chipdiffusion 這個任務上**沒有 production value**。
- 機制本身可運作
- 對沒 fine-tune / 沒 opt guidance 的弱 baseline 有用
- 對 paper 完整 setup（ablation_10k + opt）邊際效用 = 0

---

## 8. 計算成本

| Run | 7-circuit 總時間 | 相對 cost |
|-----|---|---|
| Run F (opt only) | ~64 min | 1.0× |
| Run E (svdd + opt layered) | ~70 min | 1.1× |

Generation time per circuit:
- adaptec1: F 133s → E 116s（layered actually FASTER, possibly opt convergence sped up?）
- bigblue4 (V=8170): F 2297s → E 2545s (+11%)

SVDD 的 cost 是可負擔的（K=4 forward），但既然沒贏，不必再投入。

---

## 9. 跑出來的 bug / 限制

1. **沒踩到 GPU 0 contention**（預設 `CUDA_VISIBLE_DEVICES=1` 就避開）
2. **沒踩到 K=8 batched bug**（K=4 沒事）
3. **adaptec1 legality drop 0.988→0.957** — SVDD reward × opt Lagrangian 互相干擾的微弱跡象。要救的話：把 `svdd_lambda_legality=0`（純 HPWL reward）讓 legality 完全交給 opt 處理。但既然平均沒贏，**不打算試**。

---

## 10. 行動清單

- [x] Run F: opt baseline 重現 → 44.32 ≈ 44.01 ✅
- [x] Run E: SVDD + opt layered → 44.49 = +0.37% vs F → no incremental value
- [x] 寫此 report
- [ ] 下一步：寫 `docs/next/svdd_next_3.md` ── 結案 SVDD 系列
- [ ] 下一步：開 `docs/plan/from_scratch_plan_1.md` ── 教授建議 (1)「從頭 train」track

---

## 11. 一句話結論

**SVDD-PM 機制可運作，但在 chipdiffusion 任務上 — 不管單獨用還是疊加 paper 的 opt guidance — 都沒有 marginal value。Phase 3 確認 SVDD + opt = 44.49 vs paper opt = 44.32（+0.37%, 持平）。Inference-time search 整個方向（SVDD/CoDe/TDS）正式判定 dead end，下一步轉教授建議 (1)「從頭 train」。**
