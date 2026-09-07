> # ⚠️ 更正聲明 CORRECTION（2026-09-07）— 本報告核心結論已被推翻
>
> **本報告 §1 / §5 / §11 宣告的「inference-time search（SVDD/CoDe/TDS 整族）正式判定 dead end」是錯的，已被 Phase 4-5 推翻。** 依專案慣例，report 是歷史紀錄、**不改寫**，以下原文全部保留，僅加註更正。
>
> **根本原因（framing error）**：Phase 3 的 Run F（44.321）用的是**使用者自己 fine-tune 出來的 `ablation_supervised_10k` checkpoint**，卻在本報告裡被當成「paper baseline / paper 方法」。Paper 真正的 baseline 是 **`large-v2` + opt = 46.89（paper 公告, seed 400）／ 48.691（我們環境重現, seed 300）**；44.01 / 44.321 是我們自己贏 paper ~6% 的結果。等於拿 SVDD 去挑戰一個比 paper 強 9% 的自家最佳 checkpoint —— 比出「持平」當然**不能**推論成「對 paper 沒有增量」。
>
> **在 paper 真正的 checkpoint（large-v2）上重跑之後**（數字由 `docs/all_experiments_per_circuit.csv` 重新計算確認）：
>
> | 方法（large-v2 + opt-adam legalization） | 7-circuit avg HPWL | seeds | vs paper 46.89 |
> |---|---:|---|---|
> | **SVDD_layered** | **44.845 ± 0.654** | 45.097 / 44.103 / 45.335 | **3/3 seeds 全贏** |
> | TDS_layered | 45.081 ± 0.376 | 44.973 / 45.499 / 44.772 | 3/3 seeds 全贏 |
> | CoDe_layered | 45.216 ± 0.469 | 45.361 / 44.692 / 45.596 | 3/3 seeds 全贏 |
> | 純 opt（我們重現 paper baseline） | 48.691 | 300 (n=1) | — |
>
> → 本報告 §5 觸發的「正式放棄」判定**不成立**；被判死刑的 SVDD/CoDe/TDS **三族在 paper checkpoint 上全部贏 paper**。
>
> **依據**：`docs/report/svdd_report_4.md` §7（明文指出本報告 §1 / §11 的措辭必須修正）、`docs/report/svdd_report_5.md` §8（「**完全推翻**」）、`docs/next/svdd_next_3.md` §1-4。
>
> **仍然成立的部分**：Phase 3 的程式碼、執行與數字本身沒有錯（CSV 重算 Run F = 44.321、Run E = 44.485，與內文 44.32 / 44.49 一致）。**「在已經 fine-tune 過的 ablation_10k 上，SVDD 疊加在 opt 之上沒有增量（+0.37%）」這句仍是有效結論**（headroom 假說）。錯的只是把它外推成「整個 inference-time search 方向 dead end」。
>
> **受影響章節**：§1（判定基準列 / 核心結論列 / 下一步列）、§2、§5、§7（「沒有 production value」）、§11（一句話結論）── 一律以 `docs/report/svdd_report_5.md` 為準。
>
> **小 caveat**（給後續引用者）：翻盤結論是 n=3。`svdd_report_5.md` §2.2 的 95% CI 用了 z=1.96 算成 [44.11, 45.59]；n=3 正確的 t 區間（t.975,df=2 = 4.303）是 **[43.22, 46.47]**。上界仍 < 46.89，**結論不變**，但區間比 report_5 寫的寬，引用時請用 t 區間。

---

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

> ⚠️ **[已更正 2026-09-07]** 下表第 3-5 列（「持平 → inference-time search 路線正式放棄」、「SVDD-PM 在 paper opt guidance 強度下沒有 marginal value」、下一步「轉從頭 train」）**已被推翻**。Run F 44.32 是 `ablation_10k`（我們自己 fine-tune 的 ckpt），**不是 paper baseline**；paper baseline = `large-v2` + opt = 46.89（公告）／ 48.691（我們重現）。在 large-v2 上 SVDD_layered = **44.845 ± 0.654（3 seeds，3/3 贏 paper）**。見頁首更正聲明與 `docs/report/svdd_report_5.md` §8。

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

> ⚠️ **[已更正 2026-09-07]** 這段判定**已被推翻**。plan_3 §5.2 的整套判定基準建立在「Run F = paper baseline」這個錯誤前提上 —— Run F 用的是自家 fine-tuned 的 `ablation_10k`，不是 paper 的 `large-v2`。因此 Δ=+0.37% 只能支持「**在 ablation_10k 這個已接近 floor 的 ckpt 上** SVDD 無增量」，**不能**外推到「inference-time search 整族無價值」。在 paper 真正的 large-v2 上：SVDD_layered 44.845 ± 0.654、TDS_layered 45.081 ± 0.376、CoDe_layered 45.216 ± 0.469，**三族各自 3/3 seeds 全贏 paper 46.89**。見 `docs/report/svdd_report_5.md` §8（「完全推翻」）。

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

> ⚠️ **[已更正 2026-09-07]** 下面這句的後半「Inference-time search 整個方向（SVDD/CoDe/TDS）正式判定 dead end」**已被 `docs/report/svdd_report_5.md` §8 完全推翻**（`svdd_report_4.md` §7 亦明文要求修正本節措辭）。句中「paper opt = 44.32」是誤標 —— 那是 `ablation_10k` + opt，paper 是 `large-v2` + opt = 46.89 / 48.691。數字比較本身無誤（CSV 重算 Run E = 44.485 vs Run F = 44.321），錯在把它當成「vs paper」。
>
> **正確說法**：**對已經 fine-tune 過的 checkpoint（ablation_10k）SVDD 沒有 marginal value；對 paper 原始 checkpoint（large-v2）SVDD layered on opt 3-seed = 44.845 ± 0.654，贏 paper 46.89 約 −4.4%，3/3 seeds 全贏。** Inference-time search 路線並未 dead end，且與 fine-tuning 是平行的 axis（headroom 假說）。

**SVDD-PM 機制可運作，但在 chipdiffusion 任務上 — 不管單獨用還是疊加 paper 的 opt guidance — 都沒有 marginal value。Phase 3 確認 SVDD + opt = 44.49 vs paper opt = 44.32（+0.37%, 持平）。Inference-time search 整個方向（SVDD/CoDe/TDS）正式判定 dead end，下一步轉教授建議 (1)「從頭 train」。**
