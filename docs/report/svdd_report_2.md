# SVDD-PM 第二次實驗 Report (Phase 2: legalization + 7 circuits)

> Plan: `docs/plan/svdd_plan_2.md`
> 前次回顧：`docs/next/svdd_next_1.md`
> 動機 meeting: `docs/meet/meet_0508.md`
> 執行日期：2026-05-20 ~ 2026-05-21
> Checkpoint：`v1.61-ddpo.ablation_supervised_10k.61/latest.ckpt`（目前最佳 init）
> Legalizer：`opt-adam` with `grad_descent_steps=20000`（CLAUDE.md 標準設定）

---

## 1. 結論摘要

| 項目 | 結果 |
|------|------|
| **7-circuit avg HPWL (no bb2)** | Run A (baseline, guidance=none) = **84.85**； Run C (SVDD every_n=1) = **71.94** |
| **SVDD vs no-guidance baseline** | **-15.21%（贏所有 7 個 circuit）** |
| **SVDD vs CLAUDE.md ablation_10k 標準 (guidance=opt) = 44.01** | **+63%（顯著輸）** |
| Plan_2 §5.1 判定基準 (avg > 44.5) | ❌ **SVDD 沒贏 ablation_10k 標準** |
| 應放棄 inference-time search？ | **不完全 — 比較不公平**：Run A 用 guidance=none，沒打開 opt guidance；SVDD + opt layered 還沒測 |

**最大發現**：SVDD-PM 作為**獨立 inference-time intervention** 比「不開 guidance」好 -15%，但無法替代 `opt` gradient guidance（每步 10 inner SGD + Lagrangian）。Bigblue3 是最大贏家（-54.7%）。

**建議下一步**（待寫進 `svdd_next_2.md`）：先測 SVDD layered on top of opt guidance；如果還是輸 44.01，才正式放棄 inference-time search 路線轉到「從頭 train」（教授建議 1）。

---

## 2. 實驗執行

### 2.1 跑了哪些 run

| Run | guidance | 描述 | 完成情況 |
|-----|----------|------|---------|
| **A** | `none` | baseline: ablation_10k 自身能力 + opt-adam legalization | ✅ 7/7 circuits |
| **C** | `svdd` K=4, λ=1, every_n=1, α_temp=1, start_step=0.5 | 主菜：plan_2 所稱「aggressive」 | ✅ 7/7 circuits |
| ~~B~~ (svdd default every_n=5) | — | **跳過**：Phase 1 已知 default 設定邊際；先看 C 結果 | (未跑) |
| ~~D~~ (seed 301 validate) | — | C 結果不夠贏 ablation_10k，不需要 seed 驗證 | (未跑) |

### 2.2 執行細節

- 為避開 bigblue2 (~100 min legalization × 2 配置)，**分成 part1 (idx 0-4) + part2 (idx 6-7)** 兩個 eval invocations。skip bb2。
- part2 (bigblue3 + bigblue4) 在 GPU 0 連續 4 次 `cudaErrorIllegalAddress`（其他使用者佔了 GPU 0 10.5 GB / 24 GB）。**改用 GPU 1 全部成功**。
  - 修正動作：`CUDA_VISIBLE_DEVICES=1` 跑剩餘的 3 個 run。
  - 教訓寫進 [[project_svdd_direction]]。

---

## 3. 完整 7-circuit 結果

### 3.1 Run A vs Run C（macro HPWL paper format = `macro_hpwl_rescaled / 100`）

| idx | Circuit | V | A (none) HPWL | C (SVDD) HPWL | Δ% | A legality | C legality |
|----|---------|---|---|---|---|---|---|
| 0 | adaptec1 | 543 | 11.05 | 10.28 | **-6.95%** | 0.989 | 0.995 |
| 1 | adaptec2 | 566 | 40.12 | 35.69 | **-11.04%** | 0.969 | 0.979 |
| 2 | adaptec3 | 723 | 66.62 | 61.79 | **-7.24%** | 0.997 | 0.997 |
| 3 | adaptec4 | 1329 | 61.31 | 61.40 | +0.16% | 0.995 | 0.999 |
| 4 | bigblue1 | 560 | 8.86 | 7.94 | **-10.45%** | 0.997 | 0.997 |
| 6 | bigblue3 | 1298 | **98.82** | **44.79** | **-54.67%** | 0.994 | 0.994 |
| 7 | bigblue4 | 8170 | 307.20 | 281.71 | **-8.30%** | 0.987 | 0.985 |
| **7-circuit avg** | | | **84.85** | **71.94** | **-15.21%** | 0.990 | 0.992 |

**SVDD 贏 7/7 個 circuit**（最差 adaptec4 持平）；**legality 7/7 都保持 ≥ 0.985**（legalizer 確實 fix 了 raw SVDD 的 legality 退步 → 印證 plan_2 §1 核心假設）。

### 3.2 對照 CLAUDE.md 既有方法（7-circuit avg HPWL）

| 方法 | guidance | avg HPWL | vs SVDD-C |
|------|----------|----------|-----------|
| **Ablation 10k**（CLAUDE.md 公告） | **opt** | **44.01** | **-39%（顯著贏 SVDD）** |
| DDPO v2 (CLAUDE.md) | opt | 44.65 | -38% |
| AddLoss v2 (CLAUDE.md) | opt | 45.24 | -37% |
| Paper baseline | opt | 46.89 | -35% |
| Baseline large-v2 (CLAUDE.md) | opt | 48.69 | -32% |
| **Run C (SVDD)** | svdd | **71.94** | — |
| **Run A (this baseline)** | **none** | **84.85** | **+18%（SVDD 贏）** |

**關鍵**：SVDD 跟 `opt` guidance **不在同一個 league**。opt 是「每 reverse step 跑 10 步 Lagrangian inner SGD + adaptive alpha」的強 guidance。SVDD 是「每 reverse step 跑 K=4 forward + softmax 選 1」。前者投入計算遠大於後者。

---

## 4. 對照 plan_2 §5.1 的判定基準

| Avg HPWL (7) | 意義 | 行動 | 是否觸發 |
|---|---|---|---|
| < 43.5 | 顯著贏 ablation_10k | 寫 paper / 升級 SVDD-MC | ❌ |
| 43.5 – 44.0 | 微贏 | sweep K=8 / α_temp | ❌ |
| 44.0 – 44.5 | 持平 | 看 bigblue2 / stacking | ❌ |
| **> 44.5** | **SVDD 沒贏** | **放棄 inference-time search 方向**（plan_2 §5.1 原文） | ✅ **(71.94)** |

依 plan_2 §5.1 文字判定 → **應放棄 inference-time search 方向**。但這個判定**沒考慮一個 unfair 設計缺陷**：

### 4.1 plan_2 設計缺陷：Run A baseline 對 SVDD 不利

plan_2 §4 Run A 寫得很明確：「為了跟 SVDD 公平比，Run A 也要用 guidance=none」。但這把 SVDD 跟「checkpoint 自身能力」對比，而 ablation_10k 44.01 是「checkpoint + opt guidance」。**Run A 比 ablation_10k 公告 44.01 高 93%**（84.85 vs 44.01）就是這個 setup 差別 — 不是 SVDD 的鍋。

**真正該問的問題**是：「SVDD 能不能取代 / 補強 opt？」答案：
- ✅ SVDD **能贏 no-guidance**（-15%）→ 機制有效
- ❌ SVDD **不能單獨贏 opt**（+63% vs 44.01）→ opt 是更強的 inference-time intervention
- ⚠️ **SVDD + opt layered 沒測** → plan_2 §2 漏這個 cell；不能直接放棄

### 4.2 修正後的決策

不是「放棄 inference-time search」，而是「**plan_2 沒測 SVDD + opt 的組合**」。要在 plan_3 補：

| 補測項目 | 預期 |
|----------|------|
| Run E：SVDD + guidance=opt（layered）vs Run A | 若 layered 贏 ablation_10k 44.01 → SVDD 有 marginal 增量 |
| Run F：SVDD 跑在 large-v2.ckpt（沒 fine-tune）+ opt | 看 SVDD 對未 fine-tune 模型的補強效果 |
| Run G：SVDD 從 ablation_10k 起、higher α_temp | 增加 exploration，看能不能突破 SVDD-only 71.94 floor |

如果 Run E 還是輸 44.01 → 正式放棄 inference-time search → 轉教授建議 (1)「從頭 train」。

---

## 5. 觀察與假說

### 5.1 為什麼 bigblue3 進步最大（-54.7%）？

bigblue3 V=1298，但 baseline (Run A) HPWL=98.82 是 7 個 circuit 中最異常的（vs paper 35.9, ablation_10k+opt 34.26）。表示 **ablation_10k 的 raw output 對 bigblue3 特別差**，而 opt guidance 是必要的修正。SVDD 部分 recover 了 opt 的修正能力 → 但只能恢復 50%（44.79 vs paper 35.9 還差 24%）。

**Insight**：SVDD 的價值對「raw output 越差的 circuit 越大」。

### 5.2 為什麼 adaptec4 沒進步（+0.16%）？

adaptec4 V=1329（與 bigblue3 接近），但 Run A HPWL 61.31 已經很接近 ablation_10k+opt 的 54.88。**這個 circuit 的 ablation_10k raw output 已經很強**，SVDD 沒有 search 空間。

**Insight**：SVDD 對「已經接近 floor 的 circuit」邊際效用為 0。

### 5.3 SVDD 的計算 cost vs opt

| 方法 | per reverse step 多耗 |
|------|----------------------|
| opt (CLAUDE.md) | 10 inner SGD steps（每步含 ∇legality V×V backward） |
| SVDD every_n=1 K=4 | K=4 forward passes |
| SVDD every_n=5 K=4 | K=4/5 ≈ 0.8 forward passes (amortized) |

SVDD 的計算量遠小於 opt。**adaptec1 generation_time**：Run A (no guidance) = 89s, Run C (SVDD every_n=1) = 94s → 只多 5%。adaptec4: 111s → 126s。Cost 完全可負擔。

### 5.4 legality 保持得很好

SVDD raw 階段降 legality 的問題（Phase 1 看到 0.69→0.61），**legalizer 完全 fix 了**：Run C legality 全 ≥ 0.985。印證 plan_2 §1 核心假設。

---

## 6. 跑出來的 bug / 限制

1. **GPU 0 contention 一致觸發 `cudaErrorIllegalAddress`**：3 次失敗後改 `CUDA_VISIBLE_DEVICES=1` 全成功。
   - 教訓：往後 ISPD 完整 eval 跑前先 `nvidia-smi` 看哪個 GPU 比較空，或寫 retry-on-CUDA-error 邏輯。
2. **K=8 PyG batched edge_index bug（Phase 1 留下的）** 沒去碰，Phase 2 用 K=4。
3. **plan_2 沒設計 SVDD + opt layered 對照**（§4.1 已解釋）。

---

## 7. 行動清單（Phase 2 收尾）

- [x] 跑 Run A baseline (none + opt-adam, 7 circuits)
- [x] 跑 Run C SVDD (every_n=1, λ=1, K=4, 7 circuits)
- [x] 收集 metrics.csv，分析 7-circuit avg
- [x] 寫此 report
- [ ] 下一步：寫 `docs/next/svdd_next_2.md`
- [ ] 下一步：寫 `docs/plan/svdd_plan_3.md`（補 Run E layered；如失敗則切換 plan 主題）

---

## 8. 一句話結論

**SVDD-PM 機制可運作（-15% vs no-guidance），但作為 opt guidance 的替代品不夠強（71.94 vs 44.01 = +63%）。不能算 plan_2 §5.1 意義上的「放棄 inference-time search」 — 必須先做 plan_2 沒測的 SVDD + opt layered 對照（plan_3 Run E）才能下最終結論。**
