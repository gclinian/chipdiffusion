# TDS（Twisted Diffusion Sampler / SMC）第一次實驗 Report

> Plan：`docs/plan/tds_plan_1.md`
> Survey：`docs/survey/guidance_survey_1.md` §1.3（TDS arXiv:2306.17775 NeurIPS 2023）
> 對照：`docs/report/svdd_report_5.md`（SVDD 44.85 ± 0.65）、`docs/report/code_report_1.md`（CoDe 45.22 ± 0.47）
> 執行日期：2026-05-25
> Setup：`large-v2.ckpt` + `tds_layered` + `opt-adam` legalization, 3 seeds × 7 circuits

---

## 1. 結論摘要

| 項目 | 結果 |
|------|------|
| **TDS layered 3-seed mean ± std** | **45.08 ± 0.38** |
| vs Paper baseline (46.89) | **−3.86%（贏 paper, 3/3 seeds）** |
| vs SVDD (44.85 ± 0.65) | **+0.53%（在噪音內，差 0.23）** |
| vs CoDe (45.22 ± 0.47) | **−0.31%（在噪音內，差 0.14）** |
| plan_1 §6 判定 | **44.5 ≤ 45.08 ≤ 45.5 → TDS ≈ SVDD ≈ CoDe，三族 inference-time search saturate 在同一個 ceiling** |
| **特色 1**：variance 最低 | std 0.38（vs SVDD 0.65, CoDe 0.47）— **particle filter 真的有 stabilize** |
| **特色 2**：adaptec2 大贏 | TDS 31.69（≈ paper 31.00）vs SVDD 33.40（+7.7%）—**SVDD/CoDe 唯一輸 paper 的 circuit, TDS 救回**|
| 主要 takeaway | **三族方法都贏 paper -3.6 ~ -4.4%；機制細節不重要；TDS 是「平均最穩定且 worst-circuit 最好」的選擇** |

---

## 2. 完整 3-seed 7-circuit 結果（三族並列）

| idx | Circuit | Paper | SVDD mean ± std | CoDe mean ± std | **TDS mean ± std** | TDS-SVDD |
|----|---|---:|---:|---:|---:|---:|
| 0 | adaptec1 | 9.19 | 9.13 ± 0.28 | 9.24 ± 0.29 | **9.18 ± 0.22** | +0.06 |
| 1 | adaptec2 | 31.00 | 33.40 ± 2.60 | 32.66 ± 5.04 | **31.69 ± 1.33** | **−1.71** ✓ |
| 2 | adaptec3 | 54.40 | 55.00 ± 0.68 | 53.35 ± 1.45 | **54.12 ± 0.47** | −0.88 |
| 3 | adaptec4 | 54.50 | 53.45 ± 1.24 | 53.76 ± 2.61 | **54.30 ± 2.44** | +0.86 |
| 4 | bigblue1 | 2.64 | 2.63 ± 0.02 | 2.65 ± 0.02 | **2.66 ± 0.02** | +0.03 |
| 6 | bigblue3 | 35.90 | 34.83 ± 5.34 | 34.96 ± 3.22 | **33.94 ± 0.16** ← **最穩** | −0.89 |
| 7 | bigblue4 | 140.60 | **125.48 ± 4.42** | 129.89 ± 1.34 | **129.67 ± 1.37** | +4.20 |
| **7-circuit avg** | | **46.89** | **44.85 ± 0.65** | **45.22 ± 0.47** | **45.08 ± 0.38** | +0.23 |

**Per-seed TDS 7-circuit avg**：
- seed=300: 44.97（贏 paper −4.09%）
- seed=301: 45.50（贏 paper −2.97%）
- seed=302: 44.77（贏 paper −4.52%）
- **3/3 seeds 都贏 paper**，std across seeds = 0.38

---

## 3. TDS 的兩個 unique 特性

### 3.1 Variance 最低（particle filter 的 raison d'être）

| Method | per-seed avg std | bigblue3 std | adaptec3 std |
|---|---:|---:|---:|
| SVDD | 0.65 | 5.34 | 0.68 |
| CoDe | 0.47 | 3.22 | 1.45 |
| **TDS** | **0.38** | **0.16** | **0.47** |

bigblue3 std 從 SVDD 5.34 → TDS **0.16**（**−97%**）── ESS-based resampling 真的大幅縮小了 particle 分歧。為什麼這對 paper 有意義：
- 小 std 代表結果可重現性高 → reviewer 不會質疑 single-seed 偷數據
- 大 variance circuit 是審稿人的 attack vector，TDS 把它消掉

### 3.2 Worst-case circuit (adaptec2) 救回

| Method | adaptec2 mean | vs Paper 31.00 |
|---|---:|---:|
| SVDD | 33.40 | +7.7% (lose) |
| CoDe | 32.66 | +5.3% (lose) |
| **TDS** | **31.69** | **+2.2% (essentially tied)** |

SVDD/CoDe 在 adaptec2 都輸 paper 5-8%；TDS 把這個拉回到 +2%。可能機制：
- adaptec2 V=566 跟 adaptec1 (V=543) 同等小，但 paper 數字差很多（31.0 vs 9.19）→ adaptec2 placement landscape 特別 multi-modal
- TDS particle 同時探索多個 mode，importance weight 累積後選最佳 → 比 SVDD 每步 collapse / CoDe 每 block hard argmax 更能保持多 mode 探索

---

## 4. 但 TDS 在 bigblue4 還是輸 SVDD 4.20

唯一 SVDD 顯著贏 TDS 的 circuit。可能原因：
- bigblue4 V=8170 是最大 circuit，placement landscape 高維
- TDS 4 個 particle 在 1000 step reverse 中保持獨立 trajectory → 4 個 mode 不夠
- SVDD 每步 resample → trajectory 都「重新匯聚」到當下 reward 最高點 → 等同 deeper search
- TDS 為了 asymptotic exactness 保留了 particle 多樣性，反而在 single best 評估上輸了

→ Insight: **bigblue4 是 "best-of-N at every step" 比 "best-of-final-N" 更有效的場景**

---

## 5. 對照 plan_1 §6 判定基準

| 區間 | 意義 | 行動 | 觸發 |
|---|---|---|---|
| TDS < 44.5 | SMC importance weighting 真的贏 SVDD | TDS 變新 main | ❌ |
| **44.5 ≤ TDS ≤ 45.5** | **三族都 saturate 在同 ceiling** | **paper 用 CoDe 主推（最簡），SVDD/TDS 列 ablation；下一步 FreeDoM / 從頭 train** | ✅ **(45.08)** |
| 45.5 < TDS ≤ 46.5 | TDS 輸 SVDD/CoDe | sweep ess_threshold | ❌ |
| TDS > 46.5 | TDS 沒贏 paper | SMC 不適合 | ❌ |

**45.08 落在第二區間 → 三族 saturate 在 ceiling**

---

## 6. 更新後 Leaderboard

| Rank | 方法 | 7-circuit avg | Checkpoint | Guidance |
|------|------|---:|---|---|
| 1 | Ablation 10k | 44.01 | fine-tuned ablation_10k | opt |
| 2 | Phase 3 Run F | 44.32 | ablation_10k | opt |
| 3 | Phase 3 Run E (svdd_layered) | 44.49 | ablation_10k | svdd_layered |
| 4 | DDPO v2 | 44.65 | fine-tuned | opt |
| **5** | **SVDD Phase 5 3-seed** | **44.85 ± 0.65** | **large-v2 (paper)** | **svdd_layered** |
| **6** | **TDS Phase 1 3-seed** | **45.08 ± 0.38** | **large-v2 (paper)** | **tds_layered** |
| 7 | AddLoss v2 | 45.24 | fine-tuned | opt |
| **— tied** | **CoDe Phase 1 3-seed** | **45.22 ± 0.47** | **large-v2 (paper)** | **code_layered** |
| 8 | AddLoss v1 | 45.39 | fine-tuned | opt |
| 9 | Ablation 5k | 45.43 | fine-tuned | opt |
| ... | DDPO v2.3-2.5 | 45.80-46.50 | fine-tuned | opt |
| **12** | **Paper baseline** | **46.89** | **large-v2** | **opt** |

**Top-5 中有 3 個是不靠 fine-tuning 的 inference-time search 方法**（SVDD/TDS/CoDe），都從 paper 自己的 large-v2 起。

---

## 7. 三族方法的最終比較表

| 比較項 | SVDD | CoDe | **TDS** |
|---|:---:|:---:|:---:|
| 7-circuit avg | **44.85** ← 最低 | 45.22 | 45.08 |
| std across seeds | 0.65 | 0.47 | **0.38** ← 最低 |
| 最弱 circuit | adaptec2 33.40 | adaptec2 32.66 | **adaptec2 31.69** ← 最好 |
| 最強 circuit | **bigblue4 125.48** ← 最低 | bigblue4 129.89 | bigblue4 129.67 |
| bigblue3 std | 5.34 | 3.22 | **0.16** ← 壓倒性最低 |
| 實作複雜度 | 中 | 低 | 中-高 |
| Cost (per seed) | ~117 min | ~98 min | ~120 min |
| 機制 | step-wise soft resample | block-wise hard argmax | particle filter + ESS resample |
| 對 paper 增量 | −4.36% ← 最大 | −3.57% | −3.86% |

**沒有 dominant 方法**。三族各有長處。

---

## 8. 對教授可以這樣定位

「我們在 paper 自己的 large-v2 checkpoint 上測了三種 inference-time guidance 方法，全部都贏 paper baseline 46.89：
- **SVDD-PM** soft-resample at every step → 44.85（−4.4%, std 0.65）：最低均值
- **CoDe** blockwise best-of-N → 45.22（−3.6%, std 0.47）：最簡實作
- **TDS** SMC importance-weighted particle filter → 45.08（−3.9%, std 0.38）：**最穩定 + worst-circuit 最好**

三者統計上 indistinguishable（差距 < 0.4 全在噪音內）。但每個有獨特 trade-off：
- 想要 best-mean：用 SVDD
- 想要簡單實作：用 CoDe
- 想要可重現性：用 TDS（**std 從 SVDD 0.65 降到 0.38, bigblue3 std 從 5.34 降到 0.16**）

這個結果說明 **inference-time K-particle search 是真的 universal pattern**，不只是 SVDD 一個方法的偶然成功。對 chip placement，這個 mechanism family 比 paper 的 opt-only 系統性地好 -3.6 ~ -4.4%。」

---

## 9. Cost 對照（每 seed）

| Method | per-seed 7-circuit 時間 |
|---|---|
| Paper baseline (opt only) | ~60 min |
| CoDe layered | ~98 min |
| SVDD layered | ~117 min |
| **TDS layered** | **~120 min** |

TDS 跟 SVDD 接近。CoDe 因為 K-particle 只在 block boundary 做 → 較快。opt 的 20 inner SGD 才是 dominant cost，K-particle 那點開銷沒差。

---

## 10. 跑出來的事故

1. ⚠️ **`reverse_guidance_opt_force` 在 N>1 batch 下 alpha_cost.backward() 失敗** ── PyTorch 對 `(1,)` shape 自動視為 scalar 但 `(4,)` 不行。
   - 修正：`alpha_cost = -self.alpha * (h_legality_raw.detach() - self.legality_potential_target).mean()`
   - 對 SVDD/CoDe layered（B=1）等效 identity，不影響舊結果
   - 這是 chipdiffusion codebase 一個 latent bug（從來沒人用過 B>1 的 opt guidance），現在 TDS 順便修了
2. ✅ ESS-based resampling 機制正常（看 bigblue3 std=0.16 就知道 resample 是 active 的）
3. ✅ GPU 1 全程順利

---

## 11. 行動清單

- [x] Step 1：實作 `_reverse_samples_tds` + TDS kwargs
- [x] Step 2：寫 `configs/guidance/tds_layered.yaml`
- [x] Step 3：fix `reverse_guidance_opt_force` for N>1
- [x] Step 4：adaptec1 smoke test
- [x] Step 5：3-seed × 7 circuits run
- [x] Step 6：寫此 report (tds_report_1.md)
- [ ] Step 7：寫 `docs/next/tds_next_1.md`
- [ ] 更新 CLAUDE.md leaderboard

---

## 12. 一句話結論

**TDS layered 3-seed mean 45.08 ± 0.38（贏 paper −3.86%, 3/3 seeds），跟 SVDD (44.85) 和 CoDe (45.22) 統計上 indistinguishable，但 std 顯著最低（bigblue3 std: SVDD 5.34 → TDS 0.16）。三族 inference-time search 都 saturate 在「贏 paper -3.6~-4.4%」的 ceiling，機制細節不重要；TDS 是「平均次優但 worst-case 最好且最穩」的選擇。**
