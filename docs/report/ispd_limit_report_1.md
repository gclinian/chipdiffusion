# ISPD2005 極限分析 Report（Part A oracle + Part C 文獻）

> Plan：`docs/plan/ispd_limit_plan_1.md`
> 動機：教授問（meet_0620 §3）「所有方法擠在 44-47，是不是 ISPD2005 已到極限？」
> 日期：2026-07-06。資料源：`docs/all_experiments_per_circuit.csv`（30 個 deployment-comparable 結果 / circuit，排除 raw 無 legalization 的 Phase 1 runs）

## 1. 結論（一句話）

**還沒到極限。** 把既有 30 個結果每個 circuit 取最佳（oracle），7-circuit avg = **42.09**，比目前最佳單一方法 Ablation 10k (44.01) 還低 **4.4%**，比 paper 低 10.2%。這只是「既有方法 + seed 運氣」就能構成的上界估計——真實 achievable 下限必然更低。**44.01 不是 benchmark 天花板，是方法/seed 選擇的天花板。**

## 2. Part A — 全實驗 Oracle（best-per-circuit across ~30 results/circuit）

| Circuit | Paper | Ablation 10k | **Oracle** | Oracle 來源 (seed) |
|---|---:|---:|---:|---|
| adaptec1 | 9.19 | 9.74 | **8.84** | SVDD_layered_large-v2 (s300) |
| adaptec2 | 31.00 | 32.05 | **28.36** | CoDe_layered_large-v2 (s301) |
| adaptec3 | 54.40 | 53.90 | **51.60** | AddLoss_v1 (s300) |
| adaptec4 | 54.50 | 54.16 | **51.15** | DDPO seed ensemble (s600) |
| bigblue1 | 2.64 | 2.70 | **2.59** | AddLoss_v1 (s300) |
| bigblue3 | 35.90 | 30.70 | **30.32** | AddLoss_v1 (s300) |
| bigblue4 | 140.60 | 124.85 | **121.76** | SVDD_layered_large-v2 (s301) |
| **avg(7)** | **46.89** | **44.01** | **42.09** | — |

觀察：
- **Oracle 的 7 個 winner 來自 5 個不同方法家族**（SVDD×2, AddLoss v1×3, CoDe×1, DDPO ensemble×1）— 沒有單一方法 dominate，各方法擅長不同 circuit。
- 每個 circuit 的 min 與 median 差 5-12% → **seed variance 仍是最大的未開採資源**。
- 這暗示兩條便宜的改進路：(a) **best-of-N inference**（跑 4 seeds 挑最好，可逼近 oracle，cost = 4× eval）；(b) per-circuit method selection。

## 3. Part C — 文獻對照（paper Table 10, macro-only ISPD2005）

| 方法 | 8-circuit avg | 備註 |
|---|---:|---|
| WireMask-BBO | 154 | bigblue4=798 拖垮 |
| ChiPFormer | 116 | bigblue4=548 |
| MaskPlace | — | 2/8 timeout |
| Diffusion (paper) | 45.9 | 8-circuit 含 bb2 |
| **我們 oracle (7-circuit)** | **42.09** | — |

文獻中無人做到 42 以下；diffusion 系是目前唯一在 45 級別的方法家族。**42.09 是 open literature 的新低界**（如以 best-of-N 形式呈現）。

## 4. 對教授問題的回答

1. **「是不是到極限了？」** — 不是。Oracle 42.09 證明既有方法組合就還有 ≥4.4% 空間；真實下限更低（Part B 純優化實驗可進一步確認，尚未跑）。
2. **「為什麼進步這麼少？」** — 因為所有 single-method single-seed 的結果被 **seed variance（每 circuit 5-12%）** 淹沒。方法之間的真實差距比 variance 小，所以看起來擠在一起。
3. **「要不要換 dataset？」** — ISPD2005 還能玩（seed ensemble / per-circuit selection 就能到 ~42），但**同意加測 IBM/ICCAD04**（repo 已有資料 + paper 有完整 baseline 表）作為第二 benchmark 增加說服力 — 不是因為 ISPD 飽和，而是避免 overfit 單一 benchmark 的質疑。

## 5. 後續

- [ ] Part B（純優化下界，adaptec1/bigblue1，半天 GPU）— pending，等 flow matching 訓練完 GPU 空出來
- [ ] 若要把 oracle 變成可宣稱的結果：把 best-of-4-seeds 作為正式 inference 協定重跑（cost 4×，但結果可寫進 paper）

---

## 附錄 A（Addendum, 2026-09-07）— 補登 4 筆結果後的 oracle 修正

> **上方 §1–§5 為 2026-07-06 當下的歷史紀錄，一字未改。** 本節只記錄「資料補登之後數字如何變動」。
> 觸發：2026-09-07 將 4 筆**已完成但從未登錄**的結果補進 `docs/all_experiments_summary.csv` 與
> `docs/all_experiments_per_circuit.csv`，全部直接讀自 on-disk `logs/diffusion_debug/*/metrics.csv`
> （非二手引用 report）：
> - `FromScratch_Stage2`（v2.61 stage-2 fine-tune of Run X, ckpt `v2.61.fs_p1_X_stage2_b32.61`, eval 2026-06-12）→ avg7 **46.44**
> - `FlowMatch_ODE50` / `FlowMatch_ODE1000` / `FlowMatch_ODE10` → avg7 **70.03 / 77.05 / 81.25**
>
> oracle 口徑與 §2 完全相同（排除無 legalization 的 Phase 1 raw runs、排除 paper 欄），
> pool 由 29 個 → **33 個 deployment-comparable 結果 / circuit**。

### A.1 Oracle：42.09 → **41.8444**

7 個 winner 中**只有 adaptec2 換人**，其餘 6 個原封不動：

| Circuit | §2 原 oracle | 來源 | 新 oracle | 新來源 | 變動 |
|---|---:|---|---:|---|---|
| adaptec1 | 8.84 | SVDD_layered_large-v2 (s300) | 8.8432 | 同左 | — |
| **adaptec2** | **28.36** | CoDe_layered_large-v2 (s301) | **26.6463** | **FromScratch_Stage2 (s300)** | **−1.72** |
| adaptec3 | 51.60 | AddLoss_v1 (s300) | 51.60 | 同左 | — |
| adaptec4 | 51.15 | DDPO seed ensemble (s600) | 51.15 | 同左 | — |
| bigblue1 | 2.59 | AddLoss_v1 (s300) | 2.59 | 同左 | — |
| bigblue3 | 30.32 | AddLoss_v1 (s300) | 30.32 | 同左 | — |
| bigblue4 | 121.76 | SVDD_layered_large-v2 (s301) | 121.7616 | 同左 | — |
| **avg(7)** | **42.0899** | — | **41.8444** | — | **−0.2455** |

全部改善量 = (28.3646 − 26.6463)/7 = 0.2455，**100% 來自單一格子**。
26.6463 也是全專案 adaptec2 的最低值（次低 28.3646）。
headroom 由 vs Ablation 10k −4.36% / vs paper −10.24% 變成 **−4.92% / −10.76%**。

### A.2 但這個進步是拿 legality 換來的

`FromScratch_Stage2` adaptec2 的 **macro_legality = 0.9623**，比它取代掉的
CoDe s301 那格（**0.9781**）**更差**。也就是說 41.8444 並不是「更好的擺放」，
而是「更不合法但線長更短的擺放」。**引用 oracle 數字時必須同時報 legality**，否則等於用違規面積換 HPWL。

附帶脈絡：`FromScratch_Stage2` 本身是一次 **regression** — avg7 46.44 vs 其 Stage 1 起點
(FromScratch_X_pure_500k) 45.05，**+3.08%**，主因 bigblue4 129.23 → 145.60（+12.67%）。
它整體變差，卻提供了全專案最好的一格 adaptec2 —— 正是 §2 觀察「各方法擅長不同 circuit」的又一例證。

### A.3 加上 legality 門檻後的 oracle

| macro_legality 門檻 | oracle avg(7) | adaptec2 取到誰 |
|---|---:|---|
| 無門檻 | **41.8444** | 26.6463 FromScratch_Stage2 (0.9623) |
| ≥ 0.95 / ≥ 0.96 | 42.1083 | 26.6463 FromScratch_Stage2 (0.9623) |
| ≥ 0.97 | 42.3538 | 28.3646 CoDe s301 (0.9781) |
| ≥ 0.98 | 42.8135 | 31.583 AddLoss_v2 s300 (0.9843) |
| ≥ 0.99 | **INFEASIBLE** | 全專案 adaptec2 最高 legality 僅 0.9843 → 無解 |

注意一個反直覺點：**連最鬆的 ≥0.95 門檻都讓 oracle 變差**（41.8444 → 42.1083）。
原因不是 legality 卡到，而是 A.4 的資料缺漏 —— 無門檻 oracle 的 7 個 winner 裡有
**4 個（adaptec3 / adaptec4 / bigblue1 / bigblue3）根本沒有 macro_legality 紀錄**，
一加門檻就整列被丟掉。換句話說 41.8444 有一部分是「因為那些列無法被 legality 稽核」才成立的。

### A.4 資料完整性警告（重要）

**222 列 oracle-relevant 資料中有 95 列完全沒有 macro_legality**（= 補登前 CSV 內所有數值列，
扣掉 15 列 Phase 1 raw；含 bigblue2 與 paper 欄）。→ 只有 **127/222 = 57.2%** 可用，
**任何 legality-constrained oracle 都是在不到 60% 的資料上算出來的**，A.3 那張表要照這個折扣讀。

缺漏完全集中在 data_source 為 `extracted from ...report...` 的列（二手抄自 report，report 沒抄 legality）：
`Ablation_10k`(8)、`AddLoss_v1`(8)、`DDPO_v2_3/4/5`(各 8)、`DDPO_v2_PPO_seed300`(8)、
`DDPO_v2_PPO_seed_ensemble`(32)、`Our_repro_large-v2_opt_seed300`(7)、`Paper_Table10_published`(8)。
凡是 `metrics.csv (on disk)` 的列都有 legality。

補登後（28 筆新列全帶 legality）同口徑為 95/250 → 62.0% 可用；若只看 §2 的嚴格 7-circuit oracle pool
則為 77/231 缺漏 → 66.7% 可用。三個數字都指向同一件事：**約四成資料無法做 legality 稽核。**

### A.5 對 §1 / §4 結論的影響

1. **§4 Q1「是不是到極限了？」— 結論不變**（還沒到），且 headroom 略微變大（4.36% → 4.92%）。
2. **§1 / §3 的「42.09」應改讀為「41.84」**，但**必須附帶 legality 條件**才誠實：
   在 ≥0.97 門檻下只到 42.35，≥0.99 則 adaptec2 無解。對外宣稱「open literature 新低界」時，
   建議引用**有 legality 門檻的版本**（≥0.97 → 42.35），避免被質疑用不合法擺放刷分。
3. FlowMatch 三筆對 oracle **完全無影響**（70/77/81，全面遠差），補登只是補齊 row census。

### A.6 由本次補登產生的待辦

- [ ] 回填那 95 列的 `macro_legality`（需重跑 eval 或翻舊 eval log）— 在此之前，legality-constrained oracle 只能當粗估
- [ ] `FromScratch_Stage2` 尚無對應的 `_report_N`（`from_scratch_report_1.md` 寫於 stage 2 執行之前，§10 行動清單仍把 stage 2 列為待決策項）— 需補一份，重點是「stage 2 整體 regression（+3.08%）但 adaptec2 是全專案最佳」
- [ ] 若要正式宣稱 oracle，先決定 legality 門檻並固定下來（建議 ≥0.97，與多數 deployment run 的實際水準一致）
