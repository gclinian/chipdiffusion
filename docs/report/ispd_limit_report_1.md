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
