# ISPD2005 極限分析計畫（純分析，不需訓練）

> 動機：`docs/meet/meet_0620.md` 教授建議 (3)
> 觀察：所有方法擠在 44-47（paper 46.89 → 我們 44.01，跨度僅 6%）。是方法極限還是 benchmark 極限？

## 1. 目標

估計 ISPD2005 macro-only 7-circuit 的 **achievable HPWL lower bound**，判斷「44.01 距離極限還有多遠」。

## 2. 方法（三個互補的下界估計）

### A. 已有數據的 best-per-circuit oracle
把所有實驗（~25 個 method × seeds，見 `docs/all_experiments_per_circuit.csv`）每個 circuit 取全域最低 HPWL，組出 oracle avg。
- Cost: 0（純分析既有 CSV）
- 已知 4-seed DDPO ensemble 上界是 43.89 → 全實驗 oracle 應更低

### B. per-circuit 強力優化下界
選 1-2 個小 circuit（adaptec1, bigblue1），用現有 legalizer 的 optimizer 從多個 init 直接對 HPWL 做長時間（>10× 平常步數）constrained optimization，不經 diffusion。看純優化能推到多低。
- Cost: ~半天 GPU

### C. 文獻對照
收集 ISPD2005 macro-only 各 paper 最佳值（WireMask-BBO 154, ChiPFormer 116, paper diffusion 45.9…），看 45 以下是否有人做到過。

## 3. 判定

| Oracle/優化下界 vs 44.01 | 結論 |
|---|---|
| 下界 ~43-44（差 <3%）| **ISPD2005 接近飽和** → 教授的懷疑成立，主力轉新 dataset（IBM/ICCAD04 已在 repo、或更新 benchmark）|
| 下界 <42（差 >5%）| 還有空間 → 方法還能改進，繼續在 ISPD 上做 |

## 4. 行動清單
- [ ] A: 從 all_experiments_per_circuit.csv 算全域 oracle
- [ ] B: adaptec1/bigblue1 純優化下界實驗
- [ ] C: 文獻表
- [ ] 寫 `docs/report/ispd_limit_report_1.md`（給教授的一頁結論）
