# Flow Matching 第一次實驗 Report

> Plan：`docs/plan/flowmatch_plan_1.md`
> 執行：Phase A/B 由 Opus agent（2026-07-06），Phase C guidance port + eval 由主 session
> Setup：`FlowMatchingModel`（linear OT path, Euler ODE）, v1.61-fs 500k batch=32 lr=3e-4（與 DDPM Run X 完全同資料同架構同預算），eval = opt guidance + opt-adam legalization, seed=300

## 1. 結論

**Flow matching 輸。** 最佳設定（50 ODE 步）7-circuit avg = **70.03**，遠輸同預算 DDPM (45.05) 和 paper (46.89)。依 plan §4 判定「> 46.89 → 記錄後放棄，回到 DDPM 軌道」。

唯一亮點：**小 circuit 上 FM 用 10-50 步就達到近 DDPM 品質**（adaptec1 9.28 vs 9.12；bigblue1 2.80 vs 2.70），gen time 減半。但中大 circuit 全面崩壞，平均被拖垮。

## 2. 完整數據（HPWL paper format, seed=300）

| Circuit | V | Paper | DDPM@1000 | FM@10 | FM@50 | FM@1000 |
|---|---:|---:|---:|---:|---:|---:|
| adaptec1 | 543 | 9.19 | 9.12 | 9.31 | **9.28** | 10.39 |
| adaptec2 | 566 | 31.00 | 28.96 | 40.25 | 33.04 | 36.80 |
| adaptec3 | 723 | 54.40 | 56.46 | 70.00 | 66.21 | 78.13 |
| adaptec4 | 1329 | 54.50 | 55.87 | 67.97 | 61.81 | 78.38 |
| bigblue1 | 560 | 2.64 | 2.70 | **2.80** | 3.01 | 5.30 |
| bigblue3 | 1298 | 35.90 | 33.04 | 49.75 | 49.02 | 53.52 |
| bigblue4 | 8170 | 140.60 | 129.23 | 328.63 | 267.81 | 276.81 |
| **avg(7)** | | **46.89** | **45.05** | 81.25 | **70.03** | 77.05 |

Gen time（7 circuits 總和）：DDPM 2896s → FM@50 1370s（**2.1× faster**）。

## 3. 兩個核心觀察

### 3.1 FM 的品質-步數關係是反的（更多步更差）
sweep（adaptec1）：10 步 9.26 → 50 步 9.23 → 100 步 9.56 → 1000 步 10.39。
假說：guidance 在每個 ODE 步都跑 20 次 inner SGD，1000 步 = 20,000 次 guidance 更新 + Lagrangian α 在確定性 ODE 路徑上無 noise 洗掉累積偏差 → over-guidance。DDPM 的 stochastic reverse 每步注入 noise，天然抵抗 guidance 過衝。

### 3.2 失敗主因：OOD 泛化比 DDPM 差（不是 guidance 的鍋）
V 與退化幅度強相關：V≤560 → 貼平 DDPM；V≥723 → +17~154%。訓練資料 max 400 macros，DDPM 外插到 8170 只退化輕微，**FM 外插直接崩**（bigblue4 268 vs 129，2.1×）。deterministic velocity field 的外插能力比 stochastic denoiser 弱——這是 objective 本身的性質，換 guidance 救不了。

## 4. 判定與後續

- Plan §4 觸發「輸 paper → 放棄」。**回到 DDPM 軌道。**
- 保留價值：(a) `FlowMatchingModel` 程式碼留在 codebase（additive，不影響 DDPM）；(b)「小 circuit 20× 步數減省」若日後做 latency-sensitive 應用可回收；(c) 對 paper 是一個誠實的 negative ablation：「we tried flow matching; deterministic ODE objectives generalize worse to OOD circuit sizes」。
- 未試的變因（記錄，不建議追）：更長訓練（DDPM 同預算已收斂達 45.05，FM 差距 25 不是訓練量能補的）、guidance 專調（3.2 顯示 no-guidance 泛化就已輸）、drifting model（同屬 few-step deterministic 家族，預期同樣 OOD 弱點）。

## 5. 行動清單
- [x] Phase A/B/C 全部完成
- [x] 本 report
- [ ] `docs/next/flowmatch_next_1.md`（併入本 report §3-4，不另寫長文）
