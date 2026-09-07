# 傳統 EDA 方法生成 Dataset 計畫

> 動機：`docs/meet/meet_0620.md` 教授建議 (4)
> 現況：訓練資料全是 synthetic inverse-problem 生成（先擺 placement 再生 netlist）。真實 netlist + EDA placer 的 (netlist, placement) pair 分佈可能更接近 deployment。

## 1. 目標
用傳統 EDA placer 產生訓練 label，測「更真實的 placement 分佈」能否改善 ISPD transfer。

## 2. 兩條資料來源路線

### 路線 A：synthetic netlist + EDA placer 重新標註（推薦先做）
1. 拿 v1.61-fs 的 4600 個 netlist（已有 graph 結構）
2. 用 **DREAMPlace**（paper 也用它，GPU-based，快）或現有 repo 的 `legalization.py` optimizer 對每個 netlist 從 random init 解 placement
3. 生成 (netlist, EDA placement) pairs → 新 dataset `v1.61-eda`
4. 對照訓練：同架構同 500k，只換 label 來源

**優點**：netlist 不變 → 乾淨對照「label 品質」的影響。
**注意**：synthetic 原 label 是 by-construction near-optimal；EDA label 品質若更差反而傷。先抽 100 個樣本比較兩者 HPWL 分佈再決定是否全量跑。

### 路線 B：真實 netlist（IBM/ICCAD04）+ EDA placer
- repo 已有 `benchmarks/` IBM 資料 + hmetis clustering pipeline
- 對 18 個 ibm circuit 各生多個 placement（不同 seed/參數）→ 小而真實的 fine-tune set
- 用於 stage-2 style fine-tune（取代或混合 v2.61）
- **風險**：只有 18 個 netlist，過擬合風險高；需搭配 augmentation（見 dataaug_plan_1）

## 3. 前置工程
- [ ] 裝 DREAMPlace（GPU 版）或確認 repo 內 optimizer 可當 placer 用
- [ ] 寫 netlist → DREAMPlace 格式 converter（或直接用 repo `legalization.py` 避免格式轉換）
- [ ] 100-sample 品質 pilot：比較 EDA label vs synthetic label 的 HPWL/legality 分佈

## 4. 判定
- Pilot 顯示 EDA label HPWL 比 synthetic label 差 >10% → 路線 A 直接放棄（label 更爛不可能訓更好）
- EDA label 相當或更好 → 全量生成 + 500k 對照訓練 → ISPD eval

## 5. Cost：工程 2-3 天 + 生成半天 + 訓練 1 天。四個方向中工程量最大，排最後。
