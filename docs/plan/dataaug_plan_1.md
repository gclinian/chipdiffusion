# Data Augmentation 計畫

> 動機：`docs/meet/meet_0620.md` 教授建議 (1)

## 1. 目標
在不重新生成資料的前提下，用 placement-invariant 變換擴增 v1.61-fs / v2.61，看 zero-shot ISPD transfer 是否改善。

## 2. 候選 augmentation（HPWL/legality invariant 或 equivariant）

| 變換 | 性質 | 實作 |
|---|---|---|
| **8-fold dihedral**（旋轉 90/180/270 + 鏡射）| HPWL 不變、legality 不變 | 座標 transform + pin offset 同步 transform |
| **座標平移抖動**（小幅 shift 後 re-clip）| 近似不變 | 低優先（可能破壞邊界 legality）|
| **Macro index permutation** | 完全不變（graph 同構）| GNN 本身 permutation-equivariant → 理論無效果，跳過 |
| **Net subsampling / edge dropout** | 改變 condition，是 regularizer | config 已有 `edge_dropout` 參數，直接 sweep |

主力 = **dihedral 8-fold**（等效資料量 ×8）+ `edge_dropout` sweep。

## 3. 實驗
1. 在 dataloader `get_batch` 加 on-the-fly random dihedral transform（~30 行，作用於 x 和 cond.edge_attr 的 pin offsets）
2. 訓練對照（皆 v1.61-fs, 500k, batch=32）：
   - Run A: baseline（= 既有 fs_p1_X_500k，45.05，不用重跑）
   - Run B: + dihedral aug
   - Run C: + dihedral aug + edge_dropout=0.1
3. ISPD 7-circuit eval, seed=300

## 4. 判定
- Run B/C < 44.5 → augmentation 有效，multi-seed 驗證後併入所有後續訓練
- 44.5–45.05 → 邊際，記錄即可
- > 45.05 → augmentation 傷害（synthetic data 已夠多樣），關閉此方向

## 5. Cost：~1.5 天 GPU（兩個 500k run）+ 半天實作
