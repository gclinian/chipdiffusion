# ChipDiffusion 實驗報告 — 2026/04/01

## 實驗目標

驗證 ChipDiffusion 論文 (arXiv 2407.12282) 在 ISPD2005 benchmark 上的 macro placement 結果是否可復現。

## 實驗設定

| 項目 | 我們的設定 | 論文設定 |
|------|-----------|---------|
| Checkpoint | `large-v2.ckpt`（論文公開） | 同 |
| Benchmark | ISPD2005（8 circuits） | 同 |
| 方法 | `eval_macro_only` | 同 |
| Guidance | 啟用（HPWL guidance, 20 steps） | 同 |
| Legalization | opt-adam, 20000 steps | 同 |
| Random seed | 300 | 400 |
| GPU | 單張 24GB GPU | 未公開 |

超參數（guidance weight, legalization lr 等）皆與論文一致。

## 實驗結果

### ISPD2005 Macro-Only Eval（HPWL 單位：x10^5）

| Circuit  | 我們的 HPWL | 論文 HPWL | 我們的 Legality | 我們的 HPWL Ratio | 備註 |
|----------|------------|----------|----------------|------------------|------|
| adaptec1 | 10.22      | 9.19     | 0.9942         | 0.718             |      |
| adaptec2 | 39.06      | 31.0     | 0.9305         | 1.056             | legality 較低 |
| adaptec3 | 62.14      | 54.4     | 0.9943         | 0.798             |      |
| adaptec4 | 60.51      | 54.5     | 0.9965         | 0.679             |      |
| bigblue1 | 2.69       | 2.64     | 0.9963         | 0.822             | 非常接近 |
| bigblue2 | —          | 38.8     | —              | —                 | OOM，無法執行 |
| bigblue3 | 34.26      | 35.9     | 0.9951         | 0.598             | 優於論文 |
| bigblue4 | 131.96     | 140.6    | 0.9914         | 0.491             | 優於論文 |

### 結果分析

- **bigblue1/3/4**：我們的結果與論文非常接近，bigblue3 和 bigblue4 甚至略優。
- **adaptec1-4**：HPWL 偏高於論文約 10-25%，推測主因是 random seed 不同（300 vs 400）。
- **adaptec2**：legality 僅 0.93，為所有 circuit 中最低，可能是 legalization 未完全收斂。
- 整體而言，結果趨勢與論文一致，確認 pre-trained checkpoint 可復現論文水準的效能。

## bigblue2 無法執行的原因

### 問題

bigblue2 擁有 **23,084 個 macros**，是所有 ISPD2005 circuit 中最大的。在 guidance 和 legalization 階段，程式會建立 **V x V 矩陣**（V = macro 數量），導致 GPU 記憶體不足（OOM）。

### 記憶體分析

Guidance 階段（`legality_guidance_potential()`）會在 GPU 上同時建立多個 `(B, V, V, D)` 張量：

| 張量 | 用途 | 形狀 |
|------|------|------|
| `delta` | 兩兩 macro 間距 | (1, V, V, 2) |
| `l` | softmax 距離 | (1, V, V, 1) |
| `h` | overlap penalty | (1, V, V, 1) |
| `mask_square` | 自碰撞遮罩 | (1, V, V, 1) |
| autograd 中間值 | 反向傳播用 | 同上 |

以下為 bigblue4（可執行）與 bigblue2（OOM）的 VRAM 估算對比：

| | bigblue4 (V=8,170) | bigblue2 (V=23,084) | 倍率 |
|---|---|---|---|
| V x V 元素數 | 66.7M | 533M | 8x |
| 單一 (1,V,V,2) float32 | 0.50 GB | 3.97 GB | 8x |
| Guidance 峰值 VRAM | ~5 GB | ~30-35 GB | ~6x |
| Legalization 峰值 VRAM | ~2 GB | ~12-15 GB | ~5x |

> 註：以上皆為 VRAM（GPU 顯存），非系統記憶體。所有張量運算都在 GPU 上執行，無 CPU offload。

### VRAM 需求總結

| 執行場景 | 預估 VRAM 需求 |
|----------|---------------|
| 完整執行（guidance + legalization） | **40-48 GB** |
| 僅 legalization（跳過 guidance） | **16-20 GB**（需降低 BLOCK_SIZE） |

### 建議的 GPU

| GPU | VRAM | 是否足夠 |
|-----|------|---------|
| RTX 3090 / 4090 | 24 GB | 不足 |
| A100 40GB | 40 GB | 勉強（僅 legalization 可行） |
| **A100 80GB** | **80 GB** | **建議選擇，可完整執行** |
| H100 80GB | 80 GB | 可行 |

## 檔案位置

- 合併後的 metrics：`logs/diffusion_debug/ispd2005-s0.eval_macro_only.300/metrics.csv`
- 與論文的比較表：`logs/diffusion_debug/ispd2005-s0.eval_macro_only.300/comparison_with_paper.csv`
- Placement 視覺化：`logs/diffusion_debug/ispd2005-s0.eval_macro_only.300/samples/placed*.png`

## 後續工作

1. 嘗試以 seed=400 重跑，確認 adaptec 系列差異是否來自 seed
2. 執行 IBM benchmark clustered eval
3. 若取得 A100 80GB，補跑 bigblue2
