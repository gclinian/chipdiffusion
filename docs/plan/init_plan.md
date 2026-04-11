# 改進 ChipDiffusion 的嘗試方向

> 目標：在 ISPD2005 benchmark 上擊敗目前 ChipDiffusion 的 HPWL 結果

## 現有資源盤點

| 資源 | 說明 |
|------|------|
| Pre-trained checkpoint | `large-v2.ckpt`（AttGNN backbone, large size） |
| DDPO (RL fine-tuning) | `diffusion/ddpo.py`，已實作 reward function（legality + HPWL） |
| 多種 backbone | AttGNN, GraphTransformer, ResGNN, GraphUNet, MLP, UNet, ViT |
| 多種 GNN conv layer | GCN, SAGE, GIN, GAT, Transformer, Gated |
| Guidance 機制 | HPWL guidance + legality guidance（可微分） |
| Legalization | opt-adam, scheduled weight, SGD 等多種方式 |
| Sampling 策略 | open_loop, clustered, multi-sample, iterative |
| 訓練模式 | train, finetune, ddpo |
| Benchmarks | ISPD2005 (8 circuits), IBM (18 circuits), v1 synthetic data |

---

## 方向一：DDPO 強化學習微調（最直接）

### 概念
利用已寫好的 `ddpo.py`，以 HPWL 和 legality 作為 reward signal，對 pre-trained model 進行 RL fine-tuning。讓模型從「模仿訓練資料的分佈」轉變為「直接最佳化目標指標」。

### 為何可行
- 程式碼已經寫好，包含 `legality_reward()` 和 `hpwl_reward()` 兩個 reward function
- 有現成的 config `configs/mode/ddpo.yaml`（batch=4, 100k steps）
- Pre-trained model 已經有不錯的基礎，RL 只需微調
- 論文本身沒有報告 DDPO 在 ISPD2005 上的結果，可能有提升空間

### 具體做法
1. 從 `large-v2.ckpt` 開始，用 DDPO 模式 fine-tune
2. 調整 reward 權重：`legality_weight` vs `hpwl_weight`（目前 config 是 0.0 / 1.0）
3. 可以先在小 circuit（adaptec1, bigblue1）上快速迭代

### 風險
- RL 訓練不穩定，reward hacking 可能導致品質下降
- Batch size 受限於 GPU（24GB），收斂可能慢
- 需要小心 reward shaping，避免 legality 大幅下降

### 預估難度：低（程式碼已有）

---

## 方向二：Guidance 超參數搜索

### 概念
目前 guidance 的超參數（步數、權重、learning rate）是固定的。系統性搜索這些參數可能找到更好的組合。

### 可調參數
| 參數 | 目前值 | 搜索範圍建議 |
|------|--------|-------------|
| `model.grad_descent_steps` (guidance steps) | 20 | 10-100 |
| `model.hpwl_guidance_weight` | 16e-4 | 1e-4 ~ 1e-2 |
| `legalization.grad_descent_steps` | 20000 | 10000-50000 |
| `legalization.alpha_lr` | 8e-3 | 1e-3 ~ 5e-2 |
| `legalization.hpwl_weight` | 12e-5 | 1e-5 ~ 1e-3 |

### 具體做法
1. 挑一個中等大小的 circuit（如 adaptec3）作為 proxy
2. Grid search 或 random search 上述參數
3. 找到最佳組合後套用到所有 circuits

### 風險
- 搜索空間大，每次 eval 耗時長（adaptec3 約 7 分鐘）
- 可能 overfit 到特定 circuit
- 改善幅度可能有限（guidance 本身的天花板）

### 預估難度：低

---

## 方向三：Multi-sample + Best Selection

### 概念
利用 `policies.py` 中已有的 `open_loop_multi` 策略，對同一個 circuit 生成多個 placement，選 HPWL 最低的。

### 為何可行
- 程式碼已有 `open_loop_multi` policy
- Diffusion model 本質是 stochastic，不同 noise seed 會產生不同結果
- 不需要任何模型改動

### 具體做法
1. 對每個 circuit 跑 N 次（e.g., N=5-10），用不同 random seed
2. 各自做 guidance + legalization
3. 選 HPWL 最低且 legality > 0.99 的結果

### 風險
- 計算成本線性增長（N 倍時間）
- 改善幅度可能有限（同一個 model 的 variance）

### 預估難度：很低

---

## 方向四：Graph Transformer 替換 AttGNN

### 概念
目前的 `large-v2.ckpt` 使用 AttGNN backbone。Repo 中有 Graph Transformer (`networks/gt.py`) 的實作，可能有更強的表達能力。

### 為何值得嘗試
- Graph Transformer 在其他圖任務上表現優異
- Attention 機制可以更好地捕捉 long-range dependency（重要：HPWL 涉及全域 wire length）
- 已有 config：`configs/model/graph-transformer.yaml`

### 具體做法
1. 用 Graph Transformer 從頭訓練（需要 v1 + v2 data）
2. 或嘗試從 AttGNN checkpoint 做 architecture distillation
3. 先在 synthetic data 上驗證收斂性

### 風險
- 需要完整重訓（時間長）
- Graph Transformer 的 attention 是 O(V^2)，大 circuit 可能更吃記憶體
- 不確定在這個任務上是否比 AttGNN 好

### 預估難度：中（需重訓）

---

## 方向五：GNN Conv Layer 替換

### 概念
目前 AttGNN 使用的 conv layer 可以替換。Repo 支援 GCN, SAGE, GIN, GAT, Transformer, Gated 等多種。

### 值得嘗試的組合
- **GAT (Graph Attention)**：attention-weighted aggregation，可能更好地處理異質 node
- **GIN (Graph Isomorphism Network)**：理論上最強的 message passing GNN
- **Transformer conv**：類似 GAT 但加入 key-value attention

### 具體做法
1. 修改 config 中的 conv layer 設定
2. Fine-tune from `large-v2.ckpt`（凍結部分層，只替換 conv layer）
3. 或從頭訓練（較慢但更完整）

### 風險
- Fine-tune 時 layer 不相容可能需要額外處理
- 從頭訓練成本高

### 預估難度：中

---

## 方向六：Iterative Clustering 策略

### 概念
`policies.py` 中有 `iterative_clustering` 策略：先 cluster → diffusion placement → SC placement → 再 cluster，反覆迭代。這可能比 one-shot 方法產生更好的結果。

### 為何可行
- 程式碼已實作
- Clustering 可以降低問題規模，讓模型處理更 manageable 的子問題
- 對大 circuit（adaptec4, bigblue4）可能特別有效

### 具體做法
1. 在 ISPD2005 上試跑 `iterative_clustering` policy
2. 調整 cluster size 和迭代次數
3. 與目前的 `open_loop` + guidance 結果比較

### 風險
- 多次迭代增加計算時間
- Clustering 品質影響最終結果
- 需要 hmetis（已安裝）

### 預估難度：低（程式碼已有）

---

## 方向七：Two-stage Legalization

### 概念
目前用單一階段的 opt-adam legalization。Repo 中有 `adam_2stage.yaml` 的 SC placer config，類似的 two-stage 思路可應用於 legalization。

### 具體做法
1. 第一階段：高 legality weight，快速消除 overlap
2. 第二階段：高 HPWL weight，在合法的基礎上最佳化 wire length
3. 調整 `legalization.py` 中的 weight scheduling

### 風險
- 需要修改 legalization 程式碼
- 兩階段的切換點需要調整

### 預估難度：中

---

## 建議優先順序

| 優先級 | 方向 | 理由 |
|--------|------|------|
| 1 | 方向三：Multi-sample | 零成本改動，立即可測試，建立 baseline variance |
| 2 | 方向二：Guidance 超參數搜索 | 不需改 code，可能有低垂果實 |
| 3 | 方向一：DDPO 微調 | 程式碼已有，最有潛力的改進方向 |
| 4 | 方向六：Iterative Clustering | 程式碼已有，對大 circuit 有潛力 |
| 5 | 方向四/五：換 backbone/conv | 需要重訓，但長期可能帶來根本性改進 |
| 6 | 方向七：Two-stage Legalization | 需要改 code，改善幅度不確定 |

---

## 下一步行動

1. **立即可做**：用 multi-sample (N=5) 跑一次 adaptec1，看 variance 有多大
2. **短期**：在 adaptec1 上做 guidance 超參數搜索
3. **中期**：嘗試 DDPO fine-tuning，先在小 circuit 上驗證
4. **長期**：考慮 Graph Transformer 重訓或 GNN conv layer 替換
