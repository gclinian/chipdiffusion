# DDPO 第一次實驗回顧 — 給 ddpo_plan_2 的參考

## 第一次實驗做了什麼

- 用 DDPO（REINFORCE with EMA baseline）對 pre-trained `large-v2.ckpt` 做 fine-tuning
- 在 synthetic data (v1.61-ddpo, 1600 train / 400 val) 上訓練 5000 steps
- reward 只有 HPWL（legality_weight=0.0, hpwl_weight=1.0）
- batch_size=2, num_timesteps=50, lr=1e-5
- 訓練後在 ISPD2005 samples 0-4 上 eval

## 名詞說明

- **val loss**：用 supervised 方式衡量模型的 denoising 能力。對 val data 加已知 noise，讓模型預測 noise，算 MSE。val loss 低 = denoising 能力好。DDPO 訓練時 val loss 只是觀察指標，不參與權重更新。

## 結果

- Reward 改善了（-241 → -102），但 val loss 從 0.17 爆到 130,000
- ISPD eval：adaptec3/4 改善 4-6%，adaptec1/bigblue1 變差，adaptec1 legality 暴跌（0.994 → 0.919）
- 沒有任何 circuit 超越 paper
- 結論：catastrophic forgetting 嚴重，模型不在正常 denoise

## 失敗原因分析

### 1. legality_weight=0.0（最明顯的錯誤）
- 直接沿用 ddpo.yaml 預設值，完全沒給 legality reward
- 導致模型不在意合法性，adaptec1 legality 大幅下降
- **下次必須設 legality_weight ≥ 0.5**

### 2. 沒有防止 catastrophic forgetting 的機制
- DDPO loss 只有 `-mean(log_prob * advantage)`，沒有任何東西阻止模型偏離 pre-trained 行為
- 模型找到捷徑：不好好 denoise，直接輸出固定低 HPWL pattern
- val loss 爆炸就是證據

### 3. REINFORCE 太 naive
- 目前實作是最簡單的 policy gradient，沒有 trust region 或 clipping
- 每次更新幅度不受限，容易一步走太遠

### 4. VRAM 限制
- 24GB GPU 只能跑 batch=2, timesteps=50
- batch=2 導致 gradient variance 很高
- timesteps=50 跟 eval 時的 1000 步差距太大

## 下次應該改進的方向

### 優先級 1：加 PPO clipping
```python
ratio = exp(log_prob_new - log_prob_old)
clipped_ratio = clip(ratio, 1-ε, 1+ε)  # ε=0.2
loss = -mean(min(ratio * advantage, clipped_ratio * advantage))
```
- 限制每次更新幅度，防止 policy 偏移太多
- DDPO 原始論文的 DDPO-IS 變體就是用類似機制
- 需要多存一份更新前的 log_prob_old

### 優先級 2：混合 supervised loss
```python
total_loss = ddpo_loss + α * supervised_denoising_loss
```
- 在追求 HPWL reward 的同時，保持 denoising 能力
- α 需要調，建議從 1.0 開始

### 優先級 3：legality_weight > 0
- 設 legality_weight=0.5 或 1.0
- 讓模型同時最佳化 HPWL 和 legality

### 優先級 4（如果有更大 GPU）
- batch_size=4-8（降低 variance）
- num_timesteps=100-200（更接近 eval 設定）
- 需要 A100 40-80GB

## 需要修改的程式碼

| 檔案 | 改動 | 難度 |
|------|------|------|
| `ddpo.py` | 加 PPO clipping（存 log_prob_old, 算 ratio, clip） | 中 |
| `train_graph.py` | 每個 DDPO step 同時算 supervised loss 並加到 total loss | 低 |
| `ddpo.yaml` | `legality_weight: 0.5`, 新增 `clip_epsilon: 0.2` | 低 |

## 要驗證的假設

1. PPO clipping 是否能有效防止 val loss 爆炸？
2. 混合 supervised loss 後，HPWL reward 還能改善嗎？（可能會變慢）
3. legality reward 加入後，模型能同時改善 HPWL 和 legality 嗎？
4. 在 ISPD eval 上，改進後的 DDPO 能否穩定地比 baseline 好？
