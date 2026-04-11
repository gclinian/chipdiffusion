# DDPO 第二次實驗計畫

> 參考：`docs/next/ddpo_next_1.md`（第一次實驗回顧）

## 目標

解決第一次 DDPO 實驗的三大問題：catastrophic forgetting、缺少 legality reward、REINFORCE 太 naive。在 ISPD2005 上穩定地超越 baseline。

---

## 改進項目

### 改進 1：PPO Clipping（防止 policy 偏移）

**問題**：第一次實驗用的 REINFORCE 沒有限制更新幅度，模型一步走太遠。

**做法**：在 `ddpo.py` 的 `loss()` 中加入 PPO-style clipping。

```python
# 在 reverse_samples 前，先用 detach 的模型算一次 log_prob_old
with torch.no_grad():
    _, _, log_prob_old = self.model.reverse_samples(..., output_log_prob=True)

# 正常 reverse_samples 拿 log_prob_new
x_0, _, log_prob_new = self.model.reverse_samples(..., output_log_prob=True)

# PPO clipping
ratio = torch.exp(log_prob_new.mean(dim=0) - log_prob_old.mean(dim=0))
clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
loss = -torch.mean(torch.min(ratio * advantage, clipped_ratio * advantage))
```

**注意**：需要跑兩次 reverse_samples（一次 no_grad 算 old，一次有 grad 算 new），VRAM 不會增加太多因為 old 那次不存計算圖。但時間會翻倍。

**修改檔案**：
- `diffusion/ddpo.py`：加 `clip_epsilon` 參數，修改 `loss()` 方法
- `diffusion/configs/mode/ddpo.yaml`：新增 `clip_epsilon: 0.2`

### 改進 2：混合 Supervised Loss（防止 forgetting）

**問題**：DDPO loss 不監督 denoising 品質，模型找到捷徑（不 denoise，直接輸出固定 pattern）。

**做法**：每個 training step 同時算 supervised denoising loss，加到 total loss。

```python
# 在 train_graph.py 的 training loop 中
if cfg.mode == "ddpo":
    ddpo_loss, model_metrics = ddpo_model.loss(x, cond)
    # 同時算 supervised loss
    t = torch.randint(1, cfg.model.max_diffusion_steps + 1, [x.shape[0]], device=device)
    supervised_loss, _ = model.loss(x, cond, t)
    loss = ddpo_loss + cfg.ddpo.supervised_weight * supervised_loss
```

**超參數**：`supervised_weight` 建議從 1.0 開始。太大會壓制 DDPO 效果，太小防不住 forgetting。

**修改檔案**：
- `diffusion/train_graph.py`：DDPO 分支加 supervised loss
- `diffusion/configs/mode/ddpo.yaml`：新增 `supervised_weight: 1.0`

### 改進 3：Legality Reward（必須 > 0）

**問題**：第一次 legality_weight=0.0，導致 adaptec1 legality 從 0.994 暴跌到 0.919。

**做法**：設定 `legality_weight=0.5`。

**修改檔案**：
- `diffusion/configs/mode/ddpo.yaml`：`legality_weight: 0.5`

### 改進 4：Local Reward（每步都給 reward）

**問題**：目前只在最終 step 給 global reward，中間步驟沒有回饋。1000 步中哪一步做對了？模型不知道。

**做法**：在 reverse sampling 的每個 step，用 predicted x_0 計算 HPWL/legality 作為 local reward。

```python
# 在 reverse_samples 的 output_log_prob 路徑中
# 每步已經有 predicted_x0 = (x - sigma_t * eps_predict) / alpha_t
local_reward = reward_fn(predicted_x0, cond)  # 每步的 reward

# 最終 loss 結合 local 和 global reward
global_reward = reward_fn(x_0, cond)  # 最後一步的 reward
total_reward = global_weight * global_reward + local_weight * mean(local_rewards)
```

**注意**：
- local reward 計算 legality_guidance_potential 會建 V×V 矩陣，每步都算的話 VRAM 會大增
- 建議只在每 N 步算一次 local reward（例如每 10 步），或只算 HPWL（不需要 V×V）
- 這是最大的改動，建議放在改進 1-3 驗證後再做

**修改檔案**：
- `diffusion/ddpo.py`：修改 `loss()` 支援 local reward
- `diffusion/models.py`：`reverse_samples()` 在 `output_log_prob` 路徑中回傳 intermediate predicted_x0

---

## 實驗設計

### Run 1：PPO + Supervised Loss + Legality（最小改動組合）

先加改進 1-3，不加 local reward（改動最大、風險最高）。

| 項目 | 設定 |
|------|------|
| method | ddpo_v2_ppo |
| batch_size | 2 |
| num_timesteps | 50 |
| lr | 1e-5 |
| train_steps | 5000 |
| clip_epsilon | 0.2 |
| supervised_weight | 1.0 |
| legality_weight | 0.5 |
| hpwl_weight | 1.0 |
| from_checkpoint | large-v2.ckpt |

**觀察重點**：
- val loss 是否穩定（不爆炸）
- reward 是否仍能改善
- ISPD eval 是否穩定優於 baseline

### Run 2：調整 supervised_weight

根據 Run 1 結果調整：
- 如果 val loss 穩定但 reward 不動 → 降 supervised_weight（0.5 → 0.1）
- 如果 val loss 仍爆炸 → 升 supervised_weight（1.0 → 5.0）

### Run 3（可選）：加 Local Reward

在 Run 1/2 穩定後，加入改進 4 的 local reward，驗證是否進一步改善。

---

## 執行順序

1. **修改程式碼**
   - [ ] `ddpo.py`：加 PPO clipping
   - [ ] `train_graph.py`：加混合 supervised loss
   - [ ] `ddpo.yaml`：更新超參數
2. **驗證修改**
   - [ ] 用小 batch 跑 100 steps，確認 loss 計算正確、沒有 OOM
3. **Run 1：完整訓練 5000 steps**
   - [ ] 觀察 val loss 和 reward 趨勢
   - [ ] 在 ISPD2005 上 eval
4. **Run 2：調整超參數**
5. **Run 3（可選）：加 local reward**

---

## 成功指標

| 指標 | 第一次結果 | 這次目標 |
|------|-----------|---------|
| val loss（5000 steps 後） | 264（初始 0.17） | < 10（控制 forgetting） |
| adaptec1 HPWL | 10.57（baseline 10.22） | < 10.22（至少不變差） |
| adaptec1 legality | 0.919 | > 0.99（不能下降） |
| adaptec3 HPWL | 58.48（baseline 62.14） | < 58（維持改善） |
| 所有 circuit legality | 最低 0.919 | 全部 > 0.99 |

---

## VRAM 估算

| 項目 | 估算 |
|------|------|
| 原本 DDPO（batch=2, ts=50） | ~22 GB |
| + PPO（多一次 no_grad reverse sampling） | +0（no_grad 不存計算圖，只多一點時間） |
| + Supervised loss（一次 forward + backward） | +2-3 GB |
| **Total** | **~24-25 GB** |

可能會剛好超過 24GB。如果 OOM，備案是：
- 降 num_timesteps 到 40
- 或降 batch_size 到 1（但 variance 更高）
