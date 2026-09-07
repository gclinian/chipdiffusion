# AddLoss 第二次實驗計畫（Timestep Weighting）

> 參考：`docs/report/addloss_report_1.md`
> 前次實驗：`addloss_v1` — 平均 45.39，略差於 Ablation 10k 的 44.01

## 目標

用 **timestep weighting (`alpha_t²`)** 抑制 auxiliary loss 在早期 timestep 的訊號，讓 HPWL/legality loss 只在 predicted_x0 接近 ground truth 時（late timestep）才真正影響模型。預期能解決 v1 在 **大 circuit 退步**（bigblue2 +13%, bigblue4 +7.4%）的問題。

---

## 動機（從 v1 學到的教訓）

### v1 做對了什麼

- `hpwl_loss` 從 211 下降到 149（-30%），機制確實有效
- 訓練穩定、無爆炸（對比 DDPO local reward 的 1e10 爆炸）
- 小 circuit 的 HPWL 全部改善（adaptec1, adaptec3, bigblue1, bigblue3）

### v1 的問題

- **大 circuit 退步**：bigblue2 +13%, bigblue4 +7.4%, adaptec4 +3.7%
- **Val loss 略升**（0.17 → 0.21），aux loss 輕微干擾 denoising
- 平均 HPWL 45.39 vs Ablation 10k 的 44.01

### 根本假說：早期 timestep 的 aux loss 不該存在

在 t 接近 T（純 noise 階段）時：
- `predicted_x0 = (x_t - σ_t × eps_theta) / α_t`
- α_t 小、σ_t 大 → predicted_x0 主要由 (x_t - σ_t × eps_theta) 決定，對 ground truth 的偏差被 1/α_t 放大
- **這個 predicted_x0 基本上是 noisy estimate**，對它算 HPWL/legality 沒有真正意義
- 但 v1 對所有 timestep 同等 weighting，等於讓模型浪費能量去優化這些 noisy estimates

在大 circuit 上這個問題被放大——因為 V 變大，predicted_x0 的 noise 影響更大，早期 timestep 的 "假訊號" 更有破壞力。

---

## Timestep Weighting 設計

### 公式選擇：`weight(t) = alpha_t²`

CosineScheduler 定義：
- `alpha_t = cos((π/2) × t)`, t ∈ [0, 1]
- t = 0: alpha = 1（乾淨）→ weight = 1
- t = 0.5: alpha = √2/2 → weight = 0.5
- t = 1: alpha = 0（純 noise）→ weight = 0

這個函數天然符合需求：**late timestep 全力參與，early timestep 自然衰減到 0**。

### 為什麼用 α²（不用 α 或 α⁴）

| 權重函數 | 早期 (t=0.8) | 中期 (t=0.5) | 晚期 (t=0.2) | 評估 |
|---------|-------------|-------------|-------------|------|
| α_t | 0.31 | 0.71 | 0.95 | 太溫和，早期仍有可觀訊號 |
| **α_t²** | **0.10** | **0.50** | **0.91** | **合適：早期衰減明顯，晚期幾乎滿量** |
| α_t⁴ | 0.01 | 0.25 | 0.82 | 太極端，中期都被壓掉 |

### Loss 公式

```python
# 原 v1
hpwl_loss = hpwl_potential.mean()  # scalar
total_loss += hpwl_weight * hpwl_loss

# 新 v2（timestep weighting）
hpwl_per_sample = hpwl_potential  # (B,)
timestep_weight = alpha_t.squeeze() ** 2  # (B,)
hpwl_loss = (timestep_weight * hpwl_per_sample).mean()  # scalar
total_loss += hpwl_weight * hpwl_loss
```

Legality loss 同樣處理。

---

## 實作變更

### `diffusion/models.py` `ContinuousDiffusionModel.loss()`

```python
def loss(self, x, cond, _, hpwl_weight=0.0, legality_weight=0.0,
         use_timestep_weighting=False):
    # ... (same as before: sample t, noise x, predict eps, denoising_loss)

    if hpwl_weight > 0 or legality_weight > 0:
        alpha_t = self._noise_scheduler.alpha(t).view((B, *([1] * input_dims)))
        sigma_t = self._noise_scheduler.sigma(t).view((B, *([1] * input_dims)))
        predicted_x0 = (x_perturbed - sigma_t * eps_predict) / alpha_t
        predicted_x0 = torch.where(mask, x, predicted_x0) if mask is not None else predicted_x0
        predicted_x0 = torch.clamp(predicted_x0, -2, 2)

        # Compute per-sample loss (shape B)
        total_loss = denoising_loss

        # Timestep weighting factor (shape B)
        if use_timestep_weighting:
            alpha_t_1d = self._noise_scheduler.alpha(t)  # (B,)
            weights = alpha_t_1d ** 2  # (B,)
        else:
            weights = torch.ones_like(t)  # (B,)

        if hpwl_weight > 0:
            hpwl_per_sample = guidance.hpwl_guidance_potential(predicted_x0, cond)  # (B,)
            hpwl_loss = (weights * hpwl_per_sample).mean()
            total_loss = total_loss + hpwl_weight * hpwl_loss
            metrics["hpwl_loss"] = hpwl_loss.detach().cpu().item()

        if legality_weight > 0:
            legality_per_sample = guidance.legality_guidance_potential(predicted_x0, cond, mask=mask)  # (B,)
            legality_loss = (weights * legality_per_sample).mean()
            total_loss = total_loss + legality_weight * legality_loss
            metrics["legality_loss"] = legality_loss.detach().cpu().item()

        return total_loss, metrics

    return denoising_loss, metrics
```

### `diffusion/train_graph.py`

```python
loss, model_metrics = model.loss(
    x, cond, t,
    hpwl_weight=addloss.get("hpwl_weight", 0.0),
    legality_weight=addloss.get("legality_weight", 0.0),
    use_timestep_weighting=addloss.get("use_timestep_weighting", False),
)
```

---

## 實驗設定

### Run 2：AddLoss v2（Timestep Weighting）

| 項目 | 設定 | vs v1 差異 |
|------|------|-----------|
| method | addloss_v2 | — |
| mode | finetune | 同 |
| from_checkpoint | `../public-models/large-v2/large-v2.ckpt` | **同（控制變因）** |
| train_steps | 10000 | 同 |
| lr | 1e-5 | 同 |
| batch_size | 2 | 同 |
| hpwl_weight | 1e-3 | **同**（為了純粹比較 timestep weighting 效果，先不調權重）|
| legality_weight | 1e-2 | **同** |
| **use_timestep_weighting** | **True** | **新加入** |

### 為什麼保持 weight 不變

Timestep weighting 會讓 effective weight 減半（因為 E[α²] ≈ 0.5 for uniform t），理論上可能需要把 weight 倍增才能維持同樣的「晚期訊號強度」。但為了乾淨比較 **timestep weighting 的效果**，先不調 weight。

如果 v2 結果比 v1 好但還是輸給 Ablation 10k，下一個 run 可以試 `hpwl_weight=2e-3` 補償。

---

## 預期

### 理想情況

| Circuit | v1 | v2 預期 | 機制 |
|---------|-----|---------|------|
| adaptec1 | **9.47**（改善） | 9.47 ± 0.1 | 小 circuit，aux 影響不大 |
| adaptec2 | 33.54（退） | 32-33 | 中型 circuit，timestep weighting 應幫助 |
| adaptec3 | **51.60**（改善） | 51-52 | 持平 |
| adaptec4 | 56.14（退） | 54-55 | 大 circuit，timestep weighting 關鍵 |
| bigblue1 | **2.59**（改善） | 2.59 ± 0.05 | 小 circuit，持平 |
| bigblue2 | 66.49（大退） | 58-60 | 最大 circuit，應大幅改善 |
| bigblue3 | **30.32**（改善） | 30-31 | 持平 |
| bigblue4 | 134.05（退） | 125-130 | 大 circuit，應改善 |
| **Avg (7)** | 45.39 | **< 44.5** | 目標：贏過 v1，逼近 Ablation 10k |

### 判定標準

| 結果 | 意義 | 下一步 |
|------|------|-------|
| Avg HPWL < 44.01（贏 Ablation 10k） | **timestep weighting 是關鍵**，aux loss 有效 | 調高 weight、改從 ablation_10k init |
| 44.01 < avg < 44.5（接近） | timestep weighting 部分有效 | 試 2e-3 weight 補償 |
| 44.5 < avg < 45.39（比 v1 好） | timestep weighting 改善大 circuit 退步 | 調整方向繼續 |
| avg ≥ 45.39（沒比 v1 好） | timestep weighting 無效，aux loss 本身就是冗餘 | 放棄 AddLoss 方向 |

### 訓練指標預期

| 指標 | v1 (step 10000) | v2 預期 |
|------|----------------|---------|
| hpwl_loss | 149（只算 mean，等價於對所有 t） | ~75-100（因為 early t 被壓） |
| legality_loss | 1.93 | ~1-1.5 |
| val/loss | 0.21 | ≤ 0.19（接近 Ablation 10k 的 0.17，因為早期 denoising 不被干擾） |
| 訓練穩定性 | 穩定 | 穩定 |

---

## 執行順序

1. 修改 `diffusion/models.py` 加 `use_timestep_weighting` 參數
2. 修改 `diffusion/train_graph.py` 傳遞新參數
3. 在 tmux session 中跑 training + eval（sequential）
4. 分析結果並寫 `addloss_report_2.md`

### 時間估算

| 階段 | 時間 |
|------|------|
| 實作 | ~10 分鐘 |
| 訓練（10k steps, batch=2） | ~20 分鐘 |
| ISPD eval（8 circuits） | ~2.5 小時 |
| 分析 + 報告 | ~15 分鐘 |
| **總計** | **~3 小時** |

---

## 成功指標（7 circuit 平均 HPWL）

| 排名 | 方法 | Avg HPWL |
|------|------|----------|
| 目標 | **AddLoss v2** | **< 44.01** |
| 目前最佳 | Ablation 10k | 44.01 |
| | DDPO v2 | 44.65 |
| | AddLoss v1 | 45.39 |
| | Paper | 46.89 |

**如果 v2 能達到 44.5 以下**，就證明 timestep weighting 是 AddLoss 方向的關鍵，後續可以繼續優化（調 weight、改 init）。**如果 v2 ≥ 45.39，就應該徹底放棄 AddLoss 方向**，轉向其他想法（Best-of-N、cluster-aware guidance 等）。
