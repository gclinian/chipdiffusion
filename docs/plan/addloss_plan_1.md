# AddLoss 第一次實驗計畫

> 參考：`docs/next/addloss_init.md`（可行性報告）
> 教授建議：`docs/meet/meet_0403.md` 第 1 點

## 目標

驗證在 supervised fine-tuning 的 denoising loss 之上，**加入 HPWL + legality auxiliary loss（direct gradient）** 是否能超越純 supervised fine-tuning（Ablation 10k：44.01）。

## 與過往實驗的區別

| 面向 | DDPO v2 | DDPO v2.3-v2.5 | Ablation 10k | **AddLoss v1 (新)** |
|------|---------|----------------|--------------|---------------------|
| Loss 組成 | Denoising + DDPO (global) + DDPO (local in v2.5) | + DDPO local reward | Denoising only | **Denoising + HPWL + Legality** |
| Gradient 類型 | Policy gradient | Policy gradient | Direct (MSE) | **Direct (all terms)** |
| Init | large-v2.ckpt | large-v2.ckpt | large-v2.ckpt | **large-v2.ckpt**（控制變因） |
| Train steps | 5000 | 5000 | 10000 | **10000**（對齊 Ablation 10k） |

**關鍵控制變因**：這次從 `large-v2.ckpt` 開始，跑 10000 steps，其他超參數跟 Ablation 10k 完全相同。**唯一差異就是多了 HPWL + legality loss**。這樣才能乾淨地量化 auxiliary loss 帶來的增益。

---

## 實作設計

### 修改檔案：`diffusion/models.py` 的 `ContinuousDiffusionModel.loss()`

目前的 loss 只有 denoising MSE。新版加入兩個 auxiliary term：

```python
def loss(self, x_0, cond, t, hpwl_weight=0.0, legality_weight=0.0):
    # 加 noise
    alpha_t = self._noise_scheduler.alpha(t)
    sigma_t = self._noise_scheduler.sigma(t)
    eps = torch.randn_like(x_0)
    x_t = alpha_t * x_0 + sigma_t * eps

    # Model 預測 noise
    eps_theta = self.forward(x_t, cond, t)

    # === 原本 denoising loss ===
    denoising_loss = F.mse_loss(eps_theta, eps)
    metrics = {"loss": denoising_loss.item()}

    # === 新增：predicted_x0 以及 HPWL/legality auxiliary loss ===
    if hpwl_weight > 0 or legality_weight > 0:
        predicted_x0 = (x_t - sigma_t * eps_theta) / alpha_t  # with grad
        predicted_x0 = torch.clamp(predicted_x0, -2, 2)

        total_loss = denoising_loss
        if hpwl_weight > 0:
            hpwl_loss = guidance.hpwl_guidance_potential(predicted_x0, cond).mean()
            total_loss = total_loss + hpwl_weight * hpwl_loss
            metrics["hpwl_loss"] = hpwl_loss.item()
        if legality_weight > 0:
            legality_loss = guidance.legality_guidance_potential(predicted_x0, cond).mean()
            total_loss = total_loss + legality_weight * legality_loss
            metrics["legality_loss"] = legality_loss.item()
        return total_loss, metrics

    return denoising_loss, metrics
```

### 修改檔案：`diffusion/train_graph.py`

在 `finetune` 模式下 read 新 config 並傳給 `model.loss()`：
```python
loss, model_metrics = model.loss(
    x, cond, t,
    hpwl_weight=cfg.addloss.get("hpwl_weight", 0.0),
    legality_weight=cfg.addloss.get("legality_weight", 0.0),
)
```

### 修改檔案：`diffusion/configs/mode/finetune.yaml`

新增 `addloss` 子區塊（預設 0，不啟用）：
```yaml
addloss:
  hpwl_weight: 0.0
  legality_weight: 0.0
```

---

## 實驗設定

### Run 1：AddLoss v1（保守起始）

| 項目 | 設定 | 備註 |
|------|------|------|
| method | addloss_v1 | |
| mode | finetune | 跟 Ablation 10k 一樣 |
| from_checkpoint | `../public-models/large-v2/large-v2.ckpt` | **與 Ablation/DDPO 相同 init（控制變因）** |
| train_steps | 10000 | 對齊 Ablation 10k |
| lr | 1e-5 | 同 |
| batch_size | 2 | 同 |
| **hpwl_weight** | **1e-3** | 保守起始（見下方量級分析）|
| **legality_weight** | **1e-2** | 保守起始 |
| eval_every | 5000 | |
| print_every | 200 | |

### Loss 權重量級分析

`denoising_loss` 典型值：0.1 - 0.3

`hpwl_guidance_potential` 的值：根據 ISPD 基準的 `hpwl_guidance_weight=16e-4` 可以推測它的 raw potential 量級約 100-300（與 hpwl_rescaled 類似）。

`legality_guidance_potential`：訓練初期（model 未收斂，predicted_x0 雜亂）可能較大；但因為 ground truth 是合法 placement，收斂後該趨近 0。

**選擇 alpha=1e-3, beta=1e-2 的理由**：
- `hpwl_weight × hpwl_loss ≈ 1e-3 × 200 = 0.2` → 與 denoising_loss 同量級
- `legality_weight × legality_loss ≈ 1e-2 × 10 = 0.1` → 稍小於 denoising_loss
- 保守起見寧可初期影響力小，確認不會干擾 denoising；之後可調大

如果實驗結果完全沒改善，下一個 Run 可試 `alpha=1e-2, beta=1e-1`（放大 10 倍）。

---

## 預期

### 與 Ablation 10k 的對比（7 circuit 平均，不含 bigblue2）

| 結果 | 意義 | 下一步 |
|------|------|-------|
| Avg HPWL < 44.01 | **Auxiliary loss 有效**，帶來真正的 extra signal | 調大 weight，試 timestep weighting |
| 44.01 ± 0.2 | 沒有顯著差異（誤差內），auxiliary loss 在 ground truth 下冗餘 | 試更大 weight，或換不同 init |
| > 44.5 | 干擾了 denoising，loss weight 太大或機制有問題 | 降 weight，檢查 gradient 有無爆炸 |

### 訓練穩定性指標

| 指標 | 目標 |
|------|------|
| Val loss (step 10000) | ≤ 0.25（Ablation 10k 是 0.17） |
| 無 val loss 爆炸或尖峰 | |
| hpwl_loss 下降趨勢 | 理論上應隨訓練下降 |
| legality_loss 下降趨勢 | 理論上應隨訓練下降 |

---

## 風險與應變

| 風險 | 徵兆 | 應變 |
|------|------|------|
| Loss 量級失衡 | hpwl_loss × weight 遠大於 denoising_loss | 降 weight 到 1e-4 或更小 |
| Gradient 爆炸 | Loss 突然變 NaN、val loss 尖峰 | 加 gradient clipping，或只對 late t 算 aux loss |
| Legality 早期爆炸 | legality_loss 在訓練初期 > 1000 | 加 timestep weighting（alpha_t²），或先只開 hpwl |
| 完全沒改善 | HPWL 跟 Ablation 10k 一樣 | 確認 gradient 有流過 predicted_x0；若沒問題則本方向冗餘 |

---

## 執行順序

1. 修改 `diffusion/models.py` 的 `loss()` 加 auxiliary loss 計算
2. 修改 `diffusion/train_graph.py` 傳遞 `hpwl_weight` / `legality_weight`
3. 修改 `diffusion/configs/mode/finetune.yaml` 加 `addloss` 子區塊
4. 在 tmux session 中跑 training + eval（sequential）
5. 分析結果並寫 `addloss_report_1.md`

### 時間估算

- 實作：~15 分鐘
- 訓練（10k steps × batch=2）：~20-25 分鐘（比 ablation 10k 慢一點，多算 aux loss）
- ISPD eval（8 circuits, 含 bigblue2）：~2.5 小時
- 分析 + 報告：~15 分鐘

**總計：~3 小時**

---

## 成功指標（7 circuit 平均 HPWL）

| 排名 | 方法 | Avg HPWL |
|------|------|----------|
| 目標 | **AddLoss v1** | **< 44.01** |
| 目前最佳 | Ablation 10k | 44.01 |
| | DDPO v2 | 44.65 |
| | Paper | 46.89 |

如果贏過 44.01，代表 direct gradient on auxiliary loss 真的帶來 beyond-supervised 的增益，而且路徑清晰（下一步可以試更大 weight、timestep weighting、或改從 ablation_10k init 疊加）。
