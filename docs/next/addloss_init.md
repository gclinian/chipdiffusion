# 可行性報告：Loss 加入 HPWL / Legality（用於 Fine-tuning）

> 對應教授建議：`docs/meet/meet_0403.md` 第 1 點
> 相關實驗：`ddpo_report_5.md`（local reward 三次實驗都失敗）

## 你提出的兩個疑問

**Q1：既然 DDPO fine-tune 時加 local reward 三次都失敗（v2.3, v2.4, v2.5），那把 HPWL/Legality 加到 training loss 是不是也會失敗？**

結論：**不會，這兩個是本質上不同的機制**，且這個方向**可行性高**。

**Q2：加 HPWL/Legality loss 一定要重新 train 嗎？能不能拿來 fine-tune？**

**完全可以 fine-tune，而且 fine-tune 才是正確做法**（細節見下方「階段選擇」段落）。

---

## 階段選擇：Fine-tune vs Train from Scratch

雖然教授建議是「在 training 階段加入 HPWL/legality loss」，但實作上**新增的 loss 在兩個階段都適用**。差別只在 init weights。

| 選項 | Init | 可行性 | 理由 |
|------|------|--------|------|
| Train from scratch | Random init | ❌ **不推薦** | 需要大量合成 data 和 GPU 時間，我們負擔不起，且會失去 pretrained model 的 placement 基礎知識 |
| **Fine-tune from `large-v2.ckpt`** | Paper 的原始 pretrained | ✅ 推薦 | 類似 DDPO v2 的 setup，但改用 direct gradient |
| **Fine-tune from `ablation_supervised_10k`** | 目前最佳 checkpoint（44.01） | ✅✅ **最推薦** | 從目前最強起跳，HPWL loss 做最後一哩路 polish |

**ReFL 本身就是 fine-tuning 方法**。它的完整名稱是 "Reward Feedback Learning for **Fine-tuning** Text-to-Image Diffusion Models"——這個方法就是為 fine-tune 設計的。

Code 機制上 training 和 fine-tuning 完全一樣，只是 checkpoint init 不同：

```python
# 這個 loss 在 train 或 fine-tune 階段都能直接套用
predicted_x0 = (x_t - sigma_t * eps_theta) / alpha_t
hpwl_loss = -hpwl_reward(predicted_x0, cond).mean()
legality_loss = -legality_reward(predicted_x0, cond).mean()
total_loss = denoising_loss + alpha * hpwl_loss + beta * legality_loss
```

---

---

## 兩者的關鍵差異：Policy Gradient vs Direct Gradient

### DDPO Local Reward 的運作方式

```python
# predicted_x0 是從 model 輸出算出來的 placement
local_reward = hpwl_reward(predicted_x0, cond).detach()  # ← DETACH!

# advantage 計算
advantage = (local_reward - mean) / std

# loss：透過 log_prob 回傳 gradient
loss = -log_prob(action) * advantage
```

**關鍵**：
- `local_reward` 是 **detach 的純量**（沒有 gradient 回傳）
- Gradient 只透過 `log_prob * advantage` 流回 model
- 這是 **policy gradient**（Monte Carlo estimator），variance 很大
- Advantage normalization 會丟失訊息、放大雜訊

### Training Loss 加 HPWL 的運作方式

```python
# 在 training 時算 predicted_x0（WITH gradient）
predicted_x0 = (x_t - sigma_t * eps_theta) / alpha_t  # ← 保留 grad

# 直接對 predicted_x0 算 HPWL
hpwl_loss = hpwl_reward(predicted_x0, cond)  # ← WITH gradient

# 直接加進 loss
total_loss = denoising_loss + alpha * hpwl_loss + beta * legality_loss
```

**關鍵**：
- HPWL 值本身就是 loss 的一部分，**gradient 直接流回**
- 沒有 detach、沒有 advantage、沒有 log_prob
- 這是 **direct gradient**（每個 parameter 都有精確的 partial derivative）
- Gradient 量級自動由 backward 決定，不需 normalization

### 為什麼這個差異很重要

| 面向 | DDPO Local Reward | Training Loss + HPWL |
|------|------------------|---------------------|
| Gradient 類型 | Policy gradient (REINFORCE) | Direct gradient (supervised-like) |
| 資訊量 | O(1) per trajectory（1 個 scalar） | O(V×D) per step（full vector field） |
| Variance | 高（因為 advantage normalization） | 低（backward 自動歸一化） |
| 量級爆炸風險 | 有（v2.4 看到 1e10+） | 無（backward 只看 gradient norm） |
| Sample efficiency | 很低（RL 本質） | 高（supervised 本質） |

**關鍵洞察**：DDPO local reward 失敗的三個原因（reward 爆炸、variance 高、資訊量低）都只存在於 policy gradient 下。Direct gradient **完全沒有這些問題**。

---

## 這個想法其實有文獻依據：ReFL

這個概念在 text-to-image diffusion 領域已經有成功案例：

- **DDPO / DPOK** (2023)：text-to-image 用 RL (policy gradient)，跟我們目前的 DDPO 類似
- **ReFL** (Reward Feedback Learning, 2023)：直接對 predicted x₀ 算 reward gradient，比 DDPO 好很多
- **DRaFT** (Direct Reward Fine-Tuning, 2023)：進一步改進 ReFL

ReFL 的 key insight 就是：**跳過 policy gradient，直接用 reward 作為 supervised loss**。這跟教授的建議本質上是同一件事。

差別：
- ReFL 是在 **fine-tuning** 階段用
- 教授建議是在 **training from scratch** 階段用
- 但機制（direct gradient on predicted_x0）完全一樣

---

## 但要小心的問題

### 問題 1：Predicted x₀ 在早期 timestep 仍然 ill-defined

跟 local reward 的情況一樣，predicted x₀ 在 t 接近 T 時是 noisy scaled 的值，對它算 HPWL 不一定有意義。

**但這次比 local reward 問題小**，因為：

1. **Training 時有 ground truth**：supervised loss（MSE on noise prediction）會強力把模型拉向正確方向，HPWL loss 只是輔助
2. **Gradient 量級自動穩定**：就算 HPWL 值在早期很大，gradient norm 會由 ∂HPWL/∂predicted_x0 × ∂predicted_x0/∂params 決定，不會無限爆炸
3. **可以用 timestep weighting**：簡單地對 HPWL loss 乘 `alpha_t²` 就能讓早期 loss 趨近 0

### 問題 2：Ground truth 已經是 low-HPWL placement

Training data 的 placement 已經是好的（低 HPWL、高 legality），所以 `HPWL(x_0) ≈ HPWL(ground_truth)` 已經很低。

**這帶來一個 concern**：如果 supervised loss 已經把模型推向 ground truth，HPWL loss 會不會只是**冗餘訊號**？

我的判斷：**部分冗餘，但仍有價值**：
- 在模型還沒完全收斂時，`HPWL(predicted_x0) > HPWL(ground_truth)`，此時 HPWL loss 提供有用訊號
- 在模型已收斂時，HPWL loss 趨近 0，不會造成干擾
- **本質上是 auxiliary loss（輔助 loss），不是取代 denoising loss**

### 問題 3：V×V matrix 的計算成本

Legality loss 需要算 `(B, V, V, D)` tensor。Training 時 V 比 fine-tuning 時的還大（depending on dataset），需要評估：

- v1.61 training data: V ≈ 200-500 → V×V tensor 0.6-4 MB → 沒問題
- 完整 synthetic data set: V 可能更大 → 可能需要用 `legality_guidance_potential_tiled`

VRAM 估算（對 v1.61 v=300, batch=2）：
- Per forward pass: ~1.4 MB × 多層 autograd ≈ 30 MB
- 當前 supervised training: ~5 GB → 加上 HPWL/legality ≈ 5-6 GB
- **完全 ok**，VRAM 不會是瓶頸

### 問題 4：Loss 權重調校

HPWL loss 和 legality loss 的量級不一定和 denoising loss 相同，需要調 weight：

```python
total_loss = denoising_loss + alpha * hpwl_loss + beta * legality_loss
```

建議起始值：`alpha = 1e-3, beta = 1e-2`（保守，先確認不會 harm denoising），之後再調高。

---

## 為什麼這個方法比 DDPO 可能更有效

1. **Ablation 10k 是目前最佳方法（44.01）**——純 supervised fine-tuning。這意味著 **supervised learning 是這個 task 的主要驅動力**
2. **加 HPWL/legality loss 是 supervised fine-tuning 的自然擴展**——不是改方向，是強化
3. **Direct gradient 的 sample efficiency 遠高於 policy gradient**——同樣的訓練資料能學到更多
4. **用 fine-tune 而非 train-from-scratch**——沿用 pretrained model 的基礎知識，避免大量計算

---

## 實作設計

### 最小版本（First Experiment）

```python
def loss(self, x_0, cond, t):
    # x_0: ground truth placement (B, V, D)
    # 加 noise
    eps = sample_epsilon(...)
    alpha_t = scheduler.alpha(t)
    sigma_t = scheduler.sigma(t)
    x_t = alpha_t * x_0 + sigma_t * eps

    # Model 預測 noise
    eps_theta = self.forward(x_t, cond, t)

    # === 原本 denoising loss ===
    denoising_loss = F.mse_loss(eps_theta, eps)

    # === 新增：predicted_x0 以及 HPWL/legality loss ===
    predicted_x0 = (x_t - sigma_t * eps_theta) / alpha_t  # with grad
    
    # 邊界 clip (避免 gradient 爆炸)
    predicted_x0 = torch.clamp(predicted_x0, -2, 2)
    
    # 新 loss
    hpwl_loss = -hpwl_reward(predicted_x0, cond).mean()  # negate because reward
    legality_loss = -legality_reward(predicted_x0, cond).mean()

    return denoising_loss + alpha * hpwl_loss + beta * legality_loss
```

### 進階版本（如果最小版有效）

1. **Timestep weighting**：只在 t 較小時算 HPWL/legality
   ```python
   weight = alpha_t ** 2  # 早期 ≈ 0, 後期 ≈ 1
   hpwl_loss = weight * -hpwl_reward(predicted_x0, cond).mean()
   ```
2. **Legality 分離**：可能只加 HPWL loss 就夠了，legality 由 legalization 處理
3. **Curriculum**：先 pretrain 純 denoising，後期再加 HPWL/legality

---

## 成功/失敗判定

| 結果 | 意義 |
|------|------|
| Avg HPWL < 44.01（贏 Ablation 10k） | 方法有效，HPWL loss 提供了 supervised loss 以外的信號 |
| 44.01-44.65 之間 | 方法只略好於純 supervised，但不如 DDPO v2 |
| > 44.65（輸 DDPO v2） | 可能 HPWL loss 權重沒調好，或這個方向整體不適合 |
| Training 不收斂 / val loss 爆炸 | HPWL loss 權重太高，或者 gradient 干擾了 denoising |

---

## 預估資源

- **實作難度**：低（修改 `models.py` 的 `loss()` 函數，約 20 行）
- **訓練時間**：比純 supervised 慢 ~1.5-2×（多算 HPWL/legality）
- **VRAM**：+500 MB 以內
- **實驗時間**：training 25-30 分鐘 + eval 2.5 小時 = ~3 小時

---

## 結論

### 教授的建議 vs DDPO local reward 的關鍵差異

| 面向 | DDPO Local Reward | Loss + HPWL/Legality |
|------|------------------|---------------------|
| 階段 | Fine-tuning | **Fine-tuning**（推薦從 pretrained 或 ablation_10k 開始） |
| Gradient | Policy gradient | Direct gradient |
| 失敗案例（在我們實驗） | v2.3, v2.4, v2.5 | 尚未測試 |
| 有無文獻先例 | 有（DDPO 本身） | 有（ReFL, DRaFT；都是 fine-tune 設計） |

**這兩個不是同一件事。Local reward 失敗不代表 training loss 會失敗。**

### 我的可行性評估

**可行性高（推薦嘗試）**：
- ✅ 機制與 DDPO 完全不同（direct gradient）
- ✅ 有文獻先例（ReFL）
- ✅ 實作簡單（20 行 code）
- ✅ 風險可控（可以先用小 weight 測試）
- ✅ 與目前最佳方法（supervised）同方向，屬於強化而非轉向

**主要風險**：
- ⚠️ 可能因為冗餘而效果有限（但不會 harm）
- ⚠️ Loss weight 需要調校
- ⚠️ 早期 timestep 的 gradient 訊號意義不明（但可用 weighting 解決）

### 建議執行順序

1. **先做最小版本實驗（fine-tune）**：
   - Init: `../public-models/large-v2/large-v2.ckpt` 或 `ablation_supervised_10k.61/latest.ckpt`
   - 固定 alpha=1e-3, beta=1e-2，跑 5k-10k steps，看 ISPD eval
2. **若有小幅改善**：調 loss weight，試 timestep weighting
3. **若完全沒改善**：驗證 gradient 有沒有正常流過 predicted_x0（可能是 alpha_t 太小導致 gradient vanish）
4. **若 harm denoising**：降低 alpha/beta 或只對 late timestep 算
5. **最終對比**：把 init 從 `large-v2` 換成 `ablation_supervised_10k` 再跑一次，看是否能疊加改善

### 與 local reward 實驗的比較建議

建議在 report 中明確對比：
- DDPO Local Reward (v2.3/v2.4/v2.5)：policy gradient → 失敗
- Training Loss + HPWL (NEW)：direct gradient → 預期有效

這個對比本身就是很有 research value 的 finding——**不是 per-step supervision 不適用 diffusion，而是 policy gradient 不適用這個 task**。

---

## 下一步

如果同意這個分析，下一步是寫 `addloss_plan_1.md` 規劃具體實驗。
