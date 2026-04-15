# DDPO 第四次實驗計畫

> 參考：`docs/next/ddpo_next_3.md`

## 目標

驗證把 **legality 加進 local reward** 是否能修正 DDPO v2.3（local reward only HPWL）失敗的設計。

## 動機

### 原先的設計缺陷

DDPO v2.3 的 local reward 只有 HPWL（`diffusion/ddpo.py:18`）：
```python
self.local_reward_fn = get_reward_fn(0.0, 1.0)  # HPWL only
```

當初的理由是 `legality_guidance_potential` 建立 V×V matrix 會 OOM。但重新計算：
- v1.61 training data 的 V ≈ 200-500
- `(B=2, V, V, D=2)` tensor 只 0.6-4 MB
- 加 autograd 中間值，每次 ~6-80 MB
- 每個 training step 算 10 次（`local_reward_every=5`，50 timesteps）
- 總加開銷 ~60-800 MB（24 GB 中）

**完全放得下。當初是懶人做法。**

### 為什麼 HPWL-only 可能更糟

Diffusion reverse sampling 的自然結構：
- **早期 timestep**（t 大）：predicted_x₀ 是散亂 noise → macros 分散 → HPWL 高（相連 macro 距離遠）
- **後期 timestep**（t 小）：predicted_x₀ 接近最終狀態 → HPWL 才有意義

如果 reward 只給 HPWL：
- 早期想降 HPWL → 把 macros 擠在一起 → legality 崩壞
- 沒有 legality 訊號反制，這個錯誤方向會持續

這可能是 v2.3 訓練出現 val_loss 尖峰（step 4400 飆到 13.7）的原因。

---

## 實驗設計

### Run 4：DDPO v2.4（Local Reward with Legality）

唯一改動：Local reward 從「只有 HPWL」改為「HPWL + 0.5×legality」（與 global reward 相同組成）。

| 項目 | v2.3 (舊) | **v2.4 (新)** |
|------|-----------|--------------|
| Global reward | HPWL + 0.5×legality | 同 |
| **Local reward** | **HPWL only** | **HPWL + 0.5×legality** |
| local_reward_every | 5 | 同 |
| local_reward_weight | 0.5 | 同 |
| 其餘（batch, lr, clip, supervised_weight 等） | 同 DDPO v2 | 同 |

### 程式碼變更

`diffusion/ddpo.py:18`:
```python
# 舊
self.local_reward_fn = get_reward_fn(0.0, 1.0)
# 新
self.local_reward_fn = reward_fn  # 用跟 global 相同的 reward
```

或者保留 flexibility，用新的 config 參數控制：
```python
self.local_reward_fn = get_reward_fn(local_legality_weight, local_hpwl_weight)
```

目前先選最簡單的——直接重用 global reward_fn，沒新 config。

---

## 比較對象

| 方法 | Avg HPWL (7 circuit) | Local Reward 內容 |
|------|---------------------|-------------------|
| DDPO v2 | 44.65 | 無 local reward |
| DDPO v2.3 | 45.80 | HPWL only |
| **DDPO v2.4** | ? | **HPWL + legality** |

如果 v2.4 < 44.65，證明 local reward 的想法本身可行，之前失敗只是因為沒給 legality。

如果 v2.4 ≥ 45.80，證明問題不在 legality，local reward 這個 design 根本不適合這個場景。

---

## 執行設定

| 項目 | 值 |
|------|---|
| method | ddpo_v2_4_local_legality |
| batch_size | 2 |
| num_timesteps | 50 |
| lr | 1e-5 |
| train_steps | 5000 |
| clip_epsilon | 0.2 |
| supervised_weight | 1.0 |
| legality_weight | 0.5（global & local 都是） |
| hpwl_weight | 1.0 |
| local_reward_weight | 0.5 |
| local_reward_every | 5 |
| from_checkpoint | `../public-models/large-v2/large-v2.ckpt` |
| seed | 61（訓練），300（eval） |

**預估時間**：訓練 ~108 分鐘 + eval ~2.5 小時 ≈ 4 小時

---

## VRAM 監控

訓練開始後 `nvidia-smi` 確認 VRAM < 24 GB。如果 OOM，備案：
1. 增大 `local_reward_every`（例如 10，算 5 次而非 10 次）
2. 用 `legality_guidance_potential_tiled` 取代原版

---

## 成功指標

| 指標 | 目標 |
|------|------|
| 訓練穩定性 | 無 val_loss 尖峰（v2.3 曾飆到 13.7） |
| Avg HPWL (7 circuit) | < 44.65（至少追平 DDPO v2） |
| Legality 全面保持 | 所有 circuit > 0.99 |

---

## 實驗後分析重點

1. Local reward 加 legality 後，訓練曲線是否更穩？
2. 比 DDPO v2（無 local reward）好還是壞？
3. 跟 supervised 10k 的 44.01 相比如何？
4. 如果 v2.4 仍不如 supervised 10k，那 local reward 這個方向應該放棄
