# SVDD-PM 第一次實驗計畫（Inference-time Value-Based Decoding）

> 動機文件：`docs/meet/meet_0508.md`（教授建議「RL guided」）
> 方法選擇依據：`docs/survey/guidance_survey_1.md` Tier 1
> Parent paper: Uehara/Li/Zhao et al., arXiv:2408.08252 (2024)

## 目標

在 **inference 階段**用 SVDD-PM（Soft Value-Based Decoding，Posterior-Mean variant）取代 / 補強現有 `guidance@_global_=opt`，目標：

1. 在 ISPD2005 7-circuit average HPWL 上 **打贏目前最佳 Ablation 10k (44.01)**。
2. 在 **bigblue2 (23k macros)** 上跑出有意義的 HPWL（目前 V×V backward OOM，被迫關 guidance）。
3. 不修改 pretrained model weights — 純 inference-time intervention。

---

## 1. SVDD-PM 機制

### 1.1 數學形式

對每個 reverse step `t → t-1`：

1. 從 reverse kernel sample **K 個** 候選 `x_{t-1}^{(k)} ~ p_θ(x_{t-1} | x_t)`，k = 1..K
2. 對每個候選計算 **posterior-mean rollout reward**：
   ```
   \hat x_0^{(k)} = (x_{t-1}^{(k)} - σ_{t-1} · ε_θ(x_{t-1}^{(k)}, t-1)) / α_{t-1}
   r^{(k)}        = -(HPWL(\hat x_0^{(k)}) + λ · legality(\hat x_0^{(k)}))
   ```
3. Softmax over candidates 並 resample 1 個繼續：
   ```
   w^{(k)} = exp(r^{(k)} / α_temp) / Σ_j exp(r^{(j)} / α_temp)
   x_{t-1} = sample from {x_{t-1}^{(k)}} with prob w^{(k)}
   ```
4. 繼續下一個 reverse step。

關鍵：第 2 步只需要 **forward** evaluation；不需要對 reward 取 gradient。

### 1.2 與現有 `opt` guidance 的差別

| | `opt` (Universal+Lagrangian) | SVDD-PM |
|---|---|---|
| Per-step 計算 | 10 步 inner SGD on `\hat x_0`，需 ∇HPWL + ∇legality | K 次 forward (eps + reward) |
| Memory | V×V legality Jacobian buffer | K × V×V legality forward only |
| bigblue2 (V=23k) | OOM 必須跳過 guidance | K 倍 forward，可 tile 處理 |
| 對應「RL guided」 | 否（gradient-based） | **是**（value function look-ahead） |

### 1.3 SVDD-PM vs SVDD-MC 的選擇

| | SVDD-PM | SVDD-MC |
|---|---|---|
| Value 估計 | `r(\hat x_0)` 直接代 | 學一個小 value net |
| 額外訓練 | 無 | 需要 rollout regression |
| 第一次實驗適合度 | **適合**（minimum viable） | 後續升級 |

本次計畫只做 PM 變種。

---

## 2. 現有 codebase 接口分析

### 2.1 Hook point: `ContinuousDiffusionModel.reverse_samples()`

位於 `diffusion/models.py:1430-1509`。核心 inner loop（line 1455-1500）每個 step：

```python
for i, (t, t_minus_one) in enumerate(zip(timesteps[:-1], timesteps[1:])):
    eps_predict = self(x, cond, t_vec)                         # forward U-Net
    z = self._epsilon_dist.sample(...) if not last else 0
    x, x0_predicted = self._noise_scheduler.step(              # 取得 x_{t-1}
        eps_prediction=eps_predict, t=t, t_minus_one=t_minus_one,
        xt=x, z=z, mask=mask, guidance_fn=guidance_fn,
    )
```

SVDD-PM 要做的事：在第 i 步把 single sample 改成 K candidates → 評 reward → resample。

### 2.2 可重用元件

| 元件 | 位置 | 用途 |
|------|------|------|
| `HPWL` MessagePassing | `guidance.py:191` | Forward HPWL O(E)，可重用 |
| `compute_pin_map` | `guidance.py:269` | Pin map cache per netlist |
| `legality_guidance_potential_tiled` | `guidance.py:49` | Forward legality 可 tile（不需要 backward 也行） |
| `_noise_scheduler.step` / `alpha` / `sigma` | scheduler interface | 取得 mu, eta, predicted_x0 |

### 2.3 改動範圍

| 檔案 | 改動 |
|------|------|
| `diffusion/models.py` | 新增 `reverse_samples_svdd(B, x_in, cond, num_candidates=K, alpha_temp, lambda_legality, ...)` 方法 |
| `diffusion/configs/guidance/svdd.yaml` | 新 config: `guidance_mode: svdd`, `num_candidates`, `alpha_temp`, `lambda_legality`, `every_n_steps` |
| `diffusion/eval.py` | 走 `guidance_mode == "svdd"` 分支時呼叫新方法 |
| `diffusion/guidance.py` | 新增純 forward `legality_potential_forward_tiled`（複用既有 tiled，移除 backward） |

預估 ~250-400 行新 code。

---

## 3. 實驗設定

### 3.1 Phase 1 — 概念驗證（無 SVDD baseline）

| 項目 | 設定 |
|------|------|
| Method | `eval_svdd_pm_phase1` |
| Checkpoint | `../public-models/large-v2/large-v2.ckpt`（先從 paper 模型起，確認機制） |
| Task | `ispd2005-s0`, `macros_only=True` |
| Circuits | **adaptec1, bigblue1**（V=543, 560；最小最快） |
| K (num_candidates) | 4, 8, 16（sweep） |
| α_temp | 1.0（先用 sensible default） |
| λ_legality | 0.1, 1.0（sweep） |
| `every_n_steps` | 1（每步都 SVDD）vs 5（每 5 步） |
| guidance baseline | 同時跑 `guidance=none` 和 `guidance=opt` 比較 |
| seed | 300 |

判定：SVDD-PM 對 adaptec1 / bigblue1 HPWL **比 `guidance=none` 好** ⇒ 機制有效，進 Phase 2。

### 3.2 Phase 2 — 完整 ISPD2005 eval

| 項目 | 設定 |
|------|------|
| Method | `eval_svdd_pm_full` |
| Checkpoint | **`v1.61-ddpo.ablation_supervised_10k.61/latest.ckpt`**（從目前最佳 init） |
| Task | `ispd2005-s0`, all 8 circuits |
| K | Phase 1 最好的設定 |
| α_temp, λ_legality | Phase 1 最好的設定 |
| `every_n_steps` | Phase 1 最好的設定 |
| legalizer | `opt-adam`（保留現有 legalizer，SVDD 只取代 guidance） |
| **bigblue2** | 不需要 `skip_guidance_threshold` — SVDD 可以跑（驗證！） |
| seed | 300（與其他方法可比） |

判定基準（7-circuit avg HPWL，no bb2）：

| Avg HPWL | 意義 | 下一步 |
|----------|------|--------|
| **< 43.5** | SVDD 顯著超越 Ablation 10k | 寫 `svdd_report_1.md`；升級 SVDD-MC / TDS |
| 43.5 – 44.0 | SVDD 接近最佳但贏不大 | 試 K=32、其他 hyperparam sweep |
| 44.0 – 44.5 | 持平 Ablation 10k | 看 bigblue2 表現；如 bb2 改善大則仍有價值 |
| > 44.5 | SVDD 沒贏 supervised | 試 SVDD-MC 學 value net；或放棄 inference-time search 方向 |

額外觀察點：**bigblue2 HPWL**（baseline ~57-66，paper 38.8）— 如能跑下來 <50 就有實質貢獻。

### 3.3 預估資源

| 項目 | 估算 |
|------|------|
| Implementation | ~1-2 天 |
| Phase 1 eval (2 circuits × ~6 settings) | ~3 小時 |
| Phase 2 eval (8 circuits, K=8) | 推估 `K × 現有 eval 時間`，~8-15 小時 |
| 寫報告 | ~30 分鐘 |
| **總計** | **~3-4 天** |

### 3.4 Memory 估算

Phase 2 bigblue2 (V=23k) 上每 step 需要 K=8 個 forward eps_θ + K=8 個 forward HPWL + K=8 個 tiled forward legality：

- eps_θ forward (V=23k, batch=K=8): 估 ~3-5 GB
- HPWL forward × 8: ~負擔小（O(E)）
- Legality forward tile (V=23k, block=16k, K=8): peak per tile ~K × V × block_size × 2 floats = 8 × 23k × 16k × 8 byte = ~23 GB

**Memory 風險**：legality forward 即使無 backward，K=8 同時跑可能仍 OOM。

對策：
- 把 K candidates 跑成 **chunked 序列**（一次只算 K_chunk 個 forward），最差情況 K_chunk=1。
- 用更大 block_size tiling（trade compute for memory）。
- bigblue2 上 K 設小（K=2 or 4）。

---

## 4. 風險與對策

| 風險 | 說明 | 對策 |
|------|------|------|
| K=8 forward 在 bigblue2 仍 OOM | Legality V×V forward × K | Chunked candidates；K_chunk=1 fallback |
| Value 估計 `r(\hat x_0)` 在 early t noisy | 同 DDPO local reward 失敗的根本原因 | (a) `every_n_steps=5` 只在後段做 SVDD；(b) α_temp 大讓選擇平滑；(c) 早 t 自動降權 |
| K=8 對小 circuit 太重 | 每 step 8 倍 forward | Phase 1 sweep K 找出 sweet spot |
| Resample 破壞 diversity | 重採同一條 trajectory K 次 | Track effective sample size，可加 stratified resampling |
| α_temp 調不出來 | 太小 = best-of-N，太大 = 隨機 | Phase 1 預掃 α_temp ∈ {0.1, 1, 10} |
| 從 large-v2 vs ablation_10k init 不同行為 | Phase 1 只驗 large-v2 | Phase 2 用 ablation_10k 做主實驗，附 ablation 比較 |

---

## 5. 成功指標（再列一次）

| 指標 | 現況 | 目標 |
|------|------|------|
| 7-circuit avg HPWL | 44.01 (Ablation 10k) | **< 43.5** |
| bigblue2 HPWL | skip (~57-66 with no guidance) | < 50（理想 < 45） |
| 訓練成本 | 17 min (Ablation 10k 一次性) | 0 (純 inference) |
| Eval 成本 | ~3 小時 (no SVDD) | ~8-15 小時 (K=8) |

如果 SVDD 能在 bigblue2 上跑下來且贏 paper (38.8) 的非 bigblue2 circuit，**即使 7-circuit avg 持平 Ablation 10k 也是值得寫成 paper 章節**的成果。

---

## 6. 行動清單

- [ ] **Step 0a**：讀 SVDD repo `masa-ue/SVDD`，確認 PM variant 的 reference implementation
- [ ] **Step 0b**：寫 `legality_potential_forward_tiled`（拔掉 backward 的 forward-only 版本）
- [ ] **Step 1**：實作 `ContinuousDiffusionModel.reverse_samples_svdd(...)`
- [ ] **Step 2**：寫 `diffusion/configs/guidance/svdd.yaml`，串到 `eval.py` 分支
- [ ] **Step 3**：unit test on tiny synthetic（v1.61 1 sample, V~261）確認 SVDD pipeline 跑通
- [ ] **Step 4**：Phase 1 sweep on adaptec1 + bigblue1
- [ ] **Step 5**：Phase 2 full ISPD2005 eval（含 bigblue2）
- [ ] **Step 6**：寫 `docs/report/svdd_report_1.md`，對照 §3.2 判定基準
- [ ] **Step 7**：寫 `docs/next/svdd_next_1.md`，決定要不要升級 SVDD-MC / TDS / 放棄

---

## 7. 與其他方向的關係

- **教授建議 1（從頭 train）** — 獨立 track；SVDD 是 inference-only，不衝突，可以同時做。
- **DDPO local reward 失敗** — SVDD 是它的 gradient-free 替代版，重新驗證「per-step reward signal 對 chip placement 有用嗎」。
- **AddLoss 已耗盡** — SVDD 跟 AddLoss 機制完全不同（inference search vs training-time aux loss），不會有相同 ceiling。
- **Best-of-N 方向 (`docs/context.txt` direction A)** — CoDe 是它的 block 級變種；SVDD 是每 step 帶 value function 的版本。CoDe 可作為 SVDD 的下界 baseline。
