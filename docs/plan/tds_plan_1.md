# TDS（Twisted Diffusion Sampler）第一次實驗計畫

> 動機：`docs/survey/guidance_survey_1.md` §1.3（Tier 1.3, TDS arXiv:2306.17775 NeurIPS 2023）
> 前置結果：
>   - SVDD `svdd_report_5.md` 3-seed mean 44.85 ± 0.65（贏 paper -4.4%）
>   - CoDe `code_report_1.md` 3-seed mean 45.22 ± 0.47（贏 paper -3.6%；跟 SVDD 無顯著差）
> 使用者選擇（2026-05-25）：跳過 CoDe plan_2，直接試 TDS

## 1. 目標

回答一個問題：**「importance weighting + ESS-based resampling（TDS 的特徵）能否打贏 SVDD 的 soft-resample 或 CoDe 的 hard-argmax？」**

- 若 TDS << SVDD（< 44.5）→ SMC 的 principled importance weighting 真的有用
- 若 TDS ≈ SVDD/CoDe → inference-time K-particle search 是 ceiling，weighting 細節不重要
- 若 TDS > SVDD/CoDe → 在 chip placement 上 SMC 的「不馬上 resample」反而破壞探索

→ 是 SVDD/CoDe 之外**機制最不同的第三條路徑**（particle filter 而非 step-wise resample）。

## 2. TDS 機制（vs SVDD vs CoDe）

| | SVDD-PM | CoDe | **TDS** |
|---|---|---|---|
| 結構 | K 個 candidate per step → softmax → resample 1 → collapse | K candidate every B steps → argmax → resample 1 → collapse | **N 個 particle 各自跑 trajectory，靠 importance weight 加權** |
| Selection | 每步 soft sample | 每 B 步 hard argmax | **ESS 低於閾值才 resample；平時都保留 N 個 particle** |
| Theoretical | heuristic | heuristic | **asymptotic exact**（給足 N 個 particle 收斂到 reward-tilted 真實分佈） |
| Diversity | 弱（每步 collapse） | 中（block 內 collapsed） | **強（particle 不會 collapse 除非 ESS 低）** |
| Cost per step | ~2 forwards (batch B + K) | ~1 forward (batch B) + K extra at block boundary | **~2 forwards (both batch N)** |

### 2.1 TDS 演算法

對每個 reverse step (t → t-1)：
1. **Standard reverse**：對每個 particle k，算 `eps_θ(x_t^{(k)})` → `predicted_x0` → 加 opt guidance → `mu_k` → 採樣 `x_{t-1}^{(k)} = mu_k + η·z_k`
2. **Value evaluation**：對每個新 `x_{t-1}^{(k)}` 再算 `eps_θ(x_{t-1}^{(k)}, t-1)` → Tweedie `\hat x_0` → `value_t^{(k)} = -(HPWL + λ·legality)(\hat x_0)`
3. **Importance weight update**：`log w^{(k)} += (value_t - value_{t-1}) / α_temp`（reward 改善量當權重增量）
4. **ESS check**：`ESS = 1 / Σ w_normalized²`。若 `ESS < N·ess_threshold_frac` → multinomial resample N particles by weights, 重置 `log_w = 0`
5. 最後一步不算 value（t≈0 model 不穩）

最終輸出：**all N particles 中 final reward 最高者**（best-of-N at end）。

### 2.2 預設超參（對齊 SVDD/CoDe 為公平 ablation）

| 參數 | 預設 | 理由 |
|---|---|---|
| `tds_num_particles` (N) | 4 | 對齊 SVDD K=4 / CoDe K=4 |
| `tds_alpha_temp` (α) | 1.0 | 對齊 SVDD |
| `tds_lambda_legality` (λ) | 1.0 | 對齊 SVDD |
| `tds_ess_threshold_frac` | 0.5 | 標準 SMC default（ESS < N/2 觸發 resample）|
| `tds_layer_opt` | True | 對齊 svdd_layered / code_layered |

## 3. 實作

### 3.1 程式碼改動

| 檔案 | 改動 |
|---|---|
| `diffusion/models.py` `__init__` | 加 5 個 TDS kwargs |
| `diffusion/models.py` `reverse_samples` | 加 dispatch：`if guidance_mode == "tds"` → `_reverse_samples_tds` |
| `diffusion/models.py` 新增 `_reverse_samples_tds` | N particles + delta-based log-weight + ESS resample + best-of-N output |
| `diffusion/configs/guidance/tds_layered.yaml` | merge opt params + TDS params + `guidance_mode: tds` |

### 3.2 注意事項

- B=1（eval 一次跑 1 個 sample）→ 內部 expand 到 N particles，最後 collapse 回 B=1
- `prev_value` 需要 reindex 當 resample 發生
- 數值穩定：log_w max-subtract before softmax，nan_to_num for delta

## 4. 實驗矩陣

| Run | guidance | seed | scope |
|-----|----------|---|---|
| Run J-1 | `tds_layered` | 300 | 7 circuits (skip bb2) |
| Run J-2 | `tds_layered` | 301 | 同 |
| Run J-3 | `tds_layered` | 302 | 同 |

跟 SVDD Phase 5、CoDe Phase 1 完全同 setup（large-v2 + opt-adam legalizer + skip_guidance_threshold=10000）。

## 5. 預估成本

每 seed：2 個 eps_θ forward (batch=N=4) per step × 1000 steps + opt 20-SGD per step

| Run | 估時 |
|-----|------|
| Run J-1, J-2, J-3 (sequential, GPU 1) | 預估每 seed ~140-160 min（比 SVDD 117 min 稍貴；vs CoDe 98 min 偏貴）|
| **總計** | **~7-8 hr** |

## 6. 判定基準

主指標：**7-circuit avg HPWL (no bb2), 3-seed mean ± std**

| 結果 | 意義 | 行動 |
|------|------|------|
| **TDS < 44.5** | TDS 顯著贏 SVDD → SMC importance weighting 真的有用 | TDS 變新 main method；plan_2 sweep N, α, ess_threshold |
| **44.5 ≤ TDS ≤ 45.5** | TDS ≈ SVDD ≈ CoDe → 三族 inference-time search 都 saturate 在同一個 ceiling | paper 用 CoDe 主推（最簡單），SVDD/TDS 列 ablation；下一步 FreeDoM 或從頭 train |
| **45.5 < TDS ≤ 46.5** | TDS 微贏 paper 但輸 SVDD/CoDe | 可能 ESS 不太對；try ess_threshold_frac sweep |
| **TDS > 46.5** | TDS 沒贏 paper | particle filter 不適合，**inference-time search 三族裡 SMC 是輸的那個** |

額外觀察點：
- **bigblue4 表現**：SVDD 125.48 vs CoDe 129.89（SVDD 贏 4.42 unit）。TDS 的 particle diversity 應該幫到 bb4，預期接近 SVDD 或更好
- **bigblue3 variance**：SVDD std 5.34, CoDe std 3.22。TDS 預期 std 介於兩者間
- **ESS 觸發頻率**：若 ess 幾乎從不觸發 → particles 太相似（探索不夠）；若 ess 一直觸發 → 等同 every-step resample（變 SVDD）

## 7. 風險與對策

| 風險 | 對策 |
|------|------|
| TDS 比預期慢很多（2× per-step forward） | 預估 150 min/seed 已含 buffer；若超過 250 min 中斷 |
| ESS 從不觸發（particle 都同分數）→ 退化成沒 selection 的 N parallel runs | 觀察 final 4 個 particle reward 是否 cluster；如是，降 ess_threshold_frac 或加大 α_temp |
| 數值不穩（log_w 累積到極值） | max-subtract + nan_to_num（同 SVDD 處理）|
| Best-of-N at end 是不是好 output 策略 | 也可以試 weighted average，但需小心 placement 不能 linear interpolate；先 best-of-N |
| GPU 0 contention | `CUDA_VISIBLE_DEVICES=1`（sustained from SVDD/CoDe） |

## 8. 行動清單

- [ ] Step 1：實作 `_reverse_samples_tds` + TDS kwargs
- [ ] Step 2：寫 `configs/guidance/tds_layered.yaml`
- [ ] Step 3：adaptec1 smoke test（no legalization）
- [ ] Step 4：3-seed × 7 circuits run (background, GPU 1)
- [ ] Step 5：extract metrics + 寫 `docs/report/tds_report_1.md`
- [ ] Step 6：寫 `docs/next/tds_next_1.md` 決定 plan_2 還是換方法
