# SVDD-PM 第一次實驗 Report (Phase 1)

> Plan: `docs/plan/svdd_plan_1.md`
> Survey: `docs/survey/guidance_survey_1.md`
> 動機 meeting: `docs/meet/meet_0508.md`（教授建議：denoising 過程嘗試其他 guidance，如 RL guided）
> 執行日期：2026-05-19
> Checkpoint：`../public-models/large-v2/large-v2.ckpt`（paper baseline）

---

## 1. 結論摘要

| 結論 | 判定 |
|------|------|
| **SVDD-PM 機制實作完成、能正常跑通**（K=4 default） | ✅ |
| **default 設定（K=4, every_n=5, λ=1）能贏 raw baseline** | △ 邊際（bigblue1 -3.0%, adaptec1 -0.7%） |
| **aggressive 設定（every_n=1）大幅降 HPWL（-16.5%）** | ✅ 但 legality 退步（0.69→0.61） |
| **依 plan_1 §3.2 判定基準（7-circuit avg < 43.5）** | ❌ 未達到 |
| **應該升級 SVDD-MC 或 TDS** | 暫不（Phase 1 還沒和 legalization 公平比較） |
| **Phase 2 必跑：加 legalization、完整 ISPD2005** | ✅ 必要 |

**核心觀察**：raw 模式（無 legalization）下 SVDD 帶來小幅改善但落在 seed 噪音範圍內。`every_n=1`（每 reverse step 都 SVDD）給出明顯 HPWL 降幅但犧牲 legality — 暗示 **SVDD 跟 legalizer 是互補關係**，要 Phase 2 加 legalizer 才能驗證最終 deployment 表現。

---

## 2. 實作

### 2.1 主要改動

| 檔案 | 改動 |
|------|------|
| `diffusion/models.py` | `ContinuousDiffusionModel.__init__` 新增 5 個 SVDD kwargs；`reverse_samples` 加 svdd 分流；新增 `_reverse_samples_svdd()` 方法 (~85 行) |
| `diffusion/configs/guidance/svdd.yaml` | 新 config: `guidance_mode: svdd` + SVDD 超參 |

### 2.2 SVDD-PM 機制（已 implement）

每個 reverse step（subject to `every_n_steps` 和 `start_step_frac`）：
1. 從 reverse kernel sample **K 個** 候選 `x_{t-1}^{(k)} = μ + η·z_k`（K 個不同 z）
2. 對每個候選做 Tweedie posterior-mean: `\hat x_0^{(k)} = (x_{t-1}^{(k)} - σ_{t-1}·ε_θ(x_{t-1}^{(k)}, t-1)) / α_{t-1}` （**K 次 ε_θ forward**）
3. Reward: `r^{(k)} = -(HPWL(\hat x_0^{(k)}) + λ·legality(\hat x_0^{(k)}))`
4. Softmax `w^{(k)} = exp(r^{(k)} / α_temp) / Σ` → resample 1 個繼續

關鍵：**全 forward，零 gradient backprop**，符合 plan §1.2 對應教授「RL guided」的設計。

### 2.3 程式碼小坑（在執行過程中發現並修）

- **NaN in softmax** when reward 數量級大（lambda=5、every_n=1 case）：修正
  ```python
  reward = torch.where(torch.isfinite(reward), reward, torch.full_like(reward, -1e9))
  reward = reward - reward.max(dim=0, keepdim=True).values.detach()
  weights = torch.softmax(reward / max(alpha_temp, 1e-6), dim=0)
  weights = torch.nan_to_num(weights, nan=1.0/K, posinf=1.0, neginf=0.0)
  weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-12)
  ```
- **K=8 PyG batched edge_index 崩潰** (`RuntimeError: Expected index [33312] to be smaller than self [4480]`)。是 `gatv2_conv.add_self_loops` + 自訂 wrapper.py 的 unbatch/rebatch 邏輯與 K*B=8 不相容。**Phase 1 暫時用 K=4 規避**，留 Phase 2 修。

---

## 3. 實驗設定

**通用設定**：
- task=ispd2005-s0, macros_only=True, num_output_samples=1, **legalizer=none**（Phase 1 純機制驗證）
- model: `large-v2.ckpt`（paper baseline，未經 fine-tuning）
- 量測：`macro_hpwl_rescaled / 100`（= paper 單位 HPWL ×10⁵），`macro_legality`

**Phase 1 scope**：adaptec1 (V=543), bigblue1 (V=560)。小 circuit 為主，無 legalization 加速 iteration。

---

## 4. 結果

### 4.1 adaptec1 — multi-seed 對照

| Config | seed=300 | seed=301 | seed=302 | mean | legality (mean) |
|---|---:|---:|---:|---:|---:|
| baseline `none` | 10.33 | 9.12 | 9.24 | **9.56** | 0.653 |
| SVDD K=4 λ=1 (every_n=5) | 9.89 | 9.20 | 9.39 | **9.49** | 0.649 |
| Δ vs baseline | -4.3% | +0.9% | +1.6% | **-0.7%** | -0.6% |

**結論**：3-seed mean 差距 -0.7%（單一 seed std ~0.6 HPWL），**SVDD 默認設定在 adaptec1 上落在 seed 噪音內**。

### 4.2 adaptec1 — 超參 sweep（seed=300）

| Config | HPWL | legality | 註解 |
|---|---:|---:|---|
| baseline `none` | 10.33 | 0.690 | — |
| SVDD K=4 λ=0 (HPWL only) | 10.43 | 0.664 | 沒幫助；legality 雖無 reward 但跟著掉 |
| SVDD K=4 λ=1 | 9.89 | 0.656 | 微好 |
| SVDD K=8 λ=0 | 10.42 | 0.658 | K=4→K=8 無增益 |
| **SVDD K=4 λ=1 every_n=1** | **8.63** | 0.611 | **HPWL -16.5%**；legality 大退（-11%） |
| SVDD K=4 λ=1 start_step_frac=0.3 | 10.70 | 0.698 | 後 30% 才做 → 沒幫助 |
| SVDD K=4 λ=5 | crash | — | `cudaErrorIllegalAddress`（CUDA 殘留 corruption） |

**Insight**：`every_n=1`（每步 SVDD）給出最強 HPWL signal，但 legality 從 0.69 掉到 0.61。看起來 SVDD 找到 HPWL 更低的 placement，但犧牲 macro 不重疊性 → 預期 legalization 階段能補回 legality 但 HPWL 會被推高。Phase 2 必加 legalization 驗證淨效果。

### 4.3 bigblue1 — multi-seed 對照

| Config | seed=300 | seed=301 | mean | legality (mean) |
|---|---:|---:|---:|---:|
| baseline `none` | 9.61 | 9.34 | **9.48** | 0.827 |
| SVDD K=4 λ=1 (every_n=5) | 9.51 | 8.88 | **9.20** | 0.820 |
| Δ vs baseline | -1.0% | -4.9% | **-3.0%** | -0.8% |

**結論**：bigblue1 上 SVDD default 給 **-3.0% HPWL** 改善（2 seed 平均），legality 幾乎沒退。比 adaptec1 訊號清楚。

### 4.4 bigblue1 — single-seed sweep

| Config | HPWL | legality |
|---|---:|---:|
| baseline `none` | 9.61 | 0.838 |
| SVDD K=4 λ=0 | 9.43 | 0.831 |
| SVDD K=4 λ=1 | 9.51 | 0.830 |
| SVDD K=8 λ=1 | crash | — |

K=4 λ=0 與 λ=1 接近，legality reward 看不到明顯區分。

---

## 5. 對照 plan_1 §3.2 的判定基準

| 結果情境 | 判定 | 對應行動 | 是否觸發 |
|----------|------|---------|---------|
| 7-circuit avg HPWL < 43.5 | 顯著贏 | 升級 SVDD-MC/TDS | ❌ 未達（Phase 1 只跑 2 circuit, 無 legalization） |
| 43.5–44.0 | 接近 | sweep K=32 | ❌ 未達 |
| 44.0–44.5 | 持平 Ablation 10k | 看 bigblue2 | ❌ 未達 |
| > 44.5 | SVDD 沒贏 supervised | 試 SVDD-MC 或放棄 | ❌ 未達 |

**Phase 1 結果不適用 §3.2 完整判定基準**（因為沒跑 7-circuit + legalization）。但 raw 模式單 circuit 訊號夠強到值得進 Phase 2：bigblue1 -3.0% mean improvement、aggressive every_n=1 在 HPWL 上有 -16.5% gain。

---

## 6. 與其他方法的對照（raw HPWL, large-v2 init）

> 注意：以下都是 **無 legalization** 的 raw HPWL，不能直接比 CLAUDE.md 那張表（deployment 數字）。

| 方法 | adaptec1 raw HPWL | bigblue1 raw HPWL |
|------|---|---|
| baseline (guidance=none) | 9.56 (3 seeds) | 9.48 (2 seeds) |
| **SVDD-PM K4L1 every_n=5** (this) | 9.49 (3 seeds, -0.7%) | 9.20 (2 seeds, -3.0%) |
| **SVDD-PM K4L1 every_n=1** (this) | 8.63 (1 seed, -9.7%) | — |

對 Ablation 10k (44.01) 的 7-circuit avg：本次實驗無 legalization、circuit 數不足，**無法直接比較**。

---

## 7. 觀察與假說

### 7.1 為什麼 adaptec1 沒贏、bigblue1 贏？

| 假說 | 證據 | 可信度 |
|------|------|--------|
| adaptec1 baseline 已接近 raw 最佳，improvement headroom 小 | seed=301 baseline 已是 9.12，paper 是 9.19 | 中 |
| bigblue1 macro 多 1.03×，placement landscape 更複雜，K-particle search 有用 | bigblue1 baseline std 也大（9.34→9.61, 2.9% 差距） | 中 |
| 單 seed 差距 (-4.3% / +0.9% / +1.6%) 是純隨機噪音 | seed std ≈ 0.6 HPWL，差距落在 1σ 內 | 高 |

### 7.2 為什麼 every_n=1 HPWL 大降但 legality 退？

SVDD 每步都 resample → 強化 reward signal（HPWL）但 reward 沒包含 legality 足夠權重（λ=1）。Macro 互相擠近一起讓 wire 短，但 overlap 變嚴重。**Phase 2 應該試更高 λ_legality**（fixed 後）+ legalization。

### 7.3 為什麼 K=8 PyG crash？

`networks/layers/wrapper.py:31` 的 unbatch/rebatch 對 batch_size 有結構假設（看 edge_index 重複次數和節點數推 batch_size）。K*B=8 與某些 circuit 的 edge 結構衝突 → `add_self_loops` 邊界錯。屬於現有 codebase 對 batched inference 的限制，不是 SVDD 邏輯本身的 bug。Phase 2 可以 (a) 修 wrapper.py 處理 任意 batch_size, 或 (b) 把 K candidates 分成 K_chunk 序列跑（K_chunk=4 安全）。

---

## 8. 下一步（Phase 2 設定建議）

依照 plan_1 §3.2，Phase 2 應該：

1. **加 legalization**（`legalizer@_global_=opt-adam`）做 deployment-fair 比較
2. **跑完整 7 circuits**（skip bigblue2，因為 legality 仍 V×V forward OOM；需 `legality_potential_forward_tiled`）
3. **從 `ablation_supervised_10k.61/latest.ckpt` 起**（plan_1 §3.2 原本就規劃，目前最佳 init）
4. **多 seed 重跑 default config**（K=4, λ=1, every_n=5）做完整 7-circuit mean
5. **加一個 aggressive run**（K=4, λ=1, every_n=1）看 every_n=1 + legalization 的淨效果
6. **K=8 PyG fix**：實作 K_chunk 序列化或修 wrapper.py
7. **lambda_legality sweep**：every_n=1 case 試 λ=2, 5, 10 補回 legality
8. **bigblue2 special case**：先實作 `legality_potential_forward_tiled` 再跑

**判定下次寫 `svdd_report_2.md` 的基準**（沿用 plan_1 §3.2 但用 ablation_10k init）：

| 7-circuit avg HPWL (legalized, no bb2) | 意義 | 下一步 |
|---|---|---|
| < 43.5 | SVDD 顯著贏 | 升級 SVDD-MC / TDS / 寫 paper |
| 43.5 – 44.0 | 微贏 | sweep K, λ, every_n |
| 44.0 – 44.5 | 持平 ablation_10k | 看 bigblue2 |
| > 44.5 | SVDD 沒贏 | 放棄 inference-time search 方向 |

---

## 9. 修正後寫進 `docs/next/svdd_next_1.md` 的重點

- SVDD-PM 機制本身 OK，要 Phase 2 加 legalization 才能下結論
- `every_n=1` 是最值得追的 hyperparam（強 HPWL signal）
- `λ_legality` 在 every_n=1 case 需要重 tune
- K=8 batched inference 是 codebase-wide bug，不只 SVDD 的事
- 不要直接跳到 SVDD-MC（先把 SVDD-PM 在 fair 條件下證明有效）
- 不要為了 bigblue2 馬上實作 cluster guidance（先把可 fit 24GB 的 7 circuit 做完）
