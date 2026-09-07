# CoDe（blockwise best-of-N）第一次實驗計畫

> 動機：`docs/survey/guidance_survey_1.md` §1.2（Tier 1.2，CoDe arXiv:2502.00968, 2025）
> 前置結果：SVDD Phase 5 (`docs/report/svdd_report_5.md`) 證實 inference-time search 在 paper checkpoint 上贏 paper −4.4%
> 寫作日期：2026-05-24

## 1. 目標

回答兩個問題：
1. **CoDe（hard best-of-N at block boundary）能不能贏 paper baseline (46.89)？**
2. **CoDe 跟 SVDD-PM (soft-weighted resample at every step) 比起來誰好？**

問題 1 確認「inference-time search 對 chip placement 有效」這個 hypothesis 的最簡 form。
問題 2 拆解 SVDD-PM 的 -4.4% 增量 = (a) inference-time search 本身的貢獻 + (b) soft-resample / 每步做的 incremental 貢獻。

→ 是 SVDD 論文敘事的 **必要 ablation**。

## 2. CoDe 機制

直接從 survey §1.2 抄：
- 每 reverse 跑 `block_size` 步（block，無 SVDD-like multi-particle）
- 到 block boundary 時 fork **N 個** 候選 `x_{t-1}^{(k)}`（用 K 個不同 z）
- 對每個候選計算 Tweedie posterior mean `\hat x_0^{(k)}` → reward = `-(HPWL + λ·legality)`
- **Hard argmax** 選 reward 最高的繼續（跟 SVDD-PM 的 softmax-resample 差別在這）
- 進下一個 block

vs SVDD-PM 的差別：
| | SVDD-PM (Phase 5 setup) | CoDe (this plan) |
|---|---|---|
| Resample 頻率 | every reverse step (every_n=1) | every block of B steps |
| Selection | soft (multinomial of softmax) | hard (argmax) |
| 探索 vs 最佳化 | 平衡（α_temp=1） | 偏最佳化（argmax）|
| Cost | K = 4 forward per step × 1000 steps = 4000 extra | K = 4 forward per block × 10 blocks = 40 extra |
| Layered on opt？ | 是 | 是（同 svdd_layered） |

## 3. 實作（待做）

### 3.1 程式碼改動

| 檔案 | 改動 |
|------|------|
| `diffusion/models.py` `ContinuousDiffusionModel.__init__` | 加 CoDe kwargs：`code_num_candidates=4`, `code_block_size=100`, `code_lambda_legality=1.0`, `code_layer_opt=False` |
| `diffusion/models.py` `reverse_samples` | 加 dispatch：`if self.guidance_mode == "code"` → `_reverse_samples_code` |
| `diffusion/models.py` 新增 `_reverse_samples_code` | 結構 mirror `_reverse_samples_svdd`，差別是 (1) `i % block_size == 0` 才 do_code (2) hard argmax，不 multinomial |
| `diffusion/configs/guidance/code_layered.yaml` | 新 config：merge opt params + CoDe params + `guidance_mode: code` + `code_layer_opt: True` |

### 3.2 預設超參

| 參數 | 預設 | 理由 |
|---|---|---|
| `code_num_candidates` (K) | 4 | 對齊 SVDD K=4 |
| `code_block_size` (B) | 100 | 1000 reverse steps / 10 blocks（survey 提到的 default range）|
| `code_lambda_legality` (λ) | 1.0 | 對齊 SVDD λ=1 |
| `code_layer_opt` | True | 對齊 svdd_layered |

## 4. 實驗矩陣

| Run | guidance | num_output_samples | seed | scope |
|-----|----------|---|---|---|
| **Run I-1** | `code_layered` | 7 (skip bb2) | 300 | seed 1 |
| **Run I-2** | `code_layered` | 7 (skip bb2) | 301 | seed 2 |
| **Run I-3** | `code_layered` | 7 (skip bb2) | 302 | seed 3 |

不做 `code-only`（沒 layer opt 的版本）── Phase 4 Run G (svdd only on large-v2) 已知顯著輸 (66.53)，CoDe 純版預期類似輸，不浪費 GPU。

不做 K / block_size sweep ── 先看 default 設定的 result，輸/贏再說。

從 svdd_next_3 §8 提取：**所有主結論 multi-seed (3 seeds)**。

## 5. 預估成本

每 run 對齊 SVDD Phase 5 estimates（large-v2 + svdd_layered ~70 min for 7 circuits）：
- CoDe 比 SVDD layered 計算少很多（K=40 vs K=4000 extra forwards across 1000 steps）
- **but** opt guidance 本身的 cost 沒變（grad_descent_steps=20 per step）
- 預期 CoDe ≈ SVDD layered 在 wall-clock 上（~60-70 min per seed）

| Run | 估時 |
|-----|------|
| Run I-1, I-2, I-3 (sequential, GPU 1) | 3 × ~65 min = **~3.5 hr** |

## 6. 判定基準

主指標：**7-circuit avg HPWL (no bb2), 3-seed mean ± std**

| 結果 | 意義 | 行動 |
|------|------|------|
| **CoDe mean < 44.5** | CoDe **贏 paper 且贏 SVDD**：hard argmax 比 soft-resample 強 | 改 default 用 CoDe；plan_2 試 K=8、block size sweep |
| **44.5 ≤ CoDe mean ≤ 45.5** | CoDe ≈ SVDD：**核心是 inference-time search，soft/hard 不重要** | paper 章節主推 CoDe（更簡單），SVDD 列 ablation |
| **45.5 < CoDe mean ≤ 46.5** | CoDe 微贏 paper 但輸 SVDD：soft-resample 是 SVDD 的關鍵 | 確認 SVDD 是 main method；CoDe 列為 baseline |
| **CoDe mean > 46.5** | CoDe 輸 paper：hard argmax 太貪婪 / inference-time search 不夠 | inference-time search 機制不夠普遍，**只有 SVDD soft-resample 那條 path 有用** |

額外觀察：
- **bigblue3 variance**：SVDD Phase 5 bigblue3 std = 5.34，CoDe argmax 應該 variance 較小（每 block 都選最佳）。看 std 變化能驗證 soft vs hard 的探索取捨
- **bigblue4 (-10.8% 最大贏家 in SVDD)**：CoDe 在 bb4 還能保持嗎？bb4 generation time 是瓶頸

## 7. 與 SVDD Phase 5 的對照表（填空待跑）

| Setup | 7-circuit avg HPWL | vs Paper 46.89 |
|---|---:|---:|
| Paper baseline (公告 seed=400) | 46.89 | — |
| Our reproduction (large-v2 + opt, seed=300) | 48.69 | +3.8% |
| **SVDD layered (3-seed mean)** | **44.85 ± 0.65** | **−4.36%** |
| **CoDe layered (3-seed mean)** | **TBD** | **TBD** |

## 8. 風險與對策

| 風險 | 對策 |
|------|------|
| `block_size=100` 太大，CoDe argmax 落到局部 | 預設 100；report 若顯著輸 SVDD 再 sweep block_size ∈ {50, 100, 200} |
| GPU 0 contention 再次觸發 cudaErrorIllegalAddress | 預設 `CUDA_VISIBLE_DEVICES=1` |
| CoDe argmax 變得太貪婪導致 mode collapse / legality drop | 觀察 legality；若 < 0.95 加 `code_lambda_legality=2-5` |
| bb4 (V=8170) memory OOM with K=4 batched forward | K=4 在 SVDD layered 上已驗證可跑（Phase 4 part2 OK），CoDe 同 K，無 OOM 風險 |
| 結果與 SVDD 不可區分（noise 內） | std 0.65 = 噪音地板；CoDe 若落在 44.85±1.0 範圍，標 "持平"，不再 sweep |

## 9. 行動清單

- [ ] **Step 1**：實作 `_reverse_samples_code` + `code_num_candidates/block_size/lambda_legality/layer_opt` kwargs
- [ ] **Step 2**：寫 `configs/guidance/code_layered.yaml`
- [ ] **Step 3**：adaptec1 smoke test（no legalization, 1 sample, 確認跑得通）
- [ ] **Step 4**：Run I-1 (seed=300) + I-2 (seed=301) + I-3 (seed=302) 串行 background
- [ ] **Step 5**：extract metrics, 寫 `docs/report/code_report_1.md`
- [ ] **Step 6**：寫 `docs/next/code_next_1.md` 決定 plan_2 還是換方法
