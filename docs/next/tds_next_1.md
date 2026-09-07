# TDS Phase 1 回顧（next_1）

> Report: `docs/report/tds_report_1.md`
> Plan: `docs/plan/tds_plan_1.md`
> 寫作日期：2026-05-25

## 1. 一句話總結

**TDS 45.08 ± 0.38 ≈ SVDD 44.85 ± 0.65 ≈ CoDe 45.22 ± 0.47**，三者統計上等價，都贏 paper -3.6~-4.4%。**TDS 的 unique value 是「穩定」**：std 比 SVDD 低 -42%，bigblue3 std 從 5.34 降到 0.16（−97%），adaptec2 從 SVDD/CoDe 都輸 paper 拉回到 tied paper。**inference-time K-particle search 是 universal 的核心**，三族都 saturate 在同一個 ceiling，下一步應該換軌道（FreeDoM 嘗試 layer / 從頭 train / 寫 paper）。

## 2. Plan_1 vs 實際

| Plan_1 預期 | 實際 |
|---|---|
| TDS 可能落在 44.5-45.5「saturate」區 | ✅ 命中（45.08）|
| TDS std 介於 SVDD/CoDe 之間 | ❌ TDS std=0.38 **比兩者都低**（SVDD 0.65 / CoDe 0.47）|
| bigblue4 應該接近 SVDD（particle diversity 幫到大 circuit） | ❌ TDS 129.67 ≈ CoDe 129.89，**輸 SVDD 4.20**。particle 多樣性沒幫到 bigblue4 |
| Cost ~150 min/seed | ✅ ~120 min/seed（甚至接近 SVDD 的 117） |
| ESS resample 會時不時觸發 | ✅ bigblue3 std=0.16 證明 resample 是 active 的（particles 被頻繁拉回到「集體共識」） |
| TDS 預期 ~ SVDD 或微差 | ✅ +0.23 vs SVDD（噪音內） |

**最大意外**：
- TDS std **超低**（0.38 vs SVDD 0.65, CoDe 0.47），不在預期範圍。SMC particle filter 在 chip placement 這個 task 上的「穩定性收益」比預期大。
- bigblue4 TDS 跟 CoDe 一樣輸 SVDD 4.20。原本預期 particle diversity 有幫助，結果反向。

## 3. 學到的事

### 3.1 ESS-based resampling 真的有 stabilize 作用

bigblue3 std：SVDD 5.34 → CoDe 3.22 → **TDS 0.16**（壓倒性）。

機制解釋：
- SVDD 每步 soft resample → trajectory 高度依賴採樣噪音 → 不同 seed 跑出差很多
- CoDe 每 100 步 hard argmax → 部分 collapse 但 block 內仍有隨機性
- TDS ESS-based resample → 只在 particle 分歧太大時 resample，集體向 high-reward 區域對齊 → 不同 seed 殊途同歸

對 paper：可以用「TDS 提供 95% CI 比 SVDD 窄 40%」當賣點

### 3.2 「Particle diversity」不總是好事

bigblue4 SVDD 贏 TDS 4.20。
- 預期：N=4 particles 平行探索 → 應該找到更好解
- 實際：bigblue4 V=8170 高維 → 4 particles 散開後找不到回頭路；SVDD 每步 resample 等同 deeper search at one branch
- **insight**：對「placement landscape 多 mode」(adaptec2) particle diversity 有用；對「巨大但相對 unimodal」(bigblue4) 反而散開後 ESS 觸發太晚

### 3.3 三族 saturate 在 -4% 區間 → 應該換軌道

| Method | vs Paper |
|---|---|
| SVDD | -4.36% |
| TDS | -3.86% |
| CoDe | -3.57% |
| **跨度** | **0.79%** |

三族跨度 < 1%。考慮到 seed std ~0.4-0.6%，**沒有顯著差異**。再 sweep K, λ, block_size, ess_threshold 預期增量 < 0.5%。

**下一步如果想突破 -4.4% 這個 ceiling**：
- (a) **FreeDoM time-travel** 加在這三族任一個上 → 預期再 -0.5~-1%（orthogonal）
- (b) **跟 fine-tuning 結合**：on top of ablation_10k (已知 svdd_layered 在 ablation_10k 沒增量，但 TDS 還沒測)
- (c) **教授建議 (1) 從頭 train**：跳出 inference-time search 框架

### 3.4 修了 codebase latent bug

`reverse_guidance_opt_force` 的 `alpha_cost.backward()` 在 B>1 不能跑（PyTorch 不會 auto-scalarize `(B,)` shape）。SVDD/CoDe 都 B=1 沒踩到。TDS N=4 觸發 → 修成 `.mean()` 後對 B=1 不變、對 N>1 正常運作。

**潛在影響**：往後如果有人想用 batched eval（B>1 同時跑多 placement），這個 bug 就會擋路。修了反而是順帶禮物。

## 4. 不要做的事

- ❌ **不要 sweep TDS K** — 已知 K=4 work，再大 cost 高、增量小
- ❌ **不要 sweep ess_threshold_frac** — 0.5 是 SMC standard，跑 0.3/0.7 預期差 < 0.2
- ❌ **不要 sweep tds_alpha_temp** — α=1 跟 SVDD 對齊，調整不影響 mean，只影響 variance
- ❌ **不要試 TDS on ablation_10k** — SVDD Phase 3 已知 fine-tuned ckpt 上 inference-time search 沒空間，TDS 預期同樣
- ❌ **不要再多 seed (n>3)** — 三族已穩定區分，加 seed 只是花時間

## 5. 可以做的事（plan_2 候選，按優先序）

### 5.1 (HIGH) FreeDoM time-travel layered on TDS / SVDD

- Survey §2.2 — 在 guided reverse step 後跳回 noisier state 重新 denoise，吸收 guidance 擾動
- 預期再榨 -0.5~-1%
- 5-10 行 code，便宜
- 跟 TDS/SVDD orthogonal → 可以 stack
- 成本：3-seed × 7 circuits ≈ 3-4 hr

### 5.2 (HIGH) 寫 paper draft / 跟教授 demo

3 方法 × 3 seed 已經是完整 ablation table。可以開始寫 method + results 章節。

### 5.3 (MEDIUM) 教授建議 (1) 從頭 train

完全不同 track，獨立啟動。

### 5.4 (LOW) SVDD-MC（學 value net）

survey §1.1。跟 SVDD-PM 比 marginal，需要訓 value net。

### 5.5 (LOW) Reflected / Mirror Diffusion

需要重訓，跟 (5.3) 從頭 train 同 track。

## 6. Plan_2 建議

**最有 leverage 的下一步**：**FreeDoM time-travel** layered on top of TDS（因 TDS 是最穩、worst-circuit 最好）。如果 FreeDoM 再貢獻 -0.5%，TDS+FreeDoM mean 進入 44.6 區間，可能跟 SVDD 拉開差距。

第二優先：**寫 paper**。3 個 method ablation 已經完整。

## 7. Open questions for plan_2

1. FreeDoM 加在 TDS 上還是 SVDD 上？我傾向 TDS（穩定性優勢）。SVDD 平均較低但 std 大，FreeDoM 可能放大噪音
2. FreeDoM 的「time-travel step」應該每步都做還是 every N steps？survey 沒明說，先預設 every 10
3. 「跳回 noisier 程度」應該多少？通常 1-3 個 timestep，先試 2

## 8. SVDD/TDS/CoDe paper narrative 的初步定位

如果用 CoDe 當 paper main method（最簡），章節結構建議：
- §3 Method：CoDe（blockwise best-of-N）作為 inference-time guidance
- §4 Ablations:
  - Selection rule: hard argmax (CoDe) vs soft softmax (SVDD) vs SMC importance (TDS)
  - Frequency: every step (SVDD) vs every block (CoDe) vs ESS-triggered (TDS)
  - N=K: 1 (baseline) vs 4 (ours) vs 8 (future work)
- §5 Results: 7-circuit, 3-seed, 三 method 都 beat paper -3.6~-4.4%
- §6 Discussion: stability (TDS) vs simplicity (CoDe) vs best-mean (SVDD) trade-off

可以給教授看這個架構，請他決定哪個當 main。
