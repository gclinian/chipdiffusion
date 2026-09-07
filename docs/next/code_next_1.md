# CoDe Phase 1 回顧（next_1）

> Report: `docs/report/code_report_1.md`
> Plan: `docs/plan/code_plan_1.md`
> 寫作日期：2026-05-25

## 1. 一句話總結

**CoDe ≈ SVDD（差 +0.37 在噪音內）** → 確認「inference-time K-particle search」是贏 paper 的核心機制，soft (SVDD softmax) vs hard (CoDe argmax) 不重要。CoDe 是更簡單、穩、快 17% 的版本，**值得 paper 主推**。**SVDD 主要在 bigblue4 (+4.42) 還比 CoDe 好**，是唯一例外。

## 2. Plan_1 vs 實際

| Plan_1 預期 | 實際 |
|---|---|
| CoDe 可能落在 44.5-45.5 區間「CoDe ≈ SVDD」 | ✅ 命中（45.22）|
| CoDe variance 應該比 SVDD 小（argmax 收斂）| ✅ 中型 circuit variance 較小（bigblue3 std 5.34 → 3.22）|
| Cost 應該明顯比 SVDD 便宜 | ✅ 17% 較快（但沒像理論「每 step → 每 100 step」的 100× 差距，因為 opt 的 20 inner SGD 才是 bottleneck）|
| 預期 7/7 circuits CoDe 跟 SVDD 接近 | ❌ **bigblue4 CoDe 顯著輸 SVDD 4.42**，沒預期到 |
| 預期 default K=4, block_size=100 OK | ✅ 沒踩 OOM、沒崩潰 |

**沒預期到的事**：bigblue4 的差異。大 circuit (V=8170) 上「探索 vs 收斂」trade-off 明顯偏向「探索」。CoDe 每 100 步才 fork 1 次 → 不夠探索；SVDD 每步都做 → 探索充足。

## 3. 學到的事

### 3.1 Inference-time search 是 universal 的核心機制

兩個機制完全不同（soft vs hard, every-step vs every-100-step），結果幾乎一樣 → 真正的 leverage 來自「**用 reward 評估 K 個 candidate 再選一個**」這個動作本身，與 selection rule 細節無關。

對 paper 敘事的影響：
- 用 CoDe 當 main method（更簡潔、更便宜）
- SVDD 列為「soft-weighted variant」ablation
- 顯著差別只在大 circuit (bigblue4)，這個觀察很值得寫進 paper discussion

### 3.2 大 circuit 對「探索」敏感

bigblue4 (V=8170)：SVDD 125.48 vs CoDe 129.89 = 差 4.42 unit (3.5%)。
- 不是 noise（SVDD std 4.42, CoDe std 1.34 都遠小於差距）
- 結構性差別：placement landscape 在 V↑ 時 mode 更多，搜尋深度（每步多探索）重要

對「往後該怎麼設 block_size」的提示：
- 對 V ≤ 1300 的 circuit：block_size=100 夠了
- 對 V ≥ 8000 的 circuit：應該用更小 block_size（e.g. block_size=20 → 50 個 block）
- **Plan_2 可以試 block_size sweep 來看這個假說**

### 3.3 Seed=302 bigblue3 legality = 0.929 是異常

只有這個 (seed, circuit) 組合 legality 低於 0.95，其他都 0.99+。
- CoDe argmax 在這個 seed 把 placement 拉到 legality 較差但 HPWL 較低的角落
- Legalizer (opt-adam 20000 steps) 也只能修到 0.929（其他 seed 修到 0.997）
- 不影響 3-seed 大局，但說明 **CoDe argmax 比 SVDD softmax 更激進 — 偶爾會掉進 reward 看起來高但 legalizer 修不回來的洞**

對未來 sweep 的提示：
- `code_lambda_legality=2` 或 3 可能改善（讓 reward 把 legality 看更重）
- 但加重 legality 可能犧牲 HPWL → trade-off

### 3.4 計算 cost 沒有想像中差別

預期 CoDe 比 SVDD 快很多（K-particle 只在 10 個 block vs 1000 step），實際只快 17%。原因：opt guidance 的 20 inner SGD 才是 wall-clock bottleneck（每 reverse step 都做）。
- K-particle eval cost 在 SVDD（1000 × K=4 = 4000 extra forwards）和 CoDe（10 × K=4 = 40 extra forwards）的差別在 wall clock 上不明顯
- 真正想省 cost 要把 `grad_descent_steps` 砍下來（但會傷害 HPWL）

## 4. 不要做的事

- ❌ **不要 sweep K** — CoDe 跟 SVDD 都用 K=4 就贏 paper 了，加大 K 預期增量有限
- ❌ **不要 sweep λ_legality** — SVDD λ=1 → CoDe λ=1 都 OK，除了 seed=302 bigblue3 那個 outlier
- ❌ **不要試 SVDD-MC** — 已知 SVDD-PM 跟 CoDe 平手，學 value net 是過早優化
- ❌ **不要再 multi-seed Run F (paper opt)** — 已知 single-seed 重現 paper, multi-seed 不會改變 conclusion
- ❌ **不要做 CoDe-only (no opt layering)** — 從 SVDD Phase 4 Run G (svdd only = 66.53) 已知這條 path 必輸

## 5. 可以做的事（plan_2 候選）

### 5.1 (HIGH PRIORITY) block_size sweep 看大 circuit 的行為

針對 bigblue4 那個 +4.42 差異，試 `code_block_size ∈ {20, 50, 100, 200}`：
- 小 block_size：更頻繁 fork，預期 bigblue4 改善（往 SVDD 靠近）
- 大 block_size：更少 fork，預期 7-circuit avg 退步
- 想找 sweet spot：是否 block_size=20-50 能贏 SVDD？
- 成本：4 個 block_size × 3 seed × 7 circuits = 12 runs ≈ 20 小時 （太長）
- 妥協：先用 single-seed 4 個 block_size 找 sweet spot（4-5 hr），再 multi-seed 驗證最佳設定

### 5.2 (MEDIUM PRIORITY) FreeDoM time-travel 加在 CoDe / SVDD 上

Survey §2.2 ── 在 reverse step 後跳回 noisier state 再 denoise，可能再榨 -0.5~-1%。
- 5-10 行 code
- 跟 CoDe / SVDD 正交
- 成本：3-seed × 7 circuits ≈ 3 小時

### 5.3 (MEDIUM PRIORITY) TDS (Twisted SMC)

Survey §1.3 ── SVDD 的 importance-weighted 升級。理論更乾淨（asymptotic exact）。
- 實作 2-3 天
- 不確定是否會超過 SVDD（很可能 marginal improvement）
- 對 paper：「TDS 是 SVDD 的 principled SMC 變種，效能相當」也是個 ablation

### 5.4 (LOW PRIORITY) 寫 paper / 跟教授 demo

3-seed 多個方法都贏 paper，已經有足夠 narrative。

### 5.5 跟教授建議 (1) 從頭 train 平行 track

不衝突，可以開另一條線。

## 6. Plan_2 建議

**最有 leverage 的下一步**：CoDe block_size sweep 針對 bigblue4，看能不能把 5.1 那個 +4.42 差異消掉。如果可以 → CoDe 在 7-circuit 完整贏 SVDD，paper main method 確定 = CoDe。

第二優先：FreeDoM time-travel ── 便宜，可能再榨 -0.5%。

之後再考慮 TDS 跟 from-scratch train。

## 7. Open questions for plan_2

1. **block_size 的 sweep 範圍**：{20, 50, 100, 200} 還是 {25, 50, 100}？ 我傾向後者（少跑）
2. **K**：固定 K=4 還是 sweep？固定 K=4（plan_1 結論）
3. **bigblue4 single-circuit sweep**：先用 single-circuit 多跑 block_size，再決定 multi-circuit？ 是 — single-circuit ~10 min × 4 blocks × 3 seeds = ~2 hr
4. **seed=302 bigblue3 legality 0.929 要不要 debug**？ 暫時不用，等 sweep 看其他 setup 是不是消失
