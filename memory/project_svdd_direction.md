---
name: SVDD inference-time guidance direction (planning since 2026-05-19)
description: After AddLoss exhausted, next direction is SVDD-PM inference-time value-based decoding — survey + plan_1 committed, implementation pending
metadata:
  type: project
---

2026-05-08 教授 meeting (`docs/meet/meet_0508.md`) 建議兩個方向：
(1) 從頭 train 不要 fine-tuning
(2) denoising 過程嘗試其他 guidance（例如 RL guided）

使用者選擇先做 (2)。完成的事：
- `docs/survey/guidance_survey_1.md` — 涵蓋 SVDD / CoDe / TDS / SMCDiff / DRaFT / FreeDoM / Reflected / Mirror diffusion
- `docs/plan/svdd_plan_1.md` — SVDD-PM Phase 1 (adaptec1+bigblue1 sweep) + Phase 2 (full ISPD2005 from ablation_supervised_10k init)
- Plan 的判定基準：avg HPWL <43.5 = 顯著贏；>44.5 = 放棄 inference-time search 方向

**Why:** 三個關鍵動機：(a) 對應教授「RL guided」；(b) SVDD-PM 不需要 ∇reward，直接解 bigblue2 的 V×V legality Jacobian OOM；(c) DDPO local reward 已失敗（梯度爆炸），SVDD 是它的 gradient-free 替代。

**重要 codebase 事實**（survey §1.1）：
- HPWL guidance 是 O(E) PyG message passing，**不是** V×V 瓶頸
- 真正 OOM 的是 `legality_guidance_potential` 的 (B,V,V,2) overlap tensor — 已有 `legality_guidance_potential_tiled` 但只是 backward 版
- 現有 `opt` guidance 已經是 Universal Guidance + Lagrangian constraint，所以 B 系列（Universal/FreeDoM/DPS）增益有限

**How to apply:**
- 不要再做純 AddLoss / DDPO local reward 變種（[[project_addloss_v2_status]] 已說明）
- 教授建議 (1)「從頭 train」是獨立 track，與 SVDD 不衝突，可後續再開

**Phase 1 update (2026-05-19)**：實作完成於 `models.py` `_reverse_samples_svdd()` + `configs/guidance/svdd.yaml`。詳見 `docs/report/svdd_report_1.md`。
- 機制驗證 OK；default (K=4, every_n=5, λ=1) 在 raw HPWL 上 adaptec1 -0.7%（noise 內）、bigblue1 -3.0%
- `every_n=1` 給 adaptec1 raw HPWL -16.5%，但 legality 0.69→0.61（trade-off）
- 知名 bug：K=8 觸發 PyG batched edge_index 崩潰（`add_self_loops` × wrapper.py 互動），SVDD 邏輯本身沒問題

**Phase 2 update (2026-05-21)**：見 `docs/report/svdd_report_2.md`、`docs/plan/svdd_plan_2.md`。
- 7-circuit avg HPWL（with opt-adam legalization, ablation_10k init）：Run A (guidance=none) = **84.85**, Run C (SVDD every_n=1) = **71.94**, Δ = **-15.21%**
- SVDD 贏 7/7 circuits（最差 adaptec4 持平），bigblue3 -54.7% 最大
- 但 SVDD (71.94) **遠輸** CLAUDE.md ablation_10k + guidance=opt = 44.01 (+63%)
- Plan_2 §5.1 文字判定觸發「放棄 inference-time search」，但 plan_2 設計缺陷沒測 SVDD + opt **layered** → 不能直接結案
- GPU contention 教訓：GPU 0 被其他人佔 10 GB 時 bigblue3+ 的 `compute_pin_map` 會 `cudaErrorIllegalAddress`，要 `CUDA_VISIBLE_DEVICES=1` 規避

**Phase 3 update (2026-05-21) — SVDD 系列 CLOSED**：見 `docs/report/svdd_report_3.md`、`docs/plan/svdd_plan_3.md`、`docs/next/svdd_next_2.md`。
- 實作：`svdd_layer_opt` kwarg + `_reverse_samples_svdd` 加 opt guidance 呼叫 + `configs/guidance/svdd_layered.yaml`
- Run F（重現 paper baseline ablation_10k + opt + legalizer）= **44.32**（vs CLAUDE.md 公告 44.01，差 0.7% 在重現容忍內 ✓）
- Run E（SVDD + opt layered, K=4, every_n=1, λ=1）= **44.49** vs Run F 44.32 = **+0.37%**
- Plan_3 §5.2 判定：−0.5 ≤ +0.37 ≤ +0.5 → **持平 → inference-time search 路線正式 dead end**
- 為何 SVDD 沒贏 opt：(1) opt 20-step inner SGD 已把 predicted_x0 拉到 reward-friendly 點，K=4 jitter 失去 search 空間；(2) ablation_10k init 對 7-circuit 已接近 floor，SVDD headroom 小；(3) SVDD/opt 用同一 HPWL reward → no extra info（不像 RL 中 value function 對 policy gradient 提供 variance reduction）
- adaptec1 legality 從 0.988→0.957（SVDD reward × opt Lagrangian alpha 微弱干擾，但 HPWL 沒換回好處）
- **下一步**：寫 `svdd_next_3.md` 結案 + 開 `from_scratch_plan_1.md` 走教授建議 (1)「從頭 train」

**Phase 4 update (2026-05-21) — FRAMING CORRECTION**：見 `docs/report/svdd_report_4.md`。
- **重要修正**：Phase 3 把「ablation_10k + opt = 44.32」當 paper baseline → 錯。Paper 真實 baseline = **large-v2 + opt + leg = 46.89 (paper, seed=400) / 48.69 (我們重現, seed=300)**
- Phase 4 在 **large-v2 (paper checkpoint)** 上跑 SVDD：
  - Run G (large-v2 + svdd-only + leg) = **66.53**（取代 opt 失敗）
  - **Run H (large-v2 + svdd_layered + leg) = 45.10**（贏 paper 46.89 by **−3.8%**, 贏 our reproduction 48.69 by **−7.4%**）
- Run H 對 paper 公告數字：**贏 5/7 circuits, 平 1, 輸 bigblue3 (+14%)**
- **核心 insight**：SVDD 的增量取決於 checkpoint 強度
  - 弱 checkpoint (large-v2)：opt 還有 search 空間 → SVDD layered 有增量
  - 強 checkpoint (ablation_10k)：opt 已逼近 floor → SVDD 沒空間，no marginal value
- **leaderboard rank 5**（第一個從 large-v2 出發贏 paper 的方法；前 4 都用 fine-tuned ckpt）
- Phase 3 結論「SVDD 死路」**不成立**；應該是「SVDD 對 fine-tuned ckpt 沒用，對 paper raw ckpt 有用」
- 待跑：multi-seed 驗證 Run H、bigblue3 single-circuit debug、寫 svdd_next_3 完整 retrospective

**Phase 5 update (2026-05-24) — multi-seed CONFIRMED**：見 `docs/report/svdd_report_5.md`。
- Run H 3-seed (300/301/302) on large-v2 + svdd_layered + opt-adam, 7 circuits 完整
- Per-seed avg: 45.10 / 44.10 / 45.34 → **mean 44.85 ± 0.65, vs paper 46.89 = −4.36%**
- **3/3 seeds 都 individually 贏 paper**（最差 -3.3%，最好 -5.9%）
- 95% CI [44.11, 45.59] — upper bound 仍 < paper 46.89 → 統計顯著
- bigblue3 在 Phase 4 single-seed +14% 是 bad luck，**3-seed mean -3.0% 反而贏 paper**
- bigblue4 最大贏家 (-10.8%)，所有 seed 一致；adaptec2 / adaptec3 是唯二輸 paper 的
- **leaderboard rank 5** — 唯一不靠 fine-tuning 而贏 paper 的方法
- 對教授可定位為：「inference-time guidance 補強，等同一輪輕度 fine-tuning 的效果（贏 5 個 baseline-style method，跟 AddLoss v2 同級）」
- 下一步討論方向：(a) 寫 paper SVDD chapter、(b) sweep λ 再榨 -1-2%、(c) pivot 教授建議 (1) 從頭 train
