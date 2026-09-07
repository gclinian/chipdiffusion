# SVDD-PM 第二次實驗計畫（Phase 2: Legalization + Full ISPD2005）

> 前次回顧：`docs/next/svdd_next_1.md`
> Plan_1：`docs/plan/svdd_plan_1.md`（§3.2 判定基準直接沿用）
> Report_1：`docs/report/svdd_report_1.md`

## 目標

把 Phase 1 在 raw HPWL 觀察到的 `every_n=1 → -16.5%` 訊號帶到 **deployment-fair** 條件下驗證：
1. 從 **`v1.61-ddpo.ablation_supervised_10k.61/latest.ckpt`**（目前最佳 init，7-circuit avg 44.01）起
2. 加 **opt-adam legalization**（生產環境的 default legalizer）
3. 跑 **完整 7 circuits**（skip bigblue2）
4. 套用 **plan_1 §3.2 的決策樹**

---

## 1. 核心假設

> SVDD-PM `every_n=1` 把 macro 推向更短 wire 的 placement（HPWL ↓），雖然 raw legality 退步，但 opt-adam legalization 在 ISPD scale 上能修回 legality 而 **HPWL 損失 < SVDD 帶來的增益** → 7-circuit avg HPWL 贏 ablation_10k 的 44.01。

如果這假設成立 → 寫 paper。
如果不成立 → SVDD 跟 ablation_10k 訓出來的 placement 已經處在 legality-HPWL Pareto front 上同一個位置，無法用 inference-time search 進一步推進。

---

## 2. 實驗矩陣

| Run | from_checkpoint | guidance | legalizer | num_output_samples | seed | scope | note |
|-----|----|----|----|----|----|----|----|
| **A**（reference） | ablation_10k | none | opt-adam | 7 (0..6) | 300 | 確認 baseline 重現 44.01 | 應該等於 CLAUDE.md 的數字 |
| **B**（SVDD default） | ablation_10k | svdd default (K=4, every_n=5, λ=1) | opt-adam | 7 | 300 | sanity check default | |
| **C**（SVDD aggressive） | ablation_10k | svdd K=4, every_n=1, λ=1 | opt-adam | 7 | 300 | **主菜** |
| **D**（SVDD aggressive seed 301） | ablation_10k | svdd K=4, every_n=1, λ=1 | opt-adam | 7 | 301 | seed validation 在 C 跑完且有訊號才跑 |

**先跑 A B C 三個**。如果 C 有正訊號（avg < 44.01）就跑 D 驗證 seed-stability。

### 2.1 為什麼跳過 K=8

`networks/layers/wrapper.py:31` unbatch/rebatch 限制（next_1 §3.3）。**Phase 2 用 K=4，不冒風險**；若 K=4 已贏，K=8 可在 plan_3 chunk=4 序列化下測。

### 2.2 為什麼跳過 bigblue2

Legality forward V=23k 仍是 V×V matrix，會 OOM。`skip_guidance_threshold=10000` 會把 bigblue2 的 guidance_mode 設為 "none" → fallback 到一般 reverse_samples。我們會留 bigblue2 結果在表上但**不計入 7-circuit avg**（跟 ablation_10k 一致）。

### 2.3 為什麼不 sweep lambda

next_1 §5 已經辯論過：legality 留給 legalizer，SVDD reward 不應該太重 legality。Phase 2 守 λ=1 一個值，**不要再 sweep**。

---

## 3. 預期成本

| Run | 估時 |
|-----|------|
| Run A (baseline) | 7 circuits × ~10-15 min / circuit = **~60-90 min** |
| Run B (svdd default, every_n=5) | 同 A，generation time 加 ~5% → **~60-95 min** |
| Run C (svdd every_n=1) | generation time 大概 +20-50%（Phase 1 觀察 +10%）→ **~80-120 min** |
| Run D (seed 301, 如必要) | 同 C → ~80-120 min |
| **總計（不含 D）** | **~3-5 小時** |

> Note: bigblue4 (V=8170) 是時間瓶頸（legalization ~25 min/sample）。

---

## 4. 具體 eval 指令

### Common（沿用 CLAUDE.md "Correct Eval Commands"）

```bash
PYTHONPATH=. python diffusion/eval.py \
  method=<METHOD_NAME> \
  task=ispd2005-s0 \
  from_checkpoint=v1.61-ddpo.ablation_supervised_10k.61/latest.ckpt \
  legalizer@_global_=opt-adam \
  guidance@_global_=<GUIDANCE> \
  num_output_samples=8 \
  +start_sample=0 \
  +skip_guidance_threshold=10000 \
  legalization.alpha_lr=8e-3 \
  legalization.hpwl_weight=12e-5 \
  legalization.legality_potential_target=0 \
  legalization.grad_descent_steps=20000 \
  macros_only=True \
  logger.wandb=false \
  seed=300
```

### Run A（baseline）

加 `guidance@_global_=none` 並 `model.grad_descent_steps=0`（這個 baseline 是「ablation_10k checkpoint + no guidance + opt-adam legalization」, 對照 CLAUDE.md ablation_supervised_10k 跑法）。

> ⚠️ CLAUDE.md 給的 "Correct Eval Commands" 用的是 `guidance=opt` + `model.grad_descent_steps=20 + hpwl_guidance_weight=16e-4`。為了跟 SVDD 公平比，**Run A 也要用 guidance=none**（即「ablation_10k checkpoint 自身能力」），讓 SVDD 真正測得是「inference-time intervention 的純效應」。若 Run A 比 44.01 高得多，要對照「guidance=opt」基準。

### Run B（SVDD default）

```
guidance@_global_=svdd
model.svdd_num_candidates=4
model.svdd_alpha_temp=1.0
model.svdd_lambda_legality=1.0
model.svdd_every_n_steps=5
model.svdd_start_step_frac=0.5
```

### Run C（SVDD aggressive, 主菜）

```
guidance@_global_=svdd
model.svdd_num_candidates=4
model.svdd_alpha_temp=1.0
model.svdd_lambda_legality=1.0
model.svdd_every_n_steps=1          # ← 唯一差異 vs Run B
model.svdd_start_step_frac=0.5
```

---

## 5. 量測指標 & 判定

主指標：**7-circuit avg `macro_hpwl_rescaled / 100`**（= paper HPWL ×10⁵，跟 ablation_10k 完全可比）

次指標：
- 每 circuit `macro_hpwl_rescaled / 100`
- 每 circuit `macro_legality`（>0.99 為 deployment OK）
- `generation_time` 監控 SVDD 額外 cost

### 5.1 判定（沿用 plan_1 §3.2）

| Avg HPWL (7 circuits) | 意義 | 行動 |
|---|---|---|
| **< 43.5** | **SVDD 顯著贏 Ablation 10k** | 寫 `svdd_report_2.md`；下一 plan 走升級 SVDD-MC 或 TDS；考慮寫 paper |
| 43.5 – 44.0 | SVDD 微贏 | 試 K=8 (chunk=4)、α_temp / start_step_frac sweep |
| 44.0 – 44.5 | 持平 ablation_10k | 看 bigblue2 表現；考慮 stacking SVDD on top of opt guidance |
| > 44.5 | SVDD 沒贏 | 放棄 inference-time search 方向（plan_1 §5 路線跑完）→ 開教授建議 (1)「從頭 train」track |

### 5.2 額外觀察點

- **Run C vs Run B**：差 every_n=1 vs 5。如果 C 顯著比 B 好 → 證實「越頻繁 SVDD 越好」假設。
- **Run A 跟 ablation_10k 公告 44.01 的一致性**：A 應該接近 44.01；若顯著偏離（>0.5），表示 baseline 重現有問題，要 debug。
- **bigblue2 表現**：附在表上但不計 avg。Run A/B/C 對 bigblue2 都退化成 baseline reverse（skip_guidance_threshold 觸發），三個 run 在 bigblue2 上應該完全一致 → cross-check 一下確認沒有 leak。

---

## 6. 風險與對策

| 風險 | 對策 |
|------|------|
| Run A 重現不出 44.01 | 對照 logs/diffusion_debug/ispd2005-s0.eval_macro_only.300/ 的舊紀錄找 diff；可能是 `guidance=none` vs `guidance=opt` 差別 |
| every_n=1 在 bigblue4 (V=8170) memory 不夠 | 把 K=4 candidates 分 K_chunk=2 跑（即 K=4 → 2 個 K=2 chunk） |
| Legalization 後 SVDD 的 HPWL 增益完全被吃掉 | 報告核心發現 → 寫 next_2.md 結論「inference-time search 對 placement 無增益」→ 切到 plan_3 走教授建議 (1) |
| Bigblue2 三 run 行為不一致 | 表示 SVDD code path 沒乾淨 disable，要 debug `is_guided_sampling` flag |
| 一晚跑不完 | A → C → B 排序（先把主菜 C 跑完），D 留下次 |

---

## 7. 執行順序

1. **Run A**（baseline 重現） — 確認環境/checkpoint 沒問題
2. **Run C**（aggressive，主菜） — 先跑能直接決定方向的那個
3. **Run B**（default） — 對照 C 的 every_n 效應
4. **如 C avg < 44.01**：跑 **Run D**（seed 301）validate
5. 收結果 → 寫 `docs/report/svdd_report_2.md`
6. → 寫 `docs/next/svdd_next_2.md` 決定 plan_3 方向

---

## 8. 行動清單

- [ ] Run A: baseline (none) on ablation_10k + opt-adam
- [ ] Run C: SVDD every_n=1 on ablation_10k + opt-adam
- [ ] Run B: SVDD default on ablation_10k + opt-adam
- [ ] (Conditional) Run D: SVDD every_n=1 seed=301
- [ ] 收集 metrics.csv，對照 plan §5 判定
- [ ] 寫 `docs/report/svdd_report_2.md`
