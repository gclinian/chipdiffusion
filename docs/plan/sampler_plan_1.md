# Sampler 診斷與 Best-of-N 計畫（sampler_plan_1）— 2026-09-14

> 依據：`docs/survey/method_survey_1.md`。全部 **eval-only、零訓練**，在 5090 上總計約 6 GPU-h。
> 環境：`PYTHONPATH=. /ibmnas/427/r115/gclin/miniforge3/envs/chipdiff-b/bin/python`（chipdiff-b）。
> 每個 cell 用**獨一無二的 `method=`**（eval 目錄碰撞防護會擋，但別靠它）。
> 判讀：7-circuit avg 差距 < 1.3 = 無差異；n=1 不關方向；per-circuit 主張門檻更高（bigblue3 σ≈5.3）。
> 模型分工：執行 / 蒐集 → Opus；判讀 / 決策 → Fable。

## 0. 為什麼是這個順序
新 stack 上 baseline 從 48.691 → 45.987，比任何方法的宣稱增益都大。**先重錨，再診斷，再收割。**
三個診斷各自能一次決定一整個家族的生死，且全部零訓練。Best-of-N 是已知可實現的 ~1.7 增益。

## Phase 0 — 重錨（進行中）
- **0a** paper baseline, large-v2 + opt, seed 300 → **45.987** ✅（`base_cu128_s300_part{1,2}`）
- **0b** Run F, ablation_10k + opt, seed 300（`runF_cu128_s300_part{1,2}`）— running
- **0c** baseline seeds 301, 302 → 6 便宜 circuit 全跑 + bigblue4 只 seed 300（協定 §4）

**判定 0b**：|Run F_new − 44.321| < 1.3 → 舊 stack 數字可比，leaderboard 只換 baseline。
≥ 1.3 → 舊 stack 全部作廢，SVDD seed 300 也重跑（`svdd_cu128_s300`）後再往下。

## Phase 1 — 三個診斷（各自獨立，可平行排程；GPU 單卡所以序列跑）

### 1A Unguided control：FM 的失敗到底是什麼（0 LOC）
補 `flowmatch_plan_1.md` §5 預登記但從未執行的對照。`flowmatch_report_1.md` §3.2 的「no-guidance
泛化就已輸」背後是零個 run。
- Cells（adaptec1 + bigblue4，seed 300）：
  - DDPM large-v2, `guidance@_global_=none`, T=1000
  - FM `v1.61-fs.61.fm_p1_500k.61`, `guidance@_global_=none`, `model.max_diffusion_steps=50`
  - 對照：磁碟上已有 DDPM opt（0a）、FM opt（`fm_p1_nt50_full`）
- **預登記判定**（bigblue4 為主，adaptec1 為 sanity）：
  - FM-none / DDPM-none 的 bigblue4 比值 ≥ 1.5 → 失敗在 objective/sampler，FM 家族維持關閉。
  - 比值 < 1.2 → 失敗在 guidance 交互，**重開 FM**：下一步是 guidance 只在後段 t 做（report §3.1 假說）。
  - 中間 → 記錄，不重開。
- 成本 ~40 min。

### 1B Stochasticity dial × steps（1 LOC）
`schedulers.py` 加 `eta_scale`（η=1 = 現況 DDPM；η=0 = DDIM deterministic）。
- Cells（adaptec1 + bigblue4，seed 300，large-v2 + opt）：η ∈ {0, 0.5, 1} × T ∈ {1000, 100}
  + η ∈ {0, 1} × T=1000 with `guidance=none`（去除 guidance 混淆）
- **預登記判定**（bigblue4，guidance=none 那組為準）：
  - η=0 比 η=1 差 ≥ 30% → 「deterministic 在 size-OOD 下崩」**以證據確立**；few-step / drifting 家族
    正式關閉（升級為 evidential 禁令，n=1 → 之後補 seed 301/302 再寫進 next）。
  - η=0 ≈ η=1（< 10%）→ FM 的失敗**不是** sampler 確定性，是 velocity objective；記錄，
    few-step 家族仍不做（無 OOD 證據 + 移除 search 的每步錨定），但理由改寫。
  - 順帶量到 DDPM 的 T curve（1000 vs 100）：若 T=100 在 bigblue4 < +5% → Phase 2 的 draft 用 T=100。
- 注意：η=0 讓 SVDD 候選全同，這組**只跑 opt / none**，不跑 search sampler。
- 成本 ~2 h。

### 1C Log-SNR shift by V（~8 LOC）
`schedulers.set_timesteps` 加 `t_shift`：t' = s·t/(1+(s−1)t)，s = √(V/400)（V_train = v1.61 max）。
- Cells：large-v2 + opt，7 circuits，seed 300，s 依 V 自動算；另加 s 固定 = 1（= 0a，不用重跑）。
- **預登記判定**：avg7 比 0a 好 ≥ 1.3 → 跑 seed 301/302；否則關閉（P ≈ 20–25%，成本 25 min）。
- 也試反向 s = √(400/V)（Chen 的論證方向在非影像資料未驗證，符號未知）— 第二個 cell。

## Phase 2 — Best-of-N 形式化（~30 LOC）
- 修 `policies.open_loop_multi`（`max_score` 初始化 bug）；`utils.save_outputs` 記錄 **pre-legalization
  HPWL**；加 `+num_candidates=N` 走 best-of-N，score = post-legalization HPWL 且 legality ≥ 0.97 者優先。
- **2a rank-correlation**：N=4 on adaptec1 + bigblue4，記錄 pre-leg 與 post-leg HPWL 的 Spearman ρ。
  ρ ≥ 0.7 → draft 可用 pre-leg 分數省 51% legalization 時間；否則每個候選都要 legalize。
- **2b 協定 run**：large-v2 + opt，N=4，6 便宜 circuit × seeds 300/301/302 + bigblue4 × seed 300。
- **預登記判定**：報告 best-of-4 mean（3 seeds）與 legality floor；預期 ~43.1–43.5。
  這是 **協定** 不是方法主張；論文表要標 N 與 wall-clock。
- 成本 ~2.5 h。

## Phase 3 — Skeptic 存活的三個零訓練測試（2026-09-14 追加，預登記）
GPU 序列排在 Phase 2 與 1A follow-up 之後。全部 seed 300 先行；任何 cell 通過門檻才補 seed 301/302。

### 3E Deep guidance K（MacroDiff+；config-only）
- Cells：bigblue4，K ∈ {60, 150} × `model.alpha_critical_factor` ∈ {0.5（現況）, 1.0}；adaptec1 K=150。
  對照 0a（K=20）。skeptic 警告：α dual ascent 在 K 迴圈內，phase 1 α_init=0 → 純 HPWL 深 K 會疊 macro，
  故加 acf=1.0 變因。
- **判定**：bigblue4 任一 cell 比 0a 好 ≥ 5%（≈6.5）且 legality ≥ 0.98 → 補 seeds；否則關閉。
### 3F Post-hoc 權重平均（EMA 的零訓練代理）
- 對 `v1.61-fs.61.fs_p1_X_500k.61/step_{250k..500k}.ckpt` 做均勻平均 → `fs_p1_X_500k_avg.ckpt`；
  7-circuit eval；對照 **同 stack 重跑的 Run X 500k**（舊 45.053 是舊 stack，不可用）。
- **判定**：avg7 比 Run X（新 stack）好 ≥ 1.3 → 值得做真 EMA 訓練；否則配方方向關閉。
### 3H Frame averaging（D4 test-time symmetrization）
- `+model.frame_average=true`（8 frames，4 個 ε_θ call site），large-v2 + opt，adaptec1 + bigblue4。
- **判定**：bigblue4 比 0a 好 ≥ 5% 或 adaptec1 好 ≥ 3%（其 σ 小）→ 7 circuits + seeds；否則關閉。
- 成本警告：推論 ~8× 慢；bigblue4 cell 預估 60–70 min。

## Phase 3（舊）— 視 Phase 1/2 結果（不預先承諾）
- 1B 若 T=100 可用 → Phase 2 用 draft 加 N。
- EMA + cosine LR + grad clip 配方（`recipe_plan_1.md`，8 h 訓練）— 若 GPU 有空檔就跑，作為新預設。
- 方向六 `eval_policy_algorithm=iterative_clustering` config probe on bigblue2（32 GB 現在可能放得下 guidance）。

## 4. 便宜 3-seed 協定
bigblue4 佔 eval 時間 78%。**3 seeds 跑 6 個便宜 circuit + bigblue4 只跑 1 seed**，成本 ~52% 省。
報告時 bigblue4 標 n=1。

## 5. 產出
- `docs/report/sampler_report_1.md`：每個 phase 的表 + 對照預登記判定的結論
- `docs/next/sampler_next_1.md`：學到的 / 不要做（分 evidential / prudential）
- `STATUS.md` 由 ledger 自動更新；所有 run 自動進 `docs/ledger/`

## 6. 明確不做（本輪）
Drifting model、MeanFlow/Shortcut/sCM/IMM/CTM、ODE→SDE on FM、Graph Transformer backbone（壞的）、
conv-layer swap（丟 pin offset）、AddLoss 任何變體、DDPO 任何變體。理由見 survey §6。

## Phase 4 — 收尾 seeds（2026-09-15 追加，預登記，使用者核准）
GPU 閒置；序列 ~3 h。全部 eval-only。
### 4X Run X（from-scratch 500k）seeds 301/302，6 便宜 circuit
- 對照 0c baseline 同 seeds（配對）。**判定**：paired Δavg6（n=3）≤ −1.3 且三個 seed 同號 → 專案第一個
  在乾淨 stack 上成立的正向方法結果；−1.3 < Δ ≤ −0.8 → 補 seeds 303–305；否則記錄為 n=1 假警報。
### 4C DataAug Run C eval，7 circuits，seed 300
- 對照 Run X 同 stack（44.649, n=1）。**判定**：avg7 ≤ 43.35（−1.3）→ 補 seeds；否則關閉
  （dataaug_plan_1 §4 的規則搬到同 stack 比較）。
### 4B best-of-4（全 legalize）+ baseline，seeds 303/304/305，6 便宜 circuit
- 與 2c 合併成 n=6 配對。**判定**：mean Δavg6 ≤ −1.0 且 95% CI（t, df=5）不含 0 → 可報告的協定
  （報 effect ± CI、成本 4×、legality floor 0.97）；mean ≤ −1.3 → strong；否則關閉。
