# 新方法 / 新架構 Survey（Survey #2）— 2026-09-14

> 動機：`docs/meet/meet_0620.md` 教授建議 2「嘗試新的 model 架構：Flow Matching、Drifting Model」；
> 使用者要求 survey flow matching、drifting model 及其他候選，survey 完依序實驗。
> 方法：6 個主題各一個 Opus agent 做 web search（每主題 ≥8 次搜尋、直接 fetch arXiv），
> 再對每個「strong / worth_a_pilot」候選派一個 skeptic agent 嘗試推翻，最後由 Fable 合成。
> 所有引用皆為 agent 實際 fetch 到的論文；找不到的明列於 §7。
> 判讀基準見 `CLAUDE.md` 判讀規則：7-circuit avg 差距 < 1.3 視為無差異；n=1 不關方向。

---

## 0. 一句話結論

**目前沒有任何新的生成目標或架構是好賭注。** 領域 2025–26 在 ISPD2005 上的進展全部來自
inference-time search，不是新的生成範式；所有 few-step / FM / drifting 家族都與本專案已記錄的
size-OOD 失敗共用機制（或更糟）；repo 內的替代 backbone 全部壞掉；訓練端唯一近乎確定有（小）
增益的是標準 EMA / LR schedule 配方。**兩件事排在任何新方法前面**：(1) 新 stack 上的
baseline 移動了 2.7 個單位（48.691 → 45.987），比任何方法宣稱的改進都大，所有比較都要重錨；
(2) 三個零訓練成本的診斷實驗，每個都能一次關掉或打開一整個家族。

---

## 1. 前提修正：baseline 換了

| | 舊紀錄（2026-03，old stack） | 新 stack（2026-09-14, seed 300） | paper |
|---|---:|---:|---:|
| adaptec2 | 39.06 (legality 0.93) | **30.75** | 31.0 |
| 7-circuit avg | 48.691 | **45.987** | 46.89 |

舊 3 月 baseline run 的目錄已被 eval 目錄碰撞覆蓋、config 不可驗、runtime 慢 4.5×、adaptec2
legality 0.93 —— 是壞掉的 run，不是 stack 漂移（其他舊 stack run 在同 checkpoint 上的 adaptec1
落在 8.8–9.4，與新值 9.03 一致）。後果：

- 所有「−X% vs 我們的復現」的說法灌水約 2.7 單位。
- SVDD 44.845 (3 seeds) vs 45.987 (n=1) 的差距 = 1.14 < 1.3 → **SVDD 相對 paper 方法在同環境
  的優勢尚未建立**。需要 baseline 多 seed 才能判定。
- paper 自己的 checkpoint 在我們環境贏 paper 已發表值 1.9%（seed 差異 + 硬體）。
- 5090 讓 7-circuit eval 從 63 min 降到 25 min/seed —— 3-seed 規則變得付得起。

Run F（ablation_10k + opt）正在新 stack 重跑，判定舊 stack 的微調數字是否仍可比。

---

## 2. Drifting Model —— 教授建議的那個，直接回答

**指的是**：*Generative Modeling via Drifting*（Deng, Li, Li, Du, **Kaiming He**，arXiv:2602.04770，
2026-02），後續 Sinkhorn-Drifting（2603.12366）、W-Flow（2605.11755）、Kernel-Gradient Drifting（2605.10727）。

**它是什麼**：**一步生成器**，不是 ODE/SDE 家族。沒有時間變數、沒有 noise schedule、沒有積分器。
生成器 f_θ 把 Gaussian noise 一次前向映射成樣本；訓練用 kernel 加權的「吸向真實 minibatch −
斥離自身 minibatch」的 drifting field V = V⁺_p − V⁻_q 做自回歸目標：
L = E‖f_θ(ε) − stopgrad(f_θ(ε) + V(f_θ(ε)))‖²。

**Repo 先前的關閉理由是錯的**（「同 few-step deterministic 家族」—— 它根本不是 velocity field），
**但結論以更強的理由成立**：

1. **Size-OOD 失敗的更尖銳版本**：kernel exp(−‖x−y‖/τ) 作用在整個樣本向量 R^{2V} 上。
   τ 在 V=400 調好，到 V=8,170 時 pairwise 距離以 √V 成長，field 飽和（全吸或全零）——
   正是 FM 報告記錄的「退化與 V 強相關」的機制，且更難修（Sinkhorn-Drifting 的存在本身就是
   為了 τ 敏感問題）。
2. **零 OOD 證據**：所有 drifting 論文都在固定維度上評估（ImageNet-256 latent、FFHQ、固定大小
   的球面 / DNA / 分子）。沒有任何 1-NFE 方法報告過對「比訓練大的輸入」的泛化。
3. **成本最高**：新 family（4 處註冊）+ kernel field + **重寫 data loader**（現在 `get_batch`
   給一張圖一個 ground truth，drifting 要每個 condition 一批真實樣本）+ 原論文無官方 code。
4. **價值錯位**：本專案贏的是 many-sample search（SVDD/CoDe/TDS 每步 K 個候選）；
   一步生成把「每步重新錨定」的機會全部拿掉。

**判定：不做。** 若教授堅持，最便宜的可辯護實驗是 §5 Phase 1a 的 stochasticity dial ——
它能直接回答「deterministic 在這個任務上是不是問題」，而不用實作 drifting。

---

## 3. Flow Matching 家族 —— 重開一半

### 3.1 已試過的 FM 為什麼輸（重新審視）
`flowmatch_report_1.md` 的診斷「deterministic velocity field 外插差」把兩件事混在一起：
**sampler 的確定性**（ODE vs SDE）與 **objective**（velocity regression vs ε regression）。
Repo 從未分開測。更嚴重：**§3.2 宣稱「no-guidance 泛化就已輸」，但磁碟上 13 個 FM eval 全部
`guidance_mode: opt`，沒有任何一個 no-guidance FM run。** 那句話背後是零個實驗。
（`flowmatch_plan_1.md` §5 預先登記了 unguided control，從未執行。）

### 3.2 候選

| 候選 | Skeptic 判定 | 理由 |
|---|---|---|
| **Unguided control**（FM ckpt vs DDPM ckpt，`guidance=none`）| **做，零 code** | 補上 report_1 §3.2 缺的實驗；決定 FM 的失敗是 guidance 交互還是 objective |
| ODE→SDE 轉換現有 FM ckpt | weak | score 恆等式在 t→0 發散（誤差 ∝ (1−t)/t），正是決定 legality 的資料端；guided path 不是 Euler step 所以 a=0 不還原 |
| Restart / EDM churn on FM ckpt | weak | 同上 + 磁碟資料指向 guidance 交互而非 sampler |
| Log-SNR / timestep shift（SD3 式，依 V 縮放）| worth_a_pilot | 見 §4.3 —— 對 DDPM 也適用，且更便宜 |
| FlowPlace（DAC 2026, 2604.23658）uniform prior + 硬約束投影 | weak | 從未在 ISPD2005 評估（ICCAD2015 / OpenROAD，數百 macro）；overlap 增益已被我們的 legalizer 吸收 |
| Reflow / 高階 ODE solver / stochastic interpolants / BFN | closed | 都不碰 size-OOD |

### 3.3 Few-step 家族（MeanFlow 2505.13447、Shortcut 2410.12557、sCM/TrigFlow 2410.11081、IMM、CTM 2310.02279）
全部 weak。它們把 1000 次 evaluation 壓成 1–4 次，**移除**了 many-step sampler 對 OOD 的
「每步重新錨定」；三者都在固定解析度評估，沒有 size-extrapolation 證據。
唯一的合理用法是 §4.2 的「few-step 當 draft 給 search 用」。

---

## 4. 存活的候選（skeptic 後仍 worth_a_pilot）

### 4.1 Stochasticity dial + steps curve（診斷，零訓練）
EDM S_churn（2206.00364）、Restart（2306.14878）、Schaeffer et al. KL 分析（2506.11378）的統一
說法：deterministic step 離散誤差小，但注入的 noise 會**收縮累積誤差**；score 不準時存在一個
非零最佳 noise level。OOD circuit 正是 score 最不準的情況。
- 實作：`schedulers.py:65` 的 `eta()` 是 DDIM η=1，加一個 `eta_scale` 乘數（1 行 + 1 config key）。
- Skeptic 保留意見：η 與步數交互（低步數時 stochasticity 放大 score 誤差）；η 與 guidance 混淆
  （每步 20 次 inner Adam）；**η=0 會讓 SVDD/CoDe/TDS 的候選全部相同**（`x_cand = μ + η·z`）；
  η>1 在這個 scheduler 不可表達。
- 所以正確的設計是 2-D（η × T）而且要有 unguided 對照。

### 4.2 Best-of-N 形式化（零訓練，已在 comeback_next_1 P2 2.2）
Skeptic 校正：可實現的增益是 **best-of-3 = 43.11**（Δ1.74），不是「到 41.84 oracle 的 3.0」——
41.84 是跨 checkpoint 且 legality 不乾淨的 oracle。82% 的增益來自三個高 σ circuit（noise harvesting，
須報 legality floor）。**Repo 內 `policies.open_loop_multi` 有 bug**：`max_score=0` 初始化 +
`score > max_score`，任何負的 −HPWL score 永遠回傳 sample 0。且 pre-legalization HPWL 沒被記錄，
「便宜 draft」的 rank-correlation 假設無法驗證。
- Flash-BoN、2501.09732 的觀察：等 wall-clock 下，simple BoN 打得過多數 guided search。

### 4.3 Log-SNR shift（inference-only，唯一「機制就是輸入比訓練大」的候選）
Chen 2023（2301.10972）、simple diffusion（2301.11093）、SD3（2403.03206）：固定 noise schedule
在不同輸入維度代表不同有效 corruption；SD3 明說 shift 是 token 數的函數。這裡 token = macro，
V 從訓練 400/1600 到測試 543–23,084，log-ratio 達 4 nats。
- 實作：`schedulers.set_timesteps` 的均勻 grid 換成 t' = s·t/(1+(s−1)t)，s = √(V/V_train)，
  5–10 LOC，一個 config key，一次 eval。
- Skeptic：效果的符號與存在都未在非影像資料驗證（Chen 的論證靠影像空間平滑）；P(>1.3) ≈ 20–25%，
  但成本是一次 eval，值得。

### 4.4 Deep guidance（MacroDiff+，IJCAI 2026, 2605.16451）
每個 reverse step 對 x̂₀ 做 K=300–700 次 gradient descent（我們 57 個 guided run 全是 K=20）。
- Skeptic：他們的證據是 **in-distribution**（每個 benchmark 各自用 1000 個 augmented netlist 訓練），
  8 個設計裡 7 個比我們最小的 circuit 還小，HPWL 是 mixed-size 不可比。
  Hidden cost：α dual ascent 在 K 迴圈**內**，K×15 → α 更新次數 ×15 提早飽和；phase 1（t>0.5）
  α_init=0 所以 deep K 是純 HPWL 目標 → macro 全疊在一起。
- 最便宜的測試：離線 drift probe（0.05 GPU-h），K∈{20,60,150,300} 看 ‖g‖ 與 legality potential 是否爆。

### 4.5 訓練配方：EMA + cosine LR + grad clip（EDM2, CVPR 2024）— skeptic 後 **weak**
`train_graph.py` 沒有 EMA、LR schedule、warmup、grad clip、weight decay。這是 2020 以來每篇
diffusion paper 的標配。~40–50 LOC；EMA checkpoint 是 `from_checkpoint` 的 drop-in，guidance /
search / legalizer 全不動。高信心小正效果；低信心單獨超過 1.3。
- **Skeptic（已完成）**：訓練量不是瓶頸 — FromScratch_X 500k = 45.053 vs 1.6M = 45.078（磁碟上已有）；
  偵測下限 ~1.2 vs 候選自估 0.45。**最便宜的測試零訓練**：`v1.61-fs.61.fs_p1_X_500k.61/` 有 10 個
  50k 間隔的 snapshot，做 post-hoc 權重平均（SWA/soup 式）出一個 ckpt 再 eval（~1.1 GPU-h）。
  排 Phase 3。

### 4.6 Scale conditioning / hierarchical（SBGD 2508.14352；Hier-RTLMP 2304.11761）— skeptic 後 **weak**
唯一直接**攻擊** size-OOD 機制的方向（網路現在完全不知道 circuit 多大）。但成本最高、變異最大，
且有陷阱：conditioning 值本身（V=8170）也 OOD。分階段：先跑 config 級的
`eval_policy_algorithm=iterative_clustering`（方向六，程式碼已在、123 個 run 從沒開過）。
- **Skeptic（已完成）— 前提為假**：對 `docs/ledger/results/` 全部結果回歸 quality vs macro 數：
  hpwl_ratio 在 bigblue4 (8,170) **最好** (0.615)，legality 隨 V 上升；我們對 paper / OrderPlace 的
  劣勢全在 **543–1,329 macro** 的 circuit。「size-OOD」是 FM 的故事，不是 DDPM+guidance 的。
  另：repo 內沒有 macro clustering（`utils.py:1315` 與 `parsing/utils.py:339` 都跳過 macro），
  config probe 是 no-op。**關閉。**

### 4.7 Exact D4 equivariance by frame averaging（test-time symmetrization）— skeptic 後 **worth_a_pilot**
Repo 已證明 dihedral D4 對 HPWL/legality 不變（`utils.dihedral_transform_graph`）；訓練端 augmentation
（Run C）從未 eval。推論端等價做法：ε̄ = (1/8) Σ_g g⁻¹ ε_θ(g·x, g·cond)，零訓練、任何 checkpoint。
- **Skeptic 跑了 3 個 GPU probe（~0.15 GPU-h，paper ckpt + 真實 ISPD 圖）仍無法在機制上推翻**；
  不改 sampler 的 stochasticity、步數或 ODE/SDE 性質，與 1B/1C 正交。
- Hidden cost：4 個 ε_θ call site（plain + svdd ×2 + code/tds）；`BatchWrapper` 在 B·E > 80k 會 chunk，
  8 個 frame 無法 fuse 成一個 batched forward → 推論 ~8× 慢（bigblue4 sampling ~+50 min）。
- 最便宜測試：`+model.frame_average=true`，adaptec1 + bigblue4，seed 300，~0.4–1 GPU-h。排 Phase 3。

### 4.8 其餘 skeptic 判定（全部 weak，不排）
Attention temperature log-V scaling（skeptic 實跑 probe：各 circuit 最佳溫度與 V 無序）；
AR-hybrid commit-and-freeze（mask/conditioning 通道是死碼：`is_ports` 在全部訓練資料恆為 0，
該輸入通道權重仍在初始值）；MultiDiffusion 子圖融合（訓練樣本由 75–90% 密度定義而非節點數；
子圖不能 batch）；RePaint/Restart 式 test-time refinement（Lagrangian α 是 x 之外的狀態，
re-noise 無法收縮它）；DDPM-IP / offset noise；FlowPlace 硬投影（該投影算子 repo 內不存在，
每步要比 20k-step legalizer 快 1000×）。

---

## 5. 領域現況（2025–26 ISPD2005）

| 方法 | 類型 | 6-circuit avg（無 bb2/bb4）| 備註 |
|---|---|---:|---|
| ChipDiffusion（paper）| diffusion | 31.27 | 唯一在完整 bigblue4 (8170) 報數的生成式方法 |
| 我們（新 stack baseline, n=1）| diffusion | ~32.1 | adaptec1 9.03 / adaptec2 30.75 |
| **OrderPlace**（ICML 2026, 2606.08904）| wire-mask + LLM 演化 | **29.98**（5 seeds）| adaptec1 **5.75**、bigblue1 **2.00**；bb2/bb4 只放 1024 個 module |
| EGPlace（ICML 2025）| wire-mask + 演化 | 35.7（**輸** ChipDiffusion）| 數字取自 OrderPlace Table 2 同協定重跑；同上限制。2026-09-25 校正：原本寫「相近」是錯的 |
| MacroDiff+（IJCAI 2026）| diffusion + deep guidance | 不可比（mixed-size）| in-distribution 訓練 |
| FlowPlace（DAC 2026）| flow matching | 不可比（ICCAD2015）| 未評 ISPD2005 |
| DiffPlace（2510.15897）| diffusion + CFG | — | CFG 半部與已關閉的 AddLoss 同構 |

要點：(a) 構造式 wire-mask search 在小 circuit（adaptec1、bigblue1）大幅領先，這是我們的短板；
(b) **沒有任何方法報告完整 bigblue2 / bigblue4** —— 我們在 bigblue4 (129.5) 的數字沒有對手；
(c) 領域的增益來自 inference-time outer search，這與 SVDD/CoDe/TDS 的發現一致。

---

## 6. 對比表

| # | 候選 | 改哪裡 | 訓練? | 共用 FM 失敗? | Skeptic 後 | 成本 | 決定什麼 |
|---|---|---|---|---|---|---|---|
| A | Unguided control FM vs DDPM | config | 否 | 直接檢驗 | **做** | 0 LOC, ~40 min | FM 失敗是 guidance 還是 objective |
| B | η dial × T curve (DDPM) | scheduler 1 行 | 否 | 直接檢驗 | pilot | 1 LOC, ~2 h | deterministic 是不是問題；few-step 家族生死 |
| C | Log-SNR shift by V | scheduler ~8 行 | 否 | 否 | pilot | ~25 min/seed | size-OOD 是否可用 schedule 修 |
| D | Best-of-N 協定 | policies bug fix + logging | 否 | 否 | pilot（已排程）| ~30 LOC, ~2.5 h | 可報告的 43.1 |
| E | Deep guidance K | config + probe | 否 | 否 | pilot | 0 LOC probe | K=20 是否欠調 |
| F | Post-hoc 權重平均（EMA 的零訓練版）| 小腳本 | 否 | 否 | weak→probe | ~1.1 GPU-h | 訓練配方是否值得 |
| G | Scale cond. / hierarchical | — | — | — | **關閉**（前提為假：quality 不隨 V 退化）| — | — |
| H | Frame averaging (D4) | 4 個 call site ~40 LOC | 否 | 否 | pilot | ~1 GPU-h | 對稱性是否值 |
| — | Drifting model | 新 family + loader | 是 | **更糟** | weak | 週+ | 不做 |
| — | MeanFlow / Shortcut / sCM / IMM / CTM | 新 family | 是 | **是** | weak | 天–週 | 不做 |
| — | ODE→SDE / Restart on FM ckpt | sampler | 否 | partially | weak | — | 不做 |
| — | Graph Transformer / conv swap | backbone | 8 h | 否 | weak / **壞的** | — | gt.py 缺 kwarg、丟 edge_attr、每步重算 Laplacian；gcn/sage/gin 靜默丟 pin offset |

---

## 7. 找不到的（誠實列出）
- 任何 drifting / 1-NFE 論文報告對「比訓練大的輸入」的泛化。
- 任何 ODE-vs-SDE sampler 在 **graph-size** 分佈偏移下的直接量測。
- 任何生成式 placer 在完整 ISPD2005 規模（543–23,084 movable）的 2025–26 結果。
- 任何把 SMC / particle search 用在 chip placement 的已發表工作（我們的 SVDD/CoDe/TDS 仍是唯一）。
- ChipDiffusion 作者的任何後續工作。
- MacroDiff（DAC 2025 LBR）本體的方法細節。
- EGPlace / OrderPlace 在完整 bigblue2 的數字（都只放 1024 個 module）。

---

## 8. 推薦執行順序 → `docs/plan/sampler_plan_1.md`
Phase 0 重錨 → Phase 1 三個診斷（A ✅, B, C）→ Phase 2 Best-of-N（D）→ Phase 3 三個 skeptic 存活的便宜測試（E deep-K, F post-hoc 平均, H frame averaging）。

**Skeptic pass 全部完成後的一句話**：28 個候選裡只剩 5 個零訓練測試值得跑；沒有任何一個需要新的生成範式或新 backbone。

## 9. 主要參考
- Drifting: 2602.04770, 2603.12366, 2605.11755, 2605.10727 ・ Few-step: 2505.13447, 2410.12557, 2410.11081, 2310.02279
- Stochastic sampling: 2206.00364 (EDM), 2306.14878 (Restart), 2506.11378, 2605.26582, 2511.22242
- Schedule shift: 2301.10972, 2301.11093, 2403.03206 ・ Training recipe: EDM2 CVPR 2024
- BoN / test-time scaling: 2501.09732, Flash-BoN ・ Placement: 2605.16451 (MacroDiff+), ICML 2025 PMLR 267 (EGPlace),
  2606.08904 (OrderPlace), 2604.23658 (FlowPlace), 2510.15897 (DiffPlace), 2508.14352 (SBGD), 2605.10547 (PhysEDA), 2603.28733 (VeoPlace)
