# Sampler 診斷與 Best-of-N 報告（sampler_report_1）— 2026-09-14/15（完成）

> 對應計畫：`docs/plan/sampler_plan_1.md`。每一節對照該 phase **預登記的判定標準**。
> 環境：`chipdiff-b`（torch 2.11.0+cu128, RTX 5090 32.6 GB）。所有數字 7-circuit avg HPWL ×10⁵，
> 排除 bigblue2；avg6 另排除 bigblue4。原始 metrics 在 `docs/ledger/results/`。
> 狀態標記：✅ 完成並判定 ／ ⏳ 佇列中 ／ ⬜ 未開始。

## 0. 摘要（隨進度更新）
- ✅ 0a：paper checkpoint 在新 stack = **45.987**（舊紀錄 48.691 是壞掉的 run；paper 46.89）。
- ✅ 0b：Run F 新 stack = **45.700** vs 舊 44.321 → Δ=+1.379 ≥ 1.3，**gate 觸發：舊/新 stack 不可混比**。
  同 seed 同 stack，微調 vs paper ckpt = **Δ−0.29（雜訊帶內，n=1）**。
- ✅ 1A：unguided FM / DDPM 在 bigblue4 = **2.45 ≥ 1.5** → FM 失敗在 objective/sampler，**維持關閉**。
- ✅ 0c：baseline 3 seeds avg6 = **31.212 ± 0.813** → 同 stack 2σ bar = 1.63（預登記 1.3 維持，另報 paired t）。
- ✅ re-anchor：**微調 Δ=+0.002、SVDD Δ=−0.175（paired, n=3）— 兩個標題主張都是壞 baseline 的假象。**
- ✅ 1B：unguided bigblue4 η=0/η=1 = **1.63 ≥ 1.30** → 確定性 sampler 在 OOD 崩潰，few-step 家族關閉；T=100 +7~14% → 不用 draft。
- ✅ 1C：t-shift 兩方向皆未達 −1.3（up +5.72 因 bb4 +35%；inv −0.33）→ 關閉；符號與影像直覺相反。
- ✅ 2a：ρ=+0.40 兩 circuit → pre-leg 選候選**不可行**；bigblue4 best-of-4（全 legalize）**−6.2%**（n=1）。
- ❌ 2b：無效 — floor 套錯（pre-leg 值 0.80–0.95 永遠過不了 0.97）→ 從未以 HPWL 選。我的設計錯誤。
- ⚖ 2c：best-of-4（全 legalize）paired Δavg6 **−1.02 ± 1.22（t=−1.44）** → 未達 −1.3；方向對、power 不足（要 n≈6）。
- ❌ 3E deep-K：最佳 −2.3%（n=1），K=150 +7~15% → 關閉。 ❌ 3F post-hoc 平均：兩 window 都更差 → 關閉。
  ❌ 3H frame averaging：a1 −0.4% / bb4 +2.4% → 關閉。
- ✅ 1A / 1B evidential（n=3）：FM/DDPM 2.22 ± 0.20；η=0/η=1 1.61 ± 0.02。
- ❌ 4X Run X 3 seeds：paired Δ **+0.36 ± 1.35** → 44.649 是 n=1 假警報，關閉。**四種 checkpoint 全在同一 avg6 帶內。**
- ❌ 4C DataAug Run C：avg7 45.171 vs Run X 44.649（+0.52）→ 關閉；D4 對稱性（訓練端+推論端）皆非瓶頸。
- ✅ **5B bigblue2 開 guidance：39.72（paper 38.8，+2.4%，單 seed 雜訊內）→ 復現 8/8；六個月的「輸 50%」是 24 GB 限制。**
- ✅ **5A：`lhw_24e5`（legalizer hpwl_weight ×2）三 seed 配對 −2.19 ± 0.72（−7.0%），同號，CI 不含 0 → 成立。**
  乾淨 stack 上唯一成立的正向結果，且在 legalizer 層 — 主張一的直接證據。其餘 10 組（含 lal_4e3 stage 2）null。
- **總結修正：不再是「零個正向結果」— 是「model / sampler 層零個，legalizer 層一個（校準性質）」。**
- ❌ 4B best-of-4 n=6：**−0.70 ± 1.10，CI [−1.85, +0.45]** → 未確立，關閉。
- **總結：乾淨 stack 上零個正向方法結果；負向結果乾淨且多數 evidential。**

---

## Phase 0 — 重錨

### 0a paper baseline（large-v2 + opt，seed 300）✅
| circuit | 新 stack | 舊紀錄（2026-03）| paper | legality |
|---|---:|---:|---:|---:|
| adaptec1 | 9.03 | 10.22 | 9.19 | 0.974 |
| adaptec2 | 30.75 | 39.06 | 31.0 | 0.942 |
| adaptec3 | 55.98 | 62.14 | 54.4 | 0.997 |
| adaptec4 | 57.40 | 60.51 | 54.5 | 0.999 |
| bigblue1 | 2.63 | 2.69 | 2.64 | 0.996 |
| bigblue3 | 36.59 | 34.26 | 35.9 | 0.926 |
| bigblue4 | 129.53 | 131.96 | 140.6 | 0.990 |
| **avg7** | **45.987** | 48.691 | 46.89 | |

判讀：舊 3 月 run 的目錄已被 eval 碰撞覆蓋、config 不可驗、runtime 慢 4.5×、adaptec2 legality 0.93 —
是壞掉的 run，不是 stack 漂移。**所有「−X% vs 48.691」的既有說法灌水約 2.7 單位。**
gen time 25 min（5090）。Run: `base_cu128_s300_part{1,2}`。

### 0b Run F（ablation_10k + opt，seed 300）✅ — gate 觸發
| circuit | 舊 stack | 新 stack | Δ | 新 baseline | 微調效果 |
|---|---:|---:|---:|---:|---:|
| adaptec1 | 9.44 | 9.09 | −0.35 | 9.03 | +0.7% |
| adaptec2 | 32.23 | 31.34 | −0.90 | 30.75 | +1.9% |
| adaptec3 | 54.02 | 51.86 | −2.16 | 55.98 | −7.4% |
| adaptec4 | 55.93 | 52.75 | −3.19 | 57.40 | −8.1% |
| bigblue1 | 2.74 | 2.60 | −0.13 | 2.63 | −0.8% |
| bigblue3 | 30.61 | 39.46 | **+8.85** | 36.59 | +7.9% |
| bigblue4 | 125.28 | 132.80 | **+7.52** | 129.53 | +2.5% |
| **avg7** | 44.321 | **45.700** | **+1.379** | 45.987 | **−0.6%** |

**預登記判定**：|Δ| = 1.379 ≥ 1.3 → 舊 stack 數字作廢（不可與新 stack 混比）；依計畫插入 re-anchor
（Run F s301/302、SVDD_layered s300–302）。
位移幾乎全在 bigblue3 / bigblue4 — 兩個 σ 最大的 circuit（5.3 / 4.4）；換 GPU/torch 等同換 random
stream，這是 seed 量級的變動而非系統性偏移（adaptec1–4、bigblue1 都在 ±1 內且方向為負）。
**同 seed 同 stack 的微調效果 Δ = −0.29，在 1.3 雜訊帶內（n=1）。** 專案標題「微調贏 9.6%」是對
壞掉的 baseline 量的。待 re-anchor 三 seed 配對檢定。Run: `runF_cu128_s300_part{1,2}`。

### 0c baseline seeds 300 / 301 / 302 ✅ — 同 stack 雜訊校準
| seed | adaptec1 | adaptec2 | adaptec3 | adaptec4 | bigblue1 | bigblue3 | **avg6** |
|---|---:|---:|---:|---:|---:|---:|---:|
| 300 | 9.03 | 30.75 | 55.98 | 57.40 | 2.63 | 36.59 | 32.064 |
| 301 | 9.13 | 32.08 | 52.85 | 57.35 | 2.61 | 32.76 | 31.130 |
| 302 | 9.09 | 30.24 | 54.04 | 51.97 | 2.68 | 34.65 | 30.443 |
| **mean ± sd** | 9.09 ± 0.05 | 31.02 ± 0.95 | 54.29 ± 1.58 | 55.57 ± 3.12 | 2.64 ± 0.03 | 34.67 ± 1.91 | **31.212 ± 0.813** |

**校準**：avg6 的 across-seed sd = 0.813 → **2σ = 1.63**，比預登記的 1.3 寬（1.3 來自舊 stack SVDD 的
avg7 σ=0.654）。預登記門檻不事後改；以下配對檢定同時報 paired t 與 |Δ| vs 1.3。
per-circuit 雜訊：adaptec4 5.6%、bigblue3 5.5%（單 seed 的 ±8% 逐 circuit 主張 ≈ 1.5σ）；
adaptec1 / bigblue1 < 1.5%（這兩個 circuit 的小差距反而可信）。
bigblue4 維持 seed 300（佔 eval 時間 78%）。Runs: `base_cu128_s{300,301,302}_*`。

### re-anchor：Run F ×3、SVDD_layered ×3 ✅ — **兩個標題主張都不成立**
| seed | Run F avg6 | SVDD avg6 | baseline avg6 | Δ FT | Δ SVDD |
|---|---:|---:|---:|---:|---:|
| 300 | 31.184 | 30.521 | 32.064 | −0.88 | −1.54 |
| 301 | 31.327 | 32.117 | 31.130 | +0.20 | +0.99 |
| 302 | 31.131 | 30.473 | 30.443 | +0.69 | +0.03 |
| **mean ± sd** | 31.214 ± 0.10 | 31.037 ± 0.94 | 31.212 ± 0.81 | **+0.002 ± 0.80 (t=0.00)** | **−0.175 ± 1.28 (t=−0.24)** |

avg7（seed 300 only）：Run F 45.700、SVDD 44.166、baseline 45.987；bigblue4：132.80 / 126.03 / 129.53。

**預登記判定**（|Δ| vs 1.3，並報 paired t）：
- **微調（supervised 10k on v1.61）：Δ = +0.002，在雜訊帶內。** 專案 4–9 月「微調贏 6–9%」的主張
  完全來自壞掉的 48.691 baseline。三個 seed 的 avg6 幾乎相同（sd 0.10）— 微調讓輸出更穩定，但沒有更好。
- **SVDD_layered 搜尋：Δ = −0.175，在雜訊帶內。** 「−4.36% vs paper」來自兩個效應疊加：(i) 對壞 baseline
  比較；(ii) paper 自己的 checkpoint 在我們 stack 上就是 45.99，比它自己發表的 46.89 好 2% — 所以任何
  在這裡跑的東西都「贏 paper」2%。seed 300 的 −1.54 在 301/302 反向。
- **Power**：paired sd 0.8–1.3、n=3 → 能排除的效應約 ≥1.5 單位；不能排除 0.5 單位的小效應。
  但先前宣稱的效應是 2–4 單位，已被排除。
- CoDe / TDS 與 SVDD 統計上等價（舊 stack），推定同樣為 null；不另花 GPU 重跑。

**後果**：專案至今唯一站得住的正向結果是 (a) paper checkpoint 復現 45.99（贏已發表值 2%，seed/硬體
差異），(b) 三個獨立實驗證明的「auxiliary HPWL/legality loss 冗餘」負向結果，(c) FM 在 size-OOD 下崩潰
（1A）。**Best-of-N（Phase 2）是剩下唯一有 within-protocol 量測增益的槓桿**（舊 stack best-of-3 −1.7）。
Runs: `runF_cu128_s{301,302}_*`、`svdd_cu128_s{300,301,302}_*`。

---

## Phase 1 — 診斷

### 1A unguided control（seed 300）✅ — FM 維持關閉
| | adaptec1 | bigblue4 | bb4 pre-legalization | bb4 legality |
|---|---:|---:|---:|---:|
| DDPM + opt（0a）| 9.03 | 129.53 | — | 0.990 |
| **DDPM, no guidance** | 10.65 | **269.57** | 654 | 0.986 |
| FM + opt, T=50（舊 stack）| 9.28 | 267.81 | — | 0.987 |
| **FM, no guidance, T=50** | 11.47 | **660.84** | 1,962 | 0.978 |

**預登記判定**：bigblue4 FM-none / DDPM-none = **2.45 ≥ 1.5** → FM 的失敗在 objective/sampler，
不是 guidance 交互；FM 家族維持關閉。`flowmatch_report_1.md` §3.2 的「no-guidance 泛化就已輸」
首次有實驗支撐（該 report 寫作時磁碟上零個 no-guidance FM run）。
附帶觀察：(i) pre-legalization 差距 3×（654 vs 1,962）— 損害在 raw 生成，legalizer 只是壓縮；
(ii) **DDPM 無 guidance 的 bigblue4 (269.6) ≈ FM 有 guidance (267.8)** — 所謂「DDPM OOD 泛化好」
實際上是「DDPM + opt guidance」泛化好，raw model 兩者都弱。這對 1B/1C 的判讀很重要：
guidance 在 OOD circuit 上貢獻約 −52%。
n=1；evidential 升級用 bigblue4 seed 301/302（`scripts/run_followup_1A_seeds.sh`，排 Phase 2 後）。
Runs: `diag1A_{ddpm,fm}_none_{a1,bb4}`。

### 1B stochasticity dial × steps（seed 300，large-v2）✅ — 確定性 sampler 在 OOD 崩潰
| cell | adaptec1 | Δ | bigblue4 | Δ | bb4 pre-leg | legality | gen s |
|---|---:|---:|---:|---:|---:|---:|---:|
| opt η=1.0 T=1000（0a）| 9.03 | — | 129.53 | — | — | 0.990 | 1013 |
| opt η=0.5 T=1000 | 9.21 | +2.0% | 128.52 | −0.8% | 77.2 | 0.989 | 1009 |
| opt η=0.0 T=1000 | 9.34 | +3.4% | **123.22** | **−4.9%** | 77.5 | 0.978 | 1009 |
| opt η=1.0 T=100 | 9.29 | +2.8% | 138.32 | +6.8% | 75.6 | 0.992 | 573 |
| opt η=0.5 T=100 | 9.29 | +2.8% | 144.29 | +11.4% | 76.1 | 0.993 | 592 |
| opt η=0.0 T=100 | 9.52 | +5.3% | 147.66 | +14.0% | 74.3 | 0.990 | 573 |
| **none η=1.0 T=1000（1A）** | 10.65 | — | 269.57 | — | 654 | 0.986 | 557 |
| **none η=0.0 T=1000** | 10.17 | −4.5% | **439.18** | **+62.9%** | 889 | 0.991 | 547 |

**預登記判定（unguided bigblue4，η=0 vs η=1）**：比值 **1.63 ≥ 1.30 → 確立**：deterministic sampler
（DDIM）在 size-OOD circuit 崩潰，在 in-distribution 的 adaptec1 反而略好（−4.5%）。
與 1A 合併解讀 FM 為何輸：**sampler 的確定性本身在 bigblue4 就值 +63%；FM 的 velocity objective
再讓 raw 輸出差 3×**。兩者都有貢獻，前者是主因（FM-opt 267.8 ≈ DDPM-none 269.6 ≈ 0.6 × DDPM-none-η0 439）。
few-step / drifting 家族**正式關閉（evidential；n=1，seed 301/302 已排在 3H 後）**。

附帶結論：
1. **有 guidance 時 η=0 在 bigblue4 反而 −4.9%、adaptec1 +3.4%**（adaptec1 σ=0.5% → 此差距可信；
   bb4 σ~3–4% → ~1.3σ）。guidance 把 OOD 崩潰從 439 救回 123 — 這個系統的 OOD 穩健性來自
   **stochasticity + guidance**，不在 model 本身。η 是一個可調的 in-distribution / OOD trade-off。
2. **T=100 在每個 η 下 bigblue4 都 +7~14%**，且只省 43% wall-clock（legalization 20k steps 佔大頭）。
   Plan §1B 的「T=100 < +5% → Phase 2 用 T=100 draft」分支**關閉**；Phase 2 draft 維持 T=1000。
3. 確定性 sampler 的 legality 較高（0.991 vs 0.986）但 HPWL 遠差 — legalizer 修得了 overlap，修不了
   拓撲上錯的擺放。
Runs: `diag1B_opt_eta{00,05,10}_T{1000,100}_{a1,bb4}`、`diag1B_none_eta00_T1000_{a1,bb4}`。

### 1C log-SNR shift by V（seed 300，large-v2 + opt，7 circuits）✅ — 兩個方向都關閉
| grid warp | avg7 | Δ vs 0a | 逐 circuit Δ |
|---|---:|---:|---|
| s = √(V/400)（大 circuit 往高噪聲；SD3 直覺）| 51.705 | **+5.72** | a1 +2%, a2 +3%, a3 −1%, a4 −5%, bb1 +3%, bb3 −9%, **bb4 +35%**（s=4.52）|
| s = √(400/V)（反向）| 45.656 | −0.33 | **a1 +15%, a2 +19%**, a3 +1%, a4 −4%, bb1 +4%, bb3 −5%, bb4 −5% |

**預登記判定**（Δ ≤ −1.3 → 補 seed）：兩者皆未達 → **關閉**。
觀察（n=1，per-circuit 主張依規則 6 僅為假說）：符號與影像領域的 SD3/simple-diffusion 直覺**相反** —
把大 circuit 的 grid 往高噪聲移是災難（bb4 +35%），往低噪聲移對大 circuit 略好（bb3/bb4 −5%，~1.5σ）
但重傷 in-distribution 的小 circuit（+15~19%）。這與 skeptic 的警告一致：Chen 的冗餘論證靠影像的
空間平滑性，netlist placement 沒有這個性質。若日後要追，是 per-circuit-size 的 s 曲線而非單一公式，
成本不值得。Runs: `diag1C_tshift_{up,inv}_part{1,2}`。

---

## Phase 2 — Best-of-N
### 2a rank-correlation（4 個候選全部 legalize，seed 300）✅ — gate **失敗**，便宜 draft 選擇死亡
| circuit | pre-leg HPWL（4 候選）| post-leg HPWL | Spearman ρ | argmin pre / post | 最終（best-of-4）| vs 0a |
|---|---|---|---:|---|---:|---:|
| adaptec1 | 7.35, 7.36, 7.69, 7.48 | 9.15, 9.24, 9.92, 9.14 | **+0.40** | 0 / 3 ≠ | 9.14 | +1.2% |
| bigblue4 | 81.5, 82.5, 79.3, 79.4 | 121.5, 131.7, 125.7, 131.0 | **+0.40** | 2 / 0 ≠ | **121.53** | **−6.2%** |

**預登記判定**（ρ ≥ 0.7 兩者皆須）：兩個 circuit 都只有 +0.40，且 pre-leg 的最佳候選在兩個 circuit 上都
**不是** post-leg 的最佳 → **pre-legalization HPWL 不能用來選候選**；任何 best-of-N 都必須把每個候選
legalize（N× legalization 成本）。survey §4.2 的 draft-and-refine 想法**關閉**。
附帶：這兩個 cell 本身就是有效的 best-of-4（全 legalize、以 post-leg HPWL 選）：bigblue4 −6.2%（n=1，
bb4 σ~3–4% → ~1.8σ），adaptec1 +1.2%（σ 0.5% 的 circuit 沒東西可撿）。4 個 legalized 候選在 bigblue4 的
範圍 121.5–131.7（8%）— 單 seed 內的候選變異**大於** across-seed 變異，這就是 best-of-N 要撿的東西。

### 2b 協定 run（N=4，pre-leg 選擇，floor 0.97）✅ — **無效（計畫設計錯誤）**
paired Δavg6 = +0.632 ± 2.371（t=0.46，n=3）— 但這個數字不能用：
**legality floor 0.97 是 post-legalization 的門檻，我把它套在 pre-legalization 值上。** 實際 pre-leg legality
在所有 circuit 都是 0.80–0.95（overlap 正是 legalization 要修的東西），所以 0/4 候選過門檻，每一列都落入
fallback「取 legality 最高者」— **從頭到尾沒有用 HPWL 選過**（7 個 circuit 裡 6 個 chosen ≠ argmin HPWL）。
這是我的預登記錯誤，不是程式錯誤；程式已補一個 loud warning + 純 HPWL fallback，避免再靜默發生。
為完整記錄（**不可用**）：bon2b avg6 = 30.090 (s300, avg7 44.817) / 33.794 (s301) / 31.650 (s302)。
Runs: `bon2a_legall_{a1,bb4}`、`bon2b_s300`、`bon2b_s301`、`bon2b_s302`（後三者作廢）。

### 2c 修正協定（N=4，**全部 legalize**，post-leg HPWL 選）✅ — **方向正確、n=3 power 不足**
| seed | baseline avg6 | best-of-4 avg6 | Δ | 逐 circuit Δ |
|---|---:|---:|---:|---|
| 300 | 32.064 | 30.124 | **−1.939** | a1 +1%, a2 −1%, a3 −4%, a4 −7%, bb1 0%, bb3 **−14%** |
| 301 | 31.130 | 29.646 | **−1.484** | a1 +1%, a2 **−13%**, a3 0%, a4 −8%, bb1 +1%, bb3 0% |
| 302 | 30.443 | 30.810 | +0.367 | a1 −1%, a2 −6%, a3 −3%, a4 +7%, bb1 −4%, bb3 +7% |
| **paired** | | | **−1.019 ± 1.222（t = −1.44, n=3）** | bigblue4（seed 300，2a cell）**−6.2%** |

3.67/4 候選過 post-leg legality floor 0.97 → 機制如設計運作。chosen idx 分佈 [6,2,3,7]（不是永遠選第一個）。
**預登記判定**（Δavg6 ≤ −1.3 且 t 顯著）：mean −1.02 未達 −1.3，t=−1.44（df=2，p≈0.29）→ **未確立**。
但誠實的補充：這是整輪在乾淨 stack 上**唯一**點估計方向與預期一致（min-of-4 ≈ mean − 1σ）、且幅度
最大的方法（−3.3%）；sd 1.22 意味著要 n≈6–8 才能在 2σ 解析 1.0 的效應。增益集中在高 σ circuit
（bb3 −14%、a2 −13%、a4 −7%）— 正是 noise harvesting，須報 legality floor（已報）與成本（4× sampling
+ 4× legalization ≈ 每 seed 31 min vs 14 min）。**不關閉、不宣稱；補 seeds 303–305 可解析（~1.5 h）。**
Runs: `bon2c_s{300,301,302}_*`。

---

## Phase 3 — Skeptic 存活的三個零訓練測試（seed 300）

### 3E deep guidance K（MacroDiff+ 式）✅ — **關閉**
| cell | bigblue4 | Δ | legality | gen s（×0a）|
|---|---:|---:|---:|---:|
| K=20 acf=0.5（0a）| 129.53 | — | 0.990 | 1013 |
| K=60 acf=0.5 | 126.55 | −2.3% | 0.990 | 2434（2.4×）|
| K=60 acf=1.0 | 133.67 | +3.2% | 0.991 | 2855（2.8×）|
| K=150 acf=0.5 | 138.04 | +6.6% | 0.991 | 5483（5.4×）|
| K=150 acf=1.0 | 140.10 | +8.2% | 0.989 | 5636（5.6×）|
| adaptec1 K=150 acf=0.5 | 10.40 | **+15.2%** | 0.993 | 896（9.6×）|

**預登記判定**（任一 bb4 cell ≥ 5% 且 legality ≥ 0.98）：最好的只有 −2.3%（n=1，< bb4 σ）→ **關閉**。
更多 inner step = over-guidance，與 1B 的 T curve、FM 報告 §3.1 同型；skeptic 的 hidden cost（α dual
ascent 在 K 迴圈內提早飽和）成立。MacroDiff+ 的 in-distribution 增益不轉移到 zero-shot 設定。
Runs: `deepK{60,150}_acf{05,10}_bb4`、`deepK150_acf05_a1`。

### 3F post-hoc 權重平均（EMA 零訓練代理）✅ — **關閉配方方向**
| checkpoint | avg7 | Δ vs Run X（同 stack）|
|---|---:|---:|
| Run X 500k latest（同 stack 重量）| **44.649** | —（舊 stack 紀錄 45.053）|
| avg 250k–500k uniform | 46.497 | +1.85 |
| avg 400k–500k uniform | 45.569 | +0.92 |

**預登記判定**（≤ −1.3 → 真 EMA 訓練值得）：兩個 window 都**更差** → **關閉**。與建 ckpt 的 agent 警告一致：
500k 時模型仍在移動（相鄰 snapshot 相距 rel-L2 0.18–0.25），平均落在 basin 外。
**附帶但重要**：Run X（from-scratch，合成資料，1/6 paper 步數）在新 stack 上 **44.649，是本輪最佳 avg7**，
比 paper ckpt 45.987 好 1.34（n=1，剛好在 bar 上）。逐 circuit：
| | adaptec1 | adaptec2 | adaptec3 | adaptec4 | bigblue1 | bigblue3 | bigblue4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Run X | 9.41 | 28.77 | 59.62 | 50.58 | 2.67 | 35.52 | 125.98 |
| paper ckpt | 9.03 | 30.75 | 55.98 | 57.40 | 2.63 | 36.59 | 129.53 |
| Δ | +4% | -6% | +6% | -12% | +2% | -3% | -3% |

**未被本輪任何預登記判定涵蓋；建議補 seeds 301/302（6 circuit，~25 min）。**
Runs: `runX_cu128_s300_part{1,2}`、`runXavg_{250k_500k,400k_500k}_part{1,2}`。

### 3H frame averaging（D4 exact test-time symmetrization）✅ — **關閉**
| | adaptec1 | Δ | bigblue4 | Δ | gen s |
|---|---:|---:|---:|---:|---:|
| 0a | 9.03 | — | 129.53 | — | 93 / 1013 |
| frame_average | 9.00 | −0.4% | 132.65 | +2.4% | 258（2.8×）/ 1182（1.2×）|

**預登記判定**（bb4 ≥ 5% 或 a1 ≥ 3%）：皆未達 → **關閉**。精確 D4 等變性不改善結果 — model 的
非等變性不是限制因素（訓練端 D4 augmentation 的 Run C 仍未 eval，但這個結果讓它的優先度更低）。
Runs: `frameavg_{a1,bb4}`。

### 1A / 1B evidential 升級（bigblue4，seeds 300/301/302）✅
| gate | seed 300 | 301 | 302 | mean ± sd | 判定 |
|---|---:|---:|---:|---:|---|
| 1A FM-none / DDPM-none | 2.45 | 2.16 | 2.06 | **2.22 ± 0.20** | ≥ 1.5 → **確認（n=3）** |
| 1B η=0 / η=1（unguided）| 1.63 | 1.58 | 1.61 | **1.61 ± 0.02** | ≥ 1.3 → **確認（n=3）** |

兩條都升級為 **evidential** 結論。1B 的 sd 0.02 異常緊 — 確定性採樣在 OOD 的損害是系統性的，不是 seed 運氣。
Runs: `diag1A_{ddpm,fm}_none_bb4_s30{1,2}`、`diag1B_none_eta00_T1000_bb4_s30{1,2}`。

---

## Phase 4 — 收尾 seeds（2026-09-15，預登記見 plan Phase 4）

### 4X Run X（from-scratch 500k）seeds 300–302 ✅ — **n=1 假警報，關閉**
| seed | baseline avg6 | Run X avg6 | Δ | 逐 circuit Δ |
|---|---:|---:|---:|---|
| 300 | 32.064 | 31.094 | −0.969 | a1 +4%, a2 −6%, a3 +6%, a4 −12%, bb1 +2%, bb3 −3% |
| 301 | 31.130 | 32.851 | **+1.721** | a1 +1%, **a2 +19%**, a3 +11%, a4 −1%, bb1 +7%, bb3 −5% |
| 302 | 30.443 | 30.757 | +0.314 | a1 0%, a2 0%, a3 +2%, a4 +5%, bb1 −2%, bb3 −4% |
| **paired** | | | **+0.355 ± 1.346（t=0.46, 95% CI [−2.99, +3.70]）** | 同號：否 |

**預登記判定**（≤ −1.3 且同號）：**FAIL** → 44.649 是 seed 300 的假警報，記錄並關閉。
Run X 與 paper checkpoint 統計上無差別（也與 Run F、SVDD 無差別）— 在乾淨 stack 上，
**四個不同來源的 checkpoint / sampler（paper、微調、from-scratch、SVDD）全部落在同一個 ~31.2 ± 0.8 的
avg6 帶內**。這本身是本輪最強的單一結論：目前這條 pipeline 的品質由 opt guidance + legalizer 決定，
model 來源幾乎不影響。Runs: `runX_cu128_s{300,301,302}_*`。
### 4C DataAug Run C（dihedral + edge_dropout 0.1，from-scratch 500k）seed 300 ✅ — **關閉**
| | adaptec1 | adaptec2 | adaptec3 | adaptec4 | bigblue1 | bigblue3 | bigblue4 | avg7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Run C | 9.88 | 28.37 | 55.55 | 57.13 | 2.62 | 33.86 | 128.77 | **45.171** |
| Run X（同 stack，同預算無 augmentation）| 9.41 | 28.77 | 59.62 | 50.58 | 2.67 | 35.52 | 125.98 | 44.649 |
| Δ | +5% | -1% | -7% | +13% | -2% | -5% | +2% | +0.522 |

**預登記判定**（≤ −1.3 → 補 seeds）：**FAIL** → DataAug 方向關閉。這個 checkpoint 2026-07-08 訓練完後
擱置兩個月才 eval；結果與 Run X 無差別（且 Run X 本身與 baseline 無差別，見 4X）。訓練端 D4
augmentation 與推論端 frame averaging（3H）一起說明：**D4 對稱性不是這條 pipeline 的瓶頸**。
Run B（僅 dihedral）從未重跑，故無法歸因 dropout；既然合併效應為零，歸因問題不再重要。
Runs: `runC_cu128_s300_part{1,2}`。

### 4B best-of-4（全 legalize）+ baseline，seeds 300–305 ✅ — **n=6 未確立，關閉**
| seed | baseline avg6 | best-of-4 avg6 | Δ | 逐 circuit Δ |
|---|---:|---:|---:|---|
| 300 | 32.064 | 30.124 | −1.939 | a1 +1, a2 −1, a3 −4, a4 −7, bb1 0, **bb3 −14** |
| 301 | 31.130 | 29.646 | −1.484 | a1 +1, **a2 −13**, a3 0, a4 −8, bb1 +1, bb3 0 |
| 302 | 30.443 | 30.810 | +0.367 | a1 −1, a2 −6, a3 −3, a4 +7, bb1 −4, bb3 +7 |
| 303 | 31.313 | 29.708 | −1.605 | a1 −9, **a2 −14**, a3 −4, a4 +4, bb1 −5, **bb3 −10** |
| 304 | 29.432 | 29.916 | +0.484 | a1 +5, **a2 +12**, a3 0, a4 −1, bb1 −6, bb3 0 |
| 305 | 29.937 | 29.933 | −0.005 | a1 −4, a2 +3, a3 0, a4 −3, bb1 −8, bb3 +4 |
| **paired n=6** | | | **−0.697 ± 1.095；95% CI [−1.85, +0.45]；t = −1.56** | 同號：否（3 負 / 3 零或正）|

**預登記判定**（mean ≤ −1.0 且 CI 不含 0 → 可報告）：**未達 → 關閉**。
機制面：36 個 cell、選中的候選均勻分佈 [7, 11, 7, 11]、平均 3.75/4 過 legality floor、cell 內候選
spread 8.4%（中位 6.7%）— 搜尋如設計運作。但 min-of-4 只買到每個 circuit 分佈的 ~1σ，而那個分佈本身
跨 seed 就有 ±5–14% 的擺動（a2、bb3 在六個 seed 裡正負都出現），所以 avg6 上的淨增益 ~0.7、
要 n ≈ 20 才能在 2σ 解析。**成本 4× sampling + 4× legalization，換一個解析不出來的 −2%：不值得作為
協定報告。** 若日後 seed 數夠多可重開，但本輪關閉。Runs: `bon2c_s{300..305}_*`、`base_cu128_s{303,304,305}_*`。

### 本輪總結：在乾淨 stack 上，沒有任何正向方法結果存活
| 軸 | 測試 | n | Δ（avg6 或 ratio）| 判定 |
|---|---|---:|---|---|
| 微調（supervised 10k）| Run F vs paper ckpt | 3 配對 | +0.00 ± 0.80 | null |
| Inference-time search | SVDD_layered vs paper ckpt | 3 配對 | −0.18 ± 1.28 | null |
| From-scratch 訓練 | Run X vs paper ckpt | 3 配對 | +0.36 ± 1.35 | null |
| 訓練端 D4 augmentation | Run C vs Run X | 1 | +0.52 (avg7) | null |
| Sampler 確定性 | η=0 vs η=1（unguided bb4）| 3 | ratio 1.61 ± 0.02 | **有害**（evidential）|
| FM objective | FM vs DDPM（unguided bb4）| 3 | ratio 2.22 ± 0.20 | **有害**（evidential）|
| 步數 | T=100 vs 1000 | 1 | bb4 +7~14% | 有害 |
| Grid shift by V | 兩方向 | 1 | +5.72 / −0.33 | null/有害 |
| Guidance 深度 | K=60/150 | 1 | −2.3% ~ +15% | null/有害 |
| 權重平均 | 兩 window vs Run X | 1 | +0.9 / +1.85 | 有害 |
| 推論端 D4 | frame averaging | 1 | −0.4% / +2.4% | null |
| Best-of-4 | 全 legalize vs baseline | 6 配對 | −0.70，CI 含 0 | 未確立 |

**四種 checkpoint 來源與所有 sampler 變體都落在 paper checkpoint + opt guidance + legalizer 的雜訊帶內。**
（Phase 5 後更新）唯一能移動數字的是 legalizer 的 HPWL 權重（5A，−7.0%，n=3）；bigblue2 開 guidance 後復現 8/8。


## Phase 5 — 結案前的兩個實驗（2026-09-25/26，預登記見 plan Phase 5）

### 5B bigblue2 開 guidance（32.6 GB）✅ — **跑得完；復現補成 8/8**
| | bigblue2 HPWL | legality | 備註 |
|---|---:|---:|---|
| 舊環境，無 guidance（4 seeds）| 56.9 – 65.5 | ~1.00 | 24 GB 放不下 V×V |
| **新環境，開 guidance（seed 300）** | **39.72** | **1.000** | generation 7,178 s（含 20 步 guidance × 1000），無 OOM |
| paper（seed 400）| 38.8 | — | |

**預登記判定**：跑得完 → 記錄。39.72 vs 38.8 = +2.4%，在單 seed 雜訊內（paper 是另一個 seed）。
**結論：bigblue2「輸 paper 50%」六個月來完全是 24 GB 硬體限制，不是方法問題。** 8-circuit 平均（seed 300）
= 45.20 vs paper 發表的 8-circuit 45.88（−1.5%）。復現至此 8/8 都在 paper 的雜訊帶內。
峰值記憶體未量測（下次跑加 `nvidia-smi` 取樣）。Run: `bb2_guided_s300`。

### 5A guidance / legalizer 超參數篩選（seed 300，6 便宜 circuit）✅ stage 1 ✅ stage 2
| config | 改了什麼 | avg6 | Δ vs baseline s300 (32.064) | min legality | 篩選 |
|---|---|---:|---:|---:|---|
| lhw_24e5 | legalization.hpwl_weight 12e-5 → 24e-5 | **29.69** | **−7.4%** | 0.936（adaptec2；baseline 0.942）| **PASS** |
| lal_4e3 | legalization.alpha_lr 8e-3 → 4e-3 | 30.50 | −4.9% | 0.973 | 差 0.04 未過；依 next_1「n=1 ≥1.3 自動補 seed」規則進 stage 2 |
| gdr_16e3 | model.grad_descent_rate 8e-3 → 16e-3 | 30.79 | −4.0% | 0.974 | — |
| lpt_1e3 | model.legality_potential_target 1e-4 → 1e-3 | 30.96 | −3.4% | 0.955 | — |
| hgw_32e4 | model.hpwl_guidance_weight 16e-4 → 32e-4 | 31.40 | −2.1% | 0.978 | — |
| gdr_4e3 / lpt_0 | ×0.5 / 0 | 31.52 / 31.51 | −1.7% | 0.967 / 0.976 | — |
| acf_10 / hgw_8e4 | acf 1.0 / hgw ×0.5 | 31.89 / 31.94 | −0.5% / −0.4% | 0.969 / 0.926 | — |
| lal_16e3 / lhw_6e5 | ×2 / ×0.5 | 32.53 / 32.63 | +1.5% / +1.8% | 0.973 / 0.985 | — |

篩選規則（預登記）：avg6 ≤ 0.95 × 32.064 = 30.46 且各 circuit legality ≥ 0.93。11 組裡 1 組通過。
注意：seed 300 是六個 baseline seed 裡最差的（avg6 32.06 vs 六 seed 平均 30.72），單 seed 的 −7.4% 有一部分是
回歸均值；**stage 2（seeds 301/302 配對）才算數**，判定門檻：三 seed 配對 ≤ −5% 且同號。
lhw_24e5 的增益集中在 adaptec4 −15%、adaptec3 −8%（legality 0.996 / 0.992，不是用合法度換的）。
兩個 legalizer 的旋鈕（hpwl_weight ↑、alpha_lr ↓）方向一致：都是讓 legalizer 更偏向拉 HPWL、慢一點壓合法度。
22 cells 零失敗。Runs: `hp_<config>_{part1,bb3}`。

**Stage 2（seeds 301/302 配對，判定：三 seed 配對 ≤ −5% 且同號）**
| config | seed 300 | 301 | 302 | paired Δavg6 | 判定 |
|---|---:|---:|---:|---|---|
| **lhw_24e5**（legalizer hpwl_weight ×2）| −7.4% | −9.0% | −4.6% | **−2.19 ± 0.72（−7.0%），t = -5.2，95% CI [-4.00, -0.38]** | **成立** |
| lal_4e3（legalizer alpha_lr ×0.5）| −4.9% | +2.7% | −1.0% | −0.34 ± 1.4（−1.1%），不同號 | 未確立 |

lhw_24e5 逐 circuit（三 seed 平均）：adaptec2 −10%、adaptec4 −11%、adaptec3 −5%、bigblue3 −3%、bigblue1 −2%、adaptec1 +1%。
legality：adaptec2 0.94–0.95（baseline 0.94–0.98），其餘 ≥ 0.98 — **不是用合法度換的**。
**這是整個專案在乾淨 stack 上第一個成立的正向結果，而且它動的是 legalizer（生成完之後的 20,000 步梯度下降），
不是 model、不是 sampler。** 這正是主張一（品質由 guidance + legalizer 決定）的直接檢驗：反向動這一層，數字跟著動了。
必須標明：paper 的 12e-5 是為 ISPD2005 調的，我們的 24e-5 是在同一個測試集上再調一步 — 這是 **校準發現**，不是方法貢獻；
它說明的是「原論文的 legalizer 把 HPWL 留在桌上」，而不是「我們有更好的方法」。
bigblue4 確認（post-hoc，非預登記，seed 300）：lhw_24e5 **126.68**（legality 0.975）vs baseline 129.53 → -2.2%。
7-circuit avg（seed 300）：**43.547** vs baseline 45.987（-5.3%）；paper 46.89。加 bigblue2（39.72，未用新權重）8-circuit ≈ 43.07 vs paper 45.88。
未做（可做的後續）：更大的 hpwl_weight（48e-5）、與 lal_4e3 合併、bigblue2 用新權重重跑（2 h）。
Runs: `hp_lhw_24e5_*`, `hp_lal_4e3_*`（seeds 300–302）。

---

## 附錄：流程事故
三個 ad-hoc 背景 waiter 用 `pgrep -f <pattern>` 等前一批，pattern 出現在自己的 cmdline 裡 →
一個永久死鎖（1A 從未啟動）、兩個提早觸發並同時搶 GPU（`diag1B_opt_eta00_T1000_a1` 與
`bon2a_legall_a1` 同時跑，timing 受污染但 HPWL 有效）。修正：佇列改為 `scripts/run_sampler_plan_1.sh`
（檔案、PID 等待、skip-if-done）。此事故已記入 PROGRESS.md 與 commit dd38c5b。
