# Sampler 診斷與 Best-of-N 報告（sampler_report_1）— 2026-09-14（進行中）

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
- ⏳ re-anchor（Run F ×3、SVDD ×3）、1B、1C、2a、2b、3E/3F/3H。

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

### re-anchor：Run F s301/302、SVDD_layered s300/301/302 ⏳

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

### 1B stochasticity dial × steps ⏳
先行資料（與 bon2a 同時跑，timing 受污染，HPWL 有效）：adaptec1 η=0（DDIM）= 9.34 vs η=1 = 9.03
（+3.4%，legality 0.968 vs 0.974）。決定性的是 bigblue4，待佇列。
（填表：η∈{0,0.5,1} × T∈{1000,100} × {a1,bb4}，opt；η∈{0,1} × T=1000，none。判定見 plan §1B。）

### 1C log-SNR shift by V ⏳
（填表：s=√(V/400) 與 s=√(400/V)，7 circuits，對照 0a；判定 ≥1.3 → 跑 s301/302。）

---

## Phase 2 — Best-of-N ⏳
### 2a rank-correlation（legalize all 4，adaptec1 + bigblue4）
（填：pre-leg vs post-leg 排序的 Spearman ρ；ρ ≥ 0.7 → draft 可用 pre-leg 分數。）
### 2b 協定 run（N=4，6 circuit × 3 seeds + bb4 × 1）
（填：best-of-4 mean ± sd，legality floor 0.97 下的通過率；對照 0c 三 seed baseline。）

---

## 附錄：流程事故
三個 ad-hoc 背景 waiter 用 `pgrep -f <pattern>` 等前一批，pattern 出現在自己的 cmdline 裡 →
一個永久死鎖（1A 從未啟動）、兩個提早觸發並同時搶 GPU（`diag1B_opt_eta00_T1000_a1` 與
`bon2a_legall_a1` 同時跑，timing 受污染但 HPWL 有效）。修正：佇列改為 `scripts/run_sampler_plan_1.sh`
（檔案、PID 等待、skip-if-done）。此事故已記入 PROGRESS.md 與 commit dd38c5b。
