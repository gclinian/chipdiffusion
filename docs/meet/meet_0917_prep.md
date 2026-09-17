# Meeting 準備 — 2026/09/17（進度、結果、下一步建議）

> 這是**會前準備**文件（會後的教授建議記錄另存 `meet_<MMDD>.md`）。
> 上次會議：2026/06/20。期間：Flow matching / DataAug / ISPD 極限（6–7 月）→ 兩個月中斷 →
> 9 月回歸：環境修復、方法 survey、49 個 eval cell 的 sampler 診斷（`docs/report/sampler_report_1.md`）。
> 所有數字 7-circuit avg HPWL ×10⁵（排除 bigblue2）；avg6 另排除 bigblue4。越低越好。

---

## 0. 三句話版本

1. **上次會議的四個方向都有了答案，全部關閉**：Flow matching 輸（且原因已在 n=3 下確立）、DataAug 無效、
   ISPD2005 極限分析顯示方法間差異已在雜訊帶內、EDA dataset 不值得做（見 §1）。
2. **專案過去五個月的兩個標題主張（微調 −6~9%、inference-time search −4.4%）不成立** — 它們是對一個壞掉的
   baseline 比較出來的假象。修好環境、重量 baseline、同 stack 三 seed 配對後，兩者的效應都是零（§2–3）。
3. **乾淨 stack 上的發現**：paper checkpoint + opt guidance + legalizer 就是天花板；換 model 來源（微調 / 從零訓練 /
   augmentation）、換 sampler（search / 確定性 / 步數 / 深 guidance / 對稱化 / best-of-N）**沒有一個**在統計上
   與它有差別。品質由 guidance + legalizer 決定，model 幾乎無關（§3）。建議轉向 §5。

---

## 1. 上次會議四個建議的執行結果

| 建議（06/20）| 做了什麼 | 結果 | 判定 |
|---|---|---|---|
| **1. Data augmentation** | 實作 D4 dihedral + edge dropout；從零訓練 500k（Run C）| Run C 45.17 vs 同預算無 aug 的 Run X 44.65（+0.5）；推論端 frame averaging 也 −0.4%/+2.4% | **關閉**：D4 對稱性在訓練端與推論端都不是瓶頸 |
| **2. Flow matching / Drifting** | 實作 `FlowMatchingModel`（CFM + Euler ODE）500k；survey drifting model | FM 最佳 70.0 vs DDPM 45.1；**unguided 對照（n=3）FM/DDPM = 2.22 ± 0.20；確定性 sampler 本身 η=0/η=1 = 1.61 ± 0.02** | **關閉（evidential）**：deterministic sampler 在 size-OOD 崩潰；FM objective 再壞 2×。Drifting model 是一步 kernel 生成器（何愷明 2602.04770），kernel 頻寬隨 V 飽和、零 OOD 證據 → 不做 |
| **3. ISPD2005 極限** | Oracle 分析（best-per-circuit 跨所有 run）+ 本輪同 stack 雜訊校準 | Oracle 41.84（legality 不乾淨）；同 stack across-seed sd = 0.81（avg6）→ **任何 < 1.5 的差距在 n=3 下不可解析**；所有方法落在 31.2 ± 0.8 帶內 | **教授的假說成立**：ISPD2005 macro-only 已擠不出方法差異，要看方法就得換 benchmark |
| **4. EDA-generated dataset** | 未執行（工程量最大）| Run X（合成資料從零訓練）與 paper ckpt 統計無差別 → 訓練資料來源不影響結果 | **建議不做**：換 label 來源不會改變由 guidance/legalizer 決定的結果 |

05/08 的兩個建議也已收斂：**從零訓練**（Run X，500k / 1.6M）= 與 paper ckpt 無差別；**RL/其他 guidance**（SVDD /
CoDe / TDS 三種 inference-time search）= 舊 stack 上看似 −4%，同 stack 配對後 −0.18 ± 1.28（null）。

---

## 2. 期間發現的根本問題（這是為什麼結論全變了）

### 2.1 環境：GPU 換成 RTX 5090，舊 torch 跑不了
`torch 2.2.1+cu121` 沒有 sm_120 kernel，`cuda.is_available()` 仍回 True，第一次 kernel launch 才炸。
建新 env `chipdiff-b`（torch 2.11 / cu128）。5090 讓 7-circuit eval 從 63 min 降到 25 min，**3-seed 規則變得付得起**。

### 2.2 Baseline 壞了五個月
| | 舊紀錄（2026-03）| 新 stack 重量 | paper 已發表 |
|---|---:|---:|---:|
| adaptec2 | 39.06（legality 0.93）| **30.75** | 31.0 |
| avg7 (seed 300) | **48.691** | **45.987** | 46.89 |

舊 run 的目錄被 eval 碰撞覆蓋、config 不可驗、runtime 慢 4.5× — 是壞掉的 run，不是 stack 差異。
後果：所有「−X% vs 我們的復現」灌水 2.7 單位；paper 自己的 checkpoint 在我們環境本來就贏已發表值 2%，
所以**任何東西都「贏 paper 2%」**。

### 2.3 換 stack = 換 random stream
微調 checkpoint 同 seed 跨 stack 差 +1.38（全在 σ 最大的 bigblue3/4）。舊 stack 與新 stack 數字不可混比；
本輪所有結論都在新 stack 上、同 seed 集合、配對重量。

---

## 3. 乾淨 stack 上的結果總表

Anchor：paper checkpoint（large-v2）+ opt guidance + opt-adam legalizer，**avg6 = 31.212 ± 0.813（3 seeds）**，avg7 45.987。

| 軸 | 方法 | n | Δ（配對 avg6 或比值）| 判定 |
|---|---|---:|---|---|
| Model：微調 | supervised 10k on v1.61 | 3 | **+0.00 ± 0.80** | null |
| Model：從零訓練 | Run X 500k（合成資料）| 3 | **+0.36 ± 1.35** | null（seed 300 的 44.65 是假警報）|
| Model：augmentation | Run C（D4 + dropout）| 1 | +0.52（avg7 vs Run X）| null |
| Sampler：search | SVDD-layered（K=4）| 3 | **−0.18 ± 1.28** | null |
| Sampler：確定性 | DDIM η=0（unguided bb4）| 3 | ratio **1.61 ± 0.02** | **有害** |
| Objective：FM | velocity regression（unguided bb4）| 3 | ratio **2.22 ± 0.20** | **有害** |
| Sampler：步數 | T=100 | 1 | bb4 +7~14% | 有害 |
| Sampler：grid shift | t-shift by V（兩方向）| 1 | +5.72 / −0.33 | null / 有害 |
| Guidance：深度 | K=60/150 | 1 | −2.3% ~ +15% | null / 有害 |
| Weights：平均 | post-hoc SWA 兩 window | 1 | +0.9 / +1.85 | 有害 |
| Sampler：對稱化 | frame averaging | 1 | −0.4% / +2.4% | null |
| Protocol：best-of-N | best-of-4，全 legalize | **6** | **−0.70 ± 1.10，95% CI [−1.85, +0.45]** | 未確立（機制運作、增益在雜訊內、成本 4×）|

**讀法**：四種 checkpoint 來源、八種 sampler/guidance 變體，沒有一個離開 anchor 的雜訊帶。
確定性 sampler 與 FM objective 是唯二**確立有害**的變因（n=3，sd 極小）— 這解釋了為什麼 few-step 生成範式
不適合這個任務：OOD 穩健性來自 **stochastic sampling + gradient guidance**，不在 model。

---

## 4. 對專案敘事的影響

**消失的**：「我們的方法贏 ChipDiffusion 4–9%」。
**站得住的**：
1. 復現：paper checkpoint 在現代硬體上 45.99，比已發表 46.89 好 2%（seed/硬體差異）。
2. 一組乾淨的負向結果，多數 n=3 evidential：auxiliary loss 冗餘（三個獨立實驗）、微調無效、search 無效、
   從零訓練無效、確定性 sampler 與 FM 在 size-OOD 崩潰（量化：+61% / +122%）。
3. **「Pipeline 品質由 guidance + legalizer 決定，model 幾乎無關」** — 這是本輪最強的單一發現，也是一個
   可檢驗的主張（§5 A 直接測它）。
4. 領域現況（survey）：OrderPlace（ICML'26）等構造式 wire-mask search 在 6 個小 circuit 上 29.98 vs
   ChipDiffusion 31.27；但**沒有任何方法報告完整 bigblue2 / bigblue4**（都只放 1024 個 module）。
   我們在 bigblue4（8,170 macros）的 129.5 沒有對手。

方法論產出（不是研究結果但值得提）：實驗 ledger（每個 run 自動記錄、metrics 逐字存 git）、六條判讀規則
（n=1 不關方向、baseline 必須同 stack、禁令分 evidential/prudential …）、預登記 plan → report → next 的流程。
這一輪 49 個 cell 零人工介入、零失敗。

---

## 5. 下一步建議（三條路，附成本）

### A. 轉軸：guidance / legalizer 本身（推薦，便宜）
57 個 guided run 從沒動過 `grad_descent_rate`、`alpha_critical_factor`、`legality_potential_target`、
legalization `hpwl_weight` / `alpha_lr`；本輪只掃了 K（有害）。既然 §3 說品質由這層決定，這是唯一還沒被
證明無效、且直接作用在決定品質的那一層的方向。**一輪 sweep ~3–4 GPU-h，3 seeds × 6 便宜 circuit**。
判定門檻預登記：配對 Δavg6 ≤ −1.3 才算。

### B. bigblue2 開 guidance（一次 eval，可能 OOM）
32.6 GB 現在邊緣可行。bigblue2 是唯一從未贏過 paper 的 circuit（我們 57–66 vs 38.8），也是唯一沒在乾淨
stack 測過的。成功 → 8/8 完整故事；OOM → 用已存在的 tiled legality potential 改 guidance 路徑（低風險）。

### C. 換 benchmark（教授 06/20 建議 3 的自然結論）
ISPD2005 macro-only 已證明擠不出方法差異。候選：IBM（repo 已有 `ibm.cluster512.v1` 但 eval 路徑未驗證）、
ICCAD04、或 mixed-size。這是下一個研究階段，不是這週的實驗。

### D. 論文（現在就能寫）
框架：**reproducibility + negative results + 「guidance/legalizer 決定品質」**。誠實、完整、有量化的機制解釋
（FM 為什麼輸、為什麼 few-step 不適合）。若 A 有正向結果就併入；若沒有，A 的 null 反而強化主張 3。

**我的建議順序**：A（本週，3–4 h）→ B（一次 eval）→ 同時開始寫 D → C 作為下一階段。
**不建議**：EDA-generated dataset（§1）、任何新的生成範式或 backbone（survey 28 個候選 skeptic 後全部 weak）、
更多 seeds 給 best-of-N（要 n≈20 才解析 −0.7，不值 4× 成本）。

---

## 6. 想請教教授的問題

1. **論文框架**：接受「reproducibility + 乾淨負向結果 + 機制發現」的定位嗎？目標 venue？
   （負向結果 n=3 evidential、有機制解釋，不是「我們試了沒用」。）
2. **換 benchmark**：若做 C，教授偏好 IBM / ICCAD04 / mixed-size 哪一個？考量是「能看出方法差異」而非「跟誰比」。
3. **06/20 四個方向與 05/08 兩個方向全部有答案且關閉** — 同意結案嗎？有沒有教授認為結論下太快的？
4. **A 的預登記門檻**：配對 Δavg6 ≤ −1.3（≈ 2σ）合理嗎？還是教授希望更嚴 / 更寬？

---

## 附錄：文件索引
- 本輪報告：`docs/report/sampler_report_1.md`（49 cells，每 phase 對照預登記判定）
- 回顧：`docs/next/sampler_next_1.md`（學到的 / 不要做（evidential vs prudential）/ 剩餘方向）
- 方法 survey：`docs/survey/method_survey_1.md`（drifting / FM 變體 / backbone / 領域 SOTA；28 候選 + skeptic）
- 回歸稽核：`docs/next/comeback_next_1.md`（發現壞 baseline、eval 碰撞、五個月未 commit 的那份）
- 現況數字：`STATUS.md`（自動產生，分新舊 stack）；規則：`CLAUDE.md` 判讀規則 1–7
