# 回歸回顧 (Comeback Review) — 2026-09-06

> 距上次動工 2026-07-08 約 2 個月。本文重建全專案狀態、修正若干既有結論、並排序下一步。
> 所有數字皆從 `docs/all_experiments_per_circuit.csv` 與 `logs/diffusion_debug/*/metrics.csv`
> 重算驗證，不採信報告內的轉述值。

指標約定：7-circuit avg HPWL（×10^5，排除 bigblue2），**越低越好**。
Paper Table 10 = 46.89；我們對 paper checkpoint 的復現 = 48.69（seed 300）。

---

## 0. 一句話現況

專案有**兩條各自有效、幅度相當的改進軸**——fine-tuning（44.01–44.32）與
inference-time particle search（44.85 ± 0.65, 3 seeds）——但**只被合併測試過一次、
單一 seed**，而那唯一一次的結果（Δ=+0.164）比該方法自己的 seed std（0.654）**小四倍**，
卻被寫成禁令關閉了整個方向。同時，**近 5 個月的程式與文件完全沒有 commit**，
且**環境已因 GPU 更換而無法執行任何 CUDA 運算**。

---

## 1. 阻斷問題：環境壞了（必須最先處理）

伺服器 GPU 已更換：

| | CLAUDE.md 記載 | 實際現況 |
|---|---|---|
| GPU | 24 GB | **RTX 5090, 32,607 MiB** |
| Compute capability | sm_86 / sm_89 | **sm_120 (Blackwell)** |

`chipdiff` env 為 `torch 2.2.1+cu121`，其 `arch_list` 只到 `sm_90`：

```
MATMUL FAILED: RuntimeError CUDA error: no kernel image is available for execution on the device
```

`torch.cuda.is_available()` **回傳 True**，所以這不會在啟動時報錯，而是在
**第一次 kernel launch 時才炸**，看起來像是實驗跑到一半神秘崩潰。
另一個 env `diffplace` (torch 2.6.0+cu124) 同樣不支援。

**修復範圍很小**：本專案**沒有安裝任何需編譯的 PyG extension**
（`torch-scatter` / `torch-sparse` / `torch-cluster` 皆未安裝，`diffusion/` 內零 import），
只有 `torch_geometric 2.4.0`（純 Python）+ `performer-pytorch`。
因此這是 pip 層級的 torch ≥ 2.7/cu128 升級，不是 C++ 重編。

**做法**：在**新 env**（例如 `chipdiff-b`）升級，保留 `chipdiff` 原狀；
升級後**必須先重跑一個已知結果（Run F，預期 44.32）驗證**再信任任何新數字——
CUDA/cuDNN 換版可能改變數值，否則升級後的數字與磁碟上 25 個舊結果不可比。

**附帶好消息**：VRAM 24 GB → 32.6 GB。`verify_report.md` 估 bigblue2 guidance 峰值 30–35 GB，
在 24 GB 是絕望、在 32.6 GB 是**邊緣可行**。但該估計是舊 torch/舊 allocator 下量的，
升級後需重新量測，不可直接沿用。

**另注意**：GPU 目前不是閒置的 — `PID 96770 (r139431+) train_saufno.py`，99% util。
所有 wall-clock 估計都要打折。

---

## 2. 你做過什麼：八條實驗線

| # | 方向 | 最佳結果 | 判定 |
|---|------|---------|------|
| 1 | **DDPO**（policy gradient fine-tune, run 1–5） | v2 PPO **44.65** | 已關閉。local reward 三個變體全較差（45.80 / 45.86 / 46.50）|
| 2 | **AddLoss**（ReFL-style 輔助 loss） | v2 **45.24** | 已關閉，且被 from-scratch Run Y 獨立二度證偽 |
| 3 | **Ablation supervised**（純監督微調） | 10k steps **44.01**（17 min）| 專案單一方法最佳，但 n=1 |
| 4 | **SVDD**（inference-time 粒子搜尋，Phase 1–5） | **44.845 ± 0.654**（3 seeds）| **成立**，3/3 seeds 勝 paper |
| 5 | **CoDe**（blockwise best-of-N） | **45.216 ± 0.469**（3 seeds）| 成立，與 SVDD 統計上等價 |
| 6 | **TDS**（SMC twisted sampler） | **45.081 ± 0.376**（3 seeds）| 成立，方差最低 |
| 7 | **From-scratch**（隨機初始化訓練） | Run X 500k **45.053** | 有價值的負面/對照結果 |
| 8 | **Flow Matching** | @50 steps **70.028** | 已關閉，慘敗 |
| — | **ISPD limit 分析** | oracle **42.09** | ISPD2005 未飽和 |
| — | **DataAug** | Run C 訓練完成 | **從未 eval** |

### 關鍵科學結論（三線獨立收斂，是真的）
輔助 HPWL/legality 訊號**對這個任務是冗餘的**，因為 ground-truth placement 本身
已經是低 HPWL 且合法的。三個獨立實驗同意：AddLoss v1/v2（fine-tune 端）、
DDPO local reward（policy gradient 端）、from-scratch Run Y（從 step 0 加，
45.121 vs Run X 45.053，Δ=+0.068）。這是一個乾淨、可發表的負面結果。

---

## 3. 完整 leaderboard（26 筆，重算自 CSV 與磁碟）

軸別：FT = 微調權重；FS = 從零訓練；IT = 只改 sampler，權重不動；FT+IT = 兩者

```
 #  METHOD                              avg7    seeds  std     axis   artifact
 1  Ablation_10k (supervised 10k)       44.01   1      —       FT     無（僅報告）
 2  Ablation_10k + opt (Run F 復現)     44.32   1      —       FT     有
 3  SVDD_layered on Ablation_10k (RunE) 44.49   1      —       FT+IT  有  ← 唯一一次合併
 4  DDPO_v2_PPO (seed300)               44.65   1      —       FT     無（被覆蓋）
 5  SVDD_layered on large-v2            44.845  3      0.654   IT     有
 6  FromScratch_X pure 500k             45.053  1      —       FS     有
 7  FromScratch_X 1.6M                  45.078  1      —       FS     有
 8  TDS_layered on large-v2             45.081  3      0.376   IT     有
 9  FromScratch_Y (+aux) 500k           45.121  1      —       FS     有
10  CoDe_layered on large-v2            45.216  3      0.469   IT     有
11  AddLoss_v2                          45.236  1      —       FT     有
12  AddLoss_v1                          45.387  1      —       FT     無（被覆蓋）
13  Ablation_5k                         45.43   1      —       FT     無
14  DDPO_v2 4-seed ensemble MEAN        45.447  4      0.880   FT     部分
15  DDPO_v2.3 local HPWL                45.804  1      —       FT     無
16  DDPO_v2.5 last-K                    45.857  1      —       FT     無
17  FromScratch Stage2 (v2.61)          46.441  1      —       FS     有 ← 全文件未記錄
18  DDPO_v2.4 local HPWL+legality       46.503  1      —       FT     無
19  PAPER Table 10 (seed 400)           46.89   1      —       REF    已發表
20  我們復現 large-v2 + opt (seed300)   48.691  1      —       REF    部分
21  SVDD only, large-v2, 無 opt         66.531  1      —       IT     有
22  FlowMatch @50 ODE steps             70.028  1      —       FS     有（不在 CSV）
23  SVDD only on Ablation_10k           71.944  1      —       FT+IT  有
24  FlowMatch @1000 steps               77.047  1      —       FS     有（不在 CSV）
25  FlowMatch @10 steps                 81.245  1      —       FS     有（不在 CSV）
26  Ablation_10k, 無 guidance           84.854  1      —       FT     有
```

Oracle（跨所有結果的 best-per-circuit）= **42.0899**；
加入未記錄的 fs_stage2 後 = **41.844**。

---

## 4. 三個必須修正的認知錯誤

### 4.1 「headroom 假說」是 n=1 結論，卻被寫成禁令 ← 最重要

`svdd_next_3.md` §5.1 / `svdd_report_5.md` §8 主張 SVDD 與 fine-tuning 是
「二選一、不疊加」，依據是 Phase 3：

- Run F（ablation_10k + opt）= **44.321**
- Run E（ablation_10k + svdd_layered + opt）= **44.485**
- Δ = **+0.164**，單一 seed (300)

但 SVDD 自己在同樣 7 circuits 上的 across-seed std = **0.654**，是該差值的 **4.0 倍**。
逐 circuit 看更糟：Run E vs F 是 3 勝 4 敗，總和由兩個 circuit 主導——
adaptec4 −2.96（SVDD 大勝）與 bigblue4 +4.07（而 SVDD 的 bigblue4 seed std 正好是 4.42，
即這一敗恰等於一個雜訊單位）。

**這與專案已經犯過並修正過的錯誤是同一類**：Phase 4 單 seed 的 bigblue3 = +14.2%
曾催生三個「SVDD 結構性弱點」假說，3-seed 後變成 −3.0%。`svdd_next_3.md` §5.2 因此
立下「每個主要結論至少 3 seeds」的規矩——**而這條 either/or 結論本身從未達到那個標準**。

後果：`tds_next_1.md` §4 與 `code_next_1.md` §4 都據此寫入「不要在 fine-tuned
checkpoint 上測試」的禁令，所以 **CoDe 與 TDS 從未碰過任何 fine-tuned checkpoint**
（已逐一檢查 94 個 eval dir 的 `from_checkpoint`，全部是 `large-v2.ckpt`）。

### 4.2 Leaderboard 前四名的差距小於雜訊

Ranks 1–4 = 44.01 / 44.32 / 44.49 / 44.65，全距 **0.636**，
小於 SVDD 自己的 seed std 0.654。四者全為單 seed 300。
**「supervised fine-tuning 最好」這個結論目前無統計基礎。**

另外：44.01 與 44.32 其實是**同一個組態量了兩次**（ablation_10k + opt + opt-adam, seed 300），
差 0.31。44.01 的 metrics.csv 已不存在，44.32 才是可復現的那個。

逐 circuit 雜訊更誇張：CoDe 的 adaptec2 在三個 seed 間從 28.36 到 38.21（35% 擺幅）。
**任何單 seed 的逐 circuit 主張都沒有意義。**

### 4.3 Baseline framing：同一個問題你已經被咬過一次

2026-05-21 你親自抓到 `svdd_plan_3.md` 把自己的 ablation_10k 當成 paper baseline，
導致 `svdd_report_3.md` 得出「inference-time search 是死路」的錯誤結論。
**相同的陷阱現在以反方向存在**：

`CLAUDE.md` 與 `docs/context.txt` 主張「Ablation 10k 44.01 勝 paper 46.89 達 6.1%」——
這是拿**我們在 v1.61 上微調過的模型（seed 300, n=1, 原始數據已遺失）**
去比**他們未微調的已發表數字（seed 400）**。這正是你自己否決過的比較類型。

三種站得住腳的 framing：

| framing | 最佳結果 | 說明 |
|---|---|---|
| (1) vs paper 已發表 46.89 | Ablation_10k 44.01, −6.13% | 跨 seed、跨訓練狀態，最弱 |
| (2) vs 我們自己的復現 48.69 | Ablation_10k −9.6%；SVDD −7.9% | 唯一同環境的 apples-to-apples |
| (3) 不含微調的方法比較 | **SVDD 44.845 ± 0.654, −4.36%, 3/3 seeds** | **對外最強、唯一有誤差棒** |

**對外（論文、教授）應主打 framing (3)**：在 paper 自己的 checkpoint 上、零訓練、
3 seeds 全勝。44.01/44.32 應單獨列為 fine-tuning 結果並對 48.69 比較，**絕不對 46.89 比較**。

統計修正：`svdd_report_5.md` 的 95% CI [44.11, 45.59] 用了 z=1.96 於 n=3；
正確的 t 區間（t₀.₉₇₅,df=2 = 4.303）是 **[43.22, 46.47]**。結論仍成立（46.47 < 46.89），
但邊際比宣稱的薄很多。

`code_report_1.md` 用非配對檢定判定 CoDe vs SVDD「不顯著（t≈0.80）」；
配對檢定下 SVDD 三個 seed 全勝（+0.264 / +0.590 / +0.261），paired t = 3.40, p≈0.077。
文件低估了 SVDD 對 CoDe 的優勢。（TDS vs SVDD paired t = 0.40，這兩者才是真的無法區分。）

---

## 5. 未收割的既有成果（磁碟上有、文件裡沒有）

1. **FromScratch Stage 2**（eval 於 2026-06-12）= **46.441**，
   任何報告、兩個 CSV、PROGRESS.md 皆無記錄；`from_scratch_plan_1.md` line 306
   的 Step 7 至今仍是未勾選。
   - 它是**回歸**：Stage 1 45.053 → Stage 2 46.441（+3.1%），由 bigblue4
     129.23 → 145.60（+12.7%）驅動。這是關於 **paper 自己的 two-stage recipe**
     的負面結果，目前完全隱形。
   - 它的 adaptec2 = **26.65 是全專案最低**，但 legality 僅 0.9623。
     納入後 oracle 從 42.09 → 41.844，但該改進**完全來自接受一個較不合法的擺放**。

2. **DataAug Run C**：500k steps 訓練於 2026-07-08 15:12 完成，
   `latest.ckpt` (75,655,510 B) 與 `best.ckpt` 都在，**從未做過 ISPD eval**
   （無任何 `ispd2005-s0.aug*` 目錄）。這是專案中唯一「訓練完成但零結果」的 GPU run，
   而 `dataaug_plan_1.md` §4 的判定規則早已預先登記
   （<44.5 採用 / 44.5–45.05 邊際 / >45.05 關閉，對照基準 45.053）。
   - 注意歸因問題：Run C = dihedral **且** edge_dropout=0.1；
     Run B（僅 dihedral）在 2026-07-08 01:27 於 PyG `add_self_loops` scatter 崩潰，未重跑。

3. **三個 flow-matching 結果**（70.028 / 77.047 / 81.245）已寫入報告但不在兩個 CSV 內。

4. **eval 目錄碰撞**：run dir 命名為 `<task>.<method>.<seed>`，不含 `from_checkpoint`，
   導致七個 fine-tuning eval 全部寫入 `ispd2005-s0.eval_macro_only.300/`，
   只有最後一個（AddLoss v2）倖存。Ablation_10k (44.01)、DDPO v2 (44.65)、
   AddLoss v1、DDPO v2.3/2.4/2.5 的原始 metrics 已遺失。
   **checkpoint 都還在，所以是重新 eval 而非重新訓練。**

5. **95 / 222 筆 oracle 相關資料列沒有 `macro_legality`**（所有「extracted from report」的列）。
   任何 legality-constrained 的 oracle 主張都是在 <60% 的資料上算的，必須註明。
   實際重算：無過濾 42.0899；≥0.97 → **42.354**；≥0.98 → **42.814**；
   **≥0.99 不可行**（全專案 adaptec2 最高 legality 僅 0.9843）。

---

## 6. 風險：近五個月的工作只存在一個地方

```
HEAD = 40a396d (2026-04-15) == origin/main   ← 遠端一行都沒有
7 個已追蹤檔案：+753 / −74
41 個未追蹤路徑：約 7,680 行
```

未 commit 的內容包含：`models.py` 裡的 SVDD / CoDe / TDS 三個 sampler、
`FlowMatchingModel`、`reverse_guidance_opt_force` 的 B>1 正確性修正、
`BatchWrapper` chunking 修正、dihedral augmentation、
**四個 guidance YAML（沒有它們 sampler 在 Hydra 下根本無法呼叫）**、
兩個彙整 CSV，以及 SVDD/CoDe/TDS/from-scratch/flowmatch/limit 的全部文件。

無 stash、無 bundle、無第二份 checkout。`git ls-remote` 可通（exit 0），
所以 push 是立即可執行的。**一次誤觸 `git checkout .` 會毀掉三個勝過 paper 的結果。**

`CLAUDE.md` line 35 的標題「Code Modifications (by Claude, **all committed**)」是錯的。

---

## 7. 文件矛盾（依誤導風險排序）

1. **`svdd_report_3.md` §1 與 §11 至今仍寫著「inference-time search 整個方向正式判定 dead end」**——
   與 `svdd_report_5.md` §8「完全推翻」直接相反，且從未修訂。先開這個檔的人會得到完全相反的結論。
2. `svdd_next_2.md` §3.1（已被 next_3 §4 宣告作廢）仍在磁碟上。
3. `docs/context.txt`（2026-05-18，比 SVDD 開跑早一天）不知道 SVDD / CoDe / TDS /
   from-scratch / flow-matching / oracle 六條完成的線。
4. **`memory/` 同樣過期，而且沒人發現** — `memory/MEMORY.md` 與
   `memory/project_experiment_ranking.md`（4/18）仍宣稱 leaderboard 止於 44.01；
   `memory/project_svdd_direction.md`（5/24）說 SVDD「implementation pending」。
   新 session 會從**兩個獨立來源**得到 4 個月前的舊圖像。
5. 三處排名列印順序與數值不符（`ddpo_report_5.md`、`svdd_report_3.md` §4、
   `from_scratch_report_1.md` §4）。不要把 rank 順序當數值順序讀。
6. 預先登記的判定標準被事後推翻兩次：`addloss_plan_2.md` 把 44.5–45.39 列為「繼續」，
   結果 45.236 落在區間內卻仍被放棄；`ddpo_plan_5.md` 設 ≥45.80 為「放棄」，
   v2.5 得 45.857 僅超過 0.057，直接放棄而未試 contingency K=10。
7. `from_scratch_report_1.md` §3 內文說「Run X 勝 5/7」，其自己的表格是 4 勝 3 敗。

---

## 8. 下一步（依序）

### P0 — 不做完就不能做別的

| # | 動作 | 成本 | 說明 |
|---|------|------|------|
| 0.1 | **commit + push**（分 5 個主題 commit）| 15 min | 消除全損風險；`git ls-remote` 已驗證可通 |
| 0.2 | **建新 env 升級 torch ≥2.7/cu128** | 1–2 hr | 否則零 GPU 工作可做 |
| 0.3 | **升級後重跑 Run F 驗證（預期 44.32）** | ~1 hr GPU | 不做這步，升級後所有數字與舊的 25 個結果不可比 |

### P1 — 零 GPU 或極低成本，兩個月空白後的最佳暖身

| # | 動作 | 成本 |
|---|------|------|
| 1.1 | 修 `svdd_report_3.md` §1/§11 的「dead end」錯誤結論 | 15 min |
| 1.2 | 把 4 筆磁碟上未記錄的結果補進兩個 CSV，oracle 更新為 41.844（附 legality 註記）| 1 hr |
| 1.3 | 寫 `docs/next/from_scratch_next_1.md`，記錄 Stage 2 回歸（45.05 → 46.44）| 30 min |
| 1.4 | 重寫 `docs/context.txt` **並同步更新 `memory/`** | 1 hr |
| 1.5 | **從既有資料算 best-of-N 表**：現成 9 個 search run 的 best-per-circuit = **42.488**（−9.4% vs paper），零 GPU | 30 min |
| 1.6 | 修 eval 目錄命名，讓 `from_checkpoint` 進入 run dir 名稱，防止再覆蓋 | 20 min |

### P2 — GPU 工作，依價值排序（全部卡在 P0 之後）

| # | 動作 | 預期 | 成本 |
|---|------|------|------|
| 2.1 | **在 fine-tuned checkpoint 上重測 search，3 seeds**（Run E / Run F 各補 seed 301, 302）| 推翻或坐實 §4.1 的禁令；若可疊加，fine-tune × search × best-of-N 直指 ~42.5 | ~7 hr GPU，零實作 |
| 2.2 | **best-of-N 形式化為推論協定**（SVDD best-of-3 = 43.11；配 legality 下限）| 專案最大的未開採資源；oracle 說 seed 變異 5–12% > 方法間差距 | ~3.5 hr GPU |
| 2.3 | **eval DataAug Run C**（判定規則已預先登記）| 回收兩個月前的既成訓練；回應教授 0620 建議 1 | ~1.2 hr GPU |
| 2.4 | **supervised step sweep 15k / 20k / 30k**（`ddpo_next_3.md` 列為 priority 1，從未執行）| 5k→45.43, 10k→44.01，曲線被砍在還在下降時；每個變體訓練僅 ~17 min | 訓練 ~50 min + eval |
| 2.5 | **bigblue2 在 32 GB 上重試 full guidance** | 唯一從未勝過 paper 的 circuit（57–66 vs 38.8）；成功則 8/8 全勝 | 一次 eval，可能 OOM |
| 2.6 | **paper baseline 補到 3–4 seeds** | 目前 48.69 是 n=1 且原始檔已被覆蓋；沒有它就沒有配對比較 | ~3.5 hr GPU |
| 2.7 | FreeDoM time-travel 疊在 TDS 上 | 文件自估僅 −0.5~−1% | 半天實作 + 3.5 hr |

**若只能選一件 GPU 工作：2.1。** 它是唯一可能突破 44 而不需要新想法的路，
而且無論結果如何都會產出一個可發表的 ablation。

### 從未打開過的方向（`docs/plan/init_plan.md`，被所有 next 文件遺漏）
- **方向四 — Graph Transformer backbone**（`networks/gt.py` 已在 repo 內；
  動機是 long-range dependency，而 HPWL 正是 long-range 的）
- **方向五 — GNN conv layer 置換**（GCN/SAGE/GIN/GAT/Transformer/Gated 皆已支援，
  提議從 `large-v2.ckpt` 部分凍結微調）← **全專案最便宜的未試想法，只是改 config**
- **方向六 — iterative clustering**（`policies.iterative_clustering` 已實作）

### 應正式關閉的方向
DDPO / 任何 local reward（5 runs，根因已診斷）、AddLoss（兩個獨立實驗證偽）、
flow matching 與 drifting model（70.03 vs 45.05 同預算）、EDA dataset（工程量最大、
其 plan 自己預登記了「標籤可能比現有合成資料更差」的 kill test）、
申請更大 GPU（原始動機是提高 DDPO batch size，而 DDPO 已死）。

from-scratch **不建議完全關閉**：Run X 以 1/6 步數從隨機初始化達 45.05，
是唯一不碰 paper checkpoint 就勝過 paper 的方法，適合作為論文的一節，
但不值得再投入 compute（延伸到 3M steps 約 2.3 天/變體）。

---

## 9. 論文可行性

**有論文，但故事不是「我們贏 ChipDiffusion 4.4%」。**

真正的貢獻是**組合性與 checkpoint headroom**：三個機制上截然不同的選擇規則
（per-step softmax resampling / per-100-step hard argmax / SMC importance weights + ESS）
收斂到同一個天花板（−3.6% ~ −4.4%，全距 0.79% vs seed std 0.4–0.6），
說明**搜尋預算重要、選擇規則不重要**——這是個乾淨的負面結論。
加上 negative controls 已齊備（SVDD 取代 opt = 66.53；完全無 guidance = 84.85）。

**寫之前必須先解決的三個致命缺口**：

1. **沒有 matched-compute best-of-N 對照。** SVDD/CoDe/TDS 用 K=4，
   審稿人會問「那把 paper 的 opt guidance 跑 4 個 seed 取最好呢？」——這從未跑過，
   因為 opt-only baseline 只有一個 seed。若 best-of-4 opt-only 落在 44.8 附近，
   搜尋的貢獻就縮水成「同品質、1.12× 而非 4× 計算量」，是完全不同且弱很多的論文。
   **成本為零**（同樣 4 個 baseline run 同時當 baseline 與對照）。
2. **bigblue2 會翻轉結論的正負號。** Paper 的 8-circuit 平均是 45.88；
   我們 bb2 ≈ 57 時 8-circuit 平均變成 SVDD 46.37 / TDS 46.57 / CoDe 46.69，
   **全部輸給 paper +1.1% ~ +1.8%**。損益兩平需要 bb2 ≤ 53.1 / 51.5 / 50.5。
   若能達 paper 水準的 38.8，則是 44.09 / 44.30 / 44.42（−3.2% ~ −3.9%）。
   *技術提示*：`guidance.py:49` 的 `legality_guidance_potential_tiled`（block_size=16384）
   已在 `legalization.py:52,148` 與 `sc_placement.py:281` 中實際使用（並非死碼），
   但 guidance 路徑（`models.py:1916, 1954`）與三個 search reward
   （`:1621, :1726, :1801`）走的是**未 tiled 的版本**。把 guidance 改走 tiled 版
   是低風險改動，因為 tiled kernel 已在 legalization 上經過實戰驗證。
3. **記錄中的 runtime 被 CPU 競爭汙染，直接引用會產生物理上不可能的表。**
   baseline adaptec1 `generation_time` = 419.6 s（2026-03）vs Run F 同組態 133.4 s（2026-05），
   3.1× 差距。若照抄，SVDD (132.5 s) 會看起來比 opt-only (419.6 s) **還快**，
   而它每個 reverse step 要做 K=4 次前向——任何審稿人都會直接拒。
   誠實的同日同 checkpoint 量測是 Run F 3079 s vs Run E 3455 s = **+12.2% overhead**。
   實測每 seed 7-circuit 成本：CoDe 43.5 min、opt-only 51.3 min、SVDD 57.6 min、
   TDS 95.6 min（其中 81 min 花在 bigblue4 一個 circuit 上）。

**寫作順序建議**：先跑 2.1（決定框架是「search 取代 fine-tuning」還是
「search 與 fine-tuning 可組合」——後者是強很多的論文），再開始寫。

---

## 10. 給下一個 session 的最小交接

若只讀三件事：
1. 環境壞了（§1），修好前不要規劃任何 GPU 工作。
2. 沒 commit（§6），這是唯一的全損風險。
3. 關閉最高價值方向的那條禁令是 n=1 且在自己的雜訊帶內（§4.1），值得重測。
