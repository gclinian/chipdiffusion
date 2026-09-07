# From-Scratch Phase 1 回顧（next_1）

> Report: `docs/report/from_scratch_report_1.md`
> Plan: `docs/plan/from_scratch_plan_1.md`
> 實驗執行：2026-05-26 ~ 2026-06-12
> 寫作日期：2026-09-07

> ⚠️ **這份回顧遲寫了 3 個月**。report_1 寫於 2026-05-27（只涵蓋 Stage 1 500k pilot），
> 但 plan_1 §10 Step 7 的 **Stage 2 在 2026-06-11~06-12 實際跑完並 eval 了**，
> 結果從未寫進任何 report，也沒進 `all_experiments_{summary,per_circuit}.csv`，
> plan_1 §10 那個 checkbox 至今還是 `[ ]`。本文件第 4 節就是補這個洞。

---

## 1. 一句話總結

**Run X（pure denoising, 500k, random init）= 45.053 是全 project 唯一「不碰 paper 的 large-v2 checkpoint 就贏 paper 46.89」的方法（−3.9%），而且只用了 paper 3M step 的 1/6；同一組實驗順帶給出全 project 對 AddLoss 假說最乾淨的一刀（Run Y +aux = 45.121，Δ +0.068，遠在噪音內）。但 plan_1 真正的終點 Stage 2（v2.61 fine-tune）跑完是 **46.441，比自己的 Stage-1 起點退步 +3.1%**，而且退步 100% 來自 bigblue4 一個 circuit（129.23 → 145.60, +12.7%）—— 這件事從來沒被記錄過。**

---

## 2. 這個 track 建立了什麼

### 2.1 三個數字（全部 seed=300, n=1, 7-circuit avg HPWL ×10⁵，越低越好）

| Run | 設定 | avg7 | vs paper 46.89 | 狀態 |
|---|---|---:|---:|---|
| **Run X (500k)** | pure denoising MSE, random init, v1.61-fs, batch=32 | **45.053** | **−3.92%** | on disk |
| Run X (1.6M) | 同上 resume 到 1.6M（中途 crash） | 45.078 | −3.86% | on disk |
| **Run Y (500k)** | + ReFL-style HPWL/legality aux **from step 0** | **45.121** | −3.77% | on disk |
| **Stage 2 (v2.61)** | Run X 500k → v2.61 finetune 500k, batch=32, lr=1e-4 | **46.441** | −0.96% | **on disk, 全文件未記錄** |

### 2.2 在 leaderboard 的真正位置

| Rank | 方法 | avg7 | 軸 | 需要 paper ckpt？ |
|---|---|---:|---|---|
| 1 | Ablation_10k（supervised 10k FT） | 44.01 | FT | ✅ 要 |
| 2 | Ablation_10k + opt（Run F repro） | 44.321 | FT | ✅ 要 |
| 3 | SVDD_layered on Ablation_10k（Run E） | 44.485 | FT+IT | ✅ 要 |
| 4 | DDPO_v2_PPO | 44.65 | FT | ✅ 要 |
| 5 | SVDD_layered on large-v2（n=3, std 0.654） | 44.845 | IT | ✅ 要 |
| **6** | **FromScratch_X pure 500k** | **45.053** | **FS** | **❌ 不用** |
| 7 | FromScratch_X 1.6M | 45.078 | FS | ❌ 不用 |
| 8 | TDS_layered on large-v2（n=3, std 0.376） | 45.081 | IT | ✅ 要 |
| 9 | FromScratch_Y (+aux) 500k | 45.121 | FS | ❌ 不用 |
| 10 | CoDe_layered on large-v2（n=3, std 0.469） | 45.216 | IT | ✅ 要 |
| 11 | AddLoss_v2 | 45.236 | FT | ✅ 要 |
| **17** | **FromScratch Stage2 (v2.61)** | **46.441** | **FS** | **❌ 不用** |
| 19 | **PAPER Table 10 (seed 400)** | **46.89** | — | — |
| 20 | 我們重現 large-v2 + opt（seed 300） | 48.691 | — | ✅ 要 |
| 22 | FlowMatch @50 ODE steps | 70.028 | FS | ❌ 不用 |

**這是這個 track 最該進 paper 的一句話**：leaderboard 上 rank 1-5 全部要靠 paper 自己釋出的 `large-v2.ckpt`（不管是 fine-tune 它還是在它上面做 inference-time search）。**rank 6 的 Run X 是第一個從 random init 出發還贏 paper 的**。另一條 from-scratch 線 FlowMatch 三個設定（70.0 / 77.0 / 81.2）全部大輸，所以「from-scratch 能贏」不是 trivially true —— 是 Run X 這個特定配方的功勞。

### 2.3 Compute 的故事也是賣點

- paper Stage 1 = 3,000,000 steps @ batch 64。我們 = 500,000 steps @ batch 32（batch 被 PyG `gatv2_conv.add_self_loops` batched edge_index bug 逼降）。
- 也就是 **≈ 1/12 的 sample exposure**，拿到 −3.9%。
- 實測 ms/step 67（pilot 估 229，高估 3.4×）→ 3M extension ≈ **56 hr ≈ 2.3 天/variant**，不是 plan_1 §4 寫的 8-9 天。

---

## 3. Run X vs Run Y = 全 project 對 AddLoss 假說最乾淨的一刀

這是 report_1 有寫但**份量被低估**的結果。

| 實驗設計 | aux 加在哪 | 有 aux | 沒 aux | Δ |
|---|---|---:|---:|---:|
| **From-scratch（本 track）** | random init step 0 起全程 | Run Y **45.121** | Run X **45.053** | **+0.068** |
| Fine-tune（AddLoss v2） | large-v2 之上 FT | 45.236 | （大致對照 Ablation 系列 44.01-45.43） | marginal |
| Fine-tune（AddLoss v1，無 timestep weighting） | large-v2 之上 FT | 45.387 | 同上 | marginal |

**為什麼 from-scratch 版本才是決定性的**：fine-tune 上加 aux 沒效果，永遠可以被辯護成「起點已經太好了，signal 沒空間」（這正是 svdd_next_3 §5.1 的 headroom 假說，而且那個假說是真的）。但 Run Y 是**從 random init 的第 0 步就開著 aux**，訓練 500k step，模型完全沒有「已經很好了」這個藉口 —— 結果還是 +0.068，連 3-seed SVDD 的 std 0.654 的 1/9 都不到。

**兩個 independent experimental design（from-scratch 與 fine-tune）指向同一個結論**，這才叫 negative result 而不是「一次沒做出來」：

> **ReFL-style HPWL/legality auxiliary loss 在 chipdiffusion 上是 redundant，因為 ground-truth placement 本身就已經是 low-HPWL 且 legal 的 —— aux 的梯度方向跟 denoising MSE 大幅重疊，不帶新資訊。**

這是**可以寫進 paper 的 negative result**（negative results section 或 ablation table），不是失敗紀錄。建議寫法：一張 2×2 表（{from-scratch, fine-tune} × {aux, no-aux}），四格數字都有，結論一行。

**唯一的反例、也要誠實寫**：Run Y 在 adaptec3（−2.87）和 adaptec4（−2.66）明顯贏 Run X，兩個都是中型 circuit（V=723 / 1329）。但 Run Y 在 adaptec2 輸 2.38、bigblue4 輸 1.87，7-circuit 抵銷掉。而且 Run Y 的 adaptec2 legality 只有 **0.9088**（Run X 0.9639）—— aux 裡的 legality 項反而讓 legality 變差，這件事本身就很怪，n=1 不該解釋，但值得記著。

---

## 4. 【本文件重點】Stage 2 是 regression —— 一個從未被記錄的結果

### 4.1 事實

- Checkpoint：`logs/diffusion_debug/v2.61.fs_p1_X_stage2_b32.61/latest.ckpt`
  （config 確認：`mode=finetune`, `task=v2.61`, `train_steps=500000`, `batch_size=32`, `lr=1e-4`,
  `from_checkpoint=v1.61-fs.61.fs_p1_X_500k.61/latest.ckpt` → 起點就是 Run X 500k，跑完整 500k step）
- Eval：`logs/diffusion_debug/ispd2005-s0.fs_stage2_full_part{1,2}.300/metrics.csv`，檔案時間 **2026-06-12 22:42 / 23:29**
- 出現在：**沒有任何 report、沒有 `all_experiments_summary.csv`、沒有 `all_experiments_per_circuit.csv`**
  （只有本次 audit 的 `docs/next/comeback_next_1.md` 提到）

| idx | Circuit | V | Stage 1 (Run X 500k) | **Stage 2 (v2.61)** | Δ | Stage2 legality |
|---|---|---:|---:|---:|---:|---:|
| 0 | adaptec1 | 543 | 9.121 | 9.309 | +0.19 | 0.99139 |
| 1 | adaptec2 | 566 | 28.962 | **26.646** | **−2.32** | 0.96230 |
| 2 | adaptec3 | 723 | 56.455 | 55.283 | −1.17 | 0.99668 |
| 3 | adaptec4 | 1329 | 55.866 | 53.215 | −2.65 | 0.99770 |
| 4 | bigblue1 | 560 | 2.705 | 2.681 | −0.02 | 0.99563 |
| 6 | bigblue3 | 1298 | 33.036 | 32.353 | −0.68 | 0.99736 |
| 7 | bigblue4 | 8170 | 129.226 | **145.596** | **+16.37 (+12.7%)** | 0.99016 |
| | **avg(7)** | | **45.053** | **46.441** | **+1.39 (+3.1%)** | |

### 4.2 關鍵的 nuance：Stage 2 其實贏了 6/7 個 circuit

不要把它讀成「Stage 2 全面變爛」。逐 circuit 看：

- **6/7 circuit 進步**（adaptec2 −2.32、adaptec4 −2.65 都是大幅進步），只有 adaptec1 微退 +0.19。
- **拿掉 bigblue4，6-circuit avg：31.024 → 29.915，Stage 2 反而好 −3.6%。**
- 整個 +1.39 的退步 100% 由 bigblue4 一個 circuit 造成，而 bigblue4 的絕對值（~130-146）本來就 dominate 未加權平均。

所以正確的敘述是：**Stage 2 在 V ≤ 1329 的 circuit 上有效，在 V=8170 的 bigblue4 上災難性失效，而 unweighted mean 被 bigblue4 綁架。**

bigblue4 的 legality 沒有變差（0.9877 → 0.99016 略升），所以這**不是**拿 legality 換 HPWL —— 是純粹的 HPWL 退步。

### 4.3 為什麼這值得記（而不是刪掉）

v2.61 之所以被選作 Stage 2 dataset，理由是 `max_instance=1600` 比 v1.61 的 `max_instance=400` **更接近 ISPD circuit 的實際規模**（plan_1 §1）。也就是說：我們特地換到一個「更像 target domain」的資料集，結果在 target domain 裡**最大**的那個 circuit 上退步最多。這跟直覺完全相反，所以是資訊量最高的一格。

### 4.4 假說（**n=1，這是假說不是定論**）

按可信度排：

1. **v2.61 的 max_instance=1600 仍然遠小於 bigblue4 的 8170**。Stage 2 把模型的先驗從「≤400 macro」拉到「≤1600 macro」，等於把 extrapolation 的落點從 20× 移到 5×，但**同時放棄了 Stage 1 那個恰好 generalize 得不錯的解**。中小 circuit 因為落在或接近 1600 內而受益，bigblue4 依然在分布外，卻失去了原本的 lucky extrapolation。
2. **500k step @ lr=1e-4 在 4600 個 v2.61 sample 上 = ~3.5k epoch，overfit 到 v2.61 分布**。report_1 §6.1 假說 5 已經懷疑過 Stage 1 對 v1.61 overfit；Stage 2 只是把 overfit 的目標換掉。
3. **Single-seed noise**。CLAUDE.md / context.txt 已知 adaptec2 跨 4 seed 有 29.26-36.61 的 spread；bigblue4 在 SVDD 3-seed 的 std 是 4.42。但 +16.37 遠大於 4.42，**noise 解釋不掉這一格**，所以我把它排最後。
4. **v2.61 train data 是我們自己 regen 的（4600 train / 200 val, num_workers=4），不是 paper 原始的**。gen_params 沒有像 v1.61-ddpo 那樣被逐項核對過 == paper。這條沒被驗證，是個 open risk。

### 4.5 唯一的亮點，但要打折

**fs_stage2 的 adaptec2 = 26.65 是全 project 任何方法、任何 seed 的最低值。** 對照 paper 31.00、我們重現的 large-v2 39.06。

但是 **macro_legality = 0.9623**，低於 oracle 原本在 adaptec2 選中的那筆（0.9781）。換句話說，這個 26.65 是**拿 legality 換來的 HPWL**，不是免費的。

具體影響 oracle：
- 現行 oracle（best-per-circuit，全紀錄）= **42.0899**
- 把未記錄的 fs_stage2 adaptec2 加進去 → **41.8444**
- 但套 legality filter：≥0.97 → 42.354；≥0.98 → 42.814；**≥0.99 → infeasible**（全 project adaptec2 的 legality 最高只有 0.9843）

⚠️ **這些 legality-filtered 數字的可信度有限**：222 筆 oracle 相關 CSV row 裡有 **95 筆完全沒有 macro_legality 值**（所有 "extracted from report" 的 row），所以 legality 約束只在 <60% 的資料上算得出來。要拿 oracle 進 paper 之前，這 95 筆得先從 on-disk metrics.csv 補回來。

---

## 5. report_1 的錯誤更正（**不改原檔，在此更正**）

### 5.1 §3「Run X 贏 5/7 circuits」→ 實際是 **4 勝 3 敗**

report_1 §3 表格下方寫：
> 「Run X 贏 5/7 circuits in raw terms（…X beats Y 5/7）」「Run Y 贏 2/7（adaptec3, adaptec4）」

**它自己那張表的 Y−X 欄就否定了這句話**。Y−X 為正 = X 贏：

| Circuit | Y−X | 誰贏 |
|---|---:|---|
| adaptec1 | +0.31 | X |
| adaptec2 | +2.38 | X |
| adaptec3 | **−2.87** | **Y** |
| adaptec4 | **−2.66** | **Y** |
| bigblue1 | **−0.01** | **Y**（等於平手） |
| bigblue3 | +1.47 | X |
| bigblue4 | +1.87 | X |

→ **X 4 勝、Y 3 勝**（其中 bigblue1 的 −0.01 實質是平手，2.705 vs 2.6909）。漏掉的是 bigblue1。
這不改變任何結論（avg Δ 仍是 +0.068，仍在噪音內），但**「5/7」這個數字不能寫進 paper**。

### 5.2 §11 「beats 7/7 circuits in our env」→ 實際 **6/7**

同一份 report 的一句話結論說 Run X 在我們環境下贏 7/7。對照 §3 表的 Our Repro 欄：bigblue1 = **2.69**，Run X = **2.70** → Run X 微幅落後。實際是 **6 勝 1 負**。

### 5.3 §4 leaderboard 表格 rank 6/7 順序倒了

表中 rank 6 = TDS layered 45.08、rank 7 = Run X 45.05。45.05 < 45.08，**Run X 應該排 rank 6**（本文件 §2.2 已用正確順序）。

> 三處都只更正在這裡，`from_scratch_report_1.md` 原檔**未修改**（保留當時的紀錄原貌）。

---

## 6. 誠實的限制（寫 paper 前必須說清楚）

1. **全部 n=1，全部 seed=300**。Run X、Run Y、1.6M、Stage 2 —— 四個結果沒有一個有第二個 seed。對照組 SVDD/TDS/CoDe 都是 n=3。svdd_next_3 §5.2 自己訂下的規則是「**每個主結論至少 3 seed**」，這個 track **完全沒有遵守**。
2. Run X 跟 paper 的差距（45.053 vs 46.89 = −1.84）本身就跟 3-seed 方法的 std（SVDD 0.654、CoDe 0.469、TDS 0.376）同一個數量級。**「Run X 贏 paper」在 n=1 下不是統計結論。**
3. seed 也跟 paper 不同（我們 300 / paper 400），per-circuit 差異有一部分是 seed。
4. batch=32 vs paper 64（PyG bug 逼的），所以「1/6 step count」實際上是「≈1/12 sample exposure」—— 對我們有利的說法要用後者才誠實。
5. bigblue2 全程排除（guidance V×V OOM），所以是 7-circuit 而非 paper 的 8-circuit。
6. Stage 2 用的 v2.61 train data 是我們 regen 的，gen_params 沒有跟 paper 逐項核對（見 §4.4-4）。
7. 順帶：`svdd_report_5.md` 那個 95% CI [44.11, 45.59] 用的是 z=1.96，n=3 應該用 t（t₀.₉₇₅,df=2 = 4.303）→ 正確區間是 **[43.22, 46.47]**。結論仍成立（46.47 < 46.89），但區間寬度差很多，引用時用後者。

---

## 7. 下一步建議（誠實排序）

### 🟢 最強候選：**把 inference-time search 疊到 from-scratch checkpoint 上**（從來沒人做過）

這是這次分析翻出來、最有 leverage 的一格。整個 project 的 {checkpoint} × {inference-time method} 矩陣長這樣：

| Base checkpoint | + opt only | + SVDD_layered | Δ |
|---|---:|---:|---:|
| large-v2（paper raw） | 48.691 | **44.845** | **−3.85** |
| Ablation_10k（fine-tuned） | 44.321 | 44.485 | +0.16（沒 headroom）|
| **FromScratch_X 45.053** | 45.053 | **從來沒跑過** | **?** |

svdd_next_3 §5.1 的 headroom 假說說：起點越強，inference-time search 的空間越小。**FromScratch_X (45.053) 的 headroom 比 Ablation_10k (44.321) 多**，理論上 SVDD/CoDe 應該還有得撈。而且這一格有額外的敘事價值：**它會是唯一一個「完全不碰 paper checkpoint」的 FS + IT 組合** —— 從 random init 訓練 + inference-time search，整條 pipeline 自給自足。

- 成本：SVDD_layered 或 CoDe_layered × 7 circuits × 3 seed ≈ **4-5 小時**（單次 full eval ~1.5 hr）
- 參考上界：現有 SVDD/CoDe/TDS × 3-seed 的 **best-of-9 = 42.488**，遠低於任何單一方法 —— 說明「選最好的一個」這個動作本身還有 2.5 unit 的空間
- 建議先跑 CoDe_layered（code_next_1 §3.1 已證明 CoDe ≈ SVDD 且快 17%、更簡單），單 seed 探路，有訊號再補 3 seed

### 🟡 建議「停掉」：3M step extension

plan_1 原本的終點是 3M。實測成本 **2.3 天/variant**，report_1 §4 的預期落點是 **44.0-44.5**。

對照一下：花 2.3 天換 ~0.5-1.0 unit，vs 花 4-5 小時做 best-of-N 可能拿到相近或更多。**cost/benefit 明顯輸**。建議：

> **把 3M extension 停在計畫階段，把這個 track 當成一個「paper section」收掉**，而不是繼續投 compute。

它要講的故事（random init、1/12 exposure、贏 paper、順帶 kill 掉 aux 假說）在 500k 的數字上已經完整了。3M 只會讓數字好看一點，不會改變任何結論。

### 🟡 中優先：Stage 2 regression 的 n=2 確認

§4 的結論目前 n=1。最便宜的確認方式**不是**重訓：

- 直接拿 `v2.61.fs_p1_X_stage2_b32.61/` 底下已經存在的中間 ckpt（`step_100000` … `step_500000` 都在 disk 上）跑 **bigblue4 單 circuit eval**，看退步是不是隨 step 單調發生 → ~25 min/點 × 4 點 ≈ 2 hr
- 如果 bigblue4 在 step_100000 就已經退步 → 支持假說 1/2（分布轉移），且可以直接建議 early-stop
- 如果只有 latest 退步 → 比較像 noise 或訓練後期的意外

這比重跑一個 seed 便宜一個數量級，而且用的全是已經在 disk 上的東西。

### 🔵 低優先

- Run X 補 seed 301/302（~3 hr eval，不用重訓）—— 只是為了讓「贏 paper」變成統計結論
- 把 95 筆缺 macro_legality 的 CSV row 從 on-disk metrics.csv 補回來（oracle 要用）
- 把 fs_stage2 跟 FlowMatch 三筆補進 `all_experiments_{summary,per_circuit}.csv`（本文件不動那兩個檔）

### 🔴 **當前所有 GPU 工作都被擋住（2026-09-06 起）**

上面每一項都要 GPU，而**現在一個都跑不動**：

- 機器的 GPU 已被換成 **RTX 5090（32,607 MiB, compute capability sm_120）**
- `chipdiff` env 是 **torch 2.2.1+cu121**，arch_list 只到 **sm_90** → `torch.cuda.is_available()` 回 True，但第一個 kernel launch 就爆 `no kernel image is available for execution on the device`
- `diffplace` env（torch 2.6.0+cu124）**同樣壞**
- 沒有裝任何編譯型 PyG extension（torch-scatter/sparse/cluster 都不在）→ **修法是 pip 層的 torch≥2.7/cu128 升級，不是 C++ 重編**

**在這個修好之前，§7 全部是紙上作業。** 另外 CLAUDE.md 現在整份還寫「24GB GPU」，那些 OOM gotcha（bigblue2 guidance V×V、`skip_guidance_threshold=10000`）都是基於 24GB 寫的 —— 現在是 32GB，升級完之後值得花 1 小時重測 bigblue2 能不能開 guidance（若能，就補得回第 8 個 circuit，可以直接對 paper 的 8-circuit 45.88）。

---

## 8. 不要做的事

- ❌ **不要再測 auxiliary HPWL/legality loss 的任何變體**（不同 weight、不同 timestep weighting、加在不同 stage）。from-scratch 跟 fine-tune 兩個 independent design 已經一致說 redundant。再測是在燒 GPU 買一個已知答案。
- ❌ **不要因為「Stage 2 avg 變差」就把 v2.61 整個否定**。6/7 circuit 是進步的，問題是 bigblue4 一格（§4.2）。
- ❌ **不要拿 fs_stage2 的 adaptec2 = 26.65 當 SOTA 宣傳**，它的 legality 是 0.9623（§4.5）。要報就得同時報 legality。
- ❌ **不要在 report_1 原檔上改那三個錯誤**（§5）—— 保留紀錄原貌，更正寫在這裡。
- ❌ **不要在 GPU 修好前排任何實驗時程**。

---

## 9. Open questions for plan_2

1. **這個 track 收掉當 paper section，還是繼續投 compute？** 我建議收掉（§7）。要 user 拍板。
2. **CoDe/SVDD layered on FromScratch_X 要不要跑？** 我建議跑，這是唯一沒被填的格子，且成本只有 3M extension 的 1/10。
3. **Stage 2 regression 要不要用 on-disk 中間 ckpt 做 bigblue4 單 circuit 掃描確認？**（~2 hr，全用現成檔案）
4. **v2.61 regen 出來的 train data 的 gen_params 要不要跟 paper 逐項核對？** 這是 §4.4 假說 4 的唯一驗證方式，也決定 Stage 2 的結論能不能寫進 paper。
5. **AddLoss negative result 在 paper 裡放哪？** 我建議獨立一小節 + 一張 2×2 表（{FS, FT} × {aux, no-aux}），四格數字都齊。
6. **GPU 修復（torch≥2.7/cu128）誰做、什麼時候做？** 這是所有事情的前置。順帶要更新 CLAUDE.md 的 24GB 描述。
7. **要不要趁 32GB 重測 bigblue2 guidance？** 若成功就能報 8-circuit，直接對 paper 45.88，narrative 乾淨很多。
