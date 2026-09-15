# STATUS — 自動產生，不要手改

由 `python3 scripts/ledger.py status` 產生於 **2026-09-15 10:19**。
來源：`docs/ledger/runs.jsonl`（200 runs）+ `docs/ledger/results/`（逐筆 metrics.csv 存檔）。

環境指紋：GPU `NVIDIA GeForce RTX 5090, 32607 MiB` ・ git `5b22680`（**working tree 有未 commit 變更**）

## Leaderboard（7-circuit avg HPWL，排除 bigblue2，越低越好）

參考點：paper 已發表 **46.89** ・ 我們自己跑 paper checkpoint（`base_cu128_*`，現行 stack）：avg7 **45.987**（n=1） ・ avg6 [無 bigblue4] **31.212**（n=3, sd 0.814）

avg6 = 六個便宜 circuit（bigblue4 佔 eval 時間 78%，3-seed 協定只在 seed 300 跑它）。同 stack 的 2σ 雜訊帶 = 2 × avg6 sd；差距小於它的結果不算差距。

這張表只含**磁碟上還有 metrics.csv 的 run**。有些歷史結果（Ablation_10k 44.01、DDPO v2 44.65、AddLoss v1 等）的原始檔已被 eval 目錄碰撞覆蓋，只存在於報告中 — 那些要看 `docs/all_experiments_summary.csv`。兩張表不一致是預期的，差異本身就是資訊。

### 現行 stack（2026-09-13 起，chipdiff-b）

| avg7 | avg6 | circuits | run group | seed | checkpoint |
|-----:|-----:|---------:|-----------|------|------------|
| — | 29.646 | 6 | `ispd2005-s0.bon2c_s301.301` | 301 | `large-v2 (paper)` |
| 44.817 | 30.090 | 7 | `ispd2005-s0.bon2b_s300.300` | 300 | `large-v2 (paper)` |
| — | 30.124 | 6 | `ispd2005-s0.bon2c_s300.300` | 300 | `large-v2 (paper)` |
| — | 30.443 | 6 | `ispd2005-s0.base_cu128_s302.302` | 302 | `large-v2 (paper)` |
| — | 30.473 | 6 | `ispd2005-s0.svdd_cu128_s302.302` | 302 | `large-v2 (paper)` |
| 44.166 | 30.521 | 7 | `ispd2005-s0.svdd_cu128_s300.300` | 300 | `large-v2 (paper)` |
| — | 30.810 | 6 | `ispd2005-s0.bon2c_s302.302` | 302 | `large-v2 (paper)` |
| 44.649 | 31.094 | 7 | `ispd2005-s0.runX_cu128_s300.300` | 300 | `v1.61-fs.61.fs_p1_X_500k.61/latest.ckpt` |
| — | 31.130 | 6 | `ispd2005-s0.base_cu128_s301.301` | 301 | `large-v2 (paper)` |
| — | 31.131 | 6 | `ispd2005-s0.runF_cu128_s302.302` | 302 | `v1.61-ddpo.ablation_supervised_10k.61/latest.ckpt` |
| 51.705 | 31.143 | 7 | `ispd2005-s0.diag1C_tshift_up.300` | 300 | `large-v2 (paper)` |
| 45.700 | 31.184 | 7 | `ispd2005-s0.runF_cu128_s300.300` | 300 | `v1.61-ddpo.ablation_supervised_10k.61/latest.ckpt` |
| — | 31.327 | 6 | `ispd2005-s0.runF_cu128_s301.301` | 301 | `v1.61-ddpo.ablation_supervised_10k.61/latest.ckpt` |
| 45.569 | 31.607 | 7 | `ispd2005-s0.runXavg_400k_500k.300` | 300 | `v1.61-fs.61.fs_p1_X_500k.61/avg_400k_500k_uniform.ckpt` |
| 46.497 | 31.612 | 7 | `ispd2005-s0.runXavg_250k_500k.300` | 300 | `v1.61-fs.61.fs_p1_X_500k.61/avg_250k_500k_uniform.ckpt` |
| — | 31.650 | 6 | `ispd2005-s0.bon2b_s302.302` | 302 | `large-v2 (paper)` |
| 45.987 | 32.064 | 7 | `ispd2005-s0.base_cu128_s300.300` | 300 | `large-v2 (paper)` |
| — | 32.117 | 6 | `ispd2005-s0.svdd_cu128_s301.301` | 301 | `large-v2 (paper)` |
| 45.656 | 32.754 | 7 | `ispd2005-s0.diag1C_tshift_inv.300` | 300 | `large-v2 (paper)` |
| — | 33.794 | 6 | `ispd2005-s0.bon2b_s301.301` | 301 | `large-v2 (paper)` |

### 舊 stack（2026-09-13 之前，torch 2.2.1 / 24 GB 卡）— 只能彼此比較，**不可與上表比較**

| avg7 | avg6 | circuits | run group | seed | checkpoint |
|-----:|-----:|---------:|-----------|------|------------|
| 46.441 | 29.915 | 7 | `ispd2005-s0.fs_stage2_full.300` | 300 | `v2.61.fs_p1_X_stage2_b32.61/latest.ckpt` |
| 44.692 | 30.269 | 7 | `ispd2005-s0.code_p1_s301.301` | 301 | `large-v2 (paper)` |
| 44.485 | 30.341 | 7 | `ispd2005-s0.svdd_p3_runE.300` | 300 | `v1.61-ddpo.ablation_supervised_10k.61/latest.ckpt` |
| 44.772 | 30.461 | 7 | `ispd2005-s0.tds_p1_s302.302` | 302 | `large-v2 (paper)` |
| 44.747 | 30.489 | 7 | `ispd2005-s0.eval_macro_only.600` | 600 | `v1.61-ddpo.ddpo_v2_ppo.61/latest.ckpt` |
| 44.973 | 30.756 | 7 | `ispd2005-s0.tds_p1_s300.300` | 300 | `large-v2 (paper)` |
| 45.121 | 30.792 | 7 | `ispd2005-s0.fs_p1_Y_full.300` | 300 | `v1.61-fs.61.fs_p1_Y_500k.61/latest.ckpt` |
| 44.321 | 30.829 | 7 | `ispd2005-s0.svdd_p3_runF.300` | 300 | `v1.61-ddpo.ablation_supervised_10k.61/latest.ckpt` |
| 45.053 | 31.024 | 7 | `ispd2005-s0.fs_p1_X_full.300` | 300 | `v1.61-fs.61.fs_p1_X_500k.61/latest.ckpt` |
| 45.078 | 31.063 | 7 | `ispd2005-s0.fs_p1_X_1.6M_full.300` | 300 | `v1.61-fs.61.fs_p1_X_3M.61/latest.ckpt` |
| 45.236 | 31.141 | 7 | `ispd2005-s0.eval_macro_only.300` | 300 | `v1.61-ddpo.addloss_v2.61/latest.ckpt` |
| 44.103 | 31.160 | 7 | `ispd2005-s0.svdd_p5_runH_s301.301` | 301 | `large-v2 (paper)` |
| 45.335 | 31.163 | 7 | `ispd2005-s0.svdd_p5_runH_s302.302` | 302 | `large-v2 (paper)` |
| 45.361 | 31.274 | 7 | `ispd2005-s0.code_p1_s300.300` | 300 | `large-v2 (paper)` |
| 45.499 | 31.732 | 7 | `ispd2005-s0.tds_p1_s301.301` | 301 | `large-v2 (paper)` |
| 45.596 | 31.768 | 7 | `ispd2005-s0.code_p1_s302.302` | 302 | `large-v2 (paper)` |
| 45.097 | 31.896 | 7 | `ispd2005-s0.svdd_p4_runH.300` | 300 | `large-v2 (paper)` |
| 46.014 | 32.257 | 7 | `ispd2005-s0.eval_macro_only.500` | 500 | `v1.61-ddpo.ddpo_v2_ppo.61/latest.ckpt` |
| 46.380 | 32.354 | 7 | `ispd2005-s0.eval_macro_only.400` | 400 | `v1.61-ddpo.ddpo_v2_ppo.61/latest.ckpt` |
| 71.944 | 36.983 | 7 | `ispd2005-s0.svdd_p2_runC.300` | 300 | `v1.61-ddpo.ablation_supervised_10k.61/latest.ckpt` |
| 70.027 | 37.063 | 7 | `ispd2005-s0.fm_p1_nt50_full.300` | 300 | `v1.61-fs.61.fm_p1_500k.61/latest.ckpt` |
| 66.531 | 37.478 | 7 | `ispd2005-s0.svdd_p4_runG.300` | 300 | `large-v2 (paper)` |
| 81.245 | 40.014 | 7 | `ispd2005-s0.fm_p1_nt10_full.300` | 300 | `v1.61-fs.61.fm_p1_500k.61/latest.ckpt` |
| 77.047 | 43.753 | 7 | `ispd2005-s0.fm_p1_full.300` | 300 | `v1.61-fs.61.fm_p1_500k.61/latest.ckpt` |
| 84.854 | 47.796 | 7 | `ispd2005-s0.svdd_p2_runA.300` | 300 | `v1.61-ddpo.ablation_supervised_10k.61/latest.ckpt` |

## ⚠ 中斷的 eval（跑到一半，既不是 done 也不是 orphan）

單張競爭 GPU 上被 kill / OOM / tmux 掉線的 run。
（另有 39 個只跑 adaptec1/bigblue1 的便宜篩選 run，那是刻意的形狀，不列入。）

- `ispd2005-s0.base_cu128_s301.301` — 6/7 (adaptec1,adaptec2,adaptec3,adaptec4,bigblue1,bigblue3)　_2026-09-14_
- `ispd2005-s0.base_cu128_s302.302` — 6/7 (adaptec1,adaptec2,adaptec3,adaptec4,bigblue1,bigblue3)　_2026-09-14_
- `ispd2005-s0.bon2b_s301.301` — 6/7 (adaptec1,adaptec2,adaptec3,adaptec4,bigblue1,bigblue3)　_2026-09-14_
- `ispd2005-s0.bon2b_s302.302` — 6/7 (adaptec1,adaptec2,adaptec3,adaptec4,bigblue1,bigblue3)　_2026-09-14_
- `ispd2005-s0.bon2c_s300.300` — 6/7 (adaptec1,adaptec2,adaptec3,adaptec4,bigblue1,bigblue3)　_2026-09-15_
- `ispd2005-s0.bon2c_s301.301` — 6/7 (adaptec1,adaptec2,adaptec3,adaptec4,bigblue1,bigblue3)　_2026-09-15_
- `ispd2005-s0.bon2c_s302.302` — 6/7 (adaptec1,adaptec2,adaptec3,adaptec4,bigblue1,bigblue3)　_2026-09-15_
- `ispd2005-s0.runF_cu128_s301.301` — 6/7 (adaptec1,adaptec2,adaptec3,adaptec4,bigblue1,bigblue3)　_2026-09-14_
- `ispd2005-s0.runF_cu128_s302.302` — 6/7 (adaptec1,adaptec2,adaptec3,adaptec4,bigblue1,bigblue3)　_2026-09-14_
- `ispd2005-s0.svdd_cu128_s301.301` — 6/7 (adaptec1,adaptec2,adaptec3,adaptec4,bigblue1,bigblue3)　_2026-09-14_
- `ispd2005-s0.svdd_cu128_s302.302` — 6/7 (adaptec1,adaptec2,adaptec3,adaptec4,bigblue1,bigblue3)　_2026-09-14_
- `ispd2005-s0.bon2a_legall.300` — 1/7 (bigblue4)　_2026-09-14_
- `ispd2005-s0.deepK150_acf05.300` — 1/7 (bigblue4)　_2026-09-14_
- `ispd2005-s0.deepK150_acf10.300` — 1/7 (bigblue4)　_2026-09-14_
- `ispd2005-s0.deepK60_acf05.300` — 1/7 (bigblue4)　_2026-09-14_
- `ispd2005-s0.deepK60_acf10.300` — 1/7 (bigblue4)　_2026-09-14_
- `ispd2005-s0.diag1A_ddpm_none.300` — 1/7 (bigblue4)　_2026-09-14_
- `ispd2005-s0.diag1A_ddpm_none_bb4_s301.301` — 1/7 (bigblue4)　_2026-09-14_
- `ispd2005-s0.diag1A_ddpm_none_bb4_s302.302` — 1/7 (bigblue4)　_2026-09-14_
- `ispd2005-s0.diag1A_fm_none.300` — 1/7 (bigblue4)　_2026-09-14_
- `ispd2005-s0.diag1A_fm_none_bb4_s301.301` — 1/7 (bigblue4)　_2026-09-14_
- `ispd2005-s0.diag1A_fm_none_bb4_s302.302` — 1/7 (bigblue4)　_2026-09-14_
- `ispd2005-s0.diag1B_none_eta00_T1000.300` — 1/7 (bigblue4)　_2026-09-14_
- `ispd2005-s0.diag1B_none_eta00_T1000_bb4_s301.301` — 1/7 (bigblue4)　_2026-09-15_
- `ispd2005-s0.diag1B_none_eta00_T1000_bb4_s302.302` — 1/7 (bigblue4)　_2026-09-15_
- `ispd2005-s0.diag1B_opt_eta00_T1000.300` — 1/7 (bigblue4)　_2026-09-14_
- `ispd2005-s0.diag1B_opt_eta00_T100.300` — 1/7 (bigblue4)　_2026-09-14_
- `ispd2005-s0.diag1B_opt_eta05_T1000.300` — 1/7 (bigblue4)　_2026-09-14_
- `ispd2005-s0.diag1B_opt_eta05_T100.300` — 1/7 (bigblue4)　_2026-09-14_
- `ispd2005-s0.diag1B_opt_eta10_T100.300` — 1/7 (bigblue4)　_2026-09-14_
- `ispd2005-s0.frameavg.300` — 1/7 (bigblue4)　_2026-09-15_

## ⚠ 未收割：訓練完成但從未 eval 的 checkpoint

- `v1.61-fs.61.aug_p1_C_dihedral_dropout_500k.61`　_2026-07-08_
- `v1.61-fs.61.aug_p1_ctrl.61`　_2026-07-08_
- `v1.61-fs.61.aug_p1_pilot2.61`　_2026-07-08_　（pilot，可能不需要）
- `v1.61-fs.61.aug_p1_pilot.61`　_2026-07-08_　（pilot，可能不需要）
- `v1.61-fs.61.fm_p1_pilot.61`　_2026-07-06_　（pilot，可能不需要）
- `v2.61.fs_p1_X_stage2_b32_fix_pilot.61`　_2026-06-11_　（pilot，可能不需要）
- `v2.61.fs_p1_X_stage2_b4_pilot.61`　_2026-05-29_　（pilot，可能不需要）
- `v2.61.fs_p1_X_stage2_b16_pilot.61`　_2026-05-29_　（pilot，可能不需要）
- `v1.61-fs.61.fs_p1_Y_pilot_v3.61`　_2026-05-26_　（pilot，可能不需要）
- `v1.61-fs.61.fs_p1_X_pilot_v3.61`　_2026-05-26_　（pilot，可能不需要）
- `v1.61-fs.61.fs_p1_Y_pilot_v2.61`　_2026-05-26_　（pilot，可能不需要）
- `v1.61-fs.61.fs_p1_X_pilot_v2.61`　_2026-05-26_　（pilot，可能不需要）
- `v1.61-ddpo.addloss_v1.61`　_2026-04-15_
- `v1.61-ddpo.ddpo_v2_5_last_k.61`　_2026-04-14_
- `v1.61-ddpo.ddpo_v2_4_local_legality.61`　_2026-04-14_
- `v1.61-ddpo.ddpo_v2_local.61`　_2026-04-09_
- `v1.61-ddpo.ablation_supervised_only.61`　_2026-04-08_
- `v1.61-ddpo.ddpo_v1_lr1e5.61`　_2026-04-03_
- `v1.61-ddpo.ddpo_v1_test.61`　_2026-04-03_　（pilot，可能不需要）

## ⚠ 未記錄：有完整結果，但這個數字在 docs/ 裡找不到

以 avg7 數值比對（2/3 位小數）而非目錄名，因為文件裡引用結果用的是方法名不是路徑。

- 無

## 判讀規則（同時明文寫在 CLAUDE.md，工具關掉也有效）

- 已測得的最大 across-seed σ = **0.654**（SVDD, n=3）。7-circuit avg 差距小於 **1.3** 的結果**不得用來關閉任何研究方向**。
- 任何 n=1 的結果都不能成為禁令。禁令要分 **evidential**（有 Δ、有 seed 數）與 **prudential**（成本效益判斷）— 只有前者需要過統計門檻，後者要明說是判斷不是證據。
- 決策 band 不得錨定在 n=1 或原始資料已遺失的基準數字上。
- 任何方法只能對**同 stack、我們自己跑的 paper checkpoint** 比較（上方 anchor），不能對 46.89（paper 已發表值）或任何硬編碼舊數字比較。換 GPU/torch = 換 random stream，跨 stack 不可混比。

