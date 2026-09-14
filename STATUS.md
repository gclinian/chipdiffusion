# STATUS — 自動產生，不要手改

由 `python3 scripts/ledger.py status` 產生於 **2026-09-14 10:03**。
來源：`docs/ledger/runs.jsonl`（130 runs）+ `docs/ledger/results/`（逐筆 metrics.csv 存檔）。

環境指紋：GPU `NVIDIA GeForce RTX 5090, 32607 MiB` ・ git `f0871b4`（**working tree 有未 commit 變更**）

## Leaderboard（7-circuit avg HPWL，排除 bigblue2，越低越好）

參考點：paper **46.89** ・ 我們對 paper checkpoint 的復現 **48.691**

這張表只含**磁碟上還有 metrics.csv 的 run**。有些歷史結果（Ablation_10k 44.01、DDPO v2 44.65、AddLoss v1 等）的原始檔已被 eval 目錄碰撞覆蓋，只存在於報告中 — 那些要看 `docs/all_experiments_summary.csv`。兩張表不一致是預期的，差異本身就是資訊。

| avg7 | n | run group | seed | checkpoint |
|-----:|--:|-----------|------|------------|
| 44.103 | 7 | `ispd2005-s0.svdd_p5_runH_s301.301` | 301 | `large-v2 (paper)` |
| 44.321 | 7 | `ispd2005-s0.svdd_p3_runF.300` | 300 | `v1.61-ddpo.ablation_supervised_10k.61/latest.ckpt` |
| 44.485 | 7 | `ispd2005-s0.svdd_p3_runE.300` | 300 | `v1.61-ddpo.ablation_supervised_10k.61/latest.ckpt` |
| 44.692 | 7 | `ispd2005-s0.code_p1_s301.301` | 301 | `large-v2 (paper)` |
| 44.747 | 7 | `ispd2005-s0.eval_macro_only.600` | 600 | `v1.61-ddpo.ddpo_v2_ppo.61/latest.ckpt` |
| 44.772 | 7 | `ispd2005-s0.tds_p1_s302.302` | 302 | `large-v2 (paper)` |
| 44.973 | 7 | `ispd2005-s0.tds_p1_s300.300` | 300 | `large-v2 (paper)` |
| 45.053 | 7 | `ispd2005-s0.fs_p1_X_full.300` | 300 | `v1.61-fs.61.fs_p1_X_500k.61/latest.ckpt` |
| 45.078 | 7 | `ispd2005-s0.fs_p1_X_1.6M_full.300` | 300 | `v1.61-fs.61.fs_p1_X_3M.61/latest.ckpt` |
| 45.097 | 7 | `ispd2005-s0.svdd_p4_runH.300` | 300 | `large-v2 (paper)` |
| 45.121 | 7 | `ispd2005-s0.fs_p1_Y_full.300` | 300 | `v1.61-fs.61.fs_p1_Y_500k.61/latest.ckpt` |
| 45.236 | 7 | `ispd2005-s0.eval_macro_only.300` | 300 | `v1.61-ddpo.addloss_v2.61/latest.ckpt` |
| 45.335 | 7 | `ispd2005-s0.svdd_p5_runH_s302.302` | 302 | `large-v2 (paper)` |
| 45.361 | 7 | `ispd2005-s0.code_p1_s300.300` | 300 | `large-v2 (paper)` |
| 45.499 | 7 | `ispd2005-s0.tds_p1_s301.301` | 301 | `large-v2 (paper)` |
| 45.596 | 7 | `ispd2005-s0.code_p1_s302.302` | 302 | `large-v2 (paper)` |
| 45.987 | 7 | `ispd2005-s0.base_cu128_s300.300` | 300 | `large-v2 (paper)` |
| 46.014 | 7 | `ispd2005-s0.eval_macro_only.500` | 500 | `v1.61-ddpo.ddpo_v2_ppo.61/latest.ckpt` |
| 46.380 | 7 | `ispd2005-s0.eval_macro_only.400` | 400 | `v1.61-ddpo.ddpo_v2_ppo.61/latest.ckpt` |
| 46.441 | 7 | `ispd2005-s0.fs_stage2_full.300` | 300 | `v2.61.fs_p1_X_stage2_b32.61/latest.ckpt` |
| 66.531 | 7 | `ispd2005-s0.svdd_p4_runG.300` | 300 | `large-v2 (paper)` |
| 70.027 | 7 | `ispd2005-s0.fm_p1_nt50_full.300` | 300 | `v1.61-fs.61.fm_p1_500k.61/latest.ckpt` |
| 71.944 | 7 | `ispd2005-s0.svdd_p2_runC.300` | 300 | `v1.61-ddpo.ablation_supervised_10k.61/latest.ckpt` |
| 77.047 | 7 | `ispd2005-s0.fm_p1_full.300` | 300 | `v1.61-fs.61.fm_p1_500k.61/latest.ckpt` |
| 81.245 | 7 | `ispd2005-s0.fm_p1_nt10_full.300` | 300 | `v1.61-fs.61.fm_p1_500k.61/latest.ckpt` |
| 84.854 | 7 | `ispd2005-s0.svdd_p2_runA.300` | 300 | `v1.61-ddpo.ablation_supervised_10k.61/latest.ckpt` |

## ⚠ 中斷的 eval（跑到一半，既不是 done 也不是 orphan）

單張競爭 GPU 上被 kill / OOM / tmux 掉線的 run。
（另有 28 個只跑 adaptec1/bigblue1 的便宜篩選 run，那是刻意的形狀，不列入。）

- 無

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

- `ispd2005-s0.base_cu128_s300.300` avg7=45.987　_2026-09-13_

## 判讀規則（同時明文寫在 CLAUDE.md，工具關掉也有效）

- 已測得的最大 across-seed σ = **0.654**（SVDD, n=3）。7-circuit avg 差距小於 **1.3** 的結果**不得用來關閉任何研究方向**。
- 任何 n=1 的結果都不能成為禁令。禁令要分 **evidential**（有 Δ、有 seed 數）與 **prudential**（成本效益判斷）— 只有前者需要過統計門檻，後者要明說是判斷不是證據。
- 決策 band 不得錨定在 n=1 或原始資料已遺失的基準數字上。
- 微調結果只能對 48.691（我們自己的復現）比較，不能對 46.89（paper 已發表值）比較。

