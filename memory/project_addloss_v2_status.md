---
name: AddLoss v2 experiment status (eval completed 2026-04-18)
description: AddLoss v2 timestep weighting experiment — eval already finished despite earlier belief it was stuck
type: project
---

AddLoss v2（timestep weighting, `weight=α_t²`）的 eval 其實在 Apr 15 20:50 就跑完了（metrics.csv 完整 8 筆，含 bigblue2）。使用者以為被系統負載卡住沒跑完，但實際上 retry 最終完成。

**結果:** 7-circuit avg HPWL 45.24（v1: 45.39, Ablation 10k: 44.01）。訓練指標全面改善（val_loss 0.21→0.18, hpwl_loss 149→103）但 eval 增益極小（-0.33% vs v1）。bigblue2/4/adaptec4 改善，但 bigblue3 退步 15%。

**Why:** 落在 `addloss_plan_2.md` 的「44.5–45.39 → 調整方向繼續」區間，但增益很邊際。AddLoss 方向基本已窮盡。

**How to apply:** Report 在 `docs/report/addloss_report_2.md`，建議下一步走 Best-of-N self-improve 或 bigblue2 cluster-aware（方向 A/D），**不要再做純 AddLoss 變體**。新實驗前先寫 `docs/next/addloss_next_2.md` 與 `docs/plan/<name>_plan_1.md`。
