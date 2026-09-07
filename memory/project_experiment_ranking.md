---
name: ISPD2005 method ranking (as of 2026-04-18)
description: Current leaderboard of fine-tune methods on 7-circuit avg HPWL (no bigblue2), baseline vs paper vs our variants
type: project
---

7-circuit avg HPWL (x10^5, no bigblue2) 目前排名:

1. **Ablation 10k** — 44.01（supervised fine-tune only, 目前最佳）
2. DDPO v2 (PPO+supervised) — 44.65
3. **AddLoss v2** (timestep weighting) — 45.24（新，2026-04-18 完成 eval）
4. AddLoss v1 — 45.39
5. Ablation 5k — 45.43
6. Paper — 46.89
7. Baseline (large-v2.ckpt) — 48.69

**Why:** AddLoss v1/v2 雙雙輸給 supervised-only，ground truth 已是好 target 的假說基本成立，ReFL-style direct gradient 在此 task 上是冗餘訊號。

**How to apply:** 下次 session 討論新實驗方向時，優先考慮 Best-of-N self-improve 或 bigblue2 cluster-aware guidance；不要再投資 pure AddLoss 變體。檢查最新結果用 `docs/report/`。
