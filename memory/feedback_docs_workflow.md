---
name: Docs-driven experiment workflow
description: Every experiment follows plan_N → run → report_N → next_N → plan_(N+1) in docs/; reports must compare results to plan's decision criteria
type: feedback
---

實驗流程是 `docs/plan/<name>_plan_N.md` → 執行 → `docs/report/<name>_report_N.md` → `docs/next/<name>_next_N.md` → `docs/plan/<name>_plan_(N+1).md`。所有文件集中在 `docs/`。

**Why:** 使用者靠這套流程維持多次 session 的連續性，report 必須對應 plan 的判定標準（plan 會列出類似「avg < X → 繼續，≥ Y → 放棄」的決策樹）。

**How to apply:** 寫 report 時先讀對應的 plan，在 report 的結論段明確引用 plan 的判定區間（例如「落在 44.5–45.39 區間 → 調整方向繼續」）。不是純寫結果，要幫使用者做出下一步決策。
