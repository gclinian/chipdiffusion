# 這個目錄已停用（2026-09-07）

原本這裡放的是專案記憶（leaderboard、方法排名、當前方向）。**兩個問題**：

1. **工具讀不到。** Claude Code 自動載入的記憶在
   `~/.claude/projects/-ibmnas-427-r115-gclin-Desktop-chipdiffusion/memory/`，
   不是 repo 內。所以這裡維護的東西從來沒被自動讀進任何 session。
2. **它會爛掉。** 到 2026-09 為止，這裡仍宣稱「Ablation 10k 44.01 是最佳方法」、
   「SVDD implementation pending」— 而 SVDD 早在 5 月就完成並勝出。手寫的數字
   索引一定會過期，因為數字變動比維護意願快。

## 現在數字放哪裡

| 你想知道 | 去讀 |
|---|---|
| 現況 leaderboard、哪些結果沒收割 | `STATUS.md`（自動產生，自帶時間戳） |
| 完整結果表（含只存在於報告裡的歷史值） | `docs/all_experiments_summary.csv` |
| 每個 run 的原始 metrics | `docs/ledger/results/<run_id>.csv` |
| 敘事脈絡、已關閉方向與其證據 | `docs/context.txt` |

重新產生 STATUS.md：`python3 scripts/ledger.py scan`

舊內容在 git 歷史裡（commit 300eb49 之前）。
