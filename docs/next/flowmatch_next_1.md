# Flow Matching 回顧（next_1）

> Report: `docs/report/flowmatch_report_1.md`（詳細數據與假說在 §2-3）

## 一句話
FM 同預算下 70.03 vs DDPM 45.05 — 輸在 **OOD 泛化**（V>700 全面崩，bigblue4 2.1× 差），不是 guidance 調參問題。方向關閉，回 DDPM。

## 學到的
1. **Deterministic ODE objective 對 circuit-size 外插比 stochastic denoiser 脆弱** — 訓練 max 400 macros，DDPM 外插 8170 只小退，FM 直接崩。這對「換架構」類提案是通用 caution：先看 OOD，再看 in-distribution 品質。
2. **FM + per-step guidance 會 over-guide**：品質隨 ODE 步數變多而變差（10 步最好）。若未來再碰 deterministic sampler，guidance 要 (a) 只在後段步驟做、(b) 或把 grad_descent_steps 隨步數縮放。
3. **小 circuit 10 步 ≈ DDPM 1000 步**：FM 的 few-step 優勢是真的，只是被 OOD 問題蓋掉。
4. 流程上：agent 執行 Phase A/B 順利（zero rework）；同預算同資料的對照設計讓結論乾淨，一輪就能關方向。

## 不要做
- ❌ Drifting model（同 few-step deterministic 家族，預期同樣 OOD 弱點）——除非先解 OOD
- ❌ FM 加長訓練 / 調 guidance——差距 25 個 HPWL 單位不是這些變因的量級

## 剩餘方向（meet_0620 的另外兩個 + 已完成的極限分析）
- ISPD 極限：oracle 42.09 已答（`ispd_limit_report_1.md`），best-of-4-seeds 是最便宜的正式改進
- Data augmentation（`dataaug_plan_1.md`）：dihedral 8-fold，下一個候選
- EDA dataset（`eda_dataset_plan_1.md`）：工程量大，排最後
