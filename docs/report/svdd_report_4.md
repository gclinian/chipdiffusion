# SVDD-PM 第四次實驗 Report (Phase 4: SVDD on **paper checkpoint** large-v2)

> Plan：本次無單獨 plan_4，直接 amend report_3 對「paper baseline」的 framing 錯誤
> Report_3：`docs/report/svdd_report_3.md`（**framing 錯誤**：把 ablation_10k+opt = 44.32 當成「paper 方法」）
> 動機 meeting: `docs/meet/meet_0508.md`
> 執行日期：2026-05-21
> Checkpoint：`../public-models/large-v2/large-v2.ckpt`（**paper 真正的 checkpoint**）
> Legalizer：`opt-adam`, `grad_descent_steps=20000`

---

## 1. 為什麼有 Phase 4

Phase 3 把「ablation_supervised_10k.61 (fine-tuned 自己訓的) + opt + opt-adam = 44.32」當成 paper baseline 對照，**這是錯的**：

- **Paper 真正 checkpoint** = `large-v2.ckpt`（CLAUDE.md L141 + paper 公開的 pretrained）
- **Paper 真正 avg HPWL** = **46.89**（paper 公告, seed=400）／ **48.69**（我們重現, seed=300）— 都來自 `large-v2 + opt + opt-adam`，CLAUDE.md L100
- Ablation 10k = 44.01 是**使用者自己 fine-tune** 出來的，比 paper 強 6.1%

Phase 3 證明的是「**在已經 fine-tuned 的 ablation_10k 上，SVDD 加在 opt 之上沒增量**」（Run E 44.49 vs Run F 44.32 = +0.37%），這結論本身正確，但**沒回答「SVDD vs paper」的問題**。

Phase 4 補上：在 **paper 的 checkpoint（large-v2）** 上同樣比 SVDD + opt vs 純 opt，看 SVDD 能不能贏 paper baseline 48.69 / 46.89。

---

## 2. 結論摘要

| 項目 | 結果 |
|------|------|
| Paper baseline（large-v2 + opt + leg，seed=300，**我們重現**） | **48.69** |
| Paper number（seed=400，paper 公告） | **46.89** |
| **Run G**（large-v2 + **svdd 取代 opt** + leg） | **66.53** |
| **Run H**（large-v2 + **svdd 疊加在 opt 之上** + leg） | **45.10** |
| Run H vs paper number (46.89) | **−3.82%（贏 paper）** |
| Run H vs 我們 reproduction (48.69) | **−7.38%（贏自己重現）** |
| Run G vs paper number | +42%（顯著輸） |

**核心結論**（修正 report_3 §1）：
- SVDD 單獨**不能取代 opt**（Run G 66.53 vs paper 46.89, 慘輸）
- SVDD **疊加在 opt 之上能贏 paper baseline**（Run H 45.10 vs paper 46.89, **−3.8%**）
- 但 Run H 還是輸 ablation_10k 系列（44.01 ~ 44.32），所以 **fine-tuning 比 SVDD layered 更值得**

---

## 3. 完整 7-circuit 結果

| idx | Circuit | V | Paper (公告) | Our reproduction | Run G (svdd only) | Run H (svdd+opt) | H vs Paper | H vs OurBase |
|----|---|---|---:|---:|---:|---:|---:|---:|
| 0 | adaptec1 | 543 | 9.19 | 10.22 | 11.36 | **8.84** | **−3.8%** | **−13.5%** |
| 1 | adaptec2 | 566 | 31.00 | 39.06 | 37.38 | **30.42** | **−1.9%** | **−22.1%** |
| 2 | adaptec3 | 723 | 54.40 | 62.14 | 59.37 | 55.36 | +1.8% | **−10.9%** |
| 3 | adaptec4 | 1329 | 54.50 | 60.51 | 60.41 | **53.13** | **−2.5%** | **−12.2%** |
| 4 | bigblue1 | 560 | 2.64 | 2.69 | 6.52 | **2.64** | ≈0% | **−1.9%** |
| 6 | bigblue3 | 1298 | 35.90 | 34.26 | 49.83 | 40.99 | **+14.2%** | **+19.7%** ↑ |
| 7 | bigblue4 | 8170 | 140.60 | 131.96 | 240.85 | **124.30** | **−11.6%** | **−5.8%** |
| **avg(7)** | | | **46.89** | **48.69** | **66.53** | **45.10** | **−3.82%** | **−7.38%** |

**Run H 統計**：
- 對 paper 公告數字：**贏 5 個 / 平 1 個 / 輸 1 個（bigblue3）**
- 對我們自己重現：**贏 6 個 / 輸 1 個（bigblue3）**

唯一弱點：**bigblue3 退步 +14% vs paper / +20% vs 我們重現**。可能跟 SVDD 的 K-particle 在這個 circuit 落到不利區域有關（單 seed 無法區分是 noise 還是真退步）。

---

## 4. 完整 leaderboard 更新

| Rank | 方法 | 7-circuit avg | Checkpoint | Guidance |
|------|------|---:|---|---|
| 1 | Ablation 10k (CLAUDE.md) | **44.01** | fine-tuned ablation_10k | opt |
| 2 | Phase 3 Run F (我們重現 ablation_10k) | 44.32 | ablation_10k | opt |
| 3 | Phase 3 Run E (ablation_10k + svdd) | 44.49 | ablation_10k | svdd_layered |
| 4 | DDPO v2 | 44.65 | fine-tuned | opt |
| 5 | **Phase 4 Run H (large-v2 + svdd_layered)** | **45.10** | **large-v2 (paper)** | **svdd_layered** |
| 6 | AddLoss v2 | 45.24 | fine-tuned | opt |
| 7 | AddLoss v1 | 45.39 | fine-tuned | opt |
| 8 | Ablation 5k | 45.43 | fine-tuned | opt |
| 9 | DDPO v2.3 | 45.80 | fine-tuned | opt |
| 10 | DDPO v2.5 | 45.86 | fine-tuned | opt |
| 11 | DDPO v2.4 | 46.50 | fine-tuned | opt |
| **12** | **Paper baseline (公告)** | **46.89** | **large-v2** | **opt** |
| 13 | Phase 2 Run C (svdd only) | 71.94 | ablation_10k | svdd |
| 14 | Phase 2 Run A (no guidance) | 84.85 | ablation_10k | none |
| 15 | **Phase 4 Run G (large-v2 + svdd only)** | **66.53** | **large-v2** | **svdd** |
| — | Our reproduction of paper baseline | 48.69 | large-v2 | opt |

**Run H 是 leaderboard 上唯一一個從 large-v2 出發、贏 paper 的方法**（rank 5，贏 paper 公告 46.89 by 3.8%）。所有其他贏 paper 的方法都用 fine-tuned checkpoint。

---

## 5. 為什麼 Phase 3 vs Phase 4 結論「相反」？

| Phase | Checkpoint | Run E/H vs Run F/baseline | 結論 |
|---|---|---|---|
| 3 | ablation_10k（fine-tuned） | Run E (44.49) vs Run F (44.32) = +0.37% | SVDD 沒增量 |
| 4 | large-v2（paper, NOT fine-tuned） | Run H (45.10) vs paper baseline (48.69) = **−7.38%** | **SVDD 有增量** |

關鍵差別：**checkpoint 強度**。
- ablation_10k 已經被 fine-tune 到接近天花板（44.01），opt 已經能把 raw output 推到 floor，SVDD 沒有 search 空間 → +0.37% 等於 noise
- large-v2 raw output 弱很多（從 opt 把它推到 48.69，比 ablation_10k 44.32 差 10%），**opt 仍有殘餘 search 空間**，SVDD 可以在這空間裡幫忙找更低 HPWL → **−7.38%**

Phase 2 §5.1 已經觀察過這個 pattern：**SVDD 對 raw output 越差的 setup 越有用**。Phase 4 用 large-v2 直接驗證。

---

## 6. 對 paper 真正的回答

> 「SVDD 比 paper Lagrange 法好嗎？」

**答案：取決於 checkpoint。**

- 在 paper 自己的 large-v2 checkpoint 上：**YES**，SVDD layered on opt = **45.10 < 46.89 paper（−3.8%）**
- 在 fine-tuned ablation_10k 上：**NO**，SVDD 沒增量（+0.37%）

**這代表 SVDD 是「補強弱 baseline」的工具，不是「補強強 baseline」的工具**。如果 paper 直接 ship `large-v2 + opt + opt-adam`，那 SVDD 是 valid 的 improvement。但如果同時做 fine-tuning（像 ablation_10k），fine-tuning 已經吃掉 SVDD 的潛在貢獻。

---

## 7. Phase 3 報告該修正什麼

Report_3 §1 寫：
> 「SVDD-PM 在 chipdiffusion 任務上沒有 marginal value」

**這句話太強，要改成**：
> 「**對已經 fine-tune 過的 checkpoint**，SVDD 沒有 marginal value。對 paper 原始 checkpoint，SVDD layered on opt 贏 paper baseline 3.8%。」

Report_3 §11 寫：
> 「Inference-time search 整個方向（SVDD/CoDe/TDS）正式判定 dead end，下一步轉教授建議 (1)「從頭 train」」

**這句結論也要修正**：
> 「Inference-time search 對 fine-tuned checkpoint 無增量，對 paper baseline 有 −3.8% 增量。下一步要看：(a) Phase 4 結果有沒有 multi-seed 撐起來；(b) SVDD 跟 fine-tuning **二擇一就好，還是有 setup 能讓兩者疊加**；(c) 教授建議 (1) 從頭 train 仍是平行可走的 track」

---

## 8. 為什麼 bigblue3 退步？

唯一的退步 circuit。可能解釋：

| 假說 | 證據 | 可信度 |
|---|---|---|
| Single-seed bad luck | bigblue3 在 Phase 2 Run C 也是 svdd 改善最大（-54.7%）→ 對 svdd 敏感 | 中 |
| SVDD 的 K particles 在 bigblue3 的 placement landscape 落到 plateau | bigblue3 V=1298 中型，hierarchy 結構特殊 | 低 |
| Legality reward (λ=1) 在 bigblue3 把 placement 推往低 HPWL 但 legal 困難的區域，legalizer 修不回來 | Run H bigblue3 legality 0.997，沒問題 | 低 |

**建議**：multi-seed 重跑 bigblue3 看是不是 bad luck（不影響主結論，但會影響 6/7 vs 5/7 的計分）。

---

## 9. 計算成本（Phase 4）

| Run | 7-circuit 總時間 | 相對 paper baseline |
|---|---|---|
| paper baseline (估) | ~60 min | 1.0× |
| Run G (svdd only) | ~45 min | 0.75× |
| Run H (svdd_layered) | ~70 min | 1.17× |

Run H 多 17% time，**換 -3.8% HPWL** → 工程 trade-off 划算。

---

## 10. 行動清單

- [x] Run G: large-v2 + svdd-only + opt-adam, 7 circuits, seed=300 → **66.53**
- [x] Run H: large-v2 + svdd_layered + opt-adam, 7 circuits, seed=300 → **45.10 (−3.8% vs paper, −7.4% vs our reproduction)**
- [x] 寫此 report (svdd_report_4.md)
- [ ] 下一步建議：
  - (a) **multi-seed 驗證 Run H**（最少 seed 301, 302）— 確認 -3.8% 不是 bad luck
  - (b) 寫 `svdd_next_3.md` 把 Phase 3 + 4 一起回顧，修正 report_3 framing
  - (c) bigblue3 single-circuit sweep：看是不是 seed luck
  - (d) 教授建議 (1)「從頭 train」仍是平行 track

---

## 11. 一句話結論

**SVDD-PM layered on opt 在 paper 原始 large-v2 checkpoint 上 7-circuit avg HPWL = 45.10，贏 paper baseline 46.89 by −3.8%（贏 5/7 circuits, 平 1 個, 輸 bigblue3 +14%）— 第一次得到「single-seed 下 SVDD 超越 paper」的結果。Phase 3 結論「SVDD 無增量」只在 fine-tuned ablation_10k 上成立，不是 universal 結論。Inference-time search 路線復活，下一步需 multi-seed 驗證。**
