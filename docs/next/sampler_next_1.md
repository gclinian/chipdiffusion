# Sampler 診斷與 Best-of-N 回顧（sampler_next_1）— 2026-09-15

> 對應：`docs/plan/sampler_plan_1.md` → `docs/report/sampler_report_1.md`。
> 一輪 ~18 GPU-h（5090），43 個 cell，零失敗，全部 eval-only。

## 一句話
**在修好的 stack 上做同 stack 三 seed 配對檢定後，專案過去五個月的兩個標題主張（微調 −6~9%、
inference-time search −4.4%）都消失了；它們是對一個壞掉的 baseline 比較出來的假象。**
留下的正向結果只有：paper checkpoint 復現 45.99（贏已發表值 2%）、from-scratch Run X 44.65（n=1）、
best-of-4 −3.3%（方向對、n=3 power 不足）。負向結果則非常乾淨，且多數已 n=3 evidential。

## 學到的（evidential，n≥3 或機制性）
1. **Baseline 是所有比較的地基，而它可以壞掉五個月沒人發現。** 48.691 的原始目錄被 eval 碰撞覆蓋、
   config 不可驗、runtime 4.5× 慢、adaptec2 legality 0.93 — 每個訊號都在，沒人重跑。
   教訓已進 CLAUDE.md 規則 4/7：anchor 永遠是「同 stack、我們自己跑的」，不是硬編碼數字。
2. **換 GPU/torch = 換 random stream。** Run F 同 seed 跨 stack 差 +1.38，全落在 σ 最大的兩個 circuit。
3. **同 stack 雜訊校準**：avg6 sd = 0.813（baseline, n=3）；配對設計 sd 0.8–1.3。任何 < 1.5 的差距
   在 n=3 下無法解析。per-circuit：adaptec1/bigblue1 < 1.5%，adaptec4/bigblue3 ~5.5%。
4. **Deterministic sampler 在 size-OOD 崩潰**（1B, n=3, ratio 1.61 ± 0.02）；**FM 的 velocity objective
   再壞 2.2×**（1A, n=3, 2.22 ± 0.20）。有 guidance 時 η=0 在 bb4 反而 −4.9%、a1 +3.4% —
   系統的 OOD 穩健性來自 **stochasticity + guidance**，不在 model。
5. **Few-step 沒有免費午餐**：T=100 在 bigblue4 每個 η 都 +7~14%，只省 43% 時間（legalization 佔大頭）。
6. **Pre-legalization HPWL 不能選候選**：ρ = +0.40（a1 與 bb4），argmin 都選錯。任何 best-of-N 必須
   legalize 每個候選。
7. **更多 guidance step 是 over-guidance**（3E：K=150 bb4 +7%、a1 +15%），與 1B/FM §3.1 同型。
8. **Quality 不隨 macro 數退化**（skeptic 回歸 ledger 全資料）— 我們的劣勢在 543–1,329 macro 的小
   circuit，「size-OOD」是 FM 的故事不是 DDPM+guidance 的。
9. **流程**：`pgrep -f <pattern>` 等待會自我匹配（一個死鎖、兩個提早觸發搶 GPU）→ 佇列一律寫檔、
   PID 等待、skip-if-done。預登記的 legality floor 套錯階段（2b）→ 判定門檻要寫清楚**量在哪一步**。

## 不要做（分類）
### evidential（有 Δ、有 n）
- ❌ Flow matching / drifting / MeanFlow / Shortcut / sCM / IMM / CTM 任何 few-step deterministic 家族
  （1A+1B，n=3；且 T=100 本身就 +7~14%）
- ❌ Log-SNR / t-shift by V（1C：up +5.72，inv −0.33；符號與影像直覺相反）
- ❌ Deep guidance K > 20（3E：最佳 −2.3% n=1，其餘 +3~15%）
- ❌ Post-hoc 權重平均 / 以此推論 EMA 配方（3F：兩 window 都 +0.9~1.9）
- ❌ Frame averaging（3H：−0.4% / +2.4%）
- ❌ Pre-legalization draft 選擇（2a：ρ 0.40）
- ❌ **再宣稱 supervised fine-tuning 或 SVDD/CoDe/TDS 贏 paper**（re-anchor：Δ +0.00 / −0.18，n=3）
### prudential（成本效益判斷，不是證據）
- 不重跑 CoDe / TDS 於新 stack（與 SVDD 統計等價，推定同 null；每個 ~1.5 h）
- 不做 hierarchical / scale conditioning（前提「quality 隨 V 退化」為假；repo 無 macro clustering）
- 不做 attention temperature、AR-hybrid（mask 通道死碼）、MultiDiffusion、RePaint 式 refinement

## 剩餘方向（依價值排序；2026-09-15 Phase 4 後更新）
1. ~~Run X 補 seeds~~ **已做，關閉**：paired Δ +0.36 ± 1.35，44.649 是 seed 300 假警報。
2. **best-of-4 補 seeds 303–305** — 執行中（Phase 4B，n=6 配對 + 95% CI）。若成立，這是唯一可報告的
   推論協定（附成本 4×、legality floor 0.97）；若不成立，本輪沒有任何正向方法結果。
3. ~~DataAug Run C eval~~ **已做，關閉**：45.171 vs Run X 44.649（+0.52）。D4 對稱性在訓練端與推論端
   都不是瓶頸。
4. bigblue2 with guidance on 32 GB（一次 eval，可能 OOM）— 唯一從未贏過 paper 的 circuit，也是唯一
   還沒在乾淨 stack 上測過的東西。
5. 論文框架改寫：reproducibility + 乾淨負向結果（三個獨立實驗證明 auxiliary loss 冗餘；FM/deterministic
   在 size-OOD 崩潰 n=3；fine-tuning、search、from-scratch、augmentation 在同 stack 皆與 paper ckpt
   無差別 — **pipeline 品質由 guidance + legalizer 決定，model 來源幾乎無關**）+ best-of-N（若 4B 成立）。
6. 若 4B 也不成立：下一個值得問的問題不再是「哪個 model」，而是 **guidance / legalizer 本身**
   （57 個 guided run 從沒動過 `grad_descent_rate`、`alpha_critical_factor`、`legality_potential_target`；
   3E 只掃了 K）。

## 對 plan 的修正建議（下輪）
- 判定門檻標明**量測階段**（pre-/post-legalization）與**指標**（avg6/avg7）。
- 預設 3 seeds、6 便宜 circuit；bigblue4 單 seed 只做方向確認。
- 任何 ≥ 1.3 的 n=1 結果自動排 seeds 301/302，不等人決定。
