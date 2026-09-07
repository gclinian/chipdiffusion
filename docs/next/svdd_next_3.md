# SVDD Phase 3-5 完整回顧（next_3）

> Reports：`docs/report/svdd_report_{3,4,5}.md`
> Plans：`docs/plan/svdd_plan_3.md`（Phase 4-5 沒寫 plan）
> 寫作日期：2026-05-24

## 1. 一句話總結

**Phase 3 結論「SVDD dead end」是 framing 錯誤造成的偽結論**。Phase 4 修正 framing（在 paper 真正的 large-v2 checkpoint 上跑），Phase 5 multi-seed 驗證，**最終結論翻盤：SVDD layered on opt 3-seed mean 44.85 ± 0.65，穩定贏 paper baseline 46.89 −4.4%（3/3 seeds 全贏）**。leaderboard rank 5，唯一不靠 fine-tuning 而贏 paper 的方法。

---

## 2. 三個 phase 的故事線

### Phase 3（我，2026-05-21 早）：得出錯誤結論「SVDD dead end」

Setup：在 **ablation_10k checkpoint** 上比 Run F (opt only = 44.32) vs Run E (svdd_layered = 44.49)。
- Δ = +0.37% → 套用 plan_3 §5.2 判定「持平 → 放棄 inference-time search」
- 寫 report_3 §11 結論「Inference-time search 整個方向（SVDD/CoDe/TDS）正式 dead end」

**錯誤**：我把「ablation_10k + opt = 44.32」當成 paper baseline，但 ablation_10k 是使用者**自己 fine-tune 出來**的最佳 checkpoint，paper 用的是 **large-v2 + opt = 46.89**。我寫進 next_2 §3.1 還振振有詞「Paper 用 ablation_10k + opt → 44.01」── 完全錯。

### Phase 4（同日下午，使用者糾正後）：single-seed 翻盤

使用者點出：「ablation 是你自己 finetuning 的結果，paper baseline 不是這個」 → 我才意識到 framing 錯誤。

修正：在 **large-v2（paper 真正 checkpoint）** 上重跑：
- Run G (large-v2 + svdd only) = 66.53 → 顯著輸 paper 46.89
- **Run H (large-v2 + svdd_layered) = 45.10 → 贏 paper -3.8%**

但只有 seed=300 single-seed，bigblue3 +14% 退步可疑。

### Phase 5（2026-05-24，multi-seed）：驗證 + 推翻 bigblue3 假說

跑 seed=301, 302 補上：
- s301 avg = 44.10（最強，贏 paper −5.9%）
- s302 avg = 45.34（贏 paper −3.3%）
- **3-seed mean = 44.85 ± 0.65, 贏 paper −4.4%**
- bigblue3 3-seed mean = 34.83，**反而贏 paper -3.0%**（s300 的 +14% 純 bad luck）
- bigblue4 3-seed 一致大贏 −10.8%

統計顯著：95% CI 上界 45.59 < paper 46.89。

---

## 3. Plan_3 設計缺陷的根源

Plan_3 §1 只用一句話定義「對照」：
> 「Paper 方法（ablation_10k + opt + legalizer）= 44.01（CLAUDE.md 公告）」

**這句話本身就是錯的**。CLAUDE.md L100 明白寫 `large-v2.ckpt` 的 baseline avg = 48.69 (our env) / 46.89 (paper)。CLAUDE.md L32-33 寫 "Ablation 10k... beats paper's 46.89 by 6.1%" — 「beats paper's 46.89」就是承認 paper 是 46.89 不是 44.01。

我寫 plan_3 時懶得區分「我們最佳 fine-tune 結果」vs「paper 真正 baseline」，把兩者混為一談 → 之後整個 Phase 3 都在做錯的對照。

**lesson**：寫 plan 時對「baseline」要明確指 paper 還是自己的 best result，不能模糊。

## 4. 推翻 / 修正 之前的 next 文件

### next_2 §3.1 該作廢

寫的：「Paper 用 ablation_10k + opt guidance + opt-adam legalization → 44.01。我們把 opt 拔掉之後跟 SVDD 比，等於是 strawman。」

**錯**：paper 不是用 ablation_10k。應該說：「我們在 ablation_10k 上比 SVDD 跟 opt，沒有對應到 paper baseline。要 vs paper 必須在 large-v2 上跑。」

### next_2 §4 「保留的 / 改變的」要重排

next_2 §4 寫「保留 ablation_10k 為 init checkpoint」——這正好是讓 SVDD 沒空間 search 的選擇。如果當時就在 large-v2 上做 plan_3，可以省一個 phase。

---

## 5. 學到的事

### 5.1 「Headroom 假說」是真實的

| Checkpoint | opt baseline | + svdd_layered | Δ |
|---|---:|---:|---:|
| large-v2 (paper raw) | 48.69 | **44.85** | **-7.9%** |
| ablation_10k (fine-tuned) | 44.32 | 44.49 | +0.4% |

Fine-tuning 已把 placement 推到接近 ~44 floor，SVDD 的 K=4 jitter 沒空間。raw checkpoint 有 ~4 unit headroom，SVDD 有空間 search。

**Insight**：往後評估「inference-time intervention」時，要考慮 starting checkpoint 強度。對強 ckpt 沒增量不代表方法本身無效。

### 5.2 Multi-seed 是 chip placement 評估的 must-have

Phase 4 single-seed bigblue3 +14% 看起來像「結構性退步」，差點寫進 report 變成永久結論。Phase 5 multi-seed 一跑，−3.0% 變成「實際上贏」。

bigblue3 std = 5.34（其他 circuit < 2.6），表示這個 circuit 對 SVDD K-particle 採樣的 noise 特別敏感。**single-seed 不能下結論**。

CLAUDE.md context.txt 早就講過「seed variance is significant (adaptec2: 29.26-36.61 across 4 seeds)」── 我們之前 phase 3 也是 single-seed，運氣好沒踩到。

往後 plan 規定：**每個主結論至少 3 seed**。

### 5.3 Framing 錯誤比實驗結果還危險

Phase 3 整套實驗本身沒有技術錯誤（程式碼、執行、判定基準都對）── 錯在 plan_3 對「對照組是什麼」的定義錯了。如果使用者沒 catch（"那個是我之前 finetuning 的結果"），這個錯誤會 propagate 進 next_3、plan_4，最後寫進 paper 都不知道。

**lesson**：寫 plan 時引用的數字一定要對應到 CLAUDE.md 原文，不要靠記憶。

### 5.4 程式碼層的小坑都已修

- NaN softmax fix（Phase 1 收斂）
- K=8 PyG batched bug（沒修，K=4 規避）
- CUDA error 在 GPU 0 contention（用 `CUDA_VISIBLE_DEVICES=1` 規避）
- 沒有新的程式碼 bug 從 Phase 3-5 跑出來

---

## 6. SVDD 系列現狀整理

### 6.1 已驗證的事實

| 事實 | 證據 |
|---|---|
| SVDD-PM 機制可運作（K=4, every_n=1, λ=1, α_temp=1） | Phase 1-5 都跑得通 |
| SVDD 單獨**不能**取代 opt guidance | Run G 66.53 vs Run F 44.32 (ablation_10k), 66.53 vs 48.69 (large-v2) |
| SVDD layered on opt 在 paper checkpoint 上 **贏 paper** | Run H 3-seed 44.85 vs paper 46.89, Δ −4.4% (3/3 seeds) |
| SVDD layered on opt 在 fine-tuned checkpoint 上 **無增量** | Phase 3 Run E 44.49 vs Run F 44.32, Δ +0.4% |
| bigblue4 是 SVDD 最大受益 circuit | -10.8% 3-seed mean，所有 seeds 一致 |
| bigblue3 對 SVDD K-particle variance 敏感 | std 5.34，需 multi-seed 才看得到真貌 |

### 6.2 還沒驗證的

| 待驗證 | 方法 | 成本 |
|---|---|---|
| Run F (paper opt) 自己也要 multi-seed | 跑 large-v2 + opt + leg s301, 302 | ~120 min |
| λ 對 large-v2 + svdd_layered 的影響 | sweep λ ∈ {0, 0.5, 1, 2, 5} | ~5 hr |
| K 對結果的影響（K=2, 4, 8） | K=8 要先修 PyG bug 或 K_chunk | ~3 hr + bug fix |
| every_n 在 large-v2 setup 是不是還是 1 最好 | sweep every_n ∈ {1, 2, 5} | ~3 hr |
| SVDD 加在 fine-tune 過程中？（vs inference-time） | 改 train_graph.py | 1-2 天 |

### 6.3 SVDD 對 paper 的最終定位

「**Inference-time guidance method that improves over paper's opt-only guidance on the paper's own checkpoint by 4.4% (3-seed mean), without any model fine-tuning, retraining, or extra parameters.**」

leaderboard rank 5 = 跟 fine-tune 方法平起平坐的水準，但不需要訓練。

---

## 7. Plan_4 應該回答的問題

從這個 retrospective 出來，下一個 plan 的可能方向（不互斥，可平行）：

### 7.1 把 SVDD 推到極限（continue SVDD track）

- λ sweep：adaptec2 (+7.8%) / adaptec3 (+1.1%) 是 SVDD 唯一輸 paper 的兩個 circuit。可能 SVDD reward 在這兩個 circuit 跟 opt 互相干擾，調 λ_legality 或 λ_hpwl ratio 可能補回 → 預期能把 mean 從 44.85 推到 ~44.5
- multi-seed Run F (paper opt)：補 paper-equivalent baseline 也跑 multi-seed → 真正同等條件比較
- K 跟 every_n 的 sweep（cost / benefit Pareto）

### 7.2 探索其他 guidance 方法（survey 還沒測的）

Survey 列了多種方法（CoDe / TDS / SMCDiff / DRaFT-K / FreeDoM / Reflected Diffusion / Mirror Diffusion）── SVDD 是其中 Tier 1.1，**其他幾個還沒測**。
- 如果 SVDD-PM 在 large-v2 已經 -4.4%，更強的 SMC 變種（TDS）可能能再多 -1~-2%
- CoDe（純 best-of-N）是 SVDD 的下界 baseline，可以對照「value-based vs random search」差多少
- 詳見下面 §8 給使用者的 survey 回顧

### 7.3 教授建議 (1) 從頭 train

跟 SVDD 平行，獨立的長 track。需要寫 `from_scratch_plan_1.md`。

### 7.4 寫 paper / 跟教授 demo

3-seed 結果 + headroom theory 已經是可講的 narrative。

---

## 8. 不要做的事

- ❌ 不要再用 single-seed 下結論 ── 至少 3 seed
- ❌ 不要在 fine-tuned ckpt 上測 inference-time guidance（headroom 已滿）── 用 large-v2 才有意義
- ❌ 不要用 ablation_10k 當「paper baseline」── paper 是 46.89 (large-v2 + opt + leg)
- ❌ 不要去修 K=8 PyG batched bug（K=4 + chunk 夠用）

## 9. Open questions for plan_4

1. **使用者想要 SVDD 繼續榨（λ / K sweep）還是探索其他 guidance 方法？** ← 使用者剛剛問了這個，下面 §8 給選項
2. paper baseline 要不要也 multi-seed？（目前只有 seed=300 = 48.69, 跟 paper seed=400 = 46.89 對照）
3. adaptec2 / adaptec3 為什麼 SVDD 輸 paper？需要 single-circuit 分析還是 sweep λ 自動修？
4. 教授看到 -4.4% 會傾向「寫 paper」還是「再榨一點」？
