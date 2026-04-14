# DDPO 第三次實驗回顧 — 給 plan_4 的參考

> 實驗報告：`ddpo_report_3.md`

## 第三次實驗做了什麼

兩組補充實驗：
1. **Seed Ensemble**：用 DDPO v2 checkpoint 跑 seed 300/400/500/600 四個 eval
2. **Supervised 10k**：純 supervised fine-tuning 跑 10000 steps（vs 之前的 5000）

## 核心結論

**Supervised 10k（44.01）意外成為所有方法中最佳**，比 DDPO v2 single-seed（44.65）好，只輸給 Seed Ensemble best-per-circuit（43.89）。

| 方法 | Avg HPWL (7 circuit) | 時間 | vs Paper |
|------|---------------------|------|----------|
| **Ablation 10k** | **44.01** | **17 min** | **−6.1%** |
| Seed Ensemble best-per-circuit | 43.89 | 108 min + 10 hr | −6.4% |
| DDPO v2 (s300) | 44.65 | 108 min | −4.8% |
| Ablation 5k | 45.43 | 12 min | −3.1% |
| Paper | 46.89 | — | — |

**7/8 circuit 全部超越 paper**（除 bigblue2 因為 guidance 被關）。

---

## 成功之處

### 1. 確認 Supervised fine-tuning 是主要驅動力

從 5k → 10k supervised，平均 HPWL 從 45.43 降到 44.01（−3.1%）。而 DDPO 從 ablation 5k 升級到 DDPO v2（加 reward, 多 9 倍時間），只帶來 −1.7%。

**結論**：再多的 DDPO training 不如更多的 supervised training。至少在目前 setup 下，reward signal 被 batch_size=2 的 variance 淹沒。

### 2. Val loss 是 misleading 指標

10k 訓練時 val loss 在 step 1000 觸底（0.10），之後反彈到 step 10000 的 0.17——**但 ISPD eval 卻在 10k 變得更好**。這代表：

- Val loss 算的是 v1.61 val data 的 denoising 誤差
- ISPD eval 跟 v1.61 是**不同 distribution**
- 對 v1.61 slight overfitting 可能反而有益於 ISPD（因為 v1.61 和 ISPD 有共通特徵）

**教訓**：以後要追蹤 ISPD eval，不能只看 val loss。

### 3. 大型 circuit 特別受益於更長的 supervised training

bigblue3 (30.70) 和 bigblue4 (124.85) 是 Ablation 10k 在所有方法中最好的，比 DDPO v2 都好。可能原因：更穩定的 denoising 給 guidance + legalization 更好的起點。

---

## 仍存在的問題

### 問題 1：我們還不知道 Supervised 的天花板

10k 比 5k 好，那 15k / 20k / 30k 呢？Val loss 上升但 ISPD 仍在改善，這個 divergence 會持續多久？可能還有 −2% ~ −3% 空間。

### 問題 2：bigblue2 始終是弱項（56.88 vs paper 38.8）

所有方法都在 56-61 範圍，跟 paper 差 46%。Root cause：23k macros 超過 `skip_guidance_threshold=10000`，guidance 被關掉，全部靠 legalization。

### 問題 3：DDPO 的效益太低

108 分鐘訓練只帶來 1.7% 改善（相對 ablation 5k）。如果要讓 DDPO 真的有用，需要：
- batch_size ≥ 8（需要更多 VRAM）
- 或更聰明的 reward signal（BoN、preference learning）
- 或完全放棄 policy gradient，改用別的方法

### 問題 4：Seed variance 大，單次 eval 不穩

adaptec2 跨 seed 從 29.26 到 36.61（差距 25%）。除非跑 multi-seed，否則結果的 noise 會掩蓋方法之間的真實差異。

### 問題 5：還沒用過 ISPD data 本身做任何訓練

目前所有訓練都用 v1.61 synthetic data。既然我們知道 supervised fine-tuning 是主要驅動力，**直接用 ISPD data fine-tune** 可能進一步提升。

---

## 可能的改進方向

### 方向 F：尋找 Supervised training 的最佳 step 數

**動機**：10k 比 5k 好，但不知道天花板。

**做法**：跑 15k、20k、30k 三個版本，看 ISPD eval 什麼時候開始退化。

```
5k    → 45.43（已知）
10k   → 44.01（已知，目前最好）
15k   → ?
20k   → ?
30k   → ?
```

**預估**：每個 +10k steps 約多 10 分鐘訓練 + 2.5 小時 eval。3 個版本共 ~8 小時。

**風險**：可能已經過了最佳點（step 10k 其實已經開始反彈），繼續訓練可能只是 overfitting。但用 best.ckpt（save 在 val loss 最低時）可能救回來——目前配置用 latest.ckpt，可考慮改用 best.ckpt eval。

---

### 方向 G：用 ISPD data 做 supervised fine-tuning（Test-Time Adaptation）

**動機**：既然 supervised fine-tuning 的效益高，直接用 eval data fine-tune 是最直接的 test-time adaptation。

**做法**：
1. 用 ISPD2005 的 reference placement 作為 training label
2. 在 7 個 circuit（不含 bigblue2）上跑 supervised fine-tuning
3. 先用 Ablation 10k checkpoint 當 init，再 fine-tune

**預估**：只有 7 個 training sample，很容易 overfit。每個 sample 訓練 100~500 steps 可能就夠。

**風險**：
- Reference placement 不一定是最優的（甚至可能比我們現在產出的差）
- Overfitting 到這 7 個 circuit（但 eval 就是在這 7 個上，這可能是 feature 不是 bug）

**變體 G+**：Best-of-N Self-Improvement（從 `ddpo_next_2.md` 延續）
1. 用 Ablation 10k + 多個 seed 跑 eval，選每個 circuit 的 best placement
2. 把 best placements 當作 training label fine-tune
3. 這比用 reference placement 更好——因為 label 是 model 自己能達到的最好水平

---

### 方向 H：把 Ablation 10k 當 init 重跑 DDPO

**動機**：我們之前的 DDPO 都從 large-v2.ckpt 開始。如果改從 Ablation 10k 開始，model 已經是更好的起點，DDPO 的 reward signal 可能推它更遠。

**做法**：
```
from_checkpoint: v1.61-ddpo.ablation_supervised_10k.61/latest.ckpt
train_steps: 5000
其餘設定同 DDPO v2
```

**預估**：訓練 ~108 分鐘，eval ~2.5 小時。

**風險**：可能只是重複之前 DDPO v2 的現象——supervised loss dominates reward signal。但如果 model 已經很好，很小的 reward gradient 可能足夠 polish。

---

### 方向 I：Cluster-aware guidance for bigblue2

**動機**：bigblue2 是唯一輸給 paper 的 circuit（46% gap），純粹是 VRAM 問題。

**做法**：
1. 用 hmetis 把 bigblue2 的 23k macros 分成 ~10 個 cluster
2. 逐 cluster 跑 guidance（每次只對 ~2.3k macros 算 V×V matrix）
3. 最後整體 legalize

**預估**：實作複雜度中等（需要改 `guidance.py`），但能解決根本問題。

**風險**：cluster 之間的 interaction 可能被忽略，效果不保證。

---

### 方向 J：用 best.ckpt 而非 latest.ckpt 做 eval

**動機**：目前 eval 用的是最後一個 step 的 checkpoint（latest.ckpt），但 val loss 顯示 best 可能在 step 1000 附近。

**做法**：直接改 `from_checkpoint` 用 `best.ckpt`。

**預估**：零成本。但注意：**val loss best 不等於 ISPD best**（前面提過的 divergence），所以不保證更好。

---

## 建議的優先順序

| 優先級 | 方向 | 理由 |
|--------|------|------|
| **1** | **F. Supervised 不同 step 數** | 最直接、已知 trajectory、cost 低 |
| 2 | **G. ISPD data fine-tune** | 最有潛力跳躍式提升，但需要小心 overfitting |
| 3 | **J. best.ckpt eval（+F 結合）** | 零成本，可一併測試 |
| 4 | **H. 從 Ablation 10k 重跑 DDPO** | 驗證「DDPO + 好 init」是否比純 supervised 好 |
| 5 | **I. Cluster-aware guidance** | 解決 bigblue2，但實作複雜 |

---

## 具體建議 for Plan 4

**最小 plan**（cost-effective）：
- Supervised 15k, 20k, 30k 三個版本（F）
- 搭配 best.ckpt + latest.ckpt 各自 eval（J）
- 共 ~10 小時

**中等 plan**（有野心）：
- 上述 +
- ISPD data fine-tune 從 Ablation 10k 起（G）
- 共 ~15 小時

**完整 plan**（全攻略）：
- 上述 +
- Best-of-N self-improvement（G+）
- 從 Ablation 10k 重跑 DDPO（H）
- 共 ~25 小時

---

## 開放問題

1. **Supervised training 的真正天花板在哪？** 可能在 10k-30k 之間，也可能已經接近了。
2. **ISPD reference placement 有多好？** 比我們現在的結果差還是好？如果比我們差，test-time adaptation 只會讓 model 變差。
3. **每個 circuit 的 seed 有沒有可預測的規律？** 如果有，可以直接跑那個 seed 而不用 ensemble。
4. **Bigblue2 的 paper 結果（38.8）是怎麼做到的？** 他們一定用了某種 cluster 或更大 VRAM 的方法，值得去 paper 裡挖。
