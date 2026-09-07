# SVDD Phase 1 回顧（next_1）

> Report: `docs/report/svdd_report_1.md`
> Plan: `docs/plan/svdd_plan_1.md`
> 寫作日期：2026-05-19

## 1. 一句話總結

機制驗證 OK，但 Phase 1 設計（raw / no legalization / 只跑 2 circuit）讓我們**無法套用 plan_1 §3.2 的判定基準**——必須進 Phase 2 才能下結論。最大意外是 `every_n=1` 在 raw HPWL 上的 **-16.5% 跳水**，這是 plan_1 沒預期到的訊號方向。

## 2. 跟 plan_1 比，什麼預期錯了 / 沒預期到

| plan_1 預期 | 實際 |
|------------|------|
| K=4/8 sweep 找 sweet spot | K=8 在 PyG batched edge_index 直接崩潰，根本沒跑成 |
| 預設 `every_n=5` 是合理起點 | 太保守，`every_n=1` 才是訊號最強的設定 |
| 用 large-v2 → 從 ablation_10k 兩階段 init 對比 | Phase 1 只跑了 large-v2，沒測 ablation_10k init |
| `start_step_frac=0.5` 是合理 default | `start_step_frac=0.3`（更後段）反而退步 |
| Memory 風險主要在 bigblue2 K=8 forward | 真正先觸發的是「**lambda 高 / every_n 小** 時 softmax 數值爆 NaN」 |
| Phase 1 能直接決定是否值得 Phase 2 | Phase 1 完全沒做 legalization，等於只看了「半個系統」 |

## 3. 學到的事

### 3.1 SVDD 與 legalizer 是互補關係

`every_n=1` 給 raw HPWL -16.5%、legality -11%。SVDD 在最佳化 HPWL 時把 macro 擠近彼此（短 wire 但 overlap 多）。
- **若 legalizer 能 fix legality 且 HPWL 損失 < SVDD 帶來的 -16.5%，整體就贏**
- 這正好是 Phase 2 要驗證的核心假設
- 反向觀察：plan_1 §4 的「legality reward 在 reward 內」就不該太重，避免 SVDD 自己去操心 legality

### 3.2 Seed variance 比想像大

adaptec1 baseline 3 seeds: 10.33 / 9.12 / 9.24（std ≈ 0.66 HPWL）。SVDD 的 -0.7% mean 落在 1σ 內 → **單 seed 比較完全不可信**。Phase 2 最少 2 seed 才有意義。

### 3.3 K=8 不是 SVDD 問題

`networks/layers/wrapper.py:31` 的 unbatch/rebatch 對 `batch_size > 4` 有結構性限制（GAT add_self_loops 觸發）。是 codebase-wide 的 batched-inference bug，跟 SVDD 邏輯無關。
- Phase 2 應該用 **K_chunk = 4 序列化**（K=8 跑成 2 個 chunk of 4）繞過，而不是去改 wrapper.py
- 修 wrapper.py 是另一個獨立工程

### 3.4 SVDD-PM 的計算成本可控

預估：每個 SVDD step 多 K 個 `eps_θ` forward。對 1000-step 模型 + `every_n=1` + K=4 → 約 4× model.forward。adaptec1 V=543 case 觀察到 generation_time 從 50s（baseline）到 55s（K=4 every_n=1）→ **時間幾乎沒增加**。預期是因為 GPU 對 batch K*B 是 amortized，瓶頸在 reverse loop overhead 不在 forward 本身。
- Phase 2 可以放心 every_n=1，cost 不是問題

### 3.5 數值穩定要先寫好

Phase 1 報告中 §2.3 的 fix（subtract max / nan_to_num / weight normalize）是事後補的。下次寫新 sampler 時這套要直接放進 template。

## 4. Phase 2 該怎麼設計（feed into plan_2）

**保留的**：
- SVDD-PM mechanism + 預設 K=4
- λ_legality=1 作為起點

**改變的**：
- **加 legalizer=opt-adam**（最重要，否則完全沒辦法跟 ablation_10k 比）
- **從 `ablation_supervised_10k.61/latest.ckpt` 起**（plan_1 §3.2 規劃，Phase 1 跳過了）
- **以 `every_n=1` 為主**（不是 default `every_n=5`）
- **多 seed**（最少 2，理想 3）
- **K=8 用 chunk=4 跑**，不直接傳 K=8 給 model
- **bigblue2 直接 skip**（V×V legality forward 仍會 OOM，等專門的 tiled forward implementation 再說）

**該驗證的核心假設**：
> SVDD 把 raw HPWL 降下來，後續 legalization 能修回 legality 而 HPWL 不會被推回原點 → 最終 7-circuit avg HPWL 贏 ablation_10k (44.01)。

**判定基準照 plan_1 §3.2 原樣套用**：
- < 43.5 = 顯著贏 → 升級 SVDD-MC / 寫 paper
- 43.5–44.0 = 微贏 → 繼續調參
- 44.0–44.5 = 持平 ablation_10k → 看 bigblue2 / SVDD-MC
- > 44.5 = 沒贏 → 放棄 inference-time search

## 5. 不要做的事

- **不要先跳 SVDD-MC**（學 value net）— SVDD-PM 還沒在 fair 條件證明，先把 PM 做好
- **不要先實作 cluster-aware guidance for bigblue2** — 7 circuits 沒搞定前 bigblue2 不是 critical path
- **不要去 fix wrapper.py 的 batched bug** — chunk=4 序列化夠用
- **不要把 lambda_legality 設很高** — Phase 1 證實 `every_n=1` 才是訊號來源，legality 留給 legalizer

## 6. Open questions for plan_2

1. `every_n=1` 對中型 circuit（adaptec4 V=1329, bigblue3 V=1298）是不是仍可負擔（cost / VRAM）？
2. `lambda_legality=1` 對 ablation_10k init 來說是太低還是 OK？（ablation_10k 本身 placement 比較 legal，SVDD 不需要這麼操心）
3. `α_temp=1` 預設是否合理？特別在 `every_n=1` 連續做 SVDD 的情況下，是不是 α_temp 大一點（探索性高）更好？
4. SVDD 跟現有 `opt` guidance 同時 layered 會不會更好？（plan_1 沒探討，Phase 2 還是先單獨用）
