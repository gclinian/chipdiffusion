> # ⚠️ 更正聲明 CORRECTION（2026-09-07）— §3.1 已作廢
>
> **本文件 §3.1 已由 `docs/next/svdd_next_3.md` §4 明文宣告「作廢」。** 依專案慣例本文**不改寫**，原文全部保留供歷史追溯，僅加註。
>
> §3.1 斷言「**Paper 用 ablation_10k + opt guidance + opt-adam legalization → 44.01**」── **這句是錯的**。`ablation_supervised_10k` 是**使用者自己 fine-tune 出來**的 checkpoint，不是 paper 的。Paper 真正的 baseline 是 **`large-v2` + opt = 46.89（paper 公告, seed 400）／ 48.691（我們環境重現, seed 300）**；44.01 反而是我們**贏** paper 6.1% 的自家最佳結果。
>
> **後果**：這個誤植直接 propagate 進 `docs/plan/svdd_plan_3.md` §1 與 `docs/report/svdd_report_3.md`，害 Phase 3 整場實驗都在跟錯的對照組比，得出偽結論「inference-time search（SVDD/CoDe/TDS 整族）正式 dead end」。改在 paper 真正的 large-v2 上重跑後（CSV 重算確認）：**SVDD_layered 3-seed avg = 44.845 ± 0.654（45.097 / 44.103 / 45.335），3/3 seeds 全贏 paper 46.89**；TDS_layered 45.081、CoDe_layered 45.216 亦同樣 3/3 全贏。見 `docs/report/svdd_report_5.md` §8。
>
> **另**：§4「保留的：ablation_supervised_10k 為 init checkpoint」這個決定也被 `svdd_next_3.md` §4 點名要重排 —— 正是這個選擇讓 SVDD 沒有 search headroom；當時若直接在 large-v2 上做 plan_3 可以省掉一整個 phase。§6 決策樹中「持平 → inference-time search 路線正式放棄」的分支同樣不再有效。
>
> **本文件仍然有效的部分**：§2、§3.2-3.5（GPU contention、macros_only 加速、K=4 記憶體、bigblue3 headroom 觀察）、§5、§7 未受影響。
>
> 詳見 `docs/next/svdd_next_3.md` §3、§4、§5.1。

---

# SVDD Phase 2 回顧（next_2）

> Report: `docs/report/svdd_report_2.md`
> Plan: `docs/plan/svdd_plan_2.md`
> 寫作日期：2026-05-21

## 1. 一句話總結

SVDD-PM 在 **deployment-fair 條件**（ablation_10k + opt-adam legalization）下，**贏 no-guidance baseline -15.21%**（71.94 vs 84.85），但**遠輸 paper 的 opt guidance**（71.94 vs 44.01，+63%）。Plan_2 §5.1 文字判定觸發「放棄」，但 plan_2 **沒測 SVDD + opt layered**——這個缺失對照不補就不能下結論。

## 2. Plan_2 vs 實際差異

| plan_2 預期 | 實際 |
|------------|------|
| Run A baseline 用 guidance=none 是「ablation_10k 自身能力」的公平比較 | 實際造成 baseline 比 paper 44.01 弱 93%。我們在比的是「SVDD vs 拔掉 opt 的弱化版」，不是「SVDD vs paper」 |
| 7-circuit eval ~3-5 小時 | 實際 ~30 min (part1) + ~30 min (part2)，**因為 macros_only 大幅減少節點數**讓 reverse 變快；legalization 才是真瓶頸 |
| bigblue2 用 skip_guidance_threshold 就好 | 沒踩到（我們直接 num_output_samples 切片跳過） |
| 主要風險：每 step SVDD 在大 circuit OOM | 沒踩到，但**踩到 GPU 0 contention**（其他使用者佔 10 GB）→ bigblue3/bb4 連續 `cudaErrorIllegalAddress` |
| Run B (default config) vs Run C (aggressive) 對照差異 | 跳過 Run B；先做 Run C，結果就確定 default 也不會贏 → 省 1 hr |

## 3. 學到的事

### 3.1 「跟 paper 比」必須跟 paper 的 **完整** setup 比

> 🚫 **[作廢 VOID — 2026-09-07]** **本小節整段作廢**，依據 `docs/next/svdd_next_3.md` §4「next_2 §3.1 該作廢」。
> **錯誤點**：**paper 不是用 ablation_10k**，44.01 也不是 paper 的數字。Paper baseline = `large-v2` + opt + opt-adam = **46.89（公告）／ 48.691（我們重現）**；44.01 是我們自己 fine-tune 贏 paper 6.1% 的結果。
> **應該說的是**（next_3 §4 給的正確版本）：「我們在 ablation_10k 上比 SVDD 跟 opt，**沒有對應到 paper baseline**。要 vs paper 必須在 large-v2 上跑。」
> 下方原文保留僅供歷史紀錄，**不可再作為任何 plan 的依據**。

Paper 用 ablation_10k + **opt guidance** + opt-adam legalization → 44.01。我們把 opt 拔掉之後跟 SVDD 比，等於是 strawman。要證明 SVDD 對 paper 有用 → **必須測 SVDD + opt layered**。

這是 plan_2 文字描述的 framing 缺陷，不是 SVDD 本身的問題。

### 3.2 Bigblue3 是 SVDD 受益最大的 circuit

Run A bigblue3 = 98.82（paper 35.9 的 2.8x），SVDD 拉回 44.79（接近 paper）。
- 原因：ablation_10k 對 bigblue3 raw output 特別差 → SVDD 的 K-particle search 有大 headroom
- **泛化 insight**：SVDD 對「raw output 差」的 circuit 邊際效用最大；對 raw 接近 floor 的 adaptec4 邊際效用 = 0

### 3.3 GPU contention 是真風險

GPU 0 連續 3 次 `cudaErrorIllegalAddress`（在 compute_pin_map 的 torch.unique）→ 換 GPU 1 後立刻全部成功。
- **教訓**：跑長 ISPD eval 前 `nvidia-smi` 看 GPU 哪個空，或寫 retry-on-CUDA-error 邏輯
- 這個錯誤模式跟 OOM 不太一樣——是 **kernel sync 失敗**，可能跟其他 process 寫到同一 GPU 有關
- 寫進 [[project_svdd_direction]] 給下次

### 3.4 macros_only=True 大幅縮短 reverse time

Phase 2 5-circuit part1 跑完只要 ~17 min（vs 我預期 60-90 min）。
- macros_only 把 cell-level nodes 拿掉，剩 macros，計算量大降
- **legalization** 才是真正吃時間（bb4 ~25 min, bb3 ~5 min）

### 3.5 K=4 在 bb4 (V=8170) 沒爆 GPU

擔心 K*B=4 forward 在 8170 nodes 會 OOM，實際 generation_time 226s（vs Run A 65s），慢 3.5×但能跑。**K=8 應該也撐得住** ──但 PyG batched edge_index bug 還在。

## 4. Phase 3 該怎麼設計（feed into plan_3）

**保留的**：
- SVDD-PM K=4, every_n=1, λ=1, α_temp=1, start_step_frac=0.5（Phase 2 確認的最強 SVDD 設定）
- ablation_supervised_10k 為 init checkpoint
- opt-adam legalization with grad_descent_steps=20000

**新增的（plan_2 漏的）**：
- **Run E：SVDD + opt layered** ── 把 paper 的 `opt` guidance（grad_descent_steps=20, hpwl_guidance_weight=16e-4）跟 SVDD 同步在 reverse loop 中跑
  - 實作：在 `_reverse_samples_svdd` 內部，**算完 predicted_x0 後**呼叫 `reverse_guidance_opt_force` 取得 gradient delta，加進 predicted_x0，**再算 mu / 抽 K candidates**
  - 新加 model param `svdd_layer_opt: True`
  - 新加 config `configs/guidance/svdd_layered.yaml`
- **Run F：reproduce 44.01**（ablation_10k + 純 opt + opt-adam）── 給我們自己 first-party baseline，不只信 CLAUDE.md 的字
  - 跟 Run E **同 setup** 的最近 sibling（差別只在 SVDD on/off）→ Δ 就是 SVDD 的純增量

**改變的**：
- 不再用 Run A (guidance=none) 當主 baseline；只當 sanity check 用
- GPU 預設 `CUDA_VISIBLE_DEVICES=1` 跑 bigblue3+bb4，避開 GPU 0 contention

## 5. 不要做的事

- ❌ 不要先升級 SVDD-MC（沒在 fair 條件證明 SVDD-PM 有用前，學 value net 是過早優化）
- ❌ 不要 sweep K 或 λ —— Phase 1+2 已掃過，固定 K=4 / λ=1
- ❌ 不要修 PyG K=8 batched bug —— K=4 夠用
- ❌ 不要為 bigblue2 寫 forward-only tiled legality —— 7-circuit avg 不含 bb2，先處理可 fit 24GB 的

## 6. 判定 plan_3 的決策樹

| Run F avg HPWL (reproduce paper) | 意義 |
|----|----|
| ≈ 44.01 | 我們的環境 reproduce 出 paper 數字，可信 |
| 顯著偏離（> 0.5） | 環境有差，要 debug 不能繼續 |

| Run E avg HPWL (SVDD + opt) vs Run F | 意義 | 下一步 |
|----|----|----|
| **Run E < Run F − 0.5** | **SVDD 對 paper opt guidance 有真增量** | 寫 paper 章節「SVDD 補強 opt」；plan_4 升級 SVDD-MC |
| Run F − 0.5 ≤ Run E ≤ Run F + 0.5 | 持平，SVDD 對 opt 無增量 | inference-time search 路線正式放棄；轉教授建議 (1)「從頭 train」 → 開 `from_scratch_plan_1.md` |
| Run E > Run F + 0.5 | SVDD 干擾 opt | 同上：放棄 SVDD 路線 |

## 7. Open questions for plan_3

1. SVDD + opt 的執行順序應該 `predicted_x0 + opt_grad + svdd_sample` 還是 `predicted_x0 + svdd_sample + opt_grad`？選前者（先 opt 把 predicted_x0 拉進可行空間，再 SVDD 在其周圍搜索）。
2. 兩者 weight 怎麼平衡？start 用 paper 公告值 + SVDD 預設值，**不要先 tune**。
3. 是否每 reverse step 都要做 opt？是的（沿用 paper 設定 `guidance_step=1000` ≈ 每步）。SVDD `every_n` 仍是 1。
4. legality_potential_target 怎麼設？paper opt 用 1e-4，SVDD λ_legality=1。可能需要 SVDD 把 λ 拉回 0（reward 只看 HPWL），讓 legality 完全交給 opt 的 Lagrangian alpha 機制。**第一次先不變，第二次如果 trade-off 不對再說。**
