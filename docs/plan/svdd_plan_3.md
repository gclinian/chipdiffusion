# SVDD-PM 第三次實驗計畫（Phase 3: SVDD layered on top of paper opt guidance）

> 前次回顧：`docs/next/svdd_next_2.md`
> Report_2：`docs/report/svdd_report_2.md`
> 動機：Phase 2 結論「SVDD 單獨輸 paper 44.01」缺一個對照 ── 把 SVDD 跟 paper opt guidance **layered** 才是真正 vs paper 的比較

## 1. 目標

**回答唯一一個問題：SVDD-PM 加在 paper opt guidance 之上能不能進一步降 HPWL？**

| 設定 | avg HPWL | 含意 |
|------|---|---|
| Paper 方法（ablation_10k + opt + legalizer） | **44.01**（CLAUDE.md 公告） | 對照 |
| Run F（我們重現） | 44.01 ± 0.5 (預期) | 確認環境匹配 |
| Run E（SVDD + opt layered） | **TBD** | 主要實驗 |

Run E − Run F 就是 SVDD 對 paper 方法的純增量。

## 2. 實作（已完成）

### 2.1 程式碼改動

| 檔案 | 改動 |
|------|------|
| `diffusion/models.py` `ContinuousDiffusionModel.__init__` | 加 `svdd_layer_opt: bool` kwarg |
| `diffusion/models.py` `_reverse_samples_svdd` | 加 `layer_opt` 邏輯：算完 predicted_x0 後，若 `svdd_layer_opt=True`，呼叫 `reverse_guidance_opt_force(predicted_x0, cond, t, mask)` 取 gradient delta 加進 predicted_x0，**再算 mu 抽 K candidates**。初始化時 `reset_guidance_state` for opt's stateful alpha. |
| `diffusion/configs/guidance/svdd_layered.yaml` | 新 config：merge opt.yaml + svdd.yaml + tuned params + `svdd_layer_opt=True` |

### 2.2 layered 流程（每 reverse step）

```
1. eps_predict = U-Net(x_t, t)
2. predicted_x0 = (x_t - σ_t · eps_predict) / α_t                     # Tweedie
3. predicted_x0 += reverse_guidance_opt_force(predicted_x0)           # opt 的 20-step inner SGD + Lagrangian
4. mu = α_{t-1} · predicted_x0 + direction
5. if SVDD step:
      Draw K=4 candidates x_{t-1}^{(k)} = μ + η·z_k
      Score each by -(HPWL + λ·legality)(Tweedie at t-1)              # 另 K 次 forward
      Softmax + resample 1
   else:
      x_{t-1} = μ + η·z
```

Cost：opt 的 20 inner SGD per step + SVDD K=4 forward per step。比純 opt 慢 ~30%。

## 3. 實驗矩陣

| Run | guidance | svdd_layer_opt | num_output_samples | seed | scope | note |
|-----|----------|---|---|---|---|---|
| **F** | `opt` | n/a | 5 (0..4) + 2 (6..7) | 300 | 重現 CLAUDE.md 44.01 | 我們 first-party baseline |
| **E** | `svdd_layered` | True | 5 (0..4) + 2 (6..7) | 300 | 主菜 | SVDD-PM + paper opt 同時跑 |

skip bigblue2（跟 Phase 2 一樣）。CUDA_VISIBLE_DEVICES=1 for bigblue3+bb4。

### 3.1 為什麼跳過 K 或 λ 的 sweep

Phase 1+2 已掃過 K=4/8/lambda=0/1/5/every_n=1/5/start_step_frac=0.3/0.5。**Phase 2 確認 (K=4, every_n=1, λ=1) 是最強 SVDD-only 設定**。Phase 3 只測「加上 opt 之後」這個設定，固定不動。

### 3.2 為什麼不跑多 seed

Phase 1 多 seed 已顯示 seed 變異 ~5%。Phase 3 先看 single-seed 結果是不是 vs Run F 有顯著差距（>1%）；若 marginal 才需要 seed 驗證。

### 3.3 Eval 設定（CLAUDE.md 標準）

```
legalizer@_global_=opt-adam
+skip_guidance_threshold=10000
legalization.alpha_lr=8e-3
legalization.hpwl_weight=12e-5
legalization.legality_potential_target=0
legalization.grad_descent_steps=20000
macros_only=True
seed=300
```

## 4. 預估成本

| Run | 估時 |
|-----|------|
| Run F (opt only): 7 circuits × 含 legalization | 與 Phase 2 Run A 類似 ~30 min + ~30 min = **60 min** |
| Run E (svdd_layered): 7 circuits | 比 Run F 慢 ~30%，**~80 min** |
| **總計** | **~2.5 hr** |

## 5. 判定（fed from next_2 §6）

### 5.1 Run F 重現

| Run F avg HPWL | 行動 |
|----|----|
| 44.01 ± 0.5 | ✅ 環境匹配，繼續看 Run E |
| 顯著偏離（>1） | ❌ 環境或 setup 有差，先 debug 再下結論 |

### 5.2 Run E vs Run F

| Δ = Run E − Run F | 意義 | 下一步 |
|----|----|----|
| **Δ < −0.5** | **SVDD 對 opt 有真增量** | 寫 paper 章節；plan_4 升級 SVDD-MC / 多 seed |
| −0.5 ≤ Δ ≤ +0.5 | 持平 | inference-time search 路線正式放棄；轉教授建議 (1)「從頭 train」 → 開 `from_scratch_plan_1.md` |
| Δ > +0.5 | SVDD 干擾 opt | 同上：放棄 |

## 6. 風險與對策

| 風險 | 對策 |
|------|------|
| Run F 重現不出 44.01 | 對照 CLAUDE.md 文字確認 setup；可能 paper 用不同 seed/ckpt | 
| SVDD + opt 互相干擾（opt 把 x0 推到某處，SVDD 又抽 K 個推到別處）| 觀察 legality 是否異常退步；如是，下次試 SVDD `lambda_legality=0`（純 HPWL reward） |
| bigblue3+bb4 又遇 cudaErrorIllegalAddress | 預設 `CUDA_VISIBLE_DEVICES=1` |
| Run E generation_time 過長 | adaptec1 smoke test 已知 layered ~63s（vs no-leg baseline ~35s），bb4 推估 ~3-5 min。可接受 |

## 7. 行動清單

- [x] 實作 `svdd_layer_opt` 在 `models.py`
- [x] `configs/guidance/svdd_layered.yaml`
- [x] adaptec1 smoke test（無 legalization, 確認跑得通）→ raw HPWL=7.44 vs Phase 1 svdd-only=8.63, baseline=10.33（趨勢一致）
- [ ] Run F：opt baseline 7 circuits (with legalization)
- [ ] Run E：SVDD layered 7 circuits (with legalization)
- [ ] 收 metrics.csv → 寫 `docs/report/svdd_report_3.md`
- [ ] 依結果寫 `docs/next/svdd_next_3.md` 決定 plan_4 還是切換到 from-scratch track
