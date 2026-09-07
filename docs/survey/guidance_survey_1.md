# Diffusion Guidance Methods — Survey for Chip Placement (Survey #1)

> 動機：見 `docs/meet/meet_0508.md`，教授建議「在 denoising 過程嘗試其他種 guidance 方式，例如 RL guided」。
> 目的：找出可能替代 / 補強現有 `guidance@_global_=opt` 的方法。
> 範圍：聚焦 chip macro placement 的 continuous diffusion model（座標 (x, y)）。

---

## 1. 現況：codebase 中的 guidance 實際是什麼

### 1.1 程式碼層的事實（`diffusion/guidance.py`, `configs/guidance/opt.yaml`, `models.py:1430-1509`）

**HPWL guidance**
- 實作於 `HPWL(tgn.MessagePassing)`（`guidance.py:191-224`），用 PyG message passing 沿 net edge 傳訊。
- 複雜度 **O(E)**（edges），不是 O(V²)。即使對 23k macros 的 bigblue2，HPWL 的 forward + gradient 都 OK。

**Legality guidance**（**真正的 V×V 瓶頸**）
- `legality_guidance_potential`（`guidance.py:7-47`）建出 `(B, V, V, 2)` pairwise overlap tensor。
- 已有 `legality_guidance_potential_tiled`（line 49-84），block_size=16384，每 tile 直接 `.backward()`，memory 換 compute。
- bigblue2 (V=23k) 上 OOM 的就是這個 V×V Jacobian — 不是 HPWL。

**現有 `opt` guidance 的本質**（`configs/guidance/opt.yaml`）
```yaml
guidance_mode: opt
grad_descent_steps: 10        # 每個 reverse step 跑 10 步 inner SGD
forward_guidance_weight: 0.0  # forward + backward guidance 兩個 hook
hpwl_guidance_weight: 1e-4
alpha_init: 0.0
alpha_lr: 5e-4                # Lagrangian dual variable
legality_potential_target: 1e-4   # 把 legality 當 hard constraint
use_adam: True
```
- 這已經是 **Universal Guidance (Bansal 2023) + adaptive Lagrangian constraint** 的進階版。
- Forward + backward 雙 hook、inner optimizer with Adam、Lagrangian alpha 排程 (`alpha_critical_factor=0.5`)、softmax temperature 排程。
- 結論：**B 系列方法（Universal Guidance, FreeDoM, DPS）對現況增益有限**，因為核心 mechanism 都已經內建。

**Reverse loop hook point**（`models.py:1455-1500`）
- 每個 reverse step 流程：`eps_predict` → `predicted_x0` → `guidance_fn(predicted_x0)` → mu = α·x0_guided + direction → x = mu + η·z
- SVDD / SMC / CoDe 的插入點都在這個 inner loop 的 sampling step（line 1485-1487）。

---

## 2. 候選方法分組

### Tier 1 — 最值得試（gradient-free + 對應教授「RL guided」）

#### 1.1 **SVDD — Soft Value-Based Decoding in Diffusion**
- Paper: Uehara, Li, Zhao et al., arXiv:2408.08252 (2024)
- 機制：每個 reverse step 從 reverse kernel sample **K 個** `x_{t-1}` 候選，計算 soft value
  `V_t(x_{t-1}) = log E[exp(r(x_0)) | x_{t-1}]`，依 `exp(V_t / α)` 重採樣。
- 兩變種：
  - **SVDD-PM**（posterior mean）：value = `r(\hat x_0(x_{t-1}))`，不需要訓額外網路。
  - **SVDD-MC**（Monte Carlo regression）：學一個小 value net 從 rollout 訓出來。
- **零 gradient backprop**：只需 forward HPWL/legality 評估。
- 適配 chip placement：
  - 對應教授「RL guided」 — value function look-ahead 就是 RL inference。
  - **直接避開 V×V Jacobian** 痛點：bigblue2 上只需要 K 倍 forward legality（每次仍 O(V²) compute，但無 backward buffer）。
  - 已在 molecules / DNA / RNA 驗證；reference repo `masa-ue/SVDD`。
- 實作量：中等（要新增 K-particle sampler、value 評估、resample）。

#### 1.2 **CoDe — Blockwise Best-of-N**
- Paper: arXiv:2502.00968 (2025)
- 機制：reverse 跑 B 步（一個 block），到 block boundary 時 fork N 個分支繼續，挑 reward 最高的 `\hat x_0`。
- 是 SVDD 的簡化版（block 級別、無 soft weighting）。
- 適配：**最便宜的 inference-time search baseline**，30 行就能加在 `reverse_samples` 外層。
- 用途：跑 SVDD 之前先驗證「inference-time search 對這個任務有效嗎」的 sanity check。

#### 1.3 **TDS / SMCDiff — Sequential Monte Carlo guidance**
- TDS: Wu et al., arXiv:2306.17775 (NeurIPS 2023)
- SMCDiff: Trippe et al., arXiv:2206.04119 (ICLR 2023, protein motif-scaffolding)
- 機制：N 個 particle 跑 reverse process，每 step 用 importance weight + resample。
- TDS 加 twisting function（gradient-free 即可），可選用 gradient proposal — bigblue2 時把 gradient leg 關掉。
- 比 SVDD 更原則（asymptotic exact），但實作複雜度高一級（要 ESS-based resample threshold、weight tracking）。
- 結構領域 (protein) 已有成熟應用。

### Tier 2 — 值得試但偏 incremental

#### 2.1 **DRaFT-K — Last-K reward backprop**
- Paper: Clark et al., arXiv:2309.17400 (2023)
- 機制：把 reward (HPWL + legality) 透過最後 K 個 reverse step backprop，更新 model weights。
- 是 AddLoss / ReFL 的近親，差別在於「最後 K 步」而非「random t」。
- 為什麼可能有效：解決 AddLoss 在 v1 / DDPO local reward 看到的「早期 t 的 predicted_x0 太雜訊」問題。
- 改動小：在 `train_graph.py` AddLoss branch 加 last-K mask 即可。
- 限制：本質還是 gradient-based supervised+aux 路線，bigblue2 V×V backward 問題不變。

#### 2.2 **FreeDoM — Time-travel guidance**
- Paper: Yu et al., arXiv:2303.09833 (ICCV 2023)
- 機制：當 guidance signal 在低 noise step 變弱時，跳回 noisier state 再 denoise 一次（noise-then-redenoise）吸收 guidance 擾動。
- 適配：可直接掛在現有 `opt` guidance 上，~5 行。
- 對 bigblue2 沒幫助（gradient path 沒變），但其他 7 circuit 可能再擠一點。

### Tier 3 — 想法好但代價高

#### 3.1 **Reflected Diffusion**
- Paper: Lou & Ermon, arXiv:2304.04740 (ICML 2023)
- 機制：forward 和 reverse SDE 在 data domain boundary 做 reflection，取代 thresholding hack。
- 適配：die boundary 是天然 box constraint，**Reflected Diffusion 原生處理**。
- 代價：要從頭重訓整個 diffusion model。
- 機會：剛好對應教授第一個建議「從頭 train」 — 可以合併成「從頭 train + reflected SDE」一條路。

#### 3.2 **Mirror Diffusion**
- Paper: Liu et al., arXiv:2310.01236 (NeurIPS 2023)
- 機制：用 mirror map 把 convex constraint set 映到 unconstrained 空間訓練，sample 後 closed-form 映回去。
- 適配：die rectangle 是 convex，可處理 boundary；但無法處理 non-overlap（非 convex）。
- 代價：同樣要重訓。

### Tier 4 — 已試 / 不適合

| 方法 | 為什麼跳過 |
|------|-----------|
| DDPO local reward | 已試 3 個變種全失敗（`docs/report/ddpo_report_{4,5}.md`） |
| AddLoss / ReFL | 已試 2 個變種，DRaFT-K 是其多步版本 |
| DOODL (full-trajectory grad) | 比現有 V×V 更糟 |
| DPS / ΠGDM | 機制和現有 `opt` 大致相同 |
| MPGD (autoencoder projection) | 需要先訓 placement autoencoder |
| Classifier-Free Guidance | 要重訓加 conditioning dropout，且 placement 沒有自然的 condition label |
| RB-Modulation (SOC) | 仍需 reward gradient，不解 V×V |

---

## 3. 對比表

| # | 方法 | Family | 需 ∇reward? | 需訓額外 model? | V² bottleneck? | 結構領域驗證? | 難度 |
|---|------|--------|--------------|-----------------|-----------------|---------------|------|
| 1.1 | **SVDD-PM** | RL / value | **No** | No | **No** | Yes (mol/DNA) | 中 |
| 1.1 | SVDD-MC | RL / value | No | 小 V net | No | Yes | 中-高 |
| 1.2 | **CoDe** | Block best-of-N | **No** | No | **No** | T2I | 低 |
| 1.3 | TDS | SMC + twist | 可選 | No | **No** | Yes (protein) | 中-高 |
| 1.3 | SMCDiff | SMC | No | No | No | Yes (protein) | 中-高 |
| 2.1 | DRaFT-K | Grad fine-tune | Yes | 更新 θ | Yes | Images | 低 |
| 2.2 | FreeDoM | Plug-and-play | Yes | Optional | Yes | Some | 低 |
| 3.1 | Reflected Diffusion | Constrained | No | 重訓 | No | Bounded data | 高 |
| 3.2 | Mirror Diffusion | Constrained | No | 重訓 | No | Convex sets | 高 |
| — | 現有 `opt` | Universal+Lagr | Yes | No | **Yes (legality)** | (this project) | (已有) |

---

## 4. 推薦執行順序

1. **CoDe** — sanity check (~30 行 code, 半天)
   判定：若 best-of-N at block level **完全沒效**，則 inference-time search 對這個任務無用，跳過 SVDD/TDS。
2. **SVDD-PM** — 主菜（~1-2 天 code, 1-2 天訓+eval）
   判定：對應教授「RL guided」最直接，**直接解 bigblue2 V×V 問題**。如 SVDD-PM 有效，再升級 SVDD-MC。
3. **TDS** — 如果 SVDD 有效再升級成 SMC 版本獲得 asymptotic exactness。
4. **DRaFT-K** — 平行可做，但是 incremental，優先級較低。
5. **Reflected Diffusion** — 跟「從頭 train」（教授建議 1）合併規劃，獨立的長期 track。

---

## 5. 主要參考

| Paper | arXiv |
|-------|-------|
| SVDD (Soft Value-based Decoding) | [2408.08252](https://arxiv.org/abs/2408.08252) |
| CoDe (Blockwise Control) | [2502.00968](https://arxiv.org/abs/2502.00968) |
| TDS (Twisted Diffusion Sampler) | [2306.17775](https://arxiv.org/abs/2306.17775) |
| SMCDiff (Protein motif-scaffolding) | [2206.04119](https://arxiv.org/abs/2206.04119) |
| DRaFT | [2309.17400](https://arxiv.org/abs/2309.17400) |
| FreeDoM | [2303.09833](https://arxiv.org/abs/2303.09833) |
| Reflected Diffusion | [2304.04740](https://arxiv.org/abs/2304.04740) |
| Mirror Diffusion | [2310.01236](https://arxiv.org/abs/2310.01236) |
| Universal Guidance (Bansal) | [2302.07121](https://arxiv.org/abs/2302.07121) |
| DPS | [2209.14687](https://arxiv.org/abs/2209.14687) |
| Inference-Time Alignment Tutorial | [2501.09685](https://arxiv.org/abs/2501.09685) |
| RL Fine-Tuning of Diffusion Tutorial | [2407.13734](https://arxiv.org/abs/2407.13734) |
| Chip Placement w/ Diffusion (baseline) | [2407.12282](https://arxiv.org/abs/2407.12282) |

> 下一步：見 `docs/plan/svdd_plan_1.md`（SVDD-PM 第一次實驗計畫）。
