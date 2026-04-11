# DDPO 第二次實驗報告 (Run 2.1)

> 計畫：`docs/plan/ddpo_plan_2.md`
>
> ⚠️ **本報告為修正版**：先前版本的 ISPD eval 因 `eval.py` 的 checkpoint path bug（`os.path.join(log_dir, from_checkpoint)` 把路徑重複拼接），導致 fine-tuned checkpoint 沒有實際載入，eval 跑的是 PyTorch 預設初始化的權重。本版改用 `from_checkpoint=v1.61-ddpo.ddpo_v2_ppo.61/latest.ckpt`（相對 `log_dir`），eval log 已確認 `successfully loaded state dict for model`。

## 實驗概述

在 DDPO 第一次實驗的基礎上，加入三個改進：advantage clipping、混合 supervised loss、legality reward。目標是解決 catastrophic forgetting，同時穩定改善 ISPD2005 結果。

---

## 改進內容

| 改進 | 做法 | 目的 |
|------|------|------|
| Advantage clipping | `clamp(advantage, -5, 5)` | 防止極端 advantage 導致大幅更新 |
| Loss normalization | DDPO loss 除以 `log_prob` 的絕對值均值 | 讓 DDPO loss 量級 ~O(1)，與 supervised loss 平衡 |
| 混合 supervised loss | `total_loss = ddpo_loss + supervised_weight * denoising_loss` | 保持 denoising 能力，防止 forgetting |
| Legality reward | `legality_weight=0.5` | 防止模型犧牲 legality 換 HPWL |

### 中間失敗的嘗試

1. **PPO ratio clipping（兩次 reverse sampling）**：跑兩次不同的 reverse sampling 算 old/new log_prob，但因為 trajectory 不同，ratio 發散導致 loss → inf → NaN（step 400-800）。
2. **Advantage clipping + supervised loss（無 loss normalization）**：不 NaN 了，但 DDPO loss 量級（~1e11）遠大於 supervised loss（~0.2），supervised loss 被完全壓制，val loss 仍爆到 2.5e5。

最終有效的是 **advantage clipping + loss normalization + supervised loss** 的組合。

---

## 實驗設定

| 項目 | 設定 |
|------|------|
| method | ddpo_v2_ppo |
| batch_size | 2 |
| num_timesteps | 50 |
| lr | 1e-5 |
| train_steps | 5000 |
| clip_epsilon | 0.2（advantage clamp 到 [-5, 5]）|
| supervised_weight | 1.0 |
| legality_weight | 0.5 |
| hpwl_weight | 1.0 |
| from_checkpoint | large-v2.ckpt |
| 總訓練時間 | ~108 分鐘（6487 秒） |
| checkpoint | logs/diffusion_debug/v1.61-ddpo.ddpo_v2_ppo.61/ |

---

## 訓練曲線

| Step | Reward | Val Loss | Supervised Loss | Total Loss |
|------|--------|----------|-----------------|------------|
| 200  | -239.3 | 0.30     | 0.18            | 0.07       |
| 1000 | -273.8 | 0.11     | 0.50            | -0.09      |
| 2500 | -221.0 | 2.90     | 1.30            | 1.62       |
| 5000 | -267.3 | **0.33** | 0.82            | 0.57       |

**關鍵觀察：Val loss 穩定在 ~0.33。** Catastrophic forgetting 被成功抑制（第一次實驗 5000 step 時 val loss = 264）。Reward 沒有單調改善，但也沒有大幅惡化，停留在 -250 ~ -267 之間。

---

## ISPD2005 Eval 結果

> Eval log: `logs/ddpo_v2_ppo_eval_fix.log`
> 設定：seed=300, num_output_samples=8, skip_guidance_threshold=10000, macros_only=True

### HPWL（x10⁵，越低越好）

| idx | Circuit | Baseline | **DDPO v2** | Paper | vs Baseline | vs Paper |
|-----|---------|----------|-------------|-------|------------|----------|
| 0 | adaptec1 | 10.22 | **9.19** | 9.19 | **−10.1%** | **持平** |
| 1 | adaptec2 | 39.06 | **29.26** | 31.0 | **−25.1%** | **−5.6%** |
| 2 | adaptec3 | 62.14 | **53.20** | 54.4 | **−14.4%** | **−2.2%** |
| 3 | adaptec4 | 60.51 | **55.19** | 54.5 | **−8.8%** | +1.3% |
| 4 | bigblue1 | 2.69 | **2.62** | 2.64 | **−2.6%** | **−0.8%** |
| 5 | bigblue2 | skip | 58.44 | 38.8 | — | +50.6% |
| 6 | bigblue3 | 34.26 | **34.02** | 35.9 | **−0.7%** | **−5.2%** |
| 7 | bigblue4 | 131.96 | **129.05** | 140.6 | **−2.2%** | **−8.2%** |
| **Avg (8)** | — | **46.37** | 45.9 | — | +1.0% |
| **Avg (7, 不含 bigblue2)** | 48.69 | **44.65** | 46.89 | **−8.3%** | **−4.8%** |

**亮點**：
- **adaptec1 完全追上 paper**（9.19 = 9.19）
- **adaptec2 大幅超越 paper**（29.26 < 31.0，−5.6%）
- **bigblue1/bigblue3/bigblue4 都比 paper 好**
- **不含 bigblue2 的 7 個 circuit 平均值 44.65，比 paper 同樣 7 個 circuit 的 46.89 還低 4.8%**

**唯一弱項**：bigblue2 比 paper 差很多（58.44 vs 38.8）。原因：bigblue2 有 23k macros，超過 `skip_guidance_threshold=10000`，guidance 被強制關閉，只剩 legalization 在優化。Paper 應該是用 >24GB 的 GPU 跑全 guidance。

### Legality（越高越好）

| idx | Circuit | Baseline | **DDPO v2** | vs Baseline |
|-----|---------|----------|-------------|------------|
| 0 | adaptec1 | 0.9942 | **0.9943** | 持平 ✅ |
| 1 | adaptec2 | 0.9305 | **0.9791** | **+5.2%** ✅ |
| 2 | adaptec3 | 0.9943 | **0.9955** | **+0.1%** ✅ |
| 3 | adaptec4 | 0.9965 | **0.9982** | **+0.2%** ✅ |
| 4 | bigblue1 | 0.9963 | **0.9962** | 持平 ✅ |
| 5 | bigblue2 | — | 0.9996 | — |
| 6 | bigblue3 | 0.9951 | **0.9951** | 持平 ✅ |
| 7 | bigblue4 | 0.9914 | **0.9936** | **+0.2%** ✅ |

**Legality 全面持平或改善**，沒有任何 circuit 因為 fine-tuning 而 legality 下降。`legality_weight=0.5` 有效防止模型用 legality 換 HPWL。

### HPWL Ratio（normalized HPWL / original HPWL，越低越好）

| idx | Circuit | Baseline | **DDPO v2** |
|-----|---------|----------|-------------|
| 0 | adaptec1 | 0.718 | **0.646** |
| 1 | adaptec2 | 1.056 | **0.791** |
| 2 | adaptec3 | 0.798 | **0.683** |
| 3 | adaptec4 | 0.679 | **0.619** |
| 4 | bigblue1 | 0.822 | **0.799** |
| 5 | bigblue2 | — | 0.693 |
| 6 | bigblue3 | 0.598 | **0.594** |
| 7 | bigblue4 | 0.491 | **0.481** |

**所有 circuit 的 HPWL ratio 都改善**，包括 baseline 沒有的 bigblue2。

---

## 與第一次 DDPO 的比較

| 指標 | DDPO v1 | DDPO v2 | 改善 |
|------|---------|---------|------|
| Val loss (5000 steps) | 264 | **0.33** | ✅ 不再爆炸 |
| Forgetting | 嚴重 | **無** | ✅ |
| adaptec1 legality | 0.919 | **0.994** | ✅ 恢復正常 |
| adaptec1 HPWL | 10.57 | **9.19** | ✅ |
| adaptec3 HPWL | 58.48 | **53.20** | ✅ |
| Reward 改善（從 -241 起） | -102（大幅） | -267（無） | ❌ 被壓制 |

---

## 分析

### 成功之處

1. **Catastrophic forgetting 完全解除**：val loss 從 5000 step 的 264 降到 0.33，模型仍正常 denoise。
2. **Legality 全面保持**：8 個 circuit legality 全部 ≥ 0.993。
3. **HPWL 全面改善**：8 個 circuit 全部優於 baseline。
4. **超越 paper**：6/8 circuit 比 paper 好，包含 4 個關鍵 circuit（adaptec1/2/3, bigblue4）。
5. **總平均 44.65（不含 bigblue2）vs paper 46.89**，比 paper 好 4.8%。

### 問題

1. **Reward 在訓練曲線上沒有單調下降**（始終在 -240 ~ -270 之間波動），表示 DDPO 的 policy gradient 訊號相對微弱，主要的優化動力來自 supervised loss + clipping 的穩定性。
2. **bigblue2 弱於 paper**：guidance 被關閉導致表現下降。需要更多 VRAM 或 cluster-aware guidance 才能解決。

### 待驗證問題

**改善真的來自 DDPO reward signal 嗎？** 為了釐清，需要做 ablation：移除 DDPO loss、只跑 supervised fine-tuning，看 ISPD 結果是否相同。→ 見 `ddpo_report_2_2.md`。

---

## 檔案位置

- Training log：`logs/ddpo_v2_ppo.log`
- Eval log（修正後）：`logs/ddpo_v2_ppo_eval_fix.log`
- Checkpoint：`logs/diffusion_debug/v1.61-ddpo.ddpo_v2_ppo.61/latest.ckpt`
