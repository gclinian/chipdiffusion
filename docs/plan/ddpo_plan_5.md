# DDPO 第五次實驗計畫

> 參考：`docs/report/ddpo_report_4.md`（local reward 加 legality 失敗）

## 目標

用 **last-K 策略**限制 local reward 只算在 reverse sampling 的後段 timestep，避開早期 noisy predicted x₀ 的 reward 干擾。

## 動機（從 v2.4 學到的教訓）

v2.4 把 legality 加進 local reward 後變更差（46.50 vs v2.3 的 45.80）。原因：

1. **Legality potential 在早期 timestep 爆炸**：predicted x₀ 雜亂 → pair-wise overlap 巨大 → local_reward 從 -1.2e6 漲到 -7.8e10（5 個數量級）
2. **不只是 legality 的問題**：HPWL 在早期 predicted x₀ 上也沒意義（之前以為只有 HPWL 有這個問題，現在確認 legality 也一樣）
3. **結論**：**predicted x₀ 在早期 timestep 本質上 ill-defined**，不該對它算 reward

## Last-K 策略

只在 reverse sampling 的**最後 K 個 timestep** 算 local reward。在這些步驟：
- Predicted x₀ 已經接近最終 placement
- HPWL 有意義
- Legality potential 不會爆炸（macros 已大致定位）
- Reward 量級穩定，advantage normalization 能正常運作

### 實作

`diffusion/ddpo.py`:
```python
# 原版：對所有 timestep 算
for t_idx in range(0, T, self.local_reward_every):
    ...

# 新版：只對最後 K 個 timestep 算
start_idx = max(0, T - self.local_reward_last_k)
for t_idx in range(start_idx, T, self.local_reward_every):
    ...
```

新參數 `local_reward_last_k`，預設 0 代表停用（all timestep，舊行為）。

---

## 實驗設定

### Run 5：DDPO v2.5（Last-K Local Reward）

| 項目 | 設定 | 備註 |
|------|------|------|
| method | ddpo_v2_5_last_k | |
| num_timesteps | 50 | |
| **local_reward_last_k** | **20** | **只算 last 40%（indices 30-49）** |
| local_reward_every | 5 | 每 5 步算 1 次 → 共 4 次（30, 35, 40, 45） |
| local_reward composition | HPWL + 0.5×legality | 同 global reward |
| local_reward_weight | 0.5 | 同 v2.3/v2.4 |
| 其餘（batch, lr, clip, supervised_weight 等） | 同 DDPO v2 | |
| 訓練時間估計 | ~60-70 分鐘 | 比 v2.4 少，因為 local reward 計算次數減少 |

### K 值選擇理由

- **K=20**：後 40% timestep
- **Reward 計算次數**：4（原本是 10）
- **後期性**：t_idx=30 時，reverse sampling 進度 60%，predicted x₀ 應該已經接近最終結構
- **如果 K 太小**（K=5-10）：只有 1-2 個 reward point，signal 太稀疏
- **如果 K 太大**（K=40）：仍會碰到早期 noisy 區域，重蹈 v2.3/v2.4 覆轍

---

## 預期與比較對象

### HPWL 平均（7 circuit，不含 bigblue2）

| 方法 | Avg HPWL | Local Reward 內容 |
|------|----------|-------------------|
| Ablation 10k | 44.01 | 無（supervised only） |
| DDPO v2 | 44.65 | 無 local reward |
| DDPO v2.3 | 45.80 | HPWL only, all timestep |
| DDPO v2.4 | 46.50 | HPWL+legality, all timestep |
| **DDPO v2.5** | **?** | **HPWL+legality, last K=20** |

### 成功判定

| 結果 | 意義 |
|------|------|
| v2.5 < 44.65（贏過 DDPO v2） | Local reward 的想法本身可行，只要避開早期 timestep |
| 44.65 < v2.5 < 45.80（介於 v2 和 v2.3 之間） | Last-K 有幫助但沒達到 DDPO v2 水準 |
| v2.5 ≥ 45.80 | Local reward 方向完全失敗，應放棄 |

### 訓練穩定性

| 指標 | 目標 | 備註 |
|------|------|------|
| local_reward 量級 | 穩定在 \|~1e3-1e5\| | 不該有 v2.4 那種 5 個數量級爆炸 |
| val loss (step 5000) | < 0.5 | v2.4 是 1.77（異常高） |
| 無 val loss 尖峰 | | v2.3 曾出現 13.7 尖峰 |

---

## 執行計畫

1. 改 code（`diffusion/ddpo.py` + `diffusion/configs/mode/ddpo.yaml`）加 `local_reward_last_k` 參數
2. Train 5000 steps（~65 min）
3. ISPD2005 eval 8 circuits（~2.5 hr）
4. 分析 + 寫 report

總時間：~3.5-4 小時。

---

## 風險

1. **K=20 還是太早**：可能需要降到 K=10 甚至 K=5
2. **沒有本質改善**：如果即使 last K 也沒比 DDPO v2 好，證明 local reward 這個 design 根本不適合 diffusion model，下次就應該徹底放棄
3. **Reward 計算位置太少**（4 個）：variance 可能變高，signal 太弱

如果 Run 5 結果介於 44.65-45.80，可以再試 K=10（last 20%）微調。如果 Run 5 ≥ 45.80，直接放棄 local reward，回去做 supervised 相關實驗。
