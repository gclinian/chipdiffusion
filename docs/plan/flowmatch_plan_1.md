# Flow Matching / Drifting Model 第一次實驗計畫

> 動機：`docs/meet/meet_0620.md` 教授建議 (2)
> 執行者：Opus 4.8 agent（2026-06-20 派出）
> 對照 baseline：Paper DDPM 46.89；我們 from-scratch DDPM 500k = 45.05；Ablation 10k = 44.01

## 1. 目標

回答：「**把 DDPM 換成 flow matching，同資料同架構同訓練量，ISPD 表現會不會更好？inference 能快多少？**」

## 2. 為什麼 flow matching 可能有優勢

| 面向 | DDPM (現況) | Flow Matching |
|---|---|---|
| 訓練 objective | ε-prediction MSE (stochastic) | velocity field regression (OT 直線路徑, deterministic target) |
| Inference | 1000 步 stochastic reverse | ODE solver, 通常 **10-100 步**就收斂 |
| 訓練難度 | noise schedule 敏感 | 直線 interpolation, 通常更穩 |
| Guidance 掛法 | score + ∇φ | velocity + ∇φ（同樣可掛 opt guidance / SVDD 類 particle search）|

Drifting model（few-step 生成範式）列為 stretch goal — 先把 flow matching 做通。

## 3. 實驗設計

### Phase A — 實作 + smoke test
1. 新增 `FlowMatchingModel`（重用 att_gnn backbone，改 loss 為 conditional flow matching：`||v_θ(x_t, t) − (x_1 − x_0)||²`，x_t = (1−t)·x_0 + t·x_1 線性插值）
2. 新增 Euler/Heun ODE sampler（`reverse_samples` 對應版本）
3. v1.61-fs 上 1k step pilot 確認 loss 下降

### Phase B — 500k 對照訓練（match DDPM Run X）
- dataset=v1.61-fs.61, batch=32, lr=3e-4, 500k steps, size=large
- **唯一變因 = training objective**（vs `fs_p1_X_500k`）

### Phase C — ISPD eval
- 7 circuits (skip bb2), seed=300, opt guidance + opt-adam legalizer（guidance 需 port 到 ODE sampler：在每個 ODE step 對 x̂_1 估計做 gradient descent，機制同 DDPM 的 x̂_0 guidance）
- 額外量測：**inference 步數 sweep（10/50/100/1000 步）** 的 HPWL-vs-speed curve

## 4. 判定基準

| FM 7-circuit avg | 結論 | 行動 |
|---|---|---|
| < 44.01 | FM 贏所有現有方法 | multi-seed + 寫 paper 主章節 |
| 44.01–45.05 | FM 贏同設定 DDPM (45.05) | 值得繼續：3M 訓練 / stage 2 / drifting model |
| 45.05–46.89 | 持平 DDPM | 看 inference 速度增益決定去留 |
| > 46.89 | 輸 paper | 記錄後放棄，回到 DDPM 軌道 |

即使 HPWL 持平，**若 50 步 ODE ≈ 1000 步 DDPM 品質，inference 20× 加速本身就是貢獻**（bigblue4 eval 從 38 min → 2 min）。

## 5. 風險

| 風險 | 對策 |
|---|---|
| guidance 在 ODE 路徑上不穩 | 先跑 no-guidance 對照；guidance 步長減半 |
| 500k 不夠 FM 收斂 | 觀察 500k 內 val loss 曲線，比照 DDPM 同期 |
| batch=32 PyG 限制 | 已有 wrapper.py chunking fix，直接沿用 |
| GPU 被佔 | 開跑前 nvidia-smi 選空 GPU |

## 6. 行動清單
- [x] Phase A: FlowMatchingModel + ODE sampler + pilot
  - 實作：`diffusion/models.py` 新增 `FlowMatchingModel(ContinuousDiffusionModel)`（覆寫
    `loss`/`forward_samples`/`reverse_samples`；linear/OT path，Euler ODE t=1→0，
    ports mask 每步固定，clamp [-2,2]；guidance 未 port，Phase C 再做）。
    註冊：`train_graph.py` / `eval.py` 的 model_types + family 白名單（`family=flow_matching`，
    重用同一 `model:` config 區段）。
  - Pilot (2026-07-06)：`v1.61-fs.61.fm_p1_pilot.61`，1k steps，loss 2.61 (step100) → 0.58
    (step1000)，無 NaN；step-1000 eval report 正常（hpwl_ratio 1.22, legality 0.7）。
  - Sampler sanity：pilot ckpt，num_timesteps=50 vs 1000 均 finite，值域約 [-1.14, 1.08]。
- [ ] Phase B: 500k 訓練（**執行中** 2026-07-06 起，PID 2986547，GPU 0，
  log: `logs/flowmatch_p1/fm_500k.log`，ckpt: `logs/diffusion_debug/v1.61-fs.61.fm_p1_500k.61/`）
- [ ] Phase C: ISPD 7-circuit eval + 步數 sweep
- [ ] 寫 `docs/report/flowmatch_report_1.md` + `docs/next/flowmatch_next_1.md`
