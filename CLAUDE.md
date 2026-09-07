# CLAUDE.md

> 這個檔案是**工具全部關掉也還有效**的那一層。自動化（`scripts/ledger.py`、
> `STATUS.md`）會失效、會忘記跑、會在別台機器上不存在；下面這些規則不會。

## ⛔ 環境現況：目前跑不了任何 GPU 工作（2026-09-07）

伺服器 GPU 已換成 **RTX 5090（32,607 MiB，compute capability sm_120）**。
`chipdiff` env 的 `torch 2.2.1+cu121` 編譯的 kernel 只到 `sm_90`：

```
arch_list ['sm_50','sm_60','sm_70','sm_75','sm_80','sm_86','sm_90']
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**`torch.cuda.is_available()` 會回傳 True**，所以這不會在啟動時報錯，而是在
第一次 kernel launch 才炸 —— 看起來像實驗跑到一半神秘崩潰。`diffplace` env
（torch 2.6.0+cu124）同樣壞掉。

修法：**開新 env**（保留 `chipdiff` 原狀）升到 torch ≥ 2.7 / cu128。本專案沒有
安裝任何需編譯的 PyG extension（`torch-scatter`/`sparse`/`cluster` 皆未安裝），
所以是 pip 層級升級，不是 C++ 重編。

**升級後第一件事：重跑 Run F（預期 44.32）驗證。** 不做這步，升級後的所有數字
與磁碟上既有的 25 個結果都不可比。

副作用：VRAM 24 → 32.6 GB。bigblue2 guidance 舊估 30–35 GB，現在是邊緣可行，
但那個估計是舊 torch/舊 allocator 下量的，要重新量測。

## 判讀規則（違反這些的結論不算數）

1. **n=1 不得關閉任何研究方向。** 已測得最大 across-seed σ = 0.654（SVDD, n=3），
   所以 7-circuit avg HPWL 上**小於 1.3 的差距等於沒有差距**。
2. **禁令要分類。** 寫進 `next_N` 的 do-not-do 要標明是
   **evidential**（有 Δ、有 seed 數 → 需 n≥3 且 |Δ| ≥ 2σ）還是
   **prudential**（成本效益判斷 → 明說「這是判斷不是證據」）。
3. **決策 band 不得錨定在 n=1 或原始資料已遺失的數字上。**
4. **Baseline framing。** 微調結果只能對 **48.691**（我們自己對 paper checkpoint
   的復現）比較；**不能**對 46.89（paper 已發表值、不同 seed、未微調）比較。
   對外最強且唯一有誤差棒的說法是 SVDD **44.845 ± 0.654**（3 seeds，在 paper
   自己的 frozen checkpoint 上，零訓練）。
5. **跨 track 借用證據要標來源。** CoDe 與 TDS 的多條結論其實引用的是 SVDD 的
   單 seed run。
6. **per-circuit 主張需要更高的門檻。** bigblue3 的 σ ≈ 5.34，是全專案 σ 的 8 倍；
   單 seed 的「X 在 5/7 circuits 勝 Y」在統計上是空的。

> 這些不是理論潔癖。專案最高價值的方向（fine-tuning × inference-search 組合）
> 就是被一個 Δ=0.164 的單 seed 結果關掉的，而那個 Δ 比它自己的 σ 小四倍。
> 同型錯誤另外發生過兩次（bigblue3 假警報、baseline framing 錯置）。

## Progress Tracking Rules
- **開始任何任務前**，在 `PROGRESS.md` 標記 "In progress"。
- **完成後**改成 `[x]` 並更新 Status 為 **Done**。
- 失敗或中斷要把錯誤/狀態記進 `PROGRESS.md`。

## Documentation Rules
- 所有文件統一放在 `docs/` 下
- **實驗報告**：`docs/report/<name>_report_<N>.md`
- **計畫文件**：`docs/plan/<name>_plan_<N>.md`
- **回顧筆記**：`docs/next/<name>_next_<N>.md`
- **會議記錄**：`docs/meet/meet_<MMDD>.md`
- 流程：`plan_N` → 執行 → `report_N` → `next_N` → `plan_N+1`（參考 `next_N`）
- **報告是歷史紀錄，不重寫。** 結論被推翻時在檔案頂端加更正 banner 並指向新報告，
  保留原文。範例見 `docs/report/svdd_report_3.md`。
- `report_N` 必須引用當初 `plan_N` 預先登記的判定標準。**事後推翻門檻要寫明理由** —
  專案裡發生過兩次（AddLoss v2 落在「繼續」區間卻被放棄；DDPO v2.5 只超門檻 0.057
  就被放棄且跳過 plan 寫好的 contingency K=10）。

## 實驗紀錄（ledger）

`scripts/ledger.py` 是實驗的機器可讀紀錄。設計原則：**無條件執行、衍生而非手寫、
純 stdlib（torch 壞掉也能跑）、絕不致命**。

- `eval.py` 和 `train_graph.py` 會**在自己的 process 內**自動呼叫 `record()` —
  所以在 tmux 裡手動敲的 run 也會進紀錄，不依賴任何 hook。
- `python3 scripts/ledger.py scan` — 補掃磁碟上所有 run（backstop）並重新產生 STATUS.md
- `python3 scripts/ledger.py status` — 只重新產生 STATUS.md

`STATUS.md` 是自動產生的，**不要手改**。它有三個工作佇列，是專案歷史上東西真正
掉的三個地方：中斷的 eval / 訓練完但沒 eval 的 checkpoint / 有結果但 docs 裡找不到
那個數字。每個 run 的 `metrics.csv` 逐字存進 `docs/ledger/results/`（每筆 ~2 KB），
所以數字不依賴被 gitignore 掉的 11 GB `logs/`。

## Current Experiment Status

**現況數字看 `STATUS.md`（自動產生）；敘事脈絡看 `docs/context.txt`；
2026-09 的完整交叉稽核看 `docs/next/comeback_next_1.md`。**
CLAUDE.md 只放不會變的規則與 gotchas。

不要把 leaderboard 抄到這裡 —— 這個檔案和 `docs/context.txt` 都曾因為抄了數字而
過期四個月，並讓後續 session 拿到錯誤的圖像。

## Code Modifications

- **`diffusion/eval.py`** new params (require `+` Hydra prefix):
  - `+start_sample=N`: Resume eval from sample index N
  - `+skip_guidance_threshold=N`: Auto-disable guidance for circuits with
    macros > N. **Only skips guidance, not legalization.** Legalization
    actually fits in 24GB for bigblue2 (takes ~100 min though).

- **`diffusion/ddpo.py`** params: `num_timesteps`, `clip_epsilon`,
  `supervised_weight`, `local_reward_weight`, `local_reward_every`,
  `local_reward_last_k`. Implements PPO-style clipping, log-prob scale
  normalization, mixed supervised loss, and (optional, last-K) local reward.

- **`diffusion/models.py`** `ContinuousDiffusionModel.loss()` accepts
  `hpwl_weight`, `legality_weight`, `use_timestep_weighting` for ReFL-style
  auxiliary loss on predicted_x0.

- **`diffusion/models.py`** `reverse_samples()` accepts `output_log_prob=True`
  to also return per-step log probs and predicted_x0 list (used by DDPO).

- **`diffusion/train_graph.py`** glue to pass these new params through.

- **`diffusion/configs/mode/ddpo.yaml`** legality_weight default 0.0 → 0.5.

## Repo Structure Gotchas
- **Eval 目錄碰撞（已加防護，但要知道為什麼）**: run dir 命名是
  `<task>.<method>.<seed>`，**不含 checkpoint**。七個微調 eval 都用
  `method=eval_macro_only` 寫進同一個目錄，只有最後一個活下來 —— Ablation_10k
  (44.01)、DDPO v2 (44.65)、AddLoss v1、DDPO v2.3/2.4/2.5 的原始 metrics 永久消失
  （checkpoint 還在，所以是重跑 eval 不是重訓）。現在 `eval.py` /
  `train_graph.py` 會在寫入前檢查既有 `config.yaml` 的 `from_checkpoint`，不同就
  中止。**每個 eval 用獨一無二的 `method=` 名稱**；要故意覆蓋才加 `+allow_overwrite=true`。
- **ISPD task name**: User's parsed ISPD data is at `datasets/graph/ispd2005-s0/` (not `ispd2005`). Use `task=ispd2005-s0`. The code has a special case for `dataset_name == "ispd2005"` (zeroes out is_ports) that won't trigger with `ispd2005-s0`.
- **Hydra override syntax**: Use `guidance@_global_=opt` or `legalizer@_global_=opt-adam`, NOT `guidance=opt`. The `@_global_` suffix is required for package overrides. New keys need `+` prefix (e.g., `+start_sample=5`). Same applies to `mode@_global_=finetune` / `mode@_global_=ddpo`.
- **`from_checkpoint` path bug** (IMPORTANT): `eval.py` and `train_graph.py` do `os.path.join(cfg.log_dir, cfg.from_checkpoint)`. Since `log_dir=logs/diffusion_debug`, passing `from_checkpoint=logs/diffusion_debug/X/latest.ckpt` produces `logs/diffusion_debug/logs/diffusion_debug/X/latest.ckpt` — doubled path, silent fail. Always use **relative to log_dir**:
  - From a saved run: `from_checkpoint=v1.61-ddpo.<method>.61/latest.ckpt`
  - From pretrained: `from_checkpoint=../public-models/large-v2/large-v2.ckpt`
  - **Always verify** the eval log contains "successfully loaded state dict for model" — if it says "no checkpoint at ... found", the path is wrong.
- **bigblue2 (23k macros)**: Guidance V×V matrix OOMed on the OLD 24GB card (re-measure on 32.6 GB). Set `+skip_guidance_threshold=10000` to auto-skip guidance for this circuit. **Legalization itself does NOT OOM** (takes ~100 min though). bigblue2 still loses to paper (HPWL 57-66 vs paper 38.8) because we have no guidance.
- **Guidance OOM on large circuits**: Circuits with >~10k macros OOMed on guidance (V×V matrix) on the OLD 24GB card. Use `+skip_guidance_threshold=10000` and `macros_only=True`.
- **generate_parallel.py num_workers**: Default is 64, which can OOM and kill SSH. Use fewer workers (e.g., 4) or use `generate.py` for single-process.
- **`placements/macro-ispd/`**: These results are from the **paper authors** (checkpoint `v2_gmix1.6_2x...ckpt`), NOT from our runs. They match paper Table 10 exactly.
- **System CPU contention** (NAS server): Other users (ansys.e, redhawk+) can crater eval speed 5-13x. Check `uptime` and `top` if eval seems abnormally slow. The GPU may show 11% util but actual compute is CPU-bound.

## ISPD2005 Reference Tables (stable info)

### ISPD2005 Macro Counts and runnability

> 下表的 OOM 判斷是在**舊的 24 GB 卡**上量的。現在是 32.6 GB，
> bigblue2 那一列需要重新量測（見檔案開頭的環境現況）。
| idx | Circuit  | Macros | Guidance? | Legalization? |
|-----|----------|--------|-----------|---------------|
| 0   | adaptec1 | 543    | ✓         | ✓             |
| 1   | adaptec2 | 566    | ✓         | ✓             |
| 2   | adaptec3 | 723    | ✓         | ✓             |
| 3   | adaptec4 | 1,329  | ✓         | ✓             |
| 4   | bigblue1 | 560    | ✓         | ✓             |
| 5   | bigblue2 | 23,084 | ✗ OOM (V×V) | ✓ (~100 min)  |
| 6   | bigblue3 | 1,298  | ✓         | ✓             |
| 7   | bigblue4 | 8,170  | ✓         | ✓ (~25 min)   |

For bigblue2: use `+skip_guidance_threshold=10000` to auto-disable guidance.
Without paper-style guidance, HPWL stays ~57-66 vs paper's 38.8.

### Baseline ISPD2005 Results (large-v2.ckpt, seed=300, HPWL x10^5)
| Circuit | Baseline | Paper | Note |
|---------|---------:|------:|------|
| adaptec1 | 10.22 | 9.19 | |
| adaptec2 | 39.06 | 31.0 | baseline legality low (0.93) |
| adaptec3 | 62.14 | 54.4 | |
| adaptec4 | 60.51 | 54.5 | |
| bigblue1 | 2.69 | 2.64 | very close |
| bigblue2 | skip | 38.8 | guidance OOM on the OLD 24GB card |
| bigblue3 | 34.26 | 35.9 | beats paper |
| bigblue4 | 131.96 | 140.6 | beats paper |
| Avg (7, no bb2) | **48.69** | **46.89** | |

> 這張表是**我們自己跑 paper checkpoint** 的結果，是唯一同環境的比較基準。
> 微調結果要對 **48.69** 比較，不是對 46.89。見開頭「判讀規則」第 4 條。
> 完整排名見 `STATUS.md`（自動產生）與 `docs/all_experiments_summary.csv`。

### Differences from Paper's Eval Process
- **Seed**: ours=300, paper=400 — source of per-circuit variation
- **bigblue2**: paper ran with full guidance; we disabled it via
  `skip_guidance_threshold=10000` because the old 24GB card OOMed. With 32.6 GB
  this should be re-tested — bigblue2 is the only circuit that has never beaten
  the paper, and including it flips the 8-circuit average from −3.9% to +1.1%.
- **Hyperparameters**: identical to paper (guidance steps=20, legalization
  steps=20000, etc.) for all other circuits

## Correct Eval Commands

> NOTE: `from_checkpoint` is joined with `log_dir=logs/diffusion_debug`, so
> paths MUST be relative to that. See the path-bug gotcha above.

### ISPD2005 macro-only eval (from pretrained / paper baseline)
```bash
PYTHONPATH=. python diffusion/eval.py \
  method=eval_macro_only \
  task=ispd2005-s0 \
  from_checkpoint=../public-models/large-v2/large-v2.ckpt \
  legalizer@_global_=opt-adam \
  guidance@_global_=opt \
  num_output_samples=8 \
  +start_sample=0 \
  +skip_guidance_threshold=10000 \
  model.grad_descent_steps=20 \
  model.hpwl_guidance_weight=16e-4 \
  legalization.alpha_lr=8e-3 \
  legalization.hpwl_weight=12e-5 \
  legalization.legality_potential_target=0 \
  legalization.grad_descent_steps=20000 \
  macros_only=True \
  logger.wandb=false
```

### ISPD2005 eval from a fine-tuned checkpoint
Replace the `from_checkpoint` with the run's relative path, e.g.:
```
from_checkpoint=v1.61-ddpo.ablation_supervised_10k.61/latest.ckpt
```

### IBM clustered eval (not yet validated this project run)
```bash
PYTHONPATH=. python diffusion/eval.py \
  method=eval_guided \
  task=ibm.cluster512.v1 \
  from_checkpoint=../public-models/large-v2/large-v2.ckpt \
  num_output_samples=18 \
  cluster.cached_clusters=true \
  logger.wandb=false
```

### Paper's expected ISPD2005 results (Table 10, HPWL x10^5)
| Circuit  | MaskPlace | WireMask-BBO | ChiPFormer | Diffusion (Paper) |
|----------|-----------|--------------|------------|-------------------|
| adaptec1 | 8.57      | 5.81         | 6.75       | 9.19              |
| adaptec2 | 77.7      | 54.5         | 63.8       | 31.0              |
| adaptec3 | 108       | 59.2         | 73.2       | 54.4              |
| adaptec4 | 91.9      | 62.7         | 85.8       | 54.5              |
| bigblue1 | 3.11      | 2.12         | 3.05       | 2.64              |
| bigblue2 | Timeout   | 186          | 85.8       | 38.8              |
| bigblue3 | 84.0      | 66.2         | 79.2       | 35.9              |
| bigblue4 | Timeout   | 798          | 548        | 141               |
| Average  | -         | 154          | 116        | 45.9              |

Note: `hpwl_rescaled / 100` in metrics.csv = paper HPWL value (x10^5).
