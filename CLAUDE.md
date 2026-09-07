# CLAUDE.md

## Progress Tracking Rules
- **Before starting any task**, update `PROGRESS.md` to mark the task as "In progress".
- **After completing a task**, update `PROGRESS.md` to mark it as done (change `[ ]` to `[x]` and update Status to **Done**).
- If a task fails or is interrupted, note the error/status in `PROGRESS.md`.
- This ensures continuity across sessions — the user should never have to re-explain progress.

## Documentation Rules
- 所有文件統一放在 `docs/` 下
- **實驗報告**：`docs/report/<name>_report_<N>.md`（例如 `ddpo_report_1.md`）
- **計畫文件**：`docs/plan/<name>_plan_<N>.md`（例如 `ddpo_plan_1.md`）
- **回顧筆記**：`docs/next/<name>_next_<N>.md`（例如 `ddpo_next_1.md`），記錄實驗的失敗原因、教訓、下次改進方向
- **會議記錄**：`docs/meet/meet_<MMDD>.md`（例如 `meet_0403.md`），記錄教授建議
- 流程：`plan_N` → 執行實驗 → `report_N` → `next_N`（回顧）→ `plan_N+1`（參考 `next_N`）
- 每次實驗結束或有階段性結果時，更新對應的 `_report_N.md`
- 新實驗開始前，先參考 `_next_N-1.md`，再建立 `_plan_N.md`

## Project: chipdiffusion
- Conda environment: `chipdiff` (created from `environment.yaml`; server has only one GPU, no need for `CUDA_VISIBLE_DEVICES`)
- All commands should be run with `PYTHONPATH=.` prefix from the project root
- Generated data goes to `data-gen/outputs/`
- Pre-trained model checkpoint at `logs/public-models/large-v2/large-v2.ckpt`

## Current Experiment Status

**See `docs/context.txt` for up-to-date experiment status, method rankings,
and pending directions.** This CLAUDE.md only contains persistent rules and
gotchas; experimental results change too often to track here.

Current best (as of latest update): **Ablation 10k** (純 supervised
fine-tuning, 10000 steps on v1.61-ddpo) → 7-circuit avg HPWL 44.01 (beats
paper's 46.89 by 6.1%).

## Code Modifications (by Claude, all committed)

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
- **ISPD task name**: User's parsed ISPD data is at `datasets/graph/ispd2005-s0/` (not `ispd2005`). Use `task=ispd2005-s0`. The code has a special case for `dataset_name == "ispd2005"` (zeroes out is_ports) that won't trigger with `ispd2005-s0`.
- **Hydra override syntax**: Use `guidance@_global_=opt` or `legalizer@_global_=opt-adam`, NOT `guidance=opt`. The `@_global_` suffix is required for package overrides. New keys need `+` prefix (e.g., `+start_sample=5`). Same applies to `mode@_global_=finetune` / `mode@_global_=ddpo`.
- **`from_checkpoint` path bug** (IMPORTANT): `eval.py` and `train_graph.py` do `os.path.join(cfg.log_dir, cfg.from_checkpoint)`. Since `log_dir=logs/diffusion_debug`, passing `from_checkpoint=logs/diffusion_debug/X/latest.ckpt` produces `logs/diffusion_debug/logs/diffusion_debug/X/latest.ckpt` — doubled path, silent fail. Always use **relative to log_dir**:
  - From a saved run: `from_checkpoint=v1.61-ddpo.<method>.61/latest.ckpt`
  - From pretrained: `from_checkpoint=../public-models/large-v2/large-v2.ckpt`
  - **Always verify** the eval log contains "successfully loaded state dict for model" — if it says "no checkpoint at ... found", the path is wrong.
- **bigblue2 (23k macros)**: Guidance V×V matrix OOMs on 24GB. Set `+skip_guidance_threshold=10000` to auto-skip guidance for this circuit. **Legalization itself does NOT OOM** (takes ~100 min though). bigblue2 still loses to paper (HPWL 57-66 vs paper 38.8) because we have no guidance.
- **Guidance OOM on large circuits**: Circuits with >~10k macros will OOM on guidance (V×V matrix). Use `+skip_guidance_threshold=10000` and `macros_only=True`.
- **generate_parallel.py num_workers**: Default is 64, which can OOM and kill SSH. Use fewer workers (e.g., 4) or use `generate.py` for single-process.
- **`placements/macro-ispd/`**: These results are from the **paper authors** (checkpoint `v2_gmix1.6_2x...ckpt`), NOT from our runs. They match paper Table 10 exactly.
- **System CPU contention** (NAS server): Other users (ansys.e, redhawk+) can crater eval speed 5-13x. Check `uptime` and `top` if eval seems abnormally slow. The GPU may show 11% util but actual compute is CPU-bound.

## ISPD2005 Reference Tables (stable info)

### ISPD2005 Macro Counts and runnability on 24GB GPU
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
| bigblue2 | skip | 38.8 | guidance OOM (24GB) |
| bigblue3 | 34.26 | 35.9 | beats paper |
| bigblue4 | 131.96 | 140.6 | beats paper |
| Avg (7, no bb2) | **48.69** | **46.89** | |

Our best (Ablation 10k fine-tune): 7-circuit avg **44.01** (−6.1% vs paper).
See docs/context.txt for the full method ranking and per-circuit results.

### Differences from Paper's Eval Process
- **Seed**: ours=300, paper=400 — source of per-circuit variation
- **bigblue2**: paper ran with full guidance (likely >24GB GPU or clusters);
  we disable guidance via `skip_guidance_threshold=10000`
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
