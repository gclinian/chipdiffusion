# CLAUDE.md

## Progress Tracking Rules
- **Before starting any task**, update `PROGRESS.md` to mark the task as "In progress".
- **After completing a task**, update `PROGRESS.md` to mark it as done (change `[ ]` to `[x]` and update Status to **Done**).
- If a task fails or is interrupted, note the error/status in `PROGRESS.md`.
- This ensures continuity across sessions — the user should never have to re-explain progress.

## Project: chipdiffusion
- Conda environment: `chipdiff` (created from `environment.yaml`; server has only one GPU, no need for `CUDA_VISIBLE_DEVICES`)
- All commands should be run with `PYTHONPATH=.` prefix from the project root
- Generated data goes to `data-gen/outputs/`
- Pre-trained model checkpoint at `logs/public-models/large-v2/large-v2.ckpt`

## Code Modifications (by Claude)
- **`diffusion/eval.py`** has been modified with two new parameters:
  - `start_sample`: Resume eval from a specific sample index (default: 0). Hydra requires `+start_sample=N` prefix.
  - `skip_guidance_threshold`: Auto-disable guidance for circuits with macro count > threshold to avoid OOM (default: 0 = never skip). Hydra requires `+skip_guidance_threshold=N` prefix.
  - **Note**: `skip_guidance_threshold` only skips guidance, NOT legalization. Legalization also uses V×V matrix so bigblue2 (23k macros) will OOM on both. bigblue2 must be skipped entirely on 24GB GPU.
- These are NOT in the original repo. If code is re-cloned, these changes need to be re-applied.

## Repo Structure Gotchas
- **ISPD task name**: User's parsed ISPD data is at `datasets/graph/ispd2005-s0/` (not `ispd2005`). Use `task=ispd2005-s0`. The code has a special case for `dataset_name == "ispd2005"` (zeroes out is_ports) that won't trigger with `ispd2005-s0`.
- **Hydra override syntax**: Use `guidance@_global_=opt` or `legalizer@_global_=opt-adam`, NOT `guidance=opt`. The `@_global_` suffix is required for package overrides. New keys need `+` prefix (e.g., `+start_sample=5`).
- **Guidance OOM on large circuits**: Any circuit with >~10k macros will OOM on guidance AND legalization (V×V matrix). Must use `macros_only=True` and skip bigblue2 entirely, or use clustering.
- **generate_parallel.py num_workers**: Default is 64, which can OOM and kill SSH. Use fewer workers (e.g., 4) or use `generate.py` for single-process.
- **`placements/macro-ispd/`**: These results are from the **paper authors** (checkpoint `v2_gmix1.6_2x...ckpt`), NOT from our runs. They match paper Table 10 exactly.

## Verification Progress (as of 2026-04-01)

### Goal
Verify chipdiffusion paper (arxiv 2407.12282) reported performance on real benchmarks.

### Completed
1. v1 data generated — 2000 val samples in `data-gen/outputs/v1.61/`
2. IBM benchmark downloaded & parsed — `datasets/graph/ibm.cluster512.v1/` and `ibm.cluster0.v1/` (18 circuits each)
3. ISPD2005 benchmark downloaded & parsed — `datasets/graph/ispd2005-s0/` (8 circuits)
4. hmetis installed — `hmetis-1.5-linux/` in repo root
5. ISPD2005 macro-only eval — **completed** for 7/8 circuits (samples 0-4, 6-7). Merged metrics in `logs/diffusion_debug/ispd2005-s0.eval_macro_only.300/metrics.csv`

### ISPD2005 Results (our runs vs paper, seed=300 vs paper seed=400, HPWL x10^5)

| idx | Circuit  | Ours HPWL | Paper HPWL | Ours Legality | Ours Ratio | Note |
|-----|----------|-----------|------------|---------------|------------|------|
| 0   | adaptec1 | 10.22     | 9.19       | 0.9942        | 0.718      |      |
| 1   | adaptec2 | 39.06     | 31.0       | 0.9305        | 1.056      | legality low |
| 2   | adaptec3 | 62.14     | 54.4       | 0.9943        | 0.798      |      |
| 3   | adaptec4 | 60.51     | 54.5       | 0.9965        | 0.679      |      |
| 4   | bigblue1 | 2.69      | 2.64       | 0.9963        | 0.822      | very close |
| 5   | bigblue2 | skipped   | 38.8       | -             | -          | OOM (23k macros) |
| 6   | bigblue3 | 34.26     | 35.9       | 0.9951        | 0.598      | better than paper |
| 7   | bigblue4 | 131.96    | 140.6      | 0.9914        | 0.491      | better than paper |

Detailed comparison saved in `logs/diffusion_debug/ispd2005-s0.eval_macro_only.300/comparison_with_paper.csv`.

### Differences from Paper's Eval Process
- **Seed**: ours=300, paper=400 — likely source of per-circuit variation
- **Guidance**: all 7 circuits had guidance **enabled** (skip_guidance_threshold=10000 was not triggered for any circuit we ran). Same as paper.
- **bigblue2 skipped**: paper ran it (likely with >24GB GPU), we OOM on both guidance and legalization
- **Hyperparameters**: identical to paper (guidance steps=20, legalization steps=20000, etc.)
- **Checkpoint**: same pre-trained `large-v2.ckpt`

### In Progress / Not Done
- v2 data not generated
- IBM eval not run yet

### ISPD2005 Macro Counts
| idx | Circuit  | Macros | Runnable on 24GB? |
|-----|----------|--------|-------------------|
| 0   | adaptec1 | 543    | Yes               |
| 1   | adaptec2 | 566    | Yes               |
| 2   | adaptec3 | 723    | Yes               |
| 3   | adaptec4 | 1,329  | Yes               |
| 4   | bigblue1 | 560    | Yes               |
| 5   | bigblue2 | 23,084 | No (OOM)          |
| 6   | bigblue3 | 1,298  | Yes               |
| 7   | bigblue4 | 8,170  | Yes               |

## Correct Eval Commands

### ISPD2005 macro-only eval
```bash
PYTHONPATH=. python diffusion/eval.py \
  method=eval_macro_only \
  task=ispd2005-s0 \
  from_checkpoint=logs/public-models/large-v2/large-v2.ckpt \
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

### IBM clustered eval
```bash
PYTHONPATH=. python diffusion/eval.py \
  method=eval_guided \
  task=ibm.cluster512.v1 \
  from_checkpoint=logs/public-models/large-v2/large-v2.ckpt \
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
