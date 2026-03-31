# Progress Tracker

## Current Goal: Verify pre-trained model performance matches paper

### Step 1: Generate v2 validation data (eval only)
- [x] **In progress** — Generate v2 val data: `PYTHONPATH=. python data-gen/generate_parallel.py versions@_global_=v2 num_train_samples=0 num_val_samples=200`
- Status: **In progress** (restarted 2026-03-18, previous run interrupted before generating any pkl files)

### Step 2: Run evaluation with pre-trained checkpoint
- [ ] Eval: `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python diffusion/eval.py task=v2.61 method=eval from_checkpoint=logs/public-models/large-v2/large-v2.ckpt num_output_samples=128 logger.wandb=False`
- Status: **Not started**

### Step 3: Compare metrics with paper
- [ ] Compare HPWL, legality, etc. against paper's reported numbers
- Status: **Not started**

---

## Full Data Generation (if needed later)

### v0 dataset
- [ ] Generate v0 dataset
- Status: **Not started**

### v1 dataset
- [ ] Generate v1 dataset
- Status: **Not started** (config.yaml created in `data-gen/outputs/v1.61/` but no .pkl files)

### v2 dataset (full)
- [ ] Generate v2 dataset (5000 train + 2500 val)
- Status: **Not started**

## Training / Fine-tuning (if needed later)
- [ ] Train large model on v1
- [ ] Fine-tune on v2
- Pre-trained checkpoint available: `logs/public-models/large-v2/large-v2.ckpt`
