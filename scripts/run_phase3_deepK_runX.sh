#!/usr/bin/env bash
# sampler_plan_1 Phase 3E (deep guidance K, config-only) + Run X same-stack re-measure (the pair for 3F).
# Usage: bash scripts/run_phase3_deepK_runX.sh <pid-to-wait-for>
set -u
R=/ibmnas/427/r115/gclin/Desktop/chipdiffusion; PY=/ibmnas/427/r115/gclin/miniforge3/envs/chipdiff-b/bin/python; cd "$R"
LEG="task=ispd2005-s0 legalizer@_global_=opt-adam +skip_guidance_threshold=10000 legalization.alpha_lr=8e-3 legalization.hpwl_weight=12e-5 legalization.legality_potential_target=0 legalization.grad_descent_steps=20000 macros_only=True logger.wandb=false"
HW="model.hpwl_guidance_weight=16e-4"; OPT="guidance@_global_=opt model.grad_descent_steps=20 $HW"
DDPM="from_checkpoint=../public-models/large-v2/large-v2.ckpt"; RUNX="from_checkpoint=v1.61-fs.61.fs_p1_X_500k.61/latest.ckpt"
A1="+start_sample=0 num_output_samples=1"; BB4="+start_sample=7 num_output_samples=8"; P1="+start_sample=0 num_output_samples=5"; P2="+start_sample=6 num_output_samples=8"
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 60; done; done
echo "$(date '+%m-%d %H:%M') ## Phase 3E deep-K + Run X re-measure"
run() { local name=$1 seed=$2 want=$3; shift 3; local dir="logs/diffusion_debug/ispd2005-s0.$name.$seed"
  if [ -f "$dir/metrics.csv" ] && [ "$(($(wc -l < "$dir/metrics.csv")-1))" -ge "$want" ]; then echo "$(date '+%m-%d %H:%M') $name skip (done)"; return; fi
  PYTHONPATH=. "$PY" diffusion/eval.py method="$name" seed="$seed" "$@" > "logs/eval_logs/$name.log" 2>&1
  echo "$(date '+%m-%d %H:%M') $name exit=$? $(grep -c 'Finished sample' "logs/eval_logs/$name.log")/$want samples"; }
# Run X on the current stack: the comparator for Phase 3F (old 45.053 is old-stack)
run runX_cu128_s300_part1 300 5 $RUNX $OPT $LEG $P1
run runX_cu128_s300_part2 300 2 $RUNX $OPT $LEG $P2
# 3E: deep K on bigblue4 (K=20 acf=0.5 is baseline 0a), plus one adaptec1 cell
for K in 60 150; do for acf in 0.5 1.0; do
  run deepK${K}_acf${acf/./}_bb4 300 1 $DDPM guidance@_global_=opt model.grad_descent_steps=$K $HW model.alpha_critical_factor=$acf $LEG $BB4
done; done
run deepK150_acf05_a1 300 1 $DDPM guidance@_global_=opt model.grad_descent_steps=150 $HW model.alpha_critical_factor=0.5 $LEG $A1
echo "$(date '+%m-%d %H:%M') ## Phase 3E DONE"
