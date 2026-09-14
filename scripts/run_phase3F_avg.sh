#!/usr/bin/env bash
# sampler_plan_1 Phase 3F: post-hoc weight-averaged checkpoints vs same-stack Run X. Usage: bash scripts/run_phase3F_avg.sh <pid-to-wait-for>
set -u
R=/ibmnas/427/r115/gclin/Desktop/chipdiffusion; PY=/ibmnas/427/r115/gclin/miniforge3/envs/chipdiff-b/bin/python; cd "$R"
LEG="task=ispd2005-s0 legalizer@_global_=opt-adam +skip_guidance_threshold=10000 legalization.alpha_lr=8e-3 legalization.hpwl_weight=12e-5 legalization.legality_potential_target=0 legalization.grad_descent_steps=20000 macros_only=True logger.wandb=false"
OPT="guidance@_global_=opt model.grad_descent_steps=20 model.hpwl_guidance_weight=16e-4"
P1="+start_sample=0 num_output_samples=5"; P2="+start_sample=6 num_output_samples=8"
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 60; done; done
echo "$(date '+%m-%d %H:%M') ## Phase 3F post-hoc averaged checkpoints"
run() { local name=$1 seed=$2 want=$3; shift 3; local dir="logs/diffusion_debug/ispd2005-s0.$name.$seed"
  if [ -f "$dir/metrics.csv" ] && [ "$(($(wc -l < "$dir/metrics.csv")-1))" -ge "$want" ]; then echo "$(date '+%m-%d %H:%M') $name skip (done)"; return; fi
  PYTHONPATH=. "$PY" diffusion/eval.py method="$name" seed="$seed" "$@" > "logs/eval_logs/$name.log" 2>&1
  echo "$(date '+%m-%d %H:%M') $name exit=$? $(grep -c 'Finished sample' "logs/eval_logs/$name.log")/$want samples"; }
for w in 250k_500k 400k_500k; do
  CK="from_checkpoint=v1.61-fs.61.fs_p1_X_500k.61/avg_${w}_uniform.ckpt"
  run runXavg_${w}_part1 300 5 $CK $OPT $LEG $P1
  run runXavg_${w}_part2 300 2 $CK $OPT $LEG $P2
done
echo "$(date '+%m-%d %H:%M') ## Phase 3F DONE"
