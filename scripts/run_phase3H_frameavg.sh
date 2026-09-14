#!/usr/bin/env bash
# sampler_plan_1 Phase 3H: D4 frame-averaged eps (8x forwards), adaptec1 + bigblue4. Usage: bash scripts/run_phase3H_frameavg.sh <pid-to-wait-for>
set -u
R=/ibmnas/427/r115/gclin/Desktop/chipdiffusion; PY=/ibmnas/427/r115/gclin/miniforge3/envs/chipdiff-b/bin/python; cd "$R"
LEG="task=ispd2005-s0 legalizer@_global_=opt-adam +skip_guidance_threshold=10000 legalization.alpha_lr=8e-3 legalization.hpwl_weight=12e-5 legalization.legality_potential_target=0 legalization.grad_descent_steps=20000 macros_only=True logger.wandb=false"
OPT="guidance@_global_=opt model.grad_descent_steps=20 model.hpwl_guidance_weight=16e-4"; DDPM="from_checkpoint=../public-models/large-v2/large-v2.ckpt"
A1="+start_sample=0 num_output_samples=1"; BB4="+start_sample=7 num_output_samples=8"
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 60; done; done
echo "$(date '+%m-%d %H:%M') ## Phase 3H frame averaging"
run() { local name=$1 seed=$2 want=$3; shift 3; local dir="logs/diffusion_debug/ispd2005-s0.$name.$seed"
  if [ -f "$dir/metrics.csv" ] && [ "$(($(wc -l < "$dir/metrics.csv")-1))" -ge "$want" ]; then echo "$(date '+%m-%d %H:%M') $name skip (done)"; return; fi
  PYTHONPATH=. "$PY" diffusion/eval.py method="$name" seed="$seed" "$@" > "logs/eval_logs/$name.log" 2>&1
  echo "$(date '+%m-%d %H:%M') $name exit=$? $(grep -c 'Finished sample' "logs/eval_logs/$name.log")/$want samples"; }
run frameavg_a1  300 1 $DDPM $OPT $LEG $A1  +model.frame_average=true
run frameavg_bb4 300 1 $DDPM $OPT $LEG $BB4 +model.frame_average=true
echo "$(date '+%m-%d %H:%M') ## Phase 3H DONE"
