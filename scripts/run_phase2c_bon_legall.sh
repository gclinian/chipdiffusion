#!/usr/bin/env bash
# sampler_plan_1 Phase 2c: best-of-4 with EVERY candidate legalized (post-leg HPWL selection). Usage: bash scripts/run_phase2c_bon_legall.sh <pid>
set -u
R=/ibmnas/427/r115/gclin/Desktop/chipdiffusion; PY=/ibmnas/427/r115/gclin/miniforge3/envs/chipdiff-b/bin/python; cd "$R"
LEG="task=ispd2005-s0 legalizer@_global_=opt-adam +skip_guidance_threshold=10000 legalization.alpha_lr=8e-3 legalization.hpwl_weight=12e-5 legalization.legality_potential_target=0 legalization.grad_descent_steps=20000 macros_only=True logger.wandb=false"
OPT="guidance@_global_=opt model.grad_descent_steps=20 model.hpwl_guidance_weight=16e-4"; DDPM="from_checkpoint=../public-models/large-v2/large-v2.ckpt"
P1="+start_sample=0 num_output_samples=5"; BB3="+start_sample=6 num_output_samples=7"
BON="+num_candidates=4 +candidate_legality_floor=0.97 +legalize_all_candidates=true"
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 60; done; done
echo "$(date '+%m-%d %H:%M') ## Phase 2c best-of-4 legalize-all"
run() { local name=$1 seed=$2 want=$3; shift 3; local dir="logs/diffusion_debug/ispd2005-s0.$name.$seed"
  if [ -f "$dir/metrics.csv" ] && [ "$(($(wc -l < "$dir/metrics.csv")-1))" -ge "$want" ]; then echo "$(date '+%m-%d %H:%M') $name skip (done)"; return; fi
  PYTHONPATH=. "$PY" diffusion/eval.py method="$name" seed="$seed" "$@" > "logs/eval_logs/$name.log" 2>&1
  echo "$(date '+%m-%d %H:%M') $name exit=$? $(grep -c 'Finished sample' "logs/eval_logs/$name.log")/$want samples"; }
for s in 300 301 302; do run bon2c_s${s}_part1 $s 5 $DDPM $OPT $LEG $P1 $BON; run bon2c_s${s}_bb3 $s 1 $DDPM $OPT $LEG $BB3 $BON; done
echo "$(date '+%m-%d %H:%M') ## Phase 2c DONE"
