#!/usr/bin/env bash
# Follow-up to sampler_plan_1 Phase 1A: bigblue4 unguided control at seeds 301/302 (evidential upgrade, n=3).
# Usage: bash scripts/run_followup_1A_seeds.sh <pid-to-wait-for>
set -u
R=/ibmnas/427/r115/gclin/Desktop/chipdiffusion; PY=/ibmnas/427/r115/gclin/miniforge3/envs/chipdiff-b/bin/python; cd "$R"
LEG="task=ispd2005-s0 legalizer@_global_=opt-adam +skip_guidance_threshold=10000 legalization.alpha_lr=8e-3 legalization.hpwl_weight=12e-5 legalization.legality_potential_target=0 legalization.grad_descent_steps=20000 macros_only=True logger.wandb=false"
DDPM="from_checkpoint=../public-models/large-v2/large-v2.ckpt"
FM="family=flow_matching from_checkpoint=v1.61-fs.61.fm_p1_500k.61/latest.ckpt model.max_diffusion_steps=50"
BB4="+start_sample=7 num_output_samples=8"
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 60; done; done
echo "$(date '+%m-%d %H:%M') ## Follow-up 1A seeds"
run() { local name=$1 seed=$2 want=$3; shift 3; local dir="logs/diffusion_debug/ispd2005-s0.$name.$seed"
  if [ -f "$dir/metrics.csv" ] && [ "$(($(wc -l < "$dir/metrics.csv")-1))" -ge "$want" ]; then echo "$(date '+%m-%d %H:%M') $name skip (done)"; return; fi
  PYTHONPATH=. "$PY" diffusion/eval.py method="$name" seed="$seed" "$@" > "logs/eval_logs/$name.log" 2>&1
  echo "$(date '+%m-%d %H:%M') $name exit=$? $(grep -c 'Finished sample' "logs/eval_logs/$name.log")/$want samples"; }
for s in 301 302; do
  run diag1A_ddpm_none_bb4_s$s $s 1 $DDPM guidance@_global_=none $LEG $BB4
  run diag1A_fm_none_bb4_s$s   $s 1 $FM   guidance@_global_=none $LEG $BB4
done
echo "$(date '+%m-%d %H:%M') ## Follow-up 1A DONE"
