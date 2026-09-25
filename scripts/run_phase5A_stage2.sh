#!/usr/bin/env bash
# Phase 5A stage 2: seeds 301/302 for screen-passing configs. Usage: bash scripts/run_phase5A_stage2.sh <wait-pid> <cfg> [<cfg> ...]
# <cfg> is the short name used in run_phase5_AB.sh (e.g. gdr_4e3); the override is looked up from the same table.
set -u
R=/ibmnas/427/r115/gclin/Desktop/chipdiffusion; PY=/ibmnas/427/r115/gclin/miniforge3/envs/chipdiff-b/bin/python; cd "$R"
LEG="task=ispd2005-s0 legalizer@_global_=opt-adam legalization.alpha_lr=8e-3 legalization.hpwl_weight=12e-5 legalization.legality_potential_target=0 legalization.grad_descent_steps=20000 macros_only=True logger.wandb=false"
OPT="guidance@_global_=opt model.grad_descent_steps=20 model.hpwl_guidance_weight=16e-4"
DDPM="from_checkpoint=../public-models/large-v2/large-v2.ckpt"; SKIP="+skip_guidance_threshold=10000"
P1="+start_sample=0 num_output_samples=5"; BB3="+start_sample=6 num_output_samples=7"
declare -A OV=( [gdr_4e3]="model.grad_descent_rate=4e-3" [gdr_16e3]="model.grad_descent_rate=16e-3"
  [hgw_8e4]="model.hpwl_guidance_weight=8e-4" [hgw_32e4]="model.hpwl_guidance_weight=32e-4"
  [acf_10]="model.alpha_critical_factor=1.0" [lpt_0]="model.legality_potential_target=0" [lpt_1e3]="model.legality_potential_target=1e-3"
  [lhw_6e5]="legalization.hpwl_weight=6e-5" [lhw_24e5]="legalization.hpwl_weight=24e-5"
  [lal_4e3]="legalization.alpha_lr=4e-3" [lal_16e3]="legalization.alpha_lr=16e-3" )
wait_pid=$1; shift
while kill -0 "$wait_pid" 2>/dev/null; do sleep 60; done
run() { local name=$1 seed=$2 want=$3; shift 3; local dir="logs/diffusion_debug/ispd2005-s0.$name.$seed"
  if [ -f "$dir/metrics.csv" ] && [ "$(($(wc -l < "$dir/metrics.csv")-1))" -ge "$want" ]; then echo "$(date '+%m-%d %H:%M') $name skip (done)"; return; fi
  PYTHONPATH=. "$PY" diffusion/eval.py method="$name" seed="$seed" "$@" > "logs/eval_logs/$name.log" 2>&1
  echo "$(date '+%m-%d %H:%M') $name exit=$? $(grep -c 'Finished sample' "logs/eval_logs/$name.log")/$want samples"; }
echo "$(date '+%m-%d %H:%M') ## Phase 5A stage 2: $*"
for c in "$@"; do o=${OV[$c]:?unknown config $c}
  for s in 301 302; do run hp_${c}_part1 $s 5 $DDPM $OPT $LEG $SKIP $P1 $o; run hp_${c}_bb3 $s 1 $DDPM $OPT $LEG $SKIP $BB3 $o; done
done
echo "$(date '+%m-%d %H:%M') ## Phase 5A stage 2 DONE"
