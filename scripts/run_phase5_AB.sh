#!/usr/bin/env bash
# sampler_plan_1 Phase 5: 5A guidance/legalizer hyperparameter screen (seed 300, 6 cheap circuits) then 5B bigblue2 with guidance.
set -u
R=/ibmnas/427/r115/gclin/Desktop/chipdiffusion; PY=/ibmnas/427/r115/gclin/miniforge3/envs/chipdiff-b/bin/python; cd "$R"
LEG="task=ispd2005-s0 legalizer@_global_=opt-adam legalization.alpha_lr=8e-3 legalization.hpwl_weight=12e-5 legalization.legality_potential_target=0 legalization.grad_descent_steps=20000 macros_only=True logger.wandb=false"
OPT="guidance@_global_=opt model.grad_descent_steps=20 model.hpwl_guidance_weight=16e-4"
DDPM="from_checkpoint=../public-models/large-v2/large-v2.ckpt"
P1="+start_sample=0 num_output_samples=5"; BB3="+start_sample=6 num_output_samples=7"; BB2="+start_sample=5 num_output_samples=6"
SKIP="+skip_guidance_threshold=10000"
run() { local name=$1 seed=$2 want=$3; shift 3; local dir="logs/diffusion_debug/ispd2005-s0.$name.$seed"
  if [ -f "$dir/metrics.csv" ] && [ "$(($(wc -l < "$dir/metrics.csv")-1))" -ge "$want" ]; then echo "$(date '+%m-%d %H:%M') $name skip (done)"; return; fi
  PYTHONPATH=. "$PY" diffusion/eval.py method="$name" seed="$seed" "$@" > "logs/eval_logs/$name.log" 2>&1
  echo "$(date '+%m-%d %H:%M') $name exit=$? $(grep -c 'Finished sample' "logs/eval_logs/$name.log")/$want samples"; }
# name|override  (one factor at a time around the paper recipe; later overrides win in Hydra so these go AFTER $OPT/$LEG)
CFGS=(
 "gdr_4e3|model.grad_descent_rate=4e-3"
 "gdr_16e3|model.grad_descent_rate=16e-3"
 "hgw_8e4|model.hpwl_guidance_weight=8e-4"
 "hgw_32e4|model.hpwl_guidance_weight=32e-4"
 "acf_10|model.alpha_critical_factor=1.0"
 "lpt_0|model.legality_potential_target=0"
 "lpt_1e3|model.legality_potential_target=1e-3"
 "lhw_6e5|legalization.hpwl_weight=6e-5"
 "lhw_24e5|legalization.hpwl_weight=24e-5"
 "lal_4e3|legalization.alpha_lr=4e-3"
 "lal_16e3|legalization.alpha_lr=16e-3"
)
echo "$(date '+%m-%d %H:%M') ## Phase 5A screen (seed 300)"
for c in "${CFGS[@]}"; do n=${c%%|*}; o=${c#*|}
  run hp_${n}_part1 300 5 $DDPM $OPT $LEG $SKIP $P1 $o
  run hp_${n}_bb3   300 1 $DDPM $OPT $LEG $SKIP $BB3 $o
done
echo "$(date '+%m-%d %H:%M') ## Phase 5A screen DONE"
echo "$(date '+%m-%d %H:%M') ## Phase 5B bigblue2 with guidance"
run bb2_guided_s300 300 1 $DDPM $OPT $LEG $BB2
echo "$(date '+%m-%d %H:%M') ## Phase 5B DONE"
