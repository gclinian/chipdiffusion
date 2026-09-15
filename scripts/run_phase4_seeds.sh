#!/usr/bin/env bash
# sampler_plan_1 Phase 4: Run X seeds, DataAug Run C eval, baseline + best-of-4 seeds 303-305. Usage: bash scripts/run_phase4_seeds.sh
set -u
R=/ibmnas/427/r115/gclin/Desktop/chipdiffusion; PY=/ibmnas/427/r115/gclin/miniforge3/envs/chipdiff-b/bin/python; cd "$R"
LEG="task=ispd2005-s0 legalizer@_global_=opt-adam +skip_guidance_threshold=10000 legalization.alpha_lr=8e-3 legalization.hpwl_weight=12e-5 legalization.legality_potential_target=0 legalization.grad_descent_steps=20000 macros_only=True logger.wandb=false"
OPT="guidance@_global_=opt model.grad_descent_steps=20 model.hpwl_guidance_weight=16e-4"
DDPM="from_checkpoint=../public-models/large-v2/large-v2.ckpt"
RUNX="from_checkpoint=v1.61-fs.61.fs_p1_X_500k.61/latest.ckpt"
RUNC="from_checkpoint=v1.61-fs.61.aug_p1_C_dihedral_dropout_500k.61/latest.ckpt"
P1="+start_sample=0 num_output_samples=5"; P2="+start_sample=6 num_output_samples=8"; BB3="+start_sample=6 num_output_samples=7"
BON="+num_candidates=4 +candidate_legality_floor=0.97 +legalize_all_candidates=true"
run() { local name=$1 seed=$2 want=$3; shift 3; local dir="logs/diffusion_debug/ispd2005-s0.$name.$seed"
  if [ -f "$dir/metrics.csv" ] && [ "$(($(wc -l < "$dir/metrics.csv")-1))" -ge "$want" ]; then echo "$(date '+%m-%d %H:%M') $name skip (done)"; return; fi
  PYTHONPATH=. "$PY" diffusion/eval.py method="$name" seed="$seed" "$@" > "logs/eval_logs/$name.log" 2>&1
  echo "$(date '+%m-%d %H:%M') $name exit=$? $(grep -c 'Finished sample' "logs/eval_logs/$name.log")/$want samples"; }
echo "$(date '+%m-%d %H:%M') ## Phase 4X Run X seeds"
for s in 301 302; do run runX_cu128_s${s}_part1 $s 5 $RUNX $OPT $LEG $P1; run runX_cu128_s${s}_bb3 $s 1 $RUNX $OPT $LEG $BB3; done
echo "$(date '+%m-%d %H:%M') ## Phase 4C DataAug Run C"
run runC_cu128_s300_part1 300 5 $RUNC $OPT $LEG $P1; run runC_cu128_s300_part2 300 2 $RUNC $OPT $LEG $P2
echo "$(date '+%m-%d %H:%M') ## Phase 4B baseline + best-of-4 seeds 303-305"
for s in 303 304 305; do
  run base_cu128_s${s}_part1 $s 5 $DDPM $OPT $LEG $P1; run base_cu128_s${s}_bb3 $s 1 $DDPM $OPT $LEG $BB3
  run bon2c_s${s}_part1 $s 5 $DDPM $OPT $LEG $P1 $BON;  run bon2c_s${s}_bb3 $s 1 $DDPM $OPT $LEG $BB3 $BON
done
echo "$(date '+%m-%d %H:%M') ## Phase 4 DONE"
