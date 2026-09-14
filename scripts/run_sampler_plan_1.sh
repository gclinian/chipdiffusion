#!/usr/bin/env bash
# Sequential GPU queue for docs/plan/sampler_plan_1.md. Run as `bash scripts/run_sampler_plan_1.sh [pid ...]`.
# - waits (by PID, never by pattern) for any given PIDs to exit before starting
# - skips any cell whose metrics.csv already exists with the expected sample count
# - every cell is independent: a failure logs exit!=0 and the queue continues
set -u
R=/ibmnas/427/r115/gclin/Desktop/chipdiffusion
PY=/ibmnas/427/r115/gclin/miniforge3/envs/chipdiff-b/bin/python
cd "$R"; mkdir -p logs/eval_logs
LEG="task=ispd2005-s0 legalizer@_global_=opt-adam +skip_guidance_threshold=10000 legalization.alpha_lr=8e-3 legalization.hpwl_weight=12e-5 legalization.legality_potential_target=0 legalization.grad_descent_steps=20000 macros_only=True logger.wandb=false"
GD="model.grad_descent_steps=20 model.hpwl_guidance_weight=16e-4"
OPT="guidance@_global_=opt $GD"; SVDD="guidance@_global_=svdd_layered $GD"; NONE="guidance@_global_=none"
DDPM="from_checkpoint=../public-models/large-v2/large-v2.ckpt"
FT="from_checkpoint=v1.61-ddpo.ablation_supervised_10k.61/latest.ckpt"
FM="family=flow_matching from_checkpoint=v1.61-fs.61.fm_p1_500k.61/latest.ckpt model.max_diffusion_steps=50"
A1="+start_sample=0 num_output_samples=1"; BB4="+start_sample=7 num_output_samples=8"
P1="+start_sample=0 num_output_samples=5"; P2="+start_sample=6 num_output_samples=8"; BB3="+start_sample=6 num_output_samples=7"
BON="+num_candidates=4 +candidate_legality_floor=0.97"

for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 15; done; done
echo "$(date '+%m-%d %H:%M') queue start"

# run NAME SEED EXPECTED_SAMPLES overrides...
run() {
  local name=$1 seed=$2 want=$3; shift 3
  local dir="logs/diffusion_debug/ispd2005-s0.$name.$seed"
  if [ -f "$dir/metrics.csv" ] && [ "$(($(wc -l < "$dir/metrics.csv")-1))" -ge "$want" ]; then
    echo "$(date '+%m-%d %H:%M') $name skip (done)"; return; fi
  PYTHONPATH=. "$PY" diffusion/eval.py method="$name" seed="$seed" "$@" > "logs/eval_logs/$name.log" 2>&1
  echo "$(date '+%m-%d %H:%M') $name exit=$? $(grep -c 'Finished sample' "logs/eval_logs/$name.log")/$want samples"
}

echo "## Phase 1A unguided control"
run diag1A_ddpm_none_a1  300 1 $DDPM $NONE $LEG $A1
run diag1A_fm_none_a1    300 1 $FM   $NONE $LEG $A1
run diag1A_ddpm_none_bb4 300 1 $DDPM $NONE $LEG $BB4
run diag1A_fm_none_bb4   300 1 $FM   $NONE $LEG $BB4
echo "## Phase 0c baseline seeds"
for s in 301 302; do run base_cu128_s${s}_part1 $s 5 $DDPM $OPT $LEG $P1; run base_cu128_s${s}_bb3 $s 1 $DDPM $OPT $LEG $BB3; done
echo "## Phase 0 re-anchor (gate tripped)"
for s in 301 302; do run runF_cu128_s${s}_part1 $s 5 $FT $OPT $LEG $P1; run runF_cu128_s${s}_bb3 $s 1 $FT $OPT $LEG $BB3; done
run svdd_cu128_s300_part1 300 5 $DDPM $SVDD $LEG $P1; run svdd_cu128_s300_part2 300 2 $DDPM $SVDD $LEG $P2
for s in 301 302; do run svdd_cu128_s${s}_part1 $s 5 $DDPM $SVDD $LEG $P1; run svdd_cu128_s${s}_bb3 $s 1 $DDPM $SVDD $LEG $BB3; done
echo "## Phase 1B eta x T"
for T in 1000 100; do for eta in 0.0 0.5 1.0; do
  [ "$T" = 1000 ] && [ "$eta" = 1.0 ] && continue; tag="eta${eta/./}_T${T}"
  run diag1B_opt_${tag}_a1  300 1 $DDPM $OPT $LEG $A1  +model.eta_scale=$eta model.max_diffusion_steps=$T
  run diag1B_opt_${tag}_bb4 300 1 $DDPM $OPT $LEG $BB4 +model.eta_scale=$eta model.max_diffusion_steps=$T
done; done
run diag1B_none_eta00_T1000_a1  300 1 $DDPM $NONE $LEG $A1  +model.eta_scale=0.0
run diag1B_none_eta00_T1000_bb4 300 1 $DDPM $NONE $LEG $BB4 +model.eta_scale=0.0
echo "## Phase 1C t_shift"
run diag1C_tshift_up_part1  300 5 $DDPM $OPT $LEG $P1 +model.t_shift_by_size=true
run diag1C_tshift_up_part2  300 2 $DDPM $OPT $LEG $P2 +model.t_shift_by_size=true
run diag1C_tshift_inv_part1 300 5 $DDPM $OPT $LEG $P1 +model.t_shift_by_size=true +model.t_shift_invert=true
run diag1C_tshift_inv_part2 300 2 $DDPM $OPT $LEG $P2 +model.t_shift_by_size=true +model.t_shift_invert=true
echo "## Phase 2 best-of-N"
run bon2a_legall_a1  300 1 $DDPM $OPT $LEG $A1  $BON +legalize_all_candidates=true
run bon2a_legall_bb4 300 1 $DDPM $OPT $LEG $BB4 $BON +legalize_all_candidates=true
for s in 300 301 302; do run bon2b_s${s}_part1 $s 5 $DDPM $OPT $LEG $P1 $BON; run bon2b_s${s}_bb3 $s 1 $DDPM $OPT $LEG $BB3 $BON; done
run bon2b_s300_bb4 300 1 $DDPM $OPT $LEG $BB4 $BON
echo "$(date '+%m-%d %H:%M') ALL PHASES DONE"
