#!/bin/bash
# Closed-loop noise-robustness eval sweep over 12 NavRL runs (run-20260725_160717 and later).
#
# For each run, uses the checkpoint with the highest RAW (non-EMA) logged
# eval/stats.reach_goal, found via the wandb API history (see
# /home/hanyujin/.claude/plans/help-me-run-eval-robust-wigderson.md for how these
# steps were derived). Runs eval.py --config-name eval_noise once per run,
# sequentially (single GPU), with the algo/sensor overrides needed because these
# runs used encoder_type=cnn + 60x80 depth images (not the scratch/64x64 hydra
# defaults). Each eval run's wandb.name is set to match the original training
# run's name for traceability.

set -u
cd /home/hanyujin/ReachMap/NavRL/isaac-training

LOG_DIR=/home/hanyujin/ReachMap/evaluation/results/eval_noise_logs
mkdir -p "$LOG_DIR"

# Repeat each noisy condition this many times (independent noise draws, same
# scenario per checkpoint — see eval.py's n_trials handling) to get mean/std
# instead of a single noisy point estimate. clean is unaffected (no
# randomness) and always runs once regardless.
N_TRIALS=3

# run_dir | checkpoint_step | original wandb name (== run_dir suffix minus timestamp) | training seed
# seed matters: the "eval" scenario is a fixed, deterministic obstacle course
# derived from cfg.seed (see env.set_seed() in utils.py:evaluate()), and is
# NOT comparable across seeds (see project memory: navrl-eval-scenario-is-fixed).
# Each run must be re-evaluated on its OWN training seed's scenario, or the
# policy is judged on an unfamiliar course it was never selected against.
RUNS=(
  "run-20260725_160717-depth_cnn_pretrain(ae)_attitude_depth_seed1_bnfrozen|9000|depth_cnn_pretrain(ae)_attitude_depth_seed1_bnfrozen|1"
  "run-20260726_022239-depth_cnn_pretrain(reach)_attitude_depth_seed1_bnfrozen|11000|depth_cnn_pretrain(reach)_attitude_depth_seed1_bnfrozen|1"
  "run-20260726_184527-depth_cnn_scratch_attitude_depth_seed1|3000|depth_cnn_scratch_attitude_depth_seed1|1"
  "run-20260731_020704-depth_cnn_pretrain(reach_ae)_attitude_depth_seed1_nb|22000|depth_cnn_pretrain(reach_ae)_attitude_depth_seed1_nb|1"
  "run-20260806_230332-depth_cnn_pretrain(reach_ae)_attitude_depth_seed0_nb|32000|depth_cnn_pretrain(reach_ae)_attitude_depth_seed0_nb|0"
  "run-20260807_061558-depth_cnn_pretrain(ae)_attitude_depth_seed0_nb|5000|depth_cnn_pretrain(ae)_attitude_depth_seed0_nb|0"
  "run-20260807_152322-depth_cnn_pretrain(reach)_attitude_depth_seed0_nb|39000|depth_cnn_pretrain(reach)_attitude_depth_seed0_nb|0"
  "run-20260808_040517-depth_cnn_pretrain(reach)_attitude_depth_seed2_nb|29000|depth_cnn_pretrain(reach)_attitude_depth_seed2_nb|2"
  "run-20260808_162500-depth_cnn_pretrain(ae)_attitude_depth_seed2_nb|14000|depth_cnn_pretrain(ae)_attitude_depth_seed2_nb|2"
  "run-20260808_233210-depth_cnn_pretrain(reach-ae)_attitude_depth_seed2_nb|39000|depth_cnn_pretrain(reach-ae)_attitude_depth_seed2_nb|2"
  "run-20260809_160109-depth_cnn_scratch_attitude_depth_seed2_nb|30000|depth_cnn_scratch_attitude_depth_seed2_nb|2"
  "run-20260810_173932-depth_cnn_pretrain(defm)_attitude_depth_seed2_nb|16000|depth_cnn_pretrain(defm)_attitude_depth_seed2_nb|2"
)

N=${#RUNS[@]}
FAILED=()

for idx in "${!RUNS[@]}"; do
  IFS='|' read -r run_dir step wandb_name seed <<< "${RUNS[$idx]}"
  ckpt="wandb/${run_dir}/files/checkpoint_${step}.pt"
  log_tag=$(echo "$wandb_name" | tr '/ ' '__')
  log_file="${LOG_DIR}/${log_tag}.log"

  echo "=============================================================="
  echo "[$((idx+1))/${N}] run=${run_dir}  step=${step}  seed=${seed}  wandb.name=${wandb_name}"
  echo "  checkpoint: ${ckpt}"
  echo "  log: ${log_file}"
  echo "=============================================================="

  if [ ! -f "$ckpt" ]; then
    echo "[sweep] ERROR: checkpoint not found: ${ckpt}" | tee "$log_file"
    FAILED+=("$run_dir")
    continue
  fi

  # Values containing '(' ')' must be wrapped in escaped double-quotes so
  # Hydra's override grammar treats them as string literals, not grammar
  # tokens (see https://hydra.cc/docs/1.2/advanced/override_grammar/basic).
  python training/scripts/eval.py --config-name eval_noise \
      checkpoint="\"${ckpt}\"" \
      algo.feature_extractor.encoder_type=cnn \
      algo.feature_extractor.img_h=60 \
      algo.feature_extractor.img_w=80 \
      sensor.depth_height=60 \
      sensor.depth_width=80 \
      seed="${seed}" \
      eval.n_trials="${N_TRIALS}" \
      wandb.name="\"${wandb_name}\"" \
      > "$log_file" 2>&1
  status=$?

  if [ $status -ne 0 ]; then
    echo "[sweep] FAILED (exit ${status}): ${run_dir} -- see ${log_file}"
    FAILED+=("$run_dir")
  else
    echo "[sweep] done: ${run_dir}"
  fi
done

echo
echo "=============================================================="
echo "Sweep complete. ${#FAILED[@]} / ${N} failed."
if [ ${#FAILED[@]} -gt 0 ]; then
  printf '  FAILED: %s\n' "${FAILED[@]}"
fi
echo "=============================================================="
