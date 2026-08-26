#!/bin/bash
# Max-range-fill supplement to run_eval_noise_sweep.sh: same 12 NavRL runs and
# checkpoints, but only the new "*_maxfill" dropout/cutout conditions
# (fill_value=1.0, matching this codebase's own sensor pipeline convention
# for a lost/no-return ray — see env_depth.py's nan_to_num(nan=depth_range)
# and eval_noise.yaml's "max-range fill variants" comment). The already-
# published zero-fill dropout/cutout results are NOT re-run here
# (eval.include_zerofill=false); gaussian/quantization are unaffected by the
# fill-value axis and DO re-run (cheap, and useful as a same-run sanity
# check against the original numbers). n_trials=1 (not 3): this is a
# supplementary comparison, not a from-scratch variance study.
#
# wandb.name gets a "_maxfill" suffix so these land as clearly-distinguished
# sibling runs in project NavRL-eval next to the original zero-fill runs
# (each eval.py invocation is its own wandb run; nothing is appended).

set -u
cd /home/hanyujin/ReachMap/NavRL/isaac-training

LOG_DIR=/home/hanyujin/ReachMap/evaluation/results/eval_noise_logs
mkdir -p "$LOG_DIR"

N_TRIALS=1

# run_dir | checkpoint_step | original wandb name | training seed
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
  log_tag=$(echo "${wandb_name}_maxfill" | tr '/ ' '__')
  log_file="${LOG_DIR}/${log_tag}.log"

  echo "=============================================================="
  echo "[$((idx+1))/${N}] run=${run_dir}  step=${step}  seed=${seed}  wandb.name=${wandb_name}_maxfill"
  echo "  checkpoint: ${ckpt}"
  echo "  log: ${log_file}"
  echo "=============================================================="

  if [ ! -f "$ckpt" ]; then
    echo "[sweep] ERROR: checkpoint not found: ${ckpt}" | tee "$log_file"
    FAILED+=("$run_dir")
    continue
  fi

  python training/scripts/eval.py --config-name eval_noise \
      checkpoint="\"${ckpt}\"" \
      algo.feature_extractor.encoder_type=cnn \
      algo.feature_extractor.img_h=60 \
      algo.feature_extractor.img_w=80 \
      sensor.depth_height=60 \
      sensor.depth_width=80 \
      seed="${seed}" \
      eval.include_zerofill=false \
      eval.n_trials="${N_TRIALS}" \
      wandb.name="\"${wandb_name}_maxfill\"" \
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
