#!/bin/bash
# Companion to run_eval_oriented3d_obs200_sweep.sh: same 12 checkpoints,
# same num_obstacles=200 (down from the training-matched 300), clean-only,
# no noise — but env.obstacle_mode=hf (the FLAT, axis-aligned, training-
# distribution obstacle geometry), not oriented. Isolates whether the
# success-rate recovery seen at num_obstacles=200 in the oriented sweep is
# specific to tilted-obstacle OOD, or whether these checkpoints are simply
# sensitive to obstacle density in general (in which case flat/200 should
# also beat flat/300, i.e. the training-distribution clean baseline already
# on file in zerofill_original.json).

set -u
cd /home/hanyujin/ReachMap/NavRL/isaac-training

LOG_DIR=/home/hanyujin/ReachMap/evaluation/results/eval_noise_logs
mkdir -p "$LOG_DIR"

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
  log_tag=$(echo "${wandb_name}_flat_obs200" | tr '/ ' '__')
  log_file="${LOG_DIR}/${log_tag}.log"

  echo "=============================================================="
  echo "[$((idx+1))/${N}] run=${run_dir}  step=${step}  seed=${seed}  wandb.name=${wandb_name}_flat_obs200"
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
      env.obstacle_mode=hf \
      env.policy_3d=false \
      env.num_obstacles=200 \
      eval.run_noise=false \
      eval.include_clean=true \
      wandb.name="\"${wandb_name}_flat_obs200\"" \
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
