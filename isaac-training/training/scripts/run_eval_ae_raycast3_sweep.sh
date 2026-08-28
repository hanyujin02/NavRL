#!/bin/bash
set -u
cd /home/hanyujin/ReachMap/NavRL/isaac-training
LOG_DIR=/home/hanyujin/ReachMap/evaluation/results/eval_noise_logs
mkdir -p "$LOG_DIR"
N_TRIALS=3
RUNS=(
  "run-20260826_114419-kck0wipm|4000|depth_cnn_pretrain(ae+raycast)_attitude_depth_seed0_nb|0"
  "run-20260826_211849-omgbc2xl|34000|depth_cnn_pretrain(ae+raycast)_attitude_depth_seed1_nb|1"
  "run-20260819_075525-5ktf524d|38000|depth_cnn_pretrain(ae+raycast)_attitude_depth_seed2_nb|2"
)
FAILED=()
for idx in "${!RUNS[@]}"; do
  IFS='|' read -r run_dir step wandb_name seed <<< "${RUNS[$idx]}"
  ckpt="wandb/${run_dir}/files/checkpoint_${step}.pt"
  if [ ! -f "$ckpt" ]; then
    echo "[sweep] ERROR: checkpoint not found: ${ckpt}"
    FAILED+=("${run_dir}:missing_ckpt")
    continue
  fi
  log_tag=$(echo "$wandb_name" | tr '/ ' '__')
  log_file="${LOG_DIR}/${log_tag}.log"
  echo "[noise $((idx+1))/3] run=${run_dir} step=${step} seed=${seed} wandb.name=${wandb_name}"
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
    echo "[sweep] FAILED (noise, exit ${status}): ${run_dir} -- see ${log_file}"
    FAILED+=("${run_dir}:noise")
  else
    echo "[sweep] done (noise): ${run_dir}"
  fi

  log_tag2=$(echo "${wandb_name}_oriented3d_obs200" | tr '/ ' '__')
  log_file2="${LOG_DIR}/${log_tag2}.log"
  echo "[oriented $((idx+1))/3] run=${run_dir} step=${step} seed=${seed} wandb.name=${wandb_name}_oriented3d_obs200"
  python training/scripts/eval.py --config-name eval_noise \
      checkpoint="\"${ckpt}\"" \
      algo.feature_extractor.encoder_type=cnn \
      algo.feature_extractor.img_h=60 \
      algo.feature_extractor.img_w=80 \
      sensor.depth_height=60 \
      sensor.depth_width=80 \
      seed="${seed}" \
      env.obstacle_mode=oriented \
      env.policy_3d=false \
      env.num_obstacles=200 \
      eval.run_noise=false \
      eval.include_clean=true \
      wandb.name="\"${wandb_name}_oriented3d_obs200\"" \
      > "$log_file2" 2>&1
  status2=$?
  if [ $status2 -ne 0 ]; then
    echo "[sweep] FAILED (oriented, exit ${status2}): ${run_dir} -- see ${log_file2}"
    FAILED+=("${run_dir}:oriented")
  else
    echo "[sweep] done (oriented): ${run_dir}"
  fi
done
echo
echo "Sweep complete. ${#FAILED[@]} failures."
if [ ${#FAILED[@]} -gt 0 ]; then
  printf '  FAILED: %s\n' "${FAILED[@]}"
fi
