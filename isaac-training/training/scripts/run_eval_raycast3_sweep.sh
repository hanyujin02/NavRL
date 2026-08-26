#!/bin/bash
# Closed-loop noise + oriented-OOD eval for the 3 new raycast-encoder RL runs
# (seed0/1/2, run_dir listed below), to extend the "NavRL Noise-Robustness
# Sweep" artifact with a `raycast` pretraining-source row group.
#
# Checkpoint selection: argmax of raw logged eval/stats.reach_goal per run
# (wandb API, samples=200000 — see help-me-run-eval-robust-wigderson.md for
# methodology), NOT files/best.pt (EMA-smoothed) — same rule as the existing
# 12-checkpoint sweep.
#   seed0 (run-20260825_200801-qta1mm52, finished normally): step 9000,  raw SR 0.729
#   seed1 (run-20260825_131152-foh0ppqq, killed ~step 33000): step 17000, raw SR 0.703
#   seed2 (run-20260824_170142-k8uxwopg, crashed ~step 33000): step 9000,  raw SR 0.757
#
# Two eval.py invocations per checkpoint, matching the existing sweep exactly:
#   1) eval_noise config defaults (gaussian/dropout/cutout/quantization x3
#      intensities + their maxfill variants, n_trials=3) -> run_eval_noise_sweep.sh
#   2) env.obstacle_mode=oriented, env.num_obstacles=200, eval.run_noise=false
#      -> run_eval_oriented3d_obs200_sweep.sh
# All architecture overrides (encoder_type=cnn, img_h=60/img_w=80, etc.) match
# because these runs used the same Option D block in train_depth.yaml as the
# original 12.

set -u
cd /home/hanyujin/ReachMap/NavRL/isaac-training

LOG_DIR=/home/hanyujin/ReachMap/evaluation/results/eval_noise_logs
mkdir -p "$LOG_DIR"

N_TRIALS=3

# run_dir | checkpoint_step | wandb name | training seed
RUNS=(
  "run-20260825_200801-qta1mm52|9000|depth_cnn_pretrain(raycast)_attitude_depth_seed0_nb|0"
  "run-20260825_131152-foh0ppqq|17000|depth_cnn_pretrain(raycast)_attitude_depth_seed1_nb|1"
  "run-20260824_170142-k8uxwopg|9000|depth_cnn_pretrain(raycast)_attitude_depth_seed2_nb|2"
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

  # ── 1) Noise sweep ──────────────────────────────────────────────────────
  log_tag=$(echo "$wandb_name" | tr '/ ' '__')
  log_file="${LOG_DIR}/${log_tag}.log"
  echo "=============================================================="
  echo "[noise $((idx+1))/3] run=${run_dir} step=${step} seed=${seed} wandb.name=${wandb_name}"
  echo "  checkpoint: ${ckpt}"
  echo "  log: ${log_file}"
  echo "=============================================================="

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

  # ── 2) Oriented OOD sweep (num_obstacles=200) ──────────────────────────
  log_tag2=$(echo "${wandb_name}_oriented3d_obs200" | tr '/ ' '__')
  log_file2="${LOG_DIR}/${log_tag2}.log"
  echo "=============================================================="
  echo "[oriented $((idx+1))/3] run=${run_dir} step=${step} seed=${seed} wandb.name=${wandb_name}_oriented3d_obs200"
  echo "  checkpoint: ${ckpt}"
  echo "  log: ${log_file2}"
  echo "=============================================================="

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
echo "=============================================================="
echo "Sweep complete. ${#FAILED[@]} failures."
if [ ${#FAILED[@]} -gt 0 ]; then
  printf '  FAILED: %s\n' "${FAILED[@]}"
fi
echo "=============================================================="
