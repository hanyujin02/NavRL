#!/bin/bash
# Waits for the currently running ae32 train_depth.yaml job (PID given as $1)
# to exit, then sequentially trains the 4 newly-added encoder checkpoints
# (ae64, reach64, ae128, reach128), each a straight rerun of train_depth.yaml
# with only embed_dim / encoder_ckpt / wandb.name overridden — everything
# else (seed=2, num_envs=350, max_frame_num=12e8, freeze_encoder=true,
# projection_mlp=true, img_h/w=60/80, ...) stays exactly as in the active
# "Option D" block, matching how ae32 itself is being run right now.
#
# Single GPU -> strictly sequential. Continues to the next run even if one
# fails (logs the failure, doesn't block the rest of the queue).

set -u
cd /home/hanyujin/ReachMap/NavRL/isaac-training

WAIT_PID="$1"
LOG_DIR=/home/hanyujin/ReachMap/evaluation/results/train_depth_queue_logs
mkdir -p "$LOG_DIR"

echo "[queue] waiting for PID ${WAIT_PID} (ae32 run) to exit..."
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 30
done
echo "[queue] PID ${WAIT_PID} has exited. Starting queued runs."

# tag | embed_dim
RUNS=(
  "ae64|64"
  "reach64|64"
  "ae128|128"
  "reach128|128"
)

N=${#RUNS[@]}
FAILED=()

for idx in "${!RUNS[@]}"; do
  IFS='|' read -r tag embed_dim <<< "${RUNS[$idx]}"
  ckpt="/home/hanyujin/ReachMap/checkpoints/depth-cnn-${tag}/latest.pt"
  run_name="depth_cnn_pretrain(${tag})_attitude_depth_seed2_nb"
  log_file="${LOG_DIR}/${tag}.log"

  echo "=============================================================="
  echo "[$((idx+1))/${N}] tag=${tag}  embed_dim=${embed_dim}  ckpt=${ckpt}"
  echo "  wandb.name: ${run_name}"
  echo "  log: ${log_file}"
  echo "=============================================================="

  if [ ! -f "$ckpt" ]; then
    echo "[queue] ERROR: checkpoint not found: ${ckpt}" | tee "$log_file"
    FAILED+=("$tag")
    continue
  fi

  python training/scripts/train.py --config-name=train_depth \
      algo.feature_extractor.embed_dim="${embed_dim}" \
      algo.feature_extractor.encoder_ckpt="\"${ckpt}\"" \
      wandb.name="\"${run_name}\"" \
      > "$log_file" 2>&1
  status=$?

  if [ $status -ne 0 ]; then
    echo "[queue] FAILED (exit ${status}): ${tag} -- see ${log_file}"
    FAILED+=("$tag")
  else
    echo "[queue] done: ${tag}"
  fi
done

echo
echo "=============================================================="
echo "Queue complete. ${#FAILED[@]} / ${N} failed."
if [ ${#FAILED[@]} -gt 0 ]; then
  printf '  FAILED: %s\n' "${FAILED[@]}"
fi
echo "=============================================================="
