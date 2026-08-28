#!/bin/bash
# One-off eval for the earliest "pretrain_attitude" checkpoint found
# (run-20260715_173408-depth_cnn_pretrain_attitude, wandb id ck4s1gqv),
# using an early/unlabeled cnn-pretrained encoder (depth-cnn_20260626_025327,
# checkpoint files since deleted from ReachMap/checkpoints/ — irrelevant for
# eval since the RL checkpoint already has the encoder weights baked in).
# Best checkpoint = argmax raw eval/stats.reach_goal from wandb history:
# step 16000, raw SR 0.937 (num_envs=350 training course — not directly
# comparable to the num_envs=200 eval_noise.yaml numbers below; this sweep
# gives the comparable one).
# Same methodology as every other sweep in this series: noise sweep +
# oriented3d_obs200 sweep, not added to the published artifact per request.
set -u
cd /home/hanyujin/ReachMap/NavRL/isaac-training
LOG_DIR=/home/hanyujin/ReachMap/evaluation/results/eval_noise_logs
mkdir -p "$LOG_DIR"

CKPT="wandb/run-20260715_173408-depth_cnn_pretrain_attitude/files/checkpoint_16000.pt"
WANDB_NAME="depth_cnn_pretrain_attitude_seed0"
SEED=0

echo "[noise] checkpoint=${CKPT}"
python training/scripts/eval.py --config-name eval_noise \
    checkpoint="\"${CKPT}\"" \
    algo.feature_extractor.encoder_type=cnn \
    algo.feature_extractor.img_h=60 \
    algo.feature_extractor.img_w=80 \
    sensor.depth_height=60 \
    sensor.depth_width=80 \
    seed="${SEED}" \
    eval.n_trials=3 \
    wandb.name="\"${WANDB_NAME}\"" \
    > "${LOG_DIR}/${WANDB_NAME}.log" 2>&1
echo "[noise] exit code $?"

echo "[oriented] checkpoint=${CKPT}"
python training/scripts/eval.py --config-name eval_noise \
    checkpoint="\"${CKPT}\"" \
    algo.feature_extractor.encoder_type=cnn \
    algo.feature_extractor.img_h=60 \
    algo.feature_extractor.img_w=80 \
    sensor.depth_height=60 \
    sensor.depth_width=80 \
    seed="${SEED}" \
    env.obstacle_mode=oriented \
    env.policy_3d=false \
    env.num_obstacles=200 \
    eval.run_noise=false \
    eval.include_clean=true \
    wandb.name="\"${WANDB_NAME}_oriented3d_obs200\"" \
    > "${LOG_DIR}/${WANDB_NAME}_oriented3d_obs200.log" 2>&1
echo "[oriented] exit code $?"
echo "pretrain_attitude eval complete."
