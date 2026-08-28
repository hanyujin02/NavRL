#!/bin/bash
# One-off eval for the "ae32" pretrained-encoder RL checkpoint
# (run-20260815_053331-s8odba9q, depth_cnn_pretrain(ae32)_attitude_depth_seed2).
# Only one seed found for this encoder (seed2) — analogous to the
# pretrain_attitude one-off eval, not a 3-seed sweep.
# Best checkpoint = argmax raw eval/stats.reach_goal from wandb history:
# step 17000, raw SR 0.9286.
# Same methodology as every other sweep in this series: noise sweep +
# oriented3d_obs200 sweep, not added to the published artifact per request.
set -u
cd /home/hanyujin/ReachMap/NavRL/isaac-training
LOG_DIR=/home/hanyujin/ReachMap/evaluation/results/eval_noise_logs
mkdir -p "$LOG_DIR"

CKPT="wandb/run-20260815_053331-s8odba9q/files/checkpoint_17000.pt"
WANDB_NAME="depth_cnn_pretrain_ae32_seed2"
SEED=2

echo "[noise] checkpoint=${CKPT}"
python training/scripts/eval.py --config-name eval_noise \
    checkpoint="\"${CKPT}\"" \
    algo.feature_extractor.encoder_type=cnn \
    algo.feature_extractor.embed_dim=32 \
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
    algo.feature_extractor.embed_dim=32 \
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
echo "ae32 eval complete."
