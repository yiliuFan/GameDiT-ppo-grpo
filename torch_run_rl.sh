#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

###################################
# User Configuration Section
###################################
RUN_PYTHON_PATH="/home/yiliu/miniconda3_linux/envs/diffusion_planner/bin/python3.9"
TRAIN_SET_PATH="/home/yiliu/train"
TRAIN_SET_LIST_PATH="/home/yiliu/gameformer_learning/train_set_list.json"
# Point this to the supervised warm-start checkpoint you want to fine-tune.
PRETRAINED_MODEL_PATH="/home/yiliu/gameformer_learning/Diffusion-Planner-main-2/latest.pth"
SAVE_DIR="./training_log"
RUN_NAME="diffusion-planner-rft-ppo-grpo"
###################################

$RUN_PYTHON_PATH train_rl_finetune.py \
  --ddp False \
  --device cuda \
  --save_dir "$SAVE_DIR" \
  --name "$RUN_NAME" \
  --train_set "$TRAIN_SET_PATH" \
  --train_set_list "$TRAIN_SET_LIST_PATH" \
  --pretrained_model_path "$PRETRAINED_MODEL_PATH" \
  --train_epochs 100 \
  --save_utd 5 \
  --batch_size 16 \
  --num_workers 4 \
  --learning_rate 1e-5 \
  --rl_policy_learning_rate 1e-4 \
  --rl_num_samples 4 \
  --rl_ddim_steps 5 \
  --rl_ppo_epochs 4 \
  --rl_ppo_clip 0.2 \
  --rl_value_weight 0.5 \
  --rl_value_clip 0.2 \
  --rl_grpo_epochs 2 \
  --rl_grpo_clip 0.2 \
  --rl_reference_kl_weight 0.01 \
  --rl_bc_weight 1.0 \
  --rl_trajectory_weight 0.5 \
  --rl_policy_weight 0.2 \
  --rl_entropy_weight 1e-3 \
  --rl_finetune_decoder_only True
