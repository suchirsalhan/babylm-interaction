#!/bin/bash

# Training script for Dialogue PPO with TAACO rewards
# This script sets up and runs the training with appropriate parameters

echo "Starting Dialogue PPO Training with TAACO Rewards"
echo "=================================================="

# Configuration
DATA_PATH="data/dialog_level_taaco_results_filtered.csv"
BATCH_SIZE=8
LEARNING_RATE=1.41e-5
SAVE_DIR="checkpoints_dialogue_ppo"
MAX_EPOCHS=1
DEVICE="cuda"
WANDB_PROJECT="dialogue-ppo-taaco-experiment"

echo "Configuration:"
echo "  Data Path: $DATA_PATH"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Save Directory: $SAVE_DIR"
echo "  Max Epochs: $MAX_EPOCHS"
echo "  Device: $DEVICE"
echo "  W&B Project: $WANDB_PROJECT"
echo ""

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found at $DATA_PATH"
    echo "Please ensure the data file exists before running training."
    exit 1
fi

# Create save directory if it doesn't exist
mkdir -p $SAVE_DIR

# Activate conda environment (adjust as needed)
# source activate babylm_old_trl

# Run training
python train_dialogue_ppo_taaco.py \
    --data_path "$DATA_PATH" \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --save_dir "$SAVE_DIR" \
    --max_epochs $MAX_EPOCHS \
    --device "$DEVICE" \
    --wandb_project "$WANDB_PROJECT"

echo ""
echo "Training completed!"
echo "Checkpoints saved in: $SAVE_DIR" 