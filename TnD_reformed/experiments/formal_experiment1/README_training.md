# Dialogue PPO Training with TAACO Rewards

This experiment implements a child-teacher learning framework using PPO (Proximal Policy Optimization) with TAACO (Tool for the Automatic Analysis of Cohesion) rewards for dialogue completion tasks.

## Overview

The training setup includes:
- **Child Model**: OPT-based model (Talking-Babies/opt-Talking-Babies-train_100M_2048_preprocess)
- **Teacher Model**: Llama-3.2-1B-Instruct  
- **Reward Model**: TAACO-based linguistic quality assessment
- **Dataset**: Switchboard dialogue conversations with TAACO features

## Files

- `train_dialogue_ppo_taaco.py` - Main training script
- `run_training.sh` - Shell script to start training with default parameters
- `calculate_steps.py` - Calculate training steps and checkpoints
- `README_training.md` - This documentation

## Requirements

### Python Environment
```bash
# Required packages (install in your environment)
pip install torch transformers trl datasets pandas numpy tqdm wandb
```

### Models
- Child model: `Talking-Babies/opt-Talking-Babies-train_100M_2048_preprocess`
- Child tokenizer: `Talking-Babies/opt-tokenizer`
- Teacher model: `meta-llama/Llama-3.2-1B-Instruct`

### Hardware Requirements
- **GPU Memory**: ~6GB VRAM recommended
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ for models and checkpoints

## Training Configuration

### Dataset
- **Source**: `data/dialog_level_taaco_results_filtered.csv`
- **Size**: 17,694 filtered dialogues
- **Filters**: 20-400 words per dialogue

### Training Parameters
- **Batch Size**: 8 (adjustable)
- **Learning Rate**: 1.41e-5
- **Epochs**: 1
- **Total Steps**: 2,211
- **Checkpoint Interval**: Every 221 steps (10%)

### Expected Training Time
- **Per Step**: ~3 seconds
- **Total Time**: ~1.8 hours
- **Checkpoints**: 10 saves during training

## Usage

### Testing Setup (Recommended)
Before starting training, test that all components work correctly:

```bash
# Test the complete setup
python test_training_setup.py
```

This will verify:
- Model loading (child and teacher)
- Interaction model functionality
- TAACO reward computation
- Tensor format compatibility
- Error handling
- Wandb logging capability

### Wandb Setup (For Web Logging)
To view training logs on the web, set up Weights & Biases:

```bash
# Option 1: Use the setup script
./setup_wandb.sh

# Option 2: Manual setup
pip install wandb
wandb login  # Follow prompts to get API key from https://wandb.ai/authorize
```

After setup, your training logs will be visible at:
`https://wandb.ai/[your-username]/dialogue-ppo-taaco-experiment`

### Quick Start
```bash
# 1. Test setup (recommended)
python test_training_setup.py

# 2. Calculate training steps
python calculate_steps.py

# 3. Start training with default parameters
./run_training.sh
```

### Custom Training
```bash
# Train with custom batch size
python train_dialogue_ppo_taaco.py --batch_size 16

# Train with custom learning rate
python train_dialogue_ppo_taaco.py --learning_rate 2e-5

# Train for multiple epochs
python train_dialogue_ppo_taaco.py --max_epochs 2
```

### Available Arguments
```bash
python train_dialogue_ppo_taaco.py \
    --data_path "data/dialog_level_taaco_results_filtered.csv" \
    --batch_size 8 \
    --learning_rate 1.41e-5 \
    --save_dir "checkpoints" \
    --max_epochs 1 \
    --device "cuda" \
    --wandb_project "dialogue-ppo-taaco"
```

## Training Process

### 1. Data Loading
- Loads dialogue data from CSV
- Filters by length (20-400 words)
- Creates HuggingFace Dataset format

### 2. Model Setup
- Child model: OPT-100M for learning
- Teacher model: Llama-1B for guidance
- TAACO processor for reward calculation

### 3. PPO Training Loop
- Child generates dialogue continuations
- Teacher provides high-quality completions
- TAACO scores linguistic quality
- PPO optimizes child model based on rewards

### 4. Checkpointing
- Saves every 10% of training (221 steps)
- Includes model weights and training stats
- Located in `checkpoints_dialogue_ppo/`

## Checkpoint Structure

```
checkpoints_dialogue_ppo/
├── checkpoint-step-221/
│   ├── child_model/           # Saved child model
│   └── training_info.json     # Training metadata
├── checkpoint-step-442/
├── ...
└── training_log.json          # Complete training log
```

## TAACO Reward System

The TAACO reward model evaluates:
- **Lexical Diversity**: TTR, content words
- **Coherence**: LSA, LDA, Word2Vec metrics  
- **Cohesion**: Adjacent overlap, connectives
- **Discourse**: Givenness, pronoun usage

## Monitoring

### Weights & Biases
- Project: `dialogue-ppo-taaco-experiment`
- Tracks PPO metrics, rewards, losses
- Model performance over time

### Local Logs
- Training statistics saved in checkpoints
- Error count and recovery tracking
- Step-by-step progress monitoring

## Troubleshooting

### Common Issues

**Out of Memory**
```bash
# Reduce batch size
python train_dialogue_ppo_taaco.py --batch_size 4
```

**TAACO Errors**
- Ensure TAACO dependencies are installed
- Check LSA/LDA model files are accessible

**Model Loading Issues**
- Verify internet connection for model downloads
- Check HuggingFace token if using gated models

### Recovery from Interruption
The training automatically saves checkpoints and can be resumed by loading the latest checkpoint manually.

## Results

Training produces:
1. **Improved Child Model**: Better dialogue completion abilities
2. **Training Metrics**: PPO stats, reward progression
3. **Checkpoints**: Model states at 10% intervals
4. **Analysis Data**: TAACO scores and linguistic improvements

## Next Steps

After training completion:
1. Evaluate model on test set
2. Compare TAACO scores before/after training  
3. Analyze dialogue quality improvements
4. Fine-tune hyperparameters for better results 