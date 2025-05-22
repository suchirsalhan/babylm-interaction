# Custom PPO Trainer Implementation

This repository contains a custom implementation of PPO (Proximal Policy Optimization) trainer for language model training, with specific modifications for teacher-student knowledge distillation.

## Custom Components

### Core Components Location
- Custom trainer implementations are located in `TnD_reformed/core/trainers/`
- Main PPO trainer: `TnD_reformed/core/trainers/ppo_trainer/ppo.py`
- Reward trainer: `TnD_reformed/core/trainers/reward_trainer/reward.py`

### Customization Setup
The `CustomPPOTrainer` class uses `_setup` functions to configure different components based on the configuration parameters. These setups are defined in the trainer class and can be customized through the `CustomPPOConfig`:

#### Example

1. **Optimizer Setup** (`_setup_optimizer`):
   - Configures the optimizer based on `config.optimizer_name`
   - Default: AdamW optimizer
   - Can be customized through `config.optimizer_kwargs`

2. **Scheduler Setup** (`_setup_scheduler`):
   - Configures learning rate scheduling
   - Options: linear, cosine, constant
   - Controlled by `config.scheduler_name` and `config.scheduler_kwargs`

3. **Early Stopping Setup** (`_setup_early_stopping`):
   - Configures early stopping mechanism
   - Enabled by `config.early_stopping`
   - Uses `config.target_kl` for KL divergence threshold

4. **Score Processing Setup** (`_setup_score_processing`):
   - Configures reward score processing
   - Options: scaling, normalization, clipping
   - Controlled by `config.use_score_scaling`, `config.use_score_norm`, `config.score_clip`

## Usage

### Basic Setup
```python
from TnD_reformed.core.trainers import CustomPPOTrainer, CustomPPOConfig

# Initialize configuration
config = CustomPPOConfig(
    learning_rate=1e-5,
    batch_size=8,
    mini_batch_size=4,
    early_stopping=True,
    target_kl=0.1,
    # ... other config parameters
)

# Initialize trainer
trainer = CustomPPOTrainer(
    teacher=teacher_model,
    student=student_model,
    reward_model=reward_model,
    config=config,
    train_dataset=train_dataset
)
```

## Configuration Options

### PPO Configuration
```python
config = CustomPPOConfig(
    custom_requirement_args
    **PPOConfig_default_args
)
```

## Model Requirements

### Teacher Model
- Must implement `get_teacher_logits` method
- Should provide logits for knowledge distillation
- Should be on the same device as the trainer

### Student Model
- Must implement `get_student_logits` method
- Should be trainable
- Should be on the same device as the trainer

### Reward Model
- A sequence classification model for now (from previous paper bert)
- Should output a single value per sequence
- Should be on the same device as the trainer
