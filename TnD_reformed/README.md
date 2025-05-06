# Custom PPO Trainer Implementation

This repository contains a custom implementation of PPO (Proximal Policy Optimization) trainer for language model training, with specific modifications for teacher-student knowledge distillation.

## Custom Components

### Core Components Location
- Custom trainer implementations are located in `TnD_reformed/core/trainers/`
- Main PPO trainer: `TnD_reformed/core/trainers/ppo_trainer/ppo.py`
- Reward trainer: `TnD_reformed/core/trainers/reward_trainer/reward.py`

### Customization Setup
The `CustomPPOTrainer` class uses `_setup` functions to configure different components based on the configuration parameters. These setups are defined in the trainer class and can be customized through the `CustomPPOConfig`:

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

### Training Process
1. The trainer uses a teacher model for knowledge distillation
2. A student model learns from the teacher's outputs
3. A reward model provides feedback for the PPO training
4. Early stopping prevents overfitting
5. Score processing ensures stable training

## Key Features

### 1. Knowledge Distillation
- Teacher-student architecture for efficient model training
- KL divergence-based distillation
- Configurable temperature for soft targets

### 2. Reward Processing
- Score scaling and normalization
- Configurable clipping
- Reward model integration

### 3. Training Stability
- Early stopping based on KL divergence
- Gradient clipping
- Learning rate scheduling
- Optimized CUDA memory usage

### 4. Monitoring
- Training statistics tracking
- KL divergence monitoring
- Reward statistics
- Loss tracking

## Configuration Options

### PPO Configuration
```python
config = CustomPPOConfig(
    learning_rate=1e-5,          # Learning rate
    batch_size=8,                # Batch size
    mini_batch_size=4,           # Mini-batch size
    gradient_accumulation_steps=1,# Gradient accumulation steps
    optimize_cuda_cache=True,     # Optimize CUDA memory
    early_stopping=True,         # Enable early stopping
    target_kl=0.1,               # Target KL divergence
    ppo_epochs=4,                # PPO epochs
    max_grad_norm=1.0,           # Maximum gradient norm
    init_kl_coef=0.2,            # Initial KL coefficient
    adap_kl_ctrl=True,           # Adaptive KL control
    use_score_scaling=True,      # Enable score scaling
    use_score_norm=True,         # Enable score normalization
    score_clip=None              # Score clipping value
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
- Must be a sequence classification model
- Should output a single value per sequence
- Should be on the same device as the trainer

## Best Practices

1. **Device Management**
   - Ensure all models are on the same device
   - Use `device` parameter consistently
   - Monitor device placement with debug prints

2. **Memory Management**
   - Use appropriate batch sizes
   - Enable CUDA cache optimization
   - Monitor memory usage

3. **Training Stability**
   - Start with conservative learning rates
   - Use early stopping
   - Monitor KL divergence
   - Use score normalization

4. **Debugging**
   - Enable `CUDA_LAUNCH_BLOCKING=1` for better error reporting
   - Monitor tensor shapes and devices
   - Check input validation
   - Use try-except blocks for better error handling

## Common Issues and Solutions

1. **CUDA Errors**
   - Check device placement
   - Verify tensor shapes
   - Monitor memory usage
   - Use appropriate batch sizes

2. **Training Instability**
   - Adjust learning rate
   - Enable score normalization
   - Use early stopping
   - Monitor KL divergence

3. **Memory Issues**
   - Reduce batch size
   - Enable gradient accumulation
   - Use CUDA cache optimization
   - Monitor memory usage

## Contributing

When adding new features or modifying existing ones:
1. Update the appropriate `_setup` function
2. Add configuration parameters to `CustomPPOConfig`
3. Update documentation
4. Add tests
5. Ensure backward compatibility 