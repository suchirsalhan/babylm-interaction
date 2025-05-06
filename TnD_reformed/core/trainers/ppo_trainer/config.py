from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from transformers import TrainingArguments
from trl import PPOConfig

@dataclass
class CustomPPOConfig(PPOConfig):
    """Configuration for CustomPPOTrainer."""
    
    # Model and training configuration
    teacher_model: Optional[str] = None
    student_model: Optional[str] = None
    reward_model: Optional[str] = None
    
    # Loss configuration
    loss_type: str = "kl_regularized"
    kl_coeff: float = 0.1
    entropy_coeff: float = 0.01
    lm_loss_coef: float = 0.0
    loss_scaling: float = 1.0
    ratio_threshold: float = 10.0
    vloss_only: bool = False
    
    # Reward configuration
    reward_type: str = "teacher_guided"
    reward_temperature: float = 1.0
    teacher_weight: float = 0.5
    length_reward_coef: Optional[float] = None
    score_clip: Optional[float] = None
    
    # Generation configuration
    output_min_length: int = 1
    output_max_length: int = 128
    generation_top_p: float = 1.0
    generation_top_k: int = 0
    generation_temperature: float = 1.0
    generation_do_sample: bool = True
    generation_num_beams: int = 1
    generation_num_beam_groups: int = 1
    
    # Query configuration
    query_min_length: int = 1
    query_max_length: int = 2
    
    # Training configuration
    batch_size: int = 1024
    mini_batch_size: int = 512
    gradient_scaling_type: str = "dynamic"
    max_grad_norm: float = 1.0
    min_grad_scale: float = 0.1
    patience_steps: int = 10
    
    # Evaluation configuration
    eval_freq: int = 100
    log_freq: int = 10
    metrics: List[str] = field(default_factory=lambda: ["kl_divergence", "entropy", "reward_stats"])
    
    # Logging configuration
    log_with: str = "wandb"
    project_name: str = "lm_feedback_ppo"
    run_name: Optional[str] = None
    
    # Accelerator configuration
    accelerator_kwargs: Dict[str, Any] = field(default_factory=lambda: {"mixed_precision": "bf16"})
    
    # Additional PPO parameters
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    ppo_epochs: int = 4
    backward_batch_size: int = 16
    early_stopping: bool = False
    compare_steps: int = 1 