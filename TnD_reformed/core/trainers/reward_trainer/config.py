from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from transformers import TrainingArguments
from trl import RewardConfig

@dataclass
class CustomRewardConfig(RewardConfig):
    """Extended reward configuration with additional parameters."""
    
    # Model parameters
    model_type: str = field(
        default="regression",
        metadata={"help": "Type of reward model (regression, classification)"}
    )
    
    # Loss parameters
    loss_type: str = field(
        default="mse",
        metadata={"help": "Type of loss function to use (mse, cross_entropy)"}
    )
    
    # Training parameters
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "Learning rate"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay"}
    )
    warmup_steps: int = field(
        default=500,
        metadata={"help": "Number of warmup steps"}
    )
    
    # Evaluation parameters
    eval_steps: int = field(
        default=1000,
        metadata={"help": "Number of steps between evaluations"}
    )
    save_steps: int = field(
        default=1000,
        metadata={"help": "Number of steps between checkpoints"}
    )
    
    # Custom metrics
    metrics: List[str] = field(
        default_factory=lambda: ["loss", "accuracy", "f1"],
        metadata={"help": "Additional metrics to track during training"}
    )
    
    # Reward model specific parameters
    reward_model_name: str = field(
        default="gpt2",
        metadata={"help": "Name of the reward model to use"}
    )
    reward_model_type: str = field(
        default="value",
        metadata={"help": "Type of reward model (value, policy)"}
    )
    reward_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained reward model"}
    )
    reward_model_config: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "Configuration for the reward model"}
    ) 