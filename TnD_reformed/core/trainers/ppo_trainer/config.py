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
    
    # child model generation config
    child_generation_args: Optional[Dict[str, Any]] = None
    # teacher model generation config
    teacher_generation_args: Optional[Dict[str, Any]] = None
    
    # custom interaction model
    custom_interaction_model: Optional[str] = None
    