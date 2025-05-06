import yaml
from typing import Dict, Any

class TnDConfig:
    """
    Configuration class for TnD experiments.
    
    Attributes:
        teacher_model: Name of the teacher model
        student_model: Name of the student model
        reward_model: Name of the reward model
        dataset_path: Path to the dataset
        output_dir: Directory to save outputs
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimization
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay for optimization
        max_length: Maximum sequence length
        gradient_accumulation_steps: Number of gradient accumulation steps
        entropy_coeff: Coefficient for entropy regularization
        loss_scaling: Scaling factor for the loss
    """
    def __init__(self, config_dict: Dict[str, Any]):
        self.teacher_model = config_dict.get('teacher_model', 'gpt2')
        self.student_model = config_dict.get('student_model', 'gpt2')
        self.reward_model = config_dict.get('reward_model', 'bert-base-uncased')
        self.dataset_path = config_dict.get('dataset_path', 'data/tnd')
        self.output_dir = config_dict.get('output_dir', 'outputs/tnd')
        self.num_epochs = config_dict.get('num_epochs', 3)
        self.batch_size = config_dict.get('batch_size', 32)
        self.learning_rate = config_dict.get('learning_rate', 5e-5)
        self.warmup_steps = config_dict.get('warmup_steps', 500)
        self.weight_decay = config_dict.get('weight_decay', 0.01)
        self.max_length = config_dict.get('max_length', 512)
        self.gradient_accumulation_steps = config_dict.get('gradient_accumulation_steps', 4)
        self.entropy_coeff = config_dict.get('entropy_coeff', 0.1)
        self.loss_scaling = config_dict.get('loss_scaling', 1.0)
        
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TnDConfig':
        """Create a config from a YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'teacher_model': self.teacher_model,
            'student_model': self.student_model,
            'reward_model': self.reward_model,
            'dataset_path': self.dataset_path,
            'output_dir': self.output_dir,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'warmup_steps': self.warmup_steps,
            'weight_decay': self.weight_decay,
            'max_length': self.max_length,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'entropy_coeff': self.entropy_coeff,
            'loss_scaling': self.loss_scaling
        }
        
    def save_yaml(self, yaml_path: str):
        """Save config to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f) 