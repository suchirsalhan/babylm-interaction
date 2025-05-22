import yaml
from typing import Dict, Any

class RewardModelConfig:
    """
    Configuration class for the reward model.
    
    Attributes:
        model_name: Name of the base model
        output_dir: Directory to save outputs
        num_epochs: Number of training epochs
        batch_size: Training batch size
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay for optimization
        logging_dir: Directory for logging
        logging_steps: Number of steps between logging
        save_steps: Number of steps between saving checkpoints
    """
    def __init__(self, config_dict: Dict[str, Any]):
        self.model_name = config_dict.get('model_name', 'bert-base-uncased')
        self.output_dir = config_dict.get('output_dir', 'outputs')
        self.num_epochs = config_dict.get('num_epochs', 3)
        self.batch_size = config_dict.get('batch_size', 32)
        self.warmup_steps = config_dict.get('warmup_steps', 500)
        self.weight_decay = config_dict.get('weight_decay', 0.01)
        self.logging_dir = config_dict.get('logging_dir', 'logs')
        self.logging_steps = config_dict.get('logging_steps', 100)
        self.save_steps = config_dict.get('save_steps', 1000)
        
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'RewardModelConfig':
        """Create a config from a YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model_name': self.model_name,
            'output_dir': self.output_dir,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'warmup_steps': self.warmup_steps,
            'weight_decay': self.weight_decay,
            'logging_dir': self.logging_dir,
            'logging_steps': self.logging_steps,
            'save_steps': self.save_steps
        }
        
    def save_yaml(self, yaml_path: str):
        """Save config to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f) 