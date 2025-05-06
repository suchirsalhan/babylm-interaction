import yaml
from typing import Dict, Any

class BabyLMConfig:
    """
    Configuration class for BabyLM experiments.
    
    Attributes:
        model_name: Name of the base model
        dataset_path: Path to the dataset
        output_dir: Directory to save outputs
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimization
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay for optimization
        max_length: Maximum sequence length
        gradient_accumulation_steps: Number of gradient accumulation steps
    """
    def __init__(self, config_dict: Dict[str, Any]):
        self.model_name = config_dict.get('model_name', 'gpt2')
        self.dataset_path = config_dict.get('dataset_path', 'data/babylm')
        self.output_dir = config_dict.get('output_dir', 'outputs/babylm')
        self.num_epochs = config_dict.get('num_epochs', 3)
        self.batch_size = config_dict.get('batch_size', 32)
        self.learning_rate = config_dict.get('learning_rate', 5e-5)
        self.warmup_steps = config_dict.get('warmup_steps', 500)
        self.weight_decay = config_dict.get('weight_decay', 0.01)
        self.max_length = config_dict.get('max_length', 512)
        self.gradient_accumulation_steps = config_dict.get('gradient_accumulation_steps', 4)
        
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BabyLMConfig':
        """Create a config from a YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model_name': self.model_name,
            'dataset_path': self.dataset_path,
            'output_dir': self.output_dir,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'warmup_steps': self.warmup_steps,
            'weight_decay': self.weight_decay,
            'max_length': self.max_length,
            'gradient_accumulation_steps': self.gradient_accumulation_steps
        }
        
    def save_yaml(self, yaml_path: str):
        """Save config to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f) 