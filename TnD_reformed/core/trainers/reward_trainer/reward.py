from typing import Any, Dict, Optional, Union, Tuple
import torch
from torch import nn
from trl import RewardTrainer as TRLRewardTrainer
from ...models import RewardModel
from .custom_components import CustomLossFunctions, CustomMetrics
from .config import CustomRewardConfig

class CustomRewardTrainer(TRLRewardTrainer):
    """
    Custom reward trainer that extends TRL's RewardTrainer with additional functionality.
    
    Attributes:
        model: Reward model
        config: Custom reward configuration
        device: Device to train on
    """
    def __init__(
        self,
        model: RewardModel,
        config: CustomRewardConfig,
        device: Optional[torch.device] = None
    ):
        super().__init__(model=model, args=config)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set up loss function based on config
        if config.loss_type == "mse":
            self.loss_fn = CustomLossFunctions.mse_loss
        elif config.loss_type == "cross_entropy":
            self.loss_fn = CustomLossFunctions.cross_entropy_loss
        else:
            raise ValueError(f"Unknown loss type: {config.loss_type}")
            
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute the loss using the custom loss function."""
        # Forward pass
        outputs = model(
            inputs['input_ids'],
            attention_mask=inputs.get('attention_mask')
        )
        
        # Calculate loss
        loss = self.loss_fn(
            outputs,
            inputs['labels'],
            inputs.get('attention_mask')
        )
        
        if return_outputs:
            return loss, {'outputs': outputs}
        return loss
        
    def compute_metrics(
        self,
        eval_preds: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute metrics using the custom metrics functions."""
        outputs, labels = eval_preds
        attention_mask = None  # Add attention mask if available
        
        # Calculate metrics
        metrics = CustomMetrics.calculate_metrics(
            outputs,
            labels,
            attention_mask,
            self.args.metrics
        )
        
        return metrics
        
    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create the optimizer."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
    def create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int
    ) -> Optional[Any]:
        """Create the learning rate scheduler."""
        from ...utils.utils import WarmupLinearSchedule
        return WarmupLinearSchedule(
            optimizer,
            warmup_steps=self.args.warmup_steps,
            total_steps=num_training_steps
        ) 