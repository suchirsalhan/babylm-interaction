from typing import Any, Dict, Optional, Union
import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from .base import BaseModel

class RewardModel(BaseModel):
    """
    Base class for reward models.
    
    Attributes:
        model: The reward model
        tokenizer: Tokenizer for the model
        config: Model configuration
    """
    def __init__(
        self,
        model_name: str,
        tokenizer: Optional[AutoTokenizer] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(model_name, tokenizer, config)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if not tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
    def get_reward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get reward for input sequences."""
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits

class RegressionRewardModel(RewardModel):
    """
    Reward model for regression tasks.
    
    Attributes:
        model: The reward model
        tokenizer: Tokenizer for the model
        config: Model configuration
    """
    def __init__(
        self,
        model_name: str,
        tokenizer: Optional[AutoTokenizer] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(model_name, tokenizer, config)
        
    def get_reward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get continuous reward values."""
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits.squeeze(-1)

class ClassificationRewardModel(RewardModel):
    """
    Reward model for classification tasks.
    
    Attributes:
        model: The reward model
        tokenizer: Tokenizer for the model
        config: Model configuration
    """
    def __init__(
        self,
        model_name: str,
        tokenizer: Optional[AutoTokenizer] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(model_name, tokenizer, config)
        
    def get_reward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get discrete reward values."""
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask
        )
        return torch.softmax(outputs.logits, dim=-1)

class CustomRewardModel(RewardModel):
    """
    Custom reward model with additional functionality.
    
    Attributes:
        model: The reward model
        tokenizer: Tokenizer for the model
        config: Model configuration
    """
    def __init__(
        self,
        model_name: str,
        tokenizer: Optional[AutoTokenizer] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(model_name, tokenizer, config)
        
    def get_reward_with_attention(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Get reward and attention weights."""
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        return {
            'reward': outputs.logits,
            'attention': outputs.attentions
        } 