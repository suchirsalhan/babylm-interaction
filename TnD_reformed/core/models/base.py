from typing import Any, Dict, Optional, Union
import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizer
)

class BaseModel(nn.Module):
    """
    Base class for all models in the TnD framework.
    
    Attributes:
        model: The underlying model
        tokenizer: Tokenizer for the model
        config: Model configuration
    """
    def __init__(
        self,
        model_name: str,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.model_name = model_name
        self.config = config or {}
        self.tokenizer = tokenizer
        
    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        return self.model(*args, **kwargs)
        
    def save(self, output_dir: str):
        """Save the model and tokenizer."""
        self.model.save_pretrained(output_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)
            
    def load(self, model_dir: str):
        """Load the model and tokenizer."""
        self.model = self.model.from_pretrained(model_dir)
        if self.tokenizer:
            self.tokenizer = self.tokenizer.from_pretrained(model_dir)
            
    def get_trainable_parameters(self):
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def freeze_parameters(self, freeze: bool = True):
        """Freeze or unfreeze model parameters."""
        for param in self.parameters():
            param.requires_grad = not freeze 