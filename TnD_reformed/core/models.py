from typing import Any, Dict, Union, List, Optional, Tuple
import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from transformers.generation.utils import *
from pathlib import Path

class TeacherModel(nn.Module):
    def __init__(
        self,
        model_path: str = "/nvme0n1-disk/projects/babylm/models/teacher",
        **kwargs
    ):
        super().__init__()
        self.model_path = Path(model_path)
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def get_teacher_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Get logits from teacher model."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs
            )
            return outputs.logits
            
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

class StudentModel(nn.Module):
    def __init__(
        self,
        model_path: str = "/nvme0n1-disk/projects/babylm/models/child",
        **kwargs
    ):
        super().__init__()
        self.model_path = Path(model_path)
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def get_student_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Get logits from student model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        return outputs.logits
            
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

class RewardModel(nn.Module):
    def __init__(
        self,
        model_path: str = "/nvme0n1-disk/projects/babylm/models/reward",
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        self.model_path = Path(model_path)
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def get_reward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Get reward for input sequences."""
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        return outputs.logits
            
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
