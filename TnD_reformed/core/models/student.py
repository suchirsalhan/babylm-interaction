from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional, Dict, Any
from pathlib import Path

class StudentModel:
    def __init__(
        self,
        model_path: str = "/nvme0n1-disk/projects/babylm/models/child",
        device: Optional[str] = None,
        **kwargs
    ):
        self.model_path = Path(model_path)
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        
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
            
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using student model."""
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        return {
            "sequences": outputs,
            "logits": self.get_student_logits(input_ids, attention_mask)
        }
            
    def save(self, save_path: Optional[str] = None):
        """Save model and tokenizer."""
        save_path = Path(save_path) if save_path else self.model_path
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
    def load(self, load_path: Optional[str] = None):
        """Load model and tokenizer."""
        load_path = Path(load_path) if load_path else self.model_path
        self.model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        
    def train(self):
        """Set model to training mode."""
        self.model.train()
        
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval() 