from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Union

import torch
from transformers import PreTrainedTokenizerBase
from trl import PreTrainedModelWrapper


class BaseInteractionModel(ABC):
    """Base class for interaction models that handle child-teacher model interactions.
    
    This abstract base class defines the interface that all interaction models must implement.
    The main requirement is to implement the interact method that takes a prompt and returns
    both child and teacher queries and responses.
    """
    
    def __init__(
        self,
        child_model: PreTrainedModelWrapper,
        teacher_model: PreTrainedModelWrapper,
        child_tokenizer: PreTrainedTokenizerBase,
        teacher_tokenizer: PreTrainedTokenizerBase,
        student_generation_args: dict,
        teacher_generation_args: dict,
        device: str = "cuda"
    ):
        """Initialize the interaction model with child and teacher models and tokenizers.
        
        Args:
            child_model: The child model to generate responses
            teacher_model: The teacher model to generate responses
            child_tokenizer: Tokenizer for the child model
            teacher_tokenizer: Tokenizer for the teacher model
            device: Device to run the models on ("cuda" or "cpu")
        """
        self.child_model = child_model
        self.teacher_model = teacher_model
        self.child_tokenizer = child_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.student_generation_args = student_generation_args
        self.teacher_generation_args = teacher_generation_args
        self.device = device
        
        # Move models to device
        # self.child_model.to(device)
        # self.teacher_model.to(device)
    
    @abstractmethod
    def interact(
        self,
        input_prompt: Union[str, Dict[str, str]],
        **generation_kwargs
    ) -> Dict[str, Union[torch.Tensor, str]]:
        """Process a single prompt through both child and teacher models.
        
        Args:
            input_prompt: The input prompt text (default to input for child model) or a dict containing input for child and teacher model
            max_new_tokens: Maximum number of new tokens to generate
            **generation_kwargs: Additional keyword arguments for generation
            
        Returns:
            Dictionary containing:
            - child_query: Query tensor for child model
            - child_response: Response tensor from child model
            - teacher_query: Query tensor for teacher model
            - teacher_response: Response tensor from teacher model
            Additional keys may be included by specific implementations
        """
        pass
    
    @abstractmethod
    def batch_interact(
        self,
        input_prompts: List[Union[str, Dict[str, str]]],
    ) -> Dict[str, List[Union[torch.Tensor, str]]]:
        """Process multiple prompts through both child and teacher models.
        
        Args:
            input_prompts: List of input prompts, where each prompt is either a string (default to input for child model) 
                          or a dict containing input for child and teacher model
            max_new_tokens: Maximum number of new tokens to generate
            **generation_kwargs: Additional keyword arguments for generation
            
        Returns:
            Dictionary containing:
            - child_queries: List of query tensors for child model
            - child_responses: List of response tensors from child model
            - teacher_queries: List of query tensors for teacher model
            - teacher_responses: List of response tensors from teacher model
            Additional keys may be included by specific implementations
        """
        pass
    
    def decode_tokens(
        self,
        tokens: torch.Tensor,
        is_child: bool = True,
        skip_special_tokens: bool = True
    ) -> str:
        """Decode tokens using the appropriate tokenizer.
        
        Args:
            tokens: Tensor of token ids to decode
            is_child: Whether to use child tokenizer (True) or teacher tokenizer (False)
            
        Returns:
            Decoded text string
        """
        tokenizer = self.child_tokenizer if is_child else self.teacher_tokenizer
        return tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
    
    def encode_text(
        self,
        text: str,
        is_child: bool = True
    ) -> torch.Tensor:
        """Encode text using the appropriate tokenizer.
        
        Args:
            text: Text to encode
            is_child: Whether to use child tokenizer (True) or teacher tokenizer (False)
            
        Returns:
            Tensor of token ids
        """
        tokenizer = self.child_tokenizer if is_child else self.teacher_tokenizer
        return tokenizer.encode(text, return_tensors="pt").squeeze().to(self.device)
