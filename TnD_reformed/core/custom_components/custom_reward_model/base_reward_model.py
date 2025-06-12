from abc import ABC, abstractmethod
from typing import List, Optional, Union

import torch
from transformers import PreTrainedTokenizerBase


class BaseRewardModel(ABC):
    """Base class for reward models in PPO training.
    
    This abstract base class defines the interface that all reward models must implement.
    The main requirement is to implement the compute_rewards method that takes child and teacher
    queries/responses and returns reward scores.
    """
    
    def __init__(self):
        """Initialize the reward model."""
        pass
    
    @abstractmethod
    def compute_rewards(
        self,
        child_queries: Union[List[torch.Tensor], List[str]],
        child_responses: Union[List[torch.Tensor], List[str]],
        teacher_queries: Union[List[torch.Tensor], List[str]],
        teacher_responses: Union[List[torch.Tensor], List[str]],
        child_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        teacher_tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> torch.Tensor:
        """Compute rewards for the given child and teacher responses.
        
        Args:
            child_queries: List of query tensors or strings for child model
            child_responses: List of response tensors or strings from child model
            teacher_queries: List of query tensors or strings for teacher model
            teacher_responses: List of response tensors or strings from teacher model
            child_tokenizer: Tokenizer for the child model (required if inputs are tensors)
            teacher_tokenizer: Tokenizer for the teacher model (required if inputs are tensors)
            
        Returns:
            torch.Tensor: Tensor of reward scores for each response
        """
        pass 