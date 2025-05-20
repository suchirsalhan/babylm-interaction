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
        child_queries: List[torch.Tensor],
        child_responses: List[torch.Tensor],
        teacher_queries: List[torch.Tensor],
        teacher_responses: List[torch.Tensor],
        child_tokenizer: PreTrainedTokenizerBase,
        teacher_tokenizer: PreTrainedTokenizerBase,
    ) -> torch.Tensor:
        """Compute rewards for the given child and teacher responses.
        
        Args:
            child_queries: List of query tensors for child model
            child_responses: List of response tensors from child model
            teacher_queries: List of query tensors for teacher model
            teacher_responses: List of response tensors from teacher model
            child_tokenizer: Tokenizer for the child model
            teacher_tokenizer: Tokenizer for the teacher model
            
        Returns:
            torch.Tensor: Tensor of reward scores for each response
        """
        pass 