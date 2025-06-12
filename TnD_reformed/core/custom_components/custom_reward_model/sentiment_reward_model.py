from typing import List, Union, Optional

import torch
from transformers import PreTrainedTokenizerBase, pipeline

from .base_reward_model import BaseRewardModel


class SentimentRewardModel(BaseRewardModel):
    """Reward model that uses sentiment analysis to score responses.
    
    This reward model combines the query and response, then uses a sentiment analysis
    pipeline to get a positive sentiment score as the reward.
    """
    
    def __init__(self, device: str = "cuda", sent_args: dict = None):
        """Initialize the sentiment reward model.
        
        Args:
            device: Device to run the sentiment analysis on ("cuda" or "cpu")
        """
        super().__init__()
        self.sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="lvwerra/distilbert-imdb",
            device=device
        )
        self.sent_args = sent_args or {}
    
    def _decode_input(
        self, 
        input_data: Union[torch.Tensor, str], 
        tokenizer: Optional[PreTrainedTokenizerBase] = None
    ) -> str:
        """Decode input data whether it's a tensor or string.
        
        Args:
            input_data: Either a tensor to decode or a string
            tokenizer: Tokenizer to use for decoding (required if input_data is tensor)
            
        Returns:
            Decoded string
        """
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, torch.Tensor):
            if tokenizer is None:
                raise ValueError("Tokenizer is required when input_data is a tensor")
            return tokenizer.decode(input_data.squeeze(), skip_special_tokens=True)
        else:
            raise ValueError(f"Input data must be str or torch.Tensor, got {type(input_data)}")
    
    def compute_rewards(
        self,
        child_queries: Union[List[torch.Tensor], List[str]],
        child_responses: Union[List[torch.Tensor], List[str]],
        teacher_queries: Union[List[torch.Tensor], List[str]],
        teacher_responses: Union[List[torch.Tensor], List[str]],
        child_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        teacher_tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> List[torch.Tensor]:
        """Compute rewards based on sentiment analysis of combined query and response.
        
        Args:
            child_queries: List of query tensors or strings for child model
            child_responses: List of response tensors or strings from child model
            teacher_queries: Not used in this implementation
            teacher_responses: Not used in this implementation
            child_tokenizer: Tokenizer for decoding child model outputs (required if inputs are tensors)
            teacher_tokenizer: Not used in this implementation
            
        Returns:
            List of sentiment score tensors for each response
        """
        # Check if we need tokenizers for tensor inputs
        needs_child_tokenizer = any(isinstance(item, torch.Tensor) for item in child_queries + child_responses)
        
        if needs_child_tokenizer and child_tokenizer is None:
            raise ValueError("child_tokenizer is required when child inputs contain tensors")
        
        # Decode texts (handles both tensor and string inputs)
        child_query_texts = [self._decode_input(q, child_tokenizer) for q in child_queries]
        child_response_texts = [self._decode_input(r, child_tokenizer) for r in child_responses]
        
        # Combine query and response texts
        texts = [q + r for q, r in zip(child_query_texts, child_response_texts)]
        
        # Get sentiment scores
        sentiments = self.sentiment_pipe(texts, **self.sent_args)
        rewards = [
            item["score"]
            for output in sentiments
            for item in output
            if item["label"] == "POSITIVE"
        ]
        rewards = [torch.tensor(reward) for reward in rewards]
                
        return rewards