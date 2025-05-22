from typing import List

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
        self.sent_args = sent_args
    
    def compute_rewards(
        self,
        child_queries: List[torch.Tensor],
        child_responses: List[torch.Tensor],
        teacher_queries: List[torch.Tensor],
        teacher_responses: List[torch.Tensor],
        child_tokenizer: PreTrainedTokenizerBase,
        teacher_tokenizer: PreTrainedTokenizerBase,
    ) -> List[torch.Tensor]:
        """Compute rewards based on sentiment analysis of combined query and response.
        
        Args:
            child_queries: List of query tensors for child model
            child_responses: List of response tensors from child model
            teacher_queries: Not used in this implementation
            teacher_responses: Not used in this implementation
            child_tokenizer: Tokenizer for decoding child model outputs
            teacher_tokenizer: Not used in this implementation
            
        Returns:
            torch.Tensor: Tensor of sentiment scores for each response
        """
        child_query_texts = [child_tokenizer.decode(q.squeeze()) for q in child_queries]
        child_response_texts = [child_tokenizer.decode(r.squeeze()) for r in child_responses]
        texts = [q + r for q, r in zip(child_query_texts, child_response_texts)]
        sentiments = self.sentiment_pipe(texts, **self.sent_args)
        rewards = [
            item["score"]
            for output in sentiments
            for item in output
            if item["label"] == "POSITIVE"
        ]
        rewards =[torch.tensor(reward) for reward in rewards]
                
            
        return rewards