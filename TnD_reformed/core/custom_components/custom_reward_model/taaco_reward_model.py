import sys
import os
from typing import List, Dict, Optional, Union
import numpy as np

import torch
from transformers import PreTrainedTokenizerBase

from .base_reward_model import BaseRewardModel

# Add path to access TAACO processor
taaco_path = "/workspace/TAACO"
if taaco_path not in sys.path:
    sys.path.append(taaco_path)

from TAACO_pure_txt import TAACOProcessor


class TAACORewardModel(BaseRewardModel):
    """Reward model that uses TAACO (Tool for the Automatic Analysis of Cohesion) 
    to score linguistic quality of child and teacher responses.
    
    TAACO analyzes text cohesion, coherence, lexical diversity, and other linguistic features
    to provide comprehensive text quality metrics.
    """
    
    def __init__(
        self, 
        device: str = "cuda",
        metric_weights: Optional[Dict[str, float]] = None,
        taaco_vars: Dict = None,
        focus_on_teacher: bool = True
    ):
        """Initialize the TAACO reward model.
        
        Args:
            device: Device to run computations on ("cuda" or "cpu")
            metric_weights: Dictionary of weights for different TAACO metrics
            enable_lsa: Whether to enable LSA-based coherence metrics
            enable_lda: Whether to enable LDA-based coherence metrics  
            enable_word2vec: Whether to enable Word2Vec-based coherence metrics
            focus_on_teacher: Whether to focus reward calculation on teacher responses
        """
        super().__init__()
        self.device = device
        self.focus_on_teacher = focus_on_teacher
        
        # Initialize TAACO processor
        try:
            self.taaco_processor = TAACOProcessor(taaco_vars)
            print("TAACO processor initialized successfully")
        except Exception as e:
            print(f"Warning: TAACO processor initialization failed: {e}")
            self.taaco_processor = None
        
        # Define default metric weights (higher values = more important)
        self.default_weights = {
            # Lexical diversity metrics
            "lemma_ttr": 0.15,
            "content_ttr": 0.12,
            "lexical_density_tokens": 0.10,
            
            # Coherence and cohesion
            "lsa_1_all_sent": 0.20,
            "lsa_2_all_sent": 0.15,
            "lda_1_all_sent": 0.15,
            "word2vec_1_all_sent": 0.15,
            
            # Connectives and discourse markers
            "all_connective": 0.08,
            "all_logical": 0.08,
            "all_positive": 0.06,
            
            # Givenness and referential cohesion
            "repeated_content_lemmas": 0.05,
            "pronoun_density": 0.03,
            
            # Syntactic complexity
            "basic_connectives": 0.03,
            "sentence_linking": 0.05
        }
        
        # Normalization statistics for TAACO metrics
        self.metric_means = {'noun_ttr': np.float64(8.768112449328236e-16),
                           'verb_ttr': np.float64(1.0801297944824639e-16),
                           'adj_ttr': np.float64(-2.5414818693705032e-17),
                           'lemma_ttr': np.float64(-6.862001047300358e-16),
                           'bigram_lemma_ttr': np.float64(2.827398579674685e-16),
                           'trigram_lemma_ttr': np.float64(2.90582712173729e-15),
                           'adjacent_overlap_all_sent': np.float64(-1.2707409346852516e-17),
                           'repeated_content_lemmas': np.float64(2.5414818693705032e-17),
                           'repeated_content_and_pronoun_lemmas': np.float64(-3.3674634769159165e-16)}
         
        self.metric_stds = {'noun_ttr': np.float64(0.9999999999999999),
                          'verb_ttr': np.float64(1.0),
                          'adj_ttr': np.float64(1.0),
                          'lemma_ttr': np.float64(1.0),
                          'bigram_lemma_ttr': np.float64(0.9999999999999999),
                          'trigram_lemma_ttr': np.float64(1.0),
                          'adjacent_overlap_all_sent': np.float64(1.0),
                          'repeated_content_lemmas': np.float64(1.0),
                          'repeated_content_and_pronoun_lemmas': np.float64(1.0)}
        
        # Use provided weights or defaults
        self.metric_weights = metric_weights if metric_weights is not None else self.default_weights
        
        # Normalize weights to sum to 1
        total_weight = sum(self.metric_weights.values())
        if total_weight > 0:
            self.metric_weights = {k: v/total_weight for k, v in self.metric_weights.items()}
        
    def _process_text_with_taaco(self, text: str) -> Dict[str, float]:
        """Process text with TAACO and return metrics.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of TAACO metrics
        """
        if self.taaco_processor is None:
            print("Warning: TAACO processor not available, returning empty metrics")
            return {}
        
        try:
            # Clean and preprocess text
            cleaned_text = text.strip()
            if len(cleaned_text.split()) < 2:
                print(f"Warning: Text too short for TAACO analysis: '{cleaned_text}'")
                return {}
            
            # Process with TAACO
            metrics = self.taaco_processor.process_text(cleaned_text)
            
            # Handle potential errors
            if isinstance(metrics, dict) and "error" in metrics:
                print(f"TAACO processing error: {metrics['error']}")
                return {}
                
            return metrics
            
        except Exception as e:
            print(f"Error processing text with TAACO: {e}")
            return {}
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Calculate a composite quality score from TAACO metrics using statistical normalization.
        
        Args:
            metrics: Dictionary of TAACO metrics
            
        Returns:
            Composite score between 0 and 1
        """
        if not metrics:
            return 0.0
        
        normalized_scores = []
        
        # Normalize each metric using z-score normalization
        for metric_name, metric_value in metrics.items():
            if metric_name in self.metric_means and metric_name in self.metric_stds:
                # Handle potential None or invalid values
                if metric_value is None or not isinstance(metric_value, (int, float)):
                    continue
                
                # Handle inf or nan values
                if np.isnan(metric_value) or np.isinf(metric_value):
                    continue
                
                # Z-score normalization: (value - mean) / std
                mean = float(self.metric_means[metric_name])
                std = float(self.metric_stds[metric_name])
                
                # Avoid division by zero
                if std == 0:
                    normalized_value = 0.0
                else:
                    normalized_value = (metric_value - mean) / std
                
                normalized_scores.append(normalized_value)
        
        if not normalized_scores:
            return 0.0
        
        # Take the average of normalized scores
        avg_normalized_score = np.mean(normalized_scores)
        
        # Map to [0, 1] range using sigmoid function
        # sigmoid(x) = 1 / (1 + exp(-x))
        sigmoid_score = 1.0 / (1.0 + np.exp(-avg_normalized_score))
        
        return float(sigmoid_score)
    
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
    
    def compute_individual_scores(self, responses:Union[List[torch.Tensor], List[str]], tokenizer: Optional[PreTrainedTokenizerBase]) -> List[torch.Tensor]:
        """Compute individual scores for a list of responses.
        
        Args:
            responses: List of response tensors or strings
            tokenizer: Tokenizer to use for decoding (required if responses are tensors)
        """
        scores = []
        for i in range(len(responses)):
            if isinstance(responses[i], torch.Tensor):
                response_text = self._decode_input(responses[i], tokenizer)
            else:
                response_text = responses[i]
            metrics = self._process_text_with_taaco(response_text)
            score = self._calculate_composite_score(metrics)
            scores.append(score)
        return scores
        
    def compute_rewards(
        self,
        child_queries: Union[List[torch.Tensor], List[str]],
        child_responses: Union[List[torch.Tensor], List[str]],
        teacher_queries: Union[List[torch.Tensor], List[str]],
        teacher_responses: Union[List[torch.Tensor], List[str]],
        child_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        teacher_tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> List[torch.Tensor]:
        """Compute rewards based on TAACO linguistic analysis.
        
        Args:
            child_queries: List of query tensors or strings for child model
            child_responses: List of response tensors or strings from child model
            teacher_queries: List of query tensors or strings for teacher model
            teacher_responses: List of response tensors or strings from teacher model
            child_tokenizer: Tokenizer for decoding child model outputs (required if inputs are tensors)
            teacher_tokenizer: Tokenizer for decoding teacher model outputs (required if inputs are tensors)
            
        Returns:
            List of reward tensors for each response
        """
        # Validate input lengths
        if not (len(child_queries) == len(child_responses) == len(teacher_queries) == len(teacher_responses)):
            raise ValueError("All input lists must have the same length")
        
        # Check if we need tokenizers for tensor inputs
        needs_child_tokenizer = any(isinstance(item, torch.Tensor) for item in child_queries + child_responses)
        needs_teacher_tokenizer = any(isinstance(item, torch.Tensor) for item in teacher_queries + teacher_responses)
        
        if needs_child_tokenizer and child_tokenizer is None:
            raise ValueError("child_tokenizer is required when child inputs contain tensors")
        if needs_teacher_tokenizer and teacher_tokenizer is None:
            raise ValueError("teacher_tokenizer is required when teacher inputs contain tensors")
        
        rewards = []
        
        for i in range(len(child_responses)):
            try:
                # Decode texts (handles both tensor and string inputs)
                child_query_text = self._decode_input(child_queries[i], child_tokenizer)
                child_response_text = self._decode_input(child_responses[i], child_tokenizer)
                teacher_query_text = self._decode_input(teacher_queries[i], teacher_tokenizer)
                teacher_response_text = self._decode_input(teacher_responses[i], teacher_tokenizer)
                
                # Analyze teacher text with TAACO
                primary_metrics = self._process_text_with_taaco(teacher_response_text)
                primary_score = self._calculate_composite_score(primary_metrics)
                
                # Analyze child text for comparison
                secondary_metrics = self._process_text_with_taaco(child_response_text)
                secondary_score = self._calculate_composite_score(secondary_metrics)
                
                # Calculate final reward
                final_reward = secondary_score - primary_score
                
                rewards.append(torch.tensor(final_reward, device=self.device))
                
            except Exception as e:
                print(f"Error computing reward for sample {i}: {e}")
                # Return a neutral reward for failed cases
                rewards.append(torch.tensor(0.5, device=self.device))
        
        return rewards
    
    def get_detailed_analysis(
        self,
        text: str
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Get detailed TAACO analysis for a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing detailed metrics and composite score
        """
        metrics = self._process_text_with_taaco(text)
        composite_score = self._calculate_composite_score(metrics)
        
        return {
            "composite_score": composite_score,
            "detailed_metrics": metrics,
            "weighted_metrics": {
                k: metrics.get(k, 0.0) * self.metric_weights.get(k, 0.0) 
                for k in self.metric_weights.keys()
                if k in metrics
            }
        } 