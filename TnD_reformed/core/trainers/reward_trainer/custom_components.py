from typing import Dict, Optional, Tuple
import torch
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class CustomLossFunctions:
    """Collection of custom loss functions for reward training."""
    
    @staticmethod
    def mse_loss(
        outputs: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Mean squared error loss with optional masking."""
        if attention_mask is not None:
            loss = torch.nn.functional.mse_loss(
                outputs * attention_mask.unsqueeze(-1),
                labels * attention_mask.unsqueeze(-1)
            )
        else:
            loss = torch.nn.functional.mse_loss(outputs, labels)
        return loss
        
    @staticmethod
    def cross_entropy_loss(
        outputs: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Cross entropy loss with optional masking."""
        if attention_mask is not None:
            loss = torch.nn.functional.cross_entropy(
                outputs * attention_mask.unsqueeze(-1),
                labels * attention_mask.unsqueeze(-1)
            )
        else:
            loss = torch.nn.functional.cross_entropy(outputs, labels)
        return loss

class CustomMetrics:
    """Collection of custom metrics for reward training."""
    
    @staticmethod
    def calculate_accuracy(
        outputs: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> float:
        """Calculate accuracy score."""
        if attention_mask is not None:
            outputs = outputs[attention_mask.bool()]
            labels = labels[attention_mask.bool()]
            
        preds = torch.argmax(outputs, dim=-1)
        return accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        
    @staticmethod
    def calculate_f1(
        outputs: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> float:
        """Calculate F1 score."""
        if attention_mask is not None:
            outputs = outputs[attention_mask.bool()]
            labels = labels[attention_mask.bool()]
            
        preds = torch.argmax(outputs, dim=-1)
        return f1_score(
            labels.cpu().numpy(),
            preds.cpu().numpy(),
            average='weighted'
        )
        
    @staticmethod
    def calculate_metrics(
        outputs: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Calculate multiple metrics at once."""
        metrics = metrics or ["accuracy", "f1"]
        results = {}
        
        for metric_name in metrics:
            if metric_name == "accuracy":
                results[metric_name] = CustomMetrics.calculate_accuracy(
                    outputs, labels, attention_mask
                )
            elif metric_name == "f1":
                results[metric_name] = CustomMetrics.calculate_f1(
                    outputs, labels, attention_mask
                )
                
        return results 