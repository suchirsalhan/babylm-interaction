from typing import Dict, Optional, Tuple
import torch
from torch import nn
import numpy as np

class CustomLossFunctions:
    """Collection of custom loss functions for PPO training."""
    
    @staticmethod
    def kl_regularized_loss(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        rewards: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kl_coeff: float = 0.1
    ) -> torch.Tensor:
        """PPO loss with KL divergence regularization."""
        # Calculate PPO loss
        ratio = torch.exp(student_logits - teacher_logits)
        clipped_ratio = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
        policy_loss = -torch.min(ratio * rewards, clipped_ratio * rewards)
        
        # Calculate KL divergence
        kl_div = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_logits, dim=-1),
            torch.nn.functional.softmax(teacher_logits, dim=-1),
            reduction='batchmean'
        )
        
        # Combine losses
        total_loss = policy_loss + kl_coeff * kl_div
        
        if attention_mask is not None:
            total_loss = total_loss * attention_mask.unsqueeze(-1)
            
        return total_loss.mean()
        
    @staticmethod
    def entropy_regularized_loss(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        rewards: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        entropy_coeff: float = 0.01
    ) -> torch.Tensor:
        """PPO loss with entropy regularization."""
        # Calculate PPO loss
        ratio = torch.exp(student_logits - teacher_logits)
        clipped_ratio = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
        policy_loss = -torch.min(ratio * rewards, clipped_ratio * rewards)
        
        # Calculate entropy
        probs = torch.nn.functional.softmax(student_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        
        # Combine losses
        total_loss = policy_loss - entropy_coeff * entropy
        
        if attention_mask is not None:
            total_loss = total_loss * attention_mask.unsqueeze(-1)
            
        return total_loss.mean()

class CustomRewardFunctions:
    """Collection of custom reward functions for PPO training."""
    
    @staticmethod
    def teacher_guided_reward(
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Reward based on teacher-student logit similarity."""
        # Calculate similarity between teacher and student logits
        similarity = torch.nn.functional.cosine_similarity(
            teacher_logits,
            student_logits,
            dim=-1
        )
        
        # Scale by temperature
        rewards = similarity / temperature
        
        if attention_mask is not None:
            rewards = rewards * attention_mask
            
        return rewards
        
    @staticmethod
    def mixed_reward(
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        base_reward_model: nn.Module,
        teacher_weight: float = 0.5
    ) -> torch.Tensor:
        """Combine teacher guidance with base reward model."""
        # Get base rewards
        base_rewards = base_reward_model(input_ids, attention_mask)
        
        # Get teacher-guided rewards
        teacher_rewards = CustomRewardFunctions.teacher_guided_reward(
            input_ids,
            attention_mask,
            teacher_logits,
            student_logits
        )
        
        # Combine rewards
        rewards = teacher_weight * teacher_rewards + (1 - teacher_weight) * base_rewards
        
        return rewards

class CustomGradientScaling:
    """Collection of custom gradient scaling functions."""
    
    @staticmethod
    def dynamic_gradient_scaling(
        loss: torch.Tensor,
        parameters: torch.Tensor,
        max_norm: float = 1.0,
        min_scale: float = 0.1
    ) -> torch.Tensor:
        """Scale gradients based on their magnitude."""
        # Calculate gradient norms
        grad_norms = []
        for p in parameters:
            if p.grad is not None:
                grad_norms.append(p.grad.norm().item())
        
        if not grad_norms:
            return loss
            
        # Calculate scaling factor
        max_grad_norm = max(grad_norms)
        scale = min(max_norm / (max_grad_norm + 1e-6), 1.0)
        scale = max(scale, min_scale)
        
        return loss * scale
        
    @staticmethod
    def layer_wise_gradient_scaling(
        loss: torch.Tensor,
        parameters: torch.Tensor,
        layer_weights: Dict[str, float]
    ) -> torch.Tensor:
        """Scale gradients differently for different layers."""
        scaled_loss = loss
        for name, param in parameters:
            if param.grad is not None and name in layer_weights:
                scaled_loss = scaled_loss * layer_weights[name]
                
        return scaled_loss

class CustomMetrics:
    """Collection of custom metrics for training monitoring."""
    
    @staticmethod
    def calculate_kl_divergence(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> float:
        """Calculate KL divergence between teacher and student distributions."""
        kl_div = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_logits, dim=-1),
            torch.nn.functional.softmax(teacher_logits, dim=-1),
            reduction='batchmean'
        )
        return kl_div.item()
        
    @staticmethod
    def calculate_entropy(
        logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> float:
        """Calculate entropy of the model's predictions."""
        probs = torch.nn.functional.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        
        if attention_mask is not None:
            entropy = entropy * attention_mask
            
        return entropy.mean().item()
        
    @staticmethod
    def calculate_reward_stats(
        rewards: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Calculate statistics about rewards."""
        if attention_mask is not None:
            rewards = rewards * attention_mask
            
        return {
            'reward_mean': rewards.mean().item(),
            'reward_std': rewards.std().item(),
            'reward_min': rewards.min().item(),
            'reward_max': rewards.max().item()
        } 