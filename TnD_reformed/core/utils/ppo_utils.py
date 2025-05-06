import torch
import torch.nn.functional as F

def clip_by_value(tensor: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    """
    Clip the values of a tensor to a specified minimum and maximum value.
    
    Args:
        tensor (torch.Tensor): The input tensor.
        min_value (float): The minimum value.
        max_value (float): The maximum value.
    
    Returns:
        torch.Tensor: The clipped tensor.
    """
    return torch.max(torch.min(tensor, torch.tensor(max_value)), torch.tensor(min_value))



def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute log probabilities from logits for the given labels.
    
    Args:
        logits (torch.Tensor): The logits tensor.
        labels (torch.Tensor): The ground truth labels.
    
    Returns:
        torch.Tensor: The log probabilities for each label.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
