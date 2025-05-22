import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import set_seed
from typing import Any, Dict, List, Optional, Union
import math
from dataclasses import dataclass
import torch.optim.lr_scheduler as lr_scheduler

def set_all_seeds(seed_value: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed_value)

def has_accumulated_gradients(model: nn.Module) -> bool:
    """Check if the model has accumulated gradients."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"No gradient for {name}")
                return False
            elif torch.all(param.grad == 0):
                print(f"Zero gradient for {name}")
                return False
    return True

class CosineLRScheduler(lr_scheduler._LRScheduler):
    """Cosine learning rate scheduler."""
    def __init__(self, optimizer, max_lr, steps_per_cycle, from_zero=False, last_epoch=-1):
        self.max_lr = max_lr
        self.steps_per_cycle = steps_per_cycle
        self.from_zero = from_zero
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        cycle = self.last_epoch // self.steps_per_cycle
        x = self.last_epoch % self.steps_per_cycle
        if self.from_zero:
            lr = self.max_lr * (1 - math.cos(math.pi * x / self.steps_per_cycle)) / 2
        else:
            lr = self.max_lr * (1 + math.cos(math.pi * x / self.steps_per_cycle)) / 2
        return [lr for _ in self.optimizer.param_groups]

class WarmupLinearSchedule(lr_scheduler._LRScheduler):
    """Linear warmup learning rate scheduler."""
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        return [base_lr * (self.total_steps - self.last_epoch) / (self.total_steps - self.warmup_steps) for base_lr in self.base_lrs]

class PPOLRScheduler(lr_scheduler._LRScheduler):
    """PPO learning rate scheduler."""
    def __init__(self, optimizer, L, P=0.0, Q=0.0, D=0.0, R=0.0, last_epoch=-1):
        self.L = L
        self.P = P
        self.Q = Q
        self.D = D
        self.R = R
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Implementation from original code
        pass

    def set_current_clm_lr(self, A):
        """Set the current CLM learning rate."""
        self.L = A

class AnnealScheduler:
    """Scheduler for annealing parameters."""
    def __init__(self, T, D, N_CLM, N_PPO):
        self.T = T
        self.D = D
        self.N_CLM = N_CLM
        self.N_PPO = N_PPO

    def get_update_policy(self):
        """Get the current update policy."""
        # Implementation from original code
        pass

class CustomDataset(Dataset):
    """Custom dataset class."""
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids']) 