from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk

class TnDDataset(Dataset):
    """
    Dataset class for TnD training.
    
    Attributes:
        tokenizer: Tokenizer for the model
        data: List of data samples
        max_length: Maximum sequence length
    """
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        data_path: str,
        max_length: int = 512,
        split: str = "train"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if data_path.endswith('.json'):
            self.data = load_dataset('json', data_files=data_path)[split]
        else:
            self.data = load_from_disk(data_path)[split]
            
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        
        # Tokenize the text
        encodings = self.tokenizer(
            sample['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze()
        }
        
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for the dataloader."""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

class RewardDataset(Dataset):
    """
    Dataset class for reward model training.
    
    Attributes:
        tokenizer: Tokenizer for the model
        data: List of data samples
        max_length: Maximum sequence length
    """
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        data_path: str,
        max_length: int = 512,
        split: str = "train"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if data_path.endswith('.json'):
            self.data = load_dataset('json', data_files=data_path)[split]
        else:
            self.data = load_from_disk(data_path)[split]
            
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        
        # Tokenize the text
        encodings = self.tokenizer(
            sample['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': torch.tensor(sample['label'], dtype=torch.float)
        }
        
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for the dataloader."""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        } 