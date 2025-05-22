from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk

def preprocess_text(text: str) -> str:
    """
    Preprocess text by removing extra whitespace and normalizing.
    
    Args:
        text: Input text to preprocess
        
    Returns:
        Preprocessed text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Add any additional preprocessing steps here
    return text

def create_training_pairs(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int = 512
) -> List[Dict[str, torch.Tensor]]:
    """
    Create training pairs for TnD training.
    
    Args:
        dataset: Input dataset
        tokenizer: Tokenizer for the model
        max_length: Maximum sequence length
        
    Returns:
        List of training pairs
    """
    pairs = []
    
    for sample in dataset:
        # Preprocess the text
        text = preprocess_text(sample['text'])
        
        # Tokenize the text
        encodings = tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        pairs.append({
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze()
        })
        
    return pairs

def create_reward_data(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int = 512
) -> List[Dict[str, torch.Tensor]]:
    """
    Create training data for the reward model.
    
    Args:
        dataset: Input dataset
        tokenizer: Tokenizer for the model
        max_length: Maximum sequence length
        
    Returns:
        List of training data
    """
    data = []
    
    for sample in dataset:
        # Preprocess the text
        text = preprocess_text(sample['text'])
        
        # Tokenize the text
        encodings = tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        data.append({
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': torch.tensor(sample['label'], dtype=torch.float)
        })
        
    return data

def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Dict[str, Dataset]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: Input dataset
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        
    Returns:
        Dictionary containing split datasets
    """
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # Split the dataset
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, total_size))
    
    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    } 