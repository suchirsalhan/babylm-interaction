import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from typing import Dict, List, Optional
import yaml
import os

class RewardModelTrainer:
    """
    Trainer for the reward model.
    
    Attributes:
        model: The reward model
        tokenizer: Tokenizer for the model
        config: Configuration dictionary
    """
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config['model_name']
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name']
        )
        
    def train(self, train_dataset, eval_dataset=None):
        """Train the reward model."""
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            warmup_steps=self.config['warmup_steps'],
            weight_decay=self.config['weight_decay'],
            logging_dir=self.config['logging_dir'],
            logging_steps=self.config['logging_steps'],
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            save_steps=self.config['save_steps'],
            load_best_model_at_end=True if eval_dataset else False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        trainer.train()
        
    def save_model(self, output_dir: str):
        """Save the trained model."""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
    def load_model(self, model_dir: str):
        """Load a trained model."""
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir) 