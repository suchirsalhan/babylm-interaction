#!/usr/bin/env python3
"""
Training script for dialogue PPO with TAACO rewards.
This script trains a child model using PPO with teacher guidance and TAACO-based rewards.
"""

import sys
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import json
import argparse
from datetime import datetime
import wandb

# Add paths
sys.path.append("../..")
from transformers import AutoTokenizer, GenerationConfig
from trl import AutoModelForCausalLMWithValueHead
from datasets import Dataset

# Import custom components
from core.trainers.ppo_trainer.custom_ppo import CustomPPOTrainer
from core.trainers.ppo_trainer.config import CustomPPOConfig
from core.custom_components.custom_reward_model.taaco_reward_model import TAACORewardModel
from core.custom_components.custom_interaction.exp1_dialogue_interaction_model import DialogueInteractionModel


class DialogueDataset:
    """Dataset class for handling dialogue data from CSV files."""
    
    def __init__(self, csv_path: str, text_column: str = "text", max_length: int = 500):
        """Initialize the dialogue dataset.
        
        Args:
            csv_path: Path to the CSV file containing dialogue data
            text_column: Column name containing the dialogue text
            max_length: Maximum length for dialogue texts (in words)
        """
        self.csv_path = csv_path
        self.text_column = text_column
        self.max_length = max_length
        
        # Load and preprocess data
        self.df = pd.read_csv(csv_path)
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess the dialogue data."""
        print(f"Loading data from {self.csv_path}")
        print(f"Original dataset size: {len(self.df)}")
        
        # Filter out empty texts
        self.df = self.df.dropna(subset=[self.text_column])
        
        # Filter by length (word count)
        self.df = self.df[self.df[self.text_column].str.split().str.len() <= self.max_length]
        self.df = self.df[self.df[self.text_column].str.split().str.len() >= 20]  # Min 20 words
        
        print(f"Filtered dataset size: {len(self.df)}")
        
        # Extract just the dialogue text
        self.dialogues = self.df[self.text_column].tolist()
    
    def __len__(self):
        return len(self.dialogues)
    
    def __getitem__(self, idx):
        return self.dialogues[idx]
    
    def to_dataset(self) -> Dataset:
        """Convert to HuggingFace Dataset format."""
        return Dataset.from_dict({
            "dialogue": self.dialogues,
            "input_ids": [None] * len(self.dialogues),  # Will be tokenized later
            "query": self.dialogues  # Same as dialogue for now
        })


def setup_models(device: str = "cuda") -> Tuple:
    """Setup child and teacher models with tokenizers.
    
    Args:
        device: Device to load models on
        
    Returns:
        Tuple of (child_model, teacher_model, child_tokenizer, teacher_tokenizer)
    """
    print("Loading child model...")
    # Load child model (OPT-based)
    child_tokenizer = AutoTokenizer.from_pretrained("Talking-Babies/opt-tokenizer")
    child_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        "Talking-Babies/opt-Talking-Babies-train_100M_2048_preprocess"
    )
    child_model.to(device)
    
    print("Loading teacher model...")
    # Load teacher model (Llama-based)
    teacher_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    teacher_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct"
    )
    teacher_model.to(device)
    
    return child_model, teacher_model, child_tokenizer, teacher_tokenizer


def setup_generation_configs(child_tokenizer, teacher_tokenizer) -> Tuple[Dict, Dict]:
    """Setup generation configurations for child and teacher models.
    
    Args:
        child_tokenizer: Child model tokenizer
        teacher_tokenizer: Teacher model tokenizer
        
    Returns:
        Tuple of (child_generation_args, teacher_generation_args)
    """
    child_generation_args = {
        "max_new_tokens": 80,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.8,
        "num_return_sequences": 1,
        "pad_token_id": child_tokenizer.eos_token_id,
    }
    
    teacher_generation_args = {
        "max_new_tokens": 80,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.5,
        "pad_token_id": teacher_tokenizer.eos_token_id,
    }
    
    return child_generation_args, teacher_generation_args


def calculate_training_steps(dataset_size: int, batch_size: int, num_epochs: int = 1) -> Dict[str, int]:
    """Calculate training steps and checkpoint intervals.
    
    Args:
        dataset_size: Size of the training dataset
        batch_size: Training batch size
        num_epochs: Number of training epochs
        
    Returns:
        Dictionary containing step calculations
    """
    steps_per_epoch = dataset_size // batch_size
    total_steps = steps_per_epoch * num_epochs
    checkpoint_interval = max(1, total_steps // 10)  # Save every 10%
    
    return {
        "dataset_size": dataset_size,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "checkpoint_interval": checkpoint_interval,
        "num_epochs": num_epochs
    }


def setup_taaco_reward_model(device: str = "cuda") -> TAACORewardModel:
    """Setup TAACO reward model with appropriate configuration.
    
    Args:
        device: Device to run the reward model on
        
    Returns:
        Configured TAACO reward model
    """
    # TAACO configuration for comprehensive analysis
    taaco_vars = {
        "sourceKeyOverlap": False, 
        "sourceLSA": False, 
        "sourceLDA": False, 
        "sourceWord2vec": False, 
        "wordsAll": True, 
        "wordsContent": True, 
        "wordsFunction": True, 
        "wordsNoun": True, 
        "wordsPronoun": True, 
        "wordsArgument": True, 
        "wordsVerb": True, 
        "wordsAdjective": True, 
        "wordsAdverb": True, 
        "overlapSentence": True, 
        "overlapParagraph": True, 
        "overlapAdjacent": True, 
        "overlapAdjacent2": True, 
        "otherTTR": True, 
        "otherConnectives": True, 
        "otherGivenness": True, 
        "overlapLSA": True, 
        "overlapLDA": True, 
        "overlapWord2vec": True, 
        "overlapSynonym": True, 
        "overlapNgrams": True, 
        "outputTagged": False, 
        "outputDiagnostic": False
    }
    
    return TAACORewardModel(device=device, taaco_vars=taaco_vars)


def save_checkpoint(
    child_model, 
    interaction_model, 
    step: int, 
    total_steps: int, 
    save_dir: str,
    training_stats: Dict = None
):
    """Save model checkpoint.
    
    Args:
        child_model: Child model to save
        interaction_model: Interaction model for reference
        step: Current training step
        total_steps: Total training steps
        save_dir: Directory to save checkpoints
        training_stats: Optional training statistics to save
    """
    checkpoint_dir = Path(save_dir) / f"checkpoint-step-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save child model
    child_model.save_pretrained(checkpoint_dir / "child_model")
    
    # Save training info
    training_info = {
        "step": step,
        "total_steps": total_steps,
        "progress": step / total_steps,
        "timestamp": datetime.now().isoformat(),
        "training_stats": training_stats or {}
    }
    
    with open(checkpoint_dir / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print(f"Checkpoint saved at step {step}/{total_steps} ({step/total_steps*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Train dialogue PPO with TAACO rewards")
    parser.add_argument("--data_path", type=str, 
                       default="data/dialog_level_taaco_results_filtered.csv",
                       help="Path to dialogue CSV data")
    parser.add_argument("--batch_size", type=int, default=8, 
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1.41e-5,
                       help="Learning rate for PPO training")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--max_epochs", type=int, default=1,
                       help="Maximum number of training epochs")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for training")
    parser.add_argument("--wandb_project", type=str, default="dialogue-ppo-taaco",
                       help="Weights & Biases project name")
    
    args = parser.parse_args()
    
    # Setup
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=f"dialogue-ppo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_epochs": args.max_epochs,
            "device": device,
            "child_model": "Talking-Babies/opt-Talking-Babies-train_100M_2048_preprocess",
            "teacher_model": "meta-llama/Llama-3.2-1B-Instruct",
            "reward_model": "TAACO",
            "data_path": args.data_path,
        }
    )
    print(f"Wandb run initialized: {wandb.run.name}")
    print(f"View logs at: {wandb.run.url}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = DialogueDataset(args.data_path, text_column="text", max_length=400)
    hf_dataset = dataset.to_dataset()
    
    # Calculate training steps
    step_info = calculate_training_steps(len(dataset), args.batch_size, args.max_epochs)
    print(f"Training configuration:")
    for key, value in step_info.items():
        print(f"  {key}: {value}")
    
    # Setup models
    child_model, teacher_model, child_tokenizer, teacher_tokenizer = setup_models(device)
    
    # Setup generation configs
    child_gen_args, teacher_gen_args = setup_generation_configs(child_tokenizer, teacher_tokenizer)
    
    # Setup interaction model
    interaction_model = DialogueInteractionModel(
        child_model=child_model,
        teacher_model=teacher_model,
        child_tokenizer=child_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        student_generation_args=child_gen_args,
        teacher_generation_args=teacher_gen_args,
    )
    
    # Setup reward model
    print("Setting up TAACO reward model...")
    reward_model = setup_taaco_reward_model(device)
    
    # Setup PPO configuration
    config = CustomPPOConfig(
        model_name="Talking-Babies/opt-Talking-Babies-train_100M_2048_preprocess",
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.batch_size // 2,
        log_with="wandb",
        child_generation_args=child_gen_args,
        teacher_generation_args=teacher_gen_args,
    )
    
    # Initialize PPO trainer
    print("Initializing PPO trainer...")
    ppo_trainer = CustomPPOTrainer(
        config=config,
        child_model=child_model,
        ref_model=child_model,  # Use same model as reference
        teacher_model=teacher_model,
        reward_model=reward_model,
        tokenizer=child_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        dataset=hf_dataset,
    )
    
    # Training loop
    print("Starting training...")
    step = 0
    error_count = 0
    training_stats = []
    
    try:
        for epoch in range(args.max_epochs):
            print(f"\nEpoch {epoch + 1}/{args.max_epochs}")
            
            for batch_idx, batch in enumerate(tqdm(ppo_trainer.dataloader, desc=f"Epoch {epoch+1}")):
                try:
                    # Extract dialogue prompts from batch
                    dialogue_prompts = batch.get("query", batch.get("dialogue", []))
                    
                    # Process each dialogue through interaction model
                    batch_child_queries = []
                    batch_child_responses = []
                    batch_teacher_queries = []
                    batch_teacher_responses = []
                    batch_scores = []
                    
                    print(f"Processing batch {batch_idx} with {len(dialogue_prompts)} dialogues...")
                    
                    for i, dialogue_prompt in enumerate(dialogue_prompts):
                        try:
                            # Use interaction model to get child and teacher responses
                            interaction_result = interaction_model.interact(dialogue_prompt)
                            
                            # Extract components from interaction result
                            child_query = interaction_result['child_query']
                            child_response = interaction_result['child_response']
                            teacher_query = interaction_result['teacher_query']
                            teacher_response = interaction_result['teacher_response']
                            
                            # Get decoded strings for TAACO reward computation
                            partial_dialogue = interaction_result['partial_dialogue']
                            child_query_text = interaction_result['decoded_child_query']
                            child_response_text = interaction_result['decoded_child_response']
                            teacher_query_text = interaction_result['decoded_teacher_query']
                            teacher_response_text = interaction_result['decoded_teacher_response']
                            
                            # Calculate reward using TAACO with decoded strings
                            reward_scores = reward_model.compute_rewards(
                                child_queries=[child_query_text],
                                child_responses=[partial_dialogue + child_response_text],
                                teacher_queries=[teacher_query_text],
                                teacher_responses=[partial_dialogue + teacher_response_text]
                            )
                            
                            # Store for batch processing
                            batch_child_queries.append(child_query)
                            batch_child_responses.append(child_response)
                            batch_teacher_queries.append(teacher_query)
                            batch_teacher_responses.append(teacher_response)
                            batch_scores.extend(reward_scores)
                            
                            if i == 0:  # Print first example for debugging
                                print(f"Sample child continuation: {interaction_result['child_continuation'][:100]}...")
                                print(f"Sample teacher completion: {interaction_result['teacher_completion'][:100]}...")
                                print(f"Sample TAACO score: {reward_scores[0]}")
                        
                        except Exception as e:
                            print(f"Error processing dialogue {i}: {e}")
                            continue
                    
                    if not batch_child_queries:
                        print("No valid dialogues in batch, skipping...")
                        continue
                    
                    print(f"Processed {len(batch_child_queries)} dialogues, running PPO step...")
                    
                    # Run PPO step with the collected data
                    stats, returned_scores = ppo_trainer.step(
                        child_queries=batch_child_queries,
                        child_responses=batch_child_responses,
                        teacher_queries=batch_teacher_queries,
                        teacher_responses=batch_teacher_responses,
                        scores=batch_scores
                    )
                    
                    print(f"PPO step completed. Avg original score: {sum(score.item() if hasattr(score, 'item') else score for score in batch_scores) / len(batch_scores):.4f}")
                    print(f"Policy loss: {stats.get('loss/policy', 'N/A')}, Value loss: {stats.get('loss/value', 'N/A')}")
                    
                    # Log to wandb
                    avg_original_score = sum(score.item() if hasattr(score, 'item') else score for score in batch_scores) / len(batch_scores)
                    avg_returned_score = sum(score.item() if hasattr(score, 'item') else score for score in returned_scores) / len(returned_scores)
                    
                    wandb.log({
                        "step": step,
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "avg_original_score": avg_original_score,
                        "avg_returned_score": avg_returned_score,
                        "policy_loss": stats.get('loss/policy', 0),
                        "value_loss": stats.get('loss/value', 0),
                        "total_loss": stats.get('loss/total', 0),
                        "entropy": stats.get('policy/entropy', 0),
                        "approx_kl": stats.get('policy/approxkl', 0),
                        "learning_rate": stats.get('ppo/learning_rate', args.learning_rate),
                    })
                    
                    # Log statistics
                    ppo_trainer.log_stats(stats, batch, returned_scores, columns_to_log=["query"])
                    
                    # Store training stats
                    training_stats.append({
                        "step": step,
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "stats": stats,
                        "original_scores": [score.item() if hasattr(score, 'item') else score for score in batch_scores],
                        "returned_scores": [score.item() if hasattr(score, 'item') else score for score in returned_scores],
                        "avg_original_score": avg_original_score,
                        "avg_returned_score": avg_returned_score
                    })
                    
                    step += 1
                    
                    # Save checkpoint at intervals
                    if step % step_info["checkpoint_interval"] == 0:
                        save_checkpoint(
                            child_model, 
                            interaction_model, 
                            step, 
                            step_info["total_steps"], 
                            args.save_dir,
                            training_stats[-10:]  # Last 10 training stats
                        )
                
                except Exception as e:
                    error_count += 1
                    print(f"Error in step {step}: {e}")
                    if error_count > 10:
                        print("Too many errors, stopping training")
                        break
                    continue
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        # Save final checkpoint
        print("Saving final checkpoint...")
        save_checkpoint(
            child_model, 
            interaction_model, 
            step, 
            step_info["total_steps"], 
            args.save_dir,
            training_stats[-10:] if training_stats else None
        )
        
        # Save training log
        training_log = {
            "step_info": step_info,
            "args": vars(args),
            "final_step": step,
            "error_count": error_count,
            "training_stats": training_stats
        }
        
        with open(save_dir / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2, default=str)
        
        print(f"Training completed. Final step: {step}/{step_info['total_steps']}")
        print(f"Error count: {error_count}")
        print(f"Checkpoints saved in: {args.save_dir}")
        
        # Close wandb run
        wandb.finish()


if __name__ == "__main__":
    main() 