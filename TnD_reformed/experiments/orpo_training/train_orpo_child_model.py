#!/usr/bin/env python3
"""
ORPO Training Script for Child Model
Trains the Talking-Babies child model using ORPO on preference data
"""

import os
import json
import logging
from typing import Dict, Any
from datasets import load_from_disk
from trl import ORPOConfig, ORPOTrainer
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback
)
import torch
import wandb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenUsageCallback(TrainerCallback):
    """Custom callback to track token usage during training"""
    
    def __init__(self, train_dataset):
        self.total_tokens = 0
        self.step_tokens = []
        self.train_dataset = train_dataset
        self.current_batch_start = 0
    
    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """Track tokens used at each step using actual token_count from dataset"""
        try:
            # Calculate how many samples were processed in this step
            effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
            if torch.cuda.device_count() > 1:
                effective_batch_size *= torch.cuda.device_count()
            
            # Get the actual token counts for samples in this batch
            batch_end = min(self.current_batch_start + effective_batch_size, len(self.train_dataset))
            
            # Sum up the actual token counts from the dataset
            tokens_this_step = 0
            for i in range(self.current_batch_start, batch_end):
                # Handle dataset cycling (when we go beyond dataset length in multi-epoch training)
                dataset_idx = i % len(self.train_dataset)
                tokens_this_step += self.train_dataset[dataset_idx]['token_count']
            
            self.total_tokens += tokens_this_step
            self.step_tokens.append(self.total_tokens)
            self.current_batch_start = batch_end % len(self.train_dataset)
            
            # Log token usage
            if state.global_step % args.logging_steps == 0:
                logger.info(f"Step {state.global_step}: Total tokens processed: {self.total_tokens:,} (this step: {tokens_this_step:,})")
                
                # Log to wandb if available
                if wandb.run is not None:
                    wandb.log({
                        "tokens_used": self.total_tokens,
                        "tokens_per_step": tokens_this_step,
                        "step": state.global_step
                    })
                    
        except Exception as e:
            logger.warning(f"Could not track token usage: {e}")
            # Fallback to estimation if dataset access fails
            if hasattr(args, 'per_device_train_batch_size') and hasattr(args, 'max_length'):
                tokens_this_step = args.per_device_train_batch_size * args.max_length * args.gradient_accumulation_steps
                if torch.cuda.device_count() > 1:
                    tokens_this_step *= torch.cuda.device_count()
                
                self.total_tokens += tokens_this_step
                self.step_tokens.append(self.total_tokens)
                
                if state.global_step % args.logging_steps == 0:
                    logger.info(f"Step {state.global_step}: Total tokens processed (estimated): {self.total_tokens:,}")

def setup_model_and_tokenizer(model_name: str):
    """Setup model and tokenizer"""
    logger.info(f"Loading model and tokenizer: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Talking-Babies/opt-tokenizer")
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load model with better precision handling
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    
    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def load_preference_dataset(dataset_path: str):
    """Load the preference dataset"""
    logger.info(f"Loading dataset from: {dataset_path}")
    
    # Load train and test datasets
    train_dataset = load_from_disk(os.path.join(dataset_path, "train"))
    test_dataset = load_from_disk(os.path.join(dataset_path, "test"))
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Print sample to verify format
    logger.info("Sample training example:")
    sample = train_dataset[0]
    for key, value in sample.items():
        if isinstance(value, str) and len(value) > 100:
            logger.info(f"  {key}: {value[:100]}...")
        else:
            logger.info(f"  {key}: {value}")
    
    return train_dataset, test_dataset

def calculate_save_steps(total_steps: int, save_percentage: float = 0.1) -> int:
    """Calculate save steps for checkpointing every 10%"""
    save_steps = max(1, int(total_steps * save_percentage))
    logger.info(f"Will save checkpoint every {save_steps} steps ({save_percentage*100}% of {total_steps} total steps)")
    return save_steps

def main():
    """Main training function"""
    
    # Configuration
    MODEL_NAME = "Talking-Babies/opt-Talking-Babies-train_100M_2048_preprocess"
    DATASET_PATH = "./my_dataset_short"
    OUTPUT_DIR = "./orpo_child_model_output_short"
    
    # Training hyperparameters
    LEARNING_RATE = 5e-6
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    MAX_LENGTH = 512
    NUM_EPOCHS = 2
    WARMUP_RATIO = 0.1
    
    # Setup wandb (optional)
    try:
        wandb.init(
            project="orpo-child-model-training",
            name="talking-babies-orpo",
            config={
                "model_name": MODEL_NAME,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "max_length": MAX_LENGTH,
                "num_epochs": NUM_EPOCHS,
            }
        )
        logger.info("Wandb initialized successfully")
    except Exception as e:
        logger.warning(f"Could not initialize wandb: {e}")
    
    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME)
    
    # Load dataset
    train_dataset, test_dataset = load_preference_dataset(DATASET_PATH)
    
    # Calculate total training steps
    total_steps = (len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) * NUM_EPOCHS
    save_steps = calculate_save_steps(total_steps)
    
    # Calculate total tokens in dataset for reference
    total_dataset_tokens = sum(sample['token_count'] for sample in train_dataset)
    total_training_tokens = total_dataset_tokens * NUM_EPOCHS
    logger.info(f"Total tokens in dataset: {total_dataset_tokens:,}")
    logger.info(f"Total tokens for {NUM_EPOCHS} epochs: {total_training_tokens:,}")
    
    # Setup ORPO configuration
    training_args = ORPOConfig(
        output_dir=OUTPUT_DIR,
        
        # Training parameters
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        max_length=MAX_LENGTH,
        
        # Optimization
        optim="adamw_torch",
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        
        # Logging and saving
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=5,
        
        # ORPO specific
        beta=0.1,  # ORPO regularization parameter
        
        # Memory optimization
        dataloader_drop_last=True,
        remove_unused_columns=False,
        
        # Mixed precision - use bf16 instead of fp16 to avoid gradient scaling issues
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,  # Disable fp16 to avoid gradient unscaling issues
        
        # Gradient clipping
        max_grad_norm=1.0,
        
        # Reporting
        report_to=["wandb"] if wandb.run is not None else [],
        run_name="talking-babies-orpo",
        
    )
    
    # Initialize token usage callback with the training dataset
    token_callback = TokenUsageCallback(train_dataset)
    
    # Create ORPO trainer
    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=[token_callback],
    )
    
    # Print training info
    logger.info("=" * 50)
    logger.info("ORPO Training Configuration")
    logger.info("=" * 50)
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Dataset: {DATASET_PATH}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Evaluation samples: {len(test_dataset)}")
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Checkpoint every: {save_steps} steps ({10}%)")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"Max length: {MAX_LENGTH}")
    logger.info(f"Number of epochs: {NUM_EPOCHS}")
    logger.info(f"Total dataset tokens: {total_dataset_tokens:,}")
    logger.info(f"Expected total training tokens: {total_training_tokens:,}")
    logger.info("=" * 50)
    
    # Start training
    logger.info("Starting ORPO training...")
    try:
        trainer.train()
        logger.info("Training completed successfully!")
        
        # Save final model
        final_output_dir = os.path.join(OUTPUT_DIR, "final_model")
        trainer.save_model(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        logger.info(f"Final model saved to: {final_output_dir}")
        
        # Save token usage statistics
        token_stats = {
            "total_tokens_processed": token_callback.total_tokens,
            "total_steps": len(token_callback.step_tokens),
            "average_tokens_per_step": token_callback.total_tokens / max(1, len(token_callback.step_tokens)),
            "token_progression": token_callback.step_tokens[-10:] if len(token_callback.step_tokens) > 10 else token_callback.step_tokens
        }
        
        with open(os.path.join(OUTPUT_DIR, "token_usage_stats.json"), "w") as f:
            json.dump(token_stats, f, indent=2)
        
        logger.info(f"Token usage statistics saved. Total tokens processed: {token_callback.total_tokens:,}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        # Cleanup wandb
        if wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    main() 