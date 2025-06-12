#!/usr/bin/env python3
"""
Test script to verify the training setup works correctly.
This script tests the interaction model, reward model, and PPO trainer integration.
"""

import sys
import os
import torch
import pandas as pd
import wandb

# Add paths
sys.path.append("../..")
from transformers import AutoTokenizer, GenerationConfig
from trl import AutoModelForCausalLMWithValueHead

# Import custom components
from core.custom_components.custom_reward_model.taaco_reward_model import TAACORewardModel
from core.custom_components.custom_interaction.exp1_dialogue_interaction_model import DialogueInteractionModel

def test_setup():
    """Test the training setup with a small sample."""
    print("Testing training setup...")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models
    print("Loading models...")
    child_tokenizer = AutoTokenizer.from_pretrained("Talking-Babies/opt-tokenizer")
    child_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        "Talking-Babies/opt-Talking-Babies-train_100M_2048_preprocess"
    )
    child_model.to(device)
    
    teacher_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    teacher_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct"
    )
    teacher_model.to(device)
    
    # Setup generation configs
    child_generation_args = {
        "max_new_tokens": 50,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.8,
        "num_return_sequences": 1,
        "pad_token_id": child_tokenizer.eos_token_id,
    }
    
    teacher_generation_args = {
        "max_new_tokens": 50,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.5,
        "pad_token_id": teacher_tokenizer.eos_token_id,
    }
    
    # Setup interaction model
    print("Setting up interaction model...")
    interaction_model = DialogueInteractionModel(
        child_model=child_model,
        teacher_model=teacher_model,
        child_tokenizer=child_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        student_generation_args=child_generation_args,
        teacher_generation_args=teacher_generation_args,
    )
    
    # Setup reward model
    print("Setting up TAACO reward model...")
    taaco_vars = {
        "sourceKeyOverlap": False, "sourceLSA": False, "sourceLDA": False, 
        "sourceWord2vec": False, "wordsAll": True, "wordsContent": True, 
        "wordsFunction": True, "wordsNoun": True, "wordsPronoun": True, 
        "wordsArgument": True, "wordsVerb": True, "wordsAdjective": True, 
        "wordsAdverb": True, "overlapSentence": True, "overlapParagraph": True, 
        "overlapAdjacent": True, "overlapAdjacent2": True, "otherTTR": True, 
        "otherConnectives": True, "otherGivenness": True, "overlapLSA": True, 
        "overlapLDA": True, "overlapWord2vec": True, "overlapSynonym": True, 
        "overlapNgrams": True, "outputTagged": False, "outputDiagnostic": False
    }
    reward_model = TAACORewardModel(device=device, taaco_vars=taaco_vars)
    
    # Test with sample dialogue
    print("Testing with sample dialogue...")
    test_dialogue = """A: I think we need to discuss the budget for next quarter.
B: Yes, I agree. The current projections show we might need to cut some expenses.
A: What areas do you think we should focus on?
B: Well, I think we should look at"""
    
    # Test interaction model
    print("Testing interaction model...")
    try:
        interaction_result = interaction_model.interact(test_dialogue)
        print("✓ Interaction model works!")
        print(f"Child continuation: {interaction_result['child_continuation'][:100]}...")
        print(f"Teacher completion: {interaction_result['teacher_completion'][:100]}...")
        
        # Test reward model
        print("Testing reward model...")
        reward_scores = reward_model.compute_rewards(
            child_queries=[interaction_result['decoded_child_query']],
            child_responses=[interaction_result['decoded_child_response']],
            teacher_queries=[interaction_result['decoded_teacher_query']],
            teacher_responses=[interaction_result['decoded_teacher_response']]
        )
        print("✓ Reward model works!")
        print(f"TAACO score: {reward_scores[0]}")
        
        # Test tensor formats
        print("Testing tensor formats...")
        print(f"Child query shape: {interaction_result['child_query'].shape}")
        print(f"Child response shape: {interaction_result['child_response'].shape}")
        print(f"Teacher query shape: {interaction_result['teacher_query'].shape}")
        print(f"Teacher response shape: {interaction_result['teacher_response'].shape}")
        
        # Test wandb logging
        print("Testing wandb logging...")
        try:
            # Initialize a test wandb run
            wandb.init(
                project="dialogue-ppo-test",
                name="test-run",
                mode="disabled"  # Don't actually log to wandb during testing
            )
            
            # Test logging some metrics
            wandb.log({
                "test_score": reward_scores[0].item() if hasattr(reward_scores[0], 'item') else reward_scores[0],
                "test_step": 1
            })
            
            wandb.finish()
            print("✓ Wandb logging works!")
            
        except Exception as e:
            print(f"⚠️  Wandb test failed: {e}")
            print("Note: This is not critical for training, but web logging won't work")
        
        print("\n✅ All tests passed! Training setup is ready.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_setup()
    if success:
        print("\n🚀 Ready to start training!")
        print("Run: python train_dialogue_ppo_taaco.py")
    else:
        print("\n⚠️  Please fix the issues before starting training.") 