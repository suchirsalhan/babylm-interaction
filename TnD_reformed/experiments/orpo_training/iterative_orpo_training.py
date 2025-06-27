import sys
import json
import os
import re
from typing import List, Dict, Any
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead, ORPOConfig, ORPOTrainer
import nltk
from nltk.tokenize import word_tokenize
import difflib

sys.path.append("../..")
from core.custom_components.custom_interaction.exp1_dialogue_interaction_model import DialogueInteractionModel
from core.custom_components.custom_reward_model.taaco_reward_model import TAACORewardModel

def count_words_nltk(text):
    tokens = word_tokenize(text)
    return len(tokens)

def extract_n_rounds(text, n):
    """Extract n rounds of conversation from the text"""
    turns = re.findall(r"(A::|B::)(.*?)(?=(?:A::|B::)|$)", text, flags=re.DOTALL)
    
    if not turns:
        return "", None
    
    prefix = turns[0][0][0]  # e.g., "A" or "B"
    selected_turns = turns[:2 * n]
    conversation = " ".join(f"{speaker}{utterance.strip()}" for speaker, utterance in selected_turns)
    
    return conversation, prefix

def is_response_repetitive(prompt, response, similarity_threshold=0.8):
    """
    Check if the response is just a repetition of the prompt.
    
    Args:
        prompt: The input prompt
        response: The generated response
        similarity_threshold: Threshold for considering responses as repetitive
        
    Returns:
        True if response is repetitive, False otherwise
    """
    if not response or len(response.strip()) < 10:
        return True
    
    # Check if response is shorter than expected (likely incomplete)
    if len(response.split()) < 5:
        return True
    
    # Check if response starts with the same words as the prompt
    prompt_words = prompt.split()[-20:]  # Last 20 words of prompt
    response_words = response.split()[:20]  # First 20 words of response
    
    if len(response_words) >= 10:
        # Calculate similarity using SequenceMatcher
        similarity = difflib.SequenceMatcher(None, prompt_words, response_words).ratio()
        if similarity > similarity_threshold:
            return True
    
    # Check if response contains large chunks of the prompt
    prompt_suffix = " ".join(prompt.split()[-50:])  # Last 50 words
    if prompt_suffix in response:
        return True
    
    return False

def create_enhanced_interaction_model(child_model, teacher_model, child_tokenizer, teacher_tokenizer, 
                                    child_generation_args, teacher_generation_args):
    """Create an enhanced interaction model with improved teacher prompting."""
    
    # Create the base interaction model
    interaction_model = DialogueInteractionModel(
        child_model=child_model,
        teacher_model=teacher_model,
        child_tokenizer=child_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        student_generation_args=child_generation_args,
        teacher_generation_args=teacher_generation_args,
    )
    
    # Override the teacher prompt formatting with an improved version
    def improved_format_teacher_prompt(partial_dialogue: str, child_continuation: str) -> str:
        """Format an improved prompt for the teacher model that combines original guidance with anti-repetition measures."""
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "You are an expert dialogue completion assistant. You will be given a partial dialogue and a child model's continuation attempt.\n\n"
            "YOUR TASK: Generate a high-quality completion that improves upon the child's attempt.\n\n"
            "IMPROVEMENT GUIDELINES:\n"
            "1. Take reference from the child continuation - improve its coherence, fluency, grammar, and contextual appropriateness\n"
            "2. Keep similar length and topic as the child's attempt, but make it more natural and conversational\n"
            "3. If the child's continuation is gibberish or completely inappropriate, generate your own coherent continuation\n"
            "4. Make your response flow naturally from the partial dialogue\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "- Generate ONLY the next speaker's dialogue continuation - do NOT repeat any part of the provided dialogue\n"
            "- Do NOT include speaker labels (A:: or B::) in your response\n"
            "- Do NOT copy or paraphrase the provided partial dialogue\n"
            "- Keep your response concise (10-30 words) but meaningful\n"
            "- Provide only your completion without commentary or feedback\n"
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"Partial dialogue to continue:\n{partial_dialogue}\n\n"
            f"Child model's continuation attempt: \"{child_continuation}\"\n\n"
            "Now provide an improved continuation that is more coherent, fluent, and contextually appropriate (without speaker labels):\n"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
    
    # Replace the original method
    interaction_model._format_teacher_prompt = improved_format_teacher_prompt
    
    return interaction_model

def generate_preference_data_batch_with_validation(input_texts, interaction_model, taaco_model, 
                                                 batch_size=128, teacher_tokenizer=None, 
                                                 child_tokenizer=None, add_score=False, 
                                                 max_retries=2):
    """
    Enhanced preference data generation with validation and retry logic.
    """
    print(f"Processing {len(input_texts)} samples with validation...")
    
    all_preference_data = []
    
    # Process in batches
    for batch_idx in range(0, len(input_texts), batch_size):
        batch_texts = input_texts[batch_idx:batch_idx+batch_size]
        
        print(f"Processing batch {batch_idx//batch_size + 1} ({len(batch_texts)} samples)")
        
        # Generate interactions for this batch
        retry_count = 0
        valid_results = []
        
        while retry_count <= max_retries:
            try:
                # Generate batch interactions
                batch_results = interaction_model.batch_interact(batch_texts)
                
                # Validate results and collect valid ones
                valid_batch_results = {
                    'partial_dialogues': [],
                    'child_continuations': [],
                    'teacher_completions': [],
                    'decoded_teacher_responses': []
                }
                
                invalid_indices = []
                
                for j in range(len(batch_texts)):
                    prompt = batch_results['partial_dialogues'][j]
                    teacher_completion = batch_results['teacher_completions'][j]
                    child_continuation = batch_results['child_continuations'][j]
                    teacher_decoded = batch_results['decoded_teacher_responses'][j]
                    
                    # Validate teacher response
                    if is_response_repetitive(prompt, teacher_completion):
                        print(f"  Invalid teacher response detected for sample {j}, marking for retry")
                        invalid_indices.append(j)
                    else:
                        valid_batch_results['partial_dialogues'].append(prompt)
                        valid_batch_results['child_continuations'].append(child_continuation)
                        valid_batch_results['teacher_completions'].append(teacher_completion)
                        valid_batch_results['decoded_teacher_responses'].append(teacher_decoded)
                
                # If we have enough valid results or reached max retries, break
                valid_ratio = len(valid_batch_results['partial_dialogues']) / len(batch_texts)
                print(f"  Valid responses: {len(valid_batch_results['partial_dialogues'])}/{len(batch_texts)} ({valid_ratio:.1%})")
                
                if valid_ratio >= 0.7 or retry_count >= max_retries:
                    # Use valid results
                    for i in range(len(valid_batch_results['partial_dialogues'])):
                        row_dict = {
                            'prompt': valid_batch_results['partial_dialogues'][i],
                            'chosen': valid_batch_results['teacher_completions'][i],
                            'decoded_chosen': valid_batch_results['decoded_teacher_responses'][i],
                            'rejected': valid_batch_results['child_continuations'][i],
                            'token_count': count_words_nltk(valid_batch_results['partial_dialogues'][i]),
                        }
                        
                        if add_score:
                            teacher_whole = row_dict['prompt'] + row_dict['chosen']
                            child_whole = row_dict['prompt'] + row_dict['rejected']
                            
                            try:
                                teacher_score = taaco_model.compute_individual_scores([teacher_whole], teacher_tokenizer)[0]
                                child_score = taaco_model.compute_individual_scores([child_whole], child_tokenizer)[0]
                                
                                row_dict['score_chosen'] = teacher_score
                                row_dict['score_rejected'] = child_score
                                row_dict['valid'] = teacher_score > child_score
                            except Exception as e:
                                print(f"  Error computing scores: {e}")
                                row_dict['score_chosen'] = 0.5
                                row_dict['score_rejected'] = 0.5
                                row_dict['valid'] = True
                        
                        all_preference_data.append(row_dict)
                    
                    break
                else:
                    retry_count += 1
                    print(f"  Retrying batch due to low valid ratio (attempt {retry_count}/{max_retries})")
                    
            except Exception as e:
                print(f"  Error in batch processing: {e}")
                retry_count += 1
                if retry_count > max_retries:
                    print(f"  Failed to process batch after {max_retries} retries, skipping...")
                    break
    
    print(f"Generated {len(all_preference_data)} valid preference pairs")
    return all_preference_data

class IterativeORPOTrainer:
    def __init__(self, 
                 initial_child_model_name,
                 child_tokenizer_name,
                 teacher_model_name,
                 output_dir,
                 final_json_path,
                 child_generation_args,
                 teacher_generation_args,
                 sample_vars,
                 training_config):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.final_json_path = final_json_path
        self.all_training_data = []
        self.child_generation_args = child_generation_args
        self.teacher_generation_args = teacher_generation_args
        self.training_config = training_config
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load initial child model and tokenizer
        self.child_tokenizer = AutoTokenizer.from_pretrained(child_tokenizer_name)
        self.child_tokenizer.pad_token = self.child_tokenizer.eos_token
        self.child_tokenizer.padding_side = "left"
        
        # Keep track of current child model name for reloading
        self.current_child_model_name = initial_child_model_name
        
        # Load teacher model (stays constant)
        self.teacher_model = AutoModelForCausalLMWithValueHead.from_pretrained(teacher_model_name)
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
        self.teacher_tokenizer.padding_side = "left"
        self.teacher_model.to(self.device)
        
        # Initialize reward model with provided sample_vars
        self.taaco_reward_model = TAACORewardModel(taaco_vars=sample_vars)
        
        # Load dataset
        print("Loading dataset...")
        self.dataset = load_dataset("Talking-Babies/annotated_switchboard_v1", split="train")
        self.dataset_size = len(self.dataset)
        print(f"Dataset loaded with {self.dataset_size} samples")
    
    def load_child_model(self):
        """Load/reload the child model"""
        print(f"Loading child model: {self.current_child_model_name}")
        child_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.current_child_model_name)
        child_model.to(self.device)
        return child_model
    
    def create_interaction_model(self, child_model):
        """Create enhanced interaction model with current child model"""
        return create_enhanced_interaction_model(
            child_model=child_model,
            teacher_model=self.teacher_model,
            child_tokenizer=self.child_tokenizer,
            teacher_tokenizer=self.teacher_tokenizer,
            child_generation_args=self.child_generation_args,
            teacher_generation_args=self.teacher_generation_args,
        )
    
    def generate_batch_data(self, start_idx: int, end_idx: int, interaction_model):
        """Generate preference data for a batch of the dataset"""
        print(f"Generating preference data for samples {start_idx} to {end_idx}")
        
        input_texts = []
        for i in range(start_idx, min(end_idx, self.dataset_size)):
            conv, prefix = extract_n_rounds(self.dataset[i]['text'], 1)
            input_texts.append(conv + '\n' + prefix + '::')
        
        if not input_texts:
            return []
        
        # Generate preference data with validation
        pref_data = generate_preference_data_batch_with_validation(
            input_texts, 
            interaction_model, 
            self.taaco_reward_model, 
            batch_size=64,  # Smaller batch size for better quality control
            teacher_tokenizer=self.teacher_tokenizer,
            child_tokenizer=self.child_tokenizer,
            add_score=False,
            max_retries=2
        )
        
        return pref_data
    
    def train_orpo_iteration(self, iteration: int, train_data: List[Dict]):
        """Train ORPO model for one iteration"""
        print(f"Starting ORPO training iteration {iteration}")
        
        if not train_data:
            print("No training data available, skipping training")
            return None
        
        # Convert to HuggingFace dataset
        dataset_dict = {
            'prompt': [item['prompt'] for item in train_data],
            'chosen': [item['chosen'] for item in train_data], 
            'rejected': [item['rejected'] for item in train_data]
        }
        
        hf_dataset = Dataset.from_dict(dataset_dict)
        
        # Load model for training (using regular AutoModelForCausalLM for ORPO)
        print(f"Loading model for training: {self.current_child_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            self.current_child_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
        )
        
        # Setup ORPO training with provided config
        iteration_output_dir = os.path.join(self.output_dir, f"iteration_{iteration}")
        training_args = ORPOConfig(
            output_dir=iteration_output_dir,
            **self.training_config
        )
        
        trainer = ORPOTrainer(
            model=model, 
            args=training_args, 
            processing_class=self.child_tokenizer, 
            train_dataset=hf_dataset
        )
        
        # Train
        trainer.train()
        
        # Save the trained model
        trainer.save_model()
        
        # Update current model name for next iteration
        self.current_child_model_name = iteration_output_dir
        
        # Clean up GPU memory
        del model
        del trainer
        torch.cuda.empty_cache()
        
        return iteration_output_dir
    
    def run_iterative_training(self, num_iterations: int = 5):
        """Run the full iterative training process"""
        print(f"Starting iterative ORPO training with {num_iterations} iterations")
        print(f"Dataset size: {self.dataset_size}")
        
        batch_size = self.dataset_size // num_iterations
        
        for iteration in range(num_iterations):
            print(f"\n{'='*50}")
            print(f"ITERATION {iteration + 1}/{num_iterations}")
            print(f"{'='*50}")
            
            # Calculate batch indices
            start_idx = iteration * batch_size
            end_idx = (iteration + 1) * batch_size if iteration < num_iterations - 1 else self.dataset_size
            
            print(f"Processing samples {start_idx} to {end_idx} ({end_idx - start_idx} samples)")
            
            # Load current child model
            child_model = self.load_child_model()
            
            # Create interaction model
            interaction_model = self.create_interaction_model(child_model)
            
            # Generate preference data for this batch
            batch_data = self.generate_batch_data(start_idx, end_idx, interaction_model)
            
            # Add to accumulated training data
            self.all_training_data.extend(batch_data)
            
            print(f"Generated {len(batch_data)} preference samples")
            print(f"Total accumulated samples: {len(self.all_training_data)}")
            
            # Clean up interaction model
            del child_model
            del interaction_model
            torch.cuda.empty_cache()
            
            # Train ORPO on all accumulated data
            trained_model_path = self.train_orpo_iteration(iteration + 1, self.all_training_data)
            
            # Save intermediate progress
            intermediate_json_path = os.path.join(self.output_dir, f"training_data_iteration_{iteration + 1}.json")
            with open(intermediate_json_path, 'w') as f:
                json.dump(self.all_training_data, f, indent=2)
            
            print(f"Saved intermediate training data to {intermediate_json_path}")
            print(f"Trained model saved to {trained_model_path}")
        
        # Save final training data
        print(f"\nSaving final training data to {self.final_json_path}")
        with open(self.final_json_path, 'w') as f:
            json.dump(self.all_training_data, f, indent=2)
        
        print(f"Iterative training completed!")
        print(f"Final model: {self.current_child_model_name}")
        print(f"Total training samples: {len(self.all_training_data)}")
        print(f"Final training data saved to: {self.final_json_path}")

def main():
    # Model Configuration
    model_config = {
        "initial_child_model_name": "Talking-Babies/opt-base",
        "child_tokenizer_name": "Talking-Babies/opt-tokenizer", 
        "teacher_model_name": "meta-llama/Llama-3.2-3B-Instruct",
    }
    
    # Output Configuration
    output_config = {
        "output_dir": "iterative_orpo_training_3b_results",
        "final_json_path": "final_iterative_training_3b_data.json",
        "num_iterations": 5
    }
    
    # Child Generation Arguments
    child_generation_args = {
        "max_new_tokens": 50,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.8,
        "num_return_sequences": 1,
    }
    
    # Teacher Generation Arguments
    teacher_generation_args = {
        "max_new_tokens": 50,
        "do_sample": False,
        "temperature": 0.3,  # Lower temperature for more focused responses
    }
    
    # TAACO Reward Model Sample Variables
    sample_vars = {
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
    
    # ORPO Training Configuration
    training_config = {
        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 500,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 1,
        "learning_rate": 5e-6,
        "max_length": 512,
        "max_prompt_length": 256,
        "warmup_steps": 100,
        "lr_scheduler_type": "cosine",
        "save_total_limit": 2,
    }
    
    # Create and run iterative trainer
    trainer = IterativeORPOTrainer(
        initial_child_model_name=model_config["initial_child_model_name"],
        child_tokenizer_name=model_config["child_tokenizer_name"],
        teacher_model_name=model_config["teacher_model_name"],
        output_dir=output_config["output_dir"],
        final_json_path=output_config["final_json_path"],
        child_generation_args=child_generation_args,
        teacher_generation_args=teacher_generation_args,
        sample_vars=sample_vars,
        training_config=training_config
    )
    
    trainer.run_iterative_training(num_iterations=output_config["num_iterations"])

if __name__ == "__main__":
    main() 