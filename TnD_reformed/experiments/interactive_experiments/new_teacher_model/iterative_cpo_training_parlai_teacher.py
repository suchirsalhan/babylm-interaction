import sys
import json
import os
import re
from typing import List, Dict, Any
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead, CPOConfig, CPOTrainer
import nltk
from nltk.tokenize import word_tokenize
import difflib
import wandb
import time  # Add timing import

# Add ParlAI paths
sys.path.append("../ParlAI")
sys.path.append("../ControllableComplexityChatbot")

# Import ParlAI components
from parlai.zoo.blender.blender_3B import download
from parlai.core.opt import Opt
from parlai.core.message import Message
from controllable_blender import ControllableBlender

sys.path.append("/workspace/babylm-interaction/TnD_reformed")
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

class ParlAITeacherWrapper:
    """Wrapper to make ParlAI ControllableBlender compatible with the interaction model"""
    
    def __init__(self, parlai_agent):
        self.agent = parlai_agent
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 ParlAI Teacher initialized on device: {self.device}")
        
    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Generate method compatible with HuggingFace interface"""
        # This is a dummy implementation since we'll handle generation differently
        # The actual generation will be done through the act() method
        return input_ids  # Just return input to avoid errors
    
    def to(self, device):
        """Compatibility method"""
        return self
    
    def act_on_text(self, text):
        """Generate response using ParlAI agent"""
        self.agent.reset()
        obs = {'text': text, 'episode_done': False}
        self.agent.observe(obs)
        reply = self.agent.act()
        return reply.get('text', '')
    
    def batch_act_on_texts(self, texts):
        """Generate responses for multiple texts using ParlAI batch processing"""
        try:
            # Create Message objects for batch processing
            messages = []
            for i, text in enumerate(texts):
                message = Message({'text': text, 'episode_done': False})
                messages.append(message)
            
            print(f"🔧 Created {len(messages)} Message objects for ParlAI batch processing")
            
            # Use ParlAI's batch_respond convenience method
            # This handles all the complexity of observe, batch_act, and self_observe
            responses = self.agent.batch_respond(messages)
            
            print(f"🔧 Successfully got {len(responses)} responses from ParlAI batch_respond")
            return responses
            
        except Exception as e:
            print(f"❌ ERROR in ParlAI batch_act_on_texts: {type(e).__name__}: {str(e)}")
            print(f"❌ Error details: {repr(e)}")
            import traceback
            print(f"❌ Full traceback:")
            traceback.print_exc()
            
            # Fallback to individual processing if batch fails
            print(f"🔄 Falling back to individual processing for {len(texts)} texts...")
            responses = []
            for i, text in enumerate(texts):
                try:
                    response = self.act_on_text(text)
                    responses.append(response)
                    if i % 10 == 0:
                        print(f"  Individual processing: {i+1}/{len(texts)}")
                except Exception as individual_error:
                    print(f"❌ Individual processing error for text {i+1}: {individual_error}")
                    responses.append("")  # Empty response for failed individual processing
            
            print(f"🔄 Fallback processing completed: {len(responses)} responses")
            return responses

def create_parlai_teacher_model(cefr_level="B2"):
    """Create and configure the ParlAI ControllableBlender teacher model"""
    
    # Change to the ControllableComplexityChatbot directory
    original_cwd = os.getcwd()
    parlai_dir = "/workspace/babylm-interaction/TnD_reformed/experiments/interactive_experiments/ControllableComplexityChatbot"
    os.chdir(parlai_dir)
    
    try:
        print(f"🔧 Loading ParlAI teacher configuration with CEFR level: {cefr_level}")
        
        # Load agent configuration
        agent_opt = json.load(open("blender_3B.opt", 'r'))
        
        # Configure for rerank inference
        agent_opt["inference"] = "rerank"
        agent_opt["beam_size"] = 4             
        agent_opt["topk"] = 40
        agent_opt["topp"] = 0.9
        agent_opt["temperature"] = 0.7
        
        # Settings for rerank methods - use ABSOLUTE paths to avoid cloning issues
        agent_opt["rerank_cefr"] = cefr_level  # Use the provided CEFR level
        agent_opt["rerank_tokenizer"] = "distilroberta-base"
        agent_opt["rerank_model"] = os.path.join(parlai_dir, "complexity_model")  # Absolute path
        agent_opt["rerank_model_device"] = "cuda"
        agent_opt["penalty_stddev"] = 2
        agent_opt["filter_path"] = os.path.join(parlai_dir, "data", "filter.txt")  # Absolute path
        agent_opt["wordlist_path"] = os.path.join(parlai_dir, "data", "sample_wordlist.txt")  # Absolute path
        
        # Force GPU usage for the main model
        if torch.cuda.is_available():
            agent_opt["gpu"] = 0  # Use first GPU
            print(f"🔧 Setting ParlAI to use GPU 0")
        else:
            print("⚠️  CUDA not available, ParlAI will use CPU")
        
        # Verify that the required files exist
        required_files = [
            agent_opt["filter_path"],
            agent_opt["wordlist_path"],
            agent_opt["rerank_model"]
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"⚠️  Warning: Required file/directory not found: {file_path}")
            else:
                print(f"✅ Found required file/directory: {file_path}")
        
        # Download model if needed
        print("🔧 Downloading ParlAI model if needed...")
        download(agent_opt["datapath"])
        
        # Create the agent
        print(f"🔧 Creating ControllableBlender agent with CEFR level {cefr_level}...")
        agent_creation_start = time.time()
        agent = ControllableBlender(agent_opt)
        agent_creation_end = time.time()
        print(f"🕐 ParlAI agent creation took: {agent_creation_end - agent_creation_start:.2f}s")
        
        # Check GPU memory after loading
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
            gpu_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"🔧 GPU Memory after ParlAI loading:")
            print(f"  Total: {gpu_memory:.1f}GB")
            print(f"  Allocated: {gpu_allocated:.1f}GB")
            print(f"  Reserved: {gpu_reserved:.1f}GB")
        
        # Wrap it for compatibility
        teacher_wrapper = ParlAITeacherWrapper(agent)
        
        return teacher_wrapper
        
    finally:
        # Change back to original directory
        os.chdir(original_cwd)

class ParlAIDialogueInteractionModel:
    """Custom interaction model that uses ParlAI teacher"""
    
    def __init__(self, child_model, teacher_wrapper, child_tokenizer, child_generation_args):
        self.child_model = child_model
        self.teacher_wrapper = teacher_wrapper
        self.child_tokenizer = child_tokenizer
        self.child_generation_args = child_generation_args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def batch_interact(self, input_texts):
        """Generate interactions for a batch of input texts using proper batch processing"""
        print(f"🔧 Starting BATCH interaction with {len(input_texts)} texts")
        print(f"🔧 Child model device: {self.device}")
        print(f"🔧 Child model on GPU: {next(self.child_model.parameters()).is_cuda}")
        
        batch_start_time = time.time()
        
        # Step 1: Batch process child model (like in original DialogueInteractionModel)
        print("🔧 Step 1: Batch processing child model...")
        child_batch_start = time.time()
        
        # Tokenize all child prompts with automatic padding
        child_batch_encoding = self.child_tokenizer(
            input_texts,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate child responses in batch
        with torch.no_grad():
            child_batch_outputs = self.child_model.generate(
                input_ids=child_batch_encoding['input_ids'],
                attention_mask=child_batch_encoding['attention_mask'],
                **self.child_generation_args,
                pad_token_id=self.child_tokenizer.pad_token_id
            )
        
        # Decode child responses and extract continuations
        child_continuations = []
        for i, (child_response, original_text) in enumerate(zip(child_batch_outputs, input_texts)):
            full_child_response = self.child_tokenizer.decode(child_response, skip_special_tokens=True)
            child_continuation = full_child_response[len(original_text):].strip()
            child_continuations.append(child_continuation)
        
        child_batch_end = time.time()
        child_batch_time = child_batch_end - child_batch_start
        print(f"🕐 Child batch processing took: {child_batch_time:.3f}s ({child_batch_time/len(input_texts):.3f}s per sample)")
        
        # Step 2: Batch process teacher model using ParlAI batch_act
        print("🔧 Step 2: Batch processing ParlAI teacher model...")
        teacher_batch_start = time.time()
        
        try:
            # Create teacher prompts with reference to child continuations
            teacher_inputs = []
            for i, text in enumerate(input_texts):
                teacher_input = f"{text}\nChild response: {child_continuations[i]}\nPlease give Improved response based on the child response:"
                teacher_inputs.append(teacher_input)
            
            print(f"🔧 Created {len(teacher_inputs)} teacher input prompts")
            
            # Use ParlAI batch processing
            teacher_responses = self.teacher_wrapper.batch_act_on_texts(teacher_inputs)
            
            print(f"🔧 Successfully got {len(teacher_responses)} teacher responses")
            
        except Exception as e:
            print(f"❌ ERROR in teacher batch processing: {type(e).__name__}: {str(e)}")
            print(f"❌ Error details: {repr(e)}")
            import traceback
            print(f"❌ Full traceback:")
            traceback.print_exc()
            
            # Create empty responses to maintain batch consistency
            print(f"🔄 Creating empty teacher responses to maintain batch consistency")
            teacher_responses = [""] * len(input_texts)
        
        teacher_batch_end = time.time()
        teacher_batch_time = teacher_batch_end - teacher_batch_start
        print(f"🕐 Teacher batch processing took: {teacher_batch_time:.3f}s ({teacher_batch_time/len(input_texts):.3f}s per sample)")
        
        # Prepare results
        results = {
            'partial_dialogues': input_texts,
            'child_continuations': child_continuations,
            'teacher_completions': teacher_responses,
            'decoded_teacher_responses': teacher_responses
        }
        
        batch_end_time = time.time()
        total_batch_time = batch_end_time - batch_start_time
        
        # Print detailed timing statistics
        print(f"🕐 BATCH TIMING SUMMARY for {len(input_texts)} samples:")
        print(f"  Total batch time: {total_batch_time:.2f}s")
        print(f"  Child batch time: {child_batch_time:.2f}s ({child_batch_time/total_batch_time*100:.1f}%)")
        print(f"  Teacher batch time: {teacher_batch_time:.2f}s ({teacher_batch_time/total_batch_time*100:.1f}%)")
        print(f"  Average time per sample: {total_batch_time/len(input_texts):.3f}s")
        print(f"  🚀 Speedup vs individual processing: ~{len(input_texts):.1f}x expected")
        print(f"  Estimated time for 3578 samples: {(total_batch_time/len(input_texts)) * 3578 / 60:.1f} minutes")
        
        return results

def create_enhanced_interaction_model_with_parlai(child_model, teacher_wrapper, child_tokenizer, 
                                                 child_generation_args, device):
    """Create an enhanced interaction model with ParlAI teacher."""
    
    return ParlAIDialogueInteractionModel(child_model, teacher_wrapper, child_tokenizer, child_generation_args)

def generate_preference_data_batch_with_validation(input_texts, interaction_model, taaco_model, 
                                                 batch_size=128, teacher_tokenizer=None, 
                                                 child_tokenizer=None, add_score=False, 
                                                 max_retries=2, iteration=0):
    """
    Enhanced preference data generation with validation and retry logic.
    """
    print(f"Processing {len(input_texts)} samples with validation...")
    
    total_batch_size = len(input_texts)/batch_size
    
    all_preference_data = []
    
    # Process in batches
    for batch_idx in range(0, len(input_texts), batch_size):
        batch_texts = input_texts[batch_idx:batch_idx+batch_size]
        
        print(f"Processing batch {batch_idx//batch_size + 1} ({len(batch_texts)} samples), total batch: {total_batch_size}, iteration {iteration}")
        
        # Generate interactions for this batch
        retry_count = 0
        valid_results = []
        
        while retry_count <= max_retries:
            try:
                # Generate batch interactions
                print(f"  🔧 Attempting batch interaction (attempt {retry_count + 1}/{max_retries + 1})")
                batch_results = interaction_model.batch_interact(batch_texts)
                print(f"  ✅ Batch interaction completed successfully")
                
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
                                # Note: For ParlAI teacher, we don't have a tokenizer, so we'll use child_tokenizer
                                teacher_score = taaco_model.compute_individual_scores([teacher_whole], child_tokenizer)[0]
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
                print(f"❌ ERROR in batch processing (attempt {retry_count + 1}): {type(e).__name__}: {str(e)}")
                print(f"❌ Error details: {repr(e)}")
                import traceback
                print(f"❌ Full traceback:")
                traceback.print_exc()
                
                retry_count += 1
                if retry_count > max_retries:
                    print(f"❌ Failed to process batch after {max_retries + 1} attempts, skipping batch...")
                    break
                else:
                    print(f"🔄 Retrying batch processing (attempt {retry_count + 1}/{max_retries + 1})...")
    
    print(f"Generated {len(all_preference_data)} valid preference pairs")
    return all_preference_data

class IterativeCPOTrainerWithParlAI:
    def __init__(self, 
                 initial_child_model_name,
                 child_tokenizer_name,
                 output_dir,
                 final_json_path,
                 child_generation_args,
                 sample_vars,
                 training_config):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.final_json_path = final_json_path
        self.all_training_data = []
        self.child_generation_args = child_generation_args
        self.training_config = training_config
        
        # Define progressive CEFR levels for each iteration
        self.cefr_levels = ["C2", "C1", "B2", "B1", "A2"]
        print(f"🎓 Progressive CEFR curriculum: {' → '.join(self.cefr_levels)}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load initial child model and tokenizer
        self.child_tokenizer = AutoTokenizer.from_pretrained(child_tokenizer_name)
        self.child_tokenizer.pad_token = self.child_tokenizer.eos_token
        self.child_tokenizer.padding_side = "left"
        
        # Keep track of current child model name for reloading
        self.current_child_model_name = initial_child_model_name
        
        # Note: Teacher model will be created per iteration with different CEFR levels
        self.teacher_wrapper = None
        
        # Initialize reward model with provided sample_vars
        self.taaco_reward_model = TAACORewardModel(taaco_vars=sample_vars)
        
        # Load dataset
        print("Loading dataset...")
        self.dataset = load_dataset("Talking-Babies/annotated_switchboard_v1", split="train")
        self.dataset_size = len(self.dataset)
        print(f"Dataset loaded with {self.dataset_size} samples")
    
    def create_teacher_model_for_iteration(self, iteration):
        """Create teacher model with appropriate CEFR level for the current iteration"""
        cefr_level = self.cefr_levels[iteration - 1]  # iteration is 1-based
        print(f"🎓 Creating teacher model for iteration {iteration} with CEFR level: {cefr_level}")
        
        # Clean up previous teacher model if it exists
        if self.teacher_wrapper is not None:
            print("🧹 Cleaning up previous teacher model...")
            del self.teacher_wrapper
            torch.cuda.empty_cache()
        
        # Create new teacher model with the appropriate CEFR level
        self.teacher_wrapper = create_parlai_teacher_model(cefr_level=cefr_level)
        print(f"✅ Teacher model created successfully for CEFR level: {cefr_level}")
        
        return self.teacher_wrapper
    
    def load_child_model(self):
        """Load/reload the child model"""
        print(f"Loading child model: {self.current_child_model_name}")
        
        # Check GPU memory before loading
        if torch.cuda.is_available():
            gpu_allocated_before = torch.cuda.memory_allocated(0) / 1024**3
            gpu_reserved_before = torch.cuda.memory_reserved(0) / 1024**3
            print(f"🔧 GPU Memory before child model loading:")
            print(f"  Allocated: {gpu_allocated_before:.1f}GB")
            print(f"  Reserved: {gpu_reserved_before:.1f}GB")
        
        child_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.current_child_model_name)
        child_model.to(self.device)
        
        # Check GPU memory after loading
        if torch.cuda.is_available():
            gpu_allocated_after = torch.cuda.memory_allocated(0) / 1024**3
            gpu_reserved_after = torch.cuda.memory_reserved(0) / 1024**3
            print(f"🔧 GPU Memory after child model loading:")
            print(f"  Allocated: {gpu_allocated_after:.1f}GB (+{gpu_allocated_after-gpu_allocated_before:.1f}GB)")
            print(f"  Reserved: {gpu_reserved_after:.1f}GB (+{gpu_reserved_after-gpu_reserved_before:.1f}GB)")
        
        return child_model
    
    def create_interaction_model(self, child_model, device):
        """Create enhanced interaction model with current child model and ParlAI teacher"""
        return create_enhanced_interaction_model_with_parlai(
            child_model=child_model,
            teacher_wrapper=self.teacher_wrapper,
            child_tokenizer=self.child_tokenizer,
            child_generation_args=self.child_generation_args,
            device=device
        )
    
    def generate_batch_data(self, start_idx: int, end_idx: int, interaction_model, iteration: int):
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
            batch_size=64,  # Increased batch size for better performance with batch processing
            teacher_tokenizer=None,  # ParlAI doesn't use tokenizer directly
            child_tokenizer=self.child_tokenizer,
            add_score=False,
            max_retries=2,
            iteration=iteration,
        )
        
        return pref_data
    
    def train_cpo_iteration(self, iteration: int, train_data: List[Dict]):
        """Train CPO model for one iteration"""
        print(f"Starting CPO training iteration {iteration}")
        
        # Clear GPU memory before training
        torch.cuda.empty_cache()
        
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
        
        # Load model for training (using regular AutoModelForCausalLM for CPO)
        print(f"Loading model for training: {self.current_child_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            self.current_child_model_name,
            torch_dtype=torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        
        # Setup CPO training with provided config
        iteration_output_dir = os.path.join(self.output_dir, f"iteration_{iteration}")
        
        # Create unique run name for this iteration
        unique_run_name = f"cpo_parlai_iter_{iteration}_{os.path.basename(self.output_dir)}"
        
        # Ensure wandb settings don't get overridden
        training_config_copy = self.training_config.copy()
        training_config_copy.update({
            "run_name": unique_run_name,
            "report_to": ["wandb"],
            "logging_dir": os.path.join(iteration_output_dir, "logs"),
            "logging_first_step": True,
        })
        
        print(f"🔍 Wandb run name for iteration {iteration}: {unique_run_name}")
        print(f"🔍 Report to: {training_config_copy.get('report_to')}")
        
        training_args = CPOConfig(
            output_dir=iteration_output_dir,
            **training_config_copy
        )
        
        trainer = CPOTrainer(
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
        
        # Clean up GPU memory more thoroughly
        del model
        del trainer  
        del hf_dataset
        torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        return iteration_output_dir
    
    def run_iterative_training(self, num_iterations: int = 5):
        """Run the full iterative training process"""
        print(f"Starting iterative CPO training with ParlAI teacher - {num_iterations} iterations")
        print(f"🎓 CEFR Curriculum Learning: {' → '.join(self.cefr_levels[:num_iterations])}")
        print(f"Dataset size: {self.dataset_size}")
        
        batch_size = self.dataset_size // num_iterations
        
        for iteration in range(num_iterations):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1}/{num_iterations}")
            cefr_level = self.cefr_levels[iteration]
            print(f"🎓 CEFR LEVEL: {cefr_level}")
            print(f"{'='*60}")
            
            # Calculate batch indices
            start_idx = iteration * batch_size
            end_idx = (iteration + 1) * batch_size if iteration < num_iterations - 1 else self.dataset_size
            
            print(f"Processing samples {start_idx} to {end_idx} ({end_idx - start_idx} samples)")
            
            # Create teacher model with appropriate CEFR level for this iteration
            print(f"🎓 Step 1: Creating teacher model with CEFR level {cefr_level}...")
            self.create_teacher_model_for_iteration(iteration + 1)
            
            # Load current child model
            print(f"🔧 Step 2: Loading child model...")
            child_model = self.load_child_model()
            
            # Create interaction model
            print(f"🔧 Step 3: Creating interaction model...")
            interaction_model = self.create_interaction_model(child_model, self.device)
            
            # Generate preference data for this batch
            print(f"🔧 Step 4: Generating preference data...")
            batch_data = self.generate_batch_data(start_idx, end_idx, interaction_model, iteration)
            
            # Store all data but train only on current batch
            self.all_training_data.extend(batch_data)
            
            print(f"Generated {len(batch_data)} preference samples with CEFR level {cefr_level}")
            print(f"Total accumulated samples: {len(self.all_training_data)}")
            print(f"🎯 Training on current batch only: {len(batch_data)} samples (avoiding overfitting)")
            
            # Clean up interaction model
            del child_model
            del interaction_model
            torch.cuda.empty_cache()
            
            # Ensure wandb creates a new run for this iteration
            try:
                wandb.finish()
            except:
                pass
            
            print(f"🔄 Reinitializing wandb for iteration {iteration + 1}")
            
            trained_model_path = self.train_cpo_iteration(iteration + 1, batch_data)
            
            # Save intermediate progress with CEFR level information
            intermediate_json_path = os.path.join(self.output_dir, f"training_data_iteration_{iteration + 1}_cefr_{cefr_level}.json")
            
            # Add CEFR level metadata to the training data
            batch_data_with_metadata = []
            for item in batch_data:
                item_with_metadata = item.copy()
                item_with_metadata['cefr_level'] = cefr_level
                item_with_metadata['iteration'] = iteration + 1
                batch_data_with_metadata.append(item_with_metadata)
            
            with open(intermediate_json_path, 'w') as f:
                json.dump(batch_data_with_metadata, f, indent=2)
            
            print(f"Saved intermediate training data to {intermediate_json_path}")
            print(f"Trained model saved to {trained_model_path}")
            print(f"🎓 Completed iteration {iteration + 1} with CEFR level {cefr_level}")
        
        # Add CEFR level metadata to all training data
        print(f"\nAdding CEFR level metadata to final training data...")
        final_training_data_with_metadata = []
        samples_per_iteration = len(self.all_training_data) // num_iterations
        
        for i, item in enumerate(self.all_training_data):
            iteration_idx = min(i // samples_per_iteration, num_iterations - 1)
            cefr_level = self.cefr_levels[iteration_idx]
            
            item_with_metadata = item.copy()
            item_with_metadata['cefr_level'] = cefr_level
            item_with_metadata['iteration'] = iteration_idx + 1
            final_training_data_with_metadata.append(item_with_metadata)
        
        # Save final training data
        print(f"Saving final training data to {self.final_json_path}")
        with open(self.final_json_path, 'w') as f:
            json.dump(final_training_data_with_metadata, f, indent=2)
        
        print(f"\n🎉 Iterative training completed!")
        print(f"Final model: {self.current_child_model_name}")
        print(f"Total training samples generated: {len(self.all_training_data)}")
        print(f"CEFR curriculum used: {' → '.join(self.cefr_levels[:num_iterations])}")
        
        # Print CEFR level distribution
        cefr_distribution = {}
        for item in final_training_data_with_metadata:
            cefr = item['cefr_level']
            cefr_distribution[cefr] = cefr_distribution.get(cefr, 0) + 1
        
        print(f"📊 CEFR Level Distribution:")
        for cefr, count in cefr_distribution.items():
            percentage = (count / len(final_training_data_with_metadata)) * 100
            print(f"  {cefr}: {count} samples ({percentage:.1f}%)")
        
        print(f"Final training data saved to: {self.final_json_path}")

def main():
    # Ensure wandb is properly configured
    os.environ["WANDB_PROJECT"] = "iterative-cpo-training-parlai_reverse_cefr"
    os.environ["WANDB_LOG_MODEL"] = "false"
    
    print(f"🔧 Wandb project: {os.environ.get('WANDB_PROJECT')}")
    print(f"🔧 Wandb enabled: {os.environ.get('WANDB_DISABLED', 'false') != 'true'}")
    
    # Model Configuration
    model_config = {
        "initial_child_model_name": "babylm-seqlen/opt-1024-warmup-v2",
        "child_tokenizer_name": "babylm-seqlen/tokenizer",
    }
    
    # Output Configuration
    output_config = {
        "output_dir": "iterative_cpo_training_parlai_teacher_progressive_cefr_reverse_opt_1024",
        "final_json_path": "final_iterative_training_cpo_parlai_teacher_progressive_cefr_reverse_opt_1024.json",
        "num_iterations": 5
    }
    
    # Child Generation Arguments
    child_generation_args = {
        "max_new_tokens": 100,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.8,
        "num_return_sequences": 1,
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
    
    # CPO Training Configuration
    training_config = {
        "logging_steps": 10,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "dataloader_drop_last": True,
        "fp16": False,
        "learning_rate": 1e-6,
        "num_train_epochs": 1,
        "save_steps": 500,
        "eval_steps": 500,
        "warmup_steps": 10,
        "max_grad_norm": 0.5,
        "remove_unused_columns": False,
        "optim": "adamw_torch",
        "dataloader_num_workers": 0,
    }
    
    # Create and run iterative trainer
    trainer = IterativeCPOTrainerWithParlAI(
        initial_child_model_name=model_config["initial_child_model_name"],
        child_tokenizer_name=model_config["child_tokenizer_name"],
        output_dir=output_config["output_dir"],
        final_json_path=output_config["final_json_path"],
        child_generation_args=child_generation_args,
        sample_vars=sample_vars,
        training_config=training_config
    )
    
    trainer.run_iterative_training(num_iterations=output_config["num_iterations"])

if __name__ == "__main__":
    main() 