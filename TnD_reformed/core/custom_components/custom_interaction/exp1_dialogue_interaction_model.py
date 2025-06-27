from typing import List, Tuple, Dict, Union
import torch
from transformers import AutoTokenizer, GenerationConfig
from trl import AutoModelForCausalLMWithValueHead

from .base_interaction_model import BaseInteractionModel

class DialogueInteractionModel(BaseInteractionModel):
    """Dialogue interaction model implementing child-teacher interaction for dialogue completion.
    
    The child model generates a continuation to a partial dialogue, and the teacher model
    (instruction model like llama3.2 1b instruct) generates a completion with reference 
    to the child's generation.
    """
    
    def __init__(
        self,
        child_model,
        teacher_model,
        child_tokenizer,
        teacher_tokenizer,
        student_generation_args: dict,
        teacher_generation_args: dict,
        device: str = "cuda"
    ):
        """Initialize the dialogue interaction model.
        
        Args:
            child_model: The child model to generate dialogue continuations
            teacher_model: The teacher model (instruction model) to generate completions
            child_tokenizer: Tokenizer for the child model
            teacher_tokenizer: Tokenizer for the teacher model
            student_generation_args: Generation arguments for the child model
            teacher_generation_args: Generation arguments for the teacher model
            device: Device to run the models on
        """
        
        super().__init__(
            child_model=child_model,
            teacher_model=teacher_model,
            child_tokenizer=child_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            student_generation_args=student_generation_args,
            teacher_generation_args=teacher_generation_args,
            device=device
        )
        
        # Convert generation args to GenerationConfig objects
        self.student_generation_config = GenerationConfig(**student_generation_args)
        self.teacher_generation_config = GenerationConfig(**teacher_generation_args)
        
        # Set up system prompts
        self.child_system_prompt = """You are engaging in a dialogue. 
        Continue the conversation naturally based on the context provided. 
        Generate a coherent and contextually appropriate response that maintains the flow of the dialogue."""
        
        self.teacher_system_prompt = """You are an expert dialogue completion assistant. 
        You will be given a partial dialogue and a child model's continuation attempt. 
        Your task is to generate a high-quality completion of the partial dialogue.
        Take reference from the child continuation, make a continuation that is of similar length and topic but more coherent, fluent, grammatically correct and more contextually appropriate.
        If the child's continuation is gibberish, you should generate a completion that is coherent and contextually appropriate.
        You should only provide your own completion without any added commentary or feedback.
        Dont repeat the partial dialogue, but do continuation of the dialogue.
        """
    
#     def _format_child_prompt(self, partial_dialogue: str) -> str:
#         """Format the prompt for the child model."""
#         return f"""<s>[INST] <<SYS>>
# {self.child_system_prompt}
# <</SYS>>

# Continue this dialogue:

# {partial_dialogue}

# Continue: [/INST]"""
    
    def _format_teacher_prompt(self, partial_dialogue: str, child_continuation: str) -> str:
        """Format the prompt for the teacher model with reference to child's generation."""
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{self.teacher_system_prompt}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"Original partial dialogue:\n{partial_dialogue}\n\n"
            f"Child model's continuation:\n{child_continuation}\n\n"
            "Please generate a high-quality completion of the dialogue, taking into account the child's attempt but improving:<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
    
    def interact(
        self,
        input_prompt: Union[str, Dict[str, str]],
    ) -> Dict[str, Union[torch.Tensor, str]]:
        """Process a single partial dialogue through both child and teacher models.
        
        Args:
            input_prompt: Either a partial dialogue string or a dict with 'dialogue' key
            
        Returns:
            Dictionary containing:
            - child_query: Query tensor for child model
            - child_response: Response tensor from child model
            - teacher_query: Query tensor for teacher model
            - teacher_response: Response tensor from teacher model
            - decoded_child_query: Decoded text of child query
            - decoded_child_response: Decoded text of child response
            - decoded_teacher_query: Decoded text of teacher query
            - decoded_teacher_response: Decoded text of teacher response
            - child_continuation: Just the continuation part from child
            - teacher_completion: Just the completion part from teacher
        """
        # Handle input prompt
        if isinstance(input_prompt, str):
            partial_dialogue = input_prompt
        else:
            partial_dialogue = input_prompt.get('dialogue', input_prompt.get('student', ''))
        
        # Step 1: Generate child continuation
        # child_prompt = self._format_child_prompt(partial_dialogue)
        child_prompt = partial_dialogue
        child_inputs = self.encode_text(child_prompt, is_child=True)
        child_outputs = self.child_model.generate(
            child_inputs.unsqueeze(0),
            generation_config=self.student_generation_config
        )
        child_response = child_outputs[0]
        
        # Decode child response and extract continuation
        full_child_response = self.decode_tokens(child_response, is_child=True)
        child_continuation = self._extract_response_content(full_child_response, child_prompt)
        
        # Step 2: Generate teacher completion with reference to child's generation
        teacher_prompt = self._format_teacher_prompt(partial_dialogue, child_continuation)
        teacher_inputs = self.encode_text(teacher_prompt, is_child=False)
        teacher_outputs = self.teacher_model.generate(
            teacher_inputs.unsqueeze(0),
            generation_config=self.teacher_generation_config
        )
        teacher_response = teacher_outputs[0]
        
        # Decode teacher response and extract completion
        full_teacher_response = self.decode_tokens(teacher_response, is_child=False, skip_special_tokens=False)
        teacher_completion = self._extract_response_content(full_teacher_response, teacher_prompt)
        
        # Decode all tensors for full context
        decoded_child_query = self.decode_tokens(child_inputs, is_child=True)
        decoded_child_response = full_child_response
        decoded_teacher_query = self.decode_tokens(teacher_inputs, is_child=False)
        decoded_teacher_response = full_teacher_response
        
        return {
            'child_query': child_inputs,
            'child_response': child_response,
            'teacher_query': teacher_inputs,
            'teacher_response': teacher_response,
            'decoded_child_query': decoded_child_query,
            'decoded_child_response': decoded_child_response,
            'decoded_teacher_query': decoded_teacher_query,
            'decoded_teacher_response': decoded_teacher_response,
            'child_continuation': child_continuation,
            'teacher_completion': teacher_completion,
            'partial_dialogue': partial_dialogue
        }
    
    def _extract_response_content(self, full_response: str, prompt: str) -> str:
        """Extract the generated content from the full response by removing the prompt."""
        # Handle new Llama chat format for teacher responses
        if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
            # Find the assistant section
            parts = full_response.split("assistant<|end_header_id|>")
            if len(parts) > 1:
                # Take everything after the assistant header
                content = parts[-1].strip()
                # Remove the newline that comes right after the header
                if content.startswith('\n'):
                    content = content[1:]
                # Remove any trailing eot tokens
                if "<|eot_id|>" in content:
                    content = content.split("<|eot_id|>")[0]
                return content.strip()
        
        # Handle original format for child responses
        elif "[/INST]" in full_response:
            parts = full_response.split("[/INST]", 1)
            if len(parts) > 1:
                return parts[1].strip()
        
        # Fallback: try to remove the exact prompt from the beginning
        elif full_response.startswith(prompt):
            return full_response[len(prompt):].strip()
        
        # If no clear separation, return the full response (this shouldn't happen normally)
        return full_response.strip()
    
    def batch_interact(
        self,
        input_prompts: List[Union[str, Dict[str, str]]],
    ) -> Dict[str, List[Union[torch.Tensor, str]]]:
        """Process multiple partial dialogues through child and teacher models in batches.
        
        Args:
            input_prompts: List of partial dialogues, where each is either a string or a dict
            
        Returns:
            Dictionary containing lists of interaction results for all prompts, including:
            - child_queries: List of query tensors for child model
            - child_responses: List of response tensors from child model
            - teacher_queries: List of query tensors for teacher model
            - teacher_responses: List of response tensors from teacher model
            - decoded_child_queries: List of decoded child queries
            - decoded_child_responses: List of decoded child responses
            - decoded_teacher_queries: List of decoded teacher queries
            - decoded_teacher_responses: List of decoded teacher responses
            - child_continuations: List of child continuation texts
            - teacher_completions: List of teacher completion texts
            - partial_dialogues: List of original partial dialogues
        """
        # Extract partial dialogues from input prompts
        partial_dialogues = []
        for prompt in input_prompts:
            if isinstance(prompt, str):
                partial_dialogues.append(prompt)
            else:
                partial_dialogues.append(prompt.get('dialogue', prompt.get('student', '')))
        
        # Step 1: Batch process child model
        # Tokenize all child prompts with automatic padding
        child_batch_encoding = self.child_tokenizer(
            partial_dialogues,
            padding=True,
            return_tensors="pt",
            truncation=True
        ).to(self.device)
        
        # Generate child responses in batch
        child_batch_outputs = self.child_model.generate(
            input_ids=child_batch_encoding['input_ids'],
            attention_mask=child_batch_encoding['attention_mask'],
            generation_config=self.student_generation_config
        )
        
        # Decode child responses and extract continuations
        child_continuations = []
        decoded_child_responses = []
        child_inputs_list = []
        
        for i, child_response in enumerate(child_batch_outputs):
            full_child_response = self.decode_tokens(child_response, is_child=True)
            child_continuation = self._extract_response_content(full_child_response, partial_dialogues[i])
            child_continuations.append(child_continuation)
            decoded_child_responses.append(full_child_response)
            # Store individual input tensors for compatibility
            child_inputs_list.append(child_batch_encoding['input_ids'][i])
        
        # Step 2: Batch process teacher model
        # Create teacher prompts with reference to child continuations
        teacher_prompts = []
        for i, dialogue in enumerate(partial_dialogues):
            teacher_prompt = self._format_teacher_prompt(dialogue, child_continuations[i])
            teacher_prompts.append(teacher_prompt)
        
        # Tokenize all teacher prompts with automatic padding
        teacher_batch_encoding = self.teacher_tokenizer(
            teacher_prompts,
            padding=True,
            return_tensors="pt",
            truncation=True
        ).to(self.device)
        
        # Generate teacher responses in batch
        teacher_batch_outputs = self.teacher_model.generate(
            input_ids=teacher_batch_encoding['input_ids'],
            attention_mask=teacher_batch_encoding['attention_mask'],
            generation_config=self.teacher_generation_config
        )
        
        # Decode teacher responses and extract completions
        teacher_completions = []
        decoded_teacher_responses = []
        teacher_inputs_list = []
        
        for i, teacher_response in enumerate(teacher_batch_outputs):
            full_teacher_response = self.decode_tokens(teacher_response, is_child=False, skip_special_tokens=False)
            teacher_completion = self._extract_response_content(full_teacher_response, teacher_prompts[i])
            teacher_completions.append(teacher_completion)
            decoded_teacher_responses.append(full_teacher_response)
            # Store individual input tensors for compatibility
            teacher_inputs_list.append(teacher_batch_encoding['input_ids'][i])
        
        # Decode queries for completeness
        decoded_child_queries = []
        for i, child_inputs in enumerate(child_inputs_list):
            decoded_query = self.decode_tokens(child_inputs, is_child=True)
            decoded_child_queries.append(decoded_query)
            
        decoded_teacher_queries = []
        for i, teacher_inputs in enumerate(teacher_inputs_list):
            decoded_query = self.decode_tokens(teacher_inputs, is_child=False)
            decoded_teacher_queries.append(decoded_query)
        
        return {
            'child_queries': child_inputs_list,
            'child_responses': [response for response in child_batch_outputs],
            'teacher_queries': teacher_inputs_list,
            'teacher_responses': [response for response in teacher_batch_outputs],
            'decoded_child_queries': decoded_child_queries,
            'decoded_child_responses': decoded_child_responses,
            'decoded_teacher_queries': decoded_teacher_queries,
            'decoded_teacher_responses': decoded_teacher_responses,
            'child_continuations': child_continuations,
            'teacher_completions': teacher_completions,
            'partial_dialogues': partial_dialogues
        } 