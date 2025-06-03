from typing import List, Tuple, Dict, Union
import torch
from transformers import AutoTokenizer, GenerationConfig
from trl import AutoModelForCausalLMWithValueHead

from .base_interaction_model import BaseInteractionModel

class CustomTestModel(BaseInteractionModel):
    """Custom test model implementing student-teacher interaction using Llama 2 models.
    
    The student model generates a completion to the input prompt, and the teacher model
    provides feedback on fluency and word choice.
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
        """Initialize the custom test model with Llama 3.2 models.
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
        self.student_system_prompt = """You are a student learning to write. 
        Complete the given prompt in a clear and coherent way. This topic is movie review, please complete it consisely"""
        
        self.teacher_system_prompt = """You are a writing teacher providing feedback. 
        Evaluate the student's response for:
        1. Fluency: How natural and smooth the writing flows
        2. Word Choice: The appropriateness and variety of vocabulary used
        
        also generate feedback on how the student model can be improved
        give response in json format, with keys: fluency, word_choice, and feedback"""
    
    def _format_prompt(self, prompt: str, system_prompt: str) -> str:
        """Format the prompt with system instructions."""
        return f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{prompt} [/INST]"""
    
    def interact(
        self,
        input_prompt: Union[str, Dict[str, str]],
    ) -> Dict[str, Union[torch.Tensor, str]]:
        """Process a single prompt through the models.
        
        Args:
            input_prompt: Either a single prompt string or a dict with 'student' and 'teacher' prompts
            
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
        """
        # Handle input prompt
        if isinstance(input_prompt, str):
            student_prompt = input_prompt
            teacher_prompt = input_prompt
        else:
            student_prompt = input_prompt.get('student', '')
            teacher_prompt = input_prompt.get('teacher', '')
        
        # Format prompts with system instructions
        student_prompt = self._format_prompt(student_prompt, self.student_system_prompt)
        
        # Generate student response
        student_inputs = self.encode_text(student_prompt, is_child=True)
        student_outputs = self.child_model.generate(
            student_inputs.unsqueeze(0),
            generation_config=self.student_generation_config
        )
        student_response = student_outputs[0]
        
        # Format teacher prompt with student's response
        teacher_prompt = self._format_prompt(
            f"Student's response: {self.decode_tokens(student_response, is_child=True)}\n\n"
            f"Original prompt: {teacher_prompt}\n\n"
            "Please provide feedback on fluency and word choice.",
            self.teacher_system_prompt
        )
        
        # Generate teacher feedback
        teacher_inputs = self.encode_text(teacher_prompt, is_child=False)
        teacher_outputs = self.teacher_model.generate(
            teacher_inputs.unsqueeze(0),
            generation_config=self.teacher_generation_config
        )
        teacher_response = teacher_outputs[0]
        
        # Decode all tensors
        decoded_child_query = self.decode_tokens(student_inputs, is_child=True)
        decoded_child_response = self.decode_tokens(student_response, is_child=True)
        decoded_teacher_query = self.decode_tokens(teacher_inputs, is_child=False)
        decoded_teacher_response = self.decode_tokens(teacher_response, is_child=False)
        
        return {
            'child_query': student_inputs,
            'child_response': student_response,
            'teacher_query': teacher_inputs,
            'teacher_response': teacher_response,
            'decoded_child_query': decoded_child_query,
            'decoded_child_response': decoded_child_response,
            'decoded_teacher_query': decoded_teacher_query,
            'decoded_teacher_response': decoded_teacher_response
        }
    
    def batch_interact(
        self,
        input_prompts: List[Union[str, Dict[str, str]]],
    ) -> Dict[str, List[Union[torch.Tensor, str]]]:
        """Process multiple prompts through student and teacher models.
        
        Args:
            input_prompts: List of prompts, where each prompt is either a string or a dict
            
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
        """
        results = [self.interact(prompt) for prompt in input_prompts]
        
        return {
            'child_queries': [r['child_query'] for r in results],
            'child_responses': [r['child_response'] for r in results],
            'teacher_queries': [r['teacher_query'] for r in results],
            'teacher_responses': [r['teacher_response'] for r in results],
            'decoded_child_queries': [r['decoded_child_query'] for r in results],
            'decoded_child_responses': [r['decoded_child_response'] for r in results],
            'decoded_teacher_queries': [r['decoded_teacher_query'] for r in results],
            'decoded_teacher_responses': [r['decoded_teacher_response'] for r in results]
        }
