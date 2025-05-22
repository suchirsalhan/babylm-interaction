import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from pathlib import Path

def init_models():
    # Define paths
    base_path = Path("/nvme0n1-disk/projects/babylm/models")
    teacher_path = base_path / "teacher"
    child_path = base_path / "child"
    
    # Create directories if they don't exist
    teacher_path.mkdir(parents=True, exist_ok=True)
    child_path.mkdir(parents=True, exist_ok=True)
    
    # Download and save teacher model
    print("Downloading teacher model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        device_map="auto"
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct"
    )
    
    # Save teacher model and tokenizer
    teacher_model.save_pretrained(teacher_path)
    teacher_tokenizer.save_pretrained(teacher_path)
    
    # Initialize child model with random weights
    print("Initializing child model...")
    child_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        device_map="auto"
    )
    
    # Randomize child model parameters
    for param in child_model.parameters():
        param.data = torch.randn_like(param.data) * 0.02
    
    # Save child model
    child_model.save_pretrained(child_path)
    teacher_tokenizer.save_pretrained(child_path)  # Use same tokenizer
    
    print("Models initialized successfully!")
    print(f"Teacher model saved to: {teacher_path}")
    print(f"Child model saved to: {child_path}")

if __name__ == "__main__":
    init_models() 