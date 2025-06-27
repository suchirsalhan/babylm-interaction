# train_orpo.py
from datasets import load_dataset, load_from_disk
from trl import ORPOConfig, ORPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# change params here
tokenizer = AutoTokenizer.from_pretrained("Talking-Babies/tokenizer")
model_name = "Talking-Babies/opt-Talking-Babies-train_cosmos"
output_dir = "orpo_opt_train_cosmos"
    
# Add pad token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load model with better precision handling
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
)
train_dataset = load_from_disk("./my_dataset_short_opt_base/train")

training_args = ORPOConfig(output_dir=output_dir, logging_steps=10)
trainer = ORPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()