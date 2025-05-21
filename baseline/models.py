# File: models.py
# ---------------
# All functions related to loading and saving models

import os
import torch
from utils import mkdir
import gc

import transformers
from transformers import GPT2LMHeadModel, GPT2Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from vllm import LLM, SamplingParams

DEVICE = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')


## INITIALIZATION ##
def initialize_model_and_optimizers(cfg):
    student = initialize_student_model(cfg)
    optimizer = initialize_optimizer(cfg, student)
    scheduler = initialize_scheduler(cfg, student, optimizer) 
    return student, optimizer, scheduler

def initialize_student_model(cfg):
    # First load the student
    config = GPT2Config.from_pretrained("gpt2")
    student = GPT2LMHeadModel(config).to(DEVICE)
    return student

def get_vllm_student(cfg, student, interaction_dataset, n=1):
    # Push things to the hub
    repo_name = "<repo_name>"
    student.push_to_hub(repo_name)
    interaction_dataset.student_processor.push_to_hub(repo_name)

    # Loading the vLLM wrapper
    gc.collect()
    torch.cuda.empty_cache()

    vllm_student = LLM(model=repo_name, dtype='bfloat16', enable_prefix_caching=True, gpu_memory_utilization=0.6)
    completion_tokens = cfg['datapoint_length'] - int(cfg['datapoint_length'] * cfg['context_proportion'])
    sampling_params = SamplingParams(n=n, top_p=0.8, min_tokens=completion_tokens-1, max_tokens=completion_tokens)
    return vllm_student, sampling_params

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

def initialize_optimizer(cfg, student):
    lr = cfg['learning_rate']    
    decay_parameters = get_parameter_names(student, ALL_LAYERNORM_LAYERS)    
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in student.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": cfg["weight_decay"],
        },
        {
            "params": [
                p for n, p in student.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.999) 
    )

    return optimizer

def initialize_scheduler(cfg, student, optimizer):
    num_training_steps = cfg["num_training_steps"]
    num_warmup_steps = cfg["num_warmup_steps"]
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps,
                                                             num_training_steps = num_training_steps)
    return scheduler

def get_teacher_model(cfg):
    gc.collect()
    torch.cuda.empty_cache()

    llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", dtype='bfloat16', enable_prefix_caching=True, gpu_memory_utilization=0.6)
    completion_tokens = cfg['datapoint_length'] - int(cfg['datapoint_length'] * cfg['context_proportion'])
    sampling_params = SamplingParams(top_p=0.8, min_tokens=completion_tokens-1, max_tokens=completion_tokens)
    return llm, sampling_params

## SAVING AND LOADING ##
def save_round_checkpoint(student, optimizer, scheduler, curr_round, epoch, checkpoint_dir, prefix, other_folder_name=""):
    # Open a folder for the round
    if other_folder_name == "":
        folder = os.path.join(checkpoint_dir, f'{prefix}_{curr_round}')
    else:
        folder = os.path.join(checkpoint_dir, other_folder_name)        
    mkdir(folder)

    # Save the metrics and model
    torch.save(optimizer.state_dict(), os.path.join(folder, 'latest_optimizer.pt'))
    torch.save(scheduler.state_dict(), os.path.join(folder, 'latest_scheduler.pt'))
    torch.save(student.state_dict(), os.path.join(folder, 'latest_student.pt'))

