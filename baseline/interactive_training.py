# File: interactive_training.py
# -----------------------------
# Main script for interactive LM training with a student and teacher.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from time import time
from torch.utils.data import DataLoader
import os
import math
import wandb
import gc
import pickle

from utils import get_config, setup_experiment, setup_wandb
from models import initialize_model_and_optimizers, get_teacher_model, save_round_checkpoint, get_vllm_student
from data_utils import load_babylm_data, student_po_collate_fn, InteractionDataset, teacher_correction_prompt

def full_train_loop(cfg, student, optimizer, scheduler):
    # Load the seed BabyLM dataset
    all_babylm_data = load_babylm_data(cfg)
    print("Dataset all done")

    # "Pretrain" the model via interaction
    print(f'Starting interactive training from round {cfg["start_round"]}')
    for curr_round in range(cfg["start_round"], cfg["num_rounds"]):
        interaction_loader = construct_interaction_data(cfg, student, all_babylm_data, curr_round)
        train_round(cfg, student, optimizer, scheduler, interaction_loader, curr_round, "sft_phase", cfg["n_sft_epochs"])
        train_round(cfg, student, optimizer, scheduler, interaction_loader, curr_round, "interactive_round", cfg["n_po_epochs"])

## INTERACTION SAMPLING ##
def construct_interaction_data(cfg, student, all_babylm_data, curr_round, num_rounds=1):
    # Sample input contexts from the BabyLM dataset
    savename = f'prefix_data_interaction_{curr_round}.pkl'
    sampled_context_completions = all_babylm_data.sample_round_data(cfg, savename, num_rounds=num_rounds)
    int_dataset = InteractionDataset(cfg, sampled_context_completions)

    # Sample student completions
    student_samples = sample_student_data(cfg, student, int_dataset, curr_round)
    int_dataset.add_student_completions(student_samples)

    # If there is a teacher: Generate corrections
    teacher_corrections = sample_teacher_corrections(cfg, int_dataset, curr_round)
    int_dataset.add_teacher_corrections(teacher_corrections)
    
    int_dataset.set_mode("preference_optimization")
    return DataLoader(int_dataset, batch_size=cfg['student_po_bsz'],
                      shuffle=True, collate_fn=student_po_collate_fn)

def sample_student_data(cfg, student, interaction_dataset, curr_round):
    savepath = os.path.join(cfg['logdir'], f'student_data_{curr_round}.pkl')
    vllm_student, sampling_params = get_vllm_student(cfg, student, interaction_dataset)

    # Iterate over the dataset and get student completions
    contexts = interaction_dataset.get_existing_contexts()
    num_batches = math.ceil(len(contexts) / cfg['student_sample_bsz'])
    student_samples = []

    prompts = [context['text_context'] for context in contexts]
    outputs = vllm_student.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    for output in outputs:
        student_samples.append({
            'student_losing' : list(output.outputs[0].token_ids),
            'text_losing' : output.outputs[0].text
        })

    del vllm_student
    gc.collect()
    torch.cuda.empty_cache()

    if cfg['save_generated_data'] and not os.path.exists(savepath):
        with open(savepath, 'wb') as f:
            pickle.dump(student_samples, f)

    return student_samples

def sample_teacher_corrections(cfg, interaction_dataset, curr_round):
    savepath = os.path.join(cfg['logdir'], f'teacher_data_{curr_round}.pkl')
    teacher, sampling_params = get_teacher_model(cfg)

    # Iterate over the dataset and get teacher completions
    contexts = interaction_dataset.get_existing_contexts()
    num_batches = math.ceil(len(contexts) / cfg["teacher_sample_bsz"])

    conversations = [teacher_correction_prompt(cfg, context) for context in contexts]

    outputs = teacher.chat(conversations, sampling_params=sampling_params, use_tqdm=True,
                           add_generation_prompt=False, continue_final_message=True)
    output_texts = [output.outputs[0].text for output in outputs]

    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    if cfg['save_generated_data'] and not os.path.exists(savepath):
        with open(savepath, 'wb') as f:
            pickle.dump(output_texts, f)

    return output_texts

## INNER TRAINING LOOP ##
def train_round(cfg, student, optimizer, scheduler, interaction_loader, curr_round, prefix, n_epochs):
    start_time = time()
    epoch_size = len(interaction_loader)

    for epoch in range(n_epochs):
        # Clear cache
        torch.cuda.empty_cache()

        # Train 
        no_po = prefix != "interactive_round"
        tr_metrics = round_train_epoch(cfg, student, optimizer, scheduler, interaction_loader, no_po,
                                       epoch, epoch_size, curr_round, start_time, prefix)
        print(f"Epoch {epoch}; train SFT loss: {tr_metrics['sft_loss']}, train SimPO loss: {tr_metrics['simpo_loss']}")

        # Report results
        if cfg["use_wandb"]:
            # Compute the steps
            if prefix == "sft_phase":
                # If we have performed a pretraining run
                steps = curr_round * (cfg["n_sft_epochs"] + cfg["n_po_epochs"]) * epoch_size + epoch_size * (epoch+1)
            else:
                steps = curr_round * (cfg["n_sft_epochs"] + cfg["n_po_epochs"]) * epoch_size
                steps += cfg["n_sft_epochs"] * epoch_size + epoch_size * (epoch+1)

        metric_path = os.path.join(cfg["logdir"], f"round_{curr_round}_{prefix}_epoch_{epoch}_metrics.pth")
        torch.save(tr_metrics, metric_path)

        checkpoint_dir = cfg["checkpoint_dir"]
        save_round_checkpoint(student, optimizer, scheduler, curr_round, epoch, checkpoint_dir, prefix)

def unpack_simpo_batch(minibatch, device):
    w_input_tokens = minibatch[0].to(device)
    w_target_tokens = minibatch[1].to(device)
    w_sft_mask = minibatch[2].to(device)
    w_simpo_mask = minibatch[3].to(device)
    l_input_tokens = minibatch[4].to(device)
    l_target_tokens = minibatch[5].to(device)
    l_simpo_mask = minibatch[6].to(device)

    return w_input_tokens, w_target_tokens, w_sft_mask, w_simpo_mask, \
        l_input_tokens, l_target_tokens, l_simpo_mask

def unpack_dpo_batch(minibatch, device):
    w_input_tokens = minibatch[0].to(device)
    w_target_tokens = minibatch[1].to(device)
    w_sft_mask = minibatch[2].to(device)
    w_simpo_mask = minibatch[3].to(device)
    w_ref_logp = minibatch[4].to(device)
    l_input_tokens = minibatch[5].to(device)
    l_target_tokens = minibatch[6].to(device)
    l_simpo_mask = minibatch[7].to(device)
    l_ref_logp = minibatch[8].to(device)

    return w_input_tokens, w_target_tokens, w_sft_mask, w_simpo_mask, w_ref_logp, \
        l_input_tokens, l_target_tokens, l_simpo_mask, l_ref_logp


def round_train_epoch(cfg, student, optimizer, scheduler, interaction_loader, no_po,
                      epoch, epoch_size, curr_round, start_time, prefix):
    student.train()
    total_sft_loss = 0
    total_sft_tokens = 0
    total_simpo_loss = 0
    num_pairs = 0

    temp_sft_loss = 0
    temp_sft_tokens = 0
    if not no_po:
        temp_simpo_loss = 0
        temp_num_pairs = 0
        temp_w_reward = 0
        temp_l_reward = 0
        temp_margin = 0
        temp_win_rate = 0

    device = student.device
    beta = cfg['simpo_beta']
    gamma = cfg['simpo_beta'] * cfg['simpo_gamma_ratio']
    sft_lambda = cfg['sft_lambda']
    steps_for_one_mil = len(interaction_loader) // 5
    for train_step, minibatch in enumerate(tqdm(interaction_loader)):
        w_input_tokens, w_target_tokens, w_sft_mask, w_simpo_mask, \
            l_input_tokens, l_target_tokens, l_simpo_mask = unpack_simpo_batch(minibatch, device)
        sft_tokens = torch.sum(w_sft_mask).item()
        B = w_input_tokens.shape[0]

        # Perform forward passes
        if no_po:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                w_logits = student(w_input_tokens)['logits']
                w_log_probs = F.log_softmax(w_logits, dim=2)
                w_token_log_probs = torch.gather(w_log_probs, 2, w_target_tokens.unsqueeze(2)).squeeze(2)
            
            # SFT loss
            sft_loss = - torch.sum(w_token_log_probs * w_sft_mask) / torch.sum(w_sft_mask)
            simpo_loss = torch.Tensor([0])
            loss = sft_loss
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                w_logits = student(w_input_tokens)['logits']
                w_log_probs = F.log_softmax(w_logits, dim=2)
                w_token_log_probs = torch.gather(w_log_probs, 2, w_target_tokens.unsqueeze(2)).squeeze(2)
                l_logits = student(l_input_tokens)['logits']
                l_log_probs = F.log_softmax(l_logits, dim=2)
                l_token_log_probs = torch.gather(l_log_probs, 2, l_target_tokens.unsqueeze(2)).squeeze(2)

            # SFT loss
            sft_loss = - torch.sum(w_token_log_probs * w_sft_mask) / torch.sum(w_sft_mask)

            # SimPO loss
            w_reward = beta * torch.sum(w_token_log_probs * w_simpo_mask, dim=1) / torch.sum(w_simpo_mask, dim=1)
            l_reward = beta * torch.sum(l_token_log_probs * l_simpo_mask, dim=1) / torch.sum(l_simpo_mask, dim=1)
            simpo_loss = - torch.mean(F.logsigmoid(w_reward - l_reward - gamma))

            loss = simpo_loss + sft_lambda * sft_loss

            with torch.no_grad():
                # Basic loss terms
                temp_w_reward += torch.sum(w_reward).item() / beta
                temp_l_reward += torch.sum(l_reward).item() / beta
                temp_win_rate += torch.sum(w_reward > l_reward).item()
                temp_margin += torch.sum(w_reward - l_reward).item() / beta
                temp_simpo_loss += simpo_loss.item() * B
                temp_num_pairs += B

        loss.backward()
        if cfg["gradient_clip_norm"] != -1: 
            nn.utils.clip_grad_norm_(student.parameters(), cfg['gradient_clip_norm'])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_sft_loss += sft_loss.item() * sft_tokens
        total_sft_tokens += sft_tokens
        total_simpo_loss += simpo_loss.item() * B
        num_pairs += B
        temp_sft_loss += sft_loss.item() * sft_tokens
        temp_sft_tokens += sft_tokens

        if cfg["use_wandb"] and (train_step % 10 == 0 and train_step > 0):
            # Compute the steps
            if prefix == "sft_phase":
                # If we have performed a pretraining run
                steps = curr_round * (cfg["n_sft_epochs"] + cfg["n_po_epochs"]) * epoch_size + epoch_size * epoch + train_step
            else:
                steps = curr_round * (cfg["n_sft_epochs"] + cfg["n_po_epochs"]) * epoch_size
                steps += cfg["n_epochs"] * epoch_size + epoch_size * epoch + train_step

            if no_po:
                wandb_sft_train_epoch(
                    temp_sft_loss / temp_sft_tokens, steps, curr_round, start_time, prefix
                )
            else:
                wandb_interaction_train_epoch(
                    temp_sft_loss / temp_sft_tokens, temp_simpo_loss / temp_num_pairs,
                    temp_w_reward / temp_num_pairs, temp_l_reward / temp_num_pairs,
                    temp_win_rate / temp_num_pairs, temp_margin / temp_num_pairs,
                    steps, curr_round, start_time, prefix
                )

            temp_sft_loss = 0
            temp_sft_tokens = 0
            if not no_po:
                temp_simpo_loss = 0
                temp_num_pairs = 0
                temp_w_reward = 0
                temp_l_reward = 0
                temp_win_rate = 0
                temp_margin = 0

        # Intermediate checkpoint spot for each 1M tokens words the first 10M
        if prefix == "sft_phase" and epoch < 2 and curr_round == 0 and train_step % steps_for_one_mil == 0 and train_step > 0:
            curr_million = epoch * 5 + train_step // steps_for_one_mil
            save_round_checkpoint(student, optimizer, scheduler, curr_round, epoch, cfg["checkpoint_dir"], prefix,
                                  other_folder_name=f"checkpoint_{curr_million}M")

    # Intermediate checkpoint spot for 5M, 10M and other 10M steps until 100M
    if prefix == "sft_phase" and epoch == 0 and curr_round == 0:
        save_round_checkpoint(student, optimizer, scheduler, curr_round, epoch, cfg["checkpoint_dir"], prefix,
                              other_folder_name=f"checkpoint_5M")
    elif prefix == "sft_phase" and (epoch + 1) % 2 == 0 and curr_round < 2:
        num_words = curr_round * 50 + (epoch + 1) * 5
        save_round_checkpoint(student, optimizer, scheduler, curr_round, epoch, cfg["checkpoint_dir"], prefix,
                              other_folder_name=f"checkpoint_{num_words}M")
    elif prefix != "sft_phase" and (epoch + 1) % 2 == 0 and curr_round < 2:    
        num_words = (curr_round + 1) * 50
        save_round_checkpoint(student, optimizer, scheduler, curr_round, epoch, cfg["checkpoint_dir"], prefix,
                              other_folder_name=f"checkpoint_{num_words}M")

    return {"sft_loss" : total_sft_loss / total_sft_tokens, "simpo_loss" : total_simpo_loss / num_pairs}
        
## Wandb utils ##

def wandb_interaction_train_epoch(sft_loss, simpo_loss, w_reward, l_reward, win_rate, margin,
                                  step, curr_round, start_time, prefix):
    time_elapsed = (time() - start_time) / 60
    curr_dict = {
        f"round_{curr_round}_{prefix}/time_elapsed" : time_elapsed,
        f"round_{curr_round}_{prefix}/batch_train_sft_loss" : sft_loss,
        f"round_{curr_round}_{prefix}/batch_train_po_loss" : simpo_loss,
        f"round_{curr_round}_{prefix}/batch_train_w_reward" : w_reward,
        f"round_{curr_round}_{prefix}/batch_train_l_reward" : l_reward,
        f"round_{curr_round}_{prefix}/batch_train_win_rate" : win_rate,
        f"round_{curr_round}_{prefix}/batch_train_margin" : margin,
    }
    wandb.log(curr_dict, step=step)

def wandb_sft_train_epoch(sft_loss, step, curr_round, start_time, prefix):
    time_elapsed = (time() - start_time) / 60
    curr_dict = {
        f"round_{curr_round}_{prefix}/time_elapsed" : time_elapsed,
        f"round_{curr_round}_{prefix}/batch_train_sft_loss" : sft_loss,
    }
    wandb.log(curr_dict, step=step)

def main():
    # Setup the experiment
    cfg = get_config()

    setup_experiment(cfg)
    if cfg["use_wandb"]:
        setup_wandb(cfg)
    print("Env init")

    # Load the student model and optimizers
    student, optimizer, scheduler = initialize_model_and_optimizers(cfg)
    print("Models loaded")

    # Perform training
    full_train_loop(cfg, student, optimizer, scheduler)
    

if __name__ == "__main__":
    main()
