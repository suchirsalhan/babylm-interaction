# File: data_utils.py
# -------------------
# Function for dataset loading, construction and saving + collation functions

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

import math
import random
import os
from tqdm import tqdm
import pickle

STUDENT_EOS = 50256
TRAIN_PATH_100M = 'data/text_data/clean_train_100M'
DATASETS = ['bnc_spoken', 'childes', 'gutenberg', 'open_subtitles', 'simple_wiki', 'switchboard']

class FullBabyLMDataset(Dataset):

    def __init__(self, cfg):
        # First load the tokenizer
        self.processor = student_tokenizer(cfg)

        # Tokenize, split and reconstruct each dataset
        num_chunks = cfg["num_rounds"]
        self.data = {}
        dataset_folder = TRAIN_PATH_100M

        for dataset in DATASETS:
            # Load all text in dset
            dataset_path = os.path.join(dataset_folder, f'{dataset}.train')
            with open(dataset_path, 'r') as f:
                all_text = ' '.join(f.readlines())
            print(f'Opened {dataset_path}')

            # Process full text into tokens
            tokenized_dataset = self.processor(text=[all_text])['input_ids'][0]
            print(f'Tokenized {dataset_path}; {len(tokenized_dataset)} tokens total')

            # Chunk and add
            self.data[dataset] = []
            chunk_size = len(tokenized_dataset) // num_chunks
            for curr_chunk in tqdm(range(num_chunks)):
                start = curr_chunk * chunk_size
                end = (curr_chunk+1) * chunk_size
                chunk_tokens = tokenized_dataset[start:end] if curr_chunk != num_chunks - 1 else tokenized_dataset[start:]
                chunk_text = self.processor.decode(chunk_tokens)
                self.data[dataset].append(chunk_text)
            print(f"Chunked {dataset_path}")

    def sample_round_data(self, cfg, savename, num_rounds=1):
        # Load the popped items if saved
        logdir = cfg['logdir']
        savepath = os.path.join(logdir, savename)

        # Sample the contexts
        sampled_context_completions = []
        for curr_round in range(num_rounds):
            available_chunks = len(self.data['childes'])
            for dataset in DATASETS:
                # Get the text sample
                sample_idx = random.randint(0, available_chunks-1)
                current_chunk = self.data[dataset].pop(sample_idx)

                # Tokenize the text sample
                tokenized_dset_chunk = self.processor(text=[current_chunk])['input_ids'][0]

                # Construct smaller interactions from the chunk
                interaction_size = cfg["datapoint_length"]
                context_len = int(interaction_size * cfg["context_proportion"])
                num_datapoints = len(tokenized_dset_chunk) // interaction_size
                for i in range(num_datapoints):
                    start, end = i * interaction_size, (i+1) * interaction_size
                    full_interaction = tokenized_dset_chunk[start:end]
                    
                    context = full_interaction[:context_len]
                    completion = full_interaction[context_len:]

                    sampled_context_completions.append({
                        'student_context' : context,
                        'text_context' : self.processor.decode(context),
                    })

        return sampled_context_completions

    def __len__(self):
        return 42

    def __getitem__(self, idx):
        return "NotImplemented"

class InteractionDataset(Dataset):

    def __init__(self, cfg, sampled_context_completions):
        self.mode = "student_sampling"
        self.student_processor = student_tokenizer(cfg)
        self.data = sampled_context_completions

    def __len__(self):
        return len(self.data)

    def set_mode(self, mode):
        self.mode = mode

    def add_student_completions(self, student_samples):
        for i, interaction_dict in enumerate(self.data):
            interaction_dict['text_losing'] = student_samples[i]['text_losing']
            interaction_dict['student_losing'] = student_samples[i]['student_losing']

    def add_teacher_corrections(self, teacher_corrections):
        for i, teacher_correction in enumerate(teacher_corrections):
            interaction_dict = self.data[i]
            interaction_dict['text_winning'] = teacher_correction

            student_length = len(interaction_dict['student_losing'])
            teacher_correction_tokens = self.student_processor(text=[teacher_correction])['input_ids'][0][:student_length]
            interaction_dict['student_winning'] = teacher_correction_tokens

    def add_ranked_preference_pairs(self, teacher_rankings):
        new_data = []
        for i, interaction_dict in enumerate(self.data):
            teacher_ranking = teacher_rankings[i]
            if len(teacher_ranking) == 0:
                continue

            interaction_dict['text_winning'] = teacher_ranking['text_winning']
            interaction_dict['text_losing'] = teacher_ranking['text_losing']
            interaction_dict['student_winning'] = teacher_ranking['student_winning']
            interaction_dict['student_losing'] = teacher_ranking['student_losing']
            new_data.append(interaction_dict)

        self.data = new_data

    def get_existing_contexts(self):
        return self.data

    def __getitem__(self, idx):
        if self.mode == "student_sampling":
            return self.getitem_interaction_context(idx)
        elif self.mode == "preference_optimization":
            return self.getitem_preference_optimization(idx)

    def getitem_interaction_context(self, idx):
        int_dict = self.data[idx]
        return torch.LongTensor(int_dict['student_context'])

    def getitem_preference_optimization(self, idx):
        int_dict = self.data[idx]

        # First process the context and get context length
        context = int_dict['student_context']
        context_len = len(context) + 1

        # Next process the winning and losing items
        winning_portion = int_dict['student_winning']
        w_tokens = torch.LongTensor([STUDENT_EOS] + context + winning_portion + [STUDENT_EOS])
        losing_portion = int_dict['student_losing']
        l_tokens = torch.LongTensor([STUDENT_EOS] + context + losing_portion + [STUDENT_EOS])

        return w_tokens, l_tokens, context_len

## General utilities ##
def load_babylm_data(cfg):
    # Get the overall BabyLM dataset to extract data from (behavior may vary)
    num_rounds = cfg['num_rounds']
    filename = os.path.join('data/text_data/cached_train', f'train_gpt2_100M_{num_rounds}_rounds.pkl')
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            full_babylm_dset = pickle.load(f)
    else:
        full_babylm_dset = FullBabyLMDataset(cfg)
        with open(filename, 'wb') as f:
            pickle.dump(full_babylm_dset, f)

    return full_babylm_dset

def student_tokenizer(cfg):
    return AutoTokenizer.from_pretrained("gpt2")

def student_po_collate_fn(batch):
    # Winning item processing
    w_tokens = pad_sequence([item[0] for item in batch], padding_value=STUDENT_EOS, batch_first=True)
    context_lens = [item[2] for item in batch]
    w_input_tokens = w_tokens[:, :-1]
    w_target_tokens = w_tokens[:, 1:]

    w_sft_mask = w_input_tokens != STUDENT_EOS
    w_sft_mask[:, 0] = 1
    w_simpo_mask = w_input_tokens != STUDENT_EOS
    for i, item in enumerate(batch):
        w_simpo_mask[i, :context_lens[i] - 1] = 0

    # Losing item processing
    l_tokens = pad_sequence([item[1] for item in batch], padding_value=STUDENT_EOS, batch_first=True)
    l_input_tokens = l_tokens[:, :-1]
    l_target_tokens = l_tokens[:, 1:]
    l_simpo_mask = l_input_tokens != STUDENT_EOS
    for i, item in enumerate(batch):
        l_simpo_mask[i, :context_lens[i] - 1] = 0

    return w_input_tokens, w_target_tokens, w_sft_mask, w_simpo_mask, \
        l_input_tokens, l_target_tokens, l_simpo_mask
    
## PROMPTS ##

def teacher_correction_prompt(cfg, context):
    input_context = context['text_context']
    student_completion = context['text_losing']

    conversation = [
        {
            "role" : "system",
            "content" : "You are a writer capable of continuing partial dialogues or stories that are given to you. " +\
            "You will be given a partial text (labeled 'Partial Text') and a completion of said text produced by a student of English (labeled 'Student Completion'). " +\
            "Your goal is to produce a corrected version of the student's completion. This corrected version should be grammatically " +\
            "correct, coherent and relevant to the initial partial text. If the student's response is gibberish, output your own independent completion." +\
            "You should only provide your own completion without any added commentary or feedback."
        },
        {
            "role" : "user",
            "content" : f"Partial Text: {input_context}"
        },
        {
            "role" : "user",
            "content" : f"Student Completion: {student_completion}"
        },
        {
            'role' : "user",
            'content' : "Now produce your own completion of the Partial Text. Do not include any external commentary."
        },
        {
            'role' : 'assistant',
            'content' : f'Partial Text: {input_context}\n Corrected Completion:'
        }
    ]

    return conversation
