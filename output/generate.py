#!/usr/bin/env python3
"""
multiturn_dialogue_generator.py

Run from the command line to generate multi-turn dialogs between a teacher and student model.

Example:
python multiturn_dialogue_generator.py \
  --teacher_model meta-llama/Llama-3.2-3B-Instruct \
  --student_model Talking-Babies/cpo_opt_seqlen_4096_final_checkpoint \
  --num_turns 6 \
  --max_length 100 \
  --num_samples 10 \
  --output_dir ./multiturn_dialogues

The script will save JSON files into the output_dir.
"""

import argparse
import random
import json
import os
import re
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------
# Conversation starters
# --------------------------
STARTERS = [
    "Have you been on any trips recently? Where did you go, and did anything interesting happen there?",
    "What kind of music do you usually listen to? Do you have a favorite artist or concert experience you remember?",
    "Do you enjoy cooking at home? What's the best meal you've made recently, or do you prefer eating out?",
    "Do you have any pets? How long have you had them, and what do you like most about them?",
    "Do you play any sports or keep active? Have you joined any teams or tried something new lately?",
    "What's the weather usually like where you live? Does it affect your plans or the way you spend your weekends?",
    "Have you watched any shows or movies recently? Did you enjoy them, and would you recommend them to others?",
    "How's work going these days? Have you faced any interesting challenges or had any funny moments?",
    "Do you have any hobbies you like to spend time on? How did you get into them, and what keeps you interested?",
    "Do you celebrate any holidays with your family? Are there any special traditions or funny stories from past celebrations?"
]

# --------------------------
# Model loading helpers
# --------------------------
def load_model_and_tokenizer(model_name, device=None):
    print(f"Loading model and tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Ensure tokenizer has pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            # Last resort: set pad token id to eos id
            tokenizer.pad_token_id = tokenizer.eos_token_id

    return model_name, tokenizer, model, device

# --------------------------
# Response cleaning
# --------------------------

def clean_response(response: str) -> str:
    lines = response.splitlines()
    filtered_lines = [line for line in lines if "[Teacher]" not in line and "[Student]" not in line]
    return " ".join(filtered_lines).strip()

# --------------------------
# Teacher generation
# --------------------------

def generate_teacher_response(prompt, tokenizer, model, device, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_length=inputs["input_ids"].shape[1] + max_length,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return clean_response(response)

# --------------------------
# Student generation with enforcement
# --------------------------

def get_banned_tokens(tokenizer):
    banned_strings = [
        ".", ",", "!", "?", ";", ":", "'", '"', "-", "–", "—", "(", ")",
        "[", "]", "{", "}", "…", "\n", "\t", "\r", "/", "\\", "*", "_",
        "[Teacher]", "[Student]"
    ]
    banned_ids = []
    for s in banned_strings:
        # tokenizers may return multiple ids for a string; extend
        try:
            token_ids = tokenizer(s, add_special_tokens=False)["input_ids"]
            banned_ids.extend(token_ids)
        except Exception:
            # ignore tokens that tokenizer cannot encode as-is
            continue
    # Format expected by HuggingFace generate: list of lists
    return [[tid] for tid in set(banned_ids)]


def clean_response_final(response: str) -> str:
    response = re.sub(r"\[Teacher\]|\[Student\]", " ", response)
    # keep alphanumeric and spaces
    response = re.sub(r"[^\w\s]", "", response)
    response = re.sub(r"\s+", " ", response)
    return response.strip()


def generate_student_response(prompt, tokenizer, model, device, max_length=100, max_retries=3):
    bad_words_ids = get_banned_tokens(tokenizer)

    for attempt in range(max_retries):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + max_length,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.8,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            bad_words_ids=bad_words_ids
        )

        response = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        response = clean_response_final(response)

        # Ignore empty or non-alphanumeric outputs
        if any(c.isalnum() for c in response):
            return response

    return ""

# --------------------------
# Multiturn generator
# --------------------------

def generate_multiturn_samples(
    num_samples, num_turns, teacher_tokenizer, teacher_model, teacher_device,
    student_tokenizer, student_model, student_device,
    teacher_model_name, student_model_name,
    max_length
):
    samples = []
    for i in range(num_samples):
        starter = random.choice(STARTERS)
        conversation = starter  # Start with starter as metaprompt

        transcript_lines = [f"[Starter]: {starter}"]
        student_turns = []
        teacher_turns = []

        pairs = num_turns // 2
        for turn in range(pairs):  # student-teacher pairs
            # Student turn
            student_text = generate_student_response(
                conversation, student_tokenizer, student_model, student_device, max_length=max_length
            )
            student_turns.append({"turn_index": 2*turn + 1, "text": student_text})
            transcript_lines.append(f"[Student]: {student_text}")
            conversation += f" {student_text}"

            # Teacher turn
            teacher_text = generate_teacher_response(
                conversation, teacher_tokenizer, teacher_model, teacher_device, max_length=max_length
            )
            teacher_turns.append({"turn_index": 2*turn + 2, "text": teacher_text})
            transcript_lines.append(f"[Teacher]: {teacher_text}")
            conversation += f" {teacher_text}"

        transcript = "\n".join(transcript_lines)

        samples.append({
            "teacher_model": teacher_model_name,
            "student_model": student_model_name,
            "id": str(i + 1),
            "STARTER": starter,
            "level": "Default (No Age Meta-Prompt)",
            "text": transcript,
            "student_turns": student_turns,
            "teacher_turns": teacher_turns
        })

    return samples

# --------------------------
# Command-line interface
# --------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate multi-turn dialogues between teacher and student models.")
    p.add_argument("--teacher_model", type=str, required=True, help="Hugging Face model id for the teacher")
    p.add_argument("--student_model", type=str, required=True, help="Hugging Face model id for the student")
    p.add_argument("--num_turns", type=int, default=6, help="Total number of turns (student+teacher). Will be rounded up to an even number if odd.")
    p.add_argument("--max_length", type=int, default=100, help="Max generation length per reply (tokens)")
    p.add_argument("--num_samples", type=int, default=10, help="How many conversations to generate per setting")
    p.add_argument("--output_dir", type=str, default="./multiturn_dialogues", help="Directory to save JSON dialogues")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--device", type=str, default=None, help="Device to place models on (e.g. cpu, cuda). Auto-detect if not set")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # Ensure even number of turns
    num_turns = args.num_turns
    if num_turns % 2 != 0:
        print(f"Warning: num_turns={num_turns} is odd — incrementing to {num_turns+1} to keep student-teacher pairs.")
        num_turns += 1

    # Load models
    teacher_model_name, teacher_tokenizer, teacher_model, teacher_device = load_model_and_tokenizer(args.teacher_model, device=args.device)
    student_model_name, student_tokenizer, student_model, student_device = load_model_and_tokenizer(args.student_model, device=args.device)

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating {args.num_samples} conversations with {num_turns} turns, max_length={args.max_length}...")
    samples = generate_multiturn_samples(
        num_samples=args.num_samples,
        num_turns=num_turns,
        teacher_tokenizer=teacher_tokenizer,
        teacher_model=teacher_model,
        teacher_device=teacher_device,
        student_tokenizer=student_tokenizer,
        student_model=student_model,
        student_device=student_device,
        teacher_model_name=teacher_model_name,
        student_model_name=student_model_name,
        max_length=args.max_length
    )

    output_file = os.path.join(args.output_dir, f"dialogues_{num_turns}_turns_len{args.max_length}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    print(f"Saved {len(samples)} dialogues to {output_file}")


if __name__ == "__main__":
    main()

