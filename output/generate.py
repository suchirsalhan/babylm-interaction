#!/usr/bin/env python3
"""
multiturn_dialogue_generator.py

Generates multi-turn dialogs between a teacher and student model.
Output filenames include number of turns, max_length, teacher model name, and student model name.
"""

import argparse
import random
import json
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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

def load_model_and_tokenizer(model_name, device=None):
    print(f"Loading model and tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    return model_name, tokenizer, model, device

def clean_response(response: str) -> str:
    lines = response.splitlines()
    filtered_lines = [line for line in lines if "[Teacher]" not in line and "[Student]" not in line]
    return " ".join(filtered_lines).strip()

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

def get_banned_tokens(tokenizer):
    banned_strings = [
        ".", ",", "!", "?", ";", ":", "'", '"', "-", "–", "—", "(", ")",
        "[", "]", "{", "}", "…", "\n", "\t", "\r", "/", "\\", "*", "_",
        "[Teacher]", "[Student]"
    ]
    banned_ids = []
    for s in banned_strings:
        try:
            token_ids = tokenizer(s, add_special_tokens=False)["input_ids"]
            banned_ids.extend(token_ids)
        except Exception:
            continue
    return [[tid] for tid in set(banned_ids)]

def clean_response_final(response: str) -> str:
    response = re.sub(r"\[Teacher\]|\[Student\]", " ", response)
    response = re.sub(r"[^\w\s]", "", response)
    response = re.sub(r"\s+", " ", response)
    return response.strip()

def generate_student_response(prompt, tokenizer, model, device, max_length=100, max_retries=3):
    bad_words_ids = get_banned_tokens(tokenizer)
    for _ in range(max_retries):
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
        if any(c.isalnum() for c in response):
            return response
    return ""

def generate_multiturn_samples(
    num_samples, num_turns, teacher_tokenizer, teacher_model, teacher_device,
    student_tokenizer, student_model, student_device,
    teacher_model_name, student_model_name,
    max_length
):
    samples = []
    for i in range(num_samples):
        starter = random.choice(STARTERS)
        conversation = starter
        transcript_lines = [f"[Starter]: {starter}"]
        student_turns = []
        teacher_turns = []
        for turn in range(num_turns // 2):
            student_text = generate_student_response(
                conversation, student_tokenizer, student_model, student_device, max_length=max_length
            )
            student_turns.append({"turn_index": 2*turn + 1, "text": student_text})
            transcript_lines.append(f"[Student]: {student_text}")
            conversation += f" {student_text}"
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

def parse_args():
    p = argparse.ArgumentParser(description="Generate multi-turn dialogues between teacher and student models.")
    p.add_argument("--teacher_model", type=str, required=True)
    p.add_argument("--student_model", type=str, required=True)
    p.add_argument("--num_turns", type=int, default=6)
    p.add_argument("--max_length", type=int, default=100)
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--output_dir", type=str, default="./multiturn_dialogues")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    if args.num_turns % 2 != 0:
        args.num_turns += 1
    teacher_model_name, teacher_tokenizer, teacher_model, teacher_device = load_model_and_tokenizer(args.teacher_model, device=args.device)
    student_model_name, student_tokenizer, student_model, student_device = load_model_and_tokenizer(args.student_model, device=args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    samples = generate_multiturn_samples(
        num_samples=args.num_samples,
        num_turns=args.num_turns,
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
    safe_teacher = teacher_model_name.replace("/", "-")
    safe_student = student_model_name.replace("/", "-")
    output_file = os.path.join(
        args.output_dir,
        f"dialogues_{args.num_turns}_turns_len{args.max_length}_{safe_teacher}_{safe_student}.json"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print(f"Saved {len(samples)} dialogues to {output_file}")

if __name__ == "__main__":
    main()


