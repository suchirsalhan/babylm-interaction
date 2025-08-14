!huggingface-cli login
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import torch

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import trange

# -------- Loaders --------

def load_teacher_model():
    teacher_model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    model.eval()
    device = model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return teacher_model_name, tokenizer, model, device

def load_student_model():
    student_model_name = "Talking-Babies/cpo_opt_seqlen_1024_progressive_cefr_reverse_parlai_iteration4"
    tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    model = AutoModelForCausalLM.from_pretrained(student_model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return student_model_name, tokenizer, model, device

# -------- Generation helpers --------

def generate_teacher_response(prompt, tokenizer, model, device, max_length=60):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    output_ids = model.generate(
        **inputs,
        max_length=inputs.input_ids.shape[1] + max_length,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = output_ids[0, inputs.input_ids.shape[1]:].tolist()
    text = tokenizer.decode(generated, clean_up_tokenization_spaces=True).strip()
    return text

def generate_student_response(prompt, tokenizer, model, device, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_length = inputs.input_ids.shape[1]
    output_ids = inputs.input_ids
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(output_ids)
            logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token_id = torch.argmax(probs, dim=-1).unsqueeze(-1)
            output_ids = torch.cat([output_ids, next_token_id], dim=-1)
            if eos_token_id != -1 and next_token_id.item() == eos_token_id:
                break

    generated_tokens = output_ids[0, input_length:].tolist()
    generated_text = tokenizer.decode(generated_tokens, clean_up_tokenization_spaces=True)
    return generated_text.strip()

# -------- Multi-turn sample generation --------

def generate_multiturn_samples(num_samples, num_turns, level, meta_prompt,
                               teacher_tokenizer, teacher_model, teacher_device,
                               student_tokenizer, student_model, student_device,
                               teacher_model_name, student_model_name):

    samples = []

    for i in trange(num_samples, desc="Generating multi-turn samples"):
        # Teacher first message from meta-prompt
        teacher_text = generate_teacher_response(meta_prompt, teacher_tokenizer, teacher_model, teacher_device)
        conversation = f"[Teacher]: {teacher_text}"

        # Multi-turn dialogue
        for turn in range(num_turns):
            student_input = conversation + "\n[Student]:"
            student_text = generate_student_response(student_input, student_tokenizer, student_model, student_device)
            conversation += f"\n[Student]: {student_text}"

            teacher_input = conversation + "\n[Teacher]:"
            teacher_text = generate_teacher_response(teacher_input, teacher_tokenizer, teacher_model, teacher_device)
            conversation += f"\n[Teacher]: {teacher_text}"

        samples.append({
            "teacher model": teacher_model_name,
            "student model": student_model_name,
            "id": str(i + 1),
            "level": level,
            "text": conversation
        })

    return samples

# -------- Main --------

if __name__ == "__main__":
    # Age level
    level = "6-11months"

    # Meta-prompt exactly as specified
    meta_prompt = """
You are an expert dialogue assistant.

Your task is to start a dialogue between you and a child model with the linguistic abilities of a child who is 6-11 months old.
You MUST be concise. Generate a conversation starter that consists of 1 to 2 sentences and is no more than 30 words total.
The conversation starter MUST draw upon the provided "Expected Knowledge" given below.
Output the conversation starter only. DO NOT include in the output anything else and stick to the "Generation Criteria" below.
You MUST format your answer as a text within double quotes.

## Expected knowledge
- Objects

## Generation criteria
# Tone
- Ensure the tone is friendly and conversational.
- The tone MUST be positive and sensible to the child model's age.

# Content
- The text MUST focus on recognising names of a few objects.
- The text MUST avoid questions that can be answered with a single word (e.g., 'yes', 'no', 'good', 'bad').
""".strip()

    # Load models
    teacher_model_name, teacher_tokenizer, teacher_model, teacher_device = load_teacher_model()
    student_model_name, student_tokenizer, student_model, student_device = load_student_model()

    # Generate samples
    samples = generate_multiturn_samples(
        num_samples=20,      # 20 samples
        num_turns=5,         # 5 interactions per sample
        level=level,
        meta_prompt=meta_prompt,
        teacher_tokenizer=teacher_tokenizer,
        teacher_model=teacher_model,
        teacher_device=teacher_device,
        student_tokenizer=student_tokenizer,
        student_model=student_model,
        student_device=student_device,
        teacher_model_name=teacher_model_name,
        student_model_name=student_model_name
    )

    # Create safe filename
    safe_teacher = teacher_model_name.replace("/", "-")
    safe_student = student_model_name.replace("/", "-")
    output_file = f"{safe_teacher}__{safe_student}__{level}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(samples)} multi-turn samples to {output_file}")
