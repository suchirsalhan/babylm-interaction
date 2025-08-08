!huggingface-cli login
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import torch

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------- Loaders --------

def load_teacher_model():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None  # Let HF handle device placement
    )
    model.eval()
    return tokenizer, model, model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_student_model():
    tokenizer = AutoTokenizer.from_pretrained("Talking-Babies/cpo_opt_seqlen_1024_final_checkpoint")
    model = AutoModelForCausalLM.from_pretrained("Talking-Babies/cpo_opt_seqlen_1024_final_checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

# -------- Generation helpers --------

def remove_punctuation_from_logits(logits, tokenizer):
    banned_punctuations = [".", ",", "?", "!", ":", ";", "-", "(", ")", "\"", "'"]
    banned_token_ids = [tokenizer.convert_tokens_to_ids(p) for p in banned_punctuations]
    banned_token_ids = [tid for tid in banned_token_ids if tid is not None and tid != tokenizer.unk_token_id]
    for tid in banned_token_ids:
        logits[:, tid] = -1e9
    return logits

def generate_student_response(prompt, tokenizer, model, device, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_length = inputs.input_ids.shape[1]

    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1
    output_ids = inputs.input_ids

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(output_ids)
            logits = outputs.logits[:, -1, :]

            # Optionally disable punctuation banning during debugging
            # logits = remove_punctuation_from_logits(logits, tokenizer)

            probs = torch.nn.functional.softmax(logits, dim=-1)

            next_token_id = torch.argmax(probs, dim=-1).unsqueeze(-1)

            # Validate token id range
            if next_token_id.item() < 0 or next_token_id.item() >= logits.shape[-1]:
                break

            output_ids = torch.cat([output_ids, next_token_id], dim=-1)

            if eos_token_id != -1 and next_token_id.item() == eos_token_id:
                break

    generated_tokens = output_ids[0, input_length:].tolist()
    generated_text = tokenizer.decode(generated_tokens, clean_up_tokenization_spaces=True)
    return generated_text.strip()

# -------- Teacher generation --------

def generate_teacher_response(prompt, tokenizer, model, device, max_length=100):
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

# -------- Chat loop --------

def chat_teacher_student(teacher_tokenizer, teacher_model, teacher_device,
                         student_tokenizer, student_model, student_device,
                         meta_prompt, num_turns=5):

    print(f"\n[Teacher Meta-Prompt]: {meta_prompt}\n")

    teacher_reply = generate_teacher_response(meta_prompt, teacher_tokenizer, teacher_model, teacher_device)
    print(f"[Teacher]: {teacher_reply}")

    dialogue = f"[Teacher]: {teacher_reply}"

    for turn in range(num_turns):
        print(f"\n--- Turn {turn+1} ---")

        student_input = dialogue + "\n[Student]:"
        student_reply = generate_student_response(student_input, student_tokenizer, student_model, student_device)
        print(f"[Student]: {student_reply}")

        dialogue += f"\n[Student]: {student_reply}"

        teacher_input = dialogue + "\n[Teacher]:"
        teacher_reply = generate_teacher_response(teacher_input, teacher_tokenizer, teacher_model, teacher_device)
        print(f"[Teacher]: {teacher_reply}")

        dialogue += f"\n[Teacher]: {teacher_reply}"

# -------- Usage example --------

if __name__ == "__main__":
    # Fix environment variable before launching script if needed:
    # import os
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    teacher_tokenizer, teacher_model, teacher_device = load_teacher_model()
    student_tokenizer, student_model, student_device = load_student_model()
    meta_prompt = (
        "You are an expert dialogue assistant initiating a conversation with a child language model "
        "that has the linguistic abilities of a 6 to 11 month old infant.\n\n"
        "Generate the first message to begin the dialogue. The message should:\n"
        "- Be concise: 1 to 2 short sentences, no more than 30 words total.\n"
        "- Use a friendly, positive, age-appropriate tone.\n"
        "- Mention or draw attention to a few familiar objects (e.g., ball, cup, dog, book).\n"
        "- Avoid abstract or complex concepts.\n\n"
        "Tone:\n"
        "- Conversational and nurturing\n"
        "- Suitable for a preverbal or babbling child\n\n"
        "Only output the first utterance to the child model to start the conversation."
    )
    chat_teacher_student(
        teacher_tokenizer, teacher_model, teacher_device,
        student_tokenizer, student_model, student_device,
        meta_prompt, num_turns=6
    )
