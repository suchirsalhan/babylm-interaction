!pip install --upgrade --force-reinstall huggingface_hub transformers

import huggingface_hub
print(huggingface_hub.__version__)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


import huggingface_hub
print(huggingface_hub.__version__)
# =======================
# 🔧 Install dependencies
# =======================
!pip install parlai --quiet

# ===============================
# 📚 Load the teacher model only
# ===============================

from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent

def load_teacher_agent(model_file='zoo:blender/blender_90M/model'):
    parser = ParlaiParser(True, True, "Teacher model loader")
    parser.set_params(model_file=model_file)
    opt = parser.parse_args([])
    agent = create_agent(opt, requireModelExists=True)
    return agent

# Load teacher
teacher = load_teacher_agent()


from transformers import AutoTokenizer, AutoModelForCausalLM, OPTForCausalLM
import torch

# Load student model and tokenizer
def load_student_model():
    tokenizer = AutoTokenizer.from_pretrained("Talking-Babies/cpo_opt_seqlen_1024_final_checkpoint")
    model = AutoModelForCausalLM.from_pretrained("Talking-Babies/cpo_opt_seqlen_1024_final_checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

# Generate student response using Hugging Face Transformers
def generate_student_response(prompt, tokenizer, model, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + 50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

from transformers import StoppingCriteria, StoppingCriteriaList

# Custom stopping criteria to halt on '[Teacher]:'
class StopOnTeacherToken(StoppingCriteria):
    def __init__(self, tokenizer, stop_str="[Teacher]:"):
        self.tokenizer = tokenizer
        self.stop_ids = tokenizer.encode(stop_str, add_special_tokens=False)

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the last tokens match the stop token sequence
        if input_ids[0].tolist()[-len(self.stop_ids):] == self.stop_ids:
            return True
        return False

# Generate student response using Hugging Face Transformers
def generate_student_response(prompt, tokenizer, model, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    stopping_criteria = StoppingCriteriaList([StopOnTeacherToken(tokenizer)])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + 50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )

    # Decode and postprocess
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Fallback postprocessing: cut off any hallucinated '[Teacher]:'
    if "[Teacher]:" in response:
        response = response.split("[Teacher]:")[0].strip()

    return response.strip()

# Teacher-student chat loop with meta-prompt to generate the teacher's first message
def chat_teacher_student(teacher_agent, student_tokenizer, student_model, device, meta_prompt, num_turns=5):
    print(f"\n[Teacher Meta-Prompt]: {meta_prompt}\n")

    # Step 1: Ask teacher to generate the first message using meta-prompt
    teacher_obs = {'text': meta_prompt, 'episode_done': False}
    teacher_agent.observe(teacher_obs)
    teacher_act = teacher_agent.act()
    teacher_reply = teacher_act.get('text', '[No Response]')
    print(f"[Teacher]: {teacher_reply}")

    # Initialize dialogue with teacher's generated message
    dialogue = f"[Teacher]: {teacher_reply}"

    for turn in range(num_turns):
        print(f"\n--- Turn {turn + 1} ---")

        # Student's turn
        student_input = dialogue + "\n[Student]:"
        student_reply = generate_student_response(student_input, student_tokenizer, student_model, device)
        print(f"[Student]: {student_reply}")
        dialogue += f"\n[Student]: {student_reply}"

        # Teacher's next turn
        teacher_input = dialogue + "\n[Teacher]:"
        teacher_obs = {'text': teacher_input, 'episode_done': False}
        teacher_agent.observe(teacher_obs)
        teacher_act = teacher_agent.act()
        teacher_reply = teacher_act.get('text', '[No Response]')
        print(f"[Teacher]: {teacher_reply}")
        dialogue += f"\n[Teacher]: {teacher_reply}"

teacher = load_teacher_agent()
student_tokenizer, student_model, device = load_student_model()
meta_prompt = (
    "You are an expert dialogue assistant interacting with a language model that reflects the communication skills of a 7–8-year-old child.\n\n"
    "Your job is to generate the **first message** in a conversation designed to challenge the student's ability to:\n"
    "1. Ask questions to clarify information\n"
    "2. Use appropriate grammar\n"
    "3. Problem-solve using language\n"
    "4. Recount imaginary or real events\n"
    "5. Follow multi-step instructions\n"
    "6. Express opinions, thoughts, and ideas\n\n"
    "### Message Requirements\n"
    "- Be friendly and supportive in tone\n"
    "- Use 1–2 clear, age-appropriate sentences (max 35 words total)\n"
    "- Present a **small problem**, situation, or event that invites the student to ask questions, share ideas, or solve something\n"
    "- Avoid abstract language or overly complex vocabulary\n\n"
    "### Examples of Good First Messages\n"
    "- 'I lost my keys somewhere in the house—can you help me figure out where they might be?'\n"
    "- 'A boy finds a strange box in the woods. What do you think is inside?'\n"
    "- 'Someone spilled juice on the floor. What should we do first?'\n\n"
    "Now generate the first message **you would say to the student** to begin the conversation."
)

chat_teacher_student(teacher, student_tokenizer, student_model, device, meta_prompt, num_turns=6)
