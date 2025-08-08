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


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load student model and tokenizer
def load_student_model():
    tokenizer = AutoTokenizer.from_pretrained("Talking-Babies/sam-tokenizer")
    model = AutoModelForCausalLM.from_pretrained("Talking-Babies/opt-sam-training-preshuffled")
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

# Teacher-student chat loop
def chat_teacher_student(teacher_agent, student_tokenizer, student_model, device, prompt, num_turns=5):
    print(f"\n[Start Prompt]: {prompt}\n")
    dialogue = prompt.strip()
    teacher_input = prompt.strip()

    for turn in range(num_turns):
        print(f"\n--- Turn {turn + 1} ---")

        # Teacher's turn
        teacher_obs = {'text': teacher_input, 'episode_done': False}
        teacher_agent.observe(teacher_obs)
        teacher_act = teacher_agent.act()
        teacher_reply = teacher_act.get('text', '[No Response]')
        print(f"[Teacher]: {teacher_reply}")
        dialogue += f"\n[Teacher]: {teacher_reply}"

        # Student's turn
        student_input = dialogue + "\n[Student]:"
        student_reply = generate_student_response(student_input, student_tokenizer, student_model, device)
        print(f"[Student]: {student_reply}")
        dialogue += f"\n[Student]: {student_reply}"

        # Next teacher input
        teacher_input = dialogue + "\n[Teacher]:"

# Load teacher agent (you should define this function)
teacher = load_teacher_agent()

# Load the student tokenizer/model/device
student_tokenizer, student_model, device = load_student_model()

# Set initial prompt
initial_prompt = "Hi! I'm interested in learning about space exploration."

# Run the dialogue loop
chat_teacher_student(teacher, student_tokenizer, student_model, device, initial_prompt, num_turns=6)
