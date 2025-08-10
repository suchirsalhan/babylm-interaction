"""
Full replacement training script that integrates:
 - OPT pretraining (placeholder trainer hooks - expects CustomTrainer in your repo)
 - ORPO iterative interaction phases using either a HuggingFace teacher model *or* a
   ParlAI ControllableBlender teacher agent (CEFR-controlled)
 - Four selectable curricula mapping fractions -> CEFR labels
 - Saving and pushing child models to Hugging Face Hub after each ORPO phase

Notes / requirements (you must run these once in the environment):
  - Git clone & install ControllableComplexityChatbot and ParlAI (see README comments)
  - Set environ HF_TOKEN with a valid Hugging Face token that has repo write permission
  - This script implements a lightweight IterativeORPOTrainer interface used by the ORPO callback.
    If you already have a richer implementation in your repo, you can drop that in and keep
    the `teacher_agent` acceptance behavior (we preserve both code paths: `teacher_agent` OR `teacher_model_name`).

Usage examples:
  python train_with_orpo_and_parlai_teacher.py --seq_len 128 --curriculum_id 1 --push_to_hub

"""

import os
import sys
import time
import json
import argparse
import shutil
import tempfile
import math
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import torch
from datasets import load_dataset
from transformers import (
    OPTConfig,
    OPTForCausalLM,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl

# optional: huggingface hub integration
try:
    from huggingface_hub import HfApi
except Exception:
    HfApi = None

# Ensure ParlAI/ControllableBlender imports are lazy - they are heavy
PARLAI_AVAILABLE = False
try:
    # These imports may fail if ParlAI not installed; we'll import when needed
    from parlai.zoo.blender.blender_3B import download as parlai_blender_download
    from controllable_blender import ControllableBlender
    PARLAI_AVAILABLE = True
except Exception:
    PARLAI_AVAILABLE = False

# Try to import existing IterativeORPOTrainer from repo; otherwise use fallback below
try:
    from iterative_orpo import IterativeORPOTrainer  # user-provided implementation
except Exception:
    IterativeORPOTrainer = None


# -------------------------
# Helper: push model to HF hub
# -------------------------

def push_model_to_hub(local_dir: str, repo_id: str, token: Optional[str] = None):
    """Push a model folder to Hugging Face Hub using the huggingface_hub API when available.

    Falls back to printing instructions if the library is not available.
    """
    print(f"[HUB] Pushing {local_dir} to {repo_id}")
    if token is None:
        token = os.environ.get("HF_TOKEN")
    if token is None:
        print("[HUB] WARNING: HF_TOKEN is not set. Skipping push. Set HF_TOKEN env var to push.")
        return False

    if HfApi is None:
        print("[HUB] huggingface_hub not installed. Install via `pip install huggingface_hub` to enable push.")
        return False

    api = HfApi()
    # Create repo if not exists
    try:
        api.create_repo(repo_id=repo_id, token=token, exist_ok=True)
    except Exception as e:
        print(f"[HUB] create_repo: {e}")

    # Use upload_folder to put files in the repo
    try:
        api.upload_folder(folder_path=local_dir, path_in_repo="", repo_id=repo_id, token=token)
        print(f"[HUB] Uploaded folder {local_dir} to {repo_id}")
        return True
    except Exception as e:
        print(f"[HUB] upload_folder failed: {e}")
        return False


# -------------------------
# Fallback IterativeORPOTrainer
# -------------------------
if IterativeORPOTrainer is None:
    class IterativeORPOTrainer:
        """
        Minimal fallback implementation for ORPO-style iterative interaction.

        - Accepts either teacher_agent (a ParlAI ControllableBlender instance) OR
          teacher_model_name (a HuggingFace model id string).
        - Generates (prompt, teacher_response) pairs by sampling the child model and
          querying the teacher. Then fine-tunes the child on the teacher responses
          for a small number of steps and saves the resulting child model.

        This is intentionally lightweight; replace with your repo's more complete
        implementation if available.
        """

        def __init__(
            self,
            initial_child_model_name: str,
            child_tokenizer_name: Optional[str] = None,
            teacher_agent: Optional[Any] = None,
            teacher_model_name: Optional[str] = None,
            output_dir: str = "./orpo_out",
            final_json_path: Optional[str] = None,
            child_generation_args: Optional[Dict[str, Any]] = None,
            teacher_generation_args: Optional[Dict[str, Any]] = None,
            sample_vars: Optional[Dict[str, Any]] = None,
            training_config: Optional[Dict[str, Any]] = None,
        ):
            self.initial_child = initial_child_model_name
            self.child_tokenizer_name = child_tokenizer_name or initial_child_model_name
            self.teacher_agent = teacher_agent
            self.teacher_model_name = teacher_model_name
            self.output_dir = output_dir
            self.final_json_path = final_json_path
            self.child_generation_args = child_generation_args or {}
            self.teacher_generation_args = teacher_generation_args or {}
            self.sample_vars = sample_vars or {}
            self.training_config = training_config or {}

            os.makedirs(self.output_dir, exist_ok=True)

        def _load_child_model_and_tokenizer(self):
            print(f"[IterativeORPOTrainer] Loading child from {self.initial_child}")
            tok = AutoTokenizer.from_pretrained(self.child_tokenizer_name, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(self.initial_child, trust_remote_code=True)
            model.eval()
            return model, tok

        def _query_teacher(self, prompt: str) -> str:
            # If a ParlAI teacher agent was supplied, run it's observe/act loop
            if self.teacher_agent is not None:
                # ParlAI agents expect an observation dict
                obs = {"text": prompt, "episode_done": False}
                try:
                    self.teacher_agent.observe(obs)
                    out = self.teacher_agent.act()
                    return out.get("text", "")
                except Exception as e:
                    print(f"[IterativeORPOTrainer] teacher_agent failed: {e}")
                    return ""
            elif self.teacher_model_name:
                # fallback: use a HF generative model to produce teacher response
                tok = AutoTokenizer.from_pretrained(self.teacher_model_name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(self.teacher_model_name, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")
                inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
                gen = model.generate(**inputs, max_new_tokens=self.teacher_generation_args.get("max_new_tokens", 64))
                out = tok.decode(gen[0], skip_special_tokens=True)
                # naive: return suffix after prompt
                return out[len(prompt):].strip()
            else:
                raise RuntimeError("No teacher available (neither teacher_agent nor teacher_model_name supplied)")

        def _generate_child_prompts(self, child_model, child_tokenizer, n_samples: int = 16) -> List[str]:
            prompts = []
            # generate from random seeds / short prompts sampled from training data
            for i in range(n_samples):
                seed_prompt = self.sample_vars.get("seed_prompt", "Q: How are you?\nA:")
                inputs = child_tokenizer(seed_prompt, return_tensors="pt", truncation=True, max_length=256).to(child_model.device)
                gen = child_model.generate(**inputs, max_new_tokens=self.child_generation_args.get("max_new_tokens", 32), do_sample=True, top_k=50, top_p=0.95, temperature=0.8)
                full = child_tokenizer.decode(gen[0], skip_special_tokens=True)
                prompts.append(full)
            return prompts

        def _fine_tune_child(self, train_texts: List[str], save_dir: str):
            # Very small finetune: create dataset from teacher responses and train using HF Trainer
            from datasets import Dataset
            tok = AutoTokenizer.from_pretrained(self.child_tokenizer_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(self.initial_child, trust_remote_code=True)

            def encode(example):
                enc = tok(example["text"], truncation=True, max_length=self.training_config.get("max_length", 512))
                enc["labels"] = enc["input_ids"].copy()
                return enc

            ds = Dataset.from_dict({"text": train_texts})
            ds = ds.map(encode, batched=False)
            args = TrainingArguments(
                output_dir=save_dir,
                num_train_epochs=self.training_config.get("num_train_epochs", 1),
                per_device_train_batch_size=self.training_config.get("per_device_train_batch_size", 2),
                logging_steps=self.training_config.get("logging_steps", 10),
                save_strategy="no",
                learning_rate=self.training_config.get("learning_rate", 5e-6),
            )
            trainer = Trainer(model=model, args=args, train_dataset=ds)
            trainer.train()
            model.save_pretrained(save_dir)
            tok.save_pretrained(save_dir)
            return save_dir

        def run_iterative_training(self, num_iterations: int = 1):
            child_model, child_tok = self._load_child_model_and_tokenizer()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            child_model = child_model.to(device)

            for it in range(1, num_iterations + 1):
                print(f"[IterativeORPOTrainer] Iteration {it}/{num_iterations}")
                # Create prompts from child and query teacher
                prompts = self._generate_child_prompts(child_model, child_tok, n_samples=self.sample_vars.get("n_prompts", 8))
                teacher_responses = []
                for p in prompts:
                    t = self._query_teacher(p)
                    # store (prompt + teacher response) as training example — naive formatting
                    teacher_responses.append((p + "\n" + t).strip())

                # Fine-tune the child model on teacher responses
                iter_dir = os.path.join(self.output_dir, f"iteration_{it}")
                os.makedirs(iter_dir, exist_ok=True)
                print(f"[IterativeORPOTrainer] Fine-tuning child on {len(teacher_responses)} teacher responses to {iter_dir}")
                self._fine_tune_child(["\n".join(teacher_responses)], iter_dir)

            # Write metadata
            if self.final_json_path:
                meta = {"initial_child": self.initial_child, "output_dir": self.output_dir, "iterations": num_iterations}
                with open(self.final_json_path, "w") as f:
                    json.dump(meta, f, indent=2)


# -------------------------
# Compute per-device batch size helper
# -------------------------

def compute_per_device_batch_size(global_bs: int, num_devices: int, accumulation_steps: int):
    p = global_bs / (num_devices * accumulation_steps)
    if int(p) != p:
        raise ValueError("Per-device batch size must be integer; adjust GLOBAL_BATCH_SIZE/num_devices/accumulation_steps")
    return int(p)


# -------------------------
# Custom callbacks
# -------------------------
class CustomCheckpointingCallback(TrainerCallback):
    def __init__(self, total_steps: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.total_steps = total_steps
        self.next_checkpoint_step = 0

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # simple periodic checkpoint every X steps (example)
        period = max(1, self.total_steps // 10)
        if state.global_step >= (self.next_checkpoint_step + 1) * period:
            print(f"[CheckpointCallback] Triggering checkpoint at step {state.global_step}")
            control.should_save = True
            self.next_checkpoint_step += 1
        return control


class ORPOTriggerCallback(TrainerCallback):
    def __init__(self, trainer_ref, total_steps: int, seq_len: int, fractions_to_teacher: Dict[float, str],
                 orpo_phase_config: Dict[str, Any], orpo_output_dir: str, push_to_hub: bool = False,
                 hf_hub_base: Optional[str] = None):
        super().__init__()
        self._trainer_ref = trainer_ref
        self.seq_len = seq_len
        self.total_steps = total_steps
        self.total_tokens = total_steps * GLOBAL_BATCH_SIZE * seq_len
        self.fractions_to_teacher = dict(sorted(fractions_to_teacher.items()))
        self.triggered = {frac: False for frac in self.fractions_to_teacher}
        self.orpo_phase_config = orpo_phase_config
        self.orpo_output_dir = orpo_output_dir
        self.push_to_hub = push_to_hub
        self.hf_hub_base = hf_hub_base  # string prefix to create child repo ids
        os.makedirs(self.orpo_output_dir, exist_ok=True)

    def register_trainer(self, trainer):
        self._trainer_ref = trainer

    def _is_main_process(self):
        if self._trainer_ref is not None:
            try:
                return self._trainer_ref.is_world_process_zero()
            except Exception:
                pass
        return int(os.environ.get("RANK", "0")) == 0

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        tokens_seen = state.global_step * GLOBAL_BATCH_SIZE * self.seq_len
        for frac, teacher_label in self.fractions_to_teacher.items():
            if self.triggered.get(frac, False):
                continue
            if tokens_seen >= int(self.total_tokens * frac):
                print(f"[ORPOTrigger] Reached {frac*100:.0f}% tokens -> teacher {teacher_label}")
                control.should_save = True
                # Save a checkpoint for ORPO trainer to consume
                if self._is_main_process():
                    ckpt_dir = os.path.join(args.output_dir, f"orpo_checkpoint_{int(frac*100)}pct_step{state.global_step}")
                    print(f"[ORPOTrigger] Main saving checkpoint to {ckpt_dir}")
                    trainer = self._trainer_ref
                    trainer.save_model(ckpt_dir)
                    if getattr(trainer, "tokenizer", None) is not None:
                        trainer.tokenizer.save_pretrained(ckpt_dir)
                    torch.save(trainer.args, os.path.join(ckpt_dir, "training_args.bin"))

                    # Launch ORPO and wait for result
                    try:
                        new_child_dir = self._run_orpo_phase(ckpt_dir, teacher_label, frac)
                        if new_child_dir:
                            print(f"[ORPOTrigger] ORPO returned new model at {new_child_dir}")
                            # Optionally push the model to HF hub for continued training
                            if self.push_to_hub and self.hf_hub_base:
                                repo_id = f"{self.hf_hub_base}/orpo_child_{int(frac*100)}pct"
                                push_model_to_hub(new_child_dir, repo_id)

                            # Load weights into trainer.model
                            self._load_new_weights_into_trainer(new_child_dir, trainer)
                            print("[ORPOTrigger] Loaded new ORPO weights into trainer")
                        else:
                            print("[ORPOTrigger] ORPO did not produce a new child dir; continuing")
                    except Exception as e:
                        print(f"[ORPOTrigger] ORPO phase failed: {e}")
                else:
                    print("[ORPOTrigger] Non-main process waiting for ORPO to complete...")

                # barrier for distributed
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.barrier()

                self.triggered[frac] = True
        return control

    def _prepare_parlai_teacher(self, cefr_level: str):
        if not PARLAI_AVAILABLE:
            raise RuntimeError("ParlAI/ControllableBlender not available. Install and ensure controllable_blender import succeeds.")
        # load agent_opt and set rerank_cefr dynamically
        agent_opt = json.load(open("blender_3B.opt", "r"))
        agent_opt["inference"] = "rerank"
        agent_opt["beam_size"] = 20
        agent_opt["topk"] = 40
        agent_opt["rerank_cefr"] = cefr_level
        agent_opt["rerank_tokenizer"] = "distilroberta-base"
        agent_opt["rerank_model"] = "complexity_model"
        agent_opt["rerank_model_device"] = "cuda"
        agent_opt["penalty_stddev"] = 2
        agent_opt["filter_path"] = agent_opt.get("filter_path", "data/filter.txt")
        # Ensure the blender data is downloaded
        parlai_blender_download(agent_opt["datapath"])
        teacher_agent = ControllableBlender(agent_opt)
        return teacher_agent

    def _run_orpo_phase(self, checkpoint_dir: str, teacher_label: str, frac: float) -> Optional[str]:
        # instantiate teacher — here we always prefer ParlAI CEFR-controlled agent
        teacher_agent = None
        try:
            teacher_agent = self._prepare_parlai_teacher(teacher_label)
        except Exception as e:
            print(f"[ORPOTrigger] Failed to create ParlAI teacher: {e}; falling back to HF model id if provided in config")

        # Construct IterativeORPOTrainer
        orpo_cfg = self.orpo_phase_config.copy()
        trainer_kwargs = {
            "initial_child_model_name": checkpoint_dir,
            "child_tokenizer_name": orpo_cfg.get("child_tokenizer_name", checkpoint_dir),
            "teacher_agent": teacher_agent,
            "teacher_model_name": orpo_cfg.get("teacher_model_name"),
            "output_dir": os.path.join(self.orpo_output_dir, f"phase_{int(frac*100)}"),
            "final_json_path": os.path.join(self.orpo_output_dir, f"phase_{int(frac*100)}", "final_orpo_data.json"),
            "child_generation_args": orpo_cfg.get("child_generation_args", {}),
            "teacher_generation_args": orpo_cfg.get("teacher_generation_args", {}),
            "sample_vars": orpo_cfg.get("sample_vars", {}),
            "training_config": orpo_cfg.get("training_config", {}),
        }
        orpo_trainer = IterativeORPOTrainer(**trainer_kwargs)
        orpo_trainer.run_iterative_training(num_iterations=orpo_cfg.get("orpo_iterations", 1))

        candidate_dir = os.path.join(trainer_kwargs["output_dir"], "iteration_1")
        if os.path.isdir(candidate_dir):
            return candidate_dir
        if os.path.isdir(trainer_kwargs["output_dir"]):
            return trainer_kwargs["output_dir"]
        return None

    def _load_new_weights_into_trainer(self, new_model_dir: str, trainer):
        print(f"[ORPOTrigger] Loading new weights from {new_model_dir}")
        new_model = OPTForCausalLM.from_pretrained(new_model_dir)
        new_state = new_model.state_dict()
        model_to_load = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        model_to_load.load_state_dict(new_state, strict=False)
        del new_model
        torch.cuda.empty_cache()


# -------------------------
# Main training orchestration
# -------------------------
GLOBAL_BATCH_SIZE = 64
TRAIN_EPOCHS = 10


def train_model_with_orpo(
    model_type: str = "opt",
    seq_len: int = 128,
    use_deepspeed: bool = False,
    push_to_hub: bool = False,
    dry_run: bool = False,
    num_devices: int = 1,
    accumulation_steps: int = 1,
    use_warmup: bool = False,
    special_id: str = "",
    orpo_enabled: bool = True,
    curriculum_id: int = 1,
    hf_hub_base: Optional[str] = None,
):
    per_device_batch_size = compute_per_device_batch_size(GLOBAL_BATCH_SIZE, num_devices, accumulation_steps)

    print(f"Loading dataset for seq_len={seq_len} (dry_run={dry_run})")
    ds = load_dataset(f"babylm-seqlen/train_100M_{seq_len}_single_shuffle")
    ds = ds.map(lambda x: {"labels": x["input_ids"]}, num_proc=4)
    train_dataset = ds["train"]
    if dry_run:
        train_dataset = train_dataset.select(range(200))

    suffix = ("-warmup" if use_warmup else "") + (f"-{special_id}" if special_id else "")
    output_dir = f"./checkpoints/{model_type}-babylm-{seq_len}{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    run_name = f"{model_type}_babylm_{seq_len}{suffix}_curr{curriculum_id}"

    if model_type == "opt":
        config = OPTConfig(vocab_size=50257, hidden_size=768, num_attention_heads=12, num_hidden_layers=12, ffn_dim=3072, max_position_embeddings=seq_len)
        model = OPTForCausalLM(config)
    else:
        raise NotImplementedError("Only 'opt' is supported in this demo.")

    # initialize wandb if desired (disabled for dry-run to avoid accidental logging)
    if int(os.environ.get("RANK", "0")) == 0:
        import wandb
        wandb.init(entity="babylm-seqlen", project=f"{model_type}-models", name=run_name, mode="disabled" if dry_run else "online")

    total_steps = TRAIN_EPOCHS * len(train_dataset) // GLOBAL_BATCH_SIZE
    warmup_steps = int(total_steps * 0.05) if use_warmup else 0

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=accumulation_steps,
        num_train_epochs=TRAIN_EPOCHS,
        eval_strategy="no",
        save_strategy="no",
        bf16=False,
        report_to="none",
        logging_steps=max(total_steps // 1000, 1),
    )

    # Instantiate your CustomTrainer if available, otherwise use HF Trainer as a placeholder
    try:
        from custom_trainer import CustomTrainer  # if user has CustomTrainer in repo
        TrainerClass = CustomTrainer
    except Exception:
        TrainerClass = Trainer

    trainer = TrainerClass(model=model, args=training_args, train_dataset=train_dataset)

    # checkpointing
    checkpoint_cb = CustomCheckpointingCallback(total_steps, seq_len)
    trainer.add_callback(checkpoint_cb)

    # curricula options
    curricula = {
        1: {0.20: "A2", 0.60: "B2", 1.00: "C1"},
        2: {0.20: "C1", 0.60: "B2", 1.00: "A2"},
        3: {0.20: "B2", 0.60: "A2", 1.00: "C1"},
        4: {0.20: "B2", 0.60: "C1", 1.00: "A2"},
    }
    fractions_to_teacher = curricula.get(curriculum_id, curricula[1])

    if orpo_enabled:
        orpo_phase_config = {
            # no fixed teacher_map: we will use ControllableBlender CEFR levels selected above
            "child_tokenizer_name": None,
            "child_generation_args": {"max_new_tokens": 50, "do_sample": True, "top_k": 50, "top_p": 0.95, "temperature": 0.8},
            "teacher_generation_args": {"max_new_tokens": 50, "do_sample": False, "temperature": 0.3},
            "sample_vars": {"n_prompts": 8, "seed_prompt": ""},
            "training_config": {
                "logging_steps": 10,
                "save_steps": 200,
                "eval_steps": 200,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "num_train_epochs": 1,
                "learning_rate": 5e-6,
                "max_length": 512,
            },
            "orpo_iterations": 1,
        }

        orpo_cb = ORPOTriggerCallback(
            trainer_ref=None,
            total_steps=total_steps,
            seq_len=seq_len,
            fractions_to_teacher=fractions_to_teacher,
            orpo_phase_config=orpo_phase_config,
            orpo_output_dir=os.path.join(output_dir, "orpo_phases"),
            push_to_hub=push_to_hub,
            hf_hub_base=hf_hub_base,
        )
        orpo_cb.register_trainer(trainer)
        trainer.add_callback(orpo_cb)

    print(f"Starting training; ORPO enabled? {orpo_enabled}; curriculum {curriculum_id}")
    start = time.time()
    trainer.train()
    dur = time.time() - start
    print(f"Training completed in {dur:.1f}s")


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["opt"], default="opt")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--use_warmup", action="store_true")
    parser.add_argument("--special_id", type=str, default="")
    parser.add_argument("--no_orpo", action="store_true")
    parser.add_argument("--curriculum_id", type=int, choices=[1,2,3,4], default=1)
    parser.add_argument("--push_to_hub", action="store_true", help="Push ORPO-produced child models to HF hub")
    parser.add_argument("--hf_hub_base", type=str, default=None, help="Hub repo prefix (username or org) to create child repos under, e.g. 'yourname' -> yourname/orpo_child_20pct")
    args = parser.parse_args()

    train_model_with_orpo(
        model_type=args.model_type,
        seq_len=args.seq_len,
        dry_run=args.dry_run,
        num_devices=args.num_devices,
        accumulation_steps=args.accumulation_steps,
        use_warmup=args.use_warmup,
        special_id=args.special_id,
        orpo_enabled=not args.no_orpo,
        curriculum_id=args.curriculum_id,
        push_to_hub=args.push_to_hub,
        hf_hub_base=args.hf_hub_base,
    )
