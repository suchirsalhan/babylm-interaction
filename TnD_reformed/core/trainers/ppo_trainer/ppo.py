from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import torch
from torch import nn
import numpy as np
import time
import os
from transformers import Trainer, TrainingArguments, PreTrainedTokenizerBase, PreTrainedModel
from trl import PPOTrainer
from trl.core import PPODecorators, LengthSampler, masked_mean, masked_var, flatten_dict
from ...utils.ppo_utils import clip_by_value, logprobs_from_logits
from ...models import TeacherModel, StudentModel, RewardModel
from .custom_components import (
    CustomLossFunctions,
    CustomRewardFunctions,
    CustomGradientScaling,
    CustomMetrics
)
from .config import CustomPPOConfig

class CustomPPOTrainer(PPOTrainer):
    """
    Custom PPO trainer that extends TRL's PPOTrainer with additional functionality.
    
    Attributes:
        teacher: Teacher model
        student: Student model
        reward_model: Reward model
        config: Custom PPO configuration
        device: Device to train on
    """
    def __init__(
        self,
        teacher: TeacherModel,
        student: StudentModel,
        reward_model: RewardModel,
        config: CustomPPOConfig,
        train_dataset: torch.utils.data.Dataset,
        device: Optional[torch.device] = None,
    ):
        # Use student's tokenizer for the trainer as processing_class
        tokenizer = student.tokenizer
        
        super().__init__(
            args=config,
            model=student.model,
            ref_model=teacher.model,
            reward_model=reward_model,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            value_model=teacher.model,
        )
        self.teacher = teacher
        self.student = student
        self.reward_model = reward_model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize best metrics tracking
        self.best_val_loss = float('inf')
        self.best_reward = float('-inf')
        self.patience = config.patience_steps
        self.current_patience = config.patience_steps
        
        # Set up custom components based on config
        # self._setup_loss_function()
        # self._setup_reward_function()
        # self._setup_gradient_scaling()
        
        # Initialize training statistics
        self.training_stats = {}
        self.grad_info = None
        self.ppo_loss = None
        
        # Set up length sampler for generation
        self.query_length_sampler = LengthSampler(
            config.query_min_length,
            config.query_max_length + 1
        )
        
    def _setup_loss_function(self):
        """Set up the loss function based on config."""
        if self.args.loss_type == "kl_regularized":
            self.loss_fn = lambda *args, **kwargs: CustomLossFunctions.kl_regularized_loss(
                *args, **kwargs, kl_coeff=self.args.kl_coeff
            )
        elif self.args.loss_type == "entropy_regularized":
            self.loss_fn = lambda *args, **kwargs: CustomLossFunctions.entropy_regularized_loss(
                *args, **kwargs, entropy_coeff=self.args.entropy_coeff
            )
        else:
            raise ValueError(f"Unknown loss type: {self.args.loss_type}")
            
    def _setup_reward_function(self):
        """Set up the reward function based on config."""
        if self.args.reward_type == "teacher_guided":
            self.reward_fn = lambda *args, **kwargs: CustomRewardFunctions.teacher_guided_reward(
                *args, **kwargs, temperature=self.args.reward_temperature
            )
        elif self.args.reward_type == "mixed":
            self.reward_fn = lambda *args, **kwargs: CustomRewardFunctions.mixed_reward(
                *args, **kwargs, teacher_weight=self.args.teacher_weight
            )
        else:
            raise ValueError(f"Unknown reward type: {self.args.reward_type}")
            
    def _setup_gradient_scaling(self):
        """Set up gradient scaling based on config."""
        if self.args.gradient_scaling_type == "dynamic":
            self.gradient_scaling_fn = lambda *args, **kwargs: CustomGradientScaling.dynamic_gradient_scaling(
                *args, **kwargs,
                max_norm=self.args.max_grad_norm,
                min_scale=self.args.min_grad_scale
            )
        elif self.args.gradient_scaling_type == "layer_wise":
            self.gradient_scaling_fn = CustomGradientScaling.layer_wise_gradient_scaling
        else:
            raise ValueError(f"Unknown gradient scaling type: {self.args.gradient_scaling_type}")
            
    def compute_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute rewards using the custom reward function."""
        # Get teacher and student logits
        teacher_logits = self.teacher.get_teacher_logits(input_ids)
        student_logits = self.student.get_student_logits(input_ids)
        
        # Compute rewards using custom reward function
        rewards = self.reward_fn(
            input_ids,
            attention_mask,
            teacher_logits,
            student_logits,
            self.reward_model
        )
        
        # Apply length reward if configured
        if self.args.length_reward_coef is not None:
            lengths = attention_mask.sum(dim=-1).float()
            rewards = rewards + self.args.length_reward_coef * lengths
            
        # Apply score clipping if configured
        if self.args.score_clip is not None:
            rewards = torch.clamp(rewards, -self.args.score_clip, self.args.score_clip)
            
        return rewards
        
    def compute_loss(
        self,
        logprobs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        old_logprobs: torch.Tensor,
        old_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        lm_inputs: Optional[torch.Tensor] = None,
        lm_loss_coef: float = 0.0,
        **kwargs
    ) -> torch.Tensor:
        """Compute loss using the custom loss function."""
        # Get teacher and student logits
        teacher_logits = self.teacher.get_teacher_logits(kwargs.get('input_ids'))
        student_logits = self.student.get_student_logits(kwargs.get('input_ids'))
        
        # Compute PPO loss using custom loss function
        ppo_loss = self.loss_fn(
            teacher_logits,
            student_logits,
            rewards,
            attention_mask
        )
        
        # Add language modeling loss if configured
        if lm_loss_coef > 0 and lm_inputs is not None:
            lm_loss = self._compute_lm_loss(lm_inputs)
            loss = lm_loss_coef * lm_loss + (1 - lm_loss_coef) * ppo_loss
        else:
            loss = ppo_loss
        
        # Apply gradient scaling
        loss = self.gradient_scaling_fn(loss, self.student.parameters())
        
        return loss
        
    def _compute_lm_loss(self, lm_inputs: torch.Tensor) -> torch.Tensor:
        """Compute language modeling loss."""
        batch = {"input_ids": lm_inputs}
        batch = self.tokenizer.pad(batch, padding=True, return_tensors="pt").to(self.device)
        labels = batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        self.student.train()
        lm_output = self.student.pretrained_model(**batch)
        return lm_output["loss"]
        
    def compute_metrics(
        self,
        logprobs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        old_logprobs: torch.Tensor,
        old_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Compute additional metrics."""
        metrics = {}
        
        # Get teacher and student logits
        teacher_logits = self.teacher.get_teacher_logits(kwargs.get('input_ids'))
        student_logits = self.student.get_student_logits(kwargs.get('input_ids'))
        
        # Compute requested metrics
        for metric_name in self.args.metrics:
            if metric_name == "kl_divergence":
                metrics[metric_name] = CustomMetrics.calculate_kl_divergence(
                    teacher_logits,
                    student_logits,
                    attention_mask
                )
            elif metric_name == "entropy":
                metrics[metric_name] = CustomMetrics.calculate_entropy(
                    student_logits,
                    attention_mask
                )
            elif metric_name == "reward_stats":
                metrics.update(CustomMetrics.calculate_reward_stats(
                    rewards,
                    attention_mask
                ))
                
        return metrics
        
    def generate(
        self,
        batch: Dict[str, torch.Tensor],
        use_queries: bool = True,
        **generation_kwargs
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Generate responses with optional query conditioning."""
        generation_kwargs.update({
            "min_length": -1,
            "max_new_tokens": self.args.output_max_length,
            "top_k": self.args.generation_top_k,
            "top_p": self.args.generation_top_p,
            "temperature": self.args.generation_temperature,
            "do_sample": self.args.generation_do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "num_beams": self.args.generation_num_beams,
            "num_beam_groups": self.args.generation_num_beam_groups,
        })
        
        if use_queries:
            caregiver_utts = batch["input_ids"]
            batch_query_length = self.query_length_sampler() + 1  # +1 for BOS token
            query_tensors = [utt[:batch_query_length] for utt in caregiver_utts]
        else:
            bos_tensor = torch.tensor([self.tokenizer.bos_token_id], device=self.device)
            query_tensors = self.args.batch_size * [bos_tensor]
            
        response_tensors = self.generate(query_tensors, return_prompt=False, **generation_kwargs)
        
        batch["query"] = [self.tokenizer.decode(q, skip_special_tokens=True) for q in query_tensors]
        batch["utterance"] = [self.tokenizer.decode(torch.cat((q, r)), skip_special_tokens=True) 
                            for q, r in zip(query_tensors, response_tensors)]
                            
        return batch, response_tensors, query_tensors

    def loss(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        mask: torch.LongTensor,
        advantages: torch.FloatTensor,
        returns: torch.FloatTensor,
    ):
        """Calculate policy and value losses with custom modifications."""
        vpredclipped = clip_by_value(
            vpreds,
            values - self.args.cliprange_value,
            values + self.args.cliprange_value,
        )

        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), mask)

        ratio = torch.exp(logprobs - old_logprobs)
        clipped = torch.clamp(ratio, 1.0 - self.args.cliprange, 1.0 + self.args.cliprange)
        
        # Track clipping statistics
        celing_clipfrac = masked_mean(torch.gt(ratio, clipped).float(), mask).detach().item()
        floor_clipfrac = masked_mean(torch.lt(ratio, clipped).float(), mask).detach().item()
        celing_clipfrac_value = masked_mean(torch.gt(vpreds, vpredclipped).float(), mask).detach().item()
        floor_clipfrac_value = masked_mean(torch.lt(vpreds, vpredclipped).float(), mask).detach().item()
        
        # Log clipping statistics
        if hasattr(self, 'logger'):
            self.logger.log({
                "clipped_ratio_celing_hit_ratio": celing_clipfrac,
                "clipped_ratio_floor_hit_ratio": floor_clipfrac,
                "clipped_value_celing_hit_ratio": celing_clipfrac_value,
                "clipped_value_floor_hit_ratio": floor_clipfrac_value
            })

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.args.cliprange, 1.0 + self.args.cliprange)

        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), mask)

        loss = pg_loss + self.args.vf_coef * vf_loss

        avg_ratio = masked_mean(ratio, mask).item()
        if avg_ratio > self.args.ratio_threshold:
            warnings.warn(
                f"The average ratio of batch ({avg_ratio:.2f}) exceeds threshold {self.args.ratio_threshold:.2f}. Skipping batch."
            )
            pg_loss = pg_loss * 0.0
            vf_loss = vf_loss * 0.0
            loss = loss * 0.0

        entropy = masked_mean(entropy_from_logits(logits), mask)
        entropy_loss = -entropy

        loss = loss + self.args.entropy_coeff * entropy_loss

        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
        policykl = masked_mean(old_logprobs - logprobs, mask)

        return_mean, return_var = masked_mean(returns, mask), masked_var(returns, mask)
        value_mean, value_var = masked_mean(values, mask), masked_var(values, mask)

        stats = dict(
            loss=dict(policy=pg_loss.detach(), value=vf_loss.detach(), entropy=entropy_loss.detach(), total=loss.detach()),
            policy=dict(
                entropy=entropy.detach(),
                approxkl=approxkl.detach(),
                policykl=policykl.detach(),
                clipfrac=pg_clipfrac.detach(),
                advantages=advantages.detach(),
                advantages_mean=masked_mean(advantages, mask).detach(),
                ratio=ratio.detach(),
            ),
            returns=dict(mean=return_mean.detach(), var=return_var.detach()),
            val=dict(
                vpred=masked_mean(vpreds, mask).detach(),
                error=masked_mean((vpreds - returns) ** 2, mask).detach(),
                clipfrac=vf_clipfrac.detach(),
                mean=value_mean.detach(),
                var=value_var.detach(),
            ),
        )
        return pg_loss, self.args.vf_coef * vf_loss, flatten_dict(stats)
        
    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model: Union[PreTrainedModel, nn.Module],
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
        return_logits: bool = False,
        response_masks: Optional[torch.Tensor] = None,
    ):
        """Custom batched forward pass with additional functionality."""
        bs = len(queries)
        fbs = self.args.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
                
            logits, _, values = model(**input_kwargs)

            if self.is_encoder_decoder:
                input_ids = input_kwargs["decoder_input_ids"]
                attention_mask = input_kwargs["decoder_attention_mask"]
            else:
                input_ids = input_kwargs["input_ids"]
                attention_mask = input_kwargs["attention_mask"]

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                if self.is_encoder_decoder:
                    start = 1
                    end = attention_mask[j, :].sum() - 1
                else:
                    start = len(query_batch[j]) - 1
                    if attention_mask[j, 0] == 0:  # offset left padding
                        start += attention_mask[j, :].nonzero()[0]
                    end = start + len(response_batch[j])
                    if response_masks is not None:
                        response_masks_batch[j] = torch.cat(
                            (torch.zeros_like(query_batch[j]), response_masks_batch[j])
                        )[1:]

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits
            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )
        
    def collect_ppo_loss(self, loss):
        """Collect PPO loss for tracking."""
        if self.ppo_loss is None:
            self.ppo_loss = loss
        else:
            self.ppo_loss += loss
            
    def get_ppo_loss(self):
        """Get accumulated PPO loss."""
        return self.ppo_loss
        
    @PPODecorators.empty_device_cache()
    def train_minibatch(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        mask: torch.LongTensor,
        advantages: torch.FloatTensor,
        returns: torch.FloatTensor,
    ):
        """Custom training minibatch with additional functionality."""
        loss_p, loss_v, train_stats = self.loss(
            old_logprobs, values, logits, vpreds, logprobs, mask, advantages, returns
        )
        entropy = train_stats['policy/entropy']
        loss_p = loss_p + self.args.entropy_coeff * entropy
        
        if hasattr(self, 'vloss_only') and self.vloss_only:
            loss = loss_v
        else:
            loss = loss_p + loss_v
            
        loss = loss * self.args.loss_scaling
        self.collect_ppo_loss(loss)
        
        self.accelerator.backward(loss)
        
        if self.args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()), self.args.max_grad_norm
            )

        t = time.time()
        self.optimizer.step()
        train_stats["time/ppo/optimizer_step"] = torch.Tensor([time.time() - t]).to(self.current_device)
        self.optimizer.zero_grad()
        
        return train_stats, loss
        
    def get_grads(self):
        """Get gradient information."""
        return self.grad_info
