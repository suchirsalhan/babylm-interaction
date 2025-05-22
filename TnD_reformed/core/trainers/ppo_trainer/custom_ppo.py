import math
import time
import typing
import numpy as np
from typing import Optional, Union, List, Any, Callable

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, HfArgumentParser, PreTrainedTokenizerBase, \
    AutoModelForCausalLM
from datasets import Dataset

from trl import PPOTrainer, AutoModelForCausalLMWithValueHead, PreTrainedModelWrapper
from trl.core import (
    LengthSampler,
    WANDB_PADDING,
    PPODecorators,
    clip_by_value,
    convert_to_scalar,
    entropy_from_logits,
    flatten_dict,
    logprobs_from_logits,
    masked_mean,
    masked_var,
    masked_whiten,
    set_seed,
    stack_dicts,
    stats_to_np,
)
from trl.models import (
    unwrap_model_for_generation
)

from core.trainers.ppo_trainer.config import CustomPPOConfig


class CustomPPOTrainer(PPOTrainer):
    def __init__(
            self,
            config: Optional[CustomPPOConfig] = None,
            child_model: Optional[PreTrainedModelWrapper] = None,
            # TODO: the reward model only needs to give a value, doesn't have to be a model it self
            reward_model: Optional[Any] = None,
            teacher_model: Optional[PreTrainedModelWrapper] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            teacher_tokenizer: Optional[PreTrainedTokenizerBase] = None,
            dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            num_shared_layers: Optional[int] = None,
            lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            training_data_collator: Optional[typing.Callable] = None,
    ):
        def collator(data):
            return dict((key, [d[key] for d in data]) for key in data[0])

        self.reward_model = reward_model
        self.teacher_tokenizer = teacher_tokenizer
        self.teacher_model = teacher_model

        super(CustomPPOTrainer, self).__init__(config=config, model=child_model, ref_model=teacher_model, tokenizer=tokenizer, dataset=dataset, optimizer=optimizer, data_collator=collator,
                                                num_shared_layers=num_shared_layers, lr_scheduler=lr_scheduler)
        self.device = self.accelerator.device

    @PPODecorators.empty_device_cache()
    def step(
        self,
        input_prompts : list[str],
        child_queries: List[torch.Tensor]=None,
        child_responses: List[torch.Tensor]=None,
        teacher_queries: List[torch.Tensor]=None,
        teacher_responses: List[torch.Tensor]=None,
        scores: List[torch.Tensor]=None,
        child_response_masks: Optional[List[torch.Tensor]] = None,
        teacher_response_masks: Optional[List[torch.Tensor]] = None,
    ):
        # Step1: generate response from child model
        # if queries already provided, skip this step, otherwise generate queries from input_prompts
        if child_queries and child_responses:
            pass
        else:
            child_queries = [self.tokenizer.encode(text, return_tensors="pt").squeeze().to(self.device) for text in input_prompts]
            #TODO: add generation args in config
            child_generation_args = self.config.child_generation_args
            child_responses = self.generate_child_response(child_queries, **child_generation_args)

        
        # Step2: generate teacher response
        if teacher_queries and teacher_responses:
            pass
        else:
            teacher_queries = [self.teacher_tokenizer.encode(text, return_tensors="pt").squeeze().to(self.device) for text in input_prompts]
            teacher_generation_args = self.config.teacher_generation_args
            teacher_responses = self.generate_teacher_response(teacher_queries, **teacher_generation_args)

        # Step3: compute scores from reward model
        if scores:
            pass
        else:
            # TODO: verify reward model has the method `compute_rewards`
            scores = self.reward_model.compute_rewards(child_queries, child_responses, teacher_queries, teacher_responses, self.tokenizer, self.teacher_tokenizer)
            # move score to device to avoid device mismatch
            scores = [score.to(self.current_device) for score in scores]
            
        # Step4: verify input size correct with batch size and score scaling/clipping
        # verify input size for both child and teacher
        bs = self.config.batch_size
        child_queries, child_responses, scores, child_response_masks = self._step_safety_checker(
            bs, child_queries, child_responses, scores, child_response_masks
        )
        teacher_queries, teacher_responses, scores, teacher_response_masks = self._step_safety_checker(
            bs, teacher_queries, teacher_responses, scores, teacher_response_masks
        )
        # score scaling/clipping
        scores = torch.tensor(scores, device=self.current_device)
        if self.config.use_score_scaling:
            # Score scaling
            scores_mean, scores_std = self.running.update(scores)
            tensor_to_kwargs = dict(dtype=scores.dtype, device=scores.device)
            score_scaling_factor = self.running.std.to(**tensor_to_kwargs) + torch.finfo(scores.dtype).eps
            if self.config.use_score_norm:
                scores = (scores - self.running.mean.to(**tensor_to_kwargs)) / score_scaling_factor
            else:
                scores /= score_scaling_factor

        if self.config.score_clip is not None:
            # Score clipping
            scores_dtype = scores.dtype
            scores = torch.clip(scores.float(), -self.config.score_clip, self.config.score_clip).to(dtype=scores_dtype)
        
        # if we want to push best model to the hub
        if hasattr(self, "highest_reward"):
            if self.compare_step % self.config.compare_steps == 0:
                curr_mean_reward = scores.mean()
                # if the best reward ever seen
                if curr_mean_reward > self.highest_reward:
                    self.highest_reward = curr_mean_reward
                    # push model to hub
                    self.push_to_hub(**self.push_to_hub_kwargs)
            self.compare_step += 1
        
        # step4: prepare model inputs for child and teacher to get probs and logit
        timing = dict()
        t0 = time.time()

        t = time.time()

        child_model_inputs = self.prepare_model_inputs(child_queries, child_responses)
        teacher_model_inputs = self.prepare_model_inputs(teacher_queries, teacher_responses)

        if self.is_distributed:
            child_model_inputs = self.modify_distributed_inputs(child_model_inputs, self.tokenizer)
            teacher_model_inputs = self.modify_distributed_inputs(teacher_model_inputs, self.teacher_tokenizer)
        
        # step5: get logits and logprobs for child and teacher(additionally get value from child model linear layer)
        model_inputs_names = list(child_model_inputs.keys())

        full_kl_penalty = self.config.kl_penalty == "full"

        with torch.no_grad():
            all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
                self.model,
                child_queries,
                child_responses,
                child_model_inputs,
                response_masks=child_response_masks,
                return_logits=full_kl_penalty,
            )
            with self.optional_peft_ctx():
                teacher_logprobs, teacher_logits_or_none, _, _ = self.batched_forward_pass(
                    self.model if self.is_peft_model else self.ref_model,
                    teacher_queries,
                    teacher_responses,
                    teacher_model_inputs,
                    return_logits=full_kl_penalty,
                )
        timing["time/ppo/forward_pass"] = time.time() - t
        
        # step6: compute final rewards, advantages, use scores and teacher, child logits and logprobs, default combine score and kl divergence
        with torch.no_grad():
            t = time.time()
            if full_kl_penalty:
                active_full_logprobs = logprobs_from_logits(logits_or_none, None, gather=False)
                teacher_full_logprobs = logprobs_from_logits(teacher_logits_or_none, None, gather=False)

                rewards, non_score_reward, kls = self.compute_rewards(
                    scores, active_full_logprobs, teacher_full_logprobs, masks
                )
            else:
                rewards, non_score_reward, kls = self.compute_rewards(scores, all_logprobs, teacher_logprobs, masks)
            timing["time/ppo/compute_rewards"] = time.time() - t

            t = time.time()
            values, advantages, returns = self.compute_advantages(values, rewards, masks)
            timing["time/ppo/compute_advantages"] = time.time() - t

        # upcast to float32 to avoid dataset issues
        batch_dict = {
            "queries": child_queries,
            "responses": child_responses,
            "logprobs": all_logprobs.to(torch.float32),
            "values": values.to(torch.float32),
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
        }
        batch_dict.update(child_model_inputs)
        
        # step7: train child model using the queries, logits, rewards(all combine into one loss in the train_minibatch function)
        t = time.time()
        all_stats = []
        early_stop = False
        for _ in range(self.config.ppo_epochs):
            if early_stop:
                break
            b_inds = np.random.permutation(bs)
            for backward_batch_start in range(0, bs, self.config.backward_batch_size):
                backward_batch_end = backward_batch_start + self.config.backward_batch_size
                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
                    mini_batch_end = mini_batch_start + self.config.mini_batch_size
                    mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
                    mini_batch_dict = {
                        "logprobs": batch_dict["logprobs"][mini_batch_inds],
                        "values": batch_dict["values"][mini_batch_inds],
                        "masks": batch_dict["masks"][mini_batch_inds],
                        # hacks: the queries and responses are ragged.
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                        "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                        "advantages": batch_dict["advantages"][mini_batch_inds],
                        "returns": batch_dict["returns"][mini_batch_inds],
                    }
                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

                        logprobs, logits, vpreds, _ = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"],
                            mini_batch_dict["responses"],
                            model_inputs,
                            return_logits=True,
                        )
                        train_stats = self.train_minibatch(
                            mini_batch_dict["logprobs"],
                            mini_batch_dict["values"],
                            logprobs,
                            logits,
                            vpreds,
                            mini_batch_dict["masks"],
                            mini_batch_dict["advantages"],
                            mini_batch_dict["returns"],
                        )
                        all_stats.append(train_stats)

            # typically, early stopping is done at the epoch level
            if self.config.early_stopping:
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if early_stop:
                    break
        
        # step8: record stats
        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=teacher_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
            queries=child_queries,
            responses=child_responses,
            kls=kls,
        )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(
            stats["objective/kl"],
            self.config.batch_size * self.accelerator.num_processes,
        )

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats, scores
        
    def generate_child_response(
        self,
        query_tensor: Union[torch.Tensor, List[torch.Tensor]],
        length_sampler: Optional[Callable] = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        **generation_kwargs,
    ):
        """
        Generate response with the child model given the query tensor.

        Args:
            query_tensor (`torch.LongTensor`):
                A tensor of shape (`seq_len`) containing query tokens or a list of tensors of shape (`seq_len`).
            length_sampler (`Callable`, *optional*):
                Callable that returns the number of newly generated tokens.
            batch_size (`int`, *optional):
                Batch size used for generation, defaults to `4`.
            return_prompt (`bool`, *optional*):
                If set to `False` the prompt is not returned but only the newly generated tokens, defaults to `True`.
            generation_kwargs (dict[str, Any]):
                Keyword arguments for generation.

        Returns:
            `torch.LongTensor`: A tensor of shape (`batch_size`, `gen_len`) containing response tokens.
        """
        if isinstance(query_tensor, List):
            response = self._generate_batched(
                self.model,
                query_tensor,
                length_sampler=length_sampler,
                batch_size=batch_size,
                return_prompt=return_prompt,
                **generation_kwargs,
            )
        else:
            if len(query_tensor.shape) == 2:
                raise ValueError(
                    "query_tensor must be a tensor of shape (`seq_len`) or a list of tensors of shape (`seq_len`)"
                )

            if length_sampler is not None:
                generation_kwargs["max_new_tokens"] = length_sampler()

            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                response = unwrapped_model.generate(input_ids=query_tensor.unsqueeze(dim=0), **generation_kwargs)

            if not return_prompt and not self.is_encoder_decoder:
                response = response[:, query_tensor.shape[0] :]

        return response

    def generate_teacher_response(
        self,
        query_tensor: Union[torch.Tensor, List[torch.Tensor]],
        length_sampler: Optional[Callable] = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        **generation_kwargs,
    ):
        """
        Generate response with the teacher model given the query tensor.

        Args:
            query_tensor (`torch.LongTensor`):
                A tensor of shape (`seq_len`) containing query tokens or a list of tensors of shape (`seq_len`).
            length_sampler (`Callable`, *optional*):
                Callable that returns the number of newly generated tokens.
            batch_size (`int`, *optional*):
                Batch size used for generation, defaults to `4`.
            return_prompt (`bool`, *optional*):
                If set to `False` the prompt is not returned but only the newly generated tokens, defaults to `True`.
            generation_kwargs (dict[str, Any]):
                Keyword arguments for generation.

        Returns:
            `torch.LongTensor`: A tensor of shape (`batch_size`, `gen_len`) containing response tokens.
        """
        ref_model = self.model if self.is_peft_model else self.ref_model
        
        if isinstance(query_tensor, List):
            response = self._generate_batched(
                ref_model,
                query_tensor,
                length_sampler=length_sampler,
                batch_size=batch_size,
                return_prompt=return_prompt,
                **generation_kwargs,
            )
        else:
            if len(query_tensor.shape) == 2:
                raise ValueError(
                    "query_tensor must be a tensor of shape (`seq_len`) or a list of tensors of shape (`seq_len`)"
                )

            if length_sampler is not None:
                generation_kwargs["max_new_tokens"] = length_sampler()

            with unwrap_model_for_generation(
                ref_model, self.accelerator, is_peft_model=self.is_peft_model
            ) as unwrapped_model:
                response = unwrapped_model.generate(
                    input_ids=query_tensor.unsqueeze(dim=0), **generation_kwargs
                )

            if not return_prompt and not self.is_encoder_decoder:
                response = response[:, query_tensor.shape[0] :]

        return response

    def modify_distributed_inputs(self, model_inputs, tokenizer):
        pad_first = tokenizer.padding_side == "left"

        model_inputs["input_ids"] = self.accelerator.pad_across_processes(
            model_inputs["input_ids"],
            dim=1,
            pad_index=tokenizer.pad_token_id,
            pad_first=pad_first,
        )
        model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
            model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
        )
        if self.is_encoder_decoder:
            model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["decoder_input_ids"],
                dim=1,
                pad_index=tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["decoder_attention_mask"],
                dim=1,
                pad_index=0,
                pad_first=pad_first,
            )
        return model_inputs
    