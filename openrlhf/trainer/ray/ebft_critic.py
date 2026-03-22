import math
import os
from abc import ABC
from contextlib import nullcontext
from typing import Dict, Optional, Union

import deepspeed
import ray
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.trainer import get_scheduler

from openrlhf.models import Critic, ClassifierLoss, ClassifierAccuracy, CELoss
from openrlhf.models.utils import build_strided_attention_mask_and_positions, masked_mean
from openrlhf.trainer.ppo_utils.experience_maker import Experience
from openrlhf.utils.embedding_utils import (
    compute_cf_discrepancy_loss,
    prepare_distribution_embeddings_from_split_hidden_states,
)
from openrlhf.utils import get_tokenizer
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.deepspeed.deepspeed_utils import offload_deepspeed_states, reload_deepspeed_states
from openrlhf.utils.run_config_utils import write_run_config

from ..ppo_utils import EBFTNaiveReplayBuffer
from .launcher import BaseModelActor

logger = init_logger(__name__)


class CriticEBFTTrainer(ABC):
    def __init__(
        self,
        strategy,
        critic: torch.nn.Module,
        ema_model: Critic,
        critic_optim: Optimizer,
        critic_scheduler,
        ema_beta: float = 0.992,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        value_clip: float = 0.2,
        dataloader_pin_memory: bool = True,
        **kwargs,
    ):
        self.strategy = strategy
        self.args = strategy.args
        self.ema_beta = ema_beta
        self.critic = critic
        self.ema_model = ema_model
        self.critic_optim = critic_optim
        self.critic_scheduler = critic_scheduler
        self.micro_train_batch_size = micro_train_batch_size
        self.buffer_limit = buffer_limit
        self.buffer_cpu_offload = buffer_cpu_offload
        self.value_clip = value_clip
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_epochs = self.args.max_epochs

        self.replay_buffer = EBFTNaiveReplayBuffer(
            micro_train_batch_size,
            buffer_limit,
            buffer_cpu_offload,
            getattr(self.args, "packing_samples", False),
            self.args.use_dynamic_batch,
        )

        self.critic_loss_fn = ClassifierLoss()
        self.critic_classifier_accuracy_fn = ClassifierAccuracy()

        self.aux_loss_fn = CELoss()

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8
        self._feature_adapter_init = self._snapshot_named_params(self.critic, "feature_adapter")

    def ppo_train(self):
        # replay buffer may be empty at first, we should rebuild at each training
        if self.args.use_dynamic_batch:
            self.replay_buffer.setup_dynamic_batch(self.strategy)

        not_shuffle = (
            self.strategy.ring_attn_group is not None
            or self.args.ds_tensor_parallel_size > 1
            or self.args.use_dynamic_batch
        )
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=not not_shuffle,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for step, experience in enumerate(pbar):
                experience.to_device(device)
                status = self.training_step(experience, step)

                # for DP
                status = self.strategy.all_reduce(status)
                train_critic = status.pop("train_critic")

                status_list.append(status)
                pbar.set_postfix(status)

        if not train_critic:
            # keep scheduler in sync even when critic is frozen
            if hasattr(self.critic_scheduler, "step"):
                self.critic_scheduler.step()

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean

    def training_step(self, experience: Experience, step: int) -> Dict[str, float]:
        self.critic.train()
        device = torch.cuda.current_device()

        # Extract sequence tensors from experience
        prompts = experience.prompts  # X: Original full prompt
        full_sequences = experience.full_sequences  # Full sequence (prompt + generated)

        # Decide whether critic should train this step; always train.
        train_critic = True

        # Whether we should do a gradient-carrying critic forward/backward on this step.
        # Note: if both critic LRs are 0, we treat the critic as frozen and skip autograd entirely
        # (still allowing forward passes for logging/metrics).
        freeze_critic = (float(getattr(self.args, "critic_learning_rate", 0.0)) == 0.0) and (
            float(getattr(self.args, "critic_lr_head", None) or 0.0) == 0.0
        )
        train_critic = not freeze_critic

        # Calculate sequence dimensions
        prompt_length = prompts.shape[1]  # Length of original prompt
        batch_size = prompts.shape[0]
        generate_max_len = self.args.generate_max_len  # Total tokens to generate
        context_length = self.args.context_max_len   # Total context length used for generating each block
        stride = self.args.stride  # Stride between blocks
        num_blocks = (prompt_length - generate_max_len - context_length )// stride + 1  # Number of prediction blocks
        doc_ids = experience.doc_ids  # B, S
        qa_masks = experience.qa_masks  # B, S
        attention_mask, pos_ids = build_strided_attention_mask_and_positions(
            full_sequence_length=full_sequences.size(1),  # Total sequence length
            prompt_length=prompts.size(1),  # Original prompt length
            context_length=context_length,
            generation_step=generate_max_len,  # Number of tokens generated
            max_generation_length=generate_max_len,  # Total number of tokens to generate
            stride=stride,
            num_blocks=num_blocks,
            device=device,
            doc_ids=doc_ids[:,:prompts.size(1)],
            document_masking=self.args.document_masking,
        )

        def _maybe_override_attention_impl(model_wrapper):
            restore_impl = None
            if (
                attention_mask is not None
                and attention_mask.dim() == 4
                and getattr(self.args, "gradient_checkpointing", False)
            ):
                current_impl = model_wrapper.get_attention_implementation()
                if current_impl == "flash_attention_2":
                    restore_impl = current_impl
                    model_wrapper.set_attention_implementation("eager")
            return restore_impl

        # If we are not training the critic on this step, avoid building an autograd graph.
        # This saves memory/compute when critic LR is 0.
        _forward_ctx = nullcontext() if train_critic else torch.no_grad()
        restore_attention_impl = _maybe_override_attention_impl(self.critic)
        grad_norm_value: Optional[float] = None
        try:
            with _forward_ctx:
                gt_hidden_states, gen_hidden_states, gt_classifier_logits, gen_classifier_logits, _ = self.critic(
                    full_sequences.to(device),
                    attention_mask.to(device),
                    pos_ids.to(device),
                    return_classifier_logits=True,
                    context_length=context_length,
                    prompt_length=prompt_length,
                    generate_max_len=generate_max_len,
                    stride=stride,
                    num_blocks=num_blocks,
                    hidden_state_method=self.args.hidden_state_method,
                    qa_masks=qa_masks.to(device),
                    qa_masking=self.args.qa_masking,
                )

            if self.args.critic_sequence_level == "token":
                gt_classifier_logits = gt_classifier_logits.reshape(batch_size // self.args.n_samples_per_prompt, self.args.n_samples_per_prompt, num_blocks, generate_max_len, gt_classifier_logits.shape[-1])
                gen_classifier_logits = gen_classifier_logits.reshape(batch_size // self.args.n_samples_per_prompt, self.args.n_samples_per_prompt, num_blocks, generate_max_len, gen_classifier_logits.shape[-1])
            elif self.args.critic_sequence_level in ["concat", "mean_pooling", "last_token"]:
                gt_classifier_logits = gt_classifier_logits.reshape(batch_size // self.args.n_samples_per_prompt, self.args.n_samples_per_prompt, num_blocks, gt_classifier_logits.shape[-1])
                gen_classifier_logits = gen_classifier_logits.reshape(batch_size // self.args.n_samples_per_prompt, self.args.n_samples_per_prompt, num_blocks, gt_classifier_logits.shape[-1])

            # Compute prev metrics in eval mode for deterministic results
            self.critic.eval()
            with torch.no_grad():
                prev_metrics = self.critic_classifier_accuracy_fn(gt_classifier_logits.detach(), gen_classifier_logits.detach())
                prev_classifier_accuracy = prev_metrics[0]
                prev_classifier_precision = prev_metrics[1]
                prev_classifier_recall = prev_metrics[2]
                prev_classifier_f1_score = prev_metrics[3]
                prev_classifier_pr_auc = prev_metrics[4]
                prev_classifier_roc_auc = prev_metrics[5]
                prev_classifier_mean_prob_gap = prev_metrics[6]
                prev_classifier_prob_gen = prev_metrics[7]
                prev_classifier_pred_pos_frac = prev_metrics[8]
                prev_classifier_acc_thresh0p5 = prev_metrics[9]
            self.critic.train()

            critic_loss = self.critic_loss_fn(
                gt_classifier_logits,
                gen_classifier_logits,
                sequence_selection=self.args.classifier_sequence_selection,
            )

            critic_classifier_loss = self.args.critic_classifier_loss_coef * critic_loss
            critic_direct_discrepancy_loss = torch.zeros((), device=device, dtype=critic_classifier_loss.dtype)

            direct_discrepancy_coef = float(getattr(self.args, "critic_direct_discrepancy_coef", 0.0) or 0.0)
            direct_discrepancy_target = getattr(self.args, "critic_direct_discrepancy_target", "ema_gt")
            if direct_discrepancy_coef > 0.0 and getattr(self.args, "distribution_reward_type", "pointwise") == "cf_l1oo":
                # INSTRUMENTATION: 第一次进入 discrepancy 分支时的运行时状态
                if not hasattr(self, '_instrumentation_printed'):
                    self._instrumentation_printed = True
                    import torch.distributed as dist
                    is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
                    if is_rank0:
                        print("\n" + "="*80)
                        print("[G3 INSTRUMENTATION] First entry into discrepancy path")
                        print("="*80)
                        print(f"  full_sequences.shape: {full_sequences.shape}")
                        print(f"  qa_masks.shape: {qa_masks.shape}")
                        print(f"  prompt_length: {prompt_length}")
                        print(f"  context_length: {context_length}")
                        print(f"  generate_max_len: {generate_max_len}")
                        print(f"  stride: {stride}")
                        print(f"  num_blocks: {num_blocks}")
                        print(f"  gt_hidden_states.shape: {gt_hidden_states.shape}")
                        print(f"  gen_hidden_states.shape: {gen_hidden_states.shape}")
                        print(f"  direct_discrepancy_target: {direct_discrepancy_target}")
                        print(f"  ema_model is None: {self.ema_model is None}")
                        print("="*80 + "\n")
                
                num_feat = gt_hidden_states.shape[-2]
                gt_mask_flat = qa_masks[:, context_length:prompt_length].view(batch_size, prompt_length - context_length, 1, 1).repeat(1, 1, num_feat, 1)
                gen_mask_flat = qa_masks[:, prompt_length:].view(batch_size, generate_max_len * num_blocks, 1, 1).repeat(1, 1, num_feat, 1)
                gt_mask = gt_mask_flat.unfold(-3, generate_max_len, stride).permute(0, 1, 4, 2, 3)
                gen_mask = gen_mask_flat.reshape(batch_size, generate_max_len, num_blocks, num_feat, 1).transpose(-3, -4)

                target_gt_hidden_states = gt_hidden_states.detach()
                target_gt_mask = gt_mask
                if direct_discrepancy_target == "ema_gt" and self.ema_model is not None:
                    with torch.no_grad():
                        ema_gt_hidden_states, _, _, _, _ = self.ema_model(
                            full_sequences.to(device),
                            attention_mask.to(device),
                            pos_ids.to(device),
                            return_classifier_logits=True,
                            context_length=context_length,
                            prompt_length=prompt_length,
                            generate_max_len=generate_max_len,
                            stride=stride,
                            num_blocks=num_blocks,
                            hidden_state_method=self.args.hidden_state_method,
                            qa_masks=qa_masks.to(device),
                            qa_masking=self.args.qa_masking,
                        )
                    target_gt_hidden_states = ema_gt_hidden_states.detach()
                    
                    # INSTRUMENTATION: EMA 调用后的状态
                    import torch.distributed as dist
                    is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
                    if is_rank0 and hasattr(self, '_instrumentation_printed'):
                        print("\n" + "="*80)
                        print("[G3 INSTRUMENTATION] After EMA model forward")
                        print("="*80)
                        print(f"  target_gt_hidden_states.shape: {target_gt_hidden_states.shape}")
                        print(f"  target_gt_hidden_states.dtype: {target_gt_hidden_states.dtype}")
                        print(f"  target_gt_hidden_states device: {target_gt_hidden_states.device}")
                        print("="*80 + "\n")

                target_gt_embedding, online_gen_embedding = prepare_distribution_embeddings_from_split_hidden_states(
                    gt_hidden_states=target_gt_hidden_states,
                    gen_hidden_states=gen_hidden_states,
                    gt_qa_mask=target_gt_mask,
                    gen_qa_mask=gen_mask,
                    n_samples_per_prompt=self.args.n_samples_per_prompt,
                    embed_method=self.args.embed_method,
                    use_whitening=self.args.use_whitening,
                    feature_map_type=getattr(self.args, "feature_map_type", "identity"),
                    rff_num_features=getattr(self.args, "rff_num_features", 128),
                    rff_sigma=getattr(self.args, "rff_sigma", 1.0),
                    rff_seed=getattr(self.args, "rff_seed", 43),
                    qa_masking=self.args.qa_masking,
                )
                # INSTRUMENTATION: embedding 准备后的状态
                import torch.distributed as dist
                is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
                if is_rank0 and hasattr(self, '_instrumentation_printed'):
                    print("\n" + "="*80)
                    print("[G3 INSTRUMENTATION] After prepare_distribution_embeddings")
                    print("="*80)
                    print(f"  target_gt_embedding.shape: {target_gt_embedding.shape}")
                    print(f"  online_gen_embedding.shape: {online_gen_embedding.shape}")
                    print(f"  shapes match: {target_gt_embedding.shape == online_gen_embedding.shape}")
                    print("="*80 + "\n")
                
                critic_direct_discrepancy_loss = direct_discrepancy_coef * compute_cf_discrepancy_loss(
                    online_gen_embedding,
                    target_gt_embedding.detach(),
                    cf_num_freqs=getattr(self.args, "cf_num_freqs", 128),
                    cf_sigma=getattr(self.args, "cf_sigma", 1.0),
                    cf_seed=getattr(self.args, "cf_seed", 43),
                    cf_alpha=getattr(self.args, "cf_alpha", 0.5),
                    cf_beta=getattr(self.args, "cf_beta", 0.5),
                    cf_target_mode="single",
                    cf_target_num_refs=1,
                    cf_target_std=getattr(self.args, "cf_target_std", 0.05),
                    cf_target_seed=getattr(self.args, "cf_target_seed", 43),
                )

            critic_loss = critic_classifier_loss + critic_direct_discrepancy_loss

            if self.args.use_dynamic_batch:
                critic_loss = critic_loss * self.replay_buffer.dynamic_loss_scale[step]

            if train_critic:
                self.strategy.backward(critic_loss, self.critic, self.critic_optim)

            if train_critic and self.ema_model:
                if self.args.use_dynamic_batch:
                    if self.replay_buffer.dynamic_optimizer_step[step]:
                        self.strategy.moving_average(self.critic, self.ema_model, self.ema_beta, "cuda")
                else:
                    self.strategy.moving_average(self.critic, self.ema_model, self.ema_beta, "cuda")

            if getattr(self.args, "log_gradients", False):
                should_log = True
                if self.args.use_dynamic_batch:
                    should_log = bool(self.replay_buffer.dynamic_optimizer_step[step])

                if should_log:
                    if train_critic:
                        grad_norm_value = self._compute_critic_grad_norm()
                    else:
                        grad_norm_value = 0
        
            if train_critic:
                if self.args.use_dynamic_batch:
                    if self.replay_buffer.dynamic_optimizer_step[step]:
                        self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")
                else:
                    self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")
        finally:
            if restore_attention_impl is not None:
                self.critic.set_attention_implementation(restore_attention_impl)

        # Get post-step metrics (using EMA model if available, otherwise current critic)
        with torch.no_grad():
            _post_model = self.ema_model if self.ema_model else self.critic
            restore_post_attention_impl = _maybe_override_attention_impl(_post_model)
            try:
                _, _, gt_classifier_logits, gen_classifier_logits, _ = _post_model(
                    full_sequences.to(device),
                    attention_mask.to(device),
                    pos_ids.to(device),
                    return_classifier_logits=True,
                    context_length=context_length,
                    prompt_length=prompt_length,
                    generate_max_len=generate_max_len,
                    stride=stride,
                    num_blocks=num_blocks,
                    hidden_state_method=self.args.hidden_state_method,
                    qa_masks=qa_masks.to(device),
                    qa_masking=self.args.qa_masking,
                )
            finally:
                if restore_post_attention_impl is not None:
                    _post_model.set_attention_implementation(restore_post_attention_impl)

            if self.args.critic_sequence_level == "token":
                gt_classifier_logits = gt_classifier_logits.reshape(batch_size // self.args.n_samples_per_prompt, self.args.n_samples_per_prompt, num_blocks, generate_max_len, gt_classifier_logits.shape[-1])
                gen_classifier_logits = gen_classifier_logits.reshape(batch_size // self.args.n_samples_per_prompt, self.args.n_samples_per_prompt, num_blocks, generate_max_len, gen_classifier_logits.shape[-1])
            elif self.args.critic_sequence_level in ["concat", "mean_pooling", "last_token"]:
                gt_classifier_logits = gt_classifier_logits.reshape(batch_size // self.args.n_samples_per_prompt, self.args.n_samples_per_prompt, num_blocks, gt_classifier_logits.shape[-1])
                gen_classifier_logits = gen_classifier_logits.reshape(batch_size // self.args.n_samples_per_prompt, self.args.n_samples_per_prompt, num_blocks, gen_classifier_logits.shape[-1])
            post_metrics = self.critic_classifier_accuracy_fn(gt_classifier_logits.detach(), gen_classifier_logits.detach())
            post_classifier_accuracy = post_metrics[0]
            post_classifier_precision = post_metrics[1]
            post_classifier_recall = post_metrics[2]
            post_classifier_f1_score = post_metrics[3]
            post_classifier_pr_auc = post_metrics[4]
            post_classifier_roc_auc = post_metrics[5]
            post_classifier_mean_prob_gap = post_metrics[6]
            post_classifier_prob_gen = post_metrics[7]
            post_classifier_pred_pos_frac = post_metrics[8]
            post_classifier_acc_thresh0p5 = post_metrics[9]
            critic_logit_exact_match = torch.mean((gt_classifier_logits == gen_classifier_logits).float())
        last_lrs = self.critic_scheduler.get_last_lr()
        if len(last_lrs) == 0:
            critic_lr_backbone = 0.0
            critic_lr_head = 0.0
        elif len(last_lrs) == 1:
            # Adapter-only / frozen-backbone runs can collapse to a single scheduler LR entry.
            critic_lr_backbone = 0.0
            critic_lr_head = last_lrs[0]
        else:
            critic_lr_backbone = last_lrs[0]
            critic_lr_head = last_lrs[1]

        optimizer_lrs = [float(group.get("lr", 0.0)) for group in getattr(self.critic_optim, "param_groups", [])]
        if len(optimizer_lrs) == 0:
            critic_opt_lr_group0 = 0.0
            critic_opt_lr_group1 = 0.0
        elif len(optimizer_lrs) == 1:
            critic_opt_lr_group0 = 0.0
            critic_opt_lr_group1 = optimizer_lrs[0]
        else:
            critic_opt_lr_group0 = optimizer_lrs[0]
            critic_opt_lr_group1 = optimizer_lrs[1]

        feature_adapter_drift_abs_mean, feature_adapter_drift_rel_l2 = self._compute_param_drift(
            self.critic, self._feature_adapter_init, "feature_adapter"
        )
        feature_adapter_up_drift_abs_mean, feature_adapter_up_drift_rel_l2 = self._compute_param_drift(
            self.critic, self._feature_adapter_init, "up_proj"
        )
        feature_adapter_down_drift_abs_mean, feature_adapter_down_drift_rel_l2 = self._compute_param_drift(
            self.critic, self._feature_adapter_init, "down_proj"
        )
        feature_adapter_ema_gap_abs_mean, feature_adapter_ema_gap_rel_l2 = self._compute_ema_gap(
            self.critic, self.ema_model, "feature_adapter"
        )

        status = {
            "train_critic": train_critic,
            "critic_loss": critic_loss.detach().item(),
            "critic_classifier_loss": critic_classifier_loss.detach().item(),
            "critic_direct_discrepancy_loss": critic_direct_discrepancy_loss.detach().item(),
            "critic_classifier_accuracy_before": prev_classifier_accuracy.detach().item(),
            "critic_classifier_accuracy_after": post_classifier_accuracy.detach().item(),
            "critic_classifier_precision_before": prev_classifier_precision.detach().item(),
            "critic_classifier_precision_after": post_classifier_precision.detach().item(),
            "critic_classifier_recall_before": prev_classifier_recall.detach().item(),
            "critic_classifier_recall_after": post_classifier_recall.detach().item(),
            "critic_classifier_f1_score_before": prev_classifier_f1_score.detach().item(),
            "critic_classifier_f1_score_after": post_classifier_f1_score.detach().item(),
            "critic_classifier_pr_auc_before": prev_classifier_pr_auc.detach().item(),
            "critic_classifier_pr_auc_after": post_classifier_pr_auc.detach().item(),
            "critic_classifier_roc_auc_before": prev_classifier_roc_auc.detach().item(),
            "critic_classifier_roc_auc_after": post_classifier_roc_auc.detach().item(),
            "critic_classifier_mean_prob_gap_before": prev_classifier_mean_prob_gap.detach().item(),
            "critic_classifier_mean_prob_gap_after": post_classifier_mean_prob_gap.detach().item(),
            "critic_classifier_prob_gen_before": prev_classifier_prob_gen.detach().item(),
            "critic_classifier_prob_gen_after": post_classifier_prob_gen.detach().item(),
            "critic_classifier_pred_pos_frac_before": prev_classifier_pred_pos_frac.detach().item(),
            "critic_classifier_pred_pos_frac_after": post_classifier_pred_pos_frac.detach().item(),
            "critic_classifier_acc_thresh0p5_before": prev_classifier_acc_thresh0p5.detach().item(),
            "critic_classifier_acc_thresh0p5_after": post_classifier_acc_thresh0p5.detach().item(),
            "critic_logit_exact_match": critic_logit_exact_match.detach().item(),
            "critic_lr_backbone": critic_lr_backbone,
            "critic_lr_head": critic_lr_head,
            "critic_opt_lr_group0": critic_opt_lr_group0,
            "critic_opt_lr_group1": critic_opt_lr_group1,
            "feature_adapter_drift_abs_mean": feature_adapter_drift_abs_mean,
            "feature_adapter_drift_rel_l2": feature_adapter_drift_rel_l2,
            "feature_adapter_up_drift_abs_mean": feature_adapter_up_drift_abs_mean,
            "feature_adapter_up_drift_rel_l2": feature_adapter_up_drift_rel_l2,
            "feature_adapter_down_drift_abs_mean": feature_adapter_down_drift_abs_mean,
            "feature_adapter_down_drift_rel_l2": feature_adapter_down_drift_rel_l2,
            "feature_adapter_ema_gap_abs_mean": feature_adapter_ema_gap_abs_mean,
            "feature_adapter_ema_gap_rel_l2": feature_adapter_ema_gap_rel_l2,
        }
        if grad_norm_value is not None:
            status["critic_grad_norm"] = grad_norm_value
        else:
            status["critic_grad_norm"] = 0.0
        return status

    @staticmethod
    def _normalize_param_name(name: str) -> str:
        while name.startswith("module."):
            name = name[len("module.") :]
        return name

    def _snapshot_named_params(self, model, needle: str) -> Dict[str, torch.Tensor]:
        snapshot: Dict[str, torch.Tensor] = {}
        if model is None:
            return snapshot
        with torch.no_grad():
            for name, param in model.named_parameters():
                norm_name = self._normalize_param_name(name)
                if needle in norm_name:
                    snapshot[norm_name] = param.detach().float().cpu().clone()
        return snapshot

    def _compute_param_drift(self, model, snapshot: Dict[str, torch.Tensor], needle: str) -> tuple[float, float]:
        if model is None or not snapshot:
            return 0.0, 0.0

        delta_sq = 0.0
        base_sq = 0.0
        abs_mean_acc = 0.0
        count = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                norm_name = self._normalize_param_name(name)
                if needle not in norm_name or norm_name not in snapshot:
                    continue
                current = param.detach().float().cpu()
                base = snapshot[norm_name]
                diff = current - base
                delta_sq += float(diff.pow(2).sum().item())
                base_sq += float(base.pow(2).sum().item())
                abs_mean_acc += float(diff.abs().mean().item())
                count += 1
        if count == 0:
            return 0.0, 0.0
        abs_mean = abs_mean_acc / count
        rel_l2 = math.sqrt(delta_sq) / (math.sqrt(base_sq) + 1e-12)
        return abs_mean, rel_l2

    def _compute_ema_gap(self, online_model, ema_model, needle: str) -> tuple[float, float]:
        if online_model is None or ema_model is None:
            return 0.0, 0.0
        online_params = self._snapshot_named_params(online_model, needle)
        ema_params = self._snapshot_named_params(ema_model, needle)
        if not online_params or not ema_params:
            return 0.0, 0.0

        delta_sq = 0.0
        online_sq = 0.0
        abs_mean_acc = 0.0
        count = 0
        for name, online in online_params.items():
            if name not in ema_params:
                continue
            ema = ema_params[name]
            diff = online - ema
            delta_sq += float(diff.pow(2).sum().item())
            online_sq += float(online.pow(2).sum().item())
            abs_mean_acc += float(diff.abs().mean().item())
            count += 1
        if count == 0:
            return 0.0, 0.0
        abs_mean = abs_mean_acc / count
        rel_l2 = math.sqrt(delta_sq) / (math.sqrt(online_sq) + 1e-12)
        return abs_mean, rel_l2

    def _compute_critic_grad_norm(self) -> Optional[float]:
        """Return the global gradient norm of the critic, handling ZeRO stages gracefully."""
        engine = self.critic
        grad_norm: Optional[float] = None
        try:
            if engine is not None:
                if hasattr(engine, "get_global_grad_norm"):
                    grad_norm = engine.get_global_grad_norm()
                elif hasattr(engine, "get_global_norm"):
                    grad_norm = engine.get_global_norm()
            if grad_norm is None:
                total_norm_sq = 0.0
                has_grad = False
                stage = getattr(self.strategy, "stage", None)
                gather_enabled = stage == 3
                for param in self.critic.parameters():
                    ctx = deepspeed.zero.GatheredParameters([param], enabled=gather_enabled) if gather_enabled else nullcontext()
                    with ctx:
                        grad = param.grad
                        if grad is None:
                            continue
                        has_grad = True
                        param_grad = grad.detach()
                        if not torch.is_floating_point(param_grad):
                            param_grad = param_grad.float()
                        total_norm_sq += param_grad.norm(2).item() ** 2
                if has_grad:
                    grad_norm = math.sqrt(total_norm_sq)
        except Exception as exc:
            logger.warning(f"Failed to compute critic gradient norm: {exc}")
            grad_norm = None
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.detach().cpu().item()
        return grad_norm

@ray.remote(num_gpus=1)
class EBFTCriticModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, max_steps):
        args = strategy.args

        # Drop previous references to free CUDA memory before reinit.
        if hasattr(self, "critic"):
            self.critic = None
        if hasattr(self, "ema_model"):
            self.ema_model = None
        if hasattr(self, "critic_optim"):
            self.critic_optim = None
        if hasattr(self, "critic_scheduler"):
            self.critic_scheduler = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
 
        self._setup_distributed(strategy)
        critic = Critic(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            lora_dropout=strategy.args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            packing_samples=strategy.args.packing_samples,
            temperature=strategy.args.temperature,
            use_liger_kernel=strategy.args.use_liger_kernel,
            critic_classifier_loss_choice=getattr(strategy.args, "critic_classifier_loss_choice", "log"),
            critic_sequence_level=getattr(strategy.args, "critic_sequence_level", "token"),
            gen_len=getattr(strategy.args, "generate_max_len", None),
            hidden_state_method=getattr(strategy.args, "hidden_state_method", "last_only"),
            feature_adapter_enable=getattr(strategy.args, "feature_adapter_enable", False),
            feature_adapter_type=getattr(strategy.args, "feature_adapter_type", "residual_bottleneck"),
            feature_adapter_rank=getattr(strategy.args, "feature_adapter_rank", 64),
            feature_adapter_dropout=getattr(strategy.args, "feature_adapter_dropout", 0.0),
        )
        # configure tokenizer
        if strategy.args.save_value_network:
            self.tokenizer = get_tokenizer(
                pretrain, critic, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
            )

        if args.enable_ema:
            ema_model = Critic(
                pretrain,
                use_flash_attention_2=strategy.args.flash_attn,
                bf16=strategy.args.bf16,
                load_in_4bit=strategy.args.load_in_4bit,
                lora_rank=strategy.args.lora_rank,
                lora_alpha=strategy.args.lora_alpha,
                target_modules=strategy.args.target_modules,
                lora_dropout=strategy.args.lora_dropout,
                ds_config=strategy.get_ds_train_config(is_actor=True),
                packing_samples=strategy.args.packing_samples,
                temperature=strategy.args.temperature,
                use_liger_kernel=strategy.args.use_liger_kernel,
                critic_classifier_loss_choice=getattr(strategy.args, "critic_classifier_loss_choice", "log"),
                critic_sequence_level=getattr(strategy.args, "critic_sequence_level", "token"),
                gen_len=getattr(strategy.args, "generate_max_len", None),
                hidden_state_method=getattr(strategy.args, "hidden_state_method", "last_only"),
                feature_adapter_enable=getattr(strategy.args, "feature_adapter_enable", False),
                feature_adapter_type=getattr(strategy.args, "feature_adapter_type", "residual_bottleneck"),
                feature_adapter_rank=getattr(strategy.args, "feature_adapter_rank", 64),
                feature_adapter_dropout=getattr(strategy.args, "feature_adapter_dropout", 0.0),
            )
        else:
            ema_model = None


        # configure optimizer and scheduler
        # Resolve critic_lr_head: None means "same as backbone"
        critic_lr_head = args.critic_lr_head if args.critic_lr_head is not None else args.critic_learning_rate

        # Handle critic_learning_rate = 0 case to avoid division by zero in scheduler
        if args.critic_learning_rate == 0 and critic_lr_head == 0:
            # Create optimizer with lr=0 (no critic training)
            critic_optim = strategy.create_optimizer_critic(
                critic, lr_backbone=0.0, lr_head=0.0, betas=args.adam_betas, weight_decay=args.l2
            )

            # Create dummy scheduler that maintains lr=0
            class DummyScheduler:
                """Dummy scheduler that keeps learning rate at 0."""
                def __init__(self, optimizer):
                    self.optimizer = optimizer

                def step(self):
                    # Ensure lr stays at 0
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = 0.0

                def get_last_lr(self):
                    return [0.0, 0.0]  # Two learning rates: backbone and head

                def state_dict(self):
                    return {"lr": [group["lr"] for group in self.optimizer.param_groups]}

                def load_state_dict(self, state_dict):
                    lr_list = state_dict.get("lr")
                    if lr_list is not None:
                        for group, lr in zip(self.optimizer.param_groups, lr_list):
                            group["lr"] = lr

            critic_scheduler = DummyScheduler(critic_optim)

        else:
            # Non-zero LR: standard scheduler applied to both backbone and head param groups.
            critic_optim = strategy.create_optimizer_critic(
                critic,
                lr_backbone=args.critic_learning_rate,
                lr_head=critic_lr_head,
                betas=args.adam_betas,
                weight_decay=args.l2,
            )
            warmup_ratio = args.critic_lr_warmup_ratio if args.critic_lr_warmup_ratio is not None else args.lr_warmup_ratio
            critic_scheduler = get_scheduler(
                args.critic_lr_scheduler,
                critic_optim,
                num_warmup_steps=math.ceil(max_steps * warmup_ratio),
                num_training_steps=max_steps,
                scheduler_specific_kwargs={"min_lr_rate": 0.10},
            )

        if args.gradient_checkpointing:
            critic.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # prepare models/optimizers...
        self.critic, self.critic_optim, self.critic_scheduler = strategy.prepare(
            (critic, critic_optim, critic_scheduler),
            is_rlhf=True,
        )

        # Clean up allocator after (re)initialization to mitigate fragmentation.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        if ema_model:
            ema_model._offload = True
            self.ema_model = strategy.prepare(ema_model, is_rlhf=True)
        else:
            self.ema_model = None

        # load checkpoint
        if args.load_critic_checkpoint and os.path.exists(os.path.join(args.ckpt_path, "_critic")):
            ckpt_path = os.path.join(args.ckpt_path, "_critic")
            strategy.print(f"Loading the checkpoint: {ckpt_path}")
            strategy.load_ckpt(self.critic, ckpt_path)

        # initial offload
        if strategy.args.deepspeed_enable_sleep:
            self.offload_states()

        # configure Trainer
        self.trainer = CriticEBFTTrainer(
            strategy,
            critic=self.critic,
            ema_model=self.ema_model,
            critic_optim=self.critic_optim,
            ema_beta=args.ema_beta,
            critic_scheduler=self.critic_scheduler,
            micro_train_batch_size=args.micro_train_batch_size,
            value_clip=args.value_clip,
        )

    def forward(
        self,
        sequences: torch.LongTensor,
        prompt_length: int,
        context_length: int,
        generate_max_len: int,
        stride: int,
        num_blocks: int,
        hidden_state_method: str,
        doc_ids: torch.Tensor = None,
        document_masking: bool = False,
        qa_masks: torch.Tensor = None,
        qa_masking: bool = False,
    ) -> torch.Tensor:
        """Generates critic values."""
        device = torch.cuda.current_device()
        self.critic.eval()
        with torch.no_grad():
            attention_mask, pos_ids = build_strided_attention_mask_and_positions(
                full_sequence_length=sequences.size(1),
                prompt_length=prompt_length,
                context_length=context_length,
                generation_step=generate_max_len,
                max_generation_length=generate_max_len,
                stride=stride,
                num_blocks=num_blocks,
                device=device,
                doc_ids=doc_ids[:, :prompt_length] if doc_ids is not None else None,
                document_masking=document_masking,
            )
            _model = self.ema_model if self.ema_model else self.critic
            restore_attention_impl = None
            if (
                attention_mask is not None
                and attention_mask.dim() == 4
                and getattr(self.strategy.args, "gradient_checkpointing", False)
            ):
                current_impl = _model.get_attention_implementation()
                if current_impl == "flash_attention_2":
                    restore_attention_impl = current_impl
                    _model.set_attention_implementation("eager")
            try:
                hidden_states, reward = _model(
                    sequences.to(device),
                    attention_mask=attention_mask.to(device),
                    pos_ids=pos_ids.to(device),
                    return_classifier_logits=False,
                    context_length=context_length,
                    prompt_length=prompt_length,
                    generate_max_len=generate_max_len,
                    stride=stride,
                    num_blocks=num_blocks,
                    hidden_state_method=hidden_state_method,
                    qa_masks=qa_masks.to(device),
                    qa_masking=qa_masking,
                )
            finally:
                if restore_attention_impl is not None:
                    _model.set_attention_implementation(restore_attention_impl)
        self.critic.train()  # reset model state
        return hidden_states.to("cpu"), reward.to("cpu")

    def append(self, experience):
        """Append experience to replay buffer."""
        self.trainer.replay_buffer.append(experience)
    
    def flush_buffer(self):
        """Clear the replay buffer."""
        self.trainer.replay_buffer.clear()
        torch.cuda.empty_cache()

    def fit(self):
        """Train critic model with the replay buffer."""
        torch.cuda.empty_cache()
        self.critic.train()
        status = self.trainer.ppo_train()
        self.trainer.replay_buffer.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return status

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.ema_model if args.enable_ema else self.critic,
            self.tokenizer,
            args.save_path + "_critic",
        )

    def save_checkpoint(self, tag):
        args = self.strategy.args
        # If the critic isn't being trained (LRs are 0), DeepSpeed may use a DummyOptim which
        # doesn't implement state_dict() and will crash during checkpointing. In that case,
        # skip critic checkpointing altogether (actor checkpoints can still be saved).
        if getattr(args, "disable_ds_ckpt", False):
            return
        if float(getattr(args, "critic_learning_rate", 0.0) or 0.0) == 0.0 and float(
            getattr(args, "critic_lr_head", 0.0) or 0.0
        ) == 0.0:
            return

        # Always checkpoint the training engine; the EMA engine uses a DummyOptim without state_dict.
        self.strategy.save_ckpt(
            self.critic, os.path.join(args.ckpt_path, "_critic"), tag, args.max_ckpt_num, args.max_ckpt_mem
        )
        if self.strategy.is_rank_0():
            write_run_config(os.path.join(args.ckpt_path, "_critic", tag), args, tag=tag)

    def reload_states(self):
        reload_deepspeed_states(self.critic)

    def offload_states(self):
        offload_deepspeed_states(self.critic)

    def log_memory(self, prefix: str = ""):
        """Log basic CUDA memory stats for debugging."""
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        msg = f"[critic-mem]{f' {prefix}' if prefix else ''} allocated={allocated:.1f}MB reserved={reserved:.1f}MB peak_allocated={peak:.1f}MB"
        # This can be called frequently; keep it quiet unless debugging.
        if getattr(getattr(self, "args", None), "debug", False):
            logger.info(msg)

    def sync_critic_scheduler(self, global_step: int):
        """
        Align critic scheduler counters and LRs with the provided global_step.
        This is used after the critic is reinitialized from the actor checkpoint.
        """
        scheduler = getattr(self, "critic_scheduler", None)
        if scheduler is None:
            return

        def count_training_steps(step_limit: int) -> int:
            if hasattr(scheduler, "is_training_fn"):
                return sum(1 for s in range(1, step_limit + 1) if any(scheduler.is_training_fn(s)))
            return step_limit

        train_steps = count_training_steps(global_step)

        # Update counters on custom schedulers if present.
        if hasattr(scheduler, "global_step"):
            scheduler.global_step = global_step
        if hasattr(scheduler, "train_step"):
            scheduler.train_step = train_steps
        if hasattr(scheduler, "cycle_length") and hasattr(scheduler, "cycle_train_step"):
            # Count training steps within the current cycle for restarting scheduler.
            scheduler.cycle_train_step = 0
