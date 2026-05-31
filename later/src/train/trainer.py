from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import math
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from tqdm.auto import tqdm

from later.src.train.log import (
    build_batch_diagnostics,
    build_forward_memory_breakdown,
    build_train_scalar_payload,
    count_generated_token_types,
    log_eval_generated_text,
    log_eval_sample_text_bundle,
    log_scalars,
    token_repr,
)
from later.src.train.losses import (
    compute_answer_ce,
    compute_early_exit_rank_loss,
    compute_teacher_kl,
    compute_weighted_ce,
)
from later.src.train.utils import (
    BaseTokenRowFreezeController,
    append_jsonl,
    build_cuda_memory_snapshot,
    curriculum_weight,
    get_early_exit_forbidden_token_ids,
    mean_or_zero,
    reset_cuda_peak_memory,
    safe_div,
    sync_batch_across_ranks,
    token_ids_to_list,
)


class FixedOrderSampler(Sampler[int]):
    def __init__(self, indices: List[int]) -> None:
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


@dataclass
class TrainLoopResult:
    status: str
    checkpoint_dir: str | None = None
    global_step: int = 0
    epoch_index: int = 0
    next_step_in_epoch: int = 0
    target_stage: int | None = None


class LatentSFTTrainer:
    def __init__(
        self,
        accelerator: Any,
        model: Any,
        tokenizer: Any,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        train_dataset: Any,
        val_dataset: Any,
        collator: Any,
        teacher_cache: Any,
        config: Dict[str, Any],
        logger: Any,
        writer: Any | None = None,
        base_token_row_freeze_controller: BaseTokenRowFreezeController | None = None,
        initial_training_state: Dict[str, Any] | None = None,
    ) -> None:
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.collator = collator
        self.teacher_cache = teacher_cache
        self.config = config
        self.logger = logger
        self.writer = writer
        self.base_token_row_freeze_controller = base_token_row_freeze_controller
        self.initial_training_state = dict(initial_training_state or {})
        if self.base_token_row_freeze_controller is not None:
            self.base_token_row_freeze_controller.bind_runtime(model=self.model, accelerator=self.accelerator)
            self.base_token_row_freeze_controller.sync_weight_decay_exemptions(self.optimizer)
        self.tensorboard_log_every_steps = max(int(self.config.get("tensorboard_log_every_steps", 1)), 1)
        self.distributed_backend = str(self.config.get("distributed_backend", "fsdp")).lower()
        self.enable_memory_profile = bool(self.config.get("enable_memory_profile", False))
        self.enable_forward_memory_breakdown_log = bool(self.config.get("enable_forward_memory_breakdown_log", False))
        self.enable_rank_pad_sync = bool(self.config.get("enable_rank_pad_sync", True))
        self.kl_topk_dim = max(int(self.config.get("kl_topk_dim", 128) or 0), 0)
        self.enable_alignment_debug_log = bool(self.config.get("enable_loss_alignment_debug", False))
        self.alignment_debug_log_every_steps = max(
            int(self.config.get("loss_alignment_debug_every_steps", 1)),
            1,
        )
        self.alignment_debug_max_samples = max(int(self.config.get("alignment_debug_max_samples", 1)), 1)
        self.memory_profile_log_to_console = bool(self.config.get("memory_profile_log_to_console", False))
        self.rank_boundary_probe_once = bool(self.config.get("rank_boundary_probe_once", True))
        self.rank_boundary_probe_until_step = max(int(self.config.get("rank_boundary_probe_until_step", 1)), 0)
        self.raise_on_rank_boundary_mismatch = bool(self.config.get("raise_on_rank_boundary_mismatch", False))
        self.halt_dense_token_ids = get_early_exit_forbidden_token_ids(self.train_dataset.token_constants)
        self.enable_zero3_trace_probe_log = bool(self.config.get("enable_zero3_trace_probe_log", False))
        self.zero3_trace_probe_every_steps = max(int(self.config.get("zero3_trace_probe_every_steps", 1)), 1)
        self.zero3_trace_probe_max_modules = max(int(self.config.get("zero3_trace_probe_max_modules", 64)), 1)
        self.zero3_trace_probe_record_limit = max(int(self.config.get("zero3_trace_probe_record_limit", 128)), 1)
        self.zero3_trace_probe_log_per_call = bool(self.config.get("zero3_trace_probe_log_per_call", False))
        self.zero3_trace_probe_call_log_limit = max(int(self.config.get("zero3_trace_probe_call_log_limit", 256)), 1)
        self.enable_deepspeed_submodule_trace_log = bool(self.config.get("enable_deepspeed_submodule_trace_log", False))
        self.deepspeed_submodule_trace_every_steps = max(
            int(self.config.get("deepspeed_submodule_trace_every_steps", 1)),
            1,
        )
        self.deepspeed_submodule_trace_record_limit = max(
            int(self.config.get("deepspeed_submodule_trace_record_limit", 256)),
            1,
        )
        self.deepspeed_submodule_trace_log_per_record = bool(
            self.config.get("deepspeed_submodule_trace_log_per_record", False)
        )
        self.deepspeed_submodule_trace_log_record_limit = max(
            int(self.config.get("deepspeed_submodule_trace_log_record_limit", 256)),
            1,
        )
        self.debug_first_batch_use_max_length_sample = bool(
            self.config.get("debug_first_batch_use_max_length_sample", False)
        )
        self.save_training_state = bool(self.config.get("save_training_state", True))
        self.save_on_exception = bool(self.config.get("save_on_exception", False))
        self.resume_training = bool(self.config.get("resume_training", False))
        self.resume_from_checkpoint = str(self.config.get("resume_from_checkpoint", "latest"))
        self.stage1_trainable_mode = "projector_embed_lmhead"
        self.stage2_trainable_mode = "full"
        initial_mode = self.initial_training_state.get("trainable_mode")
        self._current_trainable_mode: Optional[str] = str(initial_mode) if initial_mode else None
        self.teacher_length_mismatch_count = 0
        self.memory_log_path = (
            Path(self.config["output_dir"]) / f"memory_profile_rank{self.accelerator.process_index:02d}.jsonl"
        )
        self._memory_profile_done = False
        self._debug_first_batch_indices: Optional[List[int]] = None
        self._rank_boundary_probe_done = False
        self._zero3_trace_probe_installed = False
        self._zero3_trace_probe_handles: List[Any] = []
        self._zero3_trace_probe_active = False
        self._zero3_trace_probe_counts: Dict[str, int] = {}
        self._zero3_trace_probe_prefix: List[str] = []
        self._zero3_trace_probe_suffix: deque[str] = deque(maxlen=self.zero3_trace_probe_record_limit)
        self._zero3_trace_probe_call_index = 0
        self._zero3_trace_probe_step = -1
        self._zero3_trace_probe_epoch = -1
        self._zero3_trace_probe_batch_meta: Dict[str, Any] = {}
        self._deepspeed_trace_patch_installed = False
        self._deepspeed_trace_probe_active = False
        self._deepspeed_trace_probe_step = -1
        self._deepspeed_trace_probe_epoch = -1
        self._deepspeed_trace_probe_batch_meta: Dict[str, Any] = {}
        self._deepspeed_trace_probe_recorded: List[Dict[str, Any]] = []
        self._halt_dense_ce_loss_ema: float | None = None
        self._total_train_steps = 0
        self._best_answer_ce = float("inf")
        self._best_answer_ce_step = -1
        self._best_answer_ce_meta_path = Path(self.config["output_dir"]) / "best_answer_ce.json"
        if self._best_answer_ce_meta_path.exists():
            try:
                payload = json.loads(self._best_answer_ce_meta_path.read_text(encoding="utf-8"))
                self._best_answer_ce = float(payload.get("val_answer_ce", float("inf")))
                self._best_answer_ce_step = int(payload.get("step", -1))
            except Exception:
                self._best_answer_ce = float("inf")
                self._best_answer_ce_step = -1

    @staticmethod
    def _count_parameters(parameters: List[torch.nn.Parameter], trainable_only: bool = False) -> int:
        total = 0
        for param in parameters:
            if trainable_only and (not bool(param.requires_grad)):
                continue
            numel = int(param.numel())
            if numel <= 0:
                ds_numel = getattr(param, "ds_numel", None)
                if ds_numel is not None:
                    try:
                        numel = int(ds_numel)
                    except Exception:
                        numel = int(param.numel())
            total += int(numel)
        return int(total)

    @staticmethod
    def _count_optimizer_parameters(optimizer: torch.optim.Optimizer) -> int:
        total = 0
        for group in optimizer.param_groups:
            total += LatentSFTTrainer._count_parameters(list(group.get("params", [])), trainable_only=False)
        return int(total)

    @staticmethod
    def _count_optimizer_tensors(optimizer: torch.optim.Optimizer) -> int:
        total = 0
        for group in optimizer.param_groups:
            total += len(list(group.get("params", [])))
        return int(total)

    @staticmethod
    def _collect_trainable_parameter_names(model: Any) -> List[str]:
        names: List[str] = []
        for name, param in model.named_parameters():
            if bool(param.requires_grad):
                names.append(str(name))
        return names

    def _close_writer(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def _log_train_scalars(self, metrics: Dict[str, torch.Tensor], global_step: int, epoch_index: int) -> None:
        if global_step % self.tensorboard_log_every_steps != 0:
            return
        payload = build_train_scalar_payload(
            metrics=metrics,
            global_step=global_step,
            epoch_index=epoch_index,
            scheduler=self.scheduler,
        )
        log_scalars(self.writer, payload, global_step)

    def _log_eval_scalars(self, metrics: Dict[str, float], global_step: int) -> None:
        log_scalars(self.writer, metrics, global_step)

    def _log_eval_generated_text(
        self,
        global_step: int,
        batch_index: int,
        sample_index: int,
        record_id: str,
        generated_latent_tokens: int,
        generated_cot_tokens: int,
        original_latent_tokens: int,
        original_cot_tokens: int,
        generated_text: str,
    ) -> None:
        log_eval_generated_text(
            writer=self.writer,
            global_step=global_step,
            batch_index=batch_index,
            sample_index=sample_index,
            record_id=record_id,
            generated_latent_tokens=generated_latent_tokens,
            generated_cot_tokens=generated_cot_tokens,
            original_latent_tokens=original_latent_tokens,
            original_cot_tokens=original_cot_tokens,
            generated_text=generated_text,
        )

    def _log_eval_sample_text_bundle(
        self,
        global_step: int,
        batch_index: int,
        sample_index: int,
        record_id: str,
        generated_latent_tokens: int,
        generated_cot_tokens: int,
        original_latent_tokens: int,
        original_cot_tokens: int,
        prompt_token_ids: List[int],
        prompt_text: str,
        ground_truth: str,
        generated_token_ids: List[int],
        generated_text: str,
    ) -> None:
        log_eval_sample_text_bundle(
            writer=self.writer,
            global_step=global_step,
            batch_index=batch_index,
            sample_index=sample_index,
            record_id=record_id,
            generated_latent_tokens=generated_latent_tokens,
            generated_cot_tokens=generated_cot_tokens,
            original_latent_tokens=original_latent_tokens,
            original_cot_tokens=original_cot_tokens,
            prompt_token_ids=prompt_token_ids,
            prompt_text=prompt_text,
            ground_truth=ground_truth,
            generated_token_ids=generated_token_ids,
            generated_text=generated_text,
        )

    @staticmethod
    def _token_ids_to_list(token_ids: Any) -> List[int]:
        return token_ids_to_list(token_ids)

    @staticmethod
    def _count_generated_token_types(token_ids: List[int], token_constants: Dict[str, int]) -> Dict[str, int]:
        return count_generated_token_types(token_ids=token_ids, token_constants=token_constants)

    @staticmethod
    def _extract_checkpoint_step(path: Path) -> int:
        match = re.match(r"^step_(\d+)$", path.name)
        if match is None:
            return -1
        return int(match.group(1))

    @staticmethod
    def _checkpoint_complete_marker_name() -> str:
        return "checkpoint_complete.json"

    def _write_checkpoint_complete_marker(
        self,
        output_dir: Path,
        global_step: int,
        epoch_index: int,
        next_step_in_epoch: int,
        reason: str,
        train_stage: int,
        trainable_mode: str,
    ) -> None:
        if not self.accelerator.is_main_process:
            return
        payload = {
            "global_step": int(global_step),
            "epoch_index": int(epoch_index),
            "next_step_in_epoch": int(next_step_in_epoch),
            "reason": str(reason),
            "distributed_backend": self.distributed_backend,
            "train_stage": int(train_stage),
            "trainable_mode": str(trainable_mode),
            "timestamp_unix": int(time.time()),
        }
        marker_path = output_dir / self._checkpoint_complete_marker_name()
        with marker_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)

    def _is_checkpoint_complete(self, checkpoint_dir: Path) -> bool:
        marker_path = checkpoint_dir / self._checkpoint_complete_marker_name()
        return bool(marker_path.exists())

    def _write_checkpoint_training_state(
        self,
        output_dir: Path,
        global_step: int,
        epoch_index: int,
        next_step_in_epoch: int,
        reason: str,
        train_stage: int,
        trainable_mode: str,
    ) -> None:
        if not self.accelerator.is_main_process:
            return
        payload = {
            "global_step": int(global_step),
            "epoch_index": int(epoch_index),
            "next_step_in_epoch": int(next_step_in_epoch),
            "reason": str(reason),
            "distributed_backend": self.distributed_backend,
            "train_stage": int(train_stage),
            "trainable_mode": str(trainable_mode),
        }
        with (output_dir / "training_state.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)

    def _restore_training_state(self) -> Dict[str, int]:
        default_state = {
            "global_step": 0,
            "epoch_index": 0,
            "next_step_in_epoch": 0,
            "train_stage": 1,
            "trainable_mode": self.stage1_trainable_mode,
        }
        state = dict(default_state)
        state.update(self.initial_training_state)
        if bool(state.get("enabled", False)):
            self.logger.info(
                {
                    "tag": "resume_counters_restored",
                    "enabled": True,
                    "checkpoint_dir": str(state.get("checkpoint_dir")),
                    "restored_global_step": int(state["global_step"]),
                    "restored_epoch_index": int(state["epoch_index"]),
                    "restored_next_step_in_epoch": int(state["next_step_in_epoch"]),
                    "train_stage": int(state.get("train_stage", 1)),
                    "trainable_mode": str(state.get("trainable_mode", self.stage1_trainable_mode)),
                    "model_weights_restored": True,
                    "optimizer_scheduler_fresh_init": True,
                }
            )
        return state

    def _try_save_exception_checkpoint(
        self,
        global_step: int,
        epoch_index: int,
        step_in_epoch: int,
        error_type: str,
        error_message: str,
    ) -> None:
        if not self.save_on_exception:
            return
        self.logger.warning(
            {
                "tag": "exception_checkpoint_skipped",
                "global_step": int(global_step),
                "epoch_index": int(epoch_index),
                "step_in_epoch": int(step_in_epoch),
                "error_type": error_type,
                "error_message": error_message,
                "reason": "save_on_exception_enabled_but_disabled_to_avoid_partial_or_inconsistent_checkpoints",
            }
        )

    def _sync_checkpoint_save_success(self, local_success: bool) -> bool:
        success_value = 1 if bool(local_success) else 0
        if dist.is_available() and dist.is_initialized():
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            success_tensor = torch.tensor([success_value], device=device, dtype=torch.long)
            dist.all_reduce(success_tensor, op=dist.ReduceOp.MIN)
            success_value = int(success_tensor.item())
        return bool(success_value)

    def _sampler_for_epoch(self, epoch_index: int) -> WeightedRandomSampler:
        total_epochs = max(int(self.config["num_epochs"]), 1)
        progress = 0.0 if total_epochs == 1 else float(epoch_index / (total_epochs - 1))
        weights = [
            curriculum_weight(
                sort_key=self.train_dataset.get_curriculum_sort_key(index),
                progress=progress,
                power=float(self.config["curriculum_power"]),
            )
            for index in range(len(self.train_dataset))
        ]
        generator = torch.Generator()
        generator.manual_seed(int(self.config.get("seed", 0)) + int(epoch_index))
        return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=False, generator=generator)

    def _sample_token_length(self, index: int) -> int:
        if hasattr(self.train_dataset, "samples") and not bool(getattr(self.train_dataset, "lazy", False)):
            samples = getattr(self.train_dataset, "samples")
            if 0 <= index < len(samples):
                return int(len(samples[index].get("token_ids", [])))
        sample = self.train_dataset[index]
        return int(len(sample.get("token_ids", [])))

    def _get_debug_first_batch_indices(self) -> List[int]:
        if self._debug_first_batch_indices is not None:
            return list(self._debug_first_batch_indices)

        batch_size = max(int(self.config.get("micro_batch_size", 1)), 1)
        target_max_len = int(self.config.get("max_length", 0))
        exact_hits: List[int] = []
        fallback_candidates: List[tuple[int, int]] = []

        for index in range(len(self.train_dataset)):
            try:
                token_length = self._sample_token_length(index)
            except Exception:
                continue
            if token_length >= target_max_len and target_max_len > 0:
                exact_hits.append(index)
                if len(exact_hits) >= batch_size:
                    break
            fallback_candidates.append((token_length, index))

        if len(exact_hits) < batch_size and fallback_candidates:
            fallback_candidates.sort(key=lambda item: item[0], reverse=True)
            used = set(exact_hits)
            for _, index in fallback_candidates:
                if index in used:
                    continue
                exact_hits.append(index)
                used.add(index)
                if len(exact_hits) >= batch_size:
                    break

        self._debug_first_batch_indices = exact_hits[:batch_size]
        if self._debug_first_batch_indices:
            debug_lengths = [self._sample_token_length(index) for index in self._debug_first_batch_indices]
            self.logger.info(
                {
                    "tag": "debug_first_batch_max_length",
                    "enabled": True,
                    "selected_indices": list(self._debug_first_batch_indices),
                    "selected_token_lengths": debug_lengths,
                    "target_max_length": target_max_len,
                }
            )
        else:
            self.logger.warning(
                {
                    "tag": "debug_first_batch_max_length",
                    "enabled": True,
                    "warning": "No valid samples found for debug first batch selection",
                }
            )
        return list(self._debug_first_batch_indices)

    def _apply_debug_first_batch(self, base_sampler: Sampler[int]) -> Sampler[int]:
        first_indices = self._get_debug_first_batch_indices()
        if not first_indices:
            return base_sampler
        base_indices = list(iter(base_sampler))
        ordered_indices = list(base_indices)
        for position, index in enumerate(first_indices):
            if position < len(ordered_indices):
                ordered_indices[position] = index
            else:
                ordered_indices.append(index)
        return FixedOrderSampler(ordered_indices)

    def _compute_train_stage(self, global_step: int, total_train_steps: int) -> tuple[int, float]:
        progress = float(global_step) / float(max(int(total_train_steps), 1))
        progress = min(max(progress, 0.0), 1.0)
        train_stage = 2 if progress >= float(self.config["stage2_start_fraction"]) else 1
        if int(self.config["train_stage"]) == 2:
            train_stage = 2
        self.logger.info(
            {
                "tag": "train_stage_progress",
                "progress": float(progress),
                "global_step": int(global_step),
                "total_train_steps": int(total_train_steps),
                "train_stage": int(train_stage),
            }
        )
        return train_stage, progress

    def _compute_halt_dense_loss_weight(self, train_stage: int, train_progress: float) -> float:
        stage1_weight = float(self.config.get("halt_dense_loss_stage1_weight", 1.0))
        stage2_final_weight = float(self.config.get("halt_dense_loss_stage2_final_weight", 0.05))
        if int(train_stage) < 2:
            return float(stage1_weight)
        stage2_start = float(self.config.get("stage2_start_fraction", 1.0))
        stage2_span = max(1.0 - stage2_start, 1.0e-8)
        stage2_progress = min(max((float(train_progress) - stage2_start) / stage2_span, 0.0), 1.0)
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * stage2_progress))
        return float(stage2_final_weight + (stage1_weight - stage2_final_weight) * cosine_scale)

    def _combine_base_and_halt_dense_loss(
        self,
        *,
        base_loss_without_halt: torch.Tensor,
        ce_loss: torch.Tensor,
        halt_dense_loss: torch.Tensor,
        halt_dense_loss_weight: float,
    ) -> Dict[str, torch.Tensor | str]:
        mode = str(self.config.get("halt_dense_loss_projection_mode", "none") or "none").strip().lower()
        if mode not in {"none", "ce_quality_gate"}:
            raise ValueError(f"Unsupported halt_dense_loss_projection_mode: {mode}")

        ce_loss = ce_loss.to(dtype=base_loss_without_halt.dtype)
        weighted_halt_dense_loss = ce_loss.new_tensor(float(halt_dense_loss_weight), dtype=ce_loss.dtype) * halt_dense_loss.to(
            dtype=ce_loss.dtype
        )
        zero = ce_loss.new_zeros((), dtype=ce_loss.dtype)
        zero_f32 = ce_loss.new_zeros((), dtype=torch.float32)
        ce_loss_scalar = ce_loss.detach().to(dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            ce_loss_scalar = ce_loss_scalar.clone()
            dist.all_reduce(ce_loss_scalar, op=dist.ReduceOp.SUM)
            ce_loss_scalar = ce_loss_scalar / float(max(dist.get_world_size(), 1))

        if mode == "none":
            return {
                "total_loss": base_loss_without_halt + weighted_halt_dense_loss,
                "projected_halt_dense_loss": weighted_halt_dense_loss.detach(),
                "gated_halt_dense_loss": weighted_halt_dense_loss.detach(),
                "projection_alpha": zero_f32,
                "projection_dot": zero_f32,
                "projection_ce_grad_norm_sq": zero_f32,
                "gate_alpha": zero_f32,
                "gate_ce_loss_ema": zero_f32,
                "projection_mode": mode,
            }

        ema_beta = min(max(float(self.config.get("halt_dense_ce_gate_ema_beta", 0.98)), 0.0), 0.999999)
        current_ce_value = float(ce_loss_scalar.item())
        previous_ema = getattr(self, "_halt_dense_ce_loss_ema", None)
        if previous_ema is None:
            next_ema = current_ce_value
        else:
            next_ema = float(ema_beta * previous_ema + (1.0 - ema_beta) * current_ce_value)
        if bool(getattr(self.model, "training", False)):
            self._halt_dense_ce_loss_ema = next_ema
        else:
            next_ema = float(previous_ema if previous_ema is not None else current_ce_value)

        ce_loss_ema = ce_loss.new_tensor(next_ema, dtype=torch.float32)
        gate_alpha = torch.clamp(
            ce_loss_ema / torch.clamp(ce_loss_scalar, min=1.0e-12),
            min=0.0,
            max=1.0,
        )
        gated_halt_dense_loss = weighted_halt_dense_loss * gate_alpha.detach().to(dtype=weighted_halt_dense_loss.dtype)
        return {
            "total_loss": base_loss_without_halt + gated_halt_dense_loss,
            "projected_halt_dense_loss": gated_halt_dense_loss.detach(),
            "gated_halt_dense_loss": gated_halt_dense_loss.detach(),
            "projection_alpha": gate_alpha.detach(),
            "projection_dot": zero_f32,
            "projection_ce_grad_norm_sq": zero_f32,
            "gate_alpha": gate_alpha.detach(),
            "gate_ce_loss_ema": ce_loss_ema.detach(),
            "projection_mode": mode,
        }

    def _halt_dense_token_ids_tensor(self, device: torch.device) -> torch.Tensor | None:
        if not self.halt_dense_token_ids:
            return None
        return torch.tensor(self.halt_dense_token_ids, device=device, dtype=torch.long)

    def _resolve_eval_max_new_tokens(self, global_step: int) -> int:
        final_max_new_tokens = int(self.config.get("val_max_new_tokens", self.config["student_max_new_tokens"]))
        warmup_steps = int(self.config.get("val_max_new_tokens_warmup_steps", 0) or 0)
        if warmup_steps <= 0 or int(global_step) > warmup_steps:
            return final_max_new_tokens
        return int(self.config.get("val_max_new_tokens_warmup", final_max_new_tokens))

    @staticmethod
    def _assert_latent_label_masks(batch: Dict[str, Any]) -> None:
        labels = batch["labels"]
        latent_internal_mask = batch["latent_internal_mask"].to(torch.bool)
        latent_pad_mask = batch["latent_pad_mask"].to(torch.bool)

        latent_internal_label_leak = int(((labels != -100) & latent_internal_mask).sum().item())
        latent_pad_label_leak = int(((labels != -100) & latent_pad_mask).sum().item())
        if latent_internal_label_leak > 0 or latent_pad_label_leak > 0:
            raise ValueError(
                "Invalid CE labels detected in latent internal or latent padding regions: "
                f"latent_internal_label_leak={latent_internal_label_leak}, "
                f"latent_pad_label_leak={latent_pad_label_leak}"
            )

        loss_target_positions = batch.get("loss_target_positions")
        loss_pair_mask = batch.get("loss_pair_mask")
        if loss_target_positions is None or loss_pair_mask is None or not torch.is_tensor(loss_target_positions):
            return

        batch_size, pair_slots = loss_target_positions.shape
        if pair_slots <= 0:
            return

        pair_rows = torch.arange(batch_size, device=labels.device).unsqueeze(1).expand(batch_size, pair_slots)
        safe_target_positions = loss_target_positions.clamp(min=0, max=max(int(labels.size(1)) - 1, 0))
        valid_pairs = loss_pair_mask.to(torch.bool) & (loss_target_positions >= 0)
        target_latent_internal = valid_pairs & latent_internal_mask[pair_rows, safe_target_positions]
        target_latent_pad = valid_pairs & latent_pad_mask[pair_rows, safe_target_positions]
        target_ignored = valid_pairs & (labels[pair_rows, safe_target_positions] == -100)
        target_latent_internal_count = int(target_latent_internal.sum().item())
        target_latent_pad_count = int(target_latent_pad.sum().item())
        target_ignored_count = int(target_ignored.sum().item())
        if target_latent_internal_count > 0 or target_latent_pad_count > 0 or target_ignored_count > 0:
            raise ValueError(
                "Invalid sparse CE target mapping detected: "
                f"target_latent_internal_count={target_latent_internal_count}, "
                f"target_latent_pad_count={target_latent_pad_count}, "
                f"target_ignored_count={target_ignored_count}"
            )

    def _apply_trainable_mode(self, mode: str) -> Dict[str, float | int | str]:
        unwrapped = self.accelerator.unwrap_model(self.model)
        mode = str(mode)

        if mode not in {"projector_embed_lmhead", "full"}:
            raise ValueError(f"Unsupported fixed trainable mode: {mode}")

        if mode == "full":
            for param in unwrapped.parameters():
                param.requires_grad = True
        else:
            for param in unwrapped.parameters():
                param.requires_grad = False

            for module_name in ["latent_projector", "lm_head"]:
                module = getattr(unwrapped, module_name, None)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = True

            input_embeddings = unwrapped.get_input_embeddings() if hasattr(unwrapped, "get_input_embeddings") else None
            if input_embeddings is not None:
                for param in input_embeddings.parameters():
                    param.requires_grad = True

        trainable_params = [param for param in unwrapped.parameters() if param.requires_grad]
        trainable_count = int(sum(param.numel() for param in trainable_params))
        total_count = int(sum(param.numel() for param in unwrapped.parameters()))
        fraction = float(trainable_count / max(total_count, 1))
        return {
            "mode": mode,
            "trainable_params": trainable_count,
            "total_params": total_count,
            "trainable_fraction": fraction,
        }

    def _sync_trainable_mode_for_stage(
        self,
        train_stage: int,
        global_step: int,
        epoch_index: int,
        step_in_epoch: int,
        reason: str,
    ) -> None:
        target_mode = self.stage1_trainable_mode if int(train_stage) == 1 else self.stage2_trainable_mode
        if self._current_trainable_mode == target_mode:
            return

        stats = self._apply_trainable_mode(mode=target_mode)
        self._current_trainable_mode = target_mode
        self.logger.info(
            {
                "tag": "trainable_mode_switch",
                "reason": reason,
                "stage": int(train_stage),
                "mode": str(target_mode),
                "global_step": int(global_step),
                "epoch_index": int(epoch_index),
                "step_in_epoch": int(step_in_epoch),
                "trainable_params": int(stats["trainable_params"]),
                "total_params": int(stats["total_params"]),
                "trainable_fraction": float(stats["trainable_fraction"]),
            }
        )

    def _build_stage_transition_checkpoint(
        self,
        global_step: int,
        epoch_index: int,
        step_in_epoch: int,
        target_stage: int,
    ) -> TrainLoopResult:
        self.logger.info(
            {
                "tag": "train_stage_boundary_reached",
                "global_step": int(global_step),
                "epoch_index": int(epoch_index),
                "step_in_epoch": int(step_in_epoch),
                "from_stage": 1,
                "to_stage": int(target_stage),
                "distributed_backend": str(self.distributed_backend),
                "automatic_restart": True,
            }
        )
        self.accelerator.wait_for_everyone()
        self.save_checkpoint(
            global_step=int(global_step),
            epoch_index=int(epoch_index),
            step_in_epoch=int(step_in_epoch),
            reason="stage_transition",
            train_stage_override=int(target_stage),
            trainable_mode_override=self.stage2_trainable_mode,
        )
        checkpoint_dir = Path(self.config["output_dir"]) / f"step_{int(global_step):07d}"
        self.logger.info(
            {
                "tag": "train_stage_transition_checkpoint_saved",
                "checkpoint_dir": str(checkpoint_dir),
                "global_step": int(global_step),
                "epoch_index": int(epoch_index),
                "next_step_in_epoch": int(step_in_epoch),
                "target_stage": int(target_stage),
                "trainable_mode": str(self.stage2_trainable_mode),
            }
        )
        self.accelerator.wait_for_everyone()
        return TrainLoopResult(
            status="stage_transition_restart_required",
            checkpoint_dir=str(checkpoint_dir),
            global_step=int(global_step),
            epoch_index=int(epoch_index),
            next_step_in_epoch=int(step_in_epoch),
            target_stage=int(target_stage),
        )

    def _log_trainable_optimizer_coverage(
        self,
        *,
        tag: str,
        train_stage: int,
        global_step: int,
        epoch_index: int,
        step_in_epoch: int,
    ) -> None:
        raw_model = self.accelerator.unwrap_model(self.model)
        all_params = list(raw_model.parameters())
        trainable_params = [param for param in all_params if bool(param.requires_grad)]
        total_param_count = self._count_parameters(all_params, trainable_only=False)
        trainable_param_count = self._count_parameters(trainable_params, trainable_only=False)
        trainable_tensor_count = int(len(trainable_params))
        total_tensor_count = int(len(all_params))
        current_mode = str(self._current_trainable_mode or self.stage1_trainable_mode)
        optimizer_kind = "deepspeed_wrapped" if self.distributed_backend == "deepspeed" else "torch_optimizer"
        reliable_optimizer_counts = self.distributed_backend != "deepspeed"
        optimizer_param_count = self._count_optimizer_parameters(self.optimizer) if reliable_optimizer_counts else -1
        optimizer_tensor_count = self._count_optimizer_tensors(self.optimizer) if reliable_optimizer_counts else -1
        self.logger.info(
            {
                "tag": str(tag),
                "train_stage": int(train_stage),
                "trainable_mode": current_mode,
                "global_step": int(global_step),
                "epoch_index": int(epoch_index),
                "step_in_epoch": int(step_in_epoch),
                "trainable_param_count": int(trainable_param_count),
                "optimizer_param_count": int(optimizer_param_count),
                "total_param_count": int(total_param_count),
                "trainable_tensor_count": int(trainable_tensor_count),
                "optimizer_tensor_count": int(optimizer_tensor_count),
                "total_tensor_count": int(total_tensor_count),
                "trainable_fraction": float(trainable_param_count / max(total_param_count, 1)),
                "optimizer_fraction": float(optimizer_param_count / max(total_param_count, 1))
                if reliable_optimizer_counts
                else None,
                "optimizer_covers_all_trainable": bool(optimizer_param_count >= trainable_param_count)
                if reliable_optimizer_counts
                else None,
                "train_stage_is_full": bool(int(train_stage) >= 2),
                "trainable_is_full": bool(trainable_param_count >= total_param_count),
                "optimizer_is_full": bool(optimizer_param_count >= total_param_count) if reliable_optimizer_counts else None,
                "optimizer_kind": str(optimizer_kind),
                "optimizer_counts_reliable": bool(reliable_optimizer_counts),
            }
        )

    def _make_train_loader(self, epoch_index: int, global_step: int, total_train_steps: int) -> DataLoader:
        train_stage, progress = self._compute_train_stage(global_step=global_step, total_train_steps=total_train_steps)
        self.collator.set_stage(train_stage=train_stage, progress=progress)
        sampler: Sampler[int] = self._sampler_for_epoch(epoch_index)
        if self.debug_first_batch_use_max_length_sample and epoch_index == 0:
            sampler = self._apply_debug_first_batch(sampler)
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.config["micro_batch_size"]),
            sampler=sampler,
            num_workers=int(self.config["num_workers"]),
            pin_memory=True,
            collate_fn=self.collator,
            drop_last=False,
        )

    def _make_val_loader(self, indices: Optional[List[int]] = None) -> DataLoader:
        self.collator.set_stage(train_stage=2, progress=1.0)
        sampler: Sampler[int] | None = None
        shuffle = False
        if indices is not None:
            sampler = FixedOrderSampler(indices)
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.config["eval_batch_size"]),
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=int(self.config["num_workers"]),
            pin_memory=True,
            collate_fn=self.collator,
            drop_last=False,
        )

    def _prepare_dataloader(self, loader: DataLoader, purpose: str) -> DataLoader:
        del purpose
        return self.accelerator.prepare(loader)

    def _effective_global_batch_size(self, num_processes: int) -> int:
        micro_batch_size = max(int(self.config["micro_batch_size"]), 1)
        return int(micro_batch_size * max(int(num_processes), 1))

    def _attach_teacher_kl_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        teacher_pair_mask = batch.get("teacher_kl_pair_mask")
        if teacher_pair_mask is None:
            return batch
        if int(teacher_pair_mask.to(torch.long).sum().item()) <= 0:
            return batch
        teacher_payload = self.teacher_cache.build_batch_tensors(
            record_ids=batch["record_ids"],
            pair_mask=teacher_pair_mask,
            device=batch["input_ids"].device,
        )
        merged = dict(batch)
        merged.update(teacher_payload)
        mismatch_count = int(teacher_payload["teacher_kl_length_mismatch_count"].item())
        missing_count = int(teacher_payload.get("teacher_kl_missing_sample_count", torch.tensor(0, device=batch["input_ids"].device)).item())
        self.teacher_length_mismatch_count += mismatch_count
        if mismatch_count > 0:
            pair_count = int(teacher_pair_mask.to(torch.long).sum().item())
            pairs_per_sample = teacher_pair_mask.to(torch.long).sum(dim=1).detach().cpu().tolist()
            self.logger.warning(
                {
                    "tag": "teacher_kl_length_mismatch_detected",
                    "mismatch_count": int(mismatch_count),
                    "teacher_length_mismatch_count_total": int(self.teacher_length_mismatch_count),
                    "pair_count": int(pair_count),
                    "pairs_per_sample": [int(v) for v in pairs_per_sample],
                    "record_ids": list(batch.get("record_ids", []))[:8],
                }
            )
        if missing_count > 0:
            self.logger.warning(
                {
                    "tag": "teacher_cache_missing_samples_skipped",
                    "missing_teacher_cache_sample_count": int(missing_count),
                    "missing_teacher_cache_record_ids_preview": list(
                        teacher_payload.get("teacher_kl_missing_record_ids_preview", [])
                    ),
                    "record_ids": list(batch.get("record_ids", []))[:8],
                }
            )
        return merged

    def _sync_batch_for_rank_consistency(self, batch: Dict[str, Any], phase: str) -> Dict[str, Any]:
        if not self.enable_rank_pad_sync:
            return batch
        input_ids = batch.get("input_ids")
        if input_ids is None or not torch.is_tensor(input_ids):
            return batch

        pad_specs: Dict[str, int | float | bool] = {
            "input_ids": int(self.tokenizer.pad_token_id),
            "labels": -100,
            "loss_weights": 0.0,
            "attention_mask": 0,
            "position_ids": 0,
            "prompt_mask": False,
            "latent_internal_mask": False,
            "latent_boundary_mask": False,
            "cot_mask": False,
            "answer_mask": False,
            "teacher_kl_mask": False,
            "valid_token_mask": False,
            "latent_pad_mask": False,
            "latent_positions": -1,
            "latent_slot_mask": False,
            "latent_lengths": 0,
            "latent_start_positions": 0,
            "latent_end_positions": 0,
            "teacher_target_start": 0,
            "loss_source_positions": -1,
            "loss_target_positions": -1,
            "loss_pair_mask": False,
            "teacher_kl_source_positions": -1,
            "teacher_kl_target_positions": -1,
            "teacher_kl_pair_mask": False,
        }
        if phase == "post_teacher":
            pad_specs.update(
                {
                    "teacher_kl_topk_ids": 0,
                    "teacher_kl_topk_probs": 0.0,
                    "teacher_kl_tail": 0.0,
                    "teacher_kl_effective_mask": False,
                }
            )

        synced = sync_batch_across_ranks(batch=batch, pad_specs=pad_specs, device=input_ids.device)
        if bool(self.config.get("enable_hybrid_cache_debug_log", False)):
            self.logger.info(
                {
                    "tag": "rank_pad_sync",
                    "phase": str(phase),
                    "batch_size": int(synced["input_ids"].size(0)),
                    "seq_len": int(synced["input_ids"].size(1)),
                    "latent_slots": int(synced["latent_slot_mask"].size(1))
                    if synced.get("latent_slot_mask") is not None
                    else 0,
                    "loss_pairs": int(synced["loss_pair_mask"].size(1))
                    if synced.get("loss_pair_mask") is not None
                    else 0,
                    "kl_pairs": int(synced["teacher_kl_pair_mask"].size(1))
                    if synced.get("teacher_kl_pair_mask") is not None
                    else 0,
                    "kl_topk": int(synced["teacher_kl_topk_ids"].size(2))
                    if synced.get("teacher_kl_topk_ids") is not None
                    else 0,
                }
            )
        return synced

    def _trim_teacher_kl_dim(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.kl_topk_dim <= 0:
            return batch
        topk_ids = batch.get("teacher_kl_topk_ids")
        if topk_ids is None or not torch.is_tensor(topk_ids) or topk_ids.dim() != 3:
            return batch
        current_dim = int(topk_ids.size(-1))
        target_dim = min(int(self.kl_topk_dim), current_dim)
        if target_dim >= current_dim:
            return batch

        trimmed = dict(batch)
        trimmed["teacher_kl_topk_ids"] = topk_ids[:, :, :target_dim].contiguous()
        topk_probs = batch.get("teacher_kl_topk_probs")
        if topk_probs is not None and torch.is_tensor(topk_probs) and topk_probs.dim() == 3:
            trimmed["teacher_kl_topk_probs"] = topk_probs[:, :, :target_dim].contiguous()
        if bool(self.config.get("enable_hybrid_cache_debug_log", False)):
            self.logger.info(
                {
                    "tag": "teacher_kl_dim_trim",
                    "before": int(current_dim),
                    "after": int(target_dim),
                }
            )
        return trimmed

    @staticmethod
    def _allreduce_min_max_int(value: int, device: torch.device) -> tuple[int, int]:
        if not (dist.is_available() and dist.is_initialized()):
            return int(value), int(value)
        min_tensor = torch.tensor([int(value)], device=device, dtype=torch.long)
        max_tensor = torch.tensor([int(value)], device=device, dtype=torch.long)
        dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
        return int(min_tensor.item()), int(max_tensor.item())

    @staticmethod
    def _local_stage3_start(batch: Dict[str, Any]) -> int:
        input_ids = batch["input_ids"]
        cot_mask = batch["cot_mask"].to(torch.bool)
        seq_len = int(input_ids.size(1))
        has_true = cot_mask.any(dim=1)
        first = torch.argmax(cot_mask.to(torch.long), dim=1)
        fallback = torch.full_like(first, seq_len)
        think_start_positions = torch.where(has_true, first, fallback)
        return int(think_start_positions.max().item())

    def _compute_global_prompt_len(self, batch: Dict[str, Any]) -> int:
        local_prompt_len = 0
        if torch.is_tensor(batch.get("latent_start_positions")):
            local_prompt_len = int(batch["latent_start_positions"].max().item()) + 1
        device = batch["input_ids"].device
        if not (dist.is_available() and dist.is_initialized()):
            return int(local_prompt_len)
        prompt_tensor = torch.tensor([int(local_prompt_len)], device=device, dtype=torch.long)
        dist.all_reduce(prompt_tensor, op=dist.ReduceOp.MAX)
        return int(prompt_tensor.item())

    def _compute_global_stage3_start(self, batch: Dict[str, Any]) -> int:
        local_stage3_start = self._local_stage3_start(batch)
        device = batch["input_ids"].device
        if not (dist.is_available() and dist.is_initialized()):
            return int(local_stage3_start)
        stage3_tensor = torch.tensor([int(local_stage3_start)], device=device, dtype=torch.long)
        dist.all_reduce(stage3_tensor, op=dist.ReduceOp.MAX)
        return int(stage3_tensor.item())

    @staticmethod
    def _broadcast_token_ids_from_rank0(token_ids: List[int], device: torch.device) -> List[int]:
        local_ids = [int(v) for v in token_ids]
        if not (dist.is_available() and dist.is_initialized()):
            return local_ids

        rank = int(dist.get_rank())
        if rank == 0:
            payload = torch.tensor(local_ids, device=device, dtype=torch.long)
            length = torch.tensor([int(payload.numel())], device=device, dtype=torch.long)
        else:
            payload = torch.empty((0,), device=device, dtype=torch.long)
            length = torch.zeros((1,), device=device, dtype=torch.long)

        dist.broadcast(length, src=0)
        target_len = int(length.item())
        if rank != 0:
            payload = torch.zeros((target_len,), device=device, dtype=torch.long)
        if target_len > 0:
            dist.broadcast(payload, src=0)
        return [int(v) for v in payload.detach().cpu().tolist()]

    @staticmethod
    def _broadcast_int_from_rank0(value: int, device: torch.device) -> int:
        if not (dist.is_available() and dist.is_initialized()):
            return int(value)
        rank = int(dist.get_rank())
        payload = torch.tensor([int(value) if rank == 0 else 0], device=device, dtype=torch.long)
        dist.broadcast(payload, src=0)
        return int(payload.item())

    @staticmethod
    def _all_reduce_mean_tensor(values: torch.Tensor) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            return values
        reduced = values.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced = reduced / float(max(int(dist.get_world_size()), 1))
        return reduced

    @staticmethod
    def _all_gather_objects(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not (dist.is_available() and dist.is_initialized()):
            return [payload]
        world_size = int(dist.get_world_size())
        gathered: List[Dict[str, Any] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, payload)
        return [item for item in gathered if isinstance(item, dict)]

    def _maybe_log_rank_boundary_consistency(
        self,
        batch: Dict[str, Any],
        global_prompt_len: int,
        global_stage3_start: int,
        global_step: int = -1,
    ) -> None:
        if (
            self.rank_boundary_probe_once
            and self._rank_boundary_probe_done
            and int(global_step) > int(self.rank_boundary_probe_until_step)
        ):
            return

        input_ids = batch["input_ids"]
        device = input_ids.device
        local_seq_len = int(input_ids.size(1))
        latent_positions = batch.get("latent_positions")
        local_latent_slots = int(latent_positions.size(1)) if torch.is_tensor(latent_positions) else 0
        local_prompt_len = 0
        if torch.is_tensor(batch.get("latent_start_positions")):
            local_prompt_len = int(batch["latent_start_positions"].max().item()) + 1
        local_stage3_start = self._local_stage3_start(batch)

        prompt_min, prompt_max = self._allreduce_min_max_int(local_prompt_len, device=device)
        latent_min, latent_max = self._allreduce_min_max_int(local_latent_slots, device=device)
        stage3_min, stage3_max = self._allreduce_min_max_int(local_stage3_start, device=device)
        seq_min, seq_max = self._allreduce_min_max_int(local_seq_len, device=device)

        mismatch = (
            (prompt_min != prompt_max)
            or (latent_min != latent_max)
            or (stage3_min != stage3_max)
            or (seq_min != seq_max)
        )
        self.logger.info(
            {
                "tag": "rank_boundary_consistency",
                "prompt_len_minmax": [int(prompt_min), int(prompt_max)],
                "latent_slots_minmax": [int(latent_min), int(latent_max)],
                "stage3_start_minmax": [int(stage3_min), int(stage3_max)],
                "seq_len_minmax": [int(seq_min), int(seq_max)],
                "global_prompt_len": int(global_prompt_len),
                "global_stage3_start": int(global_stage3_start),
                "global_step": int(global_step),
                "probe_until_step": int(self.rank_boundary_probe_until_step),
                "consistent": not bool(mismatch),
            }
        )
        self._rank_boundary_probe_done = int(global_step) >= int(self.rank_boundary_probe_until_step)
        if mismatch and self.raise_on_rank_boundary_mismatch:
            raise RuntimeError("Detected cross-rank boundary mismatch before model forward.")

    def _populate_latent_memory_info(
        self,
        outputs: Any,
        batch: Dict[str, Any],
        global_prompt_len: int,
        global_stage3_start: int,
    ) -> None:
        info = dict(getattr(outputs, "latent_memory_info", {}) or {})
        seq_len = int(batch["input_ids"].size(1))
        prompt_len = int(global_prompt_len)
        info["prompt_stage_tokens"] = int(max(prompt_len, 0))
        info["latent_stage_tokens"] = int(max(global_stage3_start - prompt_len, 0))
        info["discrete_stage_tokens"] = int(max(seq_len - global_stage3_start, 0))
        if self.accelerator.is_main_process:
            sequence_limits = batch["attention_mask"].to(torch.long).sum(dim=1).detach().cpu().tolist()
            info["sequence_limits"] = [int(v) for v in sequence_limits]
        else:
            info["sequence_limits"] = []
        outputs.latent_memory_info = info

    def _should_run_zero3_trace_probe(self, global_step: int) -> bool:
        if not self.enable_zero3_trace_probe_log:
            return False
        return int(global_step) % int(self.zero3_trace_probe_every_steps) == 0

    def _should_run_deepspeed_submodule_trace_probe(self, global_step: int) -> bool:
        if not self.enable_deepspeed_submodule_trace_log:
            return False
        return int(global_step) % int(self.deepspeed_submodule_trace_every_steps) == 0

    @staticmethod
    def _trace_probe_module_name(name: str) -> str:
        return name if name else "<root>"

    def _maybe_install_zero3_trace_probe_hooks(self) -> None:
        if self._zero3_trace_probe_installed:
            return
        base_model = self.accelerator.unwrap_model(self.model)
        for name, module in base_model.named_modules():
            if not any(True for _ in module.parameters(recurse=False)):
                continue
            module_name = self._trace_probe_module_name(name)

            def _hook(_module: Any, _inputs: Any, *, _module_name: str = module_name) -> None:
                if not self._zero3_trace_probe_active:
                    return
                self._zero3_trace_probe_call_index += 1
                self._zero3_trace_probe_counts[_module_name] = self._zero3_trace_probe_counts.get(_module_name, 0) + 1
                if len(self._zero3_trace_probe_prefix) < self.zero3_trace_probe_record_limit:
                    self._zero3_trace_probe_prefix.append(_module_name)
                self._zero3_trace_probe_suffix.append(_module_name)
                if (
                    self.zero3_trace_probe_log_per_call
                    and self._zero3_trace_probe_call_index <= self.zero3_trace_probe_call_log_limit
                ):
                    self.logger.info(
                        {
                            "tag": "zero3_trace_probe_call",
                            "global_step": int(self._zero3_trace_probe_step),
                            "epoch_index": int(self._zero3_trace_probe_epoch),
                            "call_index": int(self._zero3_trace_probe_call_index),
                            "module": _module_name,
                        }
                    )

            self._zero3_trace_probe_handles.append(module.register_forward_pre_hook(_hook))
        self._zero3_trace_probe_installed = True

    def _get_deepspeed_param_coordinator(self) -> Any | None:
        engine = self.model
        candidates = [
            getattr(getattr(engine, "optimizer", None), "parameter_offload", None),
            getattr(engine, "parameter_offload", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            getter = getattr(candidate, "get_param_coordinator", None)
            if callable(getter):
                try:
                    return getter()
                except Exception:
                    continue
        return None

    def _build_deepspeed_ds_id_name_map(self) -> Dict[int, str]:
        engine = self.model
        raw_model = getattr(engine, "module", None)
        if raw_model is None:
            return {}
        mapping: Dict[int, str] = {}
        for name, module in raw_model.named_modules():
            ds_id = getattr(module, "ds_id", None)
            if ds_id is None:
                continue
            label = name if name else "<root>"
            mapping[int(ds_id)] = str(label)
        return mapping

    def _summarize_deepspeed_submodule_trace(
        self,
        coordinator: Any,
        reason: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        ds_id_to_name = self._build_deepspeed_ds_id_name_map()
        trace_mode = getattr(coordinator, "_PartitionedParameterCoordinator__trace_mode", None)
        submodule_order = getattr(coordinator, "_PartitionedParameterCoordinator__submodule_order", ())
        step_id = getattr(coordinator, "_PartitionedParameterCoordinator__step_id", None)
        ds_ids: List[int] = []
        module_names: List[str] = []
        for module in list(submodule_order):
            ds_id = getattr(module, "ds_id", None)
            if ds_id is None:
                continue
            ds_id_int = int(ds_id)
            ds_ids.append(ds_id_int)
            module_names.append(ds_id_to_name.get(ds_id_int, f"<ds_id:{ds_id_int}>"))
        counts = Counter(module_names)
        top_counts = [
            {"module": str(name), "count": int(count)}
            for name, count in counts.most_common(self.zero3_trace_probe_max_modules)
        ]
        payload: Dict[str, Any] = {
            "tag": "deepspeed_submodule_trace_summary",
            "reason": str(reason),
            "global_step": int(self._deepspeed_trace_probe_step),
            "epoch_index": int(self._deepspeed_trace_probe_epoch),
            "trace_mode": str(trace_mode),
            "coordinator_step_id": int(step_id) if step_id is not None else None,
            "submodule_order_len": int(len(ds_ids)),
            "unique_submodules": int(len(counts)),
            "prefix_ds_ids": ds_ids[: self.deepspeed_submodule_trace_record_limit],
            "suffix_ds_ids": ds_ids[-self.deepspeed_submodule_trace_record_limit :],
            "prefix_modules": module_names[: self.deepspeed_submodule_trace_record_limit],
            "suffix_modules": module_names[-self.deepspeed_submodule_trace_record_limit :],
            "top_module_counts": top_counts,
            "recorded_events_preview": list(self._deepspeed_trace_probe_recorded),
            **self._deepspeed_trace_probe_batch_meta,
        }
        payload["ds_id_sha1"] = hashlib.sha1(json.dumps(ds_ids, ensure_ascii=True).encode("utf-8")).hexdigest()
        payload["module_name_sha1"] = hashlib.sha1(
            json.dumps(module_names, ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        if extra:
            payload.update(extra)
        self.logger.info(payload)

    def _maybe_install_deepspeed_submodule_trace_patch(self) -> None:
        if self._deepspeed_trace_patch_installed:
            return
        coordinator = self._get_deepspeed_param_coordinator()
        if coordinator is None:
            return
        original_reset_step = coordinator.reset_step
        original_record_module = coordinator.record_module
        trainer = self

        def wrapped_record_module(sub_module: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_record_module(sub_module, *args, **kwargs)
            if trainer._deepspeed_trace_probe_active:
                ds_id = getattr(sub_module, "ds_id", None)
                ds_name = trainer._build_deepspeed_ds_id_name_map().get(
                    int(ds_id) if ds_id is not None else -1,
                    sub_module.__class__.__name__,
                )
                event = {
                    "index": int(len(trainer._deepspeed_trace_probe_recorded)),
                    "ds_id": int(ds_id) if ds_id is not None else None,
                    "module": str(ds_name),
                    "class": sub_module.__class__.__name__,
                }
                if len(trainer._deepspeed_trace_probe_recorded) < trainer.deepspeed_submodule_trace_record_limit:
                    trainer._deepspeed_trace_probe_recorded.append(event)
                if (
                    trainer.deepspeed_submodule_trace_log_per_record
                    and len(trainer._deepspeed_trace_probe_recorded)
                    <= trainer.deepspeed_submodule_trace_log_record_limit
                ):
                    trainer.logger.info({"tag": "deepspeed_submodule_trace_record", **event})
            return result

        def wrapped_reset_step(*args: Any, **kwargs: Any) -> Any:
            if trainer._deepspeed_trace_probe_active:
                trainer._summarize_deepspeed_submodule_trace(
                    coordinator=coordinator,
                    reason="reset_step_before",
                )
            try:
                result = original_reset_step(*args, **kwargs)
            except Exception as exc:
                trainer._summarize_deepspeed_submodule_trace(
                    coordinator=coordinator,
                    reason="reset_step_exception",
                    extra={
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
                raise
            if trainer._deepspeed_trace_probe_active:
                trainer._summarize_deepspeed_submodule_trace(
                    coordinator=coordinator,
                    reason="reset_step_after",
                )
            return result

        coordinator.record_module = wrapped_record_module
        coordinator.reset_step = wrapped_reset_step
        self._deepspeed_trace_patch_installed = True

    def _start_zero3_trace_probe(self, batch: Dict[str, Any], global_step: int, epoch_index: int) -> None:
        if not self._should_run_zero3_trace_probe(global_step):
            return
        self._maybe_install_zero3_trace_probe_hooks()
        self._zero3_trace_probe_active = True
        self._zero3_trace_probe_counts = {}
        self._zero3_trace_probe_prefix = []
        self._zero3_trace_probe_suffix = deque(maxlen=self.zero3_trace_probe_record_limit)
        self._zero3_trace_probe_call_index = 0
        self._zero3_trace_probe_step = int(global_step)
        self._zero3_trace_probe_epoch = int(epoch_index)
        self._zero3_trace_probe_batch_meta = {
            "record_ids": list(batch.get("record_ids", [])),
            "seq_len": int(batch["input_ids"].size(1)),
            "batch_size": int(batch["input_ids"].size(0)),
        }
        self.logger.info(
            {
                "tag": "zero3_trace_probe_start",
                "global_step": int(global_step),
                "epoch_index": int(epoch_index),
                **self._zero3_trace_probe_batch_meta,
            }
        )

    def _start_deepspeed_submodule_trace_probe(self, batch: Dict[str, Any], global_step: int, epoch_index: int) -> None:
        if not self._should_run_deepspeed_submodule_trace_probe(global_step):
            return
        self._maybe_install_deepspeed_submodule_trace_patch()
        self._deepspeed_trace_probe_active = True
        self._deepspeed_trace_probe_step = int(global_step)
        self._deepspeed_trace_probe_epoch = int(epoch_index)
        self._deepspeed_trace_probe_batch_meta = {
            "record_ids": list(batch.get("record_ids", [])),
            "seq_len": int(batch["input_ids"].size(1)),
            "batch_size": int(batch["input_ids"].size(0)),
        }
        self._deepspeed_trace_probe_recorded = []
        self.logger.info(
            {
                "tag": "deepspeed_submodule_trace_start",
                "global_step": int(global_step),
                "epoch_index": int(epoch_index),
                **self._deepspeed_trace_probe_batch_meta,
            }
        )

    def _finish_zero3_trace_probe(
        self,
        reason: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._zero3_trace_probe_active:
            return
        self._zero3_trace_probe_active = False
        sorted_counts = sorted(self._zero3_trace_probe_counts.items(), key=lambda item: (-item[1], item[0]))
        top_counts = [
            {"module": str(name), "count": int(count)}
            for name, count in sorted_counts[: self.zero3_trace_probe_max_modules]
        ]
        payload: Dict[str, Any] = {
            "tag": "zero3_trace_probe_summary",
            "reason": str(reason),
            "global_step": int(self._zero3_trace_probe_step),
            "epoch_index": int(self._zero3_trace_probe_epoch),
            "total_module_calls": int(sum(self._zero3_trace_probe_counts.values())),
            "unique_modules_called": int(len(self._zero3_trace_probe_counts)),
            "prefix_modules": list(self._zero3_trace_probe_prefix),
            "suffix_modules": list(self._zero3_trace_probe_suffix),
            "top_module_counts": top_counts,
            **self._zero3_trace_probe_batch_meta,
        }
        payload["prefix_sha1"] = hashlib.sha1(
            json.dumps(payload["prefix_modules"], ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        payload["suffix_sha1"] = hashlib.sha1(
            json.dumps(payload["suffix_modules"], ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        payload["count_sha1"] = hashlib.sha1(json.dumps(sorted_counts, ensure_ascii=True).encode("utf-8")).hexdigest()
        if extra:
            payload.update(extra)
        self.logger.info(payload)
        self._zero3_trace_probe_batch_meta = {}

    def _finish_deepspeed_submodule_trace_probe(self, reason: str, extra: Optional[Dict[str, Any]] = None) -> None:
        if not self._deepspeed_trace_probe_active:
            return
        coordinator = self._get_deepspeed_param_coordinator()
        if coordinator is not None:
            self._summarize_deepspeed_submodule_trace(
                coordinator=coordinator,
                reason=reason,
                extra=extra,
            )
        self._deepspeed_trace_probe_active = False
        self._deepspeed_trace_probe_batch_meta = {}
        self._deepspeed_trace_probe_recorded = []

    def _forward_loss(
        self,
        batch: Dict[str, Any],
        global_step: int = -1,
        epoch_index: int = -1,
        train_stage: Optional[int] = None,
        train_progress: Optional[float] = None,
    ) -> Dict[str, Any]:
        debug_enabled = bool(self.config.get("enable_hybrid_cache_debug_log", False))
        step_trace_enabled = False
        if step_trace_enabled:
            self.logger.info(
                {
                    "tag": "forward_stage",
                    "phase": "start",
                    "kl_enabled": bool(batch.get("teacher_kl_pair_mask") is not None),
                }
            )
        if debug_enabled:
            self.logger.info(
                {
                    "tag": "forward_loss_attach_teacher_start",
                    "teacher_pair_mask_present": bool(batch.get("teacher_kl_pair_mask") is not None),
                    "record_ids": list(batch.get("record_ids", [])),
                }
            )
        kl_loss_weight = float(self.config.get("kl_loss_weight", 0.0))
        if step_trace_enabled:
            self.logger.info({"tag": "forward_stage", "phase": "pre_teacher_sync_start"})
        batch = self._sync_batch_for_rank_consistency(batch=batch, phase="pre_teacher")
        if step_trace_enabled:
            self.logger.info({"tag": "forward_stage", "phase": "pre_teacher_sync_done"})
        teacher_pair_mask_pre = batch.get("teacher_kl_pair_mask")
        if teacher_pair_mask_pre is not None:
            has_requested_kl_pairs = int(teacher_pair_mask_pre.to(torch.long).sum().item()) > 0
        else:
            has_requested_kl_pairs = False
        kl_enabled_for_batch = bool(kl_loss_weight > 0.0 and has_requested_kl_pairs)
        if kl_enabled_for_batch:
            if step_trace_enabled:
                self.logger.info({"tag": "forward_stage", "phase": "attach_teacher_start"})
            batch = self._attach_teacher_kl_batch(batch)
            if step_trace_enabled:
                self.logger.info({"tag": "forward_stage", "phase": "attach_teacher_done"})
            batch = self._trim_teacher_kl_dim(batch)
            if step_trace_enabled:
                self.logger.info({"tag": "forward_stage", "phase": "post_teacher_sync_start"})
            batch = self._sync_batch_for_rank_consistency(batch=batch, phase="post_teacher")
            if step_trace_enabled:
                self.logger.info({"tag": "forward_stage", "phase": "post_teacher_sync_done"})
        elif debug_enabled:
            self.logger.info(
                {
                    "tag": "forward_loss_kl_hotpath_skipped",
                    "kl_loss_weight": float(kl_loss_weight),
                    "has_requested_kl_pairs": bool(has_requested_kl_pairs),
                }
            )
        self._assert_latent_label_masks(batch)
        current_stage = int(train_stage) if train_stage is not None else int(self.config.get("train_stage", 1))
        current_progress = float(train_progress) if train_progress is not None else 0.0
        halt_dense_loss_weight = self._compute_halt_dense_loss_weight(
            train_stage=current_stage,
            train_progress=current_progress,
        )
        global_prompt_len = self._compute_global_prompt_len(batch)
        global_stage3_start = self._compute_global_stage3_start(batch)
        self._maybe_log_rank_boundary_consistency(
            batch=batch,
            global_prompt_len=global_prompt_len,
            global_stage3_start=global_stage3_start,
            global_step=global_step,
        )
        if step_trace_enabled:
            self.logger.info(
                {
                    "tag": "forward_stage",
                    "phase": "stage3_decision",
                    "global_stage3_start": int(global_stage3_start),
                    "seq_len": int(batch["input_ids"].size(1)),
                    "skip_discrete": bool(int(global_stage3_start) >= int(batch["input_ids"].size(1))),
                }
            )
        if debug_enabled:
            self.logger.info(
                {
                    "tag": "forward_loss_attach_teacher_done",
                    "teacher_length_mismatch_count": int(self.teacher_length_mismatch_count),
                    "teacher_topk_shape": list(batch["teacher_kl_topk_ids"].shape)
                    if batch.get("teacher_kl_topk_ids") is not None
                    else None,
                }
            )
            self.logger.info(
                {
                    "tag": "forward_loss_model_start",
                    "input_shape": list(batch["input_ids"].shape),
                    "attention_shape": list(batch["attention_mask"].shape),
                    "latent_internal_shape": list(batch["latent_internal_mask"].shape),
                }
            )
        if step_trace_enabled:
            self.logger.info({"tag": "forward_stage", "phase": "model_forward_start"})
        self._start_zero3_trace_probe(batch=batch, global_step=global_step, epoch_index=epoch_index)
        self._start_deepspeed_submodule_trace_probe(batch=batch, global_step=global_step, epoch_index=epoch_index)
        try:
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                attention_bias=batch.get("attention_bias"),
                position_ids=batch["position_ids"],
                labels=batch["labels"],
                cot_mask=batch["cot_mask"],
                latent_internal_mask=batch["latent_internal_mask"],
                latent_positions=batch["latent_positions"],
                latent_slot_mask=batch["latent_slot_mask"],
                latent_lengths=batch["latent_lengths"],
                latent_start_positions=batch["latent_start_positions"],
                latent_end_positions=batch["latent_end_positions"],
                halt_dense_token_ids=self._halt_dense_token_ids_tensor(device=batch["input_ids"].device),
                loss_source_positions=batch.get("loss_source_positions"),
                loss_target_positions=batch.get("loss_target_positions"),
                loss_pair_mask=batch.get("loss_pair_mask"),
                teacher_kl_source_positions=batch.get("teacher_kl_source_positions") if kl_enabled_for_batch else None,
                teacher_kl_pair_mask=(
                    batch.get("teacher_kl_effective_mask", batch.get("teacher_kl_pair_mask"))
                    if kl_enabled_for_batch
                    else None
                ),
                teacher_kl_topk_ids=batch.get("teacher_kl_topk_ids") if kl_enabled_for_batch else None,
                global_prompt_len=int(global_prompt_len),
                global_stage3_start=int(global_stage3_start),
                use_cache=False,
                return_dict=True,
            )
        except Exception as exc:
            self._finish_zero3_trace_probe(
                reason="model_forward_exception",
                extra={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            self._finish_deepspeed_submodule_trace_probe(
                reason="model_forward_exception",
                extra={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            raise
        self._populate_latent_memory_info(
            outputs=outputs,
            batch=batch,
            global_prompt_len=global_prompt_len,
            global_stage3_start=global_stage3_start,
        )
        self._finish_zero3_trace_probe(
            reason="model_forward_done",
            extra={"stage_trace_info": getattr(outputs, "stage_trace_info", None)},
        )
        self._finish_deepspeed_submodule_trace_probe(
            reason="model_forward_done",
            extra={"stage_trace_info": getattr(outputs, "stage_trace_info", None)},
        )
        if step_trace_enabled:
            self.logger.info({"tag": "forward_stage", "phase": "model_forward_done"})
            self.logger.info(
                {
                    "tag": "model_sparse_anchor_summary",
                    "total_lm_head_calls": (
                        int(getattr(outputs, "stage_trace_info", {}).get("total_lm_head_calls", 0))
                        if getattr(outputs, "stage_trace_info", None) is not None
                        else 0
                    ),
                    "ce_sync_zero_loss_present": getattr(outputs, "ce_sync_zero_loss", None) is not None,
                    "kl_sync_zero_loss_present": getattr(outputs, "kl_sync_zero_loss", None) is not None,
                }
            )
            self.logger.info(
                {
                    "tag": "discrete_chunk_summary",
                    "stage3_start": int(getattr(outputs, "stage_trace_info", {}).get("stage3_start", 0)),
                    "seq_len": int(batch["input_ids"].size(1)),
                    "discrete_chunk_size": int(getattr(outputs, "stage_trace_info", {}).get("discrete_chunk_size", 0)),
                    "discrete_chunk_count": int(getattr(outputs, "stage_trace_info", {}).get("discrete_chunk_count", 0)),
                    "discrete_total_len": int(getattr(outputs, "stage_trace_info", {}).get("discrete_stage_tokens", 0)),
                }
            )
        if debug_enabled:
            self.logger.info(
                {
                    "tag": "forward_loss_model_done",
                    "halt_dense_logits_shape": list(getattr(outputs, "halt_dense_logits").shape)
                    if getattr(outputs, "halt_dense_logits", None) is not None
                    else None,
                    "loss_hidden_shape": list(getattr(outputs, "loss_hidden_states").shape)
                    if getattr(outputs, "loss_hidden_states", None) is not None
                    else None,
                    "loss_target_logits_shape": list(getattr(outputs, "loss_target_logits").shape)
                    if getattr(outputs, "loss_target_logits", None) is not None
                    else None,
                    "teacher_kl_topk_logits_shape": list(getattr(outputs, "teacher_kl_topk_logits").shape)
                    if getattr(outputs, "teacher_kl_topk_logits", None) is not None
                    else None,
                }
            )
        sparse_loss_hidden = getattr(outputs, "loss_hidden_states", None)
        logits_chunk_size = int(self.config.get("supervised_logits_chunk_size", 0) or 0)
        if step_trace_enabled:
            self.logger.info(
                {
                    "tag": "loss_chunk_summary",
                    "distributed_backend": str(self.distributed_backend),
                    "supervised_logits_chunk_size": int(logits_chunk_size),
                    "projection_chunk_size": int(getattr(self.model.config, "supervised_logits_chunk_size", 0) or 0)
                    if hasattr(self.model, "config")
                    else int(logits_chunk_size),
                    "loss_pair_slots": int(batch.get("loss_pair_mask").size(1))
                    if batch.get("loss_pair_mask") is not None
                    else 0,
                    "teacher_kl_pair_slots": (
                        int(batch.get("teacher_kl_pair_mask").size(1))
                        if batch.get("teacher_kl_pair_mask") is not None
                        else 0
                    ),
                }
            )
        if debug_enabled:
            self.logger.info({"tag": "forward_loss_ce_start"})
        ce_metrics = compute_weighted_ce(
            logits=outputs.logits,
            hidden_states=sparse_loss_hidden,
            lm_head=None,
            logits_chunk_size=logits_chunk_size,
            labels=batch["labels"],
            loss_weights=batch["loss_weights"],
            cot_mask=batch["cot_mask"],
            cot_branch_weight=batch["cot_branch_weight"],
            loss_source_positions=batch.get("loss_source_positions"),
            loss_target_positions=batch.get("loss_target_positions"),
            loss_pair_mask=batch.get("loss_pair_mask"),
            selected_target_logits=getattr(outputs, "loss_target_logits", None),
            selected_log_denom=getattr(outputs, "loss_log_denom", None),
        )
        if debug_enabled:
            self.logger.info(
                {
                    "tag": "forward_loss_ce_done",
                    "ce_loss": float(ce_metrics["ce_loss"].detach().item()),
                    "cot_ce": float(ce_metrics["cot_ce"].detach().item()),
                    "non_cot_ce": float(ce_metrics["non_cot_ce"].detach().item()),
                    "cot_weight": float(ce_metrics["cot_weight"].detach().item()),
                }
            )
        kl_stats: Dict[str, Any] | None = {} if self.enable_forward_memory_breakdown_log else None
        if debug_enabled:
            self.logger.info({"tag": "forward_loss_kl_start"})
        teacher_effective_mask = batch.get("teacher_kl_effective_mask")
        teacher_pair_mask = batch.get("teacher_kl_pair_mask")
        if teacher_effective_mask is not None:
            has_valid_kl_pairs = int(teacher_effective_mask.to(torch.long).sum().item()) > 0
        elif teacher_pair_mask is not None:
            has_valid_kl_pairs = int(teacher_pair_mask.to(torch.long).sum().item()) > 0
        else:
            has_valid_kl_pairs = False

        if kl_enabled_for_batch and has_valid_kl_pairs:
            kl_loss, kl_positions = compute_teacher_kl(
                logits=outputs.logits,
                batch=batch,
                teacher_cache=self.teacher_cache,
                kl_temperature=float(self.config["kl_temperature"]),
                stats=kl_stats,
                hidden_states=None,
                lm_head=None,
                logits_chunk_size=logits_chunk_size,
                selected_topk_logits=getattr(outputs, "teacher_kl_topk_logits", None),
                log_denom=getattr(outputs, "teacher_kl_log_denom", None),
            )
        else:
            kl_loss = batch["input_ids"].new_zeros((), dtype=torch.float32)
            kl_positions = 0
            if kl_stats is not None:
                kl_stats.update(
                    {
                        "student_logits_slice_gib": 0.0,
                        "teacher_cache_tensor_gib": 0.0,
                        "student_topk_tensor_gib": 0.0,
                        "kl_tensor_gib": 0.0,
                        "kl_positions": 0,
                        "missing_teacher_cache_sample_count": 0,
                        "missing_teacher_cache_record_ids_preview": [],
                        "skipped_kl_positions_due_to_missing_cache": 0,
                    }
                )
            if debug_enabled:
                self.logger.info(
                    {
                        "tag": "forward_loss_kl_skipped_no_valid_pairs",
                        "teacher_pair_count": int(teacher_pair_mask.to(torch.long).sum().item())
                        if teacher_pair_mask is not None
                        else 0,
                        "teacher_effective_pair_count": int(teacher_effective_mask.to(torch.long).sum().item())
                        if teacher_effective_mask is not None
                        else 0,
                    }
                )
        if debug_enabled:
            self.logger.info(
                {
                    "tag": "forward_loss_kl_done",
                    "kl_positions": int(kl_positions),
                    "kl_loss": float(kl_loss.detach().item()),
                    "missing_teacher_cache_sample_count": int((kl_stats or {}).get("missing_teacher_cache_sample_count", 0)),
                    "missing_teacher_cache_record_ids_preview": list(
                        (kl_stats or {}).get("missing_teacher_cache_record_ids_preview", [])
                    ),
                    "skipped_kl_positions_due_to_missing_cache": int(
                        (kl_stats or {}).get("skipped_kl_positions_due_to_missing_cache", 0)
                    ),
                }
            )
        halt_dense_metrics = compute_early_exit_rank_loss(
            halt_dense_logits=getattr(outputs, "halt_dense_logits", None),
            halt_dense_best_allowed_logits=getattr(outputs, "halt_dense_best_allowed_logits", None),
            latent_internal_mask=batch["latent_internal_mask"],
            attention_mask=batch["attention_mask"],
            latent_pad_mask=batch.get("latent_pad_mask"),
            front_fraction=float(self.config.get("early_exit_front_fraction", 0.25)),
            front_weight=float(self.config.get("early_exit_front_weight", 2.0)),
            nonfront_weight=float(self.config.get("early_exit_nonfront_weight", 1.0)),
            margin=float(self.config.get("early_exit_rank_margin", 0.0)),
            soft_target_curve=str(self.config.get("latent_end_soft_target_curve", "smoothstep")),
            soft_target_temperature=float(self.config.get("latent_end_soft_target_temperature", 0.2)),
            soft_target_power=float(self.config.get("latent_end_soft_target_power", 1.5)),
            soft_loss_weight=float(self.config.get("latent_end_soft_loss_weight", 1.0)),
            other_end_hard_loss_weight=float(self.config.get("other_end_hard_loss_weight", 0.25)),
            other_end_rank_margin=float(self.config.get("other_end_rank_margin", 0.0)),
        )
        halt_dense_loss = halt_dense_metrics["halt_dense_loss"]
        if debug_enabled:
            self.logger.info(
                {
                    "tag": "forward_loss_halt_dense_done",
                    "halt_dense_loss": float(halt_dense_loss.detach().item()),
                    "halt_dense_token_ids": list(self.halt_dense_token_ids),
                    "argmax_violation_rate": float(halt_dense_metrics["argmax_violation_rate"].detach().item()),
                    "front_loss": float(halt_dense_metrics["front_loss"].detach().item()),
                    "nonfront_loss": float(halt_dense_metrics["nonfront_loss"].detach().item()),
                    "latent_end_soft_loss": float(halt_dense_metrics["latent_end_soft_loss"].detach().item()),
                    "other_end_hard_loss": float(halt_dense_metrics["other_end_hard_loss"].detach().item()),
                    "latent_end_target_mean": float(halt_dense_metrics["latent_end_target_mean"].detach().item()),
                    "latent_end_score_mean": float(halt_dense_metrics["latent_end_score_mean"].detach().item()),
                }
            )
        ce_sync_zero_loss = getattr(outputs, "ce_sync_zero_loss", None)
        if ce_sync_zero_loss is None:
            ce_sync_zero_loss = batch["input_ids"].new_zeros((), dtype=torch.float32)
        kl_sync_zero_loss = getattr(outputs, "kl_sync_zero_loss", None)
        if kl_sync_zero_loss is None:
            kl_sync_zero_loss = batch["input_ids"].new_zeros((), dtype=torch.float32)
        if step_trace_enabled:
            self.logger.info(
                {
                    "tag": "sync_zero_loss_summary",
                    "ce_sync_zero_loss_requires_grad": bool(ce_sync_zero_loss.requires_grad),
                    "kl_sync_zero_loss_requires_grad": bool(kl_sync_zero_loss.requires_grad),
                    "ce_sync_zero_loss_value": float(ce_sync_zero_loss.detach().item()),
                    "kl_sync_zero_loss_value": float(kl_sync_zero_loss.detach().item()),
                }
            )
        base_loss_without_halt = (
            ce_metrics["ce_loss"]
            + float(self.config["kl_loss_weight"]) * kl_loss
            + ce_sync_zero_loss.to(dtype=ce_metrics["ce_loss"].dtype)
            + kl_sync_zero_loss.to(dtype=ce_metrics["ce_loss"].dtype)
        )
        halt_dense_combine = self._combine_base_and_halt_dense_loss(
            base_loss_without_halt=base_loss_without_halt,
            ce_loss=ce_metrics["ce_loss"],
            halt_dense_loss=halt_dense_loss,
            halt_dense_loss_weight=float(halt_dense_loss_weight),
        )
        total_loss = halt_dense_combine["total_loss"]
        answer_ce = compute_answer_ce(ce_metrics["token_ce"], batch["answer_mask"])
        halt_dense_loss_detached = halt_dense_loss.detach()
        return {
            "loss": total_loss,
            "halt_dense_loss": halt_dense_loss_detached,
            "latent_stop_loss": halt_dense_loss_detached,
            "stop_loss": halt_dense_loss_detached,
            "halt_dense_loss_weight": batch["input_ids"].new_tensor(float(halt_dense_loss_weight), dtype=torch.float32),
            "halt_dense_projected_loss": halt_dense_combine["projected_halt_dense_loss"],
            "halt_dense_gated_loss": halt_dense_combine["gated_halt_dense_loss"],
            "halt_dense_projection_alpha": halt_dense_combine["projection_alpha"],
            "halt_dense_projection_dot": halt_dense_combine["projection_dot"],
            "halt_dense_projection_ce_grad_norm_sq": halt_dense_combine["projection_ce_grad_norm_sq"],
            "halt_dense_gate_alpha": halt_dense_combine["gate_alpha"],
            "halt_dense_gate_ce_loss_ema": halt_dense_combine["gate_ce_loss_ema"],
            "halt_dense_projection_mode": halt_dense_combine["projection_mode"],
            "early_exit_rank_loss": halt_dense_loss_detached,
            "early_exit_argmax_violation_rate": halt_dense_metrics["argmax_violation_rate"].detach(),
            "early_exit_front_rank_loss": halt_dense_metrics["front_loss"].detach(),
            "early_exit_nonfront_rank_loss": halt_dense_metrics["nonfront_loss"].detach(),
            "latent_end_soft_loss": halt_dense_metrics["latent_end_soft_loss"].detach(),
            "other_end_hard_loss": halt_dense_metrics["other_end_hard_loss"].detach(),
            "latent_end_target_mean": halt_dense_metrics["latent_end_target_mean"].detach(),
            "latent_end_score_mean": halt_dense_metrics["latent_end_score_mean"].detach(),
            "latent_end_front_score_mean": halt_dense_metrics["latent_end_front_score_mean"].detach(),
            "latent_end_tail_score_mean": halt_dense_metrics["latent_end_tail_score_mean"].detach(),
            "ce_loss": ce_metrics["ce_loss"].detach(),
            "cot_ce": ce_metrics["cot_ce"].detach(),
            "non_cot_ce": ce_metrics["non_cot_ce"].detach(),
            "answer_ce": answer_ce.detach(),
            "kl_loss": kl_loss.detach(),
            "kl_positions": torch.tensor(float(kl_positions), device=batch["input_ids"].device),
            "outputs": outputs,
            "ce_metrics": ce_metrics,
            "halt_dense_metrics": halt_dense_metrics,
            "kl_stats": kl_stats or {},
            "effective_batch": batch,
        }

    def _log_memory_snapshot(
        self, tag: str, batch: Dict[str, Any] | None = None, extra: Dict[str, Any] | None = None
    ) -> None:
        if not self.enable_memory_profile or not torch.cuda.is_available():
            return
        snapshot = build_cuda_memory_snapshot(
            tag=tag,
            rank=self.accelerator.process_index,
            model=self.model,
            optimizer=self.optimizer,
            batch=batch,
            extra=extra,
        )
        append_jsonl(self.memory_log_path, snapshot)
        if self.memory_profile_log_to_console:
            self.logger.info(snapshot)

    def _batch_diagnostics(self, batch: Dict[str, Any], global_step: int, epoch_index: int) -> Dict[str, Any]:
        return build_batch_diagnostics(
            batch=batch,
            global_step=global_step,
            epoch_index=epoch_index,
            teacher_length_mismatch_count=self.teacher_length_mismatch_count,
            include_cuda_memory=bool(torch.cuda.is_available() and self.enable_forward_memory_breakdown_log),
        )

    def _log_step_diagnostics(self, tag: str, batch: Dict[str, Any], global_step: int, epoch_index: int) -> None:
        payload = self._batch_diagnostics(batch=batch, global_step=global_step, epoch_index=epoch_index)
        payload["tag"] = tag
        self.logger.info(payload)

    def _token_repr(self, token_id: int) -> str:
        return token_repr(self.tokenizer, token_id)

    def _maybe_log_alignment_debug(
        self,
        batch: Dict[str, Any],
        metrics: Dict[str, Any],
        global_step: int,
        epoch_index: int,
        tag: str,
    ) -> None:
        if not self.enable_alignment_debug_log:
            return
        if global_step % self.alignment_debug_log_every_steps != 0:
            return

        token_ce = metrics["ce_metrics"]["token_ce"].detach()
        weighted_token_ce = metrics["ce_metrics"]["weighted_token_ce"].detach()
        token_halt_dense_bce = metrics["halt_dense_metrics"]["token_halt_dense_bce"].detach()
        halt_dense_valid_mask = metrics["halt_dense_metrics"]["halt_dense_valid_mask"].detach().to(torch.bool)
        front_mask = metrics["halt_dense_metrics"]["front_mask"].detach().to(torch.bool)
        argmax_violation_mask = metrics["halt_dense_metrics"]["argmax_violation_mask"].detach().to(torch.bool)
        token_forbidden_minus_best_allowed = metrics["halt_dense_metrics"]["token_forbidden_minus_best_allowed"]
        latent_end_target = metrics["halt_dense_metrics"]["latent_end_target"].detach()
        latent_end_score = metrics["halt_dense_metrics"]["latent_end_score"].detach()
        token_latent_end_soft_loss = metrics["halt_dense_metrics"]["token_latent_end_soft_loss"].detach()
        token_other_end_hard_loss = metrics["halt_dense_metrics"]["token_other_end_hard_loss"].detach()
        distance_to_latent_end = metrics["halt_dense_metrics"]["distance_to_latent_end"].detach()
        progress_to_end = metrics["halt_dense_metrics"]["progress_to_end"].detach()
        latent_index_within_span = metrics["halt_dense_metrics"]["latent_index_within_span"].detach()
        outputs = metrics["outputs"]
        halt_dense_logits = getattr(outputs, "halt_dense_logits", None)
        halt_dense_best_allowed_logits = getattr(outputs, "halt_dense_best_allowed_logits", None)
        batch_size = int(batch["input_ids"].shape[0])
        sample_count = min(batch_size, self.alignment_debug_max_samples)

        for sample_index in range(sample_count):
            input_ids = batch["input_ids"][sample_index].detach().cpu()
            labels = batch["labels"][sample_index].detach().cpu()
            attention_mask = batch["attention_mask"][sample_index].detach().to(torch.bool).cpu()
            valid_positions = torch.nonzero(attention_mask, as_tuple=False).view(-1)
            left_pad_count = int(valid_positions[0].item()) if valid_positions.numel() > 0 else int(input_ids.numel())
            loss_pair_mask = batch["loss_pair_mask"][sample_index].detach().to(torch.bool).cpu()
            loss_source_positions = batch["loss_source_positions"][sample_index].detach().cpu()
            loss_target_positions = batch["loss_target_positions"][sample_index].detach().cpu()
            valid_pair_positions = torch.nonzero(loss_pair_mask, as_tuple=False).view(-1)
            spans = batch["spans"][sample_index]
            latent_internal_mask = batch["latent_internal_mask"][sample_index].detach().to(torch.bool).cpu()
            latent_boundary_mask = batch["latent_boundary_mask"][sample_index].detach().to(torch.bool).cpu()
            cot_mask = batch["cot_mask"][sample_index].detach().to(torch.bool).cpu()
            answer_mask = batch["answer_mask"][sample_index].detach().to(torch.bool).cpu()
            latent_pad_mask = batch["latent_pad_mask"][sample_index].detach().to(torch.bool).cpu()
            prompt_mask = batch["prompt_mask"][sample_index].detach().to(torch.bool).cpu()
            loss_weights = batch["loss_weights"][sample_index].detach().cpu()

            pad_attention_violations = int(
                (batch["attention_mask"][sample_index].detach().cpu()[~attention_mask] != 0).sum().item()
            )
            supervised_latent_internal = int(((labels != -100) & latent_internal_mask).sum().item())
            supervised_middle_pad = int(((labels != -100) & latent_pad_mask).sum().item())
            supervised_prompt = int(((labels != -100) & prompt_mask).sum().item())
            supervised_valid = int((labels != -100).sum().item())

            pair_preview: List[Dict[str, Any]] = []
            loss_pair_out_of_valid_count = 0
            loss_pair_shift_mismatch_count = 0
            for pair_tensor_index in valid_pair_positions:
                pair_idx = int(pair_tensor_index.item())
                src_pos = int(loss_source_positions[pair_idx].item())
                tgt_pos = int(loss_target_positions[pair_idx].item())
                if (
                    src_pos < 0
                    or tgt_pos < 0
                    or src_pos >= int(attention_mask.numel())
                    or tgt_pos >= int(attention_mask.numel())
                    or (not bool(attention_mask[src_pos].item()))
                    or (not bool(attention_mask[tgt_pos].item()))
                ):
                    loss_pair_out_of_valid_count += 1
                if tgt_pos <= src_pos:
                    loss_pair_shift_mismatch_count += 1
            for pair_tensor_index in valid_pair_positions[:8]:
                pair_idx = int(pair_tensor_index.item())
                src_pos = int(loss_source_positions[pair_idx].item())
                tgt_pos = int(loss_target_positions[pair_idx].item())
                pair_preview.append(
                    {
                        "src_pos": src_pos,
                        "tgt_pos": tgt_pos,
                        "src_token": self._token_repr(int(input_ids[src_pos].item())),
                        "tgt_label": self._token_repr(int(labels[tgt_pos].item()))
                        if int(labels[tgt_pos].item()) >= 0
                        else -100,
                        "tgt_weight": round(float(loss_weights[tgt_pos].item()), 4),
                        "tgt_is_latent_internal": bool(latent_internal_mask[tgt_pos].item()),
                        "tgt_is_latent_boundary": bool(latent_boundary_mask[tgt_pos].item()),
                        "tgt_is_cot": bool(cot_mask[tgt_pos].item()),
                        "tgt_is_answer": bool(answer_mask[tgt_pos].item()),
                        "token_ce": round(float(token_ce[sample_index, tgt_pos].item()), 6),
                        "weighted_token_ce": round(float(weighted_token_ce[sample_index, tgt_pos].item()), 6),
                    }
                )

            halt_dense_preview: List[Dict[str, Any]] = []
            local_halt_valid = halt_dense_valid_mask[sample_index].detach().to(torch.bool).cpu()
            halt_valid_positions = torch.nonzero(local_halt_valid, as_tuple=False).view(-1)
            halt_dense_token_ids = list(self.halt_dense_token_ids)
            for pos_tensor in halt_valid_positions[:8]:
                pos = int(pos_tensor.item())
                logits_row = []
                forbidden_minus_best_allowed = []
                if halt_dense_logits is not None:
                    logits_row = [
                        round(float(v), 6)
                        for v in halt_dense_logits[sample_index, pos].detach().float().cpu().tolist()
                    ]
                if token_forbidden_minus_best_allowed is not None:
                    forbidden_minus_best_allowed = [
                        round(float(v), 6)
                        for v in token_forbidden_minus_best_allowed[sample_index, pos].detach().float().cpu().tolist()
                    ]
                halt_dense_preview.append(
                    {
                        "pos": pos,
                        "latent_pos_index_within_span": int(latent_index_within_span[sample_index, pos].item()),
                        "distance_to_latent_end": round(float(distance_to_latent_end[sample_index, pos].item()), 4),
                        "progress_to_end": round(float(progress_to_end[sample_index, pos].item()), 6),
                        "token": self._token_repr(int(input_ids[pos].item())),
                        "forbidden_token_logits": logits_row,
                        "forbidden_token_ids": halt_dense_token_ids,
                        "best_allowed_logit": (
                            round(float(halt_dense_best_allowed_logits[sample_index, pos].detach().float().item()), 6)
                            if halt_dense_best_allowed_logits is not None
                            else None
                        ),
                        "forbidden_minus_best_allowed": forbidden_minus_best_allowed,
                        "latent_end_target": round(float(latent_end_target[sample_index, pos].item()), 6),
                        "latent_end_score": round(float(latent_end_score[sample_index, pos].item()), 6),
                        "latent_end_soft_loss": round(float(token_latent_end_soft_loss[sample_index, pos].item()), 6),
                        "other_end_hard_loss": round(float(token_other_end_hard_loss[sample_index, pos].item()), 6),
                        "rank_loss": round(float(token_halt_dense_bce[sample_index, pos].item()), 6),
                        "is_front_region": bool(front_mask[sample_index, pos].item()),
                        "is_forbidden_argmax": bool(argmax_violation_mask[sample_index, pos].item()),
                    }
                )

            payload = {
                "tag": tag,
                "global_step": int(global_step),
                "epoch_index": int(epoch_index),
                "sample_index": int(sample_index),
                "record_id": str(batch["record_ids"][sample_index]),
                "seq_len": int(input_ids.numel()),
                "valid_token_count": int(valid_positions.numel()),
                "left_pad_count": left_pad_count,
                "middle_pad_count": int(latent_pad_mask.sum().item()),
                "prompt_supervised_count": supervised_prompt,
                "latent_internal_count": int(latent_internal_mask.sum().item()),
                "latent_internal_supervised_count": supervised_latent_internal,
                "latent_boundary_supervised_count": int(((labels != -100) & latent_boundary_mask).sum().item()),
                "cot_supervised_count": int(((labels != -100) & cot_mask).sum().item()),
                "answer_supervised_count": int(((labels != -100) & answer_mask).sum().item()),
                "supervised_target_count": supervised_valid,
                "supervised_middle_pad_count": supervised_middle_pad,
                "pad_attention_violations": pad_attention_violations,
                "latent_start_pos": int(spans["latent_start"]),
                "latent_end_pos": int(spans["latent_end"]),
                "think_start_pos": int(spans["think_start"]),
                "think_end_pos": int(spans["think_end"]),
                "answer_start_pos": int(spans["answer_start"]),
                "im_end_pos": int(spans["im_end"]),
                "halt_dense_valid_positions": int(halt_valid_positions.numel()),
                "halt_dense_front_positions": int((local_halt_valid & front_mask[sample_index].cpu()).sum().item()),
                "halt_dense_argmax_violations": int((local_halt_valid & argmax_violation_mask[sample_index].cpu()).sum().item()),
                "halt_dense_preview": halt_dense_preview,
                "loss_pair_count": int(valid_pair_positions.numel()),
                "loss_pair_out_of_valid_count": int(loss_pair_out_of_valid_count),
                "loss_pair_shift_mismatch_count": int(loss_pair_shift_mismatch_count),
                "loss_pair_preview": pair_preview,
                "boundary_tokens": {
                    "latent_start": self._token_repr(int(input_ids[int(spans["latent_start"])].item())),
                    "latent_end": self._token_repr(int(input_ids[int(spans["latent_end"])].item())),
                    "think_start": self._token_repr(int(input_ids[int(spans["think_start"])].item())),
                    "think_end": self._token_repr(int(input_ids[int(spans["think_end"])].item())),
                },
            }
            if supervised_latent_internal == 0 and int(latent_internal_mask.sum().item()) > 0:
                payload["warning"] = (
                    "latent internal tokens exist but receive no direct CE supervision; "
                    "current training supervises latent boundaries with CE and latent internals with halt-dense BCE."
                )
            self.logger.info(payload)

    def _forward_memory_breakdown(
        self, batch: Dict[str, Any], metrics: Dict[str, Any], global_step: int, epoch_index: int
    ) -> Dict[str, Any]:
        return build_forward_memory_breakdown(
            model=self.model,
            optimizer=self.optimizer,
            batch=batch,
            metrics=metrics,
            global_step=global_step,
            epoch_index=epoch_index,
            teacher_length_mismatch_count=self.teacher_length_mismatch_count,
        )

    def train(self) -> TrainLoopResult:
        resume_state = self._restore_training_state()
        global_step = int(resume_state["global_step"])
        resume_epoch_index = int(resume_state["epoch_index"])
        resume_step_in_epoch = int(resume_state["next_step_in_epoch"])
        save_every = max(1, int(self.config["save_every_steps"]))
        save_latest_every = max(int(self.config.get("save_latest_every_steps", 0) or 0), 0)
        max_train_steps = int(self.config["max_train_steps"])
        num_processes = max(int(getattr(self.accelerator, "num_processes", 1)), 1)
        if max_train_steps > 0:
            total_train_steps = max_train_steps
        else:
            global_batch_size = self._effective_global_batch_size(num_processes)
            # Use global optimizer-step semantics so progress matches distributed training steps.
            steps_per_epoch = (len(self.train_dataset) + global_batch_size - 1) // global_batch_size
            total_train_steps = steps_per_epoch * max(int(self.config["num_epochs"]), 1)
        self._total_train_steps = int(total_train_steps)
        current_train_stage, _ = self._compute_train_stage(
            global_step=global_step,
            total_train_steps=total_train_steps,
        )
        self._sync_trainable_mode_for_stage(
            train_stage=current_train_stage,
            global_step=global_step,
            epoch_index=resume_epoch_index,
            step_in_epoch=max(resume_step_in_epoch - 1, 0),
            reason="train_start",
        )
        self._log_trainable_optimizer_coverage(
            tag="train_stage_optimizer_coverage",
            train_stage=current_train_stage,
            global_step=global_step,
            epoch_index=resume_epoch_index,
            step_in_epoch=max(resume_step_in_epoch - 1, 0),
        )
        final_result = TrainLoopResult(
            status="completed",
            checkpoint_dir=None,
            global_step=int(global_step),
            epoch_index=int(resume_epoch_index),
            next_step_in_epoch=int(resume_step_in_epoch),
            target_stage=int(current_train_stage),
        )
        current_epoch_index = resume_epoch_index
        current_step_in_epoch = resume_step_in_epoch
        try:
            for epoch_index in range(resume_epoch_index, int(self.config["num_epochs"])):
                current_epoch_index = int(epoch_index)
                train_loader = self._make_train_loader(
                    epoch_index=epoch_index,
                    global_step=global_step,
                    total_train_steps=total_train_steps,
                )
                train_loader = self._prepare_dataloader(train_loader, purpose="train")
                self.model.train()
                progress_bar = tqdm(
                    train_loader, disable=not self.accelerator.is_local_main_process, desc=f"train epoch {epoch_index}"
                )
                skip_until = resume_step_in_epoch if epoch_index == resume_epoch_index else 0
                for step_in_epoch, batch in enumerate(progress_bar):
                    current_step_in_epoch = int(step_in_epoch)
                    if step_in_epoch < skip_until:
                        continue
                    desired_stage, progress = self._compute_train_stage(
                        global_step=global_step,
                        total_train_steps=total_train_steps,
                    )
                    self.collator.set_stage(train_stage=desired_stage, progress=progress)
                    if desired_stage != current_train_stage:
                        self.logger.info(
                            {
                                "tag": "train_stage_switch_begin",
                                "from_stage": int(current_train_stage),
                                "to_stage": int(desired_stage),
                                "global_step": int(global_step),
                                "total_train_steps": int(total_train_steps),
                                "progress": float(progress),
                                "stage2_start_fraction": float(self.config["stage2_start_fraction"]),
                                "epoch_index": int(epoch_index),
                                "step_in_epoch": int(step_in_epoch),
                            }
                        )
                        current_train_stage = int(desired_stage)
                        if self.distributed_backend == "deepspeed":
                            final_result = self._build_stage_transition_checkpoint(
                                global_step=global_step,
                                epoch_index=epoch_index,
                                step_in_epoch=step_in_epoch,
                                target_stage=current_train_stage,
                            )
                            return final_result
                        self._sync_trainable_mode_for_stage(
                            train_stage=current_train_stage,
                            global_step=global_step,
                            epoch_index=epoch_index,
                            step_in_epoch=step_in_epoch,
                            reason="stage_switch",
                        )
                        self._log_trainable_optimizer_coverage(
                            tag="train_stage_optimizer_coverage",
                            train_stage=current_train_stage,
                            global_step=global_step,
                            epoch_index=epoch_index,
                            step_in_epoch=step_in_epoch,
                        )
                    profile_this_step = (
                        self.enable_memory_profile and not self._memory_profile_done and torch.cuda.is_available()
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    self._log_step_diagnostics(
                        tag="train_step_start", batch=batch, global_step=global_step, epoch_index=epoch_index
                    )
                    if profile_this_step:
                        reset_cuda_peak_memory()
                        self._log_memory_snapshot(
                            tag="step0_batch_ready",
                            batch=batch,
                            extra={
                                "global_step": global_step,
                                "seq_len": int(batch["input_ids"].shape[1]),
                                "micro_batch_size": int(batch["input_ids"].shape[0]),
                            },
                        )
                    try:
                        metrics = self._forward_loss(
                            batch,
                            global_step=global_step,
                            epoch_index=epoch_index,
                            train_stage=desired_stage,
                            train_progress=progress,
                        )  # NOTE
                        self._maybe_log_alignment_debug(
                            batch=batch,
                            metrics=metrics,
                            global_step=global_step,
                            epoch_index=epoch_index,
                            tag="alignment_debug_train",
                        )
                    except Exception as exc:
                        error_payload = self._batch_diagnostics(
                            batch=batch, global_step=global_step, epoch_index=epoch_index
                        )
                        error_payload["tag"] = "train_step_exception"
                        error_payload["error_type"] = type(exc).__name__
                        error_payload["error_message"] = str(exc)
                        self.logger.error(error_payload)
                        raise
                    if self.enable_forward_memory_breakdown_log:
                        self.logger.info(
                            self._forward_memory_breakdown(
                                batch=batch,
                                metrics=metrics,
                                global_step=global_step,
                                epoch_index=epoch_index,
                            )
                        )
                    if profile_this_step:
                        self._log_memory_snapshot(
                            tag="step0_after_forward",
                            batch=batch,
                            extra={
                                "global_step": global_step,
                                "loss": float(metrics["loss"].detach().item()),
                            },
                        )
                    loss_tensor = metrics["loss"]
                    step_trace_enabled = True
                    if step_trace_enabled:
                        self.logger.info(
                            {
                                "tag": "train_backward_inputs",
                                "global_step": int(global_step),
                                "epoch_index": int(epoch_index),
                                "step_in_epoch": int(step_in_epoch),
                                "loss": float(loss_tensor.detach().item()),
                                "loss_requires_grad": bool(loss_tensor.requires_grad),
                                "ce_loss": float(metrics["ce_loss"].detach().item()) if "ce_loss" in metrics else 0.0,
                                "kl_loss": float(metrics["kl_loss"].detach().item()) if "kl_loss" in metrics else 0.0,
                                "stop_loss": float(metrics["stop_loss"].detach().item()) if "stop_loss" in metrics else 0.0,
                                "halt_dense_loss_weight": float(metrics["halt_dense_loss_weight"].detach().item())
                                if "halt_dense_loss_weight" in metrics
                                else 0.0,
                            }
                        )
                        self.logger.info(
                            {
                                "tag": "train_step_stage",
                                "phase": "backward_start",
                                "global_step": int(global_step),
                                "epoch_index": int(epoch_index),
                                "step_in_epoch": int(step_in_epoch),
                            }
                        )
                    # NOTE: backward
                    self.accelerator.backward(metrics["loss"])
                    if step_trace_enabled:
                        self.logger.info(
                            {
                                "tag": "train_step_stage",
                                "phase": "backward_done",
                                "global_step": int(global_step),
                                "epoch_index": int(epoch_index),
                                "step_in_epoch": int(step_in_epoch),
                            }
                        )
                    if profile_this_step:
                        self._log_memory_snapshot(
                            tag="step0_after_backward",
                            batch=batch,
                            extra={"global_step": global_step},
                        )
                    if float(self.config["max_grad_norm"]) > 0:
                        if step_trace_enabled:
                            self.logger.info(
                                {
                                    "tag": "train_step_stage",
                                    "phase": "clip_grad_start",
                                    "global_step": int(global_step),
                                    "epoch_index": int(epoch_index),
                                    "step_in_epoch": int(step_in_epoch),
                                }
                            )
                        self.accelerator.clip_grad_norm_(self.model.parameters(), float(self.config["max_grad_norm"]))
                        if step_trace_enabled:
                            self.logger.info(
                                {
                                    "tag": "train_step_stage",
                                    "phase": "clip_grad_done",
                                    "global_step": int(global_step),
                                    "epoch_index": int(epoch_index),
                                    "step_in_epoch": int(step_in_epoch),
                                }
                            )
                    if step_trace_enabled:
                        self.logger.info(
                            {
                                "tag": "train_step_stage",
                                "phase": "optimizer_step_start",
                                "global_step": int(global_step),
                                "epoch_index": int(epoch_index),
                                "step_in_epoch": int(step_in_epoch),
                            }
                        )
                    # NOTE: optimizer step
                    self.optimizer.step()
                    restored_frozen_rows = False
                    if self.base_token_row_freeze_controller is not None:
                        restored_frozen_rows = bool(self.base_token_row_freeze_controller.restore_frozen_rows())
                    if step_trace_enabled:
                        self.logger.info(
                            {
                                "tag": "train_step_stage",
                                "phase": "optimizer_step_done",
                                "global_step": int(global_step),
                                "epoch_index": int(epoch_index),
                                "step_in_epoch": int(step_in_epoch),
                                "restored_frozen_rows": bool(restored_frozen_rows),
                            }
                        )
                        self.logger.info(
                            {
                                "tag": "train_step_stage",
                                "phase": "scheduler_step_start",
                                "global_step": int(global_step),
                                "epoch_index": int(epoch_index),
                                "step_in_epoch": int(step_in_epoch),
                            }
                        )
                    # NOTE: scheduler step
                    self.scheduler.step()
                    if step_trace_enabled:
                        self.logger.info(
                            {
                                "tag": "train_step_stage",
                                "phase": "scheduler_step_done",
                                "global_step": int(global_step),
                                "epoch_index": int(epoch_index),
                                "step_in_epoch": int(step_in_epoch),
                            }
                        )
                    if profile_this_step:
                        self._log_memory_snapshot(
                            tag="step0_after_optimizer_step",
                            batch=batch,
                            extra={"global_step": global_step},
                        )
                        self._memory_profile_done = True

                    global_step += 1
                    self._log_train_scalars(metrics, global_step=global_step, epoch_index=epoch_index)
                    progress_bar.set_postfix(
                        stage=f"{int(desired_stage)}",
                        loss=f"{metrics['loss'].item():.4f}",
                        stop=f"{metrics['latent_stop_loss'].item():.4f}",
                        ce=f"{metrics['ce_loss'].item():.4f}",
                        kl=f"{metrics['kl_loss'].item():.4f}",
                    )
                    eval_due = global_step % int(self.config["eval_every_steps"]) == 0
                    periodic_save_due = global_step % save_every == 0
                    latest_save_due = save_latest_every > 0 and global_step % save_latest_every == 0
                    numbered_saved_this_step = False
                    if periodic_save_due:
                        self.save_checkpoint(
                            global_step=global_step,
                            epoch_index=epoch_index,
                            step_in_epoch=step_in_epoch + 1,
                            reason="periodic",
                        )
                        numbered_saved_this_step = True
                    if latest_save_due:
                        self.save_latest_checkpoint(
                            global_step=global_step,
                            epoch_index=epoch_index,
                            step_in_epoch=step_in_epoch + 1,
                            reason="latest_periodic",
                        )
                    if eval_due:
                        self.evaluate(global_step=global_step)
                    if max_train_steps > 0 and global_step >= max_train_steps:
                        if not numbered_saved_this_step:
                            self.save_checkpoint(
                                global_step=global_step,
                                epoch_index=epoch_index,
                                step_in_epoch=step_in_epoch + 1,
                                reason="max_train_steps",
                            )
                        final_result = TrainLoopResult(
                            status="completed",
                            checkpoint_dir=None,
                            global_step=int(global_step),
                            epoch_index=int(epoch_index),
                            next_step_in_epoch=int(step_in_epoch + 1),
                            target_stage=int(current_train_stage),
                        )
                        return final_result
                resume_step_in_epoch = 0
                is_last_epoch = int(epoch_index) + 1 >= int(self.config["num_epochs"])
                self.save_checkpoint(
                    global_step=global_step,
                    epoch_index=int(self.config["num_epochs"]) if is_last_epoch else epoch_index,
                    step_in_epoch=0,
                    reason="final" if is_last_epoch else "epoch_end_pre_eval",
                )
                if save_latest_every > 0:
                    self.save_latest_checkpoint(
                        global_step=global_step,
                        epoch_index=int(self.config["num_epochs"]) if is_last_epoch else epoch_index,
                        step_in_epoch=0,
                        reason="latest_final" if is_last_epoch else "latest_epoch_end",
                    )
                self.evaluate(global_step=global_step)
            final_result = TrainLoopResult(
                status="completed",
                checkpoint_dir=None,
                global_step=int(global_step),
                epoch_index=int(self.config["num_epochs"]),
                next_step_in_epoch=0,
                target_stage=int(current_train_stage),
            )
        except Exception as exc:
            self._try_save_exception_checkpoint(
                global_step=global_step,
                epoch_index=current_epoch_index,
                step_in_epoch=current_step_in_epoch,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise
        finally:
            self._close_writer()
        return final_result

    def _reduce_scalar(self, value: float, device: torch.device) -> float:
        tensor = torch.tensor([float(value)], device=device, dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.item())

    def _sync_best_checkpoint_decision(self, improved: bool, device: torch.device) -> bool:
        payload = torch.tensor([1 if improved else 0], device=device, dtype=torch.long)
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(payload, src=0)
        return bool(int(payload.item()) > 0)

    @staticmethod
    def _sample_eval_generation_pool(
        pool: List[Dict[str, Any]],
        *,
        sample_count: int,
        seed: int,
    ) -> List[Dict[str, Any]]:
        if sample_count <= 0 or not pool:
            return []
        rng = random.Random(int(seed))
        if len(pool) >= sample_count:
            indices = rng.sample(range(len(pool)), k=int(sample_count))
            return [dict(pool[index]) for index in indices]
        selected = [dict(item) for item in pool]
        while len(selected) < int(sample_count):
            selected.append(dict(rng.choice(pool)))
        return selected

    def _update_best_answer_ce_link(
        self,
        *,
        global_step: int,
        val_answer_ce: float,
        per_benchmark_answer_ce: Dict[str, float],
    ) -> None:
        if not self.accelerator.is_main_process:
            return
        output_root = Path(self.config["output_dir"])
        target_dir = output_root / f"step_{int(global_step):07d}"
        link_path = output_root / "best_answer_ce"
        tmp_link_path = output_root / "best_answer_ce.tmp"
        if tmp_link_path.exists() or tmp_link_path.is_symlink():
            tmp_link_path.unlink()
        tmp_link_path.symlink_to(target_dir.name)
        tmp_link_path.replace(link_path)
        payload = {
            "step": int(global_step),
            "val_answer_ce": float(val_answer_ce),
            "per_benchmark_answer_ce": {str(key): float(value) for key, value in per_benchmark_answer_ce.items()},
        }
        self._best_answer_ce_meta_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @torch.no_grad()
    def evaluate(self, global_step: int) -> None:
        val_loader = self._make_val_loader()
        val_loader = self._prepare_dataloader(val_loader, purpose="eval")
        self.model.eval()
        token_constants = self.val_dataset.token_constants
        device = next(self.model.parameters()).device
        raw_model = self.accelerator.unwrap_model(self.model)
        rank = int(dist.get_rank()) if (dist.is_available() and dist.is_initialized()) else 0
        world_size = int(dist.get_world_size()) if (dist.is_available() and dist.is_initialized()) else 1

        per_benchmark_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "loss_sum": 0.0,
                "ce_sum": 0.0,
                "kl_sum": 0.0,
                "stop_sum": 0.0,
                "sample_count": 0.0,
                "answer_numerator": 0.0,
                "answer_denominator": 0.0,
            }
        )
        local_generation_pool: List[Dict[str, Any]] = []
        total_batches = 0
        for batch in val_loader:
            total_batches += 1
            metrics = self._forward_loss(
                batch,
                global_step=global_step,
                epoch_index=-1,
                train_stage=2,
                train_progress=1.0,
            )
            effective_batch = metrics["effective_batch"]
            batch_size = int(effective_batch["input_ids"].size(0))
            answer_token_ce = metrics["ce_metrics"]["token_ce"].detach()
            answer_mask = effective_batch["answer_mask"].detach().to(torch.bool)
            if tuple(answer_token_ce.shape) != tuple(answer_mask.shape):
                raise RuntimeError(
                    "evaluate answer_mask/token_ce shape mismatch after forward_loss sync: "
                    f"token_ce_shape={list(answer_token_ce.shape)} "
                    f"answer_mask_shape={list(answer_mask.shape)}"
                )
            benchmark_names = list(effective_batch.get("benchmark_names", ["val"] * batch_size))
            for sample_index, benchmark_name in enumerate(benchmark_names):
                bench = str(benchmark_name)
                stats = per_benchmark_stats[bench]
                stats["loss_sum"] += float(metrics["loss"].detach().item())
                stats["ce_sum"] += float(metrics["ce_loss"].detach().item())
                stats["kl_sum"] += float(metrics["kl_loss"].detach().item())
                stats["stop_sum"] += float(metrics["stop_loss"].detach().item())
                stats["sample_count"] += 1.0
                local_answer_mask = answer_mask[sample_index]
                answer_count = float(local_answer_mask.to(torch.float32).sum().item())
                if answer_count > 0.0:
                    stats["answer_numerator"] += float(
                        answer_token_ce[sample_index][local_answer_mask].sum().item()
                    )
                    stats["answer_denominator"] += float(answer_count)
                sample_spans = effective_batch["spans"][sample_index]
                local_generation_pool.append(
                    {
                        "benchmark_name": bench,
                        "record_id": str(effective_batch["record_ids"][sample_index]),
                        "prompt_only_ids": list(effective_batch["prompt_only_ids"][sample_index]),
                        "assistant_answer": str(effective_batch["assistant_answers"][sample_index]),
                        "spans": dict(sample_spans),
                        "original_latent_tokens": max(
                            int(sample_spans["latent_end"]) - int(sample_spans["latent_start"]) - 1,
                            0,
                        ),
                        "original_cot_tokens": max(
                            int(sample_spans["think_end"]) - int(sample_spans["think_start"]) - 1,
                            0,
                        ),
                        "original_total_tokens": int(effective_batch["attention_mask"][sample_index].sum().item()),
                    }
                )

        if total_batches <= 0:
            if self.accelerator.is_main_process:
                self.logger.warning({"tag": "eval_skipped", "reason": "empty_val_loader"})
            self.model.train()
            return

        reduced_stats: Dict[str, Dict[str, float]] = {}
        for benchmark_name, local_stats in per_benchmark_stats.items():
            reduced_stats[benchmark_name] = {
                key: self._reduce_scalar(value=float(value), device=device)
                for key, value in local_stats.items()
            }

        generation_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "count": 0.0,
                "generated_latent_tokens": 0.0,
                "generated_cot_tokens": 0.0,
                "generated_total_tokens": 0.0,
                "generated_stopped_normally": 0.0,
            }
        )
        sample_logs: List[Dict[str, Any]] = []
        eval_max_new_tokens = int(self._resolve_eval_max_new_tokens(global_step=global_step))
        local_pool_size_tensor = torch.tensor([int(len(local_generation_pool))], device=device, dtype=torch.long)
        min_pool_size_tensor = local_pool_size_tensor.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(min_pool_size_tensor, op=dist.ReduceOp.MIN)
        min_pool_size = int(min_pool_size_tensor.item())
        generation_rounds = int(self.config.get("eval_batch_size", 1) or 1)
        generation_enabled = min_pool_size > 0 and generation_rounds > 0
        if generation_enabled:
            sampled_generation_pool = self._sample_eval_generation_pool(
                local_generation_pool,
                sample_count=int(generation_rounds),
                seed=int(self.config.get("seed", 42)) + int(global_step) + int(rank),
            )
            prompt_ids_batch = [list(sample["prompt_only_ids"]) for sample in sampled_generation_pool]
            generated_batch = raw_model._generate_with_latent_batched(
                prompt_ids_batch=prompt_ids_batch,
                token_constants=token_constants,
                max_new_tokens=eval_max_new_tokens,
                latent_max_steps=int(self.config["latent_max_steps"]),
                distributed_lockstep=bool(world_size > 1),
                do_sample=bool(self.config.get("eval_do_sample", False)),
                temperature=float(self.config.get("eval_temperature", 1.0)),
                top_p=float(self.config.get("eval_top_p", 1.0)),
                top_k=int(self.config.get("eval_top_k", 0)),
            )
            for round_idx, sample in enumerate(sampled_generation_pool):
                prompt_token_ids = list(sample["prompt_only_ids"])
                generated_token_ids = self._token_ids_to_list(generated_batch.token_ids_batch[round_idx])
                decoded = self.tokenizer.decode(generated_token_ids, skip_special_tokens=False)
                generated_counts = self._count_generated_token_types(generated_token_ids, token_constants)
                malformed_flag = float(
                    ("</latent_think>" not in decoded) or ("</think>" not in decoded) or ("<|im_end|>" not in decoded)
                )
                local_generation = {
                    "rank": int(rank),
                    "round_idx": int(round_idx),
                    "benchmark_name": str(sample["benchmark_name"]),
                    "record_id": str(sample["record_id"]),
                    "prompt_text": self.tokenizer.decode(prompt_token_ids, skip_special_tokens=False),
                    "ground_truth": str(sample["assistant_answer"]),
                    "generated_text": str(decoded),
                    "prompt_token_ids": list(prompt_token_ids),
                    "generated_token_ids": list(generated_token_ids),
                    "original_latent_tokens": float(sample["original_latent_tokens"]),
                    "original_cot_tokens": float(sample["original_cot_tokens"]),
                    "original_total_tokens": float(sample["original_total_tokens"]),
                    "generated_latent_tokens": float(generated_counts["latent_tokens"]),
                    "generated_cot_tokens": float(generated_counts["cot_tokens"]),
                    "generated_total_tokens": float(generated_counts["total_tokens"]),
                    "generated_stopped_normally": float(generated_batch.stopped_normally_batch[round_idx]),
                    "avg_latent_steps": float(generated_batch.latent_steps_batch[round_idx]),
                    "avg_cot_tokens": float(generated_batch.cot_tokens_batch[round_idx]),
                    "latent_rollout_steps": float(generated_batch.latent_rollout_steps),
                    "malformed_rate": float(malformed_flag),
                    "sampled_from_loss_pool": True,
                }
                gathered_generation = self._all_gather_objects(local_generation)
                if self.accelerator.is_main_process:
                    for detail in sorted(
                        gathered_generation,
                        key=lambda item: (int(item.get("round_idx", 0)), int(item.get("rank", 0))),
                    ):
                        bench = str(detail.get("benchmark_name", "val"))
                        stats = generation_stats[bench]
                        stats["count"] += 1.0
                        stats["generated_latent_tokens"] += float(detail.get("generated_latent_tokens", 0.0))
                        stats["generated_cot_tokens"] += float(detail.get("generated_cot_tokens", 0.0))
                        stats["generated_total_tokens"] += float(detail.get("generated_total_tokens", 0.0))
                        stats["generated_stopped_normally"] += float(detail.get("generated_stopped_normally", 0.0))
                        if len(sample_logs) < 6:
                            sample_logs.append(dict(detail))

        metrics_out: Dict[str, float] = {"step": float(global_step)}
        global_answer_numerator = 0.0
        global_answer_denominator = 0.0
        per_benchmark_answer_ce: Dict[str, float] = {}
        for benchmark_name, stats in sorted(reduced_stats.items()):
            sample_count = max(float(stats["sample_count"]), 1.0)
            answer_ce = safe_div(float(stats["answer_numerator"]), float(stats["answer_denominator"]))
            per_benchmark_answer_ce[str(benchmark_name)] = float(answer_ce)
            global_answer_numerator += float(stats["answer_numerator"])
            global_answer_denominator += float(stats["answer_denominator"])
            prefix = f"val/{benchmark_name}"
            metrics_out[f"{prefix}/loss"] = safe_div(float(stats["loss_sum"]), sample_count)
            metrics_out[f"{prefix}/ce_loss"] = safe_div(float(stats["ce_sum"]), sample_count)
            metrics_out[f"{prefix}/kl_loss"] = safe_div(float(stats["kl_sum"]), sample_count)
            metrics_out[f"{prefix}/stop_loss"] = safe_div(float(stats["stop_sum"]), sample_count)
            metrics_out[f"{prefix}/answer_ce"] = float(answer_ce)
            if self.accelerator.is_main_process:
                gen = generation_stats.get(str(benchmark_name), {})
                gen_count = max(float(gen.get("count", 0.0)), 1.0)
                metrics_out[f"{prefix}/gen_latent_tokens"] = safe_div(
                    float(gen.get("generated_latent_tokens", 0.0)), gen_count
                )
                metrics_out[f"{prefix}/gen_cot_tokens"] = safe_div(
                    float(gen.get("generated_cot_tokens", 0.0)), gen_count
                )
                metrics_out[f"{prefix}/gen_total_tokens"] = safe_div(
                    float(gen.get("generated_total_tokens", 0.0)), gen_count
                )
                metrics_out[f"{prefix}/gen_stopped_normally_rate"] = safe_div(
                    float(gen.get("generated_stopped_normally", 0.0)), gen_count
                )

        metrics_out["val/answer_ce"] = safe_div(global_answer_numerator, global_answer_denominator)
        if reduced_stats:
            metrics_out["val/loss"] = safe_div(
                sum(float(stats["loss_sum"]) for stats in reduced_stats.values()),
                max(sum(float(stats["sample_count"]) for stats in reduced_stats.values()), 1.0),
            )
            metrics_out["val/ce_loss"] = safe_div(
                sum(float(stats["ce_sum"]) for stats in reduced_stats.values()),
                max(sum(float(stats["sample_count"]) for stats in reduced_stats.values()), 1.0),
            )
            metrics_out["val/kl_loss"] = safe_div(
                sum(float(stats["kl_sum"]) for stats in reduced_stats.values()),
                max(sum(float(stats["sample_count"]) for stats in reduced_stats.values()), 1.0),
            )
            metrics_out["val/stop_loss"] = safe_div(
                sum(float(stats["stop_sum"]) for stats in reduced_stats.values()),
                max(sum(float(stats["sample_count"]) for stats in reduced_stats.values()), 1.0),
            )

        improved_local = False
        if self.accelerator.is_main_process:
            current_answer_ce = float(metrics_out["val/answer_ce"])
            improved_local = current_answer_ce < float(self._best_answer_ce)
            if improved_local:
                self._best_answer_ce = float(current_answer_ce)
                self._best_answer_ce_step = int(global_step)
        improved = self._sync_best_checkpoint_decision(improved_local, device=device)
        if improved:
            self.save_checkpoint(
                global_step=global_step,
                epoch_index=-1,
                step_in_epoch=0,
                reason="best_answer_ce",
            )
            self._update_best_answer_ce_link(
                global_step=global_step,
                val_answer_ce=float(metrics_out["val/answer_ce"]),
                per_benchmark_answer_ce=per_benchmark_answer_ce,
            )

        if self.accelerator.is_main_process:
            for sample_index, sample_log in enumerate(sample_logs):
                self.logger.info({"tag": "val_generation_trace", "step": int(global_step), **sample_log})
                generated_token_ids = list(sample_log.get("generated_token_ids", []))
                generated_text = str(sample_log.get("generated_text", ""))
                if generated_token_ids and generated_text:
                    self._log_eval_sample_text_bundle(
                        global_step=int(global_step),
                        batch_index=0,
                        sample_index=int(sample_index),
                        record_id=str(sample_log.get("record_id", f"eval_sample_{sample_index}")),
                        generated_latent_tokens=int(sample_log.get("generated_latent_tokens", 0)),
                        generated_cot_tokens=int(sample_log.get("generated_cot_tokens", 0)),
                        original_latent_tokens=int(sample_log.get("original_latent_tokens", 0)),
                        original_cot_tokens=int(sample_log.get("original_cot_tokens", 0)),
                        prompt_token_ids=list(sample_log.get("prompt_token_ids", [])),
                        prompt_text=str(sample_log.get("prompt_text", "")),
                        ground_truth=str(sample_log.get("ground_truth", "")),
                        generated_token_ids=generated_token_ids,
                        generated_text=generated_text,
                    )
            self.logger.info(metrics_out)
            self._log_eval_scalars(metrics_out, global_step=global_step)
        self.model.train()

    def save_checkpoint(
        self,
        global_step: int,
        epoch_index: int = 0,
        step_in_epoch: int = 0,
        reason: str = "periodic",
        train_stage_override: int | None = None,
        trainable_mode_override: str | None = None,
    ) -> None:
        output_dir = Path(self.config["output_dir"]) / f"step_{global_step:07d}"
        self._save_checkpoint_to_dir(
            output_dir=output_dir,
            global_step=global_step,
            epoch_index=epoch_index,
            step_in_epoch=step_in_epoch,
            reason=reason,
            train_stage_override=train_stage_override,
            trainable_mode_override=trainable_mode_override,
            replace_existing=False,
        )

    def save_latest_checkpoint(
        self,
        global_step: int,
        epoch_index: int = 0,
        step_in_epoch: int = 0,
        reason: str = "latest",
        train_stage_override: int | None = None,
        trainable_mode_override: str | None = None,
    ) -> None:
        latest_name = str(self.config.get("latest_checkpoint_name", "latest") or "latest").strip() or "latest"
        output_dir = Path(self.config["output_dir"]) / latest_name
        self._save_checkpoint_to_dir(
            output_dir=output_dir,
            global_step=global_step,
            epoch_index=epoch_index,
            step_in_epoch=step_in_epoch,
            reason=reason,
            train_stage_override=train_stage_override,
            trainable_mode_override=trainable_mode_override,
            replace_existing=True,
        )

    def _save_checkpoint_to_dir(
        self,
        output_dir: Path,
        global_step: int,
        epoch_index: int = 0,
        step_in_epoch: int = 0,
        reason: str = "periodic",
        train_stage_override: int | None = None,
        trainable_mode_override: str | None = None,
        replace_existing: bool = False,
    ) -> None:
        staging_dir = Path(f"{str(output_dir)}.staging")
        backup_dir = Path(f"{str(output_dir)}.bak")
        self.accelerator.wait_for_everyone()
        unwrapped = self.accelerator.unwrap_model(self.model)

        # Preflight must be rank-consistent before entering any distributed save collectives.
        if output_dir.exists():
            if not bool(replace_existing) and self._is_checkpoint_complete(output_dir):
                if self.accelerator.is_main_process:
                    self.logger.warning(
                        {
                            "tag": "checkpoint_already_exists",
                            "checkpoint_dir": str(output_dir),
                            "step": int(global_step),
                            "reason": reason,
                            "action": "skip_save",
                        }
                    )
                self.accelerator.wait_for_everyone()
                return
            if not bool(replace_existing):
                raise FileExistsError(f"Checkpoint directory exists but is incomplete: {output_dir}")

        if self.accelerator.is_main_process:
            # Clean stale temporary directories from prior interrupted saves for the same step.
            stale_prefix = f"{output_dir.name}.tmp_"
            for stale_path in output_dir.parent.glob(f"{stale_prefix}*"):
                if stale_path.is_dir():
                    shutil.rmtree(stale_path, ignore_errors=True)
            existing_complete = bool(output_dir.exists() and self._is_checkpoint_complete(output_dir))
            if bool(replace_existing) and output_dir.exists() and not existing_complete:
                self.logger.warning(
                    {
                        "tag": "checkpoint_replace_incomplete_existing",
                        "checkpoint_dir": str(output_dir),
                        "step": int(global_step),
                        "reason": reason,
                        "action": "replace_existing",
                    }
                )
            if backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            staging_dir.mkdir(parents=True, exist_ok=False)
        self.accelerator.wait_for_everyone()

        def _abort_failed_save(stage: str, exc: Exception | None = None) -> None:
            if self.accelerator.is_main_process and staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            self.accelerator.wait_for_everyone()
            payload = {
                "tag": "checkpoint_save_failed",
                "step": int(global_step),
                "epoch_index": int(epoch_index),
                "next_step_in_epoch": int(step_in_epoch),
                "reason": reason,
                "stage": str(stage),
                "staging_dir": str(staging_dir),
            }
            if exc is not None:
                payload["error_type"] = type(exc).__name__
                payload["error_message"] = str(exc)
            self.logger.error(payload)
            if exc is not None:
                raise RuntimeError(f"Checkpoint save failed at stage={stage}") from exc
            raise RuntimeError(f"Checkpoint save failed at stage={stage}")

        try:
            # Execute on all ranks to avoid ZeRO/FSDP collective mismatch with barriers.
            state_dict = self.accelerator.get_state_dict(self.model)
        except Exception as exc:
            _abort_failed_save(stage="collect_state_dict", exc=exc)

        inferred_train_stage = (
            int(train_stage_override)
            if train_stage_override is not None
            else self._compute_train_stage(
                global_step=int(global_step),
                total_train_steps=max(int(self._total_train_steps), 1),
            )[0]
        )
        current_trainable_mode = str(trainable_mode_override or self._current_trainable_mode or self.stage1_trainable_mode)

        main_write_exc: Exception | None = None
        if self.accelerator.is_main_process:
            try:
                unwrapped.save_pretrained(staging_dir, state_dict=state_dict, safe_serialization=True)
                self.tokenizer.save_pretrained(staging_dir)
                self._write_checkpoint_training_state(
                    output_dir=staging_dir,
                    global_step=global_step,
                    epoch_index=epoch_index,
                    next_step_in_epoch=step_in_epoch,
                    reason=reason,
                    train_stage=int(inferred_train_stage),
                    trainable_mode=str(current_trainable_mode),
                )
            except Exception as exc:
                main_write_exc = exc
        if not self._sync_checkpoint_save_success(main_write_exc is None):
            _abort_failed_save(stage="write_model_and_metadata", exc=main_write_exc)

        finalize_exc: Exception | None = None
        if self.accelerator.is_main_process:
            try:
                self._write_checkpoint_complete_marker(
                    output_dir=staging_dir,
                    global_step=global_step,
                    epoch_index=epoch_index,
                    next_step_in_epoch=step_in_epoch,
                    reason=reason,
                    train_stage=int(inferred_train_stage),
                    trainable_mode=str(current_trainable_mode),
                )
                if bool(replace_existing) and output_dir.exists():
                    output_dir.rename(backup_dir)
                staging_dir.rename(output_dir)
                if backup_dir.exists():
                    shutil.rmtree(backup_dir, ignore_errors=True)
            except Exception as exc:
                if backup_dir.exists() and not output_dir.exists():
                    backup_dir.rename(output_dir)
                finalize_exc = exc
        if not self._sync_checkpoint_save_success(finalize_exc is None):
            _abort_failed_save(stage="finalize_checkpoint", exc=finalize_exc)

        if self.accelerator.is_main_process:
            self.logger.info(
                {
                    "checkpoint_saved": str(output_dir),
                    "step": int(global_step),
                    "epoch_index": int(epoch_index),
                    "next_step_in_epoch": int(step_in_epoch),
                    "reason": reason,
                    "distributed_backend": self.distributed_backend,
                    "save_training_state": bool(self.save_training_state),
                }
            )
        self.accelerator.wait_for_everyone()
