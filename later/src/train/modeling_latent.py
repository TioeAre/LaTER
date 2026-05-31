from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import warnings

import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


LEGACY_LATENT_PROJECTOR_KEY_MAPPING = {
    r"^latent_projector\.net\.2\.weight$": r"latent_projector.net.3.weight",
    r"^latent_projector\.net\.2\.bias$": r"latent_projector.net.3.bias",
}


class LatentProjector(nn.Module):
    def __init__(self, hidden_size: int, embed_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embed_size),
        )
        self._compat_remap_applied = False
        self._compat_remap_messages: List[str] = []
        self.register_load_state_dict_pre_hook(self._remap_legacy_state_dict)

    def _remap_legacy_state_dict(
        self,
        module: nn.Module,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        del module, local_metadata, strict, missing_keys, unexpected_keys, error_msgs

        legacy_pairs = (
            (f"{prefix}net.2.weight", f"{prefix}net.3.weight"),
            (f"{prefix}net.2.bias", f"{prefix}net.3.bias"),
        )
        has_current_layout = any(current_key in state_dict for _, current_key in legacy_pairs)
        has_legacy_layout = any(legacy_key in state_dict for legacy_key, _ in legacy_pairs)
        if has_current_layout or not has_legacy_layout:
            return

        remapped_keys: List[str] = []
        for legacy_key, current_key in legacy_pairs:
            if legacy_key in state_dict:
                state_dict[current_key] = state_dict.pop(legacy_key)
                remapped_keys.append(f"{legacy_key} -> {current_key}")

        if remapped_keys:
            self._compat_remap_applied = True
            message = (
                "Detected legacy latent_projector checkpoint layout without dropout layer indices; "
                "remapped weights for compatibility: "
                + ", ".join(remapped_keys)
            )
            self._compat_remap_messages.append(message)
            warnings.warn(message)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class LatentQwenConfig(Qwen3Config):
    model_type = "latent_qwen3"

    def __init__(
        self,
        latent_projector_hidden_size: Optional[int] = None,
        latent_projector_dropout: float = 0.1,
        discrete_chunk_size: int = 0,
        latent_bptt_window: int = 0,
        kl_temperature: float = 1.0,
        supervised_logits_chunk_size: int = 0,
        enable_forward_memory_breakdown_log: bool = False,
        enable_hybrid_cache_debug_log: bool = False,
        enable_discrete_deadlock_probe_log: bool = False,
        discrete_deadlock_probe_chunk_idx: int = -1,
        discrete_deadlock_probe_layers: Optional[List[int]] = None,
        discrete_deadlock_probe_cuda_sync: bool = False,
        enable_discrete_safe_attention: bool = True,
        discrete_attention_impl: str = "eager",
        global_stateless_forward: bool = True,
        latent_embed_builder_no_grad: bool = True,
        legacy_staged_forward_fallback: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.latent_projector_hidden_size = latent_projector_hidden_size or self.hidden_size
        self.latent_projector_dropout = float(latent_projector_dropout)
        self.train_with_latent_internal_recurrence = True
        self.train_rollout_use_cache = True
        self.train_forward_use_pastkv = True
        self.discrete_chunk_size = int(discrete_chunk_size)
        self.latent_bptt_window = int(latent_bptt_window)
        self.kl_temperature = float(kl_temperature)
        self.supervised_logits_chunk_size = int(supervised_logits_chunk_size)
        self.enable_forward_memory_breakdown_log = bool(enable_forward_memory_breakdown_log)
        self.enable_hybrid_cache_debug_log = bool(enable_hybrid_cache_debug_log)
        self.enable_discrete_deadlock_probe_log = bool(enable_discrete_deadlock_probe_log)
        self.discrete_deadlock_probe_chunk_idx = int(discrete_deadlock_probe_chunk_idx)
        self.discrete_deadlock_probe_layers = list(discrete_deadlock_probe_layers or [])
        self.discrete_deadlock_probe_cuda_sync = bool(discrete_deadlock_probe_cuda_sync)
        self.enable_discrete_safe_attention = bool(enable_discrete_safe_attention)
        self.discrete_attention_impl = str(discrete_attention_impl)
        self.global_stateless_forward = bool(global_stateless_forward)
        self.latent_embed_builder_no_grad = bool(latent_embed_builder_no_grad)
        self.legacy_staged_forward_fallback = bool(legacy_staged_forward_fallback)
        self.architectures = ["LatentQwenForCausalLM"]


@dataclass
class LatentGenerationOutput:
    token_ids: List[int]
    latent_steps: int
    cot_tokens: int
    stopped_normally: bool
    latent_rollout_steps: int = 0


@dataclass
class LatentGenerationBatchOutput:
    token_ids_batch: List[List[int]]
    latent_steps_batch: List[int]
    cot_tokens_batch: List[int]
    stopped_normally_batch: List[bool]
    latent_rollout_steps: int


class LatentQwenForCausalLM(Qwen3ForCausalLM):
    config_class = LatentQwenConfig

    def __init__(self, config: LatentQwenConfig):
        super().__init__(config)
        embed_dim = self.get_input_embeddings().embedding_dim
        self.latent_projector = LatentProjector(
            config.hidden_size,
            embed_dim,  # type: ignore
            dropout=config.latent_projector_dropout,
        )  # type: ignore[arg-type]

    @staticmethod
    def _first_true_index(mask: torch.Tensor, default_index: int) -> torch.Tensor:
        """返回每行第一个 True 的索引, 如果没有True, 则返回 default_index"""
        has_true = mask.any(dim=1)
        first = torch.argmax(mask.to(torch.long), dim=1)
        fallback = torch.full_like(first, int(default_index))
        return torch.where(has_true, first, fallback)

    @staticmethod
    def _sequence_limits_from_attention(attention_mask: torch.Tensor) -> torch.Tensor:
        """找到每个样本的右边界的索引"""
        batch_size, seq_len = attention_mask.shape
        positions = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0).expand(batch_size, -1)
        valid = attention_mask.to(torch.bool)
        return torch.where(valid, positions + 1, positions.new_zeros((batch_size, seq_len))).max(dim=1).values

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        raw_value = os.getenv(name)
        if raw_value is None:
            return bool(default)
        value = str(raw_value).strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off", ""}:
            return False
        return bool(default)

    @classmethod
    def _latent_generation_trigger_ids(cls, token_constants: Dict[str, int]) -> set[int]:
        latent_end_id = int(token_constants["latent_end_id"])
        if not cls._env_flag("LATENT_QWEN3_EARLY_EXIT_ON_SPECIAL", default=False):
            return {latent_end_id}
        return {
            int(token_constants["think_start_id"]),
            int(token_constants["think_end_id"]),
            latent_end_id,
            int(token_constants["im_end_id"]),
            int(token_constants["im_start_id"]),
        }

    @staticmethod
    def _sample_from_logits(
        logits: torch.Tensor,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> int:
        if logits.dim() != 2:
            raise ValueError(f"_sample_from_logits expects rank-2 logits, got shape={list(logits.shape)}")
        if not bool(do_sample):
            return int(torch.argmax(logits, dim=-1).item())
        if float(temperature) <= 0.0:
            raise ValueError(f"temperature must be > 0 for sampling, got temperature={float(temperature)}")

        filtered_logits = logits.float() / float(temperature)
        vocab_size = int(filtered_logits.size(-1))

        if int(top_k) > 0 and int(top_k) < vocab_size:
            topk_values, _ = torch.topk(filtered_logits, k=int(top_k), dim=-1)
            kth_values = topk_values[:, -1].unsqueeze(-1)
            filtered_logits = filtered_logits.masked_fill(filtered_logits < kth_values, float("-inf"))

        if float(top_p) < 1.0:
            if float(top_p) <= 0.0:
                raise ValueError(f"top_p must be > 0 when enabled, got top_p={float(top_p)}")
            sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_remove = cumulative_probs > float(top_p)
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove_mask = torch.zeros_like(sorted_remove, dtype=torch.bool)
            remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_remove)
            filtered_logits = filtered_logits.masked_fill(remove_mask, float("-inf"))

        probs = torch.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return int(next_token.item())

    def _sample_tokens_from_logits(
        self,
        logits: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        fallback_token_id: int,
    ) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError(f"_sample_tokens_from_logits expects rank-2 logits, got shape={list(logits.shape)}")
        active_mask = active_mask.to(device=logits.device, dtype=torch.bool).view(-1)
        next_tokens = torch.full(
            (int(logits.size(0)),),
            fill_value=int(fallback_token_id),
            device=logits.device,
            dtype=torch.long,
        )
        active_rows = torch.nonzero(active_mask, as_tuple=False).view(-1)
        for row_idx in active_rows.tolist():
            next_tokens[row_idx] = int(
                self._sample_from_logits(
                    logits[row_idx : row_idx + 1],
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            )
        return next_tokens

    @staticmethod
    def _pad_prompt_ids_batch(
        prompt_ids_batch: List[List[int]],
        *,
        pad_token_id: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(prompt_ids_batch)
        max_prompt_len = max((len(prompt_ids) for prompt_ids in prompt_ids_batch), default=0)
        input_ids = torch.full((batch_size, max_prompt_len), int(pad_token_id), device=device, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_prompt_len), device=device, dtype=torch.long)
        prompt_lengths = torch.zeros((batch_size,), device=device, dtype=torch.long)
        for row_idx, prompt_ids in enumerate(prompt_ids_batch):
            prompt_len = int(len(prompt_ids))
            prompt_lengths[row_idx] = prompt_len
            if prompt_len <= 0:
                continue
            input_ids[row_idx, :prompt_len] = torch.tensor(prompt_ids, device=device, dtype=torch.long)
            attention_mask[row_idx, :prompt_len] = 1
        return input_ids, attention_mask, prompt_lengths

    def _allocate_loss_hidden(
        self,
        source_positions: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """预填充 hidden state 以计算 loss"""
        if source_positions is None:
            return None
        return torch.zeros(
            (batch_size, source_positions.size(1), self.config.hidden_size),
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _allocate_topk_logits(
        topk_ids: Optional[torch.Tensor],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """预填充 top-k logits buffer"""
        if topk_ids is None:
            return None
        return torch.zeros(
            (topk_ids.size(0), topk_ids.size(1), topk_ids.size(2)),
            device=device,
            dtype=torch.float32,
        )

    @staticmethod
    def _allocate_log_denom(
        source_positions: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if source_positions is None:
            return None
        return torch.zeros((batch_size, source_positions.size(1)), device=device, dtype=torch.float32)

    @staticmethod
    def _allocate_loss_logits(
        source_positions: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if source_positions is None:
            return None
        return torch.zeros((batch_size, source_positions.size(1)), device=device, dtype=torch.float32)

    @staticmethod
    def _normalize_halt_dense_token_ids(
        halt_dense_token_ids: Optional[torch.Tensor | List[int]],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if halt_dense_token_ids is None:
            return None
        if torch.is_tensor(halt_dense_token_ids):
            token_ids = halt_dense_token_ids.to(device=device, dtype=torch.long).view(-1)
        else:
            token_ids = torch.as_tensor(list(halt_dense_token_ids), device=device, dtype=torch.long).view(-1)
        if int(token_ids.numel()) <= 0:
            return None
        return token_ids

    @staticmethod
    def _allocate_halt_dense_logits(
        input_ids: torch.Tensor,
        halt_dense_token_ids: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if halt_dense_token_ids is None or int(halt_dense_token_ids.numel()) <= 0:
            return None
        return torch.zeros(
            (
                int(input_ids.size(0)),
                int(input_ids.size(1)),
                int(halt_dense_token_ids.numel()),
            ),
            device=input_ids.device,
            dtype=torch.float32,
        )

    @staticmethod
    def _allocate_halt_dense_best_allowed_logits(
        input_ids: torch.Tensor,
        halt_dense_token_ids: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if halt_dense_token_ids is None or int(halt_dense_token_ids.numel()) <= 0:
            return None
        return torch.zeros(
            (
                int(input_ids.size(0)),
                int(input_ids.size(1)),
            ),
            device=input_ids.device,
            dtype=torch.float32,
        )

    @staticmethod
    def _build_step_match_mask(
        target_positions: Optional[torch.Tensor],
        pair_mask: Optional[torch.Tensor],
        step_positions: torch.Tensor,
        step_active: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if target_positions is None:
            return None
        if pair_mask is None:
            pair_mask = target_positions >= 0
        return pair_mask & step_active.unsqueeze(1) & (target_positions == step_positions.unsqueeze(1))

    @staticmethod
    def _write_sparse_hidden(
        hidden_buffer: Optional[torch.Tensor],
        source_positions: Optional[torch.Tensor],
        pair_mask: Optional[torch.Tensor],
        step_hidden: torch.Tensor,
        step_positions: torch.Tensor,
        step_active: torch.Tensor,
    ) -> None:
        if hidden_buffer is None or source_positions is None:
            return
        matches = LatentQwenForCausalLM._build_step_match_mask(
            target_positions=source_positions,
            pair_mask=pair_mask,
            step_positions=step_positions,
            step_active=step_active,
        )
        if matches is None or not bool(matches.any().item()):
            return
        expanded = step_hidden.unsqueeze(1).expand(-1, matches.size(1), -1)
        hidden_buffer[matches] = expanded[matches]

    @staticmethod
    def _detach_past_key_values(past_key_values: Any) -> Any:
        if past_key_values is None:
            return None
        if torch.is_tensor(past_key_values):
            return past_key_values.detach()
        if isinstance(past_key_values, tuple):
            return tuple(LatentQwenForCausalLM._detach_past_key_values(x) for x in past_key_values)
        if isinstance(past_key_values, list):
            return [LatentQwenForCausalLM._detach_past_key_values(x) for x in past_key_values]
        if isinstance(past_key_values, dict):
            return {k: LatentQwenForCausalLM._detach_past_key_values(v) for k, v in past_key_values.items()}
        detach_fn = getattr(past_key_values, "detach", None)
        if callable(detach_fn):
            try:
                return detach_fn()
            except Exception:
                return past_key_values
        return past_key_values

    @contextmanager
    def _builder_cache_context(self):
        gc_was_enabled = bool(self.training) and (
            bool(getattr(self, "gradient_checkpointing", False))
            or bool(getattr(getattr(self, "model", None), "gradient_checkpointing", False))
        )
        if gc_was_enabled and hasattr(self, "gradient_checkpointing_disable"):
            self.gradient_checkpointing_disable()
        try:
            yield gc_was_enabled
        finally:
            if gc_was_enabled and hasattr(self, "gradient_checkpointing_enable"):
                self.gradient_checkpointing_enable()

    @staticmethod
    def _build_chunk_pair_matches(
        target_positions: Optional[torch.Tensor],
        pair_mask: Optional[torch.Tensor],
        chunk_start: int,
        chunk_len: int,
        chunk_active_mask: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if target_positions is None or chunk_len <= 0:
            return None
        if pair_mask is None:
            pair_mask = target_positions >= 0

        local_positions = target_positions - int(chunk_start)
        in_chunk = (local_positions >= 0) & (local_positions < int(chunk_len))
        matches = pair_mask & in_chunk
        if not bool(matches.any().item()):
            return None

        safe_local = local_positions.clamp(min=0, max=max(int(chunk_len) - 1, 0)).to(torch.long)
        pair_rows = (
            torch.arange(target_positions.size(0), device=target_positions.device)
            .unsqueeze(1)
            .expand_as(target_positions)
        )
        matches = matches & chunk_active_mask[pair_rows, safe_local]
        if not bool(matches.any().item()):
            return None

        match_indices = torch.nonzero(matches, as_tuple=False)
        rows = match_indices[:, 0]
        cols = match_indices[:, 1]
        local = safe_local[rows, cols]
        return rows, cols, local

    @staticmethod
    def _write_sparse_hidden_for_chunk(
        hidden_buffer: Optional[torch.Tensor],
        source_positions: Optional[torch.Tensor],
        pair_mask: Optional[torch.Tensor],
        chunk_hidden: torch.Tensor,
        chunk_start: int,
        chunk_active_mask: torch.Tensor,
    ) -> None:
        if hidden_buffer is None or source_positions is None:
            return
        chunk_len = int(chunk_hidden.size(1))
        matches = LatentQwenForCausalLM._build_chunk_pair_matches(
            target_positions=source_positions,
            pair_mask=pair_mask,
            chunk_start=int(chunk_start),
            chunk_len=chunk_len,
            chunk_active_mask=chunk_active_mask,
        )
        if matches is None:
            return
        rows, cols, local = matches
        hidden_buffer[rows, cols] = chunk_hidden[rows, local, :]

    def _write_sparse_ce_terms_from_full_logits(
        self,
        target_logits_buffer: Optional[torch.Tensor],
        log_denom_buffer: Optional[torch.Tensor],
        source_positions: Optional[torch.Tensor],
        target_positions: Optional[torch.Tensor],
        pair_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        full_logits: torch.Tensor,
        step_positions: torch.Tensor,
        step_active: torch.Tensor,
    ) -> None:
        if (
            target_logits_buffer is None
            or log_denom_buffer is None
            or source_positions is None
            or target_positions is None
            or labels is None
        ):
            return

        matches = self._build_step_match_mask(
            target_positions=source_positions,
            pair_mask=pair_mask,
            step_positions=step_positions,
            step_active=step_active,
        )
        if matches is None:
            return

        match_indices = torch.nonzero(matches, as_tuple=False)
        if match_indices.numel() <= 0:
            return

        rows = match_indices[:, 0]
        cols = match_indices[:, 1]
        matched_logits = full_logits.index_select(0, rows)
        safe_target_positions = target_positions[rows, cols].clamp(min=0, max=max(int(labels.size(1)) - 1, 0))
        target_ids = labels[rows, safe_target_positions].to(device=matched_logits.device, dtype=torch.long)
        matched_logits_fp32 = matched_logits.float()
        selected = torch.gather(
            matched_logits_fp32,
            dim=-1,
            index=target_ids.clamp(min=0).unsqueeze(-1),
        ).squeeze(-1)
        log_denom = torch.logsumexp(matched_logits_fp32, dim=-1)

        target_logits_buffer[rows, cols] = selected
        log_denom_buffer[rows, cols] = log_denom

    def _write_sparse_teacher_topk_from_full_logits(
        self,
        topk_logits_buffer: Optional[torch.Tensor],
        log_denom_buffer: Optional[torch.Tensor],
        target_positions: Optional[torch.Tensor],
        pair_mask: Optional[torch.Tensor],
        topk_ids: Optional[torch.Tensor],
        full_logits: torch.Tensor,
        step_positions: torch.Tensor,
        step_active: torch.Tensor,
    ) -> None:
        if topk_logits_buffer is None or log_denom_buffer is None or target_positions is None or topk_ids is None:
            return

        matches = self._build_step_match_mask(
            target_positions=target_positions,
            pair_mask=pair_mask,
            step_positions=step_positions,
            step_active=step_active,
        )
        if matches is None:
            return

        match_indices = torch.nonzero(matches, as_tuple=False)
        if match_indices.numel() <= 0:
            return

        rows = match_indices[:, 0]
        cols = match_indices[:, 1]
        temperature = float(max(getattr(self.config, "kl_temperature", 1.0), 1.0e-8))
        matched_logits = full_logits.index_select(0, rows).float() / temperature
        ids = topk_ids[rows, cols].to(device=matched_logits.device, dtype=torch.long)
        selected = torch.gather(matched_logits, dim=-1, index=ids)
        log_denom = torch.logsumexp(matched_logits, dim=-1)

        topk_logits_buffer[rows, cols] = selected
        log_denom_buffer[rows, cols] = log_denom

    def _write_sparse_lm_head_terms_for_chunk(
        self,
        labels: Optional[torch.Tensor],
        loss_target_logits: Optional[torch.Tensor],
        loss_log_denom: Optional[torch.Tensor],
        loss_source_positions: Optional[torch.Tensor],
        loss_target_positions: Optional[torch.Tensor],
        loss_pair_mask: Optional[torch.Tensor],
        teacher_kl_topk_logits: Optional[torch.Tensor],
        teacher_kl_log_denom: Optional[torch.Tensor],
        teacher_kl_source_positions: Optional[torch.Tensor],
        teacher_kl_pair_mask: Optional[torch.Tensor],
        teacher_kl_topk_ids: Optional[torch.Tensor],
        halt_dense_logits: Optional[torch.Tensor],
        halt_dense_best_allowed_logits: Optional[torch.Tensor],
        halt_dense_token_ids: Optional[torch.Tensor],
        chunk_hidden: torch.Tensor,
        chunk_start: int,
        chunk_active_mask: torch.Tensor,
        ce_sync_zero_loss: Optional[torch.Tensor],
        kl_sync_zero_loss: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        ce_enabled = loss_target_logits is not None or loss_log_denom is not None
        kl_enabled = teacher_kl_topk_logits is not None or teacher_kl_log_denom is not None
        halt_dense_enabled = (
            halt_dense_logits is not None
            and halt_dense_best_allowed_logits is not None
            and halt_dense_token_ids is not None
            and int(halt_dense_token_ids.numel()) > 0
        )
        if not ce_enabled and not kl_enabled and not halt_dense_enabled:
            return ce_sync_zero_loss, kl_sync_zero_loss

        chunk_len = int(chunk_hidden.size(1))
        batch_size = int(chunk_hidden.size(0))
        total_positions = int(batch_size * chunk_len)
        if total_positions <= 0:
            return ce_sync_zero_loss, kl_sync_zero_loss

        ce_rows = ce_cols = ce_flat = ce_target_ids = None
        ce_write_enabled = (
            loss_target_logits is not None
            and loss_log_denom is not None
            and loss_source_positions is not None
            and loss_target_positions is not None
            and labels is not None
        )
        if ce_write_enabled:
            ce_matches = self._build_chunk_pair_matches(
                target_positions=loss_source_positions,
                pair_mask=loss_pair_mask,
                chunk_start=int(chunk_start),
                chunk_len=chunk_len,
                chunk_active_mask=chunk_active_mask,
            )
            if ce_matches is not None:
                ce_rows, ce_cols, ce_local = ce_matches
                safe_target_positions = loss_target_positions[ce_rows, ce_cols].clamp( # type: ignore
                    min=0,
                    max=max(int(labels.size(1)) - 1, 0),# type: ignore
                )
                ce_target_ids = labels[ce_rows, safe_target_positions].to(device=chunk_hidden.device, dtype=torch.long)# type: ignore
                ce_flat = ce_rows * chunk_len + ce_local

        kl_rows = kl_cols = kl_flat = kl_topk = None
        kl_write_enabled = (
            teacher_kl_topk_logits is not None
            and teacher_kl_log_denom is not None
            and teacher_kl_source_positions is not None
            and teacher_kl_topk_ids is not None
        )
        if kl_write_enabled:
            kl_matches = self._build_chunk_pair_matches(
                target_positions=teacher_kl_source_positions,
                pair_mask=teacher_kl_pair_mask,
                chunk_start=int(chunk_start),
                chunk_len=chunk_len,
                chunk_active_mask=chunk_active_mask,
            )
            if kl_matches is not None:
                kl_rows, kl_cols, kl_local = kl_matches
                kl_topk = teacher_kl_topk_ids[kl_rows, kl_cols].to(device=chunk_hidden.device, dtype=torch.long)# type: ignore
                kl_flat = kl_rows * chunk_len + kl_local

        projection_chunk_size = int(getattr(self.config, "supervised_logits_chunk_size", 0) or 0)
        if projection_chunk_size <= 0:
            projection_chunk_size = total_positions
        projection_chunk_size = max(projection_chunk_size, 1)
        flat_hidden = chunk_hidden.reshape(total_positions, chunk_hidden.size(-1))
        temperature = float(max(getattr(self.config, "kl_temperature", 1.0), 1.0e-8))

        halt_dense_rows = None
        halt_dense_cols = None
        halt_dense_token_ids_local = None
        if halt_dense_enabled:
            if halt_dense_logits is None or halt_dense_best_allowed_logits is None or halt_dense_token_ids is None:
                raise ValueError(
                    "halt_dense_logits, halt_dense_best_allowed_logits, and halt_dense_token_ids must all be provided"
                )
            if int(halt_dense_logits.size(2)) != int(halt_dense_token_ids.numel()):
                raise ValueError(
                    "halt_dense_logits last dim does not match halt_dense_token_ids size: "
                    f"logits_shape={list(halt_dense_logits.shape)}, token_ids={int(halt_dense_token_ids.numel())}"
                )
            if list(halt_dense_best_allowed_logits.shape) != list(halt_dense_logits.shape[:2]):
                raise ValueError(
                    "halt_dense_best_allowed_logits shape does not match halt_dense_logits prefix: "
                    f"best_allowed_shape={list(halt_dense_best_allowed_logits.shape)}, "
                    f"logits_shape={list(halt_dense_logits.shape)}"
                )
            flat_positions = torch.arange(total_positions, device=chunk_hidden.device, dtype=torch.long)
            halt_dense_rows = torch.div(flat_positions, chunk_len, rounding_mode="floor")
            halt_dense_cols = torch.remainder(flat_positions, chunk_len) + int(chunk_start)
            halt_dense_token_ids_local = halt_dense_token_ids.to(device=chunk_hidden.device, dtype=torch.long)

        for sub_start in range(0, total_positions, projection_chunk_size):
            sub_end = min(sub_start + projection_chunk_size, total_positions)
            sub_logits = self.lm_head(flat_hidden[sub_start:sub_end])
            zero_anchor = sub_logits.sum() * 0.0
            if ce_enabled and ce_sync_zero_loss is not None:
                ce_sync_zero_loss = ce_sync_zero_loss + zero_anchor
            if kl_enabled and kl_sync_zero_loss is not None:
                kl_sync_zero_loss = kl_sync_zero_loss + zero_anchor

            if halt_dense_enabled and halt_dense_rows is not None and halt_dense_cols is not None:
                forbidden_logits = sub_logits.index_select(
                    dim=-1,
                    index=halt_dense_token_ids_local, # type: ignore[arg-type]
                ).float()
                topk_size = min(int(halt_dense_token_ids_local.numel()) + 1, int(sub_logits.size(-1)))
                topk_values, topk_indices = torch.topk(sub_logits.float(), k=max(topk_size, 1), dim=-1)
                allowed_mask = ~(
                    topk_indices.unsqueeze(-1) == halt_dense_token_ids_local.view(1, 1, -1)
                ).any(dim=-1)
                best_allowed = torch.full(
                    (int(sub_logits.size(0)),),
                    fill_value=float("-inf"),
                    device=sub_logits.device,
                    dtype=torch.float32,
                )
                if bool(allowed_mask.any().item()):
                    safe_values = torch.where(
                        allowed_mask,
                        topk_values,
                        torch.full_like(topk_values, float("-inf")),
                    )
                    best_allowed = safe_values.max(dim=-1).values
                halt_dense_logits[ # type: ignore[index]
                    halt_dense_rows[sub_start:sub_end],
                    halt_dense_cols[sub_start:sub_end],
                    :,
                ] = forbidden_logits
                halt_dense_best_allowed_logits[ # type: ignore[index]
                    halt_dense_rows[sub_start:sub_end],
                    halt_dense_cols[sub_start:sub_end],
                ] = best_allowed

            if (
                ce_write_enabled
                and ce_flat is not None
                and ce_target_ids is not None
                and ce_rows is not None
                and ce_cols is not None
            ):
                ce_in_sub = (ce_flat >= sub_start) & (ce_flat < sub_end)
                if bool(ce_in_sub.any().item()):
                    local_rows = (ce_flat[ce_in_sub] - sub_start).to(torch.long)
                    matched_logits_fp32 = sub_logits.index_select(0, local_rows).float()
                    selected = torch.gather(
                        matched_logits_fp32,
                        dim=-1,
                        index=ce_target_ids[ce_in_sub].clamp(min=0).unsqueeze(-1),
                    ).squeeze(-1)
                    log_denom = torch.logsumexp(matched_logits_fp32, dim=-1)
                    loss_target_logits[ce_rows[ce_in_sub], ce_cols[ce_in_sub]] = selected# type: ignore
                    loss_log_denom[ce_rows[ce_in_sub], ce_cols[ce_in_sub]] = log_denom# type: ignore

            if (
                kl_write_enabled
                and kl_flat is not None
                and kl_topk is not None
                and kl_rows is not None
                and kl_cols is not None
            ):
                kl_in_sub = (kl_flat >= sub_start) & (kl_flat < sub_end)
                if bool(kl_in_sub.any().item()):
                    local_rows = (kl_flat[kl_in_sub] - sub_start).to(torch.long)
                    matched_logits = sub_logits.index_select(0, local_rows).float() / temperature
                    selected = torch.gather(matched_logits, dim=-1, index=kl_topk[kl_in_sub])
                    log_denom = torch.logsumexp(matched_logits, dim=-1)
                    teacher_kl_topk_logits[kl_rows[kl_in_sub], kl_cols[kl_in_sub]] = selected# type: ignore
                    teacher_kl_log_denom[kl_rows[kl_in_sub], kl_cols[kl_in_sub]] = log_denom# type: ignore

        return ce_sync_zero_loss, kl_sync_zero_loss

    def _build_latent_recursive_embeds(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cot_mask: torch.Tensor,
        latent_positions: Optional[torch.Tensor],
        latent_slot_mask: Optional[torch.Tensor],
        latent_start_positions: torch.Tensor,
        global_prompt_len: Optional[int],
        use_no_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        del cot_mask
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        valid_token_mask = attention_mask.to(torch.bool)
        embedding_table = self.get_input_embeddings()
        embed_dtype = embedding_table.weight.dtype
        embed_dim = int(embedding_table.weight.size(1)) # type: ignore
        latent_embed_overrides = torch.zeros((batch_size, seq_len, embed_dim), device=device, dtype=embed_dtype) # type: ignore
        latent_embed_mask = torch.zeros((batch_size, seq_len), device=device, dtype=torch.bool)

        prompt_len = int(latent_start_positions.max().item()) + 1
        if global_prompt_len is not None:
            prompt_len = int(global_prompt_len)
        prompt_len = max(min(prompt_len, int(seq_len)), 0)

        prompt_calls = 0
        latent_calls = 0
        grad_context = torch.no_grad if use_no_grad else nullcontext
        with self._builder_cache_context() as gc_temporarily_disabled, grad_context():
            if prompt_len > 0:
                prompt_positions = torch.arange(prompt_len, device=device).unsqueeze(0).expand(batch_size, -1)
                prompt_active_mask = valid_token_mask[:, :prompt_len] & (
                    prompt_positions <= latent_start_positions.unsqueeze(1)
                )
                prompt_outputs = self.model(
                    input_ids=input_ids[:, :prompt_len],
                    attention_mask=prompt_active_mask.to(dtype=attention_mask.dtype),
                    position_ids=position_ids[:, :prompt_len],
                    past_key_values=None,
                    use_cache=True,
                    return_dict=True,
                )
                if prompt_outputs.past_key_values is None:
                    raise RuntimeError(
                        "Expected latent embed builder prompt forward to return past_key_values. "
                        "Builder cache path may not have disabled gradient checkpointing correctly."
                    )
                prompt_calls = 1
                runtime_cache_key_mask = prompt_active_mask
                past_key_values = prompt_outputs.past_key_values
                prompt_hidden = prompt_outputs.last_hidden_state
                prev_hidden = torch.zeros(
                    (batch_size, self.config.hidden_size), device=device, dtype=prompt_hidden.dtype
                )
                row_ids = torch.arange(batch_size, device=device)
                safe_latent_start_pos = latent_start_positions.clamp(min=0, max=max(prompt_len - 1, 0))
                latent_start_active = (
                    (latent_start_positions >= 0)
                    & (latent_start_positions < prompt_len)
                    & prompt_active_mask[row_ids, safe_latent_start_pos]
                )
                if bool(latent_start_active.any().item()):
                    active_rows = torch.nonzero(latent_start_active, as_tuple=False).view(-1)
                    prev_hidden[active_rows] = prompt_hidden[active_rows, safe_latent_start_pos[active_rows], :]
            else:
                runtime_cache_key_mask = torch.zeros((batch_size, 0), device=device, dtype=torch.bool)
                past_key_values = None
                prev_hidden = torch.zeros(
                    (batch_size, self.config.hidden_size),
                    device=device,
                    dtype=embed_dtype, # type: ignore
                ) # type: ignore

            internal_slots = int(latent_positions.size(1)) if latent_positions is not None else 0
            for slot_idx in range(internal_slots):
                raw_positions = latent_positions[:, slot_idx] # type: ignore
                slot_active = raw_positions >= 0
                if latent_slot_mask is not None:
                    slot_active = slot_active & latent_slot_mask[:, slot_idx]

                step_positions = raw_positions.clamp(min=0, max=max(seq_len - 1, 0))
                in_range = (raw_positions >= 0) & (raw_positions < seq_len)
                step_active = slot_active & in_range
                if bool(step_active.any().item()):
                    rows = torch.nonzero(step_active, as_tuple=False).view(-1)
                    step_active[rows] = valid_token_mask[rows, step_positions[rows]]

                latent_embeds = self.latent_projector(prev_hidden)
                step_embeds = torch.where(step_active.unsqueeze(-1), latent_embeds, torch.zeros_like(latent_embeds))
                if bool(step_active.any().item()):
                    rows = torch.nonzero(step_active, as_tuple=False).view(-1)
                    cols = step_positions[rows]
                    latent_embed_overrides[rows, cols, :] = step_embeds[rows].to(dtype=embed_dtype) # type: ignore
                    latent_embed_mask[rows, cols] = True

                step_pos_ids = torch.zeros((batch_size,), device=device, dtype=position_ids.dtype)
                if bool(step_active.any().item()):
                    rows = torch.nonzero(step_active, as_tuple=False).view(-1)
                    step_pos_ids[rows] = position_ids[rows, step_positions[rows]]

                step_attention_mask = torch.cat([runtime_cache_key_mask, step_active.unsqueeze(1)], dim=1)
                step_outputs = self.model(
                    inputs_embeds=step_embeds.unsqueeze(1),
                    attention_mask=step_attention_mask.to(dtype=attention_mask.dtype),
                    position_ids=step_pos_ids.unsqueeze(1),
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                if step_outputs.past_key_values is None:
                    raise RuntimeError(
                        "Expected latent embed builder latent forward to return past_key_values. "
                        "Builder cache path may not have disabled gradient checkpointing correctly."
                    )
                step_hidden = step_outputs.last_hidden_state[:, -1, :]
                prev_hidden = torch.where(step_active.unsqueeze(-1), step_hidden, prev_hidden)
                runtime_cache_key_mask = step_attention_mask
                past_key_values = step_outputs.past_key_values
                latent_calls += 1

        stats = {
            "prompt_calls": int(prompt_calls),
            "latent_calls": int(latent_calls),
            "latent_builder_calls": int(prompt_calls + latent_calls),
            "latent_builder_no_grad": int(1 if use_no_grad else 0),
            "gc_temporarily_disabled": int(1 if gc_temporarily_disabled else 0),
        }
        return latent_embed_overrides, latent_embed_mask, stats

    def _forward_global_stateless_teacher_forcing(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        labels: Optional[torch.Tensor],
        cot_mask: torch.Tensor,
        latent_positions: Optional[torch.Tensor],
        latent_slot_mask: Optional[torch.Tensor],
        latent_start_positions: torch.Tensor,
        latent_end_positions: torch.Tensor,
        halt_dense_token_ids: Optional[torch.Tensor | List[int]],
        loss_source_positions: Optional[torch.Tensor],
        loss_target_positions: Optional[torch.Tensor],
        loss_pair_mask: Optional[torch.Tensor],
        teacher_kl_source_positions: Optional[torch.Tensor],
        teacher_kl_pair_mask: Optional[torch.Tensor],
        teacher_kl_topk_ids: Optional[torch.Tensor],
        global_prompt_len: Optional[int],
        global_stage3_start: Optional[int],
    ) -> CausalLMOutputWithPast:
        del latent_end_positions
        batch_size = input_ids.size(0)
        device = input_ids.device
        loss_hidden_states = None
        loss_target_logits = self._allocate_loss_logits(
            source_positions=loss_source_positions,
            batch_size=batch_size,
            device=device,
        )
        loss_log_denom = self._allocate_log_denom(
            source_positions=loss_source_positions,
            batch_size=batch_size,
            device=device,
        )
        kl_enabled = (
            teacher_kl_topk_ids is not None
            and teacher_kl_source_positions is not None
            and teacher_kl_pair_mask is not None
            and bool(torch.any(teacher_kl_pair_mask).item())
        )
        teacher_kl_topk_logits = None
        teacher_kl_log_denom = None
        if kl_enabled:
            teacher_kl_topk_logits = self._allocate_topk_logits(
                topk_ids=teacher_kl_topk_ids,
                device=device,
            )
            teacher_kl_log_denom = self._allocate_log_denom(
                source_positions=teacher_kl_source_positions,
                batch_size=batch_size,
                device=device,
            )
        ce_sync_zero_loss = input_ids.new_zeros((), dtype=torch.float32)
        kl_sync_zero_loss = input_ids.new_zeros((), dtype=torch.float32)
        normalized_halt_dense_token_ids = self._normalize_halt_dense_token_ids(
            halt_dense_token_ids=halt_dense_token_ids,
            device=device,
        )
        halt_dense_logits = self._allocate_halt_dense_logits(
            input_ids=input_ids,
            halt_dense_token_ids=normalized_halt_dense_token_ids,
        )
        halt_dense_best_allowed_logits = self._allocate_halt_dense_best_allowed_logits(
            input_ids=input_ids,
            halt_dense_token_ids=normalized_halt_dense_token_ids,
        )

        latent_builder_no_grad = bool(getattr(self.config, "latent_embed_builder_no_grad", True))
        latent_embed_overrides, latent_embed_mask, latent_builder_stats = self._build_latent_recursive_embeds(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cot_mask=cot_mask,
            latent_positions=latent_positions,
            latent_slot_mask=latent_slot_mask,
            latent_start_positions=latent_start_positions,
            global_prompt_len=global_prompt_len,
            use_no_grad=latent_builder_no_grad,
        )

        token_embeds = self.get_input_embeddings()(input_ids)
        if bool(latent_embed_mask.any().item()):
            full_inputs_embeds = token_embeds.clone()
            full_inputs_embeds[latent_embed_mask] = latent_embed_overrides[latent_embed_mask].to(
                dtype=token_embeds.dtype
            )
        else:
            full_inputs_embeds = token_embeds

        full_active_mask = attention_mask.to(torch.bool)
        full_outputs = self.model(
            inputs_embeds=full_inputs_embeds,
            attention_mask=full_active_mask.to(dtype=attention_mask.dtype),
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            return_dict=True,
        )
        full_hidden = full_outputs.last_hidden_state
        ce_sync_zero_loss, kl_sync_zero_loss = self._write_sparse_lm_head_terms_for_chunk(
            labels=labels,
            loss_target_logits=loss_target_logits,
            loss_log_denom=loss_log_denom,
            loss_source_positions=loss_source_positions,
            loss_target_positions=loss_target_positions,
            loss_pair_mask=loss_pair_mask,
            teacher_kl_topk_logits=teacher_kl_topk_logits,
            teacher_kl_log_denom=teacher_kl_log_denom,
            teacher_kl_source_positions=teacher_kl_source_positions,
            teacher_kl_pair_mask=teacher_kl_pair_mask,
            teacher_kl_topk_ids=teacher_kl_topk_ids,
            halt_dense_logits=halt_dense_logits,
            halt_dense_best_allowed_logits=halt_dense_best_allowed_logits,
            halt_dense_token_ids=normalized_halt_dense_token_ids,
            chunk_hidden=full_hidden,
            chunk_start=0,
            chunk_active_mask=full_active_mask,
            ce_sync_zero_loss=ce_sync_zero_loss,
            kl_sync_zero_loss=kl_sync_zero_loss,
        )
        outputs = CausalLMOutputWithPast(
            loss=None,
            logits=None,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
        outputs.halt_dense_logits = halt_dense_logits
        outputs.halt_dense_best_allowed_logits = halt_dense_best_allowed_logits
        outputs.halt_dense_token_ids = normalized_halt_dense_token_ids
        outputs.loss_hidden_states = loss_hidden_states
        outputs.loss_target_logits = loss_target_logits
        outputs.loss_log_denom = loss_log_denom
        outputs.teacher_kl_topk_logits = teacher_kl_topk_logits
        outputs.teacher_kl_log_denom = teacher_kl_log_denom
        outputs.ce_sync_zero_loss = ce_sync_zero_loss
        outputs.kl_sync_zero_loss = kl_sync_zero_loss

        seq_len = int(input_ids.size(1))
        prompt_len = (
            int(global_prompt_len) if global_prompt_len is not None else int(latent_start_positions.max().item()) + 1
        )
        prompt_len = max(min(prompt_len, seq_len), 0)
        stage3_start = (
            int(global_stage3_start)
            if global_stage3_start is not None
            else int(self._first_true_index(cot_mask.to(torch.bool), default_index=seq_len).max().item())
        )
        stage3_start = max(min(stage3_start, seq_len), 0)
        latent_slot_count = int(latent_positions.size(1)) if latent_positions is not None else 0
        latent_bptt_window = int(getattr(self.config, "latent_bptt_window", 0) or 0)
        discrete_stage_tokens = max(seq_len - stage3_start, 0)
        config_discrete_chunk_size = int(getattr(self.config, "discrete_chunk_size", 0) or 0)
        discrete_chunk_size = 0
        discrete_chunk_count = 1 if discrete_stage_tokens > 0 else 0
        projection_chunk_size = int(getattr(self.config, "supervised_logits_chunk_size", 0) or 0)
        total_positions = int(batch_size * seq_len)
        if total_positions <= 0:
            global_lm_head_calls = 0
        elif projection_chunk_size <= 0:
            global_lm_head_calls = 1
        else:
            global_lm_head_calls = (total_positions + projection_chunk_size - 1) // projection_chunk_size

        latent_builder_calls = int(latent_builder_stats.get("latent_builder_calls", 0))
        prompt_builder_calls = int(latent_builder_stats.get("prompt_calls", 0))
        latent_builder_latent_calls = int(latent_builder_stats.get("latent_calls", 0))
        main_model_calls = 1
        outputs.latent_memory_info = {
            "prompt_stage_tokens": max(prompt_len, 0),
            "latent_stage_tokens": max(stage3_start - prompt_len, 0),
            "discrete_stage_tokens": discrete_stage_tokens,
            "sequence_limits": [],
        }
        outputs.stage_trace_info = {
            "prompt_stage_tokens": max(prompt_len, 0),
            "stage3_start": max(stage3_start, 0),
            "latent_slot_count": latent_slot_count,
            "latent_bptt_window": int(latent_bptt_window),
            "tbptt_cut_count": 0,
            "discrete_stage_tokens": discrete_stage_tokens,
            "discrete_chunk_size": discrete_chunk_size,
            "config_discrete_chunk_size": config_discrete_chunk_size,
            "discrete_chunk_count": int(discrete_chunk_count),
            "discrete_use_cache": False,
            "discrete_replay_use_cache": False,
            "prompt_model_calls": int(prompt_builder_calls),
            "latent_model_calls": int(latent_builder_latent_calls),
            "discrete_model_calls": 0,
            "discrete_replay_model_calls": 0,
            "latent_projector_calls": int(latent_slot_count),
            "prompt_lm_head_calls": 0,
            "latent_lm_head_calls": 0,
            "discrete_lm_head_calls": 0,
            "global_lm_head_calls": int(global_lm_head_calls),
            "global_stateless_forward": True,
            "latent_builder_no_grad": bool(latent_builder_no_grad),
            "latent_builder_gc_temporarily_disabled": bool(int(latent_builder_stats.get("gc_temporarily_disabled", 0))),
            "latent_builder_calls": int(latent_builder_calls),
            "main_model_calls": int(main_model_calls),
            "total_model_calls": int(latent_builder_calls + main_model_calls),
            "total_lm_head_calls": int(global_lm_head_calls),
        }
        return outputs

    def _forward_base_causal(self, **kwargs: Any) -> CausalLMOutputWithPast:
        return super().forward(**kwargs)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        latent_internal_mask: Optional[torch.Tensor] = None,
        latent_positions: Optional[torch.Tensor] = None,
        latent_slot_mask: Optional[torch.Tensor] = None,
        latent_lengths: Optional[torch.Tensor] = None,
        latent_stop_step: Optional[torch.Tensor] = None,
        stop_targets: Optional[torch.Tensor] = None,
        stop_valid_mask: Optional[torch.Tensor] = None,
        latent_start_positions: Optional[torch.Tensor] = None,
        latent_end_positions: Optional[torch.Tensor] = None,
        latent_end_id: Optional[int] = None,
        halt_dense_token_ids: Optional[torch.Tensor | List[int]] = None,
        loss_source_positions: Optional[torch.Tensor] = None,
        loss_target_positions: Optional[torch.Tensor] = None,
        loss_pair_mask: Optional[torch.Tensor] = None,
        teacher_kl_source_positions: Optional[torch.Tensor] = None,
        teacher_kl_pair_mask: Optional[torch.Tensor] = None,
        teacher_kl_topk_ids: Optional[torch.Tensor] = None,
        cot_mask: Optional[torch.Tensor] = None,
        global_prompt_len: Optional[int] = None,
        global_stage3_start: Optional[int] = None,
        cp_forward_mode: Optional[str] = None,
        latent_hidden: Optional[torch.Tensor] = None,
        latent_embed_overrides: Optional[torch.Tensor] = None,
        latent_embed_mask: Optional[torch.Tensor] = None,
        latent_projector_hiddens: Optional[torch.Tensor] = None,
        latent_projector_positions: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        del latent_internal_mask
        del latent_lengths
        del latent_stop_step
        del stop_targets
        del stop_valid_mask
        del latent_end_id

        if attention_bias is not None:
            raise ValueError(
                "The staged SFT forward path does not support custom attention_bias. "
                "Disable use_custom_attention_bias for training."
            )

        if cp_forward_mode is not None:
            mode = str(cp_forward_mode)
            if mode == "project_latent":
                if latent_hidden is None:
                    raise ValueError("cp_forward_mode=project_latent requires latent_hidden")
                return SimpleNamespace(latent_embeds=self.latent_projector(latent_hidden))
            if mode == "mixed_inputs":
                if input_ids is None:
                    raise ValueError("cp_forward_mode=mixed_inputs requires input_ids")
                embedding_layer = self.get_input_embeddings()
                if int(input_ids.numel()) > 0:
                    min_token_id = int(input_ids.min().item())
                    max_token_id = int(input_ids.max().item())
                    vocab_size = int(getattr(embedding_layer, "num_embeddings", 0) or 0)
                    if vocab_size > 0 and (min_token_id < 0 or max_token_id >= vocab_size):
                        raise ValueError(
                            "cp_forward_mode=mixed_inputs received input_ids outside embedding range: "
                            f"min={min_token_id}, max={max_token_id}, vocab_size={vocab_size}, "
                            f"input_shape={list(input_ids.shape)}"
                        )
                token_embeds = embedding_layer(input_ids)
                if latent_projector_hiddens is not None or latent_projector_positions is not None:
                    if latent_projector_hiddens is None or latent_projector_positions is None:
                        raise ValueError(
                            "latent_projector_hiddens and latent_projector_positions must be provided together"
                        )
                    if position_ids is None:
                        raise ValueError("cp_forward_mode=mixed_inputs requires position_ids for latent projector slots")
                    if int(input_ids.size(0)) != 1:
                        raise ValueError(
                            "CP latent projector slot replacement currently supports batch size 1, "
                            f"got input_shape={list(input_ids.shape)}"
                        )
                    projector_positions = latent_projector_positions.to(device=input_ids.device, dtype=torch.long).view(-1)
                    projector_hiddens = latent_projector_hiddens.to(device=input_ids.device)
                    if int(projector_hiddens.dim()) != 2:
                        raise ValueError(
                            "latent_projector_hiddens must have shape [num_latent_slots, hidden_size], "
                            f"got {list(projector_hiddens.shape)}"
                        )
                    if int(projector_hiddens.size(0)) != int(projector_positions.numel()):
                        raise ValueError(
                            "latent_projector_hiddens row count must match latent_projector_positions: "
                            f"hiddens_shape={list(projector_hiddens.shape)}, "
                            f"positions_shape={list(projector_positions.shape)}"
                        )
                    if int(projector_hiddens.size(-1)) != int(self.config.hidden_size):
                        raise ValueError(
                            "latent_projector_hiddens hidden size must match model hidden_size: "
                            f"hiddens_shape={list(projector_hiddens.shape)}, hidden_size={int(self.config.hidden_size)}"
                        )
                    if int(projector_positions.numel()) > 0:
                        local_position_ids = position_ids.to(device=input_ids.device, dtype=torch.long)
                        projected_embeds = self.latent_projector(projector_hiddens)
                        if int(projected_embeds.size(-1)) != int(token_embeds.size(-1)):
                            raise ValueError(
                                "Projected latent embedding dim must match token embeddings: "
                                f"projected_shape={list(projected_embeds.shape)}, "
                                f"token_embeds_shape={list(token_embeds.shape)}"
                            )
                        token_embeds = token_embeds.clone()
                        for slot_idx in range(int(projector_positions.numel())):
                            slot_mask = local_position_ids == projector_positions[slot_idx]
                            if bool(slot_mask.any().item()):
                                token_embeds[slot_mask] = projected_embeds[slot_idx].to(dtype=token_embeds.dtype)
                if latent_embed_overrides is not None and latent_embed_mask is not None:
                    if list(latent_embed_overrides.shape[:2]) != list(input_ids.shape):
                        raise ValueError(
                            "latent_embed_overrides prefix shape must match input_ids: "
                            f"overrides_shape={list(latent_embed_overrides.shape)}, input_shape={list(input_ids.shape)}"
                        )
                    if int(latent_embed_overrides.size(-1)) != int(token_embeds.size(-1)):
                        raise ValueError(
                            "latent_embed_overrides hidden size must match token embeddings: "
                            f"overrides_shape={list(latent_embed_overrides.shape)}, "
                            f"token_embeds_shape={list(token_embeds.shape)}"
                        )
                    if list(latent_embed_mask.shape) != list(input_ids.shape):
                        raise ValueError(
                            "latent_embed_mask shape must match input_ids: "
                            f"mask_shape={list(latent_embed_mask.shape)}, input_shape={list(input_ids.shape)}"
                        )
                    mask = latent_embed_mask.to(torch.bool)
                    if bool(mask.any().item()):
                        token_embeds = token_embeds.clone()
                        token_embeds[mask] = latent_embed_overrides[mask].to(dtype=token_embeds.dtype)
                return self._forward_base_causal(
                    inputs_embeds=token_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=kwargs.pop("use_cache", False),
                    output_hidden_states=kwargs.pop("output_hidden_states", False),
                    return_dict=kwargs.pop("return_dict", True),
                    **kwargs,
                )
            raise ValueError(f"Unsupported cp_forward_mode: {mode}")

        has_training_contract = (
            inputs_embeds is None
            and input_ids is not None
            and latent_start_positions is not None
            and latent_end_positions is not None
        )
        if has_training_contract:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long) # type: ignore
            if position_ids is None:
                position_ids = (
                    torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids) # type: ignore
                )
            if cot_mask is None:
                raise ValueError("cot_mask is required for staged latent SFT forward")
            forward_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "labels": labels,
                "cot_mask": cot_mask,
                "latent_positions": latent_positions,
                "latent_slot_mask": latent_slot_mask,
                "latent_start_positions": latent_start_positions,
                "latent_end_positions": latent_end_positions,
                "halt_dense_token_ids": halt_dense_token_ids,
                "loss_source_positions": loss_source_positions,
                "loss_target_positions": loss_target_positions,
                "loss_pair_mask": loss_pair_mask,
                "teacher_kl_source_positions": teacher_kl_source_positions,
                "teacher_kl_pair_mask": teacher_kl_pair_mask,
                "teacher_kl_topk_ids": teacher_kl_topk_ids,
                "global_prompt_len": global_prompt_len,
                "global_stage3_start": global_stage3_start,
            }
            use_global_stateless = bool(getattr(self.config, "global_stateless_forward", True))
            is_gc_enabled = bool(getattr(self, "gradient_checkpointing", False)) or bool(
                getattr(getattr(self, "model", None), "gradient_checkpointing", False)
            )
            if (
                (not use_global_stateless)
                and bool(getattr(self.config, "legacy_staged_forward_fallback", True))
                and is_gc_enabled
            ):
                raise RuntimeError(
                    "Staged forward path requires use_cache/past_key_values, but gradient_checkpointing is enabled "
                    "and disables cache. Set gradient_checkpointing=false for staged path or enable "
                    "global_stateless_forward."
                )
            if use_global_stateless:
                return self._forward_global_stateless_teacher_forcing(**forward_kwargs)
            raise RuntimeError(
                "global_stateless_forward is disabled; no valid forward path"
            )

        if input_ids is None and inputs_embeds is None:
            raise ValueError("forward requires input_ids or inputs_embeds")

        # Fallback for eval/generation and generic causal LM usage.
        return self._forward_base_causal(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    @torch.no_grad()
    def _generate_with_latent_batched(
        self,
        prompt_ids_batch: List[List[int]],
        token_constants: Dict[str, int],
        max_new_tokens: int,
        latent_max_steps: int,
        distributed_lockstep: bool = False,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> LatentGenerationBatchOutput:
        device = next(self.parameters()).device
        def _module_device(module: nn.Module, fallback: torch.device) -> torch.device:
            try:
                return next(module.parameters()).device
            except StopIteration:
                try:
                    return next(module.buffers()).device
                except StopIteration:
                    return fallback

        batch_size = int(len(prompt_ids_batch))
        latent_rollout_steps = max(int(latent_max_steps), 0)
        if batch_size <= 0:
            return LatentGenerationBatchOutput([], [], [], [], latent_rollout_steps)
        if max_new_tokens <= 0:
            return LatentGenerationBatchOutput(
                token_ids_batch=[[] for _ in range(batch_size)],
                latent_steps_batch=[0 for _ in range(batch_size)],
                cot_tokens_batch=[0 for _ in range(batch_size)],
                stopped_normally_batch=[False for _ in range(batch_size)],
                latent_rollout_steps=latent_rollout_steps,
            )

        latent_start_id = int(token_constants["latent_start_id"])
        latent_end_id = int(token_constants["latent_end_id"])
        think_start_id = int(token_constants["think_start_id"])
        im_end_id = int(token_constants["im_end_id"])
        latent_trigger_ids = self._latent_generation_trigger_ids(token_constants)
        hidden_size = int(self.config.hidden_size)
        embedding_table = self.get_input_embeddings()
        embed_dim = int(embedding_table.embedding_dim)
        embed_dtype = embedding_table.weight.dtype
        lm_head_device = _module_device(self.lm_head, device)
        latent_projector_device = _module_device(self.latent_projector, device)
        pad_token_id = int(getattr(self.config, "pad_token_id", None) or im_end_id)
        lockstep_enabled = bool(
            distributed_lockstep and dist.is_available() and dist.is_initialized() and int(dist.get_world_size()) > 1
        )
        world_size = int(dist.get_world_size()) if (dist.is_available() and dist.is_initialized()) else 1

        prompt_input_ids, prompt_attention_mask, prompt_lengths = self._pad_prompt_ids_batch(
            prompt_ids_batch,
            pad_token_id=pad_token_id,
            device=device,
        )
        generated_batch: List[List[int]] = [[latent_start_id] for _ in range(batch_size)]
        stopped_normally = torch.zeros((batch_size,), device=device, dtype=torch.bool)
        visible_budget_exhausted = torch.tensor(
            [len(tokens) >= max_new_tokens for tokens in generated_batch],
            device=device,
            dtype=torch.bool,
        )
        latent_end_seen = torch.zeros((batch_size,), device=device, dtype=torch.bool)
        first_latent_end_step = torch.full(
            (batch_size,),
            fill_value=int(latent_rollout_steps),
            device=device,
            dtype=torch.long,
        )
        visible_latent_counts = torch.zeros((batch_size,), device=device, dtype=torch.long)

        prompt_outputs = self._forward_base_causal(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            use_cache=True,
            past_key_values=None,
            output_hidden_states=True,
            return_dict=True,
        )
        if prompt_outputs.past_key_values is None or prompt_outputs.hidden_states is None:
            raise RuntimeError("Expected prompt generation forward to return past_key_values and hidden_states")
        past_key_values = prompt_outputs.past_key_values
        runtime_cache_key_mask = prompt_attention_mask.to(torch.bool)
        prompt_hidden = prompt_outputs.hidden_states[-1]
        hidden_device = prompt_hidden.device
        last_prompt_indices = torch.clamp(prompt_lengths - 1, min=0).to(device=hidden_device, dtype=torch.long)
        row_ids = torch.arange(batch_size, device=hidden_device)
        latent_hidden = prompt_hidden[row_ids, last_prompt_indices, :]
        next_position_ids = prompt_lengths.to(device=device, dtype=torch.long)
        if self._env_flag("LATENT_QWEN3_DEBUG", default=False):
            print(
                "[latent_qwen3][device_debug]",
                {
                    "input_device": str(device),
                    "hidden_device": str(hidden_device),
                    "lm_head_device": str(lm_head_device),
                    "latent_projector_device": str(latent_projector_device),
                },
            )

        latent_start_ids = torch.full((batch_size, 1), latent_start_id, device=device, dtype=torch.long)
        latent_start_mask = torch.ones((batch_size, 1), device=device, dtype=torch.bool)
        latent_start_outputs = self._forward_base_causal(
            input_ids=latent_start_ids,
            attention_mask=torch.cat([runtime_cache_key_mask, latent_start_mask], dim=1).to(
                dtype=prompt_attention_mask.dtype
            ),
            position_ids=next_position_ids.unsqueeze(1),
            use_cache=True,
            past_key_values=past_key_values,
            output_hidden_states=True,
            return_dict=True,
        )
        if latent_start_outputs.past_key_values is None or latent_start_outputs.hidden_states is None:
            raise RuntimeError("Expected latent start generation forward to return past_key_values and hidden_states")
        past_key_values = latent_start_outputs.past_key_values
        runtime_cache_key_mask = torch.cat([runtime_cache_key_mask, latent_start_mask], dim=1)
        next_position_ids = next_position_ids + latent_start_mask.view(-1).to(torch.long)
        latent_hidden = latent_start_outputs.hidden_states[-1][:, -1, :]

        active_step_key_mask = torch.ones((batch_size, 1), device=device, dtype=torch.bool)
        dummy_embed = torch.zeros((batch_size, 1, embed_dim), device=device, dtype=embed_dtype)
        for rollout_step in range(latent_rollout_steps):
            stop_hidden = latent_hidden.to(device=lm_head_device)
            stop_logits = self.lm_head(stop_hidden)
            stop_tokens = torch.argmax(stop_logits, dim=-1).to(device=device, dtype=torch.long)
            append_visible_mask = (~latent_end_seen) & (~visible_budget_exhausted)
            append_rows = torch.nonzero(append_visible_mask, as_tuple=False).view(-1)
            for row_idx in append_rows.tolist():
                token_id = int(stop_tokens[row_idx].item())
                is_latent_stop = token_id in latent_trigger_ids
                if len(generated_batch[row_idx]) < max_new_tokens:
                    if is_latent_stop:
                        generated_batch[row_idx].append(latent_end_id)
                    else:
                        generated_batch[row_idx].append(token_id)
                        visible_latent_counts[row_idx] += 1
                if len(generated_batch[row_idx]) >= max_new_tokens:
                    visible_budget_exhausted[row_idx] = True
            trigger_mask = torch.zeros_like(latent_end_seen)
            for trigger_id in latent_trigger_ids:
                trigger_mask = trigger_mask | (stop_tokens == int(trigger_id))
            newly_seen_end = (~latent_end_seen) & trigger_mask
            if bool(newly_seen_end.any().item()):
                first_latent_end_step = torch.where(
                    newly_seen_end,
                    torch.full_like(first_latent_end_step, fill_value=int(rollout_step)),
                    first_latent_end_step,
                )
                latent_end_seen = latent_end_seen | newly_seen_end

            projector_hidden = latent_hidden.to(device=latent_projector_device)
            latent_embeds = self.latent_projector(projector_hidden).unsqueeze(1).to(device=device)
            step_outputs = self._forward_base_causal(
                inputs_embeds=latent_embeds,
                attention_mask=torch.cat([runtime_cache_key_mask, active_step_key_mask], dim=1).to(
                    dtype=prompt_attention_mask.dtype
                ),
                position_ids=next_position_ids.unsqueeze(1),
                use_cache=True,
                past_key_values=past_key_values,
                output_hidden_states=True,
                return_dict=True,
            )
            if step_outputs.past_key_values is None or step_outputs.hidden_states is None:
                raise RuntimeError("Expected latent rollout forward to return past_key_values and hidden_states")
            past_key_values = step_outputs.past_key_values
            runtime_cache_key_mask = torch.cat([runtime_cache_key_mask, active_step_key_mask], dim=1)
            next_position_ids = next_position_ids + 1
            latent_hidden = step_outputs.hidden_states[-1][:, -1, :]

        forced_end_rows = torch.nonzero((~latent_end_seen) & (~visible_budget_exhausted), as_tuple=False).view(-1)
        for row_idx in forced_end_rows.tolist():
            if len(generated_batch[row_idx]) < max_new_tokens:
                generated_batch[row_idx].append(latent_end_id)
            if len(generated_batch[row_idx]) >= max_new_tokens:
                visible_budget_exhausted[row_idx] = True

        visible_rollout_lengths = torch.where(
            latent_end_seen,
            first_latent_end_step + 1,
            torch.full_like(visible_latent_counts, fill_value=int(latent_rollout_steps)),
        ).clamp(min=0, max=int(latent_rollout_steps))
        latent_visible_mask = (
            torch.arange(latent_rollout_steps, device=device).unsqueeze(0).expand(batch_size, -1)
            < visible_rollout_lengths.unsqueeze(1)
        )
        runtime_cache_key_mask = torch.cat(
            [
                prompt_attention_mask.to(torch.bool),
                torch.ones((batch_size, 1), device=device, dtype=torch.bool),
                latent_visible_mask,
            ],
            dim=1,
        )

        think_rows = torch.nonzero(~visible_budget_exhausted, as_tuple=False).view(-1)
        for row_idx in think_rows.tolist():
            if len(generated_batch[row_idx]) < max_new_tokens:
                generated_batch[row_idx].append(think_start_id)
            if len(generated_batch[row_idx]) >= max_new_tokens:
                visible_budget_exhausted[row_idx] = True

        normal_input_ids = torch.full((batch_size, 1), think_start_id, device=device, dtype=torch.long)
        decode_finished = visible_budget_exhausted.clone()
        while True:
            local_done = bool(decode_finished.all().item())
            if (not lockstep_enabled) and local_done:
                break

            step_key_mask = (~decode_finished).unsqueeze(1)
            step_inputs_embeds = torch.where(
                step_key_mask.unsqueeze(-1),
                embedding_table(normal_input_ids),
                dummy_embed,
            )
            step_outputs = self._forward_base_causal(
                inputs_embeds=step_inputs_embeds,
                attention_mask=torch.cat([runtime_cache_key_mask, step_key_mask], dim=1).to(
                    dtype=prompt_attention_mask.dtype
                ),
                position_ids=next_position_ids.unsqueeze(1),
                use_cache=True,
                past_key_values=past_key_values,
                output_hidden_states=False,
                return_dict=True,
            )
            if step_outputs.past_key_values is not None:
                past_key_values = step_outputs.past_key_values
            runtime_cache_key_mask = torch.cat([runtime_cache_key_mask, step_key_mask], dim=1)
            next_position_ids = next_position_ids + step_key_mask.view(-1).to(torch.long)
            next_tokens = self._sample_tokens_from_logits(
                step_outputs.logits[:, -1, :],
                ~decode_finished,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                fallback_token_id=im_end_id,
            )
            active_rows = torch.nonzero(~decode_finished, as_tuple=False).view(-1)
            for row_idx in active_rows.tolist():
                if len(generated_batch[row_idx]) >= max_new_tokens:
                    decode_finished[row_idx] = True
                    continue
                next_token = int(next_tokens[row_idx].item())
                generated_batch[row_idx].append(next_token)
                if next_token == im_end_id:
                    stopped_normally[row_idx] = True
                    decode_finished[row_idx] = True
                elif len(generated_batch[row_idx]) >= max_new_tokens:
                    decode_finished[row_idx] = True
                else:
                    normal_input_ids[row_idx, 0] = next_token

            local_done = bool(decode_finished.all().item())
            if lockstep_enabled:
                done_tensor = torch.tensor([1 if local_done else 0], device=device, dtype=torch.long)
                dist.all_reduce(done_tensor, op=dist.ReduceOp.SUM)
                if int(done_tensor.item()) >= world_size:
                    break
            elif local_done:
                break

        latent_steps_batch: List[int] = []
        cot_tokens_batch: List[int] = []
        for token_ids in generated_batch:
            latent_token_count = 0
            after_latent = False
            cot_count = 0
            for token_id in token_ids:
                if token_id == latent_start_id:
                    continue
                if not after_latent:
                    if token_id == latent_end_id:
                        after_latent = True
                    else:
                        latent_token_count += 1
                    continue
                if token_id not in {latent_start_id, latent_end_id}:
                    cot_count += 1
            latent_steps_batch.append(int(latent_token_count))
            cot_tokens_batch.append(int(cot_count))

        return LatentGenerationBatchOutput(
            token_ids_batch=[list(token_ids) for token_ids in generated_batch],
            latent_steps_batch=latent_steps_batch,
            cot_tokens_batch=cot_tokens_batch,
            stopped_normally_batch=[bool(flag) for flag in stopped_normally.detach().cpu().tolist()],
            latent_rollout_steps=int(latent_rollout_steps),
        )

    @torch.no_grad()
    def generate_with_latent(
        self,
        prompt_ids: List[int],
        token_constants: Dict[str, int],
        max_new_tokens: int,
        latent_max_steps: int,
        distributed_lockstep: bool = False,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> LatentGenerationOutput:
        batch_output = self._generate_with_latent_batched(
            prompt_ids_batch=[list(prompt_ids)],
            token_constants=token_constants,
            max_new_tokens=max_new_tokens,
            latent_max_steps=latent_max_steps,
            distributed_lockstep=distributed_lockstep,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        return LatentGenerationOutput(
            token_ids=list(batch_output.token_ids_batch[0]),
            latent_steps=int(batch_output.latent_steps_batch[0]),
            cot_tokens=int(batch_output.cot_tokens_batch[0]),
            stopped_normally=bool(batch_output.stopped_normally_batch[0]),
            latent_rollout_steps=int(batch_output.latent_rollout_steps),
        )


def register_latent_qwen3_with_transformers() -> Dict[str, str]:
    status: Dict[str, str] = {}
    try:
        AutoConfig.register(LatentQwenConfig.model_type, LatentQwenConfig)
        status["config"] = "registered"
    except ValueError as exc:
        if "already used" not in str(exc):
            raise
        status["config"] = "already_registered"

    try:
        AutoModelForCausalLM.register(LatentQwenConfig, LatentQwenForCausalLM)
        status["causal_lm"] = "registered"
    except ValueError as exc:
        if "already used" not in str(exc):
            raise
        status["causal_lm"] = "already_registered"
    return status


def read_model_type_from_config_dir(model_path: str | Path) -> Optional[str]:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    model_type = payload.get("model_type")
    return str(model_type) if model_type is not None else None
