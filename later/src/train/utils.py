from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import yaml
from transformers import AutoTokenizer

from later.src.utils.utils import ensure_latent_think_special_tokens, validate_latent_think_tokenizer_contract


ASSISTANT_PREFIX = "<|im_start|>assistant\n"
USER_PREFIX = "<|im_start|>user\n"
IM_END = "<|im_end|>\n"


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def render_manual_chat(messages: Sequence[Dict[str, str]]) -> str:
    parts: List[str] = []
    for message in messages:
        role = str(message["role"])
        content = str(message["content"])
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(parts)


def render_student_messages(user_content: str, assistant_content: str) -> str:
    return f"{USER_PREFIX}{user_content}<|im_end|>\n{ASSISTANT_PREFIX}{assistant_content}<|im_end|>\n"


def render_prompt_only(user_content: str) -> str:
    return f"{USER_PREFIX}{user_content}<|im_end|>\n{ASSISTANT_PREFIX}"

def build_messages_user_content_only(question: str, task):
    if task in ["gsm8k", "aime2024", "aime2025", "math500", "prosqa"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the provided **Target Question**.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    elif task in ["arc_easy", "arc_challenge", "gpqa", "medqa", "commonsense_qa"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the provided **Target Question**.
Your final answer must be selected from A,B,C,D... For example \\boxed{{A}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    elif task in ["mbppplus", "humanevalplus"]:
        user_content = f"""
Target Question: {question}

You must put all python code as self-contained Python function(s) in markdown code blocks. For example:
```python
import math
def add(a, b):
    return a + b
```
Do not add any other contents inside the markdown code block.
Now, reason step by step and output the final answer:
"""

    elif task in ["winogrande"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the provided **Target Question**.
Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    else:
        user_content = f"""
Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the question.
Present your reasoning, and then clearly state your final answer at the end.
"""
    return user_content

def find_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> int:
    if not subsequence:
        return -1
    limit = len(sequence) - len(subsequence) + 1
    for idx in range(max(limit, 0)):
        if list(sequence[idx : idx + len(subsequence)]) == list(subsequence):
            return idx
    return -1


def last_non_pad_index(mask: torch.Tensor) -> int:
    indices = torch.nonzero(mask, as_tuple=False)
    return int(indices[-1].item()) if len(indices) else -1


def maybe_slice_attention_mask(attention_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
    if attention_mask.dim() == 2:
        return attention_mask[:, :seq_len]
    if attention_mask.dim() == 4:
        return attention_mask[:, :, :seq_len, :seq_len]
    raise ValueError(f"Unsupported attention mask rank: {attention_mask.dim()}")


def build_position_ids(valid_token_mask: torch.Tensor) -> torch.Tensor:
    running = torch.cumsum(valid_token_mask.to(torch.long), dim=0) - 1
    running = torch.clamp(running, min=0)
    return running


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def extract_answer_region(text: str) -> str:
    text = text.strip()
    if "</think>" in text:
        return text.split("</think>", 1)[-1].strip()
    return text


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def chunked(iterable: Iterable[Any], chunk_size: int) -> Iterable[List[Any]]:
    chunk: List[Any] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def resolve_context_parallel_padding_multiple(config: Dict[str, Any]) -> int:
    cp_size = max(int(config.get("context_parallel_size", 1) or 1), 1)
    if cp_size <= 1:
        return 1
    padding_factor = max(int(config.get("context_parallel_padding_factor", 2) or 2), 1)
    return int(cp_size * padding_factor)


def resolve_hf_causal_base_model(model: Any) -> Any:
    prefix = getattr(model, "base_model_prefix", "model")
    candidate = getattr(model, prefix, None)
    if candidate is None:
        candidate = getattr(model, "model", None)
    if candidate is None:
        raise ValueError("Model does not expose a callable base model for hidden-state extraction.")
    return candidate


def compute_topk_from_hidden(
    hidden_states: torch.Tensor,
    lm_head: Any,
    topk: int,
    temperature: float,
    projection_chunk_size: int,
    prob_dtype: np.dtype[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rows = int(hidden_states.size(0))
    if rows <= 0:
        return (
            np.zeros((0, topk), dtype=np.int32),
            np.zeros((0, topk), dtype=prob_dtype),
            np.zeros((0,), dtype=prob_dtype),
            0.0,
        )

    ids_chunks: List[np.ndarray] = []
    probs_chunks: List[np.ndarray] = []
    tail_chunks: List[np.ndarray] = []
    captured_mass_sum = 0.0
    chunk_size = max(int(projection_chunk_size), 1)
    temperature = float(max(temperature, 1.0e-8))

    for start in range(0, rows, chunk_size):
        end = min(start + chunk_size, rows)
        logits = lm_head(hidden_states[start:end]).float() / temperature
        vocab_size = int(logits.size(-1))
        keep_topk = min(int(topk), vocab_size)
        topk_logits, topk_ids = torch.topk(logits, k=keep_topk, dim=-1)
        log_denom = torch.logsumexp(logits, dim=-1, keepdim=True)
        topk_probs = (topk_logits - log_denom).exp()
        tail = torch.clamp(1.0 - topk_probs.sum(dim=-1), min=0.0)
        captured_mass_sum += float(topk_probs.sum().detach().item())

        ids_np = topk_ids.detach().cpu().to(torch.int32).numpy()
        probs_np = topk_probs.detach().cpu().numpy().astype(prob_dtype, copy=False)
        tail_np = tail.detach().cpu().numpy().astype(prob_dtype, copy=False)

        if keep_topk < int(topk):
            ids_pad = np.zeros((ids_np.shape[0], int(topk)), dtype=np.int32)
            probs_pad = np.zeros((probs_np.shape[0], int(topk)), dtype=prob_dtype)
            ids_pad[:, :keep_topk] = ids_np
            probs_pad[:, :keep_topk] = probs_np
            ids_np = ids_pad
            probs_np = probs_pad

        ids_chunks.append(ids_np)
        probs_chunks.append(probs_np)
        tail_chunks.append(tail_np)

    return (
        np.concatenate(ids_chunks, axis=0),
        np.concatenate(probs_chunks, axis=0),
        np.concatenate(tail_chunks, axis=0),
        float(captured_mass_sum),
    )


def build_sparse_supervision_from_hidden(
    hidden_states: torch.Tensor,
    lm_head: Any,
    labels: torch.Tensor | None,
    loss_source_positions: torch.Tensor | None,
    loss_target_positions: torch.Tensor | None,
    loss_pair_mask: torch.Tensor | None,
    teacher_kl_source_positions: torch.Tensor | None,
    teacher_kl_pair_mask: torch.Tensor | None,
    teacher_kl_topk_ids: torch.Tensor | None,
    supervised_logits_chunk_size: int,
    kl_temperature: float,
) -> Dict[str, torch.Tensor | None]:
    batch_size, seq_len, _ = hidden_states.shape
    device = hidden_states.device
    projection_chunk_size = max(int(supervised_logits_chunk_size) if int(supervised_logits_chunk_size) > 0 else 1, 1)
    kl_temperature = float(max(kl_temperature, 1.0e-8))

    loss_target_logits = None
    loss_log_denom = None
    if loss_target_positions is not None and loss_pair_mask is not None:
        loss_slots = int(loss_target_positions.size(1))
        loss_target_logits = hidden_states.new_zeros((batch_size, loss_slots), dtype=torch.float32)
        loss_log_denom = hidden_states.new_zeros((batch_size, loss_slots), dtype=torch.float32)
        if (
            labels is not None
            and loss_source_positions is not None
            and int(loss_slots) > 0
        ):
            row_ids = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(loss_source_positions)
            valid = loss_pair_mask & (loss_source_positions >= 0) & (loss_target_positions >= 0)
            safe_source = loss_source_positions.clamp(min=0, max=max(seq_len - 1, 0))
            safe_target = loss_target_positions.clamp(min=0, max=max(seq_len - 1, 0))
            gathered_labels = labels[row_ids, safe_target]
            keep = valid & (gathered_labels != -100)
            if bool(keep.any().item()):
                kept_rows = row_ids[keep]
                kept_cols = torch.arange(loss_slots, device=device).unsqueeze(0).expand(batch_size, loss_slots)[keep]
                kept_source = safe_source[keep]
                kept_target_ids = gathered_labels[keep].to(torch.long)
                kept_hidden = hidden_states[kept_rows, kept_source, :]
                for start in range(0, int(kept_hidden.size(0)), projection_chunk_size):
                    end = min(start + projection_chunk_size, int(kept_hidden.size(0)))
                    chunk_logits = lm_head(kept_hidden[start:end]).float()
                    selected = torch.gather(
                        chunk_logits,
                        dim=-1,
                        index=kept_target_ids[start:end].clamp(min=0).unsqueeze(-1),
                    ).squeeze(-1)
                    log_denom = torch.logsumexp(chunk_logits, dim=-1)
                    loss_target_logits[kept_rows[start:end], kept_cols[start:end]] = selected
                    loss_log_denom[kept_rows[start:end], kept_cols[start:end]] = log_denom

    teacher_kl_topk_logits = None
    teacher_kl_log_denom = None
    if teacher_kl_pair_mask is not None:
        kl_slots = int(teacher_kl_pair_mask.size(1))
        teacher_kl_log_denom = hidden_states.new_zeros((batch_size, kl_slots), dtype=torch.float32)
        if teacher_kl_topk_ids is not None:
            teacher_kl_topk_logits = hidden_states.new_zeros(
                (batch_size, kl_slots, int(teacher_kl_topk_ids.size(-1))),
                dtype=torch.float32,
            )
        if (
            teacher_kl_source_positions is not None
            and teacher_kl_topk_ids is not None
            and int(kl_slots) > 0
        ):
            row_ids = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(teacher_kl_source_positions)
            valid = teacher_kl_pair_mask & (teacher_kl_source_positions >= 0)
            if bool(valid.any().item()):
                kept_rows = row_ids[valid]
                kept_cols = torch.arange(kl_slots, device=device).unsqueeze(0).expand(batch_size, kl_slots)[valid]
                safe_source = teacher_kl_source_positions.clamp(min=0, max=max(seq_len - 1, 0))
                kept_source = safe_source[valid]
                kept_topk_ids = teacher_kl_topk_ids[kept_rows, kept_cols].to(device=device, dtype=torch.long)
                kept_hidden = hidden_states[kept_rows, kept_source, :]
                for start in range(0, int(kept_hidden.size(0)), projection_chunk_size):
                    end = min(start + projection_chunk_size, int(kept_hidden.size(0)))
                    chunk_logits = lm_head(kept_hidden[start:end]).float() / kl_temperature
                    selected = torch.gather(chunk_logits, dim=-1, index=kept_topk_ids[start:end])
                    log_denom = torch.logsumexp(chunk_logits, dim=-1)
                    teacher_kl_topk_logits[kept_rows[start:end], kept_cols[start:end]] = selected
                    teacher_kl_log_denom[kept_rows[start:end], kept_cols[start:end]] = log_denom

    return {
        "loss_target_logits": loss_target_logits,
        "loss_log_denom": loss_log_denom,
        "teacher_kl_topk_logits": teacher_kl_topk_logits,
        "teacher_kl_log_denom": teacher_kl_log_denom,
    }


def token_ids_to_list(token_ids: Any) -> List[int]:
    if isinstance(token_ids, torch.Tensor):
        return [int(x) for x in token_ids.detach().cpu().tolist()]
    if isinstance(token_ids, list):
        return [int(x) for x in token_ids]
    try:
        return [int(x) for x in list(token_ids)]
    except Exception:
        return []



def token_repr(tokenizer: Any, token_id: int) -> str:
    try:
        tokens = tokenizer.convert_ids_to_tokens([int(token_id)])
        if isinstance(tokens, list) and tokens:
            return str(tokens[0])
    except Exception:
        pass
    try:
        return str(tokenizer.decode([int(token_id)], skip_special_tokens=False))
    except Exception:
        return str(int(token_id))


def _format_visible_token_text(text: str) -> str:
    if text == "":
        return "<EMPTY>"
    if text == " ":
        return "<SP>"
    if text == "\n":
        return "<NL>"
    if text == "\t":
        return "<TAB>"

    has_control = any((ord(ch) < 32 and ch not in {"\n", "\t"}) or ord(ch) == 127 for ch in text)
    if has_control:
        return text.encode("unicode_escape").decode("ascii")
    return text


def _format_generated_token_piece(tokenizer: Any, token_id: int) -> str:
    raw_token = token_repr(tokenizer, token_id)
    try:
        decoded = str(tokenizer.decode([int(token_id)], skip_special_tokens=False))
    except Exception:
        decoded = ""

    if raw_token.startswith("<|") and raw_token.endswith("|>"):
        return raw_token
    if raw_token.startswith("<") and raw_token.endswith(">") and decoded == raw_token:
        return raw_token

    formatted = _format_visible_token_text(decoded)
    if formatted != decoded or decoded == "":
        return formatted
    return decoded


def format_latent_generated_text(tokenizer: Any, token_ids: Any, token_constants: Dict[str, int], latent_steps: int) -> str:
    ids = token_ids_to_list(token_ids)
    latent_start_id = int(token_constants["latent_start_id"])
    latent_end_id = int(token_constants["latent_end_id"])

    latent_started = False
    latent_closed = False
    latent_parts: List[str] = []
    cot_parts: List[str] = []
    latent_budget = max(int(latent_steps), 0)
    assigned_latent = 0

    try:
        latent_start_idx = ids.index(latent_start_id)
        latent_started = True
    except ValueError:
        latent_start_idx = -1

    def _append_piece(token_id: int, to_latent: bool) -> None:
        piece = _format_generated_token_piece(tokenizer, token_id)
        if to_latent:
            latent_parts.append(piece)
        else:
            cot_parts.append(piece)

    for idx, token_id in enumerate(ids):
        if token_id == latent_start_id:
            latent_started = True
            continue
        if token_id == latent_end_id:
            latent_closed = True
            continue

        # Use latent_steps as the authoritative split budget after <latent_think>.
        in_latent_region = latent_started and idx > latent_start_idx and assigned_latent < latent_budget
        if in_latent_region:
            _append_piece(token_id, to_latent=True)
            assigned_latent += 1
        else:
            _append_piece(token_id, to_latent=False)

    rendered_latent = "".join(latent_parts) if latent_parts else "<EMPTY>"
    rendered_cot = "".join(cot_parts) if cot_parts else "<EMPTY>"

    sections: List[str] = []
    if latent_started:
        sections.append("[LATENT_START]")
        sections.append(f"[LATENT] {rendered_latent}")
        if latent_closed:
            sections.append("[LATENT_END]")
    sections.append(f"[COT] {rendered_cot}")
    print(sections)
    return "\n".join(sections)


def decode_token_ids_for_debug(
    tokenizer: Any,
    token_ids: Any,
    positions: Optional[Sequence[int] | torch.Tensor] = None,
    skip_special_tokens: bool = False,
) -> Dict[str, Any]:
    ids = token_ids_to_list(token_ids)
    selected_positions: List[int] = []
    selected_ids: List[int] = []

    if positions is None:
        selected_ids = ids
    else:
        if isinstance(positions, torch.Tensor):
            raw_positions = [int(x) for x in positions.detach().cpu().tolist()]
        else:
            raw_positions = [int(x) for x in positions]
        for pos in raw_positions:
            if 0 <= pos < len(ids):
                selected_positions.append(int(pos))
                selected_ids.append(int(ids[pos]))

    text = ""
    if tokenizer is not None and selected_ids:
        text = str(tokenizer.decode(selected_ids, skip_special_tokens=skip_special_tokens))

    return {
        "positions": selected_positions,
        "token_ids": selected_ids,
        "text": text,
    }


_DEBUG_TOKENIZER_CACHE: Dict[str, Any] = {}


@dataclass
class BaseTokenRowFreezeController:
    base_vocab_size: int
    new_token_start: int
    new_token_end: int
    apply_to_lm_head: bool
    freeze_scope: str
    uses_tied_word_embeddings: bool
    input_weight_handle: Any | None
    lm_head_weight_handle: Any | None
    lm_head_bias_handle: Any | None
    input_weight_snapshot: torch.Tensor
    lm_head_weight_snapshot: torch.Tensor | None
    lm_head_bias_snapshot: torch.Tensor | None
    input_weight_decay_exempt: bool = False
    lm_head_weight_decay_exempt: bool = False
    lm_head_bias_weight_decay_exempt: bool = False
    _runtime_model: Any | None = None
    _runtime_accelerator: Any | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.base_vocab_size > 0 and self.new_token_end > self.new_token_start)

    @property
    def frozen_row_count(self) -> int:
        return int(self.base_vocab_size)

    @property
    def new_token_count(self) -> int:
        return max(int(self.new_token_end - self.new_token_start), 0)

    @property
    def hook_count(self) -> int:
        return int(
            sum(
                handle is not None
                for handle in (
                    self.input_weight_handle,
                    self.lm_head_weight_handle,
                    self.lm_head_bias_handle,
                )
            )
        )

    def bind_runtime(self, model: Any, accelerator: Any | None = None) -> None:
        self._runtime_model = model
        self._runtime_accelerator = accelerator

    def weight_decay_exempt_parameter_ids(self, model: Any | None = None) -> set[int]:
        target_model = model if model is not None else self._runtime_model
        if target_model is None:
            return set()
        unwrapped = self._unwrap_model(target_model)
        param_ids: set[int] = set()

        input_embeddings = unwrapped.get_input_embeddings() if hasattr(unwrapped, 'get_input_embeddings') else None
        if input_embeddings is not None and hasattr(input_embeddings, 'weight'):
            param_ids.add(id(input_embeddings.weight))

        if self.apply_to_lm_head:
            lm_head = getattr(unwrapped, 'lm_head', None)
            if lm_head is not None and hasattr(lm_head, 'weight'):
                param_ids.add(id(lm_head.weight))
            lm_head_bias = getattr(lm_head, 'bias', None) if lm_head is not None else None
            if lm_head_bias is not None:
                param_ids.add(id(lm_head_bias))
        return param_ids

    def sync_weight_decay_exemptions(self, optimizer: torch.optim.Optimizer) -> None:
        if optimizer is None:
            return
        target_model = self._runtime_model
        param_ids = self.weight_decay_exempt_parameter_ids(target_model)
        if not param_ids:
            return
        self.input_weight_decay_exempt = False
        self.lm_head_weight_decay_exempt = False
        self.lm_head_bias_weight_decay_exempt = False

        unwrapped = self._unwrap_model(target_model)
        input_embeddings = unwrapped.get_input_embeddings() if hasattr(unwrapped, 'get_input_embeddings') else None
        input_weight = getattr(input_embeddings, 'weight', None)
        lm_head = getattr(unwrapped, 'lm_head', None)
        lm_head_weight = getattr(lm_head, 'weight', None)
        lm_head_bias = getattr(lm_head, 'bias', None)

        for group in optimizer.param_groups:
            weight_decay = float(group.get('weight_decay', 0.0) or 0.0)
            if weight_decay != 0.0:
                continue
            for param in group.get('params', []):
                param_id = id(param)
                if input_weight is not None and param_id == id(input_weight):
                    self.input_weight_decay_exempt = True
                if lm_head_weight is not None and param_id == id(lm_head_weight):
                    self.lm_head_weight_decay_exempt = True
                if lm_head_bias is not None and param_id == id(lm_head_bias):
                    self.lm_head_bias_weight_decay_exempt = True

    def restore_frozen_rows(self) -> bool:
        if not self.enabled:
            return False
        target_model = self._runtime_model
        if target_model is None:
            return False

        restored = False
        unwrapped = self._unwrap_model(target_model)
        input_embeddings = unwrapped.get_input_embeddings() if hasattr(unwrapped, 'get_input_embeddings') else None
        if input_embeddings is None or not hasattr(input_embeddings, 'weight'):
            return False

        input_weight = input_embeddings.weight
        if not self.input_weight_decay_exempt:
            restored = self._restore_param_prefix(input_weight, self.input_weight_snapshot) or restored

        if self.apply_to_lm_head:
            lm_head = getattr(unwrapped, 'lm_head', None)
            if lm_head is not None and hasattr(lm_head, 'weight') and (not self.uses_tied_word_embeddings):
                if (not self.lm_head_weight_decay_exempt) and self.lm_head_weight_snapshot is not None:
                    restored = self._restore_param_prefix(lm_head.weight, self.lm_head_weight_snapshot) or restored
            lm_head_bias = getattr(lm_head, 'bias', None) if lm_head is not None else None
            if (
                lm_head_bias is not None
                and self.lm_head_bias_snapshot is not None
                and (not self.lm_head_bias_weight_decay_exempt)
            ):
                restored = self._restore_param_prefix(lm_head_bias, self.lm_head_bias_snapshot) or restored
        return restored

    def _unwrap_model(self, model: Any) -> Any:
        if self._runtime_accelerator is None:
            return model
        try:
            return self._runtime_accelerator.unwrap_model(model)
        except Exception:
            return model

    def _restore_param_prefix(self, param: torch.Tensor, snapshot: torch.Tensor) -> bool:
        if param is None or snapshot is None or int(param.size(0)) < int(self.base_vocab_size):
            return False
        with torch.no_grad():
            target_prefix = snapshot.to(device=param.device, dtype=param.dtype)
            param[: self.base_vocab_size].copy_(target_prefix)
        return True


def _zero_frozen_rows_hook(base_vocab_size: int):
    frozen_rows = max(int(base_vocab_size), 0)

    def _hook(grad: torch.Tensor | None) -> torch.Tensor | None:
        if grad is None or frozen_rows <= 0 or grad.dim() <= 0:
            return grad
        limit = min(frozen_rows, int(grad.size(0)))
        if limit <= 0:
            return grad
        grad = grad.clone()
        grad.narrow(0, 0, limit).zero_()
        return grad

    return _hook


def configure_base_token_row_freezing(
    model: Any,
    base_vocab_size: int,
    apply_to_lm_head: bool = True,
    freeze_scope: str = 'always',
) -> BaseTokenRowFreezeController:
    if model is None:
        raise ValueError('model is required')
    base_vocab_size = int(base_vocab_size)
    if base_vocab_size <= 0:
        raise ValueError(f'base_vocab_size must be positive, got {base_vocab_size}')

    input_embeddings = model.get_input_embeddings() if hasattr(model, 'get_input_embeddings') else None
    if input_embeddings is None or not hasattr(input_embeddings, 'weight'):
        raise ValueError('model must expose get_input_embeddings().weight')
    input_weight = input_embeddings.weight
    total_vocab_size = int(input_weight.size(0))
    if base_vocab_size >= total_vocab_size:
        raise ValueError(
            'Row freezing requires added tokenizer rows. '
            f'Got base_vocab_size={base_vocab_size}, total_vocab_size={total_vocab_size}.'
        )

    lm_head = getattr(model, 'lm_head', None)
    apply_to_lm_head = bool(apply_to_lm_head and lm_head is not None and hasattr(lm_head, 'weight'))
    uses_tied_word_embeddings = bool(apply_to_lm_head and getattr(lm_head, 'weight', None) is input_weight)

    input_handle = input_weight.register_hook(_zero_frozen_rows_hook(base_vocab_size))
    lm_head_handle = None
    lm_head_bias_handle = None
    lm_head_weight_snapshot = None
    lm_head_bias_snapshot = None

    if apply_to_lm_head and (not uses_tied_word_embeddings):
        lm_head_handle = lm_head.weight.register_hook(_zero_frozen_rows_hook(base_vocab_size))
        lm_head_weight_snapshot = lm_head.weight[:base_vocab_size].detach().cpu().clone()
    if apply_to_lm_head and getattr(lm_head, 'bias', None) is not None:
        lm_head_bias_handle = lm_head.bias.register_hook(_zero_frozen_rows_hook(base_vocab_size))
        lm_head_bias_snapshot = lm_head.bias[:base_vocab_size].detach().cpu().clone()

    controller = BaseTokenRowFreezeController(
        base_vocab_size=base_vocab_size,
        new_token_start=base_vocab_size,
        new_token_end=total_vocab_size,
        apply_to_lm_head=apply_to_lm_head,
        freeze_scope=str(freeze_scope),
        uses_tied_word_embeddings=uses_tied_word_embeddings,
        input_weight_handle=input_handle,
        lm_head_weight_handle=lm_head_handle,
        lm_head_bias_handle=lm_head_bias_handle,
        input_weight_snapshot=input_weight[:base_vocab_size].detach().cpu().clone(),
        lm_head_weight_snapshot=lm_head_weight_snapshot,
        lm_head_bias_snapshot=lm_head_bias_snapshot,
    )
    controller.bind_runtime(model=model, accelerator=None)
    return controller


def _get_debug_tokenizer(model_name_or_path: str) -> Any | None:
    key = str(model_name_or_path).strip()
    if not key:
        return None
    if key in _DEBUG_TOKENIZER_CACHE:
        return _DEBUG_TOKENIZER_CACHE[key]
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            key,
            trust_remote_code=True,
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = "<|endoftext|>"
        _DEBUG_TOKENIZER_CACHE[key] = tokenizer
        return tokenizer
    except Exception:
        return None


def decode_stage3_inputs_with_auto_tokenizer(
    model_name_or_path: str,
    input_ids: torch.Tensor,
    stage3_start: int,
    decode_scope: str = "stage3_start",
    max_rows: int = 1,
    max_tokens: int = 64,
    skip_special_tokens: bool = False,
) -> List[Dict[str, Any]]:
    scope = str(decode_scope).strip().lower()
    if scope == "batch":
        return decode_input_ids_slice_with_auto_tokenizer(
            model_name_or_path=model_name_or_path,
            input_ids=input_ids,
            slice_start=0,
            slice_end=None,
            max_rows=-1,
            skip_special_tokens=skip_special_tokens,
        )
    if scope == "stage3_all":
        return decode_input_ids_slice_with_auto_tokenizer(
            model_name_or_path=model_name_or_path,
            input_ids=input_ids,
            slice_start=int(stage3_start),
            slice_end=int(stage3_start) + int(max_tokens),
            max_rows=max_rows,
            skip_special_tokens=skip_special_tokens,
        )
    return decode_input_ids_slice_with_auto_tokenizer(
        model_name_or_path=model_name_or_path,
        input_ids=input_ids,
        slice_start=int(stage3_start),
        slice_end=int(stage3_start) + 1,
        max_rows=max_rows,
        skip_special_tokens=skip_special_tokens,
    )


def decode_input_ids_slice_with_auto_tokenizer(
    model_name_or_path: str,
    input_ids: torch.Tensor,
    slice_start: int = 0,
    slice_end: Optional[int] = None,
    max_rows: int = 1,
    skip_special_tokens: bool = False,
) -> List[Dict[str, Any]]:
    tokenizer = _get_debug_tokenizer(model_name_or_path)
    if tokenizer is None or input_ids.dim() != 2:
        return []

    batch_size = int(input_ids.size(0))
    seq_len = int(input_ids.size(1))
    row_count = batch_size if int(max_rows) <= 0 else min(batch_size, max(int(max_rows), 1))
    start_pos = max(int(slice_start), 0)
    end_pos = seq_len if slice_end is None else min(max(int(slice_end), start_pos), seq_len)
    positions = list(range(start_pos, end_pos))

    payloads: List[Dict[str, Any]] = []
    for row_idx in range(row_count):
        decoded = decode_token_ids_for_debug(
            tokenizer=tokenizer,
            token_ids=input_ids[row_idx],
            positions=positions,
            skip_special_tokens=skip_special_tokens,
        )
        payloads.append(
            {
                "row": int(row_idx),
                "slice_start": int(start_pos),
                "slice_end": int(end_pos),
                "positions": decoded["positions"],
                "token_ids": decoded["token_ids"],
                "text": decoded["text"],
            }
        )
    return payloads


@dataclass
class TeacherShardHandle:
    ids: np.memmap
    probs: np.memmap
    tail: np.memmap


class PrecomputedTeacherCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Teacher cache directory not found: {self.root}")
        index_path = self.root / "index.pt"
        if not index_path.exists():
            raise FileNotFoundError(f"Teacher cache index missing: {index_path}")
        index = torch.load(index_path, map_location="cpu")
        self.record_to_meta = {
            record_id: (
                int(index["shard_ids"][i].item()),
                int(index["row_starts"][i].item()),
                int(index["row_counts"][i].item()),
            )
            for i, record_id in enumerate(index["record_ids"])
        }
        self.handles: Dict[int, TeacherShardHandle] = {}

    def _load_shard(self, shard_id: int) -> TeacherShardHandle:
        if shard_id in self.handles:
            return self.handles[shard_id]
        rank = shard_id // 100000
        shard = shard_id % 100000
        base = self.root / f"rank_{rank:05d}_shard_{shard:05d}"
        handle = TeacherShardHandle(
            ids=np.load(str(base) + "_ids.npy", mmap_mode="r"),
            probs=np.load(str(base) + "_probs.npy", mmap_mode="r"),
            tail=np.load(str(base) + "_tail.npy", mmap_mode="r"),
        )
        self.handles[shard_id] = handle
        return handle

    def get(self, record_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if record_id not in self.record_to_meta:
            raise KeyError(f"Missing teacher cache entry for record_id={record_id}")
        shard_id, row_start, row_count = self.record_to_meta[record_id]
        shard = self._load_shard(shard_id)
        row_end = row_start + row_count
        return (
            np.asarray(shard.ids[row_start:row_end]),
            np.asarray(shard.probs[row_start:row_end]),
            np.asarray(shard.tail[row_start:row_end]),
        )

    def get_optional(self, record_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if record_id not in self.record_to_meta:
            return None
        return self.get(record_id)

    def build_batch_tensors(
        self,
        record_ids: Sequence[str],
        pair_mask: torch.Tensor | None,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        batch_size = len(record_ids)
        if pair_mask is not None and pair_mask.dim() != 2:
            raise ValueError("pair_mask must be rank-2 when provided")

        max_pairs = int(pair_mask.size(1)) if pair_mask is not None else 0
        teacher_topk = 0
        teacher_entries: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        teacher_lengths: List[int] = []
        expected_lengths: List[int] = []
        mismatch_count = 0
        missing_record_ids: List[str] = []

        for row_idx, record_id in enumerate(record_ids):
            maybe_entry = self.get_optional(record_id)
            if maybe_entry is None:
                missing_record_ids.append(str(record_id))
                ids = np.zeros((0, 0), dtype=np.int32)
                probs = np.zeros((0, 0), dtype=np.float32)
                tail = np.zeros((0,), dtype=np.float32)
            else:
                ids, probs, tail = maybe_entry
            teacher_entries.append((ids, probs, tail))
            teacher_len = int(ids.shape[0]) if ids.ndim >= 1 else 0
            teacher_lengths.append(teacher_len)
            teacher_topk = max(teacher_topk, int(ids.shape[1]) if ids.ndim == 2 else 0)
            expected_len = int(pair_mask[row_idx].sum().item()) if pair_mask is not None else teacher_len
            expected_lengths.append(expected_len)
            if pair_mask is None:
                max_pairs = max(max_pairs, teacher_len)
            if expected_len != teacher_len:
                mismatch_count += 1

        topk_ids = torch.zeros((batch_size, max_pairs, teacher_topk), device=device, dtype=torch.long)
        topk_probs = torch.zeros((batch_size, max_pairs, teacher_topk), device=device, dtype=torch.float32)
        tail_mass = torch.zeros((batch_size, max_pairs), device=device, dtype=torch.float32)
        valid_mask = torch.zeros((batch_size, max_pairs), device=device, dtype=torch.bool)
        teacher_lengths_tensor = torch.tensor(teacher_lengths, device=device, dtype=torch.long)
        expected_lengths_tensor = torch.tensor(expected_lengths, device=device, dtype=torch.long)

        for row_idx, (ids, probs, tail) in enumerate(teacher_entries):
            usable = min(
                int(ids.shape[0]),
                int(expected_lengths_tensor[row_idx].item()),
            )
            if usable <= 0:
                continue
            row_topk = int(ids.shape[1]) if ids.ndim == 2 else 0
            if row_topk > 0:
                topk_ids[row_idx, :usable, :row_topk] = torch.as_tensor(
                    ids[:usable].copy(),
                    device=device,
                    dtype=torch.long,
                )
                topk_probs[row_idx, :usable, :row_topk] = torch.as_tensor(
                    probs[:usable].copy(),
                    device=device,
                    dtype=torch.float32,
                )
            tail_mass[row_idx, :usable] = torch.as_tensor(
                tail[:usable].copy(),
                device=device,
                dtype=torch.float32,
            )
            valid_mask[row_idx, :usable] = True

        return {
            "teacher_kl_topk_ids": topk_ids,
            "teacher_kl_topk_probs": topk_probs,
            "teacher_kl_tail": tail_mass,
            "teacher_kl_effective_mask": valid_mask,
            "teacher_kl_topk": torch.tensor(int(teacher_topk), device=device, dtype=torch.long),
            "teacher_kl_teacher_lengths": teacher_lengths_tensor,
            "teacher_kl_expected_lengths": expected_lengths_tensor,
            "teacher_kl_length_mismatch_count": torch.tensor(int(mismatch_count), device=device, dtype=torch.long),
            "teacher_kl_missing_sample_count": torch.tensor(int(len(missing_record_ids)), device=device, dtype=torch.long),
            "teacher_kl_missing_record_ids_preview": list(missing_record_ids[:8]),
        }


def get_token_constants(tokenizer: Any) -> Dict[str, int]:
    validate_latent_think_tokenizer_contract(tokenizer)
    tokens = {
        "latent_start_id": tokenizer.convert_tokens_to_ids("<latent_think>"),
        "latent_end_id": tokenizer.convert_tokens_to_ids("</latent_think>"),
        "think_start_id": tokenizer.convert_tokens_to_ids("<think>"),
        "think_end_id": tokenizer.convert_tokens_to_ids("</think>"),
        "im_start_id": tokenizer.convert_tokens_to_ids("<|im_start|>"),
        "im_end_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "eot_id": tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    }
    missing = [name for name, token_id in tokens.items() if token_id is None or int(token_id) < 0]
    if missing:
        raise ValueError(f"Tokenizer is missing required special tokens: {missing}")
    return {name: int(value) for name, value in tokens.items()}


def get_early_exit_forbidden_token_ids(token_constants: Dict[str, int]) -> List[int]:
    forbidden_names = (
        "latent_start_id",
        "latent_end_id",
        "think_start_id",
        "think_end_id",
        "im_start_id",
        "im_end_id",
    )
    forbidden_ids: List[int] = []
    seen_ids = set()
    for token_name in forbidden_names:
        token_id = token_constants.get(token_name)
        if token_id is None:
            continue
        token_id = int(token_id)
        if token_id < 0 or token_id in seen_ids:
            continue
        seen_ids.add(token_id)
        forbidden_ids.append(token_id)
    return forbidden_ids


def get_halt_dense_forbidden_token_ids(token_constants: Dict[str, int]) -> List[int]:
    return get_early_exit_forbidden_token_ids(token_constants)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def curriculum_weight(sort_key: Sequence[int | float], progress: float, power: float) -> float:
    difficulty_rank = float(sort_key[0])
    latent_steps = float(sort_key[1])
    cot_tokens = float(sort_key[2])
    ease = 1.0 / (1.0 + difficulty_rank + 0.05 * latent_steps + 0.01 * cot_tokens)
    flat = 1.0
    return (1.0 - progress) * (ease**power) + progress * flat


def mean_or_zero(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    denom = torch.clamp(weights.sum(), min=1.0)
    return (values * weights).sum() / denom


def distributed_max_int(value: int, device: torch.device) -> int:
    tensor = torch.tensor([int(value)], device=device, dtype=torch.long)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def distributed_min_int(value: int, device: torch.device) -> int:
    tensor = torch.tensor([int(value)], device=device, dtype=torch.long)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return int(tensor.item())


def rebuild_position_ids_2d(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.dim() != 2:
        raise ValueError(f"rebuild_position_ids_2d expects rank-2 tensor, got rank={attention_mask.dim()}")
    valid = attention_mask.to(torch.long)
    running = torch.cumsum(valid, dim=1) - 1
    return torch.clamp(running, min=0)


def pad_tensor_to_shape(
    value: torch.Tensor,
    target_shape: Sequence[int],
    pad_value: int | float | bool = 0,
) -> torch.Tensor:
    if value.dim() != len(target_shape):
        raise ValueError(f"pad_tensor_to_shape rank mismatch: value_rank={value.dim()} target_rank={len(target_shape)}")

    result = value
    for dim, target in enumerate(target_shape):
        current = int(result.size(dim))
        if int(target) <= current:
            continue
        pad_shape = list(result.shape)
        pad_shape[dim] = int(target) - current
        pad_tensor = result.new_full(pad_shape, pad_value)
        result = torch.cat([result, pad_tensor], dim=dim)
    return result


def _first_true_index_per_row(mask: torch.Tensor, default_index: int) -> torch.Tensor:
    if mask.dim() != 2:
        raise ValueError(f"Expected rank-2 mask, got rank={mask.dim()}")
    first_indices = torch.full((mask.size(0),), int(default_index), device=mask.device, dtype=torch.long)
    for row_idx in range(mask.size(0)):
        row_true = torch.nonzero(mask[row_idx], as_tuple=False).view(-1)
        if row_true.numel() > 0:
            first_indices[row_idx] = int(row_true[0].item())
    return first_indices


def _pad_1d_with_left_and_middle(
    row: torch.Tensor,
    left_pad: int,
    middle_pad: int,
    middle_insert_at: int,
    pad_value: int | float | bool,
) -> torch.Tensor:
    left_pad = max(int(left_pad), 0)
    middle_pad = max(int(middle_pad), 0)
    middle_insert_at = min(max(int(middle_insert_at), 0), int(row.numel()))

    chunks: List[torch.Tensor] = []
    if left_pad > 0:
        chunks.append(row.new_full((left_pad,), pad_value))
    chunks.append(row[:middle_insert_at])
    if middle_pad > 0:
        chunks.append(row.new_full((middle_pad,), pad_value))
    chunks.append(row[middle_insert_at:])
    return torch.cat(chunks, dim=0)


def _pad_rank_local_sequence_tensors(
    tensor: torch.Tensor,
    left_pads: Sequence[int],
    middle_pads: Sequence[int],
    split_positions: Sequence[int],
    pad_value: int | float | bool,
) -> torch.Tensor:
    if tensor.dim() != 2:
        raise ValueError(f"Expected rank-2 tensor for sequence padding, got rank={tensor.dim()}")
    if tensor.size(0) != len(left_pads):
        raise ValueError("Batch size mismatch while rank-local sequence padding")

    rows: List[torch.Tensor] = []
    max_len = 0
    for row_idx in range(tensor.size(0)):
        row = _pad_1d_with_left_and_middle(
            row=tensor[row_idx],
            left_pad=int(left_pads[row_idx]),
            middle_pad=int(middle_pads[row_idx]),
            middle_insert_at=int(split_positions[row_idx]),
            pad_value=pad_value,
        )
        rows.append(row)
        max_len = max(max_len, int(row.numel()))

    if max_len <= 0:
        return tensor.new_empty((tensor.size(0), 0))

    padded_rows: List[torch.Tensor] = []
    for row in rows:
        if int(row.numel()) < max_len:
            row = torch.cat([row, row.new_full((max_len - int(row.numel()),), pad_value)], dim=0)
        padded_rows.append(row)
    return torch.stack(padded_rows, dim=0)


def _shift_position_tensor_for_rank_alignment(
    value: torch.Tensor,
    left_pads: Sequence[int],
    middle_pads: Sequence[int],
    split_positions: Sequence[int],
    invalid_value: int = -1,
) -> torch.Tensor:
    if value.dim() == 1:
        result = value.clone()
        for row_idx in range(result.size(0)):
            pos = int(result[row_idx].item())
            if pos < 0:
                continue
            shifted = pos + int(left_pads[row_idx])
            if pos >= int(split_positions[row_idx]):
                shifted += int(middle_pads[row_idx])
            result[row_idx] = int(shifted)
        return result

    if value.dim() == 2:
        result = value.clone()
        for row_idx in range(result.size(0)):
            row = result[row_idx]
            valid = row != int(invalid_value)
            if not bool(valid.any().item()):
                continue
            shifted = row[valid] + int(left_pads[row_idx])
            shifted = shifted + (row[valid] >= int(split_positions[row_idx])).to(row.dtype) * int(middle_pads[row_idx])
            row[valid] = shifted
            result[row_idx] = row
        return result

    return value


def _align_rank_local_boundaries_for_sequence(
    synced: Dict[str, Any],
    pad_specs: Dict[str, int | float | bool],
    device: torch.device,
) -> Dict[str, Any]:
    input_ids = synced.get("input_ids")
    cot_mask = synced.get("cot_mask")
    latent_start_positions = synced.get("latent_start_positions")
    if (
        input_ids is None
        or cot_mask is None
        or latent_start_positions is None
        or not torch.is_tensor(input_ids)
        or not torch.is_tensor(cot_mask)
        or not torch.is_tensor(latent_start_positions)
    ):
        return synced

    if input_ids.dim() != 2 or cot_mask.dim() != 2 or latent_start_positions.dim() != 1:
        return synced

    batch_size, local_seq_len = int(input_ids.size(0)), int(input_ids.size(1))
    if batch_size <= 0 or local_seq_len <= 0:
        return synced

    think_starts = _first_true_index_per_row(cot_mask.to(torch.bool), default_index=local_seq_len)
    local_latent_start_max = int(latent_start_positions.max().item()) if latent_start_positions.numel() > 0 else 0
    local_delta_max = int((think_starts - latent_start_positions.to(torch.long)).max().item())

    latent_tensor = torch.tensor([local_latent_start_max], device=device, dtype=torch.long)
    delta_tensor = torch.tensor([local_delta_max], device=device, dtype=torch.long)
    dist.all_reduce(latent_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(delta_tensor, op=dist.ReduceOp.MAX)

    aligned_latent_start = int(latent_tensor.item())
    aligned_think_start = int(latent_tensor.item() + delta_tensor.item())

    left_pads = []
    middle_pads = []
    split_positions = []
    for row_idx in range(batch_size):
        local_latent = int(latent_start_positions[row_idx].item())
        local_think = int(think_starts[row_idx].item())
        left_pad = max(aligned_latent_start - local_latent, 0)
        middle_pad = max(aligned_think_start - (local_think + left_pad), 0)
        left_pads.append(left_pad)
        middle_pads.append(middle_pad)
        split_positions.append(local_think)

    sequence_keys = []
    for key, value in synced.items():
        if (
            key in pad_specs
            and torch.is_tensor(value)
            and value.dim() == 2
            and int(value.size(0)) == batch_size
            and int(value.size(1)) == local_seq_len
        ):
            sequence_keys.append(key)

    for key in sequence_keys:
        synced[key] = _pad_rank_local_sequence_tensors(
            tensor=synced[key],
            left_pads=left_pads,
            middle_pads=middle_pads,
            split_positions=split_positions,
            pad_value=pad_specs[key],
        )

    for key in [
        "latent_start_positions",
        "latent_end_positions",
        "teacher_target_start",
        "latent_positions",
        "loss_source_positions",
        "loss_target_positions",
        "teacher_kl_source_positions",
        "teacher_kl_target_positions",
    ]:
        value = synced.get(key)
        if value is None or not torch.is_tensor(value):
            continue
        if value.dim() not in {1, 2}:
            continue
        invalid = (
            -1 if key.endswith("positions") and key not in {"latent_start_positions", "latent_end_positions"} else -1
        )
        synced[key] = _shift_position_tensor_for_rank_alignment(
            value=value,
            left_pads=left_pads,
            middle_pads=middle_pads,
            split_positions=split_positions,
            invalid_value=invalid,
        )

    spans = synced.get("spans")
    if isinstance(spans, list) and len(spans) == batch_size:
        for row_idx, span in enumerate(spans):
            if not isinstance(span, dict):
                continue
            left = int(left_pads[row_idx])
            middle = int(middle_pads[row_idx])
            split = int(split_positions[row_idx])
            for pos_key in [
                "assistant_prefix_start",
                "assistant_content_start",
                "latent_start",
                "latent_end",
                "think_start",
                "think_end",
                "answer_start",
                "im_end",
            ]:
                if pos_key not in span:
                    continue
                pos = int(span[pos_key])
                shifted = pos + left
                if pos >= split:
                    shifted += middle
                span[pos_key] = int(shifted)

    return synced


def sync_batch_across_ranks(
    batch: Dict[str, Any],
    pad_specs: Dict[str, int | float | bool],
    device: torch.device,
) -> Dict[str, Any]:
    if not (dist.is_available() and dist.is_initialized()):
        return batch

    synced = _align_rank_local_boundaries_for_sequence(dict(batch), pad_specs=pad_specs, device=device)
    for key, pad_value in pad_specs.items():
        value = synced.get(key)
        if value is None:
            continue
        if not torch.is_tensor(value):
            continue

        local_rank_tensor = torch.tensor([int(value.dim())], device=device, dtype=torch.long)
        dist.all_reduce(local_rank_tensor, op=dist.ReduceOp.MAX)
        global_rank = int(local_rank_tensor.item())
        if global_rank <= 0:
            continue
        if value.dim() != global_rank:
            raise ValueError(
                f"sync_batch_across_ranks expects key={key} to have same rank on all ranks, "
                f"local_rank={value.dim()}, global_rank={global_rank}"
            )

        local_shape = torch.tensor(list(value.shape), device=device, dtype=torch.long)
        dist.all_reduce(local_shape, op=dist.ReduceOp.MAX)
        global_shape = [int(v) for v in local_shape.tolist()]
        synced[key] = pad_tensor_to_shape(value, target_shape=global_shape, pad_value=pad_value)

    if "attention_mask" in synced and torch.is_tensor(synced["attention_mask"]) and "position_ids" in synced:
        synced["position_ids"] = rebuild_position_ids_2d(synced["attention_mask"])
    return synced


def bytes_to_gib(num_bytes: int | float) -> float:
    return float(num_bytes) / float(1024**3)


def tensor_gib(value: Any) -> float:
    return bytes_to_gib(tensor_nbytes(value))


def tensor_nbytes(value: Any) -> int:
    if not isinstance(value, torch.Tensor):
        return 0
    return int(value.numel() * value.element_size())


def nested_tensor_nbytes(value: Any, device_type: Optional[str] = None) -> int:
    if isinstance(value, torch.Tensor):
        if device_type is not None and value.device.type != device_type:
            return 0
        return tensor_nbytes(value)
    if isinstance(value, dict):
        return sum(nested_tensor_nbytes(item, device_type=device_type) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(nested_tensor_nbytes(item, device_type=device_type) for item in value)
    return 0


def module_parameter_nbytes(module: torch.nn.Module, device_type: Optional[str] = "cuda") -> int:
    total = 0
    for parameter in module.parameters():
        if device_type is not None and parameter.device.type != device_type:
            continue
        total += tensor_nbytes(parameter)
    return total


def module_buffer_nbytes(module: torch.nn.Module, device_type: Optional[str] = "cuda") -> int:
    total = 0
    for buffer in module.buffers():
        if device_type is not None and buffer.device.type != device_type:
            continue
        total += tensor_nbytes(buffer)
    return total


def module_gradient_nbytes(module: torch.nn.Module, device_type: Optional[str] = "cuda") -> int:
    total = 0
    for parameter in module.parameters():
        grad = parameter.grad
        if grad is None:
            continue
        if device_type is not None and grad.device.type != device_type:
            continue
        total += tensor_nbytes(grad)
    return total


def optimizer_state_nbytes(optimizer: torch.optim.Optimizer, device_type: Optional[str] = "cuda") -> int:
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                if device_type is not None and value.device.type != device_type:
                    continue
                total += tensor_nbytes(value)
    return total


def current_cuda_memory_gib(device: Optional[torch.device] = None) -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {
            "allocated_gib": 0.0,
            "reserved_gib": 0.0,
            "max_allocated_gib": 0.0,
            "max_reserved_gib": 0.0,
        }
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.synchronize(device)
    return {
        "allocated_gib": bytes_to_gib(torch.cuda.memory_allocated(device)),
        "reserved_gib": bytes_to_gib(torch.cuda.memory_reserved(device)),
        "max_allocated_gib": bytes_to_gib(torch.cuda.max_memory_allocated(device)),
        "max_reserved_gib": bytes_to_gib(torch.cuda.max_memory_reserved(device)),
    }


def build_cuda_memory_snapshot(
    tag: str,
    rank: int,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    batch: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())

    snapshot: Dict[str, Any] = {
        "tag": tag,
        "rank": int(rank),
    }
    if device is not None:
        snapshot["device"] = str(device)
        snapshot.update(current_cuda_memory_gib(device))
    else:
        snapshot["device"] = "cpu"
        snapshot.update(current_cuda_memory_gib(device))

    param_bytes = module_parameter_nbytes(model) if model is not None else 0
    buffer_bytes = module_buffer_nbytes(model) if model is not None else 0
    grad_bytes = module_gradient_nbytes(model) if model is not None else 0
    optimizer_bytes = optimizer_state_nbytes(optimizer) if optimizer is not None else 0
    batch_bytes = nested_tensor_nbytes(batch, device_type="cuda") if batch is not None else 0

    snapshot.update(
        {
            "param_gib": bytes_to_gib(param_bytes),
            "buffer_gib": bytes_to_gib(buffer_bytes),
            "grad_gib": bytes_to_gib(grad_bytes),
            "optimizer_state_gib": bytes_to_gib(optimizer_bytes),
            "batch_tensor_gib": bytes_to_gib(batch_bytes),
        }
    )
    if extra:
        snapshot.update(extra)
    return snapshot


def reset_cuda_peak_memory(device: Optional[torch.device] = None) -> None:
    if not torch.cuda.is_available():
        return
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)


def append_jsonl(path: str | Path, payload: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
