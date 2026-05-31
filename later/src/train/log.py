from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.tensorboard import SummaryWriter

from later.src.train.utils import (
    bytes_to_gib,
    current_cuda_memory_gib,
    module_buffer_nbytes,
    module_gradient_nbytes,
    module_parameter_nbytes,
    nested_tensor_nbytes,
    optimizer_state_nbytes,
    tensor_nbytes,
)


class RankFilter(logging.Filter):
    def __init__(self, rank: int) -> None:
        super().__init__()
        self.rank = int(rank)

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.rank
        return True


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_file: str | Path | None = None,
    is_main_process: bool = True,
    rank: int = 0,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, str(log_level).upper(), logging.INFO))
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    for existing_filter in list(logger.filters):
        logger.removeFilter(existing_filter)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | rank=%(rank)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    rank_filter = RankFilter(rank=rank)
    logger.addFilter(rank_filter)

    if is_main_process:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logger.level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(rank_filter)
        logger.addHandler(console_handler)

    if log_file is not None:
        output_path = Path(log_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_path, encoding="utf-8")
        file_handler.setLevel(logger.level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(rank_filter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def tensorboard_dir_from_config(config: Dict[str, Any]) -> Path:
    return Path(str(config.get("tensorboard_root", "runs/tensorboard"))) / str(
        config.get("tensorboard_run_name", "sft_latent_qwen3")
    )


def attach_timestamp_to_tensorboard_run_name(config: Dict[str, Any], now: datetime | None = None) -> None:
    dt = now or datetime.now()
    base_run_name = str(config.get("tensorboard_run_name", "sft_latent_qwen3"))
    config["tensorboard_run_name"] = f"{base_run_name}_{dt.strftime('%Y-%m-%d_%H-%M-%S')}"


def create_summary_writer(config: Dict[str, Any], is_main_process: bool) -> SummaryWriter | None:
    if not is_main_process or not bool(config.get("enable_tensorboard", False)):
        return None
    log_dir = tensorboard_dir_from_config(config)
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir), flush_secs=int(config.get("tensorboard_flush_secs", 30)))


def log_scalars(writer: SummaryWriter | None, metrics: Dict[str, Any], step: int) -> None:
    if writer is None:
        return
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            writer.add_scalar(key, float(value.detach().item()), int(step))
        elif isinstance(value, (int, float)):
            writer.add_scalar(key, float(value), int(step))


def _tensorboard_code_block(text: str) -> str:
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    fence = "```"
    while fence in normalized:
        fence += "`"
    return f"{fence}\n{normalized}\n{fence}"


def _tensorboard_step_prefixed_record_id(global_step: int, record_id: str, width: int = 8) -> str:
    safe_record_id = str(record_id).replace("/", "_")
    return f"{int(global_step):0{int(width)}d}_{safe_record_id}"


def _eval_text_header(
    global_step: int,
    batch_index: int,
    sample_index: int,
    generated_latent_tokens: int,
    generated_cot_tokens: int,
    original_latent_tokens: int,
    original_cot_tokens: int,
) -> str:
    return (
        f"step={int(global_step)} batch_index={int(batch_index)} sample_index={int(sample_index)} "
        f"generated_latent_tokens={int(generated_latent_tokens)} "
        f"generated_cot_tokens={int(generated_cot_tokens)} "
        f"original_latent_tokens={int(original_latent_tokens)} "
        f"original_cot_tokens={int(original_cot_tokens)}"
    )


def build_train_scalar_payload(
    metrics: Dict[str, torch.Tensor],
    global_step: int,
    epoch_index: int,
    scheduler: Any,
) -> Dict[str, float]:
    lr = None
    if hasattr(scheduler, "get_last_lr"):
        last_lr = scheduler.get_last_lr()
        if last_lr:
            lr = float(last_lr[0])
    halt_dense_value = float(metrics.get("halt_dense_loss", metrics["latent_stop_loss"]).item())
    payload: Dict[str, float] = {
        "train/loss": float(metrics["loss"].detach().item()),
        "train/halt_dense_loss": halt_dense_value,
        "train/latent_stop_loss": halt_dense_value,
        "train/stop_loss": halt_dense_value,
        "train/halt_dense_loss_weight": float(metrics.get("halt_dense_loss_weight", 0.0)),
        "train/halt_dense_projected_loss": float(
            metrics.get("halt_dense_projected_loss", 0.0).item()
            if isinstance(metrics.get("halt_dense_projected_loss", 0.0), torch.Tensor)
            else metrics.get("halt_dense_projected_loss", 0.0)
        ),
        "train/halt_dense_gated_loss": float(
            metrics.get("halt_dense_gated_loss", metrics.get("halt_dense_projected_loss", 0.0)).item()
            if isinstance(metrics.get("halt_dense_gated_loss", metrics.get("halt_dense_projected_loss", 0.0)), torch.Tensor)
            else metrics.get("halt_dense_gated_loss", metrics.get("halt_dense_projected_loss", 0.0))
        ),
        "train/halt_dense_projection_alpha": float(
            metrics.get("halt_dense_projection_alpha", 0.0).item()
            if isinstance(metrics.get("halt_dense_projection_alpha", 0.0), torch.Tensor)
            else metrics.get("halt_dense_projection_alpha", 0.0)
        ),
        "train/halt_dense_projection_dot": float(
            metrics.get("halt_dense_projection_dot", 0.0).item()
            if isinstance(metrics.get("halt_dense_projection_dot", 0.0), torch.Tensor)
            else metrics.get("halt_dense_projection_dot", 0.0)
        ),
        "train/halt_dense_projection_ce_grad_norm_sq": float(
            metrics.get("halt_dense_projection_ce_grad_norm_sq", 0.0).item()
            if isinstance(metrics.get("halt_dense_projection_ce_grad_norm_sq", 0.0), torch.Tensor)
            else metrics.get("halt_dense_projection_ce_grad_norm_sq", 0.0)
        ),
        "train/halt_dense_gate_alpha": float(
            metrics.get("halt_dense_gate_alpha", metrics.get("halt_dense_projection_alpha", 0.0)).item()
            if isinstance(metrics.get("halt_dense_gate_alpha", metrics.get("halt_dense_projection_alpha", 0.0)), torch.Tensor)
            else metrics.get("halt_dense_gate_alpha", metrics.get("halt_dense_projection_alpha", 0.0))
        ),
        "train/halt_dense_gate_ce_loss_ema": float(
            metrics.get("halt_dense_gate_ce_loss_ema", 0.0).item()
            if isinstance(metrics.get("halt_dense_gate_ce_loss_ema", 0.0), torch.Tensor)
            else metrics.get("halt_dense_gate_ce_loss_ema", 0.0)
        ),
        "train/early_exit_rank_loss": float(metrics.get("early_exit_rank_loss", metrics["latent_stop_loss"]).item()),
        "train/early_exit_argmax_violation_rate": float(
            metrics.get("early_exit_argmax_violation_rate", 0.0).item()
            if isinstance(metrics.get("early_exit_argmax_violation_rate", 0.0), torch.Tensor)
            else metrics.get("early_exit_argmax_violation_rate", 0.0)
        ),
        "train/early_exit_front_rank_loss": float(
            metrics.get("early_exit_front_rank_loss", 0.0).item()
            if isinstance(metrics.get("early_exit_front_rank_loss", 0.0), torch.Tensor)
            else metrics.get("early_exit_front_rank_loss", 0.0)
        ),
        "train/early_exit_nonfront_rank_loss": float(
            metrics.get("early_exit_nonfront_rank_loss", 0.0).item()
            if isinstance(metrics.get("early_exit_nonfront_rank_loss", 0.0), torch.Tensor)
            else metrics.get("early_exit_nonfront_rank_loss", 0.0)
        ),
        "train/latent_end_soft_loss": float(
            metrics.get("latent_end_soft_loss", 0.0).item()
            if isinstance(metrics.get("latent_end_soft_loss", 0.0), torch.Tensor)
            else metrics.get("latent_end_soft_loss", 0.0)
        ),
        "train/other_end_hard_loss": float(
            metrics.get("other_end_hard_loss", 0.0).item()
            if isinstance(metrics.get("other_end_hard_loss", 0.0), torch.Tensor)
            else metrics.get("other_end_hard_loss", 0.0)
        ),
        "train/latent_end_target_mean": float(
            metrics.get("latent_end_target_mean", 0.0).item()
            if isinstance(metrics.get("latent_end_target_mean", 0.0), torch.Tensor)
            else metrics.get("latent_end_target_mean", 0.0)
        ),
        "train/latent_end_score_mean": float(
            metrics.get("latent_end_score_mean", 0.0).item()
            if isinstance(metrics.get("latent_end_score_mean", 0.0), torch.Tensor)
            else metrics.get("latent_end_score_mean", 0.0)
        ),
        "train/latent_end_front_score_mean": float(
            metrics.get("latent_end_front_score_mean", 0.0).item()
            if isinstance(metrics.get("latent_end_front_score_mean", 0.0), torch.Tensor)
            else metrics.get("latent_end_front_score_mean", 0.0)
        ),
        "train/latent_end_tail_score_mean": float(
            metrics.get("latent_end_tail_score_mean", 0.0).item()
            if isinstance(metrics.get("latent_end_tail_score_mean", 0.0), torch.Tensor)
            else metrics.get("latent_end_tail_score_mean", 0.0)
        ),
        "train/ce_loss": float(metrics["ce_loss"].item()),
        "train/cot_ce": float(
            metrics.get("cot_ce", 0.0).item() if isinstance(metrics.get("cot_ce", 0.0), torch.Tensor) else metrics.get("cot_ce", 0.0)
        ),
        "train/non_cot_ce": float(
            metrics.get("non_cot_ce", 0.0).item()
            if isinstance(metrics.get("non_cot_ce", 0.0), torch.Tensor)
            else metrics.get("non_cot_ce", 0.0)
        ),
        "train/answer_ce": float(metrics["answer_ce"].item()),
        "train/kl_loss": float(metrics["kl_loss"].item()),
        "train/epoch": float(epoch_index),
        "train/global_step": float(global_step),
    }
    if lr is not None:
        payload["train/lr"] = lr
    return payload


def log_eval_generated_text(
    writer: SummaryWriter | None,
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
    if writer is None:
        return
    safe_record_id = _tensorboard_step_prefixed_record_id(global_step=global_step, record_id=record_id)
    tag = f"val/generated_text/{safe_record_id}"
    header = _eval_text_header(
        global_step=global_step,
        batch_index=batch_index,
        sample_index=sample_index,
        generated_latent_tokens=generated_latent_tokens,
        generated_cot_tokens=generated_cot_tokens,
        original_latent_tokens=original_latent_tokens,
        original_cot_tokens=original_cot_tokens,
    )
    writer.add_text(tag, f"{header}\n\n{_tensorboard_code_block(generated_text)}", int(global_step))


def log_eval_sample_text_bundle(
    writer: SummaryWriter | None,
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
    if writer is None:
        return
    safe_record_id = _tensorboard_step_prefixed_record_id(global_step=global_step, record_id=record_id)
    header = f"step={int(global_step)} batch_index={int(batch_index)} sample_index={int(sample_index)}"
    bundle_header = _eval_text_header(
        global_step=global_step,
        batch_index=batch_index,
        sample_index=sample_index,
        generated_latent_tokens=generated_latent_tokens,
        generated_cot_tokens=generated_cot_tokens,
        original_latent_tokens=original_latent_tokens,
        original_cot_tokens=original_cot_tokens,
    )
    writer.add_text(
        f"val/prompt_token_ids/{safe_record_id}",
        f"{header}\n\n{_tensorboard_code_block(str(prompt_token_ids))}",
        int(global_step),
    )
    writer.add_text(f"val/prompt/{safe_record_id}", f"{header}\n\n{_tensorboard_code_block(prompt_text)}", int(global_step))
    writer.add_text(
        f"val/ground_truth/{safe_record_id}",
        f"{header}\n\n{_tensorboard_code_block(ground_truth)}",
        int(global_step),
    )
    writer.add_text(
        f"val/generated_token_ids/{safe_record_id}",
        f"{header}\n\n{_tensorboard_code_block(str(generated_token_ids))}",
        int(global_step),
    )
    writer.add_text(
        f"val/eval_sample_bundle/{safe_record_id}",
        (
            f"{bundle_header}\n\n"
            f"[prompt_token_ids]\n{_tensorboard_code_block(str(prompt_token_ids))}\n\n"
            f"[prompt]\n{_tensorboard_code_block(prompt_text)}\n\n"
            f"[ground_truth]\n{_tensorboard_code_block(ground_truth)}\n\n"
            f"[generated_token_ids]\n{_tensorboard_code_block(str(generated_token_ids))}\n\n"
            f"[generated]\n{_tensorboard_code_block(generated_text)}\n\n"
        ),
        int(global_step),
    )


def count_generated_token_types(token_ids: List[int], token_constants: Dict[str, int]) -> Dict[str, int]:
    latent_start_id = int(token_constants["latent_start_id"])
    latent_end_id = int(token_constants["latent_end_id"])
    think_start_id = int(token_constants["think_start_id"])
    think_end_id = int(token_constants["think_end_id"])

    in_latent = False
    in_cot = False
    latent_tokens = 0
    cot_tokens = 0

    for token_id in token_ids:
        if token_id == latent_start_id:
            in_latent = True
            continue
        if token_id == latent_end_id:
            in_latent = False
            continue
        if token_id == think_start_id:
            in_cot = True
            continue
        if token_id == think_end_id:
            in_cot = False
            continue
        if in_latent:
            latent_tokens += 1
        if in_cot:
            cot_tokens += 1

    return {
        "latent_tokens": int(latent_tokens),
        "cot_tokens": int(cot_tokens),
        "total_tokens": int(len(token_ids)),
    }


def token_repr(tokenizer: Any, token_id: int) -> str:
    try:
        tokens = tokenizer.convert_ids_to_tokens([int(token_id)])
        if isinstance(tokens, list) and tokens:
            return str(tokens[0])
    except Exception:
        pass
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return str(int(token_id))


def build_batch_diagnostics(
    batch: Dict[str, Any],
    global_step: int,
    epoch_index: int,
    teacher_length_mismatch_count: int,
    include_cuda_memory: bool,
) -> Dict[str, Any]:
    seq_len = int(batch["input_ids"].shape[1])
    latent_lengths_tensor = batch["latent_lengths"].detach().to(torch.long).cpu()
    halt_dense_valid_mask = (
        batch["latent_internal_mask"].detach().to(torch.bool).cpu()
        & batch["attention_mask"].detach().to(torch.bool).cpu()
    )
    halt_dense_positions_tensor = halt_dense_valid_mask.sum(dim=1).to(torch.long)
    attention_mask = batch["attention_mask"].detach().to(torch.long).cpu()
    token_lengths_tensor = attention_mask.sum(dim=1).to(torch.long)
    diag: Dict[str, Any] = {
        "global_step": int(global_step),
        "epoch_index": int(epoch_index),
        "micro_batch_size": int(batch["input_ids"].shape[0]),
        "padded_seq_len": seq_len,
        "token_lengths": token_lengths_tensor.tolist(),
        "max_token_length": int(token_lengths_tensor.max().item()),
        "latent_lengths": latent_lengths_tensor.tolist(),
        "max_latent_length": int(latent_lengths_tensor.max().item()) if latent_lengths_tensor.numel() > 0 else 0,
        "halt_dense_positions": halt_dense_positions_tensor.tolist(),
        "max_halt_dense_positions": int(halt_dense_positions_tensor.max().item())
        if halt_dense_positions_tensor.numel() > 0
        else 0,
        "record_ids": list(batch.get("record_ids", []))[:4],
        "teacher_length_mismatch_count": int(teacher_length_mismatch_count),
    }
    if include_cuda_memory:
        diag.update(current_cuda_memory_gib())
    return diag


def build_forward_memory_breakdown(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, Any],
    metrics: Dict[str, Any],
    global_step: int,
    epoch_index: int,
    teacher_length_mismatch_count: int,
) -> Dict[str, Any]:
    outputs = metrics["outputs"]
    ce_metrics = metrics["ce_metrics"]
    kl_stats = dict(metrics.get("kl_stats", {}))
    current = current_cuda_memory_gib() if torch.cuda.is_available() else {
        "allocated_gib": 0.0,
        "reserved_gib": 0.0,
        "max_allocated_gib": 0.0,
        "max_reserved_gib": 0.0,
    }
    model_param_gib = bytes_to_gib(module_parameter_nbytes(model))
    model_buffer_gib = bytes_to_gib(module_buffer_nbytes(model))
    grad_gib = bytes_to_gib(module_gradient_nbytes(model))
    optimizer_state_gib = bytes_to_gib(optimizer_state_nbytes(optimizer))
    batch_tensor_gib = bytes_to_gib(nested_tensor_nbytes(batch, device_type="cuda"))
    logits_gib = bytes_to_gib(tensor_nbytes(outputs.logits))
    sparse_loss_hidden_gib = bytes_to_gib(tensor_nbytes(getattr(outputs, "loss_hidden_states", None)))
    sparse_kl_topk_logits_gib = bytes_to_gib(tensor_nbytes(getattr(outputs, "teacher_kl_topk_logits", None)))
    sparse_kl_log_denom_gib = bytes_to_gib(tensor_nbytes(getattr(outputs, "teacher_kl_log_denom", None)))
    halt_dense_logits = getattr(outputs, "halt_dense_logits", getattr(outputs, "stop_logits", None))
    halt_dense_logits_gib = bytes_to_gib(tensor_nbytes(halt_dense_logits))
    token_ce_gib = bytes_to_gib(tensor_nbytes(ce_metrics.get("token_ce")))
    weighted_token_ce_gib = bytes_to_gib(tensor_nbytes(ce_metrics.get("weighted_token_ce")))
    latent_memory_info = getattr(outputs, "latent_memory_info", {}) or {}
    latent_rollout_embeds_gib = float(latent_memory_info.get("rollout_embeddings_gib", 0.0))
    latent_base_embeds_gib = float(latent_memory_info.get("rollout_base_embeddings_gib", 0.0))
    latent_step_input_gib = float(latent_memory_info.get("current_embeddings_peak_gib", 0.0))
    latent_prefix_hidden_gib = float(latent_memory_info.get("prefix_hidden_gib", 0.0))
    latent_prev_hidden_gib = float(latent_memory_info.get("prev_hidden_gib", 0.0))
    known_gib = (
        model_param_gib + model_buffer_gib + grad_gib + optimizer_state_gib + batch_tensor_gib
        + logits_gib + sparse_loss_hidden_gib + sparse_kl_topk_logits_gib + sparse_kl_log_denom_gib
        + halt_dense_logits_gib + token_ce_gib + weighted_token_ce_gib
        + latent_rollout_embeds_gib + latent_base_embeds_gib + latent_step_input_gib
        + latent_prefix_hidden_gib + latent_prev_hidden_gib
        + float(kl_stats.get("student_logits_slice_gib", 0.0))
        + float(kl_stats.get("teacher_cache_tensor_gib", 0.0))
        + float(kl_stats.get("student_topk_tensor_gib", 0.0))
        + float(kl_stats.get("kl_tensor_gib", 0.0))
    )
    residual_gib = max(float(current["allocated_gib"]) - known_gib, 0.0)
    payload = {
        "tag": "forward_memory_breakdown",
        "global_step": int(global_step),
        "epoch_index": int(epoch_index),
        "allocated_gib": float(current["allocated_gib"]),
        "reserved_gib": float(current["reserved_gib"]),
        "max_allocated_gib": float(current["max_allocated_gib"]),
        "max_reserved_gib": float(current["max_reserved_gib"]),
        "model_param_gib": model_param_gib,
        "model_buffer_gib": model_buffer_gib,
        "model_grad_gib": grad_gib,
        "optimizer_state_gib": optimizer_state_gib,
        "batch_tensor_gib": batch_tensor_gib,
        "token_logits_gib": logits_gib,
        "sparse_loss_hidden_gib": sparse_loss_hidden_gib,
        "sparse_kl_topk_logits_gib": sparse_kl_topk_logits_gib,
        "sparse_kl_log_denom_gib": sparse_kl_log_denom_gib,
        "halt_dense_logits_gib": halt_dense_logits_gib,
        "token_ce_gib": token_ce_gib,
        "weighted_token_ce_gib": weighted_token_ce_gib,
        "latent_rollout_embeddings_gib": latent_rollout_embeds_gib,
        "latent_rollout_base_embeddings_gib": latent_base_embeds_gib,
        "latent_current_step_embeddings_peak_gib": latent_step_input_gib,
        "latent_prefix_hidden_gib": latent_prefix_hidden_gib,
        "latent_prev_hidden_gib": latent_prev_hidden_gib,
        "latent_rollout_total_len": int(latent_memory_info.get("rollout_total_len", 0)),
        "latent_global_seq_len": int(latent_memory_info.get("global_seq_len", 0)),
        "latent_prefix_len": int(latent_memory_info.get("prefix_len", 0)),
        "latent_global_stop_steps": int(latent_memory_info.get("global_stop_steps", 0)),
        "latent_normal_mode": str(latent_memory_info.get("normal_mode", "")),
        "latent_normal_suffix_max_len": int(latent_memory_info.get("normal_suffix_max_len", 0)),
        "latent_normal_local_suffix_len": int(latent_memory_info.get("normal_suffix_local_max_len", 0)),
        "latent_normal_global_suffix_len": int(latent_memory_info.get("normal_global_suffix_len", 0)),
        "latent_normal_global_num_chunks": int(latent_memory_info.get("normal_global_num_chunks", 0)),
        "latent_normal_chunk_size": int(latent_memory_info.get("normal_chunk_size", 0)),
        "latent_normal_rank_sync_enabled": bool(latent_memory_info.get("normal_rank_sync_enabled", False)),
        "latent_runtime_active_count": int(latent_memory_info.get("runtime_active_count", 0)),
        "latent_runtime_continue_count": int(latent_memory_info.get("runtime_continue_count", 0)),
        "latent_runtime_stop_count": int(latent_memory_info.get("runtime_stop_count", 0)),
        "kl_student_logits_slice_gib": float(kl_stats.get("student_logits_slice_gib", 0.0)),
        "kl_teacher_cache_tensor_gib": float(kl_stats.get("teacher_cache_tensor_gib", 0.0)),
        "kl_student_topk_tensor_gib": float(kl_stats.get("student_topk_tensor_gib", 0.0)),
        "kl_tensor_gib": float(kl_stats.get("kl_tensor_gib", 0.0)),
        "kl_positions": int(kl_stats.get("kl_positions", 0)),
        "deepspeed_other_runtime_gib": residual_gib,
        "teacher_length_mismatch_count": int(teacher_length_mismatch_count),
        "record_ids": list(batch.get("record_ids", []))[:4],
    }
    hybrid_cache_lengths = latent_memory_info.get("hybrid_cache_lengths")
    hybrid_cache_padding_lengths = latent_memory_info.get("hybrid_cache_padding_lengths")
    if hybrid_cache_lengths is not None:
        payload["latent_hybrid_cache_lengths"] = list(hybrid_cache_lengths)
    if hybrid_cache_padding_lengths is not None:
        payload["latent_hybrid_cache_padding_lengths"] = list(hybrid_cache_padding_lengths)
    return payload


def build_dry_run_payload(
    train_dataset: Any,
    val_dataset: Any,
    batch: Dict[str, Any],
    teacher_cache: Any,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "train_rows": len(train_dataset),
        "val_rows": len(val_dataset),
        "train_skipped_rows": train_dataset.skipped_rows,
        "val_skipped_rows": val_dataset.skipped_rows,
        "batch_shape": tuple(batch["input_ids"].shape),
        "teacher_cache_entries": len(teacher_cache.record_to_meta),
        "tensorboard_enabled": bool(config.get("enable_tensorboard", False)),
        "tensorboard_dir": str(tensorboard_dir_from_config(config)),
    }


def build_distributed_init_payload(config: Dict[str, Any], accelerator: Any) -> Dict[str, Any]:
    return {
        "distributed_backend": str(config.get("distributed_backend", "fsdp")),
        "distributed_type": str(accelerator.distributed_type),
        "num_processes": accelerator.num_processes,
        "process_index": accelerator.process_index,
        "tensorboard_enabled": bool(config.get("enable_tensorboard", False)),
        "tensorboard_dir": str(tensorboard_dir_from_config(config)),
    }


def build_training_setup_payload(
    config: Dict[str, Any],
    trainable_param_count: int,
    optimizer_param_count: int,
    total_param_count: int,
    global_batch_size: int,
    steps_per_epoch: int,
    total_steps: int,
) -> Dict[str, Any]:
    return {
        "optimizer": "adamw_fixed",
        "train_path": "stagewise_fixed",
        "stage1_trainable_mode": "projector_embed_lmhead",
        "stage2_trainable_mode": "full",
        "latent_stop_loss": "ce_fixed",
        "train_with_latent_internal_recurrence": True,
        "train_rollout_use_cache": True,
        "train_forward_use_pastkv": True,
        "cot_ce_loss_weight": float(config.get("cot_ce_loss_weight", 0.3)),
        "latent_start_ce_loss_weight": float(config.get("latent_start_ce_loss_weight", 1.0)),
        "latent_end_ce_loss_weight": float(config.get("latent_end_ce_loss_weight", 1.0)),
        "enable_loss_alignment_debug": bool(config.get("enable_loss_alignment_debug", False)),
        "loss_alignment_debug_every_steps": int(config.get("loss_alignment_debug_every_steps", 1)),
        "alignment_debug_max_samples": int(config.get("alignment_debug_max_samples", 1)),
        "trainable_params_m": round(trainable_param_count / 1.0e6, 3),
        "optimizer_params_m": round(optimizer_param_count / 1.0e6, 3),
        "total_params_b": round(total_param_count / 1.0e9, 3),
        "latent_bptt_window": int(config.get("latent_bptt_window", 0) or 0),
        "discrete_chunk_size": int(config.get("discrete_chunk_size", 0) or 0),
        "enable_discrete_deadlock_probe_log": bool(config.get("enable_discrete_deadlock_probe_log", False)),
        "discrete_deadlock_probe_chunk_idx": int(config.get("discrete_deadlock_probe_chunk_idx", -1)),
        "discrete_deadlock_probe_layers": list(config.get("discrete_deadlock_probe_layers", [])),
        "discrete_deadlock_probe_cuda_sync": bool(config.get("discrete_deadlock_probe_cuda_sync", False)),
        "enable_discrete_safe_attention": bool(config.get("enable_discrete_safe_attention", True)),
        "discrete_attention_impl": str(config.get("discrete_attention_impl", "eager")),
        "enable_rank_pad_sync": bool(config.get("enable_rank_pad_sync", True)),
        "kl_topk_dim": int(config.get("kl_topk_dim", 128) or 0),
        "distributed_backend": str(config.get("distributed_backend", "fsdp")),
        "deepspeed_zero_stage": int(config.get("deepspeed_zero_stage", 0) or 0),
        "global_batch_size": int(global_batch_size),
        "steps_per_epoch": int(steps_per_epoch),
        "total_steps": int(total_steps),
    }
