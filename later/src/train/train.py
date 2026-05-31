from __future__ import annotations

import gc
import os
import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import torch.distributed as dist
from accelerate import Accelerator, DeepSpeedPlugin, FullyShardedDataParallelPlugin
from transformers import AutoConfig, AutoTokenizer, get_cosine_schedule_with_warmup

from later.src.train.dataset import LatentSFTCollator, LatentSFTDataset, load_sft_frame, load_sft_split
from later.src.train.modeling_latent import (
    LEGACY_LATENT_PROJECTOR_KEY_MAPPING,
    LatentQwenConfig,
    LatentQwenForCausalLM,
    read_model_type_from_config_dir,
    register_latent_qwen3_with_transformers,
)
from later.src.train.log import (
    attach_timestamp_to_tensorboard_run_name,
    build_distributed_init_payload,
    build_dry_run_payload,
    build_training_setup_payload,
    create_summary_writer,
    setup_logger,
)
from later.src.train.trainer import LatentSFTTrainer, TrainLoopResult
from later.src.train.trainer_latent_cp import LatentCPSFTTrainer
from later.src.train.utils import (
    BaseTokenRowFreezeController,
    PrecomputedTeacherCache,
    append_jsonl,
    build_cuda_memory_snapshot,
    configure_base_token_row_freezing,
    get_token_constants,
    load_yaml,
    set_seed,
)
from later.src.utils.utils import ensure_latent_think_special_tokens, validate_latent_think_tokenizer_contract

STAGE1_TRAINABLE_MODE = "projector_embed_lmhead"
STAGE2_TRAINABLE_MODE = "full"


def _resolve_deepspeed_value(value: Any) -> Any:
    if isinstance(value, str) and value.lower() == "auto":
        return None
    return value


def build_deepspeed_config(config: Dict[str, Any]) -> Dict[str, Any]:
    mixed_precision = str(config["mixed_precision"]).lower()
    zero_stage = int(config.get("deepspeed_zero_stage", 3))
    zero_optimization: Dict[str, Any] = {
        "stage": zero_stage,
        "contiguous_gradients": bool(config.get("deepspeed_contiguous_gradients", True)),
        "overlap_comm": bool(config.get("deepspeed_overlap_comm", True)),
        "stage3_gather_16bit_weights_on_model_save": bool(
            config.get("deepspeed_gather_16bit_weights_on_model_save", False)
        ),
    }

    reduce_bucket_size = _resolve_deepspeed_value(config.get("deepspeed_reduce_bucket_size", "auto"))
    if reduce_bucket_size is not None:
        zero_optimization["reduce_bucket_size"] = reduce_bucket_size

    stage3_prefetch_bucket_size = _resolve_deepspeed_value(config.get("deepspeed_stage3_prefetch_bucket_size", "auto"))
    if stage3_prefetch_bucket_size is not None:
        zero_optimization["stage3_prefetch_bucket_size"] = stage3_prefetch_bucket_size

    stage3_param_persistence_threshold = _resolve_deepspeed_value(
        config.get("deepspeed_stage3_param_persistence_threshold", "auto")
    )
    if stage3_param_persistence_threshold is not None:
        zero_optimization["stage3_param_persistence_threshold"] = stage3_param_persistence_threshold

    sub_group_size = _resolve_deepspeed_value(config.get("deepspeed_sub_group_size", "auto"))
    if sub_group_size is not None:
        zero_optimization["sub_group_size"] = sub_group_size

    stage3_max_live_parameters = _resolve_deepspeed_value(config.get("deepspeed_stage3_max_live_parameters", "auto"))
    if stage3_max_live_parameters is not None:
        zero_optimization["stage3_max_live_parameters"] = stage3_max_live_parameters

    stage3_max_reuse_distance = _resolve_deepspeed_value(config.get("deepspeed_stage3_max_reuse_distance", "auto"))
    if stage3_max_reuse_distance is not None:
        zero_optimization["stage3_max_reuse_distance"] = stage3_max_reuse_distance

    offload_optimizer_device = str(config.get("deepspeed_offload_optimizer_device", "none")).lower()
    if offload_optimizer_device != "none":
        zero_optimization["offload_optimizer"] = {"device": offload_optimizer_device}

    offload_param_device = str(config.get("deepspeed_offload_param_device", "none")).lower()
    if zero_stage == 3 and offload_param_device != "none":
        zero_optimization["offload_param"] = {"device": offload_param_device}

    return {
        "train_micro_batch_size_per_gpu": int(config["micro_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "gradient_clipping": float(config["max_grad_norm"]),
        "zero_optimization": zero_optimization,
        "bf16": {"enabled": mixed_precision == "bf16"},
        "fp16": {"enabled": mixed_precision == "fp16"},
    }


def validate_backend_config(config: Dict[str, Any]) -> None:
    backend = str(config.get("distributed_backend", "fsdp")).lower()
    if (
        backend == "deepspeed"
        and int(config.get("deepspeed_zero_stage", 3)) == 3
        and not bool(config.get("deepspeed_gather_16bit_weights_on_model_save", False))
    ):
        raise ValueError(
            "distributed_backend=deepspeed with ZeRO-3 requires deepspeed_gather_16bit_weights_on_model_save=true "
            "to save Hugging Face checkpoints via save_pretrained"
        )


def _is_staged_forward_enabled(config: Dict[str, Any]) -> bool:
    return (not bool(config.get("global_stateless_forward", True))) and bool(
        config.get("legacy_staged_forward_fallback", True)
    )


def _resolve_effective_runtime_flags(config: Dict[str, Any], logger: Any) -> Dict[str, Any]:
    requested_gc = bool(config.get("gradient_checkpointing", False))
    staged_forward_enabled = _is_staged_forward_enabled(config)
    effective_gc = bool(requested_gc and (not staged_forward_enabled))
    halt_dense_projection_mode = str(config.get("halt_dense_loss_projection_mode", "none") or "none").strip().lower()
    if halt_dense_projection_mode == "ce_positive_projection":
        raise ValueError(
            "halt_dense_loss_projection_mode=ce_positive_projection is unsupported with the current training path. "
            "Use halt_dense_loss_projection_mode=ce_quality_gate or none."
        )
    if halt_dense_projection_mode not in {"none", "ce_quality_gate"}:
        raise ValueError(f"Unsupported halt_dense_loss_projection_mode: {halt_dense_projection_mode}")
    if requested_gc and staged_forward_enabled:
        logger.warning(
            {
                "tag": "gc_auto_disabled_for_staged_path",
                "reason": "staged_forward_requires_past_key_values_and_use_cache",
                "gradient_checkpointing_requested": bool(requested_gc),
                "gradient_checkpointing_effective": bool(effective_gc),
                "global_stateless_forward": bool(config.get("global_stateless_forward", True)),
                "legacy_staged_forward_fallback": bool(config.get("legacy_staged_forward_fallback", True)),
            }
        )
    config["gradient_checkpointing"] = bool(effective_gc)
    return {
        "staged_forward_enabled": bool(staged_forward_enabled),
        "gradient_checkpointing_requested": bool(requested_gc),
        "gradient_checkpointing_effective": bool(effective_gc),
        "halt_dense_loss_projection_mode": halt_dense_projection_mode,
        "attn_implementation_requested": str(config.get("attn_implementation", "sdpa")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def build_accelerator(config: Dict[str, Any]) -> Accelerator:
    backend = str(config.get("distributed_backend", "fsdp")).lower()
    context_parallel_size = int(config.get("context_parallel_size", 1) or 1)
    common_kwargs = {
        "mixed_precision": str(config["mixed_precision"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "log_with": None,
    }
    if context_parallel_size > 1:
        if backend != "fsdp":
            raise ValueError("Accelerate context parallelism requires distributed_backend=fsdp/FSDP2.")
        if str(config.get("attn_implementation", "sdpa")).lower() != "sdpa":
            raise ValueError("Accelerate context parallelism only supports SDPA/no-mask causal attention here.")
        try:
            from accelerate.utils import ParallelismConfig, TorchContextParallelConfig
        except Exception as exc:  # pragma: no cover - depends on runtime accelerate version
            raise RuntimeError(
                "Installed accelerate does not expose ParallelismConfig/TorchContextParallelConfig. "
                "Upgrade accelerate to a version with context parallelism support."
            ) from exc
        cp_handler = TorchContextParallelConfig(
            cp_comm_strategy=str(config.get("context_parallel_comm_strategy", "allgather"))
        )
        common_kwargs["parallelism_config"] = ParallelismConfig(
            cp_size=int(context_parallel_size),
            cp_handler=cp_handler,
        )
    if backend == "deepspeed":
        ds_plugin = DeepSpeedPlugin(
            hf_ds_config=build_deepspeed_config(config),
            gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
            gradient_clipping=float(config["max_grad_norm"]),
            zero_stage=int(config.get("deepspeed_zero_stage", 3)),
            offload_optimizer_device=str(config.get("deepspeed_offload_optimizer_device", "none")),
            offload_param_device=str(config.get("deepspeed_offload_param_device", "none")),
            zero3_init_flag=bool(config.get("deepspeed_zero3_init_flag", True)),
            zero3_save_16bit_model=bool(config.get("deepspeed_gather_16bit_weights_on_model_save", False)),
        )
        return Accelerator(deepspeed_plugin=ds_plugin, **common_kwargs)
    plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        reshard_after_forward=True,
        mixed_precision_policy=str(config["mixed_precision_policy"]),
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=[str(config["transformer_layer_cls"])],
        state_dict_type="sharded_state_dict",
        cpu_ram_efficient_loading=True,
        limit_all_gathers=True,
        activation_checkpointing=bool(config.get("gradient_checkpointing", False)),
    )
    return Accelerator(fsdp_plugin=plugin, **common_kwargs)


def _apply_trainable_mode(model: Any, mode: str) -> None:
    mode = str(mode)
    if mode not in {"projector_embed_lmhead", "full"}:
        raise ValueError(f"Unsupported fixed trainable mode: {mode}")

    if mode == "full":
        for param in model.parameters():
            param.requires_grad = True
        return

    for param in model.parameters():
        param.requires_grad = False

    for module_name in ["latent_projector", "lm_head"]:
        module = getattr(model, module_name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True

    if mode == "projector_embed_lmhead":
        input_embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        if input_embeddings is not None:
            for param in input_embeddings.parameters():
                param.requires_grad = True
        return


def configure_trainable_parameters(model: Any, mode: str = STAGE1_TRAINABLE_MODE) -> List[torch.nn.Parameter]:
    _apply_trainable_mode(model=model, mode=mode)
    return list(model.parameters())


def _extract_checkpoint_step(path: Path) -> int:
    match = re.match(r"^step_(\d+)$", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def _checkpoint_complete_marker_name() -> str:
    return "checkpoint_complete.json"


def _is_checkpoint_complete(checkpoint_dir: Path) -> bool:
    return bool((checkpoint_dir / _checkpoint_complete_marker_name()).exists())


def _list_complete_checkpoint_dirs(output_dir: Path) -> List[Path]:
    if not output_dir.exists():
        return []
    candidates: List[Path] = []
    for path in output_dir.iterdir():
        if not path.is_dir():
            continue
        if _extract_checkpoint_step(path) < 0:
            continue
        if not _is_checkpoint_complete(path):
            continue
        candidates.append(path)
    candidates.sort(key=_extract_checkpoint_step)
    return candidates


def _resolve_resume_checkpoint_dir(config: Dict[str, Any]) -> Path:
    requested = str(config.get("resume_from_checkpoint", "latest") or "").strip()
    if requested == "":
        requested = "latest"

    output_dir = Path(config["output_dir"])
    if requested.lower() == "latest":
        latest_name = str(config.get("latest_checkpoint_name", "latest") or "latest").strip() or "latest"
        latest_dir = output_dir / latest_name
        if latest_dir.exists():
            if _is_checkpoint_complete(latest_dir):
                return latest_dir
            warnings.warn(
                (
                    "resume_training=true but latest checkpoint path is incomplete; "
                    f"falling back to the newest complete numbered checkpoint: {latest_dir}"
                ),
                RuntimeWarning,
                stacklevel=2,
            )
        checkpoints = _list_complete_checkpoint_dirs(output_dir)
        if not checkpoints:
            raise FileNotFoundError(
                f"resume_training=true but no complete checkpoint was found under {output_dir}"
            )
        return checkpoints[-1]

    requested_path = Path(requested)
    if not requested_path.is_absolute():
        requested_path = output_dir / requested_path
    if not requested_path.exists():
        raise FileNotFoundError(
            f"resume_training=true but requested checkpoint does not exist: {requested_path}"
        )
    if not _is_checkpoint_complete(requested_path):
        raise RuntimeError(
            f"resume_training=true but requested checkpoint is incomplete: {requested_path}"
        )
    return requested_path


def _load_resume_metadata(checkpoint_dir: Path) -> Dict[str, Any]:
    state_path = checkpoint_dir / "training_state.json"
    default_global_step = max(_extract_checkpoint_step(checkpoint_dir), 0)
    if not state_path.exists():
        return {
            "global_step": int(default_global_step),
            "epoch_index": 0,
            "next_step_in_epoch": 0,
            "train_stage": None,
            "trainable_mode": None,
        }
    with state_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    train_stage = payload.get("train_stage")
    trainable_mode = payload.get("trainable_mode")
    return {
        "global_step": int(payload.get("global_step", default_global_step)),
        "epoch_index": int(payload.get("epoch_index", 0)),
        "next_step_in_epoch": int(payload.get("next_step_in_epoch", 0)),
        "train_stage": int(train_stage) if train_stage is not None else None,
        "trainable_mode": str(trainable_mode) if trainable_mode is not None else None,
    }


def compute_total_train_steps(config: Dict[str, Any], train_dataset_len: int, num_processes: int) -> int:
    max_train_steps = int(config["max_train_steps"])
    if max_train_steps > 0:
        return max_train_steps
    micro_batch_size = max(int(config["micro_batch_size"]), 1)
    global_batch_size = micro_batch_size * max(int(num_processes), 1)
    steps_per_epoch = max(1, (int(train_dataset_len) + global_batch_size - 1) // global_batch_size)
    return steps_per_epoch * max(int(config["num_epochs"]), 1)


def infer_train_stage(global_step: int, total_train_steps: int, config: Dict[str, Any]) -> int:
    progress = float(global_step) / float(max(int(total_train_steps), 1))
    progress = min(max(progress, 0.0), 1.0)
    train_stage = 2 if progress >= float(config["stage2_start_fraction"]) else 1
    if int(config["train_stage"]) == 2:
        train_stage = 2
    return int(train_stage)


def trainable_mode_for_stage(train_stage: int) -> str:
    return STAGE1_TRAINABLE_MODE if int(train_stage) == 1 else STAGE2_TRAINABLE_MODE


def resolve_resume_plan(
    config: Dict[str, Any],
    train_dataset_len: int,
    num_processes: int,
) -> Dict[str, Any]:
    total_train_steps = compute_total_train_steps(
        config=config,
        train_dataset_len=train_dataset_len,
        num_processes=num_processes,
    )
    micro_batch_size = max(int(config["micro_batch_size"]), 1)
    global_batch_size = micro_batch_size * max(int(num_processes), 1)
    steps_per_epoch = max(1, (int(train_dataset_len) + global_batch_size - 1) // global_batch_size)
    default_stage = infer_train_stage(
        global_step=0,
        total_train_steps=total_train_steps,
        config=config,
    )
    default_mode = trainable_mode_for_stage(default_stage)
    if not bool(config.get("resume_training", False)):
        return {
            "enabled": False,
            "checkpoint_dir": None,
            "global_step": 0,
            "epoch_index": 0,
            "next_step_in_epoch": 0,
            "train_stage": int(default_stage),
            "trainable_mode": str(default_mode),
            "model_load_path": str(config["model_path"]),
            "total_train_steps": int(total_train_steps),
            "steps_per_epoch": int(steps_per_epoch),
        }

    checkpoint_dir = _resolve_resume_checkpoint_dir(config)
    state = _load_resume_metadata(checkpoint_dir)
    resume_override_global_step = config.get("resume_override_global_step", None)
    resume_step_overridden = resume_override_global_step is not None
    checkpoint_global_step = int(state["global_step"])
    checkpoint_epoch_index = int(state["epoch_index"])
    checkpoint_next_step_in_epoch = int(state["next_step_in_epoch"])
    if resume_step_overridden:
        override_global_step = min(max(int(resume_override_global_step), 0), int(total_train_steps))
        state["global_step"] = int(override_global_step)
        state["epoch_index"] = int(override_global_step // max(int(steps_per_epoch), 1))
        state["next_step_in_epoch"] = int(override_global_step % max(int(steps_per_epoch), 1))
        state["train_stage"] = None
        state["trainable_mode"] = None
    train_stage = state["train_stage"]
    if train_stage is None:
        train_stage = infer_train_stage(
            global_step=int(state["global_step"]),
            total_train_steps=total_train_steps,
            config=config,
        )
    trainable_mode = state["trainable_mode"] or trainable_mode_for_stage(int(train_stage))
    return {
        "enabled": True,
        "checkpoint_dir": str(checkpoint_dir),
        "global_step": int(state["global_step"]),
        "epoch_index": int(state["epoch_index"]),
        "next_step_in_epoch": int(state["next_step_in_epoch"]),
        "train_stage": int(train_stage),
        "trainable_mode": str(trainable_mode),
        "model_load_path": str(checkpoint_dir),
        "total_train_steps": int(total_train_steps),
        "steps_per_epoch": int(steps_per_epoch),
        "resume_step_overridden": bool(resume_step_overridden),
        "checkpoint_global_step": int(checkpoint_global_step),
        "checkpoint_epoch_index": int(checkpoint_epoch_index),
        "checkpoint_next_step_in_epoch": int(checkpoint_next_step_in_epoch),
    }


def _parameter_numel(param: torch.nn.Parameter) -> int:
    numel = int(param.numel())
    if numel > 0:
        return numel
    ds_numel = getattr(param, "ds_numel", None)
    if ds_numel is None:
        return numel
    try:
        return int(ds_numel)
    except Exception:
        return numel


def _count_parameters(parameters: Iterable[torch.nn.Parameter], trainable_only: bool = False) -> int:
    total = 0
    for param in parameters:
        if trainable_only and (not bool(param.requires_grad)):
            continue
        total += _parameter_numel(param)
    return int(total)


def _count_parameter_tensors(parameters: Iterable[torch.nn.Parameter], trainable_only: bool = False) -> int:
    total = 0
    for param in parameters:
        if trainable_only and (not bool(param.requires_grad)):
            continue
        total += 1
    return int(total)


def _collect_trainable_module_names(model: Any, max_items: int = 32) -> List[str]:
    names: List[str] = []
    for module_name, module in model.named_modules():
        for param in module.parameters(recurse=False):
            if bool(param.requires_grad) and _parameter_numel(param) > 0:
                names.append(module_name if module_name else "<root>")
                break
        if len(names) >= int(max_items):
            break
    return names


def build_optimizer(
    trainable_params: Iterable[torch.nn.Parameter],
    config: Dict[str, Any],
    zero_weight_decay_param_ids: set[int] | None = None,
) -> torch.optim.Optimizer:
    params = [param for param in trainable_params if bool(param.requires_grad)]
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    zero_weight_decay_param_ids = set(zero_weight_decay_param_ids or set())

    param_groups: List[Dict[str, Any]] = []
    if zero_weight_decay_param_ids:
        decay_params = [param for param in params if id(param) not in zero_weight_decay_param_ids]
        no_decay_params = [param for param in params if id(param) in zero_weight_decay_param_ids]
        if decay_params:
            param_groups.append({"params": decay_params, "weight_decay": weight_decay})
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    else:
        param_groups.append({"params": params, "weight_decay": weight_decay})

    return torch.optim.AdamW(
        param_groups,
        lr=learning_rate,
        betas=(float(config["adam_beta1"]), float(config["adam_beta2"])),
        eps=float(config["adam_eps"]),
        weight_decay=weight_decay,
        foreach=bool(config.get("optimizer_foreach", False)),
    )


def build_scheduler_with_resume(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
    resume_global_step: int = 0,
) -> tuple[Any, int, int, float]:
    total_steps = max(int(total_steps), 1)
    warmup_steps = min(max(int(total_steps * float(warmup_ratio)), 0), total_steps)
    scheduler_resume_step = min(max(int(resume_global_step), 0), total_steps)
    scheduler_last_epoch = int(scheduler_resume_step - 1)

    for param_group in optimizer.param_groups:
        param_group.setdefault("initial_lr", float(param_group["lr"]))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        last_epoch=scheduler_last_epoch,
    )

    lr_after_scheduler_init = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0
    return scheduler, int(warmup_steps), int(scheduler_resume_step), lr_after_scheduler_init


def load_tokenizer(
    config: Dict[str, Any],
    return_base_vocab_size: bool = False,
    model_path_override: str | None = None,
) -> Any | tuple[Any, int]:
    model_path = str(model_path_override or config["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    base_vocab_size = int(len(tokenizer))
    ensure_latent_think_special_tokens(tokenizer)
    validate_latent_think_tokenizer_contract(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "<|endoftext|>"
    if return_base_vocab_size:
        return tokenizer, base_vocab_size
    return tokenizer


def load_model_and_tokenizer(
    config: Dict[str, Any],
    logger: Any,
    model_path_override: str | None = None,
) -> tuple[Any, Any, str, int]:
    model_path = str(model_path_override or config["model_path"])
    tokenizer, base_vocab_size = load_tokenizer(
        config,
        return_base_vocab_size=True,
        model_path_override=model_path,
    )

    base_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    latent_config = LatentQwenConfig.from_dict(base_config.to_dict())
    max_length = int(config.get("max_length", 0) or 0)
    original_rope_max = int(config.get("rope_original_max_position_embeddings", 32768) or 32768)
    if max_length > original_rope_max:
        factor = float(config.get("rope_scaling_factor", 0.0) or 0.0)
        if factor <= 0.0:
            factor = float(max(1.25, math.ceil(max_length / max(original_rope_max, 1))))
        rope_scaling = dict(config.get("rope_scaling") or {})
        if not rope_scaling:
            rope_scaling = {
                "rope_type": str(config.get("rope_scaling_type", "yarn")),
                "factor": factor,
                "original_max_position_embeddings": original_rope_max,
            }
        latent_config.rope_scaling = rope_scaling
        latent_config.max_position_embeddings = max(
            int(getattr(latent_config, "max_position_embeddings", 0) or 0),
            max_length,
        )
        logger.info(
            {
                "tag": "latent_rope_scaling_configured",
                "max_length": int(max_length),
                "model_max_position_embeddings": int(latent_config.max_position_embeddings),
                "rope_scaling": dict(latent_config.rope_scaling or {}),
            }
        )
    latent_config.use_cache = bool(config.get("use_cache", True))
    latent_config.train_with_latent_internal_recurrence = bool(
        config.get("train_with_latent_internal_recurrence", True)
    )
    latent_config.train_rollout_use_cache = bool(config.get("train_rollout_use_cache", True))
    latent_config.train_forward_use_pastkv = bool(config.get("train_forward_use_pastkv", True))
    latent_config.discrete_chunk_size = int(config.get("discrete_chunk_size", 0) or 0)
    latent_config.latent_bptt_window = int(config.get("latent_bptt_window", 0) or 0)
    latent_config.kl_temperature = float(config.get("kl_temperature", 1.0))
    latent_config.supervised_logits_chunk_size = int(config.get("supervised_logits_chunk_size", 0) or 0)
    latent_config.enable_forward_memory_breakdown_log = bool(config.get("enable_forward_memory_breakdown_log", False))
    latent_config.enable_hybrid_cache_debug_log = bool(config.get("enable_hybrid_cache_debug_log", False))
    latent_config.enable_discrete_deadlock_probe_log = bool(config.get("enable_discrete_deadlock_probe_log", False))
    latent_config.discrete_deadlock_probe_chunk_idx = int(config.get("discrete_deadlock_probe_chunk_idx", -1))
    latent_config.discrete_deadlock_probe_layers = list(config.get("discrete_deadlock_probe_layers", []))
    latent_config.discrete_deadlock_probe_cuda_sync = bool(config.get("discrete_deadlock_probe_cuda_sync", False))
    latent_config.enable_discrete_safe_attention = bool(config.get("enable_discrete_safe_attention", True))
    latent_config.discrete_attention_impl = str(config.get("discrete_attention_impl", "eager"))
    latent_config.global_stateless_forward = bool(config.get("global_stateless_forward", True))
    latent_config.legacy_staged_forward_fallback = bool(config.get("legacy_staged_forward_fallback", True))
    latent_config.latent_projector_dropout = float(config.get("latent_projector_dropout", 0.1))

    attn_impl = str(config.get("attn_implementation", "sdpa"))
    effective_attn_impl = attn_impl
    if int(config.get("context_parallel_size", 1) or 1) > 1 and attn_impl.lower() != "sdpa":
        raise ValueError("Latent CP training requires attn_implementation=sdpa.")
    try:
        model = LatentQwenForCausalLM.from_pretrained(
            model_path,
            config=latent_config,
            torch_dtype=getattr(torch, str(config["torch_dtype"])),
            trust_remote_code=True,
            attn_implementation=attn_impl,
            key_mapping=LEGACY_LATENT_PROJECTOR_KEY_MAPPING,
        )
    except Exception as exc:
        if attn_impl == "flash_attention_2":
            logger.warning(
                {
                    "tag": "attn_fallback",
                    "requested": attn_impl,
                    "fallback": "sdpa",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            model = LatentQwenForCausalLM.from_pretrained(
                model_path,
                config=latent_config,
                torch_dtype=getattr(torch, str(config["torch_dtype"])),
                trust_remote_code=True,
                attn_implementation="sdpa",
                key_mapping=LEGACY_LATENT_PROJECTOR_KEY_MAPPING,
            )
            effective_attn_impl = "sdpa"
        else:
            raise
    ensure_latent_think_special_tokens(tokenizer, model=model)
    validate_latent_think_tokenizer_contract(tokenizer)
    return model, tokenizer, str(effective_attn_impl), int(base_vocab_size)


def _cleanup_training_runtime(*objects: Any) -> None:
    for obj in objects:
        try:
            close_writer = getattr(obj, "_close_writer", None)
            if callable(close_writer):
                close_writer()
        except Exception:
            pass
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _load_train_and_val_frames(config: Dict[str, Any]) -> tuple[Any, Any]:
    external_val_path = str(config.get("val_data", "") or "").strip()
    if external_val_path:
        train_df = load_sft_frame(str(config["train_data"]))
        val_df = load_sft_frame(external_val_path)
        return train_df, val_df
    return load_sft_split(config["train_data"], float(config["val_ratio"]))


def _run_training_once(config: Dict[str, Any], args: argparse.Namespace) -> TrainLoopResult:
    registration_status = register_latent_qwen3_with_transformers()
    alloc_conf = str(config.get("pytorch_cuda_alloc_conf", "")).strip()
    if alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf

    validate_backend_config(config)
    log_dir = Path(config.get("log_dir") or config["output_dir"])
    train_log_file = str(config.get("train_log_file", "train.log"))
    logger = setup_logger(
        name="later.train",
        log_level=str(config.get("log_level", "INFO")),
        log_file=log_dir / train_log_file,
        is_main_process=True,
        rank=0,
    )
    logger.info(
        {
            "tag": "latent_qwen3_registry",
            "status": dict(registration_status),
            "configured_model_path": str(config.get("model_path")),
            "configured_model_type": read_model_type_from_config_dir(str(config.get("model_path"))),
        }
    )
    runtime_flags = _resolve_effective_runtime_flags(config=config, logger=logger)
    set_seed(int(config["seed"]))

    train_df, val_df = _load_train_and_val_frames(config)
    max_train_samples = int(config.get("max_train_samples", 0) or 0)
    max_val_samples = int(config.get("max_val_samples", 0) or 0)
    if max_train_samples > 0:
        train_df = train_df.iloc[:max_train_samples].reset_index(drop=True)
    if max_val_samples > 0:
        val_df = val_df.iloc[:max_val_samples].reset_index(drop=True)
    teacher_cache = PrecomputedTeacherCache(config["teacher_cache_dir"])

    train_dataset = LatentSFTDataset(
        frame=train_df,
        tokenizer=load_tokenizer(config),
        max_length=int(config["max_length"]),
        seed=int(config["seed"]),
        lazy=bool(config.get("lazy_dataset", False)),
        cot_ce_loss_weight=float(config.get("cot_ce_loss_weight", 0.3)),
        latent_start_ce_loss_weight=float(config.get("latent_start_ce_loss_weight", 1.0)),
        latent_end_ce_loss_weight=float(config.get("latent_end_ce_loss_weight", 1.0)),
        im_end_ce_loss_weight=float(config.get("im_end_ce_loss_weight", 1.0)),
    )
    val_dataset = LatentSFTDataset(
        frame=val_df,
        tokenizer=train_dataset.tokenizer,
        max_length=int(config["max_length"]),
        seed=int(config["seed"]),
        lazy=bool(config.get("lazy_dataset", False)),
        cot_ce_loss_weight=float(config.get("cot_ce_loss_weight", 0.3)),
        latent_start_ce_loss_weight=float(config.get("latent_start_ce_loss_weight", 1.0)),
        latent_end_ce_loss_weight=float(config.get("latent_end_ce_loss_weight", 1.0)),
        im_end_ce_loss_weight=float(config.get("im_end_ce_loss_weight", 1.0)),
    )
    tokenizer = train_dataset.tokenizer
    get_token_constants(tokenizer)
    collator = LatentSFTCollator(tokenizer=tokenizer, config=config)

    if args.dry_run:
        dry_batch_size = 1 if int(config.get("context_parallel_size", 1) or 1) > 1 else min(2, len(train_dataset))
        batch = collator([train_dataset[i] for i in range(int(dry_batch_size))])
        logger.info(
            build_dry_run_payload(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                batch=batch,
                teacher_cache=teacher_cache,
                config=config,
            )
        )
        return

    accelerator = build_accelerator(config)
    logger = setup_logger(
        name="later.train",
        log_level=str(config.get("log_level", "INFO")),
        log_file=log_dir
        / (train_log_file if accelerator.is_main_process else f"train_rank{accelerator.process_index:02d}.log"),
        is_main_process=accelerator.is_main_process,
        rank=accelerator.process_index,
    )
    writer = create_summary_writer(config=config, is_main_process=accelerator.is_main_process)
    logger.info(build_distributed_init_payload(config=config, accelerator=accelerator))
    context_parallel_size = int(config.get("context_parallel_size", 1) or 1)
    latent_cp_enabled = context_parallel_size > 1
    if latent_cp_enabled:
        if int(accelerator.num_processes) != int(context_parallel_size):
            raise ValueError(
                "Latent CP expects data_parallel_size=1 and world_size == context_parallel_size; "
                f"world_size={int(accelerator.num_processes)}, context_parallel_size={context_parallel_size}"
            )
        if int(config.get("micro_batch_size", 1)) != 1 or int(config.get("gradient_accumulation_steps", 1)) != 1:
            raise ValueError("Latent CP requires micro_batch_size=1 and gradient_accumulation_steps=1.")
    resume_plan = resolve_resume_plan(
        config=config,
        train_dataset_len=len(train_dataset),
        num_processes=1 if latent_cp_enabled else int(accelerator.num_processes),
    )
    if bool(resume_plan["enabled"]):
        resume_log_payload = {
            "tag": "resume_counters_restored",
            "enabled": True,
            "checkpoint_dir": str(resume_plan["checkpoint_dir"]),
            "restored_global_step": int(resume_plan["global_step"]),
            "restored_epoch_index": int(resume_plan["epoch_index"]),
            "restored_next_step_in_epoch": int(resume_plan["next_step_in_epoch"]),
            "train_stage": int(resume_plan["train_stage"]),
            "trainable_mode": str(resume_plan["trainable_mode"]),
            "model_weights_restored": True,
            "optimizer_scheduler_fresh_init": True,
        }
        if bool(resume_plan.get("resume_step_overridden", False)):
            resume_log_payload.update(
                {
                    "tag": "resume_counters_overridden",
                    "checkpoint_global_step": int(resume_plan.get("checkpoint_global_step", 0)),
                    "checkpoint_epoch_index": int(resume_plan.get("checkpoint_epoch_index", 0)),
                    "checkpoint_next_step_in_epoch": int(resume_plan.get("checkpoint_next_step_in_epoch", 0)),
                    "override_global_step": int(resume_plan["global_step"]),
                    "steps_per_epoch_current_run": int(resume_plan.get("steps_per_epoch", 0)),
                    "derived_epoch_index": int(resume_plan["epoch_index"]),
                    "derived_next_step_in_epoch": int(resume_plan["next_step_in_epoch"]),
                    "override_step_semantics": "current_run_optimizer_steps",
                }
            )
        logger.info(resume_log_payload)
    logger.info(
        {
            "tag": "model_load_path_resolution",
            "model_path": str(resume_plan["model_load_path"]),
            "model_type_from_config": read_model_type_from_config_dir(str(resume_plan["model_load_path"])),
            "latent_qwen3_registered": True,
        }
    )
    tokenizer = load_tokenizer(config, model_path_override=str(resume_plan["model_load_path"]))
    get_token_constants(tokenizer)
    train_dataset = LatentSFTDataset(
        frame=train_df,
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
        seed=int(config["seed"]),
        lazy=bool(config.get("lazy_dataset", False)),
        cot_ce_loss_weight=float(config.get("cot_ce_loss_weight", 0.3)),
        latent_start_ce_loss_weight=float(config.get("latent_start_ce_loss_weight", 1.0)),
        latent_end_ce_loss_weight=float(config.get("latent_end_ce_loss_weight", 1.0)),
        im_end_ce_loss_weight=float(config.get("im_end_ce_loss_weight", 1.0)),
    )
    val_dataset = LatentSFTDataset(
        frame=val_df,
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
        seed=int(config["seed"]),
        lazy=bool(config.get("lazy_dataset", False)),
        cot_ce_loss_weight=float(config.get("cot_ce_loss_weight", 0.3)),
        latent_start_ce_loss_weight=float(config.get("latent_start_ce_loss_weight", 1.0)),
        latent_end_ce_loss_weight=float(config.get("latent_end_ce_loss_weight", 1.0)),
        im_end_ce_loss_weight=float(config.get("im_end_ce_loss_weight", 1.0)),
    )
    collator = LatentSFTCollator(tokenizer=tokenizer, config=config)
    model, tokenizer, effective_attn_impl, base_vocab_size = load_model_and_tokenizer(
        config=config,
        logger=logger,
        model_path_override=str(resume_plan["model_load_path"]),
    )
    logger.info(
        {
            "tag": "effective_runtime_flags",
            "global_stateless_forward": bool(config.get("global_stateless_forward", True)),
            "legacy_staged_forward_fallback": bool(config.get("legacy_staged_forward_fallback", True)),
            "gradient_checkpointing_requested": bool(runtime_flags["gradient_checkpointing_requested"]),
            "gradient_checkpointing_effective": bool(runtime_flags["gradient_checkpointing_effective"]),
            "attn_implementation_requested": str(runtime_flags["attn_implementation_requested"]),
            "attn_implementation_effective": str(effective_attn_impl),
        }
    )
    if bool(runtime_flags["staged_forward_enabled"]):
        model.config.use_cache = True
        if hasattr(model, "gradient_checkpointing_disable"):
            try:
                model.gradient_checkpointing_disable()
            except Exception:
                pass
    if bool(config.get("gradient_checkpointing", False)):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    base_token_row_freeze_controller: BaseTokenRowFreezeController | None = None
    if bool(config.get("freeze_base_token_rows", False)):
        base_token_row_freeze_controller = configure_base_token_row_freezing(
            model=model,
            base_vocab_size=int(base_vocab_size),
            apply_to_lm_head=bool(config.get("freeze_base_token_rows_apply_to_lm_head", True)),
            freeze_scope=str(config.get("freeze_base_token_rows_scope", "always")),
        )
        embedding_vocab_size = int(model.get_input_embeddings().weight.size(0))
        if int(base_vocab_size) >= embedding_vocab_size:
            raise RuntimeError(
                "Expected tokenizer extension before base-row freezing. "
                f"base_vocab_size={int(base_vocab_size)}, embedding_vocab_size={embedding_vocab_size}"
            )
        token_constants = get_token_constants(tokenizer)
        added_token_ids = [int(token_id) for token_id in token_constants.values() if int(token_id) >= int(base_vocab_size)]
        if not added_token_ids:
            raise RuntimeError(
                "freeze_base_token_rows is enabled but no newly-added special token ids were found above base_vocab_size. "
                f"base_vocab_size={int(base_vocab_size)}, token_constants={token_constants}"
            )
        if not all(int(base_vocab_size) <= token_id < int(len(tokenizer)) for token_id in added_token_ids):
            raise RuntimeError(
                "Detected special token ids outside the expected new-token range. "
                f"base_vocab_size={int(base_vocab_size)}, tokenizer_len={int(len(tokenizer))}, added_token_ids={added_token_ids}"
            )
        model.config.base_vocab_size = int(base_vocab_size)
        model.config.new_token_start = int(base_token_row_freeze_controller.new_token_start)
        model.config.new_token_end = int(base_token_row_freeze_controller.new_token_end)
        logger.info(
            {
                "tag": "freeze_base_token_rows",
                "enabled": True,
                "base_vocab_size": int(base_vocab_size),
                "new_token_start": int(base_token_row_freeze_controller.new_token_start),
                "new_token_end": int(base_token_row_freeze_controller.new_token_end),
                "new_token_count": int(base_token_row_freeze_controller.new_token_count),
                "apply_to_lm_head": bool(base_token_row_freeze_controller.apply_to_lm_head),
                "freeze_scope": str(base_token_row_freeze_controller.freeze_scope),
                "uses_tied_word_embeddings": bool(base_token_row_freeze_controller.uses_tied_word_embeddings),
                "hook_count": int(base_token_row_freeze_controller.hook_count),
            }
        )

    trainable_params = configure_trainable_parameters(model, mode=str(resume_plan["trainable_mode"]))
    trainable_param_count = _count_parameters(model.parameters(), trainable_only=True)
    optimizer_param_count = _count_parameters(trainable_params, trainable_only=False)
    total_param_count = _count_parameters(model.parameters(), trainable_only=False)
    trainable_tensor_count = _count_parameter_tensors(model.parameters(), trainable_only=True)
    if trainable_tensor_count <= 0:
        raise RuntimeError(
            "No trainable parameters after trainable mode setup. "
            f"trainable_param_count={int(trainable_param_count)}, "
            f"trainable_tensor_count={int(trainable_tensor_count)}, "
            f"trainable_modules={_collect_trainable_module_names(model)}"
        )
    if trainable_param_count <= 0:
        logger.warning(
            {
                "tag": "trainable_param_count_zero_but_tensors_present",
                "trainable_param_count": int(trainable_param_count),
                "optimizer_param_count": int(optimizer_param_count),
                "total_param_count": int(total_param_count),
                "trainable_tensor_count": int(trainable_tensor_count),
            }
        )

    optimizer = build_optimizer(
        trainable_params,
        config,
        zero_weight_decay_param_ids=(
            base_token_row_freeze_controller.weight_decay_exempt_parameter_ids(model)
            if base_token_row_freeze_controller is not None
            else None
        ),
    )
    micro_batch_size = max(1, int(config["micro_batch_size"]))
    num_processes = 1 if latent_cp_enabled else max(1, int(accelerator.num_processes))
    global_batch_size = micro_batch_size * num_processes
    steps_per_epoch = max(1, (len(train_dataset) + global_batch_size - 1) // global_batch_size)
    total_steps = int(resume_plan["total_train_steps"])
    scheduler, warmup_steps, scheduler_resume_step, lr_after_scheduler_init = build_scheduler_with_resume(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_ratio=float(config["warmup_ratio"]),
        resume_global_step=int(resume_plan["global_step"]),
    )
    scheduler_last_epoch = int(scheduler_resume_step - 1)

    logger.info(
        build_training_setup_payload(
            config=config,
            trainable_param_count=trainable_param_count,
            optimizer_param_count=optimizer_param_count,
            total_param_count=total_param_count,
            global_batch_size=global_batch_size,
            steps_per_epoch=steps_per_epoch,
            total_steps=total_steps,
        )
    )
    logger.info(
        {
            "tag": "scheduler_resume_alignment",
            "restored_global_step": int(resume_plan["global_step"]),
            "scheduler_resume_step": int(scheduler_resume_step),
            "scheduler_last_epoch": int(scheduler_last_epoch),
            "warmup_steps": int(warmup_steps),
            "total_train_steps": int(total_steps),
            "lr_after_scheduler_init": float(lr_after_scheduler_init),
        }
    )

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    if base_token_row_freeze_controller is not None:
        base_token_row_freeze_controller.bind_runtime(model=model, accelerator=accelerator)
        base_token_row_freeze_controller.sync_weight_decay_exemptions(optimizer)
    if bool(config.get("enable_memory_profile", False)) and torch.cuda.is_available():
        snapshot = build_cuda_memory_snapshot(
            tag="after_accelerator_prepare",
            rank=accelerator.process_index,
            model=model,
            optimizer=optimizer,
            extra={"output_dir": str(config["output_dir"])},
        )
        memory_log_path = Path(config["output_dir"]) / f"memory_profile_rank{accelerator.process_index:02d}.jsonl"
        append_jsonl(memory_log_path, snapshot)
        if bool(config.get("memory_profile_log_to_console", False)):
            logger.info(snapshot)
    trainer_cls = LatentCPSFTTrainer if latent_cp_enabled else LatentSFTTrainer
    trainer = trainer_cls(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collator=collator,
        teacher_cache=teacher_cache,
        config=config,
        logger=logger,
        writer=writer,
        base_token_row_freeze_controller=base_token_row_freeze_controller,
        initial_training_state=resume_plan,
    )
    result = trainer.train()
    if not isinstance(result, TrainLoopResult):
        result = TrainLoopResult(status="completed")
    _cleanup_training_runtime(trainer, model, optimizer, scheduler, accelerator, writer)
    return result


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    attach_timestamp_to_tensorboard_run_name(config)
    max_auto_stage_restarts = max(int(config.get("max_auto_stage_restarts", 1) or 1), 0)
    restart_count = 0
    while True:
        result = _run_training_once(config=config, args=args)
        if str(result.status) != "stage_transition_restart_required":
            return
        if restart_count >= max_auto_stage_restarts:
            raise RuntimeError(
                "Exceeded max_auto_stage_restarts while attempting automatic stage transition restart. "
                f"restart_count={int(restart_count)}, max_auto_stage_restarts={int(max_auto_stage_restarts)}, "
                f"checkpoint_dir={str(result.checkpoint_dir)}"
            )
        restart_count += 1
        config["resume_training"] = True
        if result.checkpoint_dir:
            config["resume_from_checkpoint"] = str(result.checkpoint_dir)
        stage_logger = setup_logger(
            name="later.train",
            log_level=str(config.get("log_level", "INFO")),
            log_file=Path(config.get("log_dir") or config["output_dir"]) / str(config.get("train_log_file", "train.log")),
            is_main_process=True,
            rank=0,
        )
        stage_logger.info(
            {
                "tag": "train_stage_auto_restart_begin",
                "restart_count": int(restart_count),
                "checkpoint_dir": str(result.checkpoint_dir),
                "global_step": int(result.global_step),
                "epoch_index": int(result.epoch_index),
                "next_step_in_epoch": int(result.next_step_in_epoch),
                "target_stage": int(result.target_stage or 2),
            }
        )
        stage_logger.info(
            {
                "tag": "train_stage_auto_restart_resume",
                "restart_count": int(restart_count),
                "resume_training": bool(config.get("resume_training", False)),
                "resume_from_checkpoint": str(config.get("resume_from_checkpoint", "")),
            }
        )


if __name__ == "__main__":
    main()
