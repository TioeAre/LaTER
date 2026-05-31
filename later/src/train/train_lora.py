from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import torch

from later.src.train.dataset import LatentSFTCollator, LatentSFTDataset, load_sft_split
from later.src.train.log import (
    attach_timestamp_to_tensorboard_run_name,
    build_distributed_init_payload,
    build_dry_run_payload,
    build_training_setup_payload,
    create_summary_writer,
    setup_logger,
)
from later.src.train.lora_utils import (
    count_trainable_adapter_parameters,
    read_base_model_path_from_adapter,
    resolve_lora_modules_to_save,
    wrap_model_with_lora,
)
from later.src.train.modeling_latent import read_model_type_from_config_dir, register_latent_qwen3_with_transformers
from later.src.train.train import (
    _cleanup_training_runtime,
    _collect_trainable_module_names,
    _count_parameter_tensors,
    _count_parameters,
    _load_resume_metadata,
    _resolve_effective_runtime_flags,
    _resolve_resume_checkpoint_dir,
    build_accelerator,
    build_optimizer,
    build_scheduler_with_resume,
    compute_total_train_steps,
    configure_base_token_row_freezing,
    get_token_constants,
    load_model_and_tokenizer,
    load_tokenizer,
    validate_backend_config,
)
from later.src.train.trainer import LatentSFTTrainer
from later.src.train.utils import BaseTokenRowFreezeController, PrecomputedTeacherCache, load_yaml, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def normalize_lora_config(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(config)
    normalized["train_stage"] = 2
    normalized["stage2_start_fraction"] = 0.0
    normalized["max_auto_stage_restarts"] = 0
    normalized.setdefault("context_parallel_size", 1)
    normalized.setdefault("freeze_base_token_rows", True)
    normalized.setdefault("freeze_base_token_rows_apply_to_lm_head", True)
    normalized.setdefault("freeze_base_token_rows_scope", "always")
    normalized.setdefault("lora_r", 64)
    normalized.setdefault("lora_alpha", 128)
    normalized.setdefault("lora_dropout", 0.05)
    normalized.setdefault("lora_bias", "none")
    normalized.setdefault("lora_target_modules", "all-linear")
    normalized.setdefault("lora_task_type", "CAUSAL_LM")
    normalized.setdefault("lora_use_rslora", True)
    normalized.setdefault("lora_init_lora_weights", "pissa_niter_16")
    normalized.setdefault("lora_ensure_weight_tying", True)
    normalized.setdefault("lora_modules_to_save", ["latent_projector", "lm_head", "model.embed_tokens"])
    return normalized


def resolve_lora_resume_plan(config: Dict[str, Any], train_dataset_len: int, num_processes: int) -> Dict[str, Any]:
    total_train_steps = compute_total_train_steps(
        config=config,
        train_dataset_len=train_dataset_len,
        num_processes=num_processes,
    )
    if not bool(config.get("resume_training", False)):
        return {
            "enabled": False,
            "checkpoint_dir": None,
            "adapter_load_path": None,
            "base_model_load_path": str(config["model_path"]),
            "tokenizer_load_path": str(config["model_path"]),
            "global_step": 0,
            "epoch_index": 0,
            "next_step_in_epoch": 0,
            "train_stage": 2,
            "trainable_mode": "full",
            "total_train_steps": int(total_train_steps),
        }

    checkpoint_dir = _resolve_resume_checkpoint_dir(config)
    state = _load_resume_metadata(checkpoint_dir)
    base_model_path = read_base_model_path_from_adapter(checkpoint_dir) or str(config["model_path"])
    return {
        "enabled": True,
        "checkpoint_dir": str(checkpoint_dir),
        "adapter_load_path": str(checkpoint_dir),
        "base_model_load_path": str(base_model_path),
        "tokenizer_load_path": str(checkpoint_dir),
        "global_step": int(state["global_step"]),
        "epoch_index": int(state["epoch_index"]),
        "next_step_in_epoch": int(state["next_step_in_epoch"]),
        "train_stage": 2,
        "trainable_mode": "full",
        "total_train_steps": int(total_train_steps),
    }


def main() -> None:
    args = parse_args()
    config = normalize_lora_config(load_yaml(args.config))
    attach_timestamp_to_tensorboard_run_name(config)

    registration_status = register_latent_qwen3_with_transformers()
    alloc_conf = str(config.get("pytorch_cuda_alloc_conf", "")).strip()
    if alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf

    validate_backend_config(config)
    if int(config.get("context_parallel_size", 1) or 1) > 1:
        raise ValueError("train_lora.py currently supports non-CP latent SFT only; set context_parallel_size=1.")

    log_dir = Path(config.get("log_dir") or config["output_dir"])
    train_log_file = str(config.get("train_log_file", "train.log"))
    logger = setup_logger(
        name="later.train_lora",
        log_level=str(config.get("log_level", "INFO")),
        log_file=log_dir / train_log_file,
        is_main_process=True,
        rank=0,
    )
    logger.info(
        {
            "tag": "latent_qwen3_lora_registry",
            "status": dict(registration_status),
            "configured_model_path": str(config.get("model_path")),
            "configured_model_type": read_model_type_from_config_dir(str(config.get("model_path"))),
            "lora_modules_to_save": list(resolve_lora_modules_to_save(config)),
        }
    )
    runtime_flags = _resolve_effective_runtime_flags(config=config, logger=logger)
    set_seed(int(config["seed"]))

    train_df, val_df = load_sft_split(config["train_data"], float(config["val_ratio"]))
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
        batch = collator([train_dataset[i] for i in range(int(min(2, len(train_dataset))))])
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
        name="later.train_lora",
        log_level=str(config.get("log_level", "INFO")),
        log_file=log_dir / (train_log_file if accelerator.is_main_process else f"train_rank{accelerator.process_index:02d}.log"),
        is_main_process=accelerator.is_main_process,
        rank=accelerator.process_index,
    )
    writer = create_summary_writer(config=config, is_main_process=accelerator.is_main_process)
    logger.info(build_distributed_init_payload(config=config, accelerator=accelerator))

    resume_plan = resolve_lora_resume_plan(
        config=config,
        train_dataset_len=len(train_dataset),
        num_processes=max(1, int(accelerator.num_processes)),
    )
    if bool(resume_plan["enabled"]):
        logger.info(
            {
                "tag": "resume_counters_restored",
                "enabled": True,
                "checkpoint_dir": str(resume_plan["checkpoint_dir"]),
                "restored_global_step": int(resume_plan["global_step"]),
                "restored_epoch_index": int(resume_plan["epoch_index"]),
                "restored_next_step_in_epoch": int(resume_plan["next_step_in_epoch"]),
                "train_stage": int(resume_plan["train_stage"]),
                "trainable_mode": str(resume_plan["trainable_mode"]),
                "adapter_load_path": str(resume_plan["adapter_load_path"]),
                "base_model_load_path": str(resume_plan["base_model_load_path"]),
                "optimizer_scheduler_fresh_init": True,
            }
        )

    tokenizer = load_tokenizer(config, model_path_override=str(resume_plan["tokenizer_load_path"]))
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
        model_path_override=str(resume_plan["base_model_load_path"]),
    )
    logger.info(
        {
            "tag": "effective_runtime_flags",
            "gradient_checkpointing_requested": bool(runtime_flags["gradient_checkpointing_requested"]),
            "gradient_checkpointing_effective": bool(runtime_flags["gradient_checkpointing_effective"]),
            "attn_implementation_requested": str(runtime_flags["attn_implementation_requested"]),
            "attn_implementation_effective": str(effective_attn_impl),
            "lora_training": True,
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

    model = wrap_model_with_lora(
        model,
        config,
        logger=logger,
        adapter_path=str(resume_plan["adapter_load_path"]) if resume_plan["adapter_load_path"] else None,
    )

    base_token_row_freeze_controller: BaseTokenRowFreezeController | None = None
    if bool(config.get("freeze_base_token_rows", False)):
        base_token_row_freeze_controller = configure_base_token_row_freezing(
            model=model,
            base_vocab_size=int(base_vocab_size),
            apply_to_lm_head=bool(config.get("freeze_base_token_rows_apply_to_lm_head", True)),
            freeze_scope=str(config.get("freeze_base_token_rows_scope", "always")),
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
            }
        )

    trainable_params = [param for param in model.parameters() if bool(param.requires_grad)]
    trainable_param_count = _count_parameters(model.parameters(), trainable_only=True)
    optimizer_param_count = _count_parameters(trainable_params, trainable_only=False)
    total_param_count = _count_parameters(model.parameters(), trainable_only=False)
    trainable_tensor_count = _count_parameter_tensors(model.parameters(), trainable_only=True)
    if trainable_tensor_count <= 0:
        raise RuntimeError(
            "No trainable parameters after LoRA attachment. "
            f"trainable_modules={_collect_trainable_module_names(model)}"
        )
    logger.info(
        {
            "tag": "lora_trainable_summary",
            "trainable_param_count": int(trainable_param_count),
            "optimizer_param_count": int(optimizer_param_count),
            "total_param_count": int(total_param_count),
            "trainable_tensor_count": int(trainable_tensor_count),
            "trainable_adapter_param_count": int(count_trainable_adapter_parameters(model)),
            "trainable_modules_preview": _collect_trainable_module_names(model),
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
    num_processes = max(1, int(accelerator.num_processes))
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

    trainer = LatentSFTTrainer(
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
    trainer.train()
    _cleanup_training_runtime(trainer, model, optimizer, scheduler, accelerator, writer)


if __name__ == "__main__":
    main()
