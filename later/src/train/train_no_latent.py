from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from later.src.train.dataset import load_sft_split
from later.src.train.dataset_no_latent import NoLatentSFTCollator, NoLatentSFTDataset
from later.src.train.log import (
    attach_timestamp_to_tensorboard_run_name,
    build_distributed_init_payload,
    build_training_setup_payload,
    create_summary_writer,
    setup_logger,
)
from later.src.train.train import (
    _collect_trainable_module_names,
    _count_parameter_tensors,
    _count_parameters,
    _cleanup_training_runtime,
    build_accelerator,
    build_optimizer,
    build_scheduler_with_resume,
    compute_total_train_steps,
    resolve_resume_plan,
    validate_backend_config,
)
from later.src.train.trainer import TrainLoopResult
from later.src.train.trainer_no_latent import NoLatentSFTTrainer
from later.src.train.utils import PrecomputedTeacherCache, load_yaml, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def validate_think_tokenizer_contract(tokenizer: Any) -> Dict[str, int]:
    required = {
        "think_start_id": tokenizer.convert_tokens_to_ids("<think>"),
        "think_end_id": tokenizer.convert_tokens_to_ids("</think>"),
        "im_end_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
    }
    missing = [name for name, token_id in required.items() if token_id is None or int(token_id) < 0]
    if missing:
        raise ValueError(f"Tokenizer is missing no-latent required tokens: {missing}")
    if list(tokenizer.encode("<think>", add_special_tokens=False)) != [int(required["think_start_id"])]:
        raise ValueError("Tokenizer must encode <think> as one dedicated token")
    if list(tokenizer.encode("</think>", add_special_tokens=False)) != [int(required["think_end_id"])]:
        raise ValueError("Tokenizer must encode </think> as one dedicated token")
    return {name: int(value) for name, value in required.items()}


def load_no_latent_tokenizer(config: Dict[str, Any], model_path_override: str | None = None) -> Any:
    model_path = str(model_path_override or config["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "<|endoftext|>"
    validate_think_tokenizer_contract(tokenizer)
    return tokenizer


def load_no_latent_model_and_tokenizer(
    config: Dict[str, Any],
    logger: Any,
    model_path_override: str | None = None,
) -> tuple[Any, Any, str]:
    model_path = str(model_path_override or config["model_path"])
    tokenizer = load_no_latent_tokenizer(config=config, model_path_override=model_path)
    attn_impl = str(config.get("attn_implementation", "sdpa")).lower()
    if attn_impl != "sdpa":
        raise ValueError("No-latent CP-only training requires attn_implementation=sdpa.")
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    max_length = int(config.get("max_length", 0) or 0)
    original_rope_max = int(config.get("rope_original_max_position_embeddings", 32768) or 32768)
    if max_length > original_rope_max:
        factor = float(config.get("rope_scaling_factor", 0.0) or 0.0)
        if factor <= 0.0:
            factor = float(max(2, math.ceil(max_length / max(original_rope_max, 1))))
        rope_scaling = dict(config.get("rope_scaling") or {})
        if not rope_scaling:
            rope_scaling = {
                "rope_type": str(config.get("rope_scaling_type", "yarn")),
                "factor": factor,
                "original_max_position_embeddings": original_rope_max,
            }
        model_config.rope_scaling = rope_scaling
        model_config.max_position_embeddings = max(int(getattr(model_config, "max_position_embeddings", 0) or 0), max_length)
        logger.info(
            {
                "tag": "no_latent_rope_scaling_configured",
                "max_length": int(max_length),
                "model_max_position_embeddings": int(model_config.max_position_embeddings),
                "rope_scaling": dict(model_config.rope_scaling or {}),
            }
        )
    kwargs = {
        "config": model_config,
        "torch_dtype": getattr(torch, str(config["torch_dtype"])),
        "trust_remote_code": True,
        "attn_implementation": attn_impl,
    }
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    return model, tokenizer, str(attn_impl)


def _normalize_no_latent_stage_config(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(config)
    normalized.setdefault("train_stage", 2)
    normalized.setdefault("stage2_start_fraction", 0.0)
    normalized.setdefault("max_auto_stage_restarts", 1)
    return normalized


def resolve_no_latent_resume_plan(
    config: Dict[str, Any],
    train_dataset_len: int,
) -> Dict[str, Any]:
    plan = resolve_resume_plan(
        config=_normalize_no_latent_stage_config(config),
        train_dataset_len=train_dataset_len,
        num_processes=1,
    )
    plan["trainable_mode"] = "full"
    return plan


def _run_training_once(config: Dict[str, Any], args: argparse.Namespace) -> TrainLoopResult:
    validate_backend_config(config)
    if str(config.get("distributed_backend", "")).lower() != "fsdp":
        raise ValueError("No-latent training is CP-only and requires distributed_backend=fsdp.")
    if int(config.get("context_parallel_size", 1) or 1) <= 1:
        raise ValueError("No-latent training is CP-only and requires context_parallel_size > 1.")
    if str(config.get("attn_implementation", "")).lower() != "sdpa":
        raise ValueError("No-latent CP-only training requires attn_implementation=sdpa.")
    if int(config.get("micro_batch_size", 1)) != 1 or int(config.get("gradient_accumulation_steps", 1)) != 1:
        raise ValueError("No-latent CP-only training requires micro_batch_size=1 and gradient_accumulation_steps=1.")

    log_dir = Path(config.get("log_dir") or config["output_dir"])
    train_log_file = str(config.get("train_log_file", "train.log"))
    logger = setup_logger(
        name="later.train_no_latent",
        log_level=str(config.get("log_level", "INFO")),
        log_file=log_dir / train_log_file,
        is_main_process=True,
        rank=0,
    )
    set_seed(int(config["seed"]))

    train_df, val_df = load_sft_split(config["train_data"], float(config["val_ratio"]))
    max_train_samples = int(config.get("max_train_samples", 0) or 0)
    max_val_samples = int(config.get("max_val_samples", 0) or 0)
    if max_train_samples > 0:
        train_df = train_df.iloc[:max_train_samples].reset_index(drop=True)
    if max_val_samples > 0:
        val_df = val_df.iloc[:max_val_samples].reset_index(drop=True)
    teacher_cache = PrecomputedTeacherCache(config["teacher_cache_dir"])

    tokenizer = load_no_latent_tokenizer(config)
    train_dataset = NoLatentSFTDataset(
        frame=train_df,
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
        seed=int(config["seed"]),
        lazy=bool(config.get("lazy_dataset", False)),
        cot_ce_loss_weight=float(config.get("cot_ce_loss_weight", 0.25)),
        im_end_ce_loss_weight=float(config.get("im_end_ce_loss_weight", 1.0)),
    )
    val_dataset = NoLatentSFTDataset(
        frame=val_df,
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
        seed=int(config["seed"]),
        lazy=bool(config.get("lazy_dataset", False)),
        cot_ce_loss_weight=float(config.get("cot_ce_loss_weight", 0.25)),
        im_end_ce_loss_weight=float(config.get("im_end_ce_loss_weight", 1.0)),
    )
    collator = NoLatentSFTCollator(tokenizer=tokenizer, config=config)

    if args.dry_run:
        batch = collator([train_dataset[0]])
        logger.info(
            {
                "tag": "dry_run_no_latent",
                "train_dataset_size": int(len(train_dataset)),
                "val_dataset_size": int(len(val_dataset)),
                "batch_shape": list(batch["input_ids"].shape),
                "loss_pair_slots": int(batch["loss_pair_mask"].size(1)),
                "teacher_kl_pair_slots": int(batch["teacher_kl_pair_mask"].size(1)),
            }
        )
        return

    accelerator = build_accelerator(config)
    logger = setup_logger(
        name="later.train_no_latent",
        log_level=str(config.get("log_level", "INFO")),
        log_file=log_dir
        / (train_log_file if accelerator.is_main_process else f"train_rank{accelerator.process_index:02d}.log"),
        is_main_process=accelerator.is_main_process,
        rank=accelerator.process_index,
    )
    writer = create_summary_writer(config=config, is_main_process=accelerator.is_main_process)
    logger.info(build_distributed_init_payload(config=config, accelerator=accelerator))
    context_parallel_size = int(config.get("context_parallel_size", 1) or 1)
    if int(accelerator.num_processes) != int(context_parallel_size):
        raise ValueError(
            "No-latent CP-only training expects dp_size=1 and world_size == context_parallel_size; "
            f"world_size={int(accelerator.num_processes)}, context_parallel_size={context_parallel_size}"
        )

    resume_plan = resolve_no_latent_resume_plan(
        config=config,
        train_dataset_len=len(train_dataset),
    )
    logger.info(
        {
            "tag": "model_load_path_resolution",
            "model_path": str(resume_plan["model_load_path"]),
            "model_type_from_config": AutoConfig.from_pretrained(
                str(resume_plan["model_load_path"]), trust_remote_code=True
            ).model_type,
            "no_latent_baseline": True,
        }
    )

    tokenizer = load_no_latent_tokenizer(config, model_path_override=str(resume_plan["model_load_path"]))
    train_dataset = NoLatentSFTDataset(
        frame=train_df,
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
        seed=int(config["seed"]),
        lazy=bool(config.get("lazy_dataset", False)),
        cot_ce_loss_weight=float(config.get("cot_ce_loss_weight", 0.25)),
        im_end_ce_loss_weight=float(config.get("im_end_ce_loss_weight", 1.0)),
    )
    val_dataset = NoLatentSFTDataset(
        frame=val_df,
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
        seed=int(config["seed"]),
        lazy=bool(config.get("lazy_dataset", False)),
        cot_ce_loss_weight=float(config.get("cot_ce_loss_weight", 0.25)),
        im_end_ce_loss_weight=float(config.get("im_end_ce_loss_weight", 1.0)),
    )
    collator = NoLatentSFTCollator(tokenizer=tokenizer, config=config)
    model, tokenizer, effective_attn_impl = load_no_latent_model_and_tokenizer(
        config=config,
        logger=logger,
        model_path_override=str(resume_plan["model_load_path"]),
    )
    if bool(config.get("gradient_checkpointing", False)):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    trainable_params = list(model.parameters())
    trainable_param_count = _count_parameters(model.parameters(), trainable_only=True)
    optimizer_param_count = _count_parameters(trainable_params, trainable_only=False)
    total_param_count = _count_parameters(model.parameters(), trainable_only=False)
    trainable_tensor_count = _count_parameter_tensors(model.parameters(), trainable_only=True)
    if trainable_tensor_count <= 0:
        raise RuntimeError(
            "No trainable parameters in no-latent baseline. "
            f"trainable_modules={_collect_trainable_module_names(model)}"
        )

    optimizer = build_optimizer(trainable_params, config)
    global_batch_size = 1
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
            "attn_implementation_effective": str(effective_attn_impl),
        }
    )

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    trainer = NoLatentSFTTrainer(
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
        base_token_row_freeze_controller=None,
        initial_training_state=resume_plan,
    )
    result = trainer.train()
    if not isinstance(result, TrainLoopResult):
        result = TrainLoopResult(status="completed")
    _cleanup_training_runtime(trainer, model, optimizer, scheduler, accelerator, writer)
    return result


def main() -> None:
    args = parse_args()
    config = _normalize_no_latent_stage_config(load_yaml(args.config))
    attach_timestamp_to_tensorboard_run_name(config)
    max_auto_stage_restarts = max(int(config.get("max_auto_stage_restarts", 1) or 1), 0)
    restart_count = 0
    while True:
        result = _run_training_once(config=config, args=args)
        if str(result.status) != "stage_transition_restart_required":
            return
        if restart_count >= max_auto_stage_restarts:
            raise RuntimeError(
                "Exceeded max_auto_stage_restarts while attempting automatic no-latent stage transition restart. "
                f"restart_count={int(restart_count)}, max_auto_stage_restarts={int(max_auto_stage_restarts)}, "
                f"checkpoint_dir={str(result.checkpoint_dir)}"
            )
        restart_count += 1
        config["resume_training"] = True
        if result.checkpoint_dir:
            config["resume_from_checkpoint"] = str(result.checkpoint_dir)
        stage_logger = setup_logger(
            name="later.train_no_latent",
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
                "no_latent_baseline": True,
            }
        )
        stage_logger.info(
            {
                "tag": "train_stage_auto_restart_resume",
                "restart_count": int(restart_count),
                "resume_training": bool(config.get("resume_training", False)),
                "resume_from_checkpoint": str(config.get("resume_from_checkpoint", "")),
                "no_latent_baseline": True,
            }
        )


if __name__ == "__main__":
    main()
