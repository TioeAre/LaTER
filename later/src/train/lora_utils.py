from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import LoraConfig, PeftModel, get_peft_model


def resolve_lora_modules_to_save(config: Dict[str, Any]) -> List[str]:
    configured = config.get("lora_modules_to_save")
    if configured is None:
        return ["latent_projector", "lm_head", "model.embed_tokens"]
    if isinstance(configured, str):
        return [configured]
    return [str(item) for item in list(configured)]


def build_lora_config(config: Dict[str, Any]) -> LoraConfig:
    return LoraConfig(
        r=int(config.get("lora_r", 64)),
        lora_alpha=int(config.get("lora_alpha", 128)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias=str(config.get("lora_bias", "none")),
        target_modules=str(config.get("lora_target_modules", "all-linear")),
        task_type=str(config.get("lora_task_type", "CAUSAL_LM")),
        use_rslora=bool(config.get("lora_use_rslora", True)),
        init_lora_weights=str(config.get("lora_init_lora_weights", "pissa_niter_16")),
        ensure_weight_tying=bool(config.get("lora_ensure_weight_tying", True)),
        modules_to_save=resolve_lora_modules_to_save(config),
    )


def wrap_model_with_lora(
    model: Any,
    config: Dict[str, Any],
    *,
    logger: Any,
    adapter_path: str | None = None,
) -> Any:
    lora_config = build_lora_config(config)
    if adapter_path is not None:
        wrapped = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        action = "resume"
    else:
        wrapped = get_peft_model(model, lora_config)
        action = "fresh"
    if bool(config.get("gradient_checkpointing", False)) and hasattr(wrapped, "enable_input_require_grads"):
        wrapped.enable_input_require_grads()
    if hasattr(wrapped, "tie_weights"):
        try:
            wrapped.tie_weights()
        except Exception:
            pass
    logger.info(
        {
            "tag": "lora_attached",
            "action": str(action),
            "adapter_path": str(adapter_path) if adapter_path is not None else None,
            "target_modules": str(lora_config.target_modules),
            "task_type": str(lora_config.task_type),
            "use_rslora": bool(lora_config.use_rslora),
            "init_lora_weights": str(lora_config.init_lora_weights),
            "ensure_weight_tying": bool(getattr(lora_config, "ensure_weight_tying", False)),
            "modules_to_save": list(resolve_lora_modules_to_save(config)),
            "lora_r": int(lora_config.r),
            "lora_alpha": int(lora_config.lora_alpha),
            "lora_dropout": float(lora_config.lora_dropout),
        }
    )
    return wrapped


def read_base_model_path_from_adapter(adapter_dir: str | Path) -> str | None:
    config_path = Path(adapter_dir) / "adapter_config.json"
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    base_model_name_or_path = payload.get("base_model_name_or_path")
    if base_model_name_or_path is None:
        return None
    return str(base_model_name_or_path)


def count_trainable_adapter_parameters(model: Any) -> int:
    total = 0
    for param in model.parameters():
        if bool(param.requires_grad):
            total += int(param.numel())
    return int(total)

