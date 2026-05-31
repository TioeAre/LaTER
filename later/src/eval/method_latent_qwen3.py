import argparse
import asyncio
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from later.src.train.lora_utils import read_base_model_path_from_adapter
from later.src.train.modeling_latent import (
    LEGACY_LATENT_PROJECTOR_KEY_MAPPING,
    LatentQwenForCausalLM,
    read_model_type_from_config_dir,
    register_latent_qwen3_with_transformers,
)
from later.src.train.utils import format_latent_generated_text, get_token_constants, render_prompt_only, build_messages_user_content_only
from later.src.utils.utils import collect_result_async


class LatentQwen3_Method:
    def __init__(
        self,
        *,
        max_new_tokens: int = 22048,
        latent_steps: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 0,
        do_sample: bool = False,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        if args is None:
            raise ValueError("args is required for LatentQwen3_Method")

        self.args = args
        self.max_new_tokens = int(max_new_tokens)
        self.latent_steps = int(latent_steps)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.do_sample = bool(do_sample)
        self.generate_bs = max(1, int(generate_bs))
        self.method_name = "latent_qwen3"
        self.task = args.task
        self.debug_generation = str(os.getenv("LATENT_QWEN3_DEBUG", "0")).strip().lower() in {"1", "true", "yes"}

        registry_status = register_latent_qwen3_with_transformers()
        step_dir, checkpoint_kind = self._validate_step_checkpoint_dir(str(args.model_name))
        self.checkpoint_kind = str(checkpoint_kind)
        self.checkpoint_load_plan = self._build_checkpoint_load_plan(step_dir=step_dir, checkpoint_kind=checkpoint_kind)
        print(
            "[latent_qwen3][registry]",
            {
                "status": registry_status,
                "model_path": str(step_dir),
                "checkpoint_kind": str(checkpoint_kind),
                "base_model_path": self.checkpoint_load_plan.get("base_model_path"),
                "tokenizer_source": self.checkpoint_load_plan.get("tokenizer_path"),
                "model_type_from_config": self.checkpoint_load_plan.get("model_type_from_config"),
            },
        )
        self.model, self.tokenizer = self._load_model_and_tokenizer(
            step_dir=step_dir,
            checkpoint_kind=checkpoint_kind,
            load_plan=self.checkpoint_load_plan,
        )
        self.generate_with_latent_batched_fn = self._resolve_generate_with_latent_batched_fn(self.model)
        self.generate_with_latent_fn = self._resolve_generate_with_latent_fn(self.model)
        self.token_constants = get_token_constants(self.tokenizer)

    def _validate_step_checkpoint_dir(self, model_name: str) -> Tuple[Path, str]:
        step_dir = Path(model_name).expanduser().resolve()
        if not step_dir.exists():
            raise FileNotFoundError(
                f"--model_name must point to an existing step checkpoint directory, got: {step_dir}"
            )
        if not step_dir.is_dir():
            raise ValueError(f"--model_name must be a directory, got: {step_dir}")
        # if re.fullmatch(r"step_\d+", step_dir.name) is None:
            # raise ValueError(f"--model_name must be a concrete step directory like '.../step_0002889', got: {step_dir}")

        has_tokenizer = (step_dir / "tokenizer_config.json").exists() or (step_dir / "tokenizer.json").exists()
        has_full_config = (step_dir / "config.json").exists()
        has_full_weights = (
            (step_dir / "model.safetensors").exists()
            or (step_dir / "model.safetensors.index.json").exists()
            or (step_dir / "pytorch_model.bin").exists()
            or (step_dir / "pytorch_model.bin.index.json").exists()
        )
        has_adapter_config = (step_dir / "adapter_config.json").exists()
        has_adapter_weights = (step_dir / "adapter_model.safetensors").exists() or (step_dir / "adapter_model.bin").exists()
        if has_tokenizer and has_full_config and has_full_weights:
            return step_dir, "full"
        if has_tokenizer and has_adapter_config and has_adapter_weights:
            return step_dir, "lora"

        full_missing = []
        if not has_full_config:
            full_missing.append("config.json")
        if not has_tokenizer:
            full_missing.append("tokenizer_config.json/tokenizer.json")
        if not has_full_weights:
            full_missing.append("model.safetensors(.index.json)/pytorch_model.bin(.index.json)")

        lora_missing = []
        if not has_adapter_config:
            lora_missing.append("adapter_config.json")
        if not has_tokenizer:
            lora_missing.append("tokenizer_config.json/tokenizer.json")
        if not has_adapter_weights:
            lora_missing.append("adapter_model.safetensors/adapter_model.bin")
        raise ValueError(
            "Invalid checkpoint directory: "
            f"{step_dir}. Expected either a full latent checkpoint or a LoRA adapter checkpoint. "
            f"full_missing={full_missing} lora_missing={lora_missing}"
        )

    def _build_checkpoint_load_plan(self, step_dir: Path, checkpoint_kind: str) -> Dict[str, Any]:
        if str(checkpoint_kind) == "full":
            return {
                "checkpoint_kind": "full",
                "model_load_path": str(step_dir),
                "tokenizer_path": str(step_dir),
                "base_model_path": str(step_dir),
                "adapter_path": None,
                "model_type_from_config": read_model_type_from_config_dir(step_dir),
            }
        if str(checkpoint_kind) != "lora":
            raise ValueError(f"Unsupported checkpoint kind: {checkpoint_kind}")
        base_model_path = read_base_model_path_from_adapter(step_dir)
        if not base_model_path:
            raise ValueError(
                f"LoRA checkpoint is missing a valid base_model_name_or_path in adapter_config.json: {step_dir}"
            )
        return {
            "checkpoint_kind": "lora",
            "model_load_path": str(base_model_path),
            "tokenizer_path": str(step_dir),
            "base_model_path": str(base_model_path),
            "adapter_path": str(step_dir),
            "model_type_from_config": read_model_type_from_config_dir(base_model_path),
        }

    def _load_base_latent_model(
        self,
        *,
        model_path: str,
        preferred_attn: str,
    ) -> Tuple[LatentQwenForCausalLM, Dict[str, Any]]:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        try:
            model, loading_info = LatentQwenForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                device_map="auto",
                attn_implementation=preferred_attn,
                key_mapping=LEGACY_LATENT_PROJECTOR_KEY_MAPPING,
                output_loading_info=True,
            ) # type: ignore
        except Exception as exc:
            if preferred_attn == "flash_attention_2":
                print(
                    "[latent_qwen3][load_warning]",
                    {
                        "message": "flash_attention_2 load failed, falling back to sdpa",
                        "model_path": str(model_path),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
                model, loading_info = LatentQwenForCausalLM.from_pretrained(
                    str(model_path),
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    device_map="auto",
                    attn_implementation="sdpa",
                    key_mapping=LEGACY_LATENT_PROJECTOR_KEY_MAPPING,
                    output_loading_info=True,
                ) # type: ignore
            else:
                raise
        return model, dict(loading_info)

    def _load_model_and_tokenizer(
        self,
        *,
        step_dir: Path,
        checkpoint_kind: str,
        load_plan: Dict[str, Any],
    ) -> Tuple[Any, AutoTokenizer]:
        preferred_attn = "flash_attention_2" if torch.cuda.is_available() else "sdpa"
        model, loading_info = self._load_base_latent_model(
            model_path=str(load_plan["model_load_path"]),
            preferred_attn=preferred_attn,
        )
        if str(checkpoint_kind) == "lora":
            model = PeftModel.from_pretrained(model, str(load_plan["adapter_path"]), is_trainable=False)
        tokenizer = AutoTokenizer.from_pretrained(str(load_plan["tokenizer_path"]), trust_remote_code=True, use_fast=True)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<pad>"})
                model.resize_token_embeddings(len(tokenizer))
        model.eval()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True
        unexpected_keys = list(loading_info.get("unexpected_keys", []))
        missing_keys = list(loading_info.get("missing_keys", []))
        projector_related = [
            key
            for key in [*unexpected_keys, *missing_keys]
            if key.startswith("latent_projector.")
        ]
        if projector_related:
            print(
                "[latent_qwen3][load_warning]",
                {
                    "unexpected_keys": unexpected_keys,
                    "missing_keys": missing_keys,
                },
            )
        return model, tokenizer

    def _resolve_generate_with_latent_fn(self, model: Any) -> Any:
        direct = getattr(model, "generate_with_latent", None)
        if callable(direct):
            return direct
        if hasattr(model, "get_base_model"):
            base_model = model.get_base_model()
            base_callable = getattr(base_model, "generate_with_latent", None)
            if callable(base_callable):
                return base_callable
        raise AttributeError(f"Model does not expose generate_with_latent: {type(model).__name__}")

    def _resolve_generate_with_latent_batched_fn(self, model: Any) -> Any | None:
        direct = getattr(model, "_generate_with_latent_batched", None)
        if callable(direct):
            return direct
        if hasattr(model, "get_base_model"):
            base_model = model.get_base_model()
            base_callable = getattr(base_model, "_generate_with_latent_batched", None)
            if callable(base_callable):
                return base_callable
        return None

    def _render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def _prepare_chat_batch(
        self, batch_messages: List[List[Dict]], add_generation_prompt: bool = True
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        prompts: List[str] = []
        for messages in batch_messages:
            user_content = ""
            for message in messages:
                if str(message.get("role", "")).lower() == "user":
                    user_content = str(message.get("content", ""))
                    break
            if add_generation_prompt:
                # Match training-time prompt shape: user turn + assistant prefix only.
                prompts.append(render_prompt_only(user_content=user_content))
            else:
                prompts.append(user_content)
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ) # type: ignore
        input_ids = encoded["input_ids"].to(self.model.device)
        attention_mask = encoded["attention_mask"].to(self.model.device)
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids)) # type: ignore
        return prompts, input_ids, attention_mask, tokens_batch

    async def run_batch_async(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_messages = [
            # [{"role": "user", "content": str(item["question"])}]
            [{"role": "user", "content": build_messages_user_content_only(str(item["question"]), self.task)}]
            for item in items
        ]
        prompts, input_ids, attention_mask, tokens_batch = self._prepare_chat_batch(
            batch_messages, add_generation_prompt=True
        )

        generated_batch: List[str] = []
        all_generated_ids: List[List[int]] = []
        generated_token_counts: List[int] = []
        think_end_indices: List[int] = []
        think_end_id = int(self.token_constants["think_end_id"])
        prompt_ids_batch: List[List[int]] = []
        for idx in range(len(items)):
            mask = attention_mask[idx].bool()
            prompt_ids_batch.append(input_ids[idx][mask].to("cpu").tolist())

        if self.generate_with_latent_batched_fn is not None:
            generated = self.generate_with_latent_batched_fn(
                prompt_ids_batch=prompt_ids_batch,
                token_constants=self.token_constants,
                max_new_tokens=self.max_new_tokens,
                latent_max_steps=self.latent_steps,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )
            token_ids_batch = [[int(x) for x in token_ids] for token_ids in generated.token_ids_batch]
            latent_steps_batch = [int(x) for x in generated.latent_steps_batch]
            cot_tokens_batch = [int(x) for x in generated.cot_tokens_batch]
            stopped_normally_batch = [bool(x) for x in generated.stopped_normally_batch]
        else:
            token_ids_batch = []
            latent_steps_batch = []
            cot_tokens_batch = []
            stopped_normally_batch = []
            for prompt_ids in prompt_ids_batch:
                single_generated = self.generate_with_latent_fn(
                    prompt_ids=prompt_ids,
                    token_constants=self.token_constants,
                    max_new_tokens=self.max_new_tokens,
                    latent_max_steps=self.latent_steps,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                )
                token_ids_batch.append([int(x) for x in single_generated.token_ids])
                latent_steps_batch.append(int(single_generated.latent_steps))
                cot_tokens_batch.append(int(single_generated.cot_tokens))
                stopped_normally_batch.append(bool(single_generated.stopped_normally))

        for idx, token_ids in enumerate(token_ids_batch):
            all_generated_ids.append(list(token_ids))
            raw_decoded = self.tokenizer.decode(token_ids, skip_special_tokens=False) # type: ignore
            if self.debug_generation:
                print(
                    "[latent_qwen3][generation_debug]",
                    {
                        "sample_index": idx,
                        "prompt_len": len(prompt_ids_batch[idx]),
                        "generated_len": len(token_ids),
                        "latent_steps": int(latent_steps_batch[idx]),
                        "cot_tokens": int(cot_tokens_batch[idx]),
                        "stopped_normally": bool(stopped_normally_batch[idx]),
                        "do_sample": bool(self.do_sample),
                        "temperature": float(self.temperature),
                        "top_p": float(self.top_p),
                        "top_k": int(self.top_k),
                        "token_ids_head": token_ids[:64],
                        "raw_decoded_head": raw_decoded[:400],
                    },
                )
            generated_batch.append(raw_decoded)
            generated_token_counts.append(len(token_ids))
            think_end_indices.append(token_ids.index(think_end_id) if think_end_id in token_ids else -1)

        return await collect_result_async(
            items=items,
            generated_batch=generated_batch,
            task=self.task,
            attention_mask=attention_mask,
            tokens_batch=tokens_batch,
            prompts=prompts,
            input_ids=input_ids,
            batch_start=0,
            tokenizer=self.tokenizer,
            generated_token_counts=generated_token_counts,
            think_end_indices=think_end_indices,
            experiment_data={},
            all_generated_ids=all_generated_ids,
            entropies_list=[],
            save_dir="",
            token_types_batch=None,
            persist_results=False,
        )

    def run_batch(self, items: List[Dict]) -> List[Dict]:
        return asyncio.run(self.run_batch_async(items))

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
