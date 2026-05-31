import asyncio
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from transformers import AddedToken


LATENT_THINK_START_TOKEN = "<latent_think>"
LATENT_THINK_END_TOKEN = "</latent_think>"


def _can_use_flash_attention_2() -> Tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is unavailable"

    try:
        import flash_attn  # noqa: F401
    except Exception as exc:
        return False, str(exc)

    return True, ""


def build_role_attn_kwargs(
    config: Dict[str, Any],
    role: str,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    attn_key = f"{role}_attn_implementation"
    default_attn = config.get("attn_implementation")
    attn_impl = str(config.get(attn_key, default_attn) or "").strip()

    if attn_impl == "flash_attention_2":
        can_use_flash_attn, reason = _can_use_flash_attention_2()
        if not can_use_flash_attn:
            fallback_key = f"{role}_fallback_attn_implementation"
            fallback_attn = str(config.get(fallback_key, config.get("fallback_attn_implementation", "sdpa")) or "").strip()
            if logger is not None:
                if fallback_attn:
                    logger.warning(
                        "flash_attention_2 requested for %s, but flash_attn is unavailable (%s). Falling back to %s.",
                        role,
                        reason,
                        fallback_attn,
                    )
                else:
                    logger.warning(
                        "flash_attention_2 requested for %s, but flash_attn is unavailable (%s). "
                        "Proceeding without an explicit attention override.",
                        role,
                        reason,
                    )
            attn_impl = fallback_attn

    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    if config.get("trust_remote_code", False):
        kwargs["trust_remote_code"] = True
    return kwargs


def ensure_latent_think_special_tokens(
    tokenizer: Any,
    model: Optional[Any] = None,
    pad_token: str = "<pad>",
) -> int:
    """Register latent-think boundary tokens as dedicated tokenizer entries."""
    if tokenizer is None:
        raise ValueError("tokenizer is required")

    existing_specials = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    existing_special_text = {str(token) for token in existing_specials}
    merged_specials = list(existing_specials)
    for token in (LATENT_THINK_START_TOKEN, LATENT_THINK_END_TOKEN):
        if token not in existing_special_text:
            merged_specials.append(
                AddedToken(
                    token,
                    special=True,
                    normalized=False,
                    lstrip=False,
                    rstrip=False,
                    single_word=False,
                )
            )
            existing_special_text.add(token)

    special_tokens: Dict[str, Any] = {}
    if len(merged_specials) != len(existing_specials):
        special_tokens["additional_special_tokens"] = merged_specials
    if getattr(tokenizer, "pad_token", None) is None:
        special_tokens["pad_token"] = pad_token

    num_added = 0
    if special_tokens:
        num_added = int(tokenizer.add_special_tokens(special_tokens))
    if model is not None and hasattr(model, "resize_token_embeddings") and hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        current_vocab_size = int(embeddings.weight.size(0)) if embeddings is not None else 0
        target_vocab_size = int(len(tokenizer))
        if current_vocab_size != target_vocab_size:
            model.resize_token_embeddings(target_vocab_size)
    return num_added


def get_registered_token_id(tokenizer: Any, token: str) -> Optional[int]:
    """Return a token id only when `token` is already a dedicated vocab entry."""
    if tokenizer is None or not hasattr(tokenizer, "convert_tokens_to_ids"):
        return None

    token_id = tokenizer.convert_tokens_to_ids(token)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is None or token_id == unk_id:
        return None
    try:
        token_id = int(token_id)
    except Exception:
        return None
    if token_id < 0:
        return None
    return token_id


def validate_latent_think_tokenizer_contract(tokenizer: Any) -> Dict[str, int]:
    """Fail fast when latent-think tokens are not unique dedicated tokenizer entries."""
    ensure_latent_think_special_tokens(tokenizer)

    latent_start_id = get_registered_token_id(tokenizer, LATENT_THINK_START_TOKEN)
    latent_end_id = get_registered_token_id(tokenizer, LATENT_THINK_END_TOKEN)
    if latent_start_id is None or latent_end_id is None:
        raise ValueError("Tokenizer failed to register latent-think boundary tokens as dedicated entries.")

    checks = [
        (LATENT_THINK_START_TOKEN, latent_start_id),
        (LATENT_THINK_END_TOKEN, latent_end_id),
    ]
    for token_text, token_id in checks:
        encoded = tokenizer.encode(token_text, add_special_tokens=False)
        if list(encoded) != [int(token_id)]:
            raise ValueError(
                f"Tokenizer must encode {token_text!r} as exactly one token id. "
                f"Got ids={list(encoded)} expected={[int(token_id)]}."
            )
        decoded = str(tokenizer.decode([int(token_id)], skip_special_tokens=False))
        if decoded != token_text:
            raise ValueError(
                f"Tokenizer must decode id {int(token_id)} back to {token_text!r}. "
                f"Got decoded={decoded!r}."
            )

    adjacency_text = f"{LATENT_THINK_START_TOKEN}<|endoftext|>{LATENT_THINK_END_TOKEN}<think>"
    adjacency_ids = [int(x) for x in tokenizer.encode(adjacency_text, add_special_tokens=False)]
    if adjacency_ids.count(int(latent_start_id)) != 1 or adjacency_ids.count(int(latent_end_id)) != 1:
        raise ValueError(
            "Tokenizer split latent-think boundary tokens in concatenated context. "
            f"adjacency_ids={adjacency_ids}, latent_start_id={int(latent_start_id)}, "
            f"latent_end_id={int(latent_end_id)}."
        )

    return {
        "latent_start_id": int(latent_start_id),
        "latent_end_id": int(latent_end_id),
    }


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return None
        return result
    text = str(value).strip()
    if not text:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def write_jsonl_rows(
    path: Union[str, Path],
    records: Iterable[Any],
    transform: Optional[Callable[[Any], Any]] = None,
) -> int:
    n = 0
    path_obj = Path(path)
    with path_obj.open("w", encoding="utf-8") as f:
        for rec in records:
            payload = transform(rec) if transform is not None else rec
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl_row(
    path: Union[str, Path],
    record: Any,
    transform: Optional[Callable[[Any], Any]] = None,
) -> None:
    path_obj = Path(path)
    payload = transform(record) if transform is not None else record
    with path_obj.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl_rows(path: Union[str, Path], dict_only: bool = False) -> List[Any]:
    path_obj = Path(path)
    rows: List[Any] = []
    if not path_obj.exists():
        return rows
    with path_obj.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if dict_only and not isinstance(row, dict):
                continue
            rows.append(row)
    return rows


def normalize_group_key(value: Any, missing_token: str = "<NA>") -> str:
    """Normalize grouping key values into a stable, hashable string."""
    if value is None:
        return missing_token

    try:
        import pandas as pd

        if pd.api.types.is_scalar(value) and pd.isna(value):
            return missing_token
    except Exception:
        if isinstance(value, float) and math.isnan(value):
            return missing_token

    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), ensure_ascii=False, sort_keys=True)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if hasattr(value, "tolist"):
        try:
            return json.dumps(value.tolist(), ensure_ascii=False, sort_keys=True) # type: ignore
        except Exception:
            return str(value)
    return str(value)


def extract_final_answer(response: str) -> Optional[str]:
    """Extract the final answer from model response.

    Tries multiple formats:
      1. \\boxed{...}
      2. The answer is ...
      3. Last number in the text
    """
    matches = list(re.finditer(r"\\boxed\{", response))
    if matches:
        match = matches[-1]
        start_idx = match.end()
        brace_count = 1
        for i in range(start_idx, len(response)):
            if response[i] == "{":
                brace_count += 1
            elif response[i] == "}":
                brace_count -= 1
            if brace_count == 0:
                return response[start_idx:i].strip()

    pattern = re.search(r"(?:the\s+)?answer\s+is[:\s]*([^\.\n]+)", response, re.IGNORECASE)
    if pattern:
        return pattern.group(1).strip()

    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", response)
    if numbers:
        return numbers[-1]

    return None


def normalize_answer_text(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    normalized = answer.strip().lower()
    normalized = normalized.rstrip(".,;")
    normalized = normalized.replace("$", "").replace("%", "")
    normalized = re.sub(r"(\d),(\d)", r"\1\2", normalized)
    return normalized


def has_latent_think_block(response: str) -> bool:
    # TODO: 还要确保 tag 内有 latent token, 并且 latent token 只在 tag 内
    start_tag = "<latent_think>"
    end_tag = "</latent_think>"
    if start_tag not in response or end_tag not in response:
        return False
    return response.index(end_tag) > response.index(start_tag)


def calculate_entropy(logits: Union[torch.Tensor, Sequence[torch.Tensor], None]) -> torch.Tensor:
    """Calculate the entropy of the logits.

    Args:
        logits: The logits to calculate the entropy of.
            Can be a Tensor of shape [batch, seq_len, vocab_size] or [batch, vocab_size],
            or a Sequence of tensors (e.g. from generate output_logits=True).

    Returns:
        torch.Tensor: The entropy.
    """
    if logits is None:
        raise ValueError("Cannot calculate entropy because logits is None.")

    if isinstance(logits, torch.Tensor):
        next_token_logits = logits[:, -1, :] if logits.dim() == 3 else logits
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        current_step_entropy = -torch.sum(probs * log_probs, dim=-1)
        return current_step_entropy.cpu()
    elif isinstance(logits, (list, tuple)):
        all_scores = torch.stack(list(logits), dim=1).to("cpu")
        probs = torch.nn.functional.softmax(all_scores, dim=-1)
        log_probs = torch.nn.functional.log_softmax(all_scores, dim=-1)
        entropy_tensor = -torch.sum(probs * log_probs, dim=-1)
        return entropy_tensor
    else:
        raise TypeError(f"Unsupported type for logits: {type(logits)}")


def collect_model_attentions(outputs_attentions: Tuple[torch.Tensor]) -> torch.Tensor:
    # 将其沿第0维堆叠，得到形状: [num_layers, batch, heads, q_len, k_len]
    all_layers_attn = torch.stack(outputs_attentions, dim=0)
    num_layers = all_layers_attn.shape[0]
    # 定义你想观察的层索引，例如：底层, 1/4处, 中间层, 3/4处, 顶层
    layer_indices = [0, num_layers // 4, num_layers // 2, num_layers * 3 // 4, num_layers - 1]
    # 提取这些特定层，形状变为: [num_selected_layers, batch, heads, q_len(此时为1), k_len]
    selected_attn = all_layers_attn[layer_indices, :, :, -1, :]
    # 仅对多头(dim=2)求平均，保留特定的层维度
    # 结果形状: [num_selected_layers, batch, k_len]
    curr_attn = selected_attn.mean(dim=2).detach().cpu().half()
    return curr_attn


def logits_to_tokens(next_token_logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature > 0:
        scaled_logits = next_token_logits / temperature
        sample_probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(sample_probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            scaled_logits[indices_to_remove] = -float("Inf")
            sample_probs = torch.nn.functional.softmax(scaled_logits, dim=-1)

        next_tokens = torch.multinomial(sample_probs, num_samples=1).squeeze(-1)
    else:
        next_tokens = torch.argmax(next_token_logits, dim=-1)
    return next_tokens


async def collect_result_async(
    items,
    generated_batch,
    task,
    attention_mask,
    tokens_batch,
    prompts,
    input_ids,
    batch_start,
    tokenizer,
    generated_token_counts=[],
    think_end_indices=[],
    experiment_data={},
    all_generated_ids=[],
    entropies_list=[],
    save_dir="",
    token_types_batch=None,
    gpt_max_concurrency: int = 8,
    persist_results: bool = True,
) -> List[Dict]:
    from loguru import logger

    from later.src.config.project_config import project_config
    from later.src.eval.gpt_evaluator import batch_gpt_acc_async, extract_content_outside_think
    from utils import (
        extract_boxed_answer,
        extract_gsm8k_answer,
        extract_markdown_python_block,
        normalize_answer,
        run_with_timeout,
    )
    EntropyVisualizer = None
    if project_config.DRAW_ENTROPY or persist_results:
        try:
            from later.src.utils.visualizer import EntropyVisualizer as _EntropyVisualizer

            EntropyVisualizer = _EntropyVisualizer
        except Exception as e:
            logger.warning(
                f"collect_result_async visualizer unavailable: {type(e).__name__}: {e}. "
                "Skip entropy/experiment visualization persistence."
            )

    experiment_data_dir = os.path.join(save_dir, "experiment_data")
    results = []
    save_results = {}
    save_path = os.path.join(experiment_data_dir, "save_results.json")
    if persist_results and os.path.exists(save_path):
        with open(save_path, "r") as f:
            try:
                save_results = json.load(f)
            except json.JSONDecodeError:
                save_results = {}

    gpt_tasks = {"math500", "prosqa", "dolci", "distilled_reasoning"}
    gpt_requests = []
    gpt_item_indices = []
    precomputed = {}

    for idx, item in enumerate(items):
        generated_text = generated_batch[idx]
        if task in gpt_tasks:
            if task in {"dolci", "distilled_reasoning"}:
                _, answer = extract_content_outside_think(generated_text)
                pred = answer.strip() if answer else generated_text.strip()
            else:
                pred = normalize_answer(extract_boxed_answer(generated_text))
            gold = item.get("gold", "")
            precomputed[idx] = (pred, gold)
            gpt_item_indices.append(idx)
            gpt_requests.append(
                {
                    "question": item.get("question", ""),
                    "predict_answer": pred if pred is not None else "",
                    "ground_truth": str(gold),
                    "answer_type": str(item.get("answer_type", task)),
                }
            )

    gpt_ok_map = {}
    if gpt_requests:
        try:
            gpt_results = await batch_gpt_acc_async(
                gpt_requests,
                max_concurrency=gpt_max_concurrency,
            )
            for req_idx, item_idx in enumerate(gpt_item_indices):
                acc, _ = gpt_results.get(req_idx, (0.0, ""))
                gpt_ok_map[item_idx] = True if int(acc) == 1 else False
        except Exception as e:
            logger.error(f"collect_result_async batch GPT scoring failed: {type(e).__name__}: {e}")
            for item_idx in gpt_item_indices:
                gpt_ok_map[item_idx] = False

    for idx, item in enumerate(items):
        generated_text = generated_batch[idx]
        if task in ["mbppplus", "humanevalplus"]:
            pred = extract_markdown_python_block(generated_text)
            gold = item.get("gold", "")

            if pred is None:
                ok = False
                error_msg = "python error: No python code block found"
            else:
                python_code_to_exe = pred + "\n" + gold
                ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)

            logger.info("=========================================")
            logger.info(f"Question {idx}")
            logger.info(f"error_msg: {error_msg}")
            # logger.info(f'=========================================')
        elif task in ["aime2024", "aime2025"]:
            pred = normalize_answer(extract_gsm8k_answer(generated_text))
            gold = str(item.get("gold", "")).strip()
            try:
                if pred:
                    pred_int = int(pred)
                else:
                    pred_int = None
                gold_int = int(gold)
                ok = pred_int == gold_int
                error_msg = None
            except ValueError:
                ok = False
                error_msg = f"Value error in parsing answer. Pred: {pred}, Gold: {gold}"
        elif task in gpt_tasks:
            pred, gold = precomputed.get(idx, (None, item.get("gold", "")))
            ok = gpt_ok_map.get(idx, False)
            error_msg = None
        else:
            pred = normalize_answer(extract_gsm8k_answer(generated_text))
            gold = item.get("gold", "")
            ok = (pred == gold) if (pred and gold) else False
            error_msg = None

        mask = attention_mask[idx].bool()
        trimmed_ids = input_ids[idx][mask].to("cpu").tolist()
        agent_trace = {
            "name": "SingleAgent",
            "role": "singleagent",
            "input": prompts[idx],
            "input_ids": trimmed_ids,
            "input_tokens": tokens_batch[idx],
            "output": generated_text,
        }

        problem_idx = batch_start + idx + 1
        entropies = entropies_list[idx] if idx < len(entropies_list) else []
        if token_types_batch is not None and idx < len(token_types_batch):
            current_token_types = token_types_batch[idx]
        else:
            current_token_types = None
        img_path = ""

        if EntropyVisualizer is not None and project_config.DRAW_ENTROPY and idx < len(all_generated_ids):
            full_ids = all_generated_ids[idx]
            img_path = os.path.join(save_dir, f"sample_{problem_idx}_entropy.png")
            EntropyVisualizer.draw_text_with_entropy_from_ids_with_latent(
                tokenizer=tokenizer,
                token_ids=full_ids,
                entropies=entropies,
                save_path=img_path,
                token_types=current_token_types,
                solution=item.get("solution", "None"),
            )

        max_latent_entropy = 0.0
        if current_token_types is not None:
            latent_entropies = [ent for t, ent in zip(current_token_types, entropies) if t == 0]
            if latent_entropies:
                max_latent_entropy = max(latent_entropies)

        if persist_results and EntropyVisualizer is not None:
            EntropyVisualizer.save_experiment_data(
                batch_start=problem_idx,
                experiment_data=experiment_data,
                experiment_data_dir=experiment_data_dir,
                SAVE_STATES=project_config.SAVE_STATES,
                DRAW_ATTENTION=project_config.DRAW_ATTENTION,
            )

        generated_token_count = generated_token_counts[idx] if idx < len(generated_token_counts) else 0
        think_end_index = think_end_indices[idx] if idx < len(think_end_indices) else 0

        results.append(
            {
                "question": item["question"],
                "gold": gold,
                "solution": item["solution"],
                "prediction": pred,
                "raw_prediction": generated_text,
                "agents": [agent_trace],
                "correct": ok,
                "entropy_viz": img_path,
                "avg_entropy": sum(entropies) / len(entropies) if entropies else 0,
                "generated_token_counts": generated_token_count,
                "think_end_indices": think_end_index,
            }
        )
        if persist_results:
            save_results[problem_idx] = {
                "problem_idx": problem_idx,
                "question": item["question"],
                "gold": gold,
                "prediction": pred,
                "correct": ok,
                "switch_cot_type": experiment_data.get("switch_cot_type", [0] * len(items))[idx],
                "entropy_viz": img_path,
                # "avg_entropy": sum(entropies) / len(entropies) if entropies else 0,
                "max_latent_entropy": max_latent_entropy,
                "generated_token_counts": generated_token_count,
                "latent_tokens": generated_token_count
                - sum(current_token_types if current_token_types is not None else []),
                "think_end_indices": think_end_index,
            }
    if persist_results:
        try:
            with open(save_path, "w") as f:
                json.dump(save_results, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"collect_result_async save_results write failed: {type(e).__name__}: {e}")

    return results


def collect_result(
    items,
    generated_batch,
    task,
    attention_mask,
    tokens_batch,
    prompts,
    input_ids,
    batch_start,
    tokenizer,
    generated_token_counts=[],
    think_end_indices=[],
    experiment_data={},
    all_generated_ids=[],
    entropies_list=[],
    save_dir="",
    token_types_batch=None,
    gpt_max_concurrency: int = 8,
    persist_results: bool = True,
) -> List[Dict]:
    return asyncio.run(
        collect_result_async(
            items=items,
            generated_batch=generated_batch,
            task=task,
            attention_mask=attention_mask,
            tokens_batch=tokens_batch,
            prompts=prompts,
            input_ids=input_ids,
            batch_start=batch_start,
            tokenizer=tokenizer,
            generated_token_counts=generated_token_counts,
            think_end_indices=think_end_indices,
            experiment_data=experiment_data,
            all_generated_ids=all_generated_ids,
            entropies_list=entropies_list,
            save_dir=save_dir,
            token_types_batch=token_types_batch,
            gpt_max_concurrency=gpt_max_concurrency,
            persist_results=persist_results,
        )
    )
