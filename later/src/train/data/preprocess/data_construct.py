"""Construct two-stage latent-reasoning data with DeepSeek thinking mode.

Stage 1 extracts only high-level insights from the original long reasoning.
Stage 2 asks the model to continue from a selected insight and answer the
problem again, then uses the hidden reasoning trace as the distilled CoT.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
from json_repair import repair_json
from loguru import logger
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase


from later.src.eval.gpt_evaluator import gpt_acc_async
from later.src.utils.utils import (
    append_jsonl_row,
    ensure_latent_think_special_tokens,
    read_jsonl_rows as utils_read_jsonl_rows,
    validate_latent_think_tokenizer_contract,
)


DEFAULT_INPUT_DIR = "data/external/Dolci-Think-SFT-32B_sampled"
DEFAULT_OUTPUT_DIR = "data/latent_reasoning_distill"
DEFAULT_TOKENIZER_PATH = "Qwen/Qwen3-14B"


STAGE1_SYSTEM_PROMPT = """You are an expert reasoning data curator.

Extract only the key insights from the source reasoning.

Rules:
- Return valid JSON only.
- Do not produce a short chain of thought.
- Do not provide any final answers in your response.
- correct_insight must be the coarse but correct high-level solution idea.
- incorrect_insights must only include wrong ideas that are explicitly evidenced in the source outputs.
- Do not invent errors that are not present in the source outputs.
"""


STAGE1_USER_TEMPLATE = """You will receive a problem, one or more source outputs, and optional ground truth.

Return JSON with this schema:
{{
  "task_summary": "short task type description",
  "correct_insight": "2-10 sentences of high-level correct plan",
  "incorrect_insights": [
    {{
      "idea": "2-10 sentences of a wrong high-level plan evidenced in the source outputs",
      "why_wrong": "brief reason it fails"
    }}
  ],
  "source_answer_correct": true,
  "contains_reflection": true
}}

REMEMBER: Only extract insights that reflect on the high-level idea, **do not** give any actual answer in correct_insights or incorrect_insights.

Only give your complete plan to solve the question, do not directly state whether it is right or wrong or other content that has nothing to do with the idea in the correct_insight or incorrect_insights.

Problem prompt:
<<<PROMPT>>>
{prompt}
<<<END_PROMPT>>>

Source outputs:
<<<OUTPUTS>>>
{outputs}
<<<END_OUTPUTS>>>

Ground truth:
<<<GROUND_TRUTH>>>
{ground_truth}
<<<END_GROUND_TRUTH>>>

Metadata:
{metadata_json}
"""


STAGE2_SYSTEM_PROMPT = """You are solving a problem from your previous intuition.

Rules:
- Continue from the your intuition instead of restarting from scratch.
- Your intuition may be correct or incorrect.
- Follow your intuition and finish the reasoning efficiently.
- Keep private reasoning compact and avoid repeated planning.
- In the visible answer, provide the final answer and a brief justification only.
"""


STAGE2_USER_TEMPLATE = """Solve the following problem by continuing from the your intuition.

Your intuition may be correct or incorrect. Do not ignore it. Continue from it and finish the solution.

REMEMBER: The intuition comes from your previous conversation with yourself. It's not the user's iintuition.

Problem prompt:
<<<PROMPT>>>
{prompt}
<<<END_PROMPT>>>

Your Intuition:
<<<INTUITION>>>
{insight_text}
<<<END_INTUITION>>>
"""


@dataclass
class SourceExample:
	uid: str
	prompt: str
	outputs: List[str]
	ground_truth: str
	dataset_source: str
	original_dataset: str
	metadata: Dict[str, Any]


def _setup_logging(output_dir: str) -> None:
	os.makedirs(output_dir, exist_ok=True)
	logs_dir = os.path.join(output_dir, "logs")
	os.makedirs(logs_dir, exist_ok=True)
	logger.remove()
	logger.add(sys.stderr, level="INFO", backtrace=True, diagnose=True)
	logger.add(
		os.path.join(logs_dir, "distill_debug.log"),
		level="DEBUG",
		rotation="10 MB",
		retention="60 days",
		backtrace=True,
		diagnose=True,
	)
	logger.add(
		os.path.join(logs_dir, "distill_error.log"),
		level="ERROR",
		rotation="10 MB",
		retention="60 days",
		backtrace=True,
		diagnose=True,
	)


def _convert(value: Any) -> Any:
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, Path):
		return str(value)
	if hasattr(value, "model_dump"):
		return _convert(value.model_dump())
	if isinstance(value, dict):
		return {key: _convert(item) for key, item in value.items()}
	if isinstance(value, (list, tuple, set)):
		return [_convert(item) for item in value]
	try:
		json.dumps(value)
		return value
	except TypeError:
		return str(value)


def _safe_json_dumps(obj: Any) -> str:
	return json.dumps(_convert(obj), ensure_ascii=False, indent=2)


def _stringify_cell(value: Any) -> str:
	"""递归将原本数据集中的数据类型返回成文本"""
	if value is None:
		return ""
	if isinstance(value, str):
		return value
	if isinstance(value, (list, tuple)):
		parts = [part for part in (_stringify_cell(item).strip() for item in value) if part]
		return "\n\n".join(parts)
	if isinstance(value, dict):
		return _safe_json_dumps(value)
	return str(value)


def _string_list(value: Any) -> List[str]:
	if value is None:
		return []
	if isinstance(value, (list, tuple)):
		return [str(item) for item in value if str(item).strip()]
	text = str(value)
	return [text] if text.strip() else []


def _normalize_ground_truth(value: Any) -> str:
	if isinstance(value, list):
		cleaned = [str(item).strip() for item in value if str(item).strip()]
		return " || ".join(cleaned)
	return _stringify_cell(value).strip()


def _split_think_content(text: str) -> Tuple[str, str]:
	if not text:
		return "", ""
	reasoning_parts: List[str] = []
	visible_parts: List[str] = []
	last_end = 0
	for match in re.finditer(r"<think>.*?</think>", text, flags=re.DOTALL):
		outside = text[last_end:match.start()].strip()
		if outside:
			visible_parts.append(outside)
		inside = re.sub(r"^<think>|</think>$", "", match.group(0).strip(), flags=re.DOTALL).strip()
		if inside:
			reasoning_parts.append(inside)
		last_end = match.end()
	tail = text[last_end:].strip()
	if tail:
		visible_parts.append(tail)
	reasoning = "\n\n".join(reasoning_parts).strip()
	visible = "\n\n".join(visible_parts).strip()
	return reasoning, visible


def _extract_visible_answer(text: str) -> str:
	if not text:
		return ""
	_, visible = _split_think_content(text)
	return visible or text.strip()


def _extract_from_messages(messages: Any) -> Tuple[str, List[str], str]:
	prompt_parts: List[str] = []
	outputs: List[str] = []
	ground_truth = ""
	if isinstance(messages, np.ndarray):
		messages = messages.tolist()
	elif isinstance(messages, tuple):
		messages = list(messages)
	if not isinstance(messages, list):
		return "", [], ""

	for message in messages:
		if not isinstance(message, dict):
			continue
		message_dict = cast(Dict[str, Any], message)
		role = str(message_dict.get("role", "")).strip().lower()
		content = _stringify_cell(message_dict.get("content", "")).strip()
		if not content:
			continue
		if role == "user":
			prompt_parts.append(content)
		elif role == "assistant":
			outputs.append(content)
			ground_truth = _extract_visible_answer(content)

	return "\n\n".join(prompt_parts).strip(), outputs, ground_truth.strip()


def _extract_example_fields(row_dict: Dict[str, Any]) -> Tuple[str, List[str], str]:
	prompt = _stringify_cell(row_dict.get("prompt", "")).strip()
	outputs = _string_list(row_dict.get("outputs", []))
	ground_truth = _normalize_ground_truth(row_dict.get("ground_truth", ""))
	if prompt and outputs:
		return prompt, outputs, ground_truth

	message_prompt, message_outputs, message_ground_truth = _extract_from_messages(row_dict.get("messages", []))
	prompt = prompt or message_prompt
	outputs = outputs or message_outputs
	ground_truth = ground_truth or message_ground_truth
	return prompt, outputs, ground_truth


def _make_uid(row: Dict[str, Any], shard_name: str, row_idx: int) -> str:
	raw = row.get("id") or row.get("custom_id") or row.get("key") or f"{shard_name}:{row_idx}"
	seed = f"{shard_name}::{raw}::{row.get('conversation_hash', '')}"
	return hashlib.md5(seed.encode("utf-8")).hexdigest()


def iter_parquet_rows(input_dir: str, limit: Optional[int] = None) -> Iterator[SourceExample]:
	parquet_dir = Path(input_dir) / "data"
	emitted = 0
	for file_path in sorted(parquet_dir.glob("*.parquet")):
		df = pd.read_parquet(file_path)
		for row_position, (_, row) in enumerate(df.iterrows()):
			row_dict = row.to_dict()
			prompt, outputs, ground_truth = _extract_example_fields(row_dict)
			if not prompt or not outputs:
				continue
			metadata = {
				"dataset": row_dict.get("dataset"),
				"source": row_dict.get("source"),
				"dataset_source": row_dict.get("dataset_source"),
				"original_dataset": row_dict.get("original_dataset"),
				"predicted_label": row_dict.get("predicted_label"),
				"constraint_type": row_dict.get("constraint_type"),
				"constraint": row_dict.get("constraint"),
				"model": row_dict.get("model"),
				"source_file": file_path.name,
				"row_index": row_position,
			}
			yield SourceExample(
				uid=_make_uid(row_dict, file_path.name, row_position),
				prompt=prompt,
				outputs=outputs,
				ground_truth=ground_truth,
				dataset_source=str(row_dict.get("dataset_source", "") or row_dict.get("source", "") or ""),
				original_dataset=str(row_dict.get("original_dataset", "") or row_dict.get("source", "") or ""),
				metadata={key: value for key, value in metadata.items() if value not in (None, "")},
			)
			emitted += 1
			if limit is not None and emitted >= limit:
				return


def load_processed_uids(output_jsonl: str) -> set[str]:
	processed: set[str] = set()
	if not os.path.exists(output_jsonl):
		return processed
	with open(output_jsonl, "r", encoding="utf-8") as fh:
		for line in fh:
			line = line.strip()
			if not line:
				continue
			try:
				uid = json.loads(line).get("uid")
			except json.JSONDecodeError:
				continue
			if uid:
				processed.add(uid)
	return processed


def load_jsonl_rows(path: str) -> List[Dict[str, Any]]:
	return cast(List[Dict[str, Any]], utils_read_jsonl_rows(path, dict_only=True))


def save_jsonl_record(path: str, record: Dict[str, Any]) -> None:
	append_jsonl_row(path, record, transform=_convert)


def load_deferred_samples(path: str) -> List[Dict[str, Any]]:
	if not os.path.exists(path):
		return []
	try:
		with open(path, "r", encoding="utf-8") as fh:
			payload = json.load(fh)
	except Exception as exc:  # noqa: BLE001
		logger.warning(f"Failed to load deferred samples file {path}: {exc}")
		return []

	raw_samples = payload.get("samples", []) if isinstance(payload, dict) else payload
	if not isinstance(raw_samples, list):
		return []

	order: List[str] = []
	items_by_uid: Dict[str, Dict[str, Any]] = {}
	for item in raw_samples:
		if not isinstance(item, dict):
			continue
		uid = str(item.get("uid", "")).strip()
		if not uid:
			continue
		if uid not in items_by_uid:
			order.append(uid)
		items_by_uid[uid] = dict(item)

	return [items_by_uid[uid] for uid in order]


def save_deferred_samples(path: str, samples: Sequence[Dict[str, Any]]) -> None:
	payload = {
		"updated_at": datetime.now(timezone.utc).isoformat(),
		"num_samples": len(samples),
		"samples": [_convert(item) for item in samples],
	}
	with open(path, "w", encoding="utf-8") as fh:
		json.dump(payload, fh, ensure_ascii=False, indent=2)


def _upsert_deferred_sample(
	deferred_samples: List[Dict[str, Any]],
	deferred_uid_to_index: Dict[str, int],
	entry: Dict[str, Any],
) -> None:
	uid = str(entry.get("uid", "")).strip()
	if not uid:
		return
	index = deferred_uid_to_index.get(uid)
	if index is None:
		deferred_uid_to_index[uid] = len(deferred_samples)
		deferred_samples.append(entry)
		return
	deferred_samples[index] = entry


def _remove_deferred_sample(
	deferred_samples: List[Dict[str, Any]],
	deferred_uid_to_index: Dict[str, int],
	uid: str,
) -> None:
	index = deferred_uid_to_index.pop(uid, None)
	if index is None:
		return
	deferred_samples.pop(index)
	deferred_uid_to_index.clear()
	for idx, sample in enumerate(deferred_samples):
		sample_uid = str(sample.get("uid", "")).strip()
		if sample_uid:
			deferred_uid_to_index[sample_uid] = idx


def _build_deferred_entry(
	example: SourceExample,
	status: str,
	reason: str,
	stage2_score: Optional[Any] = None,
) -> Dict[str, Any]:
	entry: Dict[str, Any] = {
		"uid": example.uid,
		"status": status,
		"reason": reason,
		"dataset_source": example.dataset_source,
		"original_dataset": example.original_dataset,
		"updated_at": datetime.now(timezone.utc).isoformat(),
	}
	if stage2_score is not None:
		entry["stage2_score"] = stage2_score
	return entry


def _reorder_pending_examples(pending: Sequence[SourceExample], deferred_uids: set[str]) -> List[SourceExample]:
	if not deferred_uids:
		return list(pending)
	normal: List[SourceExample] = []
	deferred: List[SourceExample] = []
	for example in pending:
		if example.uid in deferred_uids:
			deferred.append(example)
		else:
			normal.append(example)
	return normal + deferred


def export_jsonl_to_parquet(input_jsonl: str, output_parquet: str) -> None:
	rows = load_jsonl_rows(input_jsonl)
	if rows:
		pd.DataFrame(rows).to_parquet(output_parquet, index=False)


def build_outputs_block(outputs: Sequence[str], max_chars_per_output: int) -> str:
	return "\n\n".join(
		f"[OUTPUT {idx}]\n{output[:max_chars_per_output]}" for idx, output in enumerate(outputs)
	)


def build_stage1_messages(example: SourceExample, max_chars_per_output: int) -> List[Dict[str, str]]:
	return [
		{"role": "system", "content": STAGE1_SYSTEM_PROMPT},
		{
			"role": "user",
			"content": STAGE1_USER_TEMPLATE.format(
				prompt=example.prompt,
				outputs=build_outputs_block(example.outputs, max_chars_per_output=max_chars_per_output),
				ground_truth=example.ground_truth or "",
				metadata_json=_safe_json_dumps(example.metadata),
			),
		},
	]


def _hash_fraction(text: str, salt: str = "") -> float:
	digest = hashlib.md5(f"{salt}::{text}".encode("utf-8")).hexdigest()
	return int(digest[:8], 16) / 0xFFFFFFFF


def _score_or_neg_inf(value: Any) -> float:
	try:
		return float(value)
	except (TypeError, ValueError):
		return float("-inf")


def _needs_rerollout(validation: Dict[str, Any], threshold: float) -> bool:
	return _score_or_neg_inf(validation.get("score")) < threshold


def _select_stage2_insight(uid: str, correct_insight: str, incorrect_insights: List[Dict[str, str]]) -> Tuple[str, str, Optional[int]]:
	# randomly select an incorrect insight 10% of the time, otherwise use the correct insight
	use_incorrect = bool(incorrect_insights) and _hash_fraction(uid, salt="incorrect") < 0.1
	if not use_incorrect:
		return correct_insight, "correct_insight", None
	# randomly select one of the incorrect insights based on the uid hash to ensure reproducibility
	index = int(_hash_fraction(uid, salt="incorrect_index") * len(incorrect_insights))
	index = min(index, len(incorrect_insights) - 1)
	return incorrect_insights[index]["idea"], "incorrect_insight", index


def build_stage2_messages(example: SourceExample, stage1_payload: Dict[str, Any]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
	insight_text, insight_type, insight_index = _select_stage2_insight(
		example.uid,
		stage1_payload["correct_insight"],
		stage1_payload["incorrect_insights"],
	)
	selection = {
		"selected_insight_type": insight_type,
		"selected_insight_index": insight_index,
		"selected_insight_text": insight_text,
	}
	return [
		{"role": "system", "content": STAGE2_SYSTEM_PROMPT},
		{
			"role": "user",
			"content": STAGE2_USER_TEMPLATE.format(
				# insight_type=insight_type,
                prompt=example.prompt,
				insight_text=insight_text,
			),
		},
	], selection


def _balanced_json_substring(text: str) -> Optional[str]:
	start = text.find("{")
	if start < 0:
		return None
	depth = 0
	in_string = False
	escaped = False
	for idx in range(start, len(text)):
		char = text[idx]
		if in_string:
			if escaped:
				escaped = False
			elif char == "\\":
				escaped = True
			elif char == '"':
				in_string = False
			continue
		if char == '"':
			in_string = True
		elif char == "{":
			depth += 1
		elif char == "}":
			depth -= 1
			if depth == 0:
				return text[start : idx + 1]
	return None


def _json_loads_with_fallbacks(text: str) -> Dict[str, Any]:
	try:
		return json.loads(text)
	except json.JSONDecodeError as exc:
		logger.warning(f"Trying json_repair after malformed model JSON: {exc}")
		repaired = repair_json(
			text,
			return_objects=True,
			ensure_ascii=False,
			skip_json_loads=True,
		)
		if not isinstance(repaired, dict):
			raise ValueError(f"json_repair did not return a JSON object: {type(repaired).__name__}")
		return repaired


def parse_json_response(text: str) -> Dict[str, Any]:
	cleaned = text.strip()
	if cleaned.startswith("```"):
		cleaned = cleaned.split("\n", 1)[-1]
		cleaned = cleaned.rsplit("```", 1)[0]
	try:
		return _json_loads_with_fallbacks(cleaned)
	except json.JSONDecodeError:
		candidate = _balanced_json_substring(cleaned)
		if candidate is None:
			raise ValueError("No JSON object found in model response.")
		return _json_loads_with_fallbacks(candidate)


def normalize_stage1_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
	incorrect_items = payload.get("incorrect_insights") or []
	normalized_incorrect: List[Dict[str, str]] = []
	if isinstance(incorrect_items, list):
		for item in incorrect_items:
			if isinstance(item, str):
				idea = item.strip()
				if idea:
					normalized_incorrect.append({"idea": idea, "why_wrong": ""})
			elif isinstance(item, dict):
				idea = str(item.get("idea", "")).strip()
				why_wrong = str(item.get("why_wrong", "")).strip()
				if idea:
					normalized_incorrect.append({"idea": idea, "why_wrong": why_wrong})

	correct_insight = str(payload.get("correct_insight", "")).strip()
	if not correct_insight:
		raise ValueError("Missing correct_insight in stage-1 response.")

	return {
		"task_summary": str(payload.get("task_summary", "")).strip(),
		"correct_insight": correct_insight,
		"incorrect_insights": normalized_incorrect,
		"source_answer_correct": payload.get("source_answer_correct"),
		"contains_reflection": bool(payload.get("contains_reflection", False)),
	}


def create_openai_client(base_url: str, api_key: str, timeout: int) -> AsyncOpenAI:
	return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


async def validate_stage2_answer_async(
	*,
	question: str,
	predict_answer: str,
	ground_truth: str,
	answer_type: str = "distilled_reasoning",
	correct_threshold: float = 0.7,
	judge_client: Optional[AsyncOpenAI] = None,
) -> Dict[str, Any]:
	normalized_ground_truth = _extract_visible_answer(ground_truth).strip()
	if not predict_answer.strip() or not normalized_ground_truth:
		return {
			"score": None,
			"is_correct": None,
			"judge_response": "",
			"skipped": True,
			"reason": "empty_answer_or_ground_truth",
		}

	try:
		score, judge_response = await gpt_acc_async(
			question=question,
			predict_answer=predict_answer,
			ground_truth=normalized_ground_truth,
			answer_type=answer_type,
			client=judge_client,
		)
		return {
			"score": score,
			"is_correct": bool(score >= correct_threshold),
			"judge_response": judge_response,
			"skipped": False,
			"reason": "",
		}
	except Exception as exc:  # noqa: BLE001
		logger.error(f"validate_stage2_answer_async failed: {type(exc).__name__}: {exc}")
		return {
			"score": 0.0,
			"is_correct": False,
			"judge_response": "",
			"skipped": False,
			"reason": repr(exc),
		}

async def rollout_stage2_once(
	*,
	example: SourceExample,
	stage1_payload: Dict[str, Any],
	worker_args: Dict[str, Any],
	teacher_client: AsyncOpenAI,
	judge_client: AsyncOpenAI,
	seed: Optional[int],
	correct_threshold: float,
) -> Tuple[List[Dict[str, str]], Dict[str, Any], Dict[str, Any]]:
	stage2_messages, stage2_selection = build_stage2_messages(example, stage1_payload)
	stage2_response = await call_chat_completion(
		client=teacher_client,
		model=worker_args["model"],
		messages=stage2_messages,
		temperature=worker_args["temperature"],
		max_tokens=worker_args["stage2_max_tokens"],
		max_retries=worker_args["max_retries"],
		seed=seed,
	)
	stage2_validation = await validate_stage2_answer_async(
		question=example.prompt,
		predict_answer=stage2_response["content"],
		ground_truth=example.ground_truth,
		correct_threshold=correct_threshold,
		judge_client=judge_client,
	)
	return stage2_messages, stage2_response, {**stage2_selection, "validation": stage2_validation}


async def call_chat_completion(
	*,
	client: AsyncOpenAI,
	model: str,
	messages: List[Dict[str, str]],
	temperature: float,
	max_tokens: int,
	max_retries: int,
	seed: Optional[int],
) -> Dict[str, Any]:
	last_error: Optional[Exception] = None
	for attempt in range(max_retries + 1):
		try:
			request_kwargs: Dict[str, Any] = {
				"model": model,
				"messages": messages,
				"temperature": temperature,
				# "max_tokens": max_tokens,
				"extra_body": {"thinking": {"type": "enabled"}},
				# "reasoning": {"effort": "high"},	# for openai gpt-5-mini
			}
			if seed is not None:
				request_kwargs["seed"] = seed
			response = await client.chat.completions.create(**request_kwargs)
			message = response.choices[0].message
			content = message.content or ""
			reasoning_content = getattr(message, "reasoning_content", "") or ""
			if not reasoning_content and "<think>" in content and "</think>" in content:
				reasoning_content, content = _split_think_content(content)
			return {
				"response_id": getattr(response, "id", None),
				"model": getattr(response, "model", model),
				"usage": _convert(getattr(response, "usage", None)),
				"content": content,
				"reasoning_content": reasoning_content,
			}
		except Exception as exc:  # noqa: BLE001
			last_error = exc
			if attempt >= max_retries:
				break
			await asyncio.sleep(min(60.0, (2 ** attempt) + _hash_fraction(str(exc), salt=str(attempt))))
	raise RuntimeError(f"API request failed after retries: {last_error}")


async def process_one(
	example: SourceExample,
	worker_args: Dict[str, Any],
	teacher_client: AsyncOpenAI,
	judge_client: AsyncOpenAI,
	stage1_teacher_client: AsyncOpenAI,
) -> Dict[str, Any]:
	seed = int(hashlib.md5(example.uid.encode("utf-8")).hexdigest()[:8], 16) if worker_args["use_seed"] else None

	stage1_messages = build_stage1_messages(example, worker_args["max_chars_per_output"])
	stage1_response = await call_chat_completion(
		client=stage1_teacher_client,
		model=worker_args["model"],
		messages=stage1_messages,
		temperature=worker_args["temperature"],
		max_tokens=worker_args["stage1_max_tokens"],
		max_retries=worker_args["max_retries"],
		seed=seed,
	)
	stage1_payload = normalize_stage1_payload(parse_json_response(stage1_response["content"]))

	stage2_messages, stage2_response, stage2_bundle = await rollout_stage2_once(
		example=example,
		stage1_payload=stage1_payload,
		worker_args=worker_args,
		teacher_client=teacher_client,
		judge_client=judge_client,
		seed=seed,
		correct_threshold=worker_args["stage2_correct_threshold"],
	)
	stage2_selection = {
		"selected_insight_type": stage2_bundle["selected_insight_type"],
		"selected_insight_index": stage2_bundle["selected_insight_index"],
		"selected_insight_text": stage2_bundle["selected_insight_text"],
	}
	stage2_validation = stage2_bundle["validation"]

	return {
		"uid": example.uid,
		"question": example.prompt,
		"ground_truth": example.ground_truth,
		"dataset_source": example.dataset_source,
		"original_dataset": example.original_dataset,
		"metadata": example.metadata,
		"source_outputs": example.outputs,
		"stage1": {
			"messages": stage1_messages,
			"task_summary": stage1_payload["task_summary"],
			"correct_insight": stage1_payload["correct_insight"],
			"incorrect_insights": stage1_payload["incorrect_insights"],
			"source_answer_correct": stage1_payload["source_answer_correct"],
			"contains_reflection": stage1_payload["contains_reflection"],
			"reasoning_content": stage1_response["reasoning_content"],
			"content": stage1_response["content"],
			"response_id": stage1_response["response_id"],
			"model": stage1_response["model"],
			"usage": stage1_response["usage"],
		},
		"stage2": {
			"messages": stage2_messages,
			**stage2_selection,
			"distilled_cot": stage2_response["reasoning_content"],
			"answer": stage2_response["content"],
			"validation": stage2_validation,
			"rerollout": {
				"threshold": worker_args["stage2_correct_threshold"],
				"max_attempts": worker_args["stage2_rerollouts"],
				"attempts": [],
			},
			"response_id": stage2_response["response_id"],
			"model": stage2_response["model"],
			"usage": stage2_response["usage"],
		},
	}


async def rerollout_record(
	*,
	example: SourceExample,
	record: Dict[str, Any],
	worker_args: Dict[str, Any],
	teacher_client: AsyncOpenAI,
	judge_client: AsyncOpenAI,
) -> Dict[str, Any]:
	stage2 = record["stage2"]
	stage2_validation = stage2["validation"]
	if not _needs_rerollout(stage2_validation, worker_args["stage2_correct_threshold"]):
		return record

	seed = int(hashlib.md5(example.uid.encode("utf-8")).hexdigest()[:8], 16) if worker_args["use_seed"] else None
	stage1 = record["stage1"]
	stage1_payload = {
		"task_summary": stage1.get("task_summary", ""),
		"correct_insight": stage1.get("correct_insight", ""),
		"incorrect_insights": stage1.get("incorrect_insights", []),
		"source_answer_correct": stage1.get("source_answer_correct"),
		"contains_reflection": stage1.get("contains_reflection", False),
	}

	stage2_messages = stage2["messages"]
	stage2_response: Dict[str, Any] = {
		"reasoning_content": stage2.get("distilled_cot", ""),
		"content": stage2.get("answer", ""),
		"response_id": stage2.get("response_id"),
		"model": stage2.get("model"),
		"usage": stage2.get("usage"),
	}
	stage2_selection: Dict[str, Any] = {
		"selected_insight_type": stage2.get("selected_insight_type"),
		"selected_insight_index": stage2.get("selected_insight_index"),
		"selected_insight_text": stage2.get("selected_insight_text"),
	}

	max_rerollouts = max(0, worker_args["stage2_rerollouts"])
	logger.info(f"{example.uid} start deferred rerollouts")
	found_correct = False
	retry_records: List[Dict[str, Any]] = []
	best_failed: Dict[str, Any] = {
		"messages": stage2_messages,
		"response": stage2_response,
		"selection": stage2_selection,
		"validation": stage2_validation,
	}
	best_failed_score = _score_or_neg_inf(stage2_validation.get("score"))

	for reroll_idx in range(max_rerollouts):
		reroll_seed = None if seed is None else seed + reroll_idx + 1
		candidate_messages, candidate_response, candidate_bundle = await rollout_stage2_once(
			example=example,
			stage1_payload=stage1_payload,
			worker_args=worker_args,
			teacher_client=teacher_client,
			judge_client=judge_client,
			seed=reroll_seed,
			correct_threshold=worker_args["stage2_correct_threshold"],
		)
		candidate_validation = candidate_bundle["validation"]
		retry_records.append(
			{
				"attempt": reroll_idx + 1,
				"score": candidate_validation.get("score"),
				"is_correct": candidate_validation.get("is_correct"),
				"selected_insight_type": candidate_bundle["selected_insight_type"],
				"selected_insight_index": candidate_bundle["selected_insight_index"],
			}
		)
		candidate_score = _score_or_neg_inf(candidate_validation.get("score"))
		if candidate_validation.get("is_correct"):
			stage2_messages = candidate_messages
			stage2_response = candidate_response
			stage2_selection = {
				"selected_insight_type": candidate_bundle["selected_insight_type"],
				"selected_insight_index": candidate_bundle["selected_insight_index"],
				"selected_insight_text": candidate_bundle["selected_insight_text"],
			}
			stage2_validation = candidate_validation
			found_correct = True
			break
		if candidate_score > best_failed_score:
			best_failed = {
				"messages": candidate_messages,
				"response": candidate_response,
				"selection": {
					"selected_insight_type": candidate_bundle["selected_insight_type"],
					"selected_insight_index": candidate_bundle["selected_insight_index"],
					"selected_insight_text": candidate_bundle["selected_insight_text"],
				},
				"validation": candidate_validation,
			}
			best_failed_score = candidate_score

	if not found_correct:
		stage2_messages = best_failed["messages"]
		stage2_response = best_failed["response"]
		stage2_selection = best_failed["selection"]
		stage2_validation = best_failed["validation"]
		logger.info(
			f"{example.uid} all deferred rerollouts below threshold, best score={stage2_validation.get('score')}"
		)

	record["stage2"] = {
		"messages": stage2_messages,
		**stage2_selection,
		"distilled_cot": stage2_response["reasoning_content"],
		"answer": stage2_response["content"],
		"validation": stage2_validation,
		"rerollout": {
			"threshold": worker_args["stage2_correct_threshold"],
			"max_attempts": worker_args["stage2_rerollouts"],
			"attempts": retry_records,
		},
		"response_id": stage2_response["response_id"],
		"model": stage2_response["model"],
		"usage": stage2_response["usage"],
	}
	return record


def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerBase:
	logger.info(f"Loading tokenizer from {tokenizer_path}")
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
	ensure_latent_think_special_tokens(tokenizer)
	validate_latent_think_tokenizer_contract(tokenizer)
	return tokenizer


def count_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
	if not text:
		return 0
	return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def enrich_with_token_stats(
	record: Dict[str, Any],
	tokenizer: PreTrainedTokenizerBase,
	tokenizer_path: str,
) -> Dict[str, Any]:
	source_token_counts = [count_tokens(tokenizer, output) for output in record["source_outputs"]]
	primary_source_tokens = source_token_counts[0] if source_token_counts else 0
	distilled_cot_tokens = count_tokens(tokenizer, record["stage2"]["distilled_cot"] + record["stage2"]["answer"])
	compression_ratio = distilled_cot_tokens / primary_source_tokens if primary_source_tokens else None
	record["token_stats"] = {
		"tokenizer_path": tokenizer_path,
		"source_output_token_counts": source_token_counts,
		"source_primary_output_tokens": primary_source_tokens,
		"distilled_cot_tokens": distilled_cot_tokens,
		"compression_ratio_vs_primary_output": compression_ratio,
		"is_shorter_than_source": distilled_cot_tokens < primary_source_tokens if primary_source_tokens else False,
		"is_half_or_less": distilled_cot_tokens <= primary_source_tokens * 0.5 if primary_source_tokens else False,
	}
	return record


def build_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
	if not rows:
		return {"num_records": 0}

	source_tokens = [row.get("token_stats", {}).get("source_primary_output_tokens", 0) for row in rows]
	distilled_tokens = [row.get("token_stats", {}).get("distilled_cot_tokens", 0) for row in rows]
	ratios = [
		row.get("token_stats", {}).get("compression_ratio_vs_primary_output")
		for row in rows
		if row.get("token_stats", {}).get("compression_ratio_vs_primary_output") is not None
	]
	shorter_flags = [bool(row.get("token_stats", {}).get("is_shorter_than_source", False)) for row in rows]
	half_flags = [bool(row.get("token_stats", {}).get("is_half_or_less", False)) for row in rows]

	summary: Dict[str, Any] = {
		"num_records": len(rows),
		"avg_source_primary_output_tokens": float(np.mean(source_tokens)),
		"median_source_primary_output_tokens": float(np.median(source_tokens)),
		"avg_distilled_cot_tokens": float(np.mean(distilled_tokens)),
		"median_distilled_cot_tokens": float(np.median(distilled_tokens)),
		"avg_compression_ratio": float(np.mean(ratios)) if ratios else None,
		"median_compression_ratio": float(np.median(ratios)) if ratios else None,
		"shorter_than_source_rate": float(np.mean(shorter_flags)),
		"half_or_less_rate": float(np.mean(half_flags)),
	}

	dataset_groups: Dict[str, List[float]] = {}
	for row in rows:
		dataset_name = row.get("original_dataset") or row.get("dataset_source") or "unknown"
		ratio = row.get("token_stats", {}).get("compression_ratio_vs_primary_output")
		if ratio is None:
			continue
		dataset_groups.setdefault(dataset_name, []).append(ratio)

	summary["dataset_compression_ratio"] = {
		name: {
			"count": len(group),
			"avg_ratio": float(np.mean(group)),
			"median_ratio": float(np.median(group)),
		}
		for name, group in sorted(dataset_groups.items())
	}
	return summary


def save_summary(summary_path: str, rows: List[Dict[str, Any]]) -> None:
	with open(summary_path, "w", encoding="utf-8") as fh:
		json.dump(_convert(build_summary(rows)), fh, ensure_ascii=False, indent=2)


def _iter_batches(examples: Sequence[SourceExample], batch_size: int) -> Iterator[List[SourceExample]]:
	for start in range(0, len(examples), batch_size):
		yield list(examples[start : start + batch_size])


async def _close_async_client(client: AsyncOpenAI) -> None:
	close = getattr(client, "close", None)
	if callable(close):
		await close()


def _format_elapsed(seconds: float) -> str:
	seconds_int = max(0, int(seconds))
	hours, remainder = divmod(seconds_int, 3600)
	minutes, secs = divmod(remainder, 60)
	return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _update_worker_status(
	progress_bar: Any,
	task_started_monotonic: Dict[asyncio.Task[Any], float],
	latest_added_at: Optional[datetime],
) -> None:
	now = time.monotonic()
	longest_running = max((now - started for started in task_started_monotonic.values()), default=0.0)
	latest_added_text = latest_added_at.strftime("%H:%M:%S") if latest_added_at is not None else "-"
	progress_bar.set_postfix(
		{
			"running": len(task_started_monotonic),
			"longest": _format_elapsed(longest_running),
			"latest_add": latest_added_text,
		},
		refresh=False,
	)


async def _run_bounded_example(
	*,
	example: SourceExample,
	worker_args: Dict[str, Any],
	stage1_teacher_client: AsyncOpenAI,
	teacher_client: AsyncOpenAI,
	judge_client: AsyncOpenAI,
	semaphore: asyncio.Semaphore,
) -> Tuple[SourceExample, Optional[Dict[str, Any]], Optional[Exception]]:
	async with semaphore:
		try:
			record = await process_one(
				example=example,
				worker_args=worker_args,
				stage1_teacher_client=stage1_teacher_client,
				teacher_client=teacher_client,
				judge_client=judge_client,
			)
			return example, record, None
		except Exception as exc:  # noqa: BLE001
			return example, None, exc


async def _run_bounded_rerollout(
	*,
	example: SourceExample,
	record: Dict[str, Any],
	worker_args: Dict[str, Any],
	teacher_client: AsyncOpenAI,
	judge_client: AsyncOpenAI,
	semaphore: asyncio.Semaphore,
) -> Tuple[SourceExample, Optional[Dict[str, Any]], Optional[Exception]]:
	async with semaphore:
		try:
			updated_record = await rerollout_record(
				example=example,
				record=record,
				worker_args=worker_args,
				teacher_client=teacher_client,
				judge_client=judge_client,
			)
			return example, updated_record, None
		except Exception as exc:  # noqa: BLE001
			return example, None, exc


async def run_distillation_async(args: argparse.Namespace) -> None:
	os.makedirs(args.output_dir, exist_ok=True)
	output_jsonl = os.path.join(args.output_dir, args.output_name + ".jsonl")
	error_jsonl = os.path.join(args.output_dir, args.output_name + ".errors.jsonl")
	output_parquet = os.path.join(args.output_dir, args.output_name + ".parquet")
	summary_json = os.path.join(args.output_dir, args.output_name + ".summary.json")
	deferred_samples_json = args.deferred_samples_file or os.path.join(
		args.output_dir,
		args.output_name + ".deferred.json",
	)

	_setup_logging(args.output_dir)
	processed = load_processed_uids(output_jsonl) if args.resume else set()
	if processed:
		logger.info(f"Found {len(processed)} processed examples in existing output.")

	deferred_samples = load_deferred_samples(deferred_samples_json)
	if processed and deferred_samples:
		deferred_samples = [
			sample
			for sample in deferred_samples
			if str(sample.get("uid", "")).strip() not in processed
		]
	deferred_uid_to_index: Dict[str, int] = {
		str(sample.get("uid", "")).strip(): idx
		for idx, sample in enumerate(deferred_samples)
		if str(sample.get("uid", "")).strip()
	}
	deferred_uids = set(deferred_uid_to_index)
	if deferred_uids:
		logger.info(
			f"Loaded {len(deferred_uids)} deferred samples from previous runs: {deferred_samples_json}"
		)

	pending: List[SourceExample] = []
	for example in iter_parquet_rows(args.input_dir, limit=args.limit):
		if example.uid in processed:
			continue
		pending.append(example)

	pending = _reorder_pending_examples(pending, deferred_uids)
	if args.max_samples > 0:
		pending = pending[: args.max_samples]

	logger.info(f"Pending examples: {len(pending)}")
	if not pending:
		existing_rows = load_jsonl_rows(output_jsonl)
		export_jsonl_to_parquet(output_jsonl, output_parquet)
		save_summary(summary_json, existing_rows)
		save_deferred_samples(deferred_samples_json, deferred_samples)
		return

	tokenizer = await asyncio.to_thread(load_tokenizer, args.tokenizer_path)
	worker_args = {
		"base_url": args.base_url,
		"api_key": args.api_key,
		"model": args.model,
		"temperature": args.temperature,
		"timeout": args.timeout,
		"max_retries": args.max_retries,
		"use_seed": not args.disable_seed,
		"stage1_max_tokens": args.stage1_max_tokens,
		"stage2_max_tokens": args.stage2_max_tokens,
		"stage2_correct_threshold": args.stage2_correct_threshold,
		"stage2_rerollouts": args.stage2_rerollouts,
		"max_chars_per_output": args.max_chars_per_output,
	}
	max_in_flight = max(1, args.workers)
	stage1_teacher_client = create_openai_client(
		base_url=os.getenv("STAGE1_OPENAI_BASE_URL", args.base_url),
		api_key=args.api_key,
		timeout=args.timeout,
	)
	teacher_client = create_openai_client(
		base_url=args.base_url,
		api_key=args.api_key,
		timeout=args.timeout,
	)
	judge_client = create_openai_client(
		base_url=os.getenv("JUDGE_OPENAI_BASE_URL", args.base_url),
		api_key=os.getenv("API_KEY", args.api_key),
		timeout=args.timeout,
	)
	semaphore = asyncio.Semaphore(max(1, args.workers))

	completed = 0
	rerollout_queue: List[Tuple[SourceExample, Dict[str, Any]]] = []
	try:
		with tqdm(
			total=len(pending),
			desc="Current run",
			position=0,
			dynamic_ncols=True,
		) as current_progress, tqdm(
			total=len(processed) + len(pending),
			initial=len(processed),
			desc="Overall",
			position=1,
			dynamic_ncols=True,
		) as overall_progress:
			pending_iter = iter(pending)
			in_flight: set[asyncio.Task[Tuple[SourceExample, Optional[Dict[str, Any]], Optional[Exception]]]] = set()
			task_started_monotonic: Dict[asyncio.Task[Tuple[SourceExample, Optional[Dict[str, Any]], Optional[Exception]]], float] = {}
			latest_added_at: Optional[datetime] = None

			def _submit_main_task() -> bool:
				nonlocal latest_added_at
				try:
					example = next(pending_iter)
				except StopIteration:
					return False
				task = asyncio.create_task(
					_run_bounded_example(
						example=example,
						worker_args=worker_args,
						stage1_teacher_client=stage1_teacher_client,
						teacher_client=teacher_client,
						judge_client=judge_client,
						semaphore=semaphore,
					)
				)
				in_flight.add(task)
				task_started_monotonic[task] = time.monotonic()
				latest_added_at = datetime.now()
				_update_worker_status(current_progress, task_started_monotonic, latest_added_at)
				return True

			while len(in_flight) < max_in_flight and _submit_main_task():
				pass

			while in_flight:
				done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
				for task in done:
					task_started_monotonic.pop(task, None)
					example, record, error = await task
					try:
						if error is not None or record is None:
							await asyncio.to_thread(
								save_jsonl_record,
								error_jsonl,
								{
									"uid": example.uid,
									"question": example.prompt,
									"dataset_source": example.dataset_source,
									"original_dataset": example.original_dataset,
									"error": repr(error),
								},
							)
							_upsert_deferred_sample(
								deferred_samples,
								deferred_uid_to_index,
								_build_deferred_entry(
									example=example,
									status="error",
									reason=repr(error),
								),
							)
							logger.error(f"Failed processing example {example.uid}: {error}")
						elif _needs_rerollout(record["stage2"]["validation"], worker_args["stage2_correct_threshold"]):
							rerollout_queue.append((example, record))
							_upsert_deferred_sample(
								deferred_samples,
								deferred_uid_to_index,
								_build_deferred_entry(
									example=example,
									status="rerollout_pending",
									reason="initial_stage2_below_threshold",
									stage2_score=record["stage2"]["validation"].get("score"),
								),
							)
							logger.info(
								f"Queued deferred rerollout for {example.uid}, score={record['stage2']['validation'].get('score')}"
							)
						else:
							record = await asyncio.to_thread(
								enrich_with_token_stats,
								record,
								tokenizer,
								args.tokenizer_path,
							)
							await asyncio.to_thread(save_jsonl_record, output_jsonl, record)
							_remove_deferred_sample(deferred_samples, deferred_uid_to_index, example.uid)
					finally:
						completed += 1
						current_progress.update(1)
						overall_progress.update(1)
						if completed % args.log_every == 0:
							logger.info(
								f"Completed current run {completed}/{len(pending)} | overall {len(processed) + completed}/{len(processed) + len(pending)}"
							)
						_update_worker_status(current_progress, task_started_monotonic, latest_added_at)

				while len(in_flight) < max_in_flight and _submit_main_task():
					pass

		if rerollout_queue:
			logger.info(f"Start deferred rerollout pass for {len(rerollout_queue)} examples.")
			with tqdm(
				total=len(rerollout_queue),
				desc="Deferred rerollout",
				position=2,
				dynamic_ncols=True,
			) as rerollout_progress:
				reroll_completed = 0
				reroll_iter = iter(rerollout_queue)
				reroll_in_flight: set[asyncio.Task[Tuple[SourceExample, Optional[Dict[str, Any]], Optional[Exception]]]] = set()
				reroll_started_monotonic: Dict[asyncio.Task[Tuple[SourceExample, Optional[Dict[str, Any]], Optional[Exception]]], float] = {}
				reroll_latest_added_at: Optional[datetime] = None

				def _submit_reroll_task() -> bool:
					nonlocal reroll_latest_added_at
					try:
						example, record = next(reroll_iter)
					except StopIteration:
						return False
					task = asyncio.create_task(
						_run_bounded_rerollout(
							example=example,
							record=record,
							worker_args=worker_args,
							teacher_client=teacher_client,
							judge_client=judge_client,
							semaphore=semaphore,
						)
					)
					reroll_in_flight.add(task)
					reroll_started_monotonic[task] = time.monotonic()
					reroll_latest_added_at = datetime.now()
					_update_worker_status(rerollout_progress, reroll_started_monotonic, reroll_latest_added_at)
					return True

				while len(reroll_in_flight) < max_in_flight and _submit_reroll_task():
					pass

				while reroll_in_flight:
					done, reroll_in_flight = await asyncio.wait(
						reroll_in_flight,
						return_when=asyncio.FIRST_COMPLETED,
					)
					for task in done:
						reroll_started_monotonic.pop(task, None)
						example, record, error = await task
						try:
							if error is not None or record is None:
								await asyncio.to_thread(
									save_jsonl_record,
									error_jsonl,
									{
										"uid": example.uid,
										"question": example.prompt,
										"dataset_source": example.dataset_source,
										"original_dataset": example.original_dataset,
										"error": f"rerollout_exception: {repr(error)}",
									},
								)
								_upsert_deferred_sample(
									deferred_samples,
									deferred_uid_to_index,
									_build_deferred_entry(
										example=example,
										status="error",
										reason=f"rerollout_exception: {repr(error)}",
									),
								)
								logger.error(f"Deferred rerollout failed for {example.uid}: {error}")
							elif _needs_rerollout(record["stage2"]["validation"], worker_args["stage2_correct_threshold"]):
								await asyncio.to_thread(
									save_jsonl_record,
									error_jsonl,
									{
										"uid": example.uid,
										"question": example.prompt,
										"dataset_source": example.dataset_source,
										"original_dataset": example.original_dataset,
										"error": "rerollout_unfinished",
										"stage2_score": record["stage2"]["validation"].get("score"),
									},
								)
								_upsert_deferred_sample(
									deferred_samples,
									deferred_uid_to_index,
									_build_deferred_entry(
										example=example,
										status="rerollout_pending",
										reason="rerollout_unfinished",
										stage2_score=record["stage2"]["validation"].get("score"),
									),
								)
								logger.warning(
									f"Deferred rerollout unfinished for {example.uid}, score={record['stage2']['validation'].get('score')}"
								)
							else:
								record = await asyncio.to_thread(
									enrich_with_token_stats,
									record,
									tokenizer,
									args.tokenizer_path,
								)
								await asyncio.to_thread(save_jsonl_record, output_jsonl, record)
								_remove_deferred_sample(deferred_samples, deferred_uid_to_index, example.uid)
						finally:
							reroll_completed += 1
							rerollout_progress.update(1)
							if reroll_completed % args.log_every == 0:
								logger.info(
									f"Deferred rerollout progress {reroll_completed}/{len(rerollout_queue)}"
								)
							_update_worker_status(rerollout_progress, reroll_started_monotonic, reroll_latest_added_at)

					while len(reroll_in_flight) < max_in_flight and _submit_reroll_task():
						pass
	finally:
		await _close_async_client(teacher_client)
		await _close_async_client(stage1_teacher_client)
		await _close_async_client(judge_client)
		save_deferred_samples(deferred_samples_json, deferred_samples)
		logger.info(f"Saved deferred samples ({len(deferred_samples)}) to {deferred_samples_json}")

	rows = load_jsonl_rows(output_jsonl)
	export_jsonl_to_parquet(output_jsonl, output_parquet)
	save_summary(summary_json, rows)
	logger.info(f"Saved records to {output_jsonl}")
	logger.info(f"Saved parquet to {output_parquet}")
	logger.info(f"Saved summary to {summary_json}")
	if os.path.exists(error_jsonl):
		logger.info(f"Failures logged to {error_jsonl}")


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Construct two-stage latent reasoning data")
	parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR, help="Directory containing source parquet shards")
	parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory for JSONL, parquet, and summary outputs")
	parser.add_argument("--output_name", default="distilled_latent_reasoning", help="Base filename for outputs")
	parser.add_argument("--model", required=True, help="Teacher model name, for example deepseek-chat")
	parser.add_argument("--base_url", required=True, help="OpenAI-compatible base URL ending with /v1")
	parser.add_argument("--api_key", default=None, help="API key; if omitted, read from --api_key_env")
	parser.add_argument("--api_key_env", default="OPENAI_API_KEY", help="Environment variable containing the API key")
	parser.add_argument("--tokenizer_path", default=DEFAULT_TOKENIZER_PATH, help="Tokenizer used for CoT compression statistics")
	parser.add_argument("--workers", type=int, default=max(1, min(32, os.cpu_count() or 1)), help="Number of concurrent API workers")
	parser.add_argument(
		"--batch_size",
		type=int,
		default=0,
		help="Deprecated and ignored: scheduler now uses a continuous worker pool controlled by --workers",
	)
	parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
	parser.add_argument("--stage1_max_tokens", type=int, default=2048, help="Max completion tokens for stage 1")
	parser.add_argument("--stage2_max_tokens", type=int, default=8192, help="Max completion tokens for stage 2")
	parser.add_argument("--stage2_correct_threshold", type=float, default=0.7, help="Minimum GPT judge score to accept stage-2 answer")
	parser.add_argument("--stage2_rerollouts", type=int, default=5, help="Max rerollout attempts if stage-2 score is below threshold")
	parser.add_argument("--timeout", type=int, default=1800, help="Per-request timeout in seconds")
	parser.add_argument("--max_retries", type=int, default=8, help="Retry count per request")
	parser.add_argument("--max_chars_per_output", type=int, default=18000, help="Max characters kept from each source output")
	parser.add_argument("--limit", type=int, default=None, help="Optional raw row limit for debugging")
	parser.add_argument("--max_samples", type=int, default=-1, help="Number of pending samples to process; -1 means all")
	parser.add_argument("--log_every", type=int, default=50, help="Progress logging interval")
	parser.add_argument("--resume", action="store_true", help="Resume from existing JSONL output")
	parser.add_argument(
		"--deferred_samples_file",
		default=None,
		help="Local file storing failed or unfinished rerollout samples",
	)
	parser.add_argument("--disable_seed", action="store_true", help="Disable deterministic seed passed to the API")
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()
	if args.api_key is None:
		args.api_key = os.environ.get(args.api_key_env)
	if not args.api_key:
		parser.error("API key is required via --api_key or the environment variable from --api_key_env")
	asyncio.run(run_distillation_async(args))


if __name__ == "__main__":
	main()
