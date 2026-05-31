"""
Preprocess distilled latent reasoning data for verl-compatible SFT/RL training.

Major capabilities:
1) Resumable progress (byte-offset checkpoints) for long JSONL preprocessing.
2) Distilled data filtering, difficulty bucketing, curriculum sorting.
3) SFT samples with <latent_think>...</latent_think> + <think>...</think>.
4) Latent-state alignment metadata for Markov-style training objectives.
5) RL (GRPO) prompt data construction from GooseReason-0.7M.

Notes:
- This script intentionally writes JSONL as the stable intermediate/final artifact.
- If parquet dependencies are available, parquet files are also exported.
- Several training-side requirements (custom losses, state alignment backprop) are
  represented as explicit metadata fields for downstream trainer consumption.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

from later.src.utils.utils import (
    append_jsonl_row,
    ensure_latent_think_special_tokens,
    read_jsonl_rows,
    safe_float,
    safe_int,
    validate_latent_think_tokenizer_contract,
    write_jsonl_rows,
)


try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

pa: Any = None
pq: Any = None
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pass

pd: Any = None
try:
    import pandas as pd
except Exception:
    pass


LATENT_THINK_START = "<latent_think>"
LATENT_THINK_END = "</latent_think>"
THINK_START = "<think>"
THINK_END = "</think>"

DEFAULT_DISTILLED_INPUT = (
    "data/latent_reasoning_distill_ds/"
    "distilled_latent_reasoning.jsonl,"
    "data/latent_reasoning_distill/"
    "distilled_latent_reasoning.jsonl"
)
DEFAULT_GOOSE_DIR = "data/external/Nemotron-Research-GooseReason-0.7M"

STATE_VERSION = 3
BLOCKED_DATA_TYPE_KEYWORDS = (
    "olmo identity hardcoded data",
)
MAX_ALLOWED_LATENT_STEPS = 256
DISTILLED_SOURCE_DS = "latent_reasoning_distill_ds"
DISTILLED_SOURCE_ORIGINAL = "latent_reasoning_distill"


def utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def hash_config(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compact_text(text: str) -> str:
    # text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_for_match(value: Any) -> str:
    return normalize_space(str(value or "")).lower()


def should_filter_blocked_data_type(row: Dict[str, Any]) -> bool:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    fields = [
        row.get("dataset_source", ""),
        row.get("original_dataset", ""),
        row.get("dataset", ""),
        row.get("source", ""),
        metadata.get("dataset_source", ""),
        metadata.get("original_dataset", ""),
        metadata.get("dataset", ""),
        metadata.get("source", ""),
    ]

    candidates = [_normalize_for_match(field) for field in fields if str(field or "").strip()]
    for candidate in candidates:
        if any(keyword in candidate for keyword in BLOCKED_DATA_TYPE_KEYWORDS):
            return True
    return False


def option_label(index: int) -> str:
    label = ""
    current = index
    while current >= 0:
        current, remainder = divmod(current, 26)
        label = chr(ord("a") + remainder) + label
        current -= 1
    return label


def format_options_for_prompt(options: Any) -> str:
    if isinstance(options, str):
        text = options.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    options = parsed
            except Exception:
                pass

    if not isinstance(options, list) or not options:
        return ""

    # Choose a case for the whole group: all-upper or all-lower
    case_upper = random.choice([True, False])

    formatted: List[str] = []
    for idx, option in enumerate(options):
        option_text = option
        option_text = compact_text(str(option_text))
        if option_text:
            label = option_label(idx)
            label = label.upper() if case_upper else label.lower()
            formatted.append(f"{label}. {option_text}")

    if not formatted:
        return ""

    return "\nOptions:\n" + "\n".join(formatted)


class TokenCounter:
    """Tokenizer-backed counter with a safe fallback."""

    def __init__(self, tokenizer_name_or_path: str):
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer = None
        self.mode = "whitespace_fallback"

        if AutoTokenizer is None:
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
            ensure_latent_think_special_tokens(self.tokenizer)
            validate_latent_think_tokenizer_contract(self.tokenizer)
            self.mode = "hf_tokenizer"
        except Exception:
            self.tokenizer = None

    def count(self, text: str) -> int:
        text = text or ""
        if self.tokenizer is not None:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            return len(ids)
        if not text.strip():
            return 0
        # Fallback: coarse approximation, still deterministic.
        return len(text.split())


def normalize_selected_insight_text(raw: Any) -> str:
    """
    Requirement #9:
    - If selected_insight_text is a list[str], join by spaces.
    - If it is a string that encodes a JSON list, json.loads then join by spaces.
    """
    if raw is None:
        return ""

    if isinstance(raw, list):
        return normalize_space(" ".join(str(x).strip() for x in raw if str(x).strip()))

    text = str(raw).strip()
    if not text:
        return ""

    # Stringified JSON list case.
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return normalize_space(" ".join(str(x).strip() for x in parsed if str(x).strip()))
        except Exception:
            pass

    return normalize_space(text)


def replace_user_intuition_phrases(text: str) -> str:
    """
    Requirement #10:
    Replace variants of user's intuition with my intuition while preserving
    reasonable capitalization.
    """
    if not text:
        return ""

    pattern = re.compile(r"\b(?:the\s+)?user[\'’]s intuition\b", flags=re.IGNORECASE)

    def repl(match: re.Match[str]) -> str:
        matched = match.group(0)
        # Uppercase-leading variant -> "My intuition", else "my intuition".
        if matched and matched[0].isupper():
            return "My intuition"
        return "my intuition"

    return pattern.sub(repl, text)


def difficulty_bucket(is_correct: bool, distilled_cot_tokens: int) -> Tuple[str, int]:
    """
    Requirement #2:
    - incorrect => hard
    - correct and < 500 => easy
    - correct and 500~8192 => medium
    - > 8192 => hard
    """
    if not is_correct:
        return "hard", 2
    if distilled_cot_tokens < 500:
        return "easy", 0
    if distilled_cot_tokens <= 8192:
        return "medium", 1
    return "hard", 2


def maybe_build_parquet(records: List[Dict[str, Any]], parquet_path: Path) -> Tuple[bool, str]:
    """Best-effort parquet writer."""
    if not records:
        return False, "no records"

    # Prefer pyarrow for nested structs/lists.
    if pa is not None and pq is not None:
        try:
            table = pa.Table.from_pylist(records)
            pq.write_table(table, str(parquet_path))
            return True, "pyarrow"
        except Exception as exc:
            return False, f"pyarrow_failed: {exc}"

    if pd is not None:
        try:
            df = pd.DataFrame(records)
            df.to_parquet(str(parquet_path), index=False)
            return True, "pandas"
        except Exception as exc:
            return False, f"pandas_failed: {exc}"

    return False, "missing pyarrow/pandas"


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    return write_jsonl_rows(path, records)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    append_jsonl_row(path, record)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return cast(List[Dict[str, Any]], read_jsonl_rows(path, dict_only=True))


def parse_distilled_input_paths(raw_input: str) -> List[Path]:
    paths = [Path(part.strip()).resolve() for part in str(raw_input).split(",") if part.strip()]
    if not paths:
        raise ValueError("At least one distilled input path must be provided.")
    if len(paths) > 2:
        raise ValueError("At most two distilled input paths are supported.")
    return paths


def classify_distilled_input_path(path: Path) -> str:
    path_str = str(path)
    if DISTILLED_SOURCE_DS in path_str:
        return DISTILLED_SOURCE_DS
    if DISTILLED_SOURCE_ORIGINAL in path_str:
        return DISTILLED_SOURCE_ORIGINAL
    raise ValueError(
        "Dual distilled input mode only supports paths containing "
        f"'{DISTILLED_SOURCE_DS}' or '{DISTILLED_SOURCE_ORIGINAL}': {path}"
    )


def infer_distilled_input_source_type(path: Path, allow_unknown_single: bool = False) -> str:
    try:
        return classify_distilled_input_path(path)
    except ValueError:
        if allow_unknown_single:
            return "single_input"
        raise


def validate_distilled_input_paths(paths: List[Path]) -> List[Path]:
    resolved = [path.resolve() for path in paths]
    for path in resolved:
        if not path.exists():
            raise FileNotFoundError(f"Distilled input not found: {path}")
        if not path.is_file():
            raise ValueError(f"Distilled input must be a file: {path}")

    if len(resolved) == 2:
        source_types = {classify_distilled_input_path(path) for path in resolved}
        expected = {DISTILLED_SOURCE_DS, DISTILLED_SOURCE_ORIGINAL}
        if source_types != expected:
            raise ValueError(
                "Dual input mode requires exactly one ds distilled file and one original distilled file."
            )

    return resolved


def extract_row_ratio(row: Dict[str, Any]) -> float:
    token_stats = row.get("token_stats", {}) if isinstance(row.get("token_stats"), dict) else {}
    return safe_float(token_stats.get("compression_ratio_vs_primary_output"), default=999.0)


def extract_stage2_is_correct(row: Dict[str, Any]) -> bool:
    stage2 = row.get("stage2", {}) if isinstance(row.get("stage2"), dict) else {}
    validation = stage2.get("validation", {}) if isinstance(stage2.get("validation"), dict) else {}
    return bool(validation.get("is_correct", False))


def resolve_row_uid(row: Dict[str, Any], source_path: Path, line_no: int) -> str:
    uid = compact_text(str(row.get("uid", "") or ""))
    if uid:
        return uid
    question = compact_text(str(row.get("question", "") or ""))
    ground_truth = compact_text(str(row.get("ground_truth", "") or ""))
    uid_seed = f"{source_path.resolve()}:{line_no}||{question}||{ground_truth}"
    return hashlib.md5(uid_seed.encode("utf-8")).hexdigest()


def build_sft_record(
    row: Dict[str, Any],
    token_counter: TokenCounter,
    compression_ratio_threshold: float,
    max_distilled_cot_tokens: int,
    latent_pad_token: str,
    latent_min: int,
    latent_max: int,
    cot_loss_weight: float,
    answer_loss_weight: float,
    state_align_loss_weight: float,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Convert one distilled row -> SFT sample.

    Returns (record, reason).
    reason in {
        "ok",
        "filtered_blocked_data_type",
        "filtered_ratio",
        "filtered_distilled_cot_tokens_ge_threshold",
        "filtered_latent_steps_gt_256",
        "missing_question",
        "missing_cot",
        "missing_answer",
    }
    """
    if should_filter_blocked_data_type(row):
        return None, "filtered_blocked_data_type"

    token_stats = row.get("token_stats", {}) if isinstance(row.get("token_stats"), dict) else {}
    stage1 = row.get("stage1", {}) if isinstance(row.get("stage1"), dict) else {}
    stage2 = row.get("stage2", {}) if isinstance(row.get("stage2"), dict) else {}
    validation = stage2.get("validation", {}) if isinstance(stage2.get("validation"), dict) else {}

    ratio = safe_float(token_stats.get("compression_ratio_vs_primary_output"), default=999.0)
    if ratio >= compression_ratio_threshold:
        return None, "filtered_ratio"

    question = compact_text(str(row.get("question", "") or ""))
    if not question:
        return None, "missing_question"

    distilled_cot = compact_text(str(stage2.get("distilled_cot", "") or ""))
    distilled_cot = replace_user_intuition_phrases(distilled_cot)
    if not distilled_cot:
        return None, "missing_cot"

    answer_text = compact_text(str(stage2.get("answer", "") or row.get("ground_truth", "") or ""))
    if not answer_text:
        return None, "missing_answer"

    correct_insight = compact_text(str(stage1.get("correct_insight", "") or ""))
    selected_insight_text = normalize_selected_insight_text(stage2.get("selected_insight_text"))

    # Requirement #3: latent token count = token_count(stage1.correct_insight) / 2.
    insight_for_latent = correct_insight if correct_insight else selected_insight_text
    insight_token_len = token_counter.count(insight_for_latent)
    n_latent_steps = max(latent_min, insight_token_len // 2)
    if latent_max > 0:
        n_latent_steps = min(n_latent_steps, latent_max)
    if n_latent_steps > MAX_ALLOWED_LATENT_STEPS:
        return None, "filtered_latent_steps_gt_256"

    is_correct = bool(validation.get("is_correct", False))
    distilled_cot_tokens = safe_int(token_stats.get("distilled_cot_tokens"), default=0)
    if distilled_cot_tokens >= max_distilled_cot_tokens:
        return None, "filtered_distilled_cot_tokens_ge_threshold"
    difficulty, difficulty_rank = difficulty_bucket(is_correct=is_correct, distilled_cot_tokens=distilled_cot_tokens)

    # Requirement #7: use <|endoftext|> style token as latent padding placeholder.
    latent_placeholder = latent_pad_token * n_latent_steps

    assistant_content = (
        f"{LATENT_THINK_START}{latent_placeholder}{LATENT_THINK_END}"
        f"{THINK_START}{distilled_cot}{THINK_END}{answer_text}"
    )

    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant_content},
    ]

    # Requirement #6: provide reference prompt containing selected insight,
    # then assistant starts CoT. This supports state alignment loss construction.
    state_align_user_prompt = question
    if selected_insight_text:
        state_align_user_prompt = (
            "Solve the following problem by continuing from the your intuition.\n\nYour intuition may be correct or incorrect. Do not ignore it. Continue from it and finish the solution.\n"
            f"{question}\n\n"
            f"Your Intuition: {selected_insight_text}"
        )

    state_align_reference_messages = [
        {"role": "user", "content": state_align_user_prompt},
        {"role": "assistant", "content": f"{THINK_START}{distilled_cot}{THINK_END}{answer_text}"},
    ]

    uid = str(row.get("uid", "") or "")
    if not uid:
        uid_seed = f"{question}||{answer_text}||{distilled_cot[:200]}"
        uid = hashlib.md5(uid_seed.encode("utf-8")).hexdigest()

    record = {
        "record_id": uid,
        "source_uid": uid,
        "question": question,
        "ground_truth": answer_text,
        "messages": messages,
        "assistant_cot": distilled_cot,
        "assistant_answer": answer_text,
        "difficulty": difficulty,
        "difficulty_rank": difficulty_rank,
        "n_latent_steps": n_latent_steps,
        "insight_token_len": insight_token_len,
        "correct_insight": correct_insight,
        "selected_insight_text": selected_insight_text,
        "compression_ratio_vs_primary_output": ratio,
        "distilled_cot_tokens": distilled_cot_tokens,
        "stage2_is_correct": is_correct,
        "latent_pad_token": latent_pad_token,
        # Requirement #5 + #8 as explicit training metadata.
        "latent_loss_weight": 0.0,
        "cot_loss_weight": cot_loss_weight,
        "answer_loss_weight": answer_loss_weight,
        "mask_prompt_loss": True,
        "mask_system_loss": True,
        "latent_backprop_strategy": "markov_state_only",
        "state_align_enabled": True,
        "state_align_loss_weight": state_align_loss_weight,
        "state_align_reference_messages": state_align_reference_messages,
        "state_align_target": "assistant_cot_start_state",
        "curriculum_sort_key": [difficulty_rank, n_latent_steps, distilled_cot_tokens],
        "dataset_source": row.get("dataset_source", ""),
        "original_dataset": row.get("original_dataset", ""),
    }
    return record, "ok"


def build_goose_rl_record(
    row: Dict[str, Any],
    source_file: str,
    line_no: int,
    latent_min_steps: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Build one RL prompt sample from GooseReason JSONL.

    Requirement #11:
    Include latent_think/think/final-answer supervision intent in metadata.
    """
    question = compact_text(str(row.get("question", "") or ""))
    options = row.get("options", [])
    answer = compact_text(str(row.get("answer", "") or ""))
    if not question:
        return None, "missing_question"
    if not answer:
        return None, "missing_answer"

    src_name = Path(source_file).name
    domain = src_name.split("-")[0].strip().lower() if "-" in src_name else "general"
    options_text = format_options_for_prompt(options)

    user_prompt = f"{question}{options_text}\n\n"
    user_prompt += (
        "Please solve in the following structure:\n"
        "1) Output latent reasoning between <latent_think> and </latent_think>.\n"
        "2) Then output explicit reasoning between <think> and </think>.\n"
        "3) Then output a concise final answer."
    )

    record_id = f"{src_name}:{line_no}"
    prompt_messages = [{"role": "user", "content": user_prompt}]

    record = {
        "record_id": record_id,
        "data_source": f"goosereason_{domain}",
        "ability": domain,
        "prompt": prompt_messages,
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "source_file": src_name,
            "source_line": line_no,
            "question": question,
            "options": options,
            "answer": answer,
            "n_latent_steps": latent_min_steps,
            "loss_targets": ["latent_think", "think", "final_answer"],
        },
    }
    return record, "ok"


def load_or_init_state(
    state_path: Path,
    config_hash_value: str,
    reset_progress: bool,
    ignore_config_change: bool,
    mode_name: str,
) -> Dict[str, Any]:
    if reset_progress and state_path.exists():
        state_path.unlink()

    if state_path.exists():
        state = load_json(state_path)
        if state.get("version") != STATE_VERSION:
            raise RuntimeError(
                f"State version mismatch: found={state.get('version')} expected={STATE_VERSION}. "
                "Use --reset_progress to rebuild."
            )
        if state.get("mode") != mode_name:
            raise RuntimeError(
                f"State mode mismatch: found={state.get('mode')} expected={mode_name}. "
                "Use --reset_progress."
            )
        old_hash = str(state.get("config_hash", ""))
        if old_hash != config_hash_value and not ignore_config_change:
            raise RuntimeError(
                "Preprocess config changed since previous run. "
                "Use --ignore_config_change or --reset_progress."
            )
        state["config_hash"] = config_hash_value
        return state

    return {
        "version": STATE_VERSION,
        "mode": mode_name,
        "config_hash": config_hash_value,
        "created_at": utcnow_iso(),
        "updated_at": utcnow_iso(),
    }


def _increment_reason(counter: Dict[str, Any], reason: str) -> None:
    counter[reason] = safe_int(counter.get(reason), 0) + 1


def _read_distilled_rows(
    input_paths: List[Path],
    badline_path: Path,
    reset_progress: bool,
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Any]]:
    allow_unknown_single = len(input_paths) == 1
    input_sources = {
        str(path): infer_distilled_input_source_type(path, allow_unknown_single=allow_unknown_single)
        for path in input_paths
    }
    if reset_progress and badline_path.exists():
        badline_path.unlink()

    rows_by_uid: Dict[str, Dict[str, Dict[str, Any]]] = {}
    stats: Dict[str, Any] = {
        "input_paths": [str(path) for path in input_paths],
        "input_sources": input_sources,
        "processed_lines": 0,
        "invalid_lines": 0,
        "blank_lines": 0,
        "rows_with_uid": 0,
        "rows_without_uid": 0,
        "single_source_candidates": 0,
        "duplicate_uid_count": 0,
        "selection_reasons": {},
    }

    for input_path in input_paths:
        source_type = input_sources[str(input_path)]
        with input_path.open("r", encoding="utf-8") as fin:
            for line_no, line in enumerate(fin, 1):
                stats["processed_lines"] += 1
                stripped = line.strip()
                if not stripped:
                    stats["blank_lines"] += 1
                    continue

                try:
                    row = json.loads(stripped)
                except Exception:
                    stats["invalid_lines"] += 1
                    append_jsonl(
                        badline_path,
                        {
                            "file": str(input_path),
                            "line_no": line_no,
                            "line_preview": stripped[:1000],
                            "error": "json_decode_error",
                        },
                    )
                    continue

                uid = resolve_row_uid(row=row, source_path=input_path, line_no=line_no)
                if row.get("uid"):
                    stats["rows_with_uid"] += 1
                else:
                    stats["rows_without_uid"] += 1

                candidates = rows_by_uid.setdefault(uid, {})
                if candidates:
                    stats["duplicate_uid_count"] += int(not candidates.get("__counted_duplicate__", False))
                candidates[source_type] = row
                candidates["__counted_duplicate__"] = True

    for uid, candidates in list(rows_by_uid.items()):
        candidates.pop("__counted_duplicate__", None)
        if len(candidates) == 1:
            stats["single_source_candidates"] += 1
        rows_by_uid[uid] = candidates

    return rows_by_uid, stats


def select_preferred_distilled_rows(
    rows_by_uid: Dict[str, Dict[str, Dict[str, Any]]],
    compression_ratio_threshold: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selected_rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "selected_rows": 0,
        "selected_from_ds": 0,
        "selected_from_original": 0,
        "single_source_kept": 0,
        "single_source_filtered_ratio": 0,
        "duplicate_uid_count": 0,
        "fallback_to_original": 0,
        "dropped_both_ratio_ge_threshold": 0,
        "dropped_original_not_validated": 0,
        "dropped_other_duplicate_case": 0,
        "selection_reasons": {},
    }

    for uid, candidates in rows_by_uid.items():
        ds_row = candidates.get(DISTILLED_SOURCE_DS)
        orig_row = candidates.get(DISTILLED_SOURCE_ORIGINAL)

        if len(candidates) == 1:
            only_source, only_row = next(iter(candidates.items()))
            ratio = extract_row_ratio(only_row)
            if ratio < compression_ratio_threshold:
                selected_rows.append(only_row)
                stats["selected_rows"] += 1
                stats["single_source_kept"] += 1
                if only_source == DISTILLED_SOURCE_DS:
                    stats["selected_from_ds"] += 1
                    _increment_reason(stats["selection_reasons"], "single_source_ds_kept")
                else:
                    stats["selected_from_original"] += 1
                    _increment_reason(stats["selection_reasons"], "single_source_original_kept")
            else:
                stats["single_source_filtered_ratio"] += 1
                _increment_reason(stats["selection_reasons"], "single_source_filtered_ratio")
            continue

        if ds_row is None and orig_row is None:
            continue

        stats["duplicate_uid_count"] += 1
        ds_ratio = extract_row_ratio(ds_row)
        orig_ratio = extract_row_ratio(orig_row)
        orig_is_correct = extract_stage2_is_correct(orig_row)

        if ds_ratio < compression_ratio_threshold:
            selected_rows.append(ds_row)
            stats["selected_rows"] += 1
            stats["selected_from_ds"] += 1
            _increment_reason(stats["selection_reasons"], "duplicate_prefer_ds")
            continue

        if (
            ds_ratio >= compression_ratio_threshold
            and orig_ratio < compression_ratio_threshold
            and orig_is_correct
        ):
            selected_rows.append(orig_row)
            stats["selected_rows"] += 1
            stats["selected_from_original"] += 1
            stats["fallback_to_original"] += 1
            _increment_reason(stats["selection_reasons"], "duplicate_fallback_to_original")
            continue

        if ds_ratio >= compression_ratio_threshold and orig_ratio >= compression_ratio_threshold:
            stats["dropped_both_ratio_ge_threshold"] += 1
            _increment_reason(stats["selection_reasons"], "duplicate_drop_both_ratio_ge_threshold")
            continue

        if ds_ratio >= compression_ratio_threshold and orig_ratio < compression_ratio_threshold and not orig_is_correct:
            stats["dropped_original_not_validated"] += 1
            _increment_reason(stats["selection_reasons"], "duplicate_drop_original_not_validated")
            continue

        stats["dropped_other_duplicate_case"] += 1
        _increment_reason(stats["selection_reasons"], "duplicate_drop_other_case")

    return selected_rows, stats


def process_distilled_sft_dataset(
    input_paths: List[Path],
    cache_path: Path,
    badline_path: Path,
    state_path: Path,
    token_counter: TokenCounter,
    compression_ratio_threshold: float,
    max_distilled_cot_tokens: int,
    latent_pad_token: str,
    latent_min: int,
    latent_max: int,
    cot_loss_weight: float,
    answer_loss_weight: float,
    state_align_loss_weight: float,
    commit_every: int,
    reset_progress: bool,
    ignore_config_change: bool,
) -> Dict[str, Any]:
    ensure_dir(cache_path.parent)
    ensure_dir(state_path.parent)

    config_obj = {
        "input_paths": [str(path.resolve()) for path in input_paths],
        "compression_ratio_threshold": compression_ratio_threshold,
        "max_distilled_cot_tokens": max_distilled_cot_tokens,
        "blocked_data_type_keywords": list(BLOCKED_DATA_TYPE_KEYWORDS),
        "latent_pad_token": latent_pad_token,
        "latent_min": latent_min,
        "latent_max": latent_max,
        "max_allowed_latent_steps": MAX_ALLOWED_LATENT_STEPS,
        "cot_loss_weight": cot_loss_weight,
        "answer_loss_weight": answer_loss_weight,
        "state_align_loss_weight": state_align_loss_weight,
        "tokenizer": token_counter.tokenizer_name_or_path,
        "selection_rules": {
            "keep_ratio_strictly_less_than_threshold": True,
            "prefer_ds_when_ratio_lt_threshold": True,
            "fallback_to_original_only_when_ds_ge_threshold_and_original_lt_threshold_and_original_is_correct": True,
        },
    }
    state = load_or_init_state(
        state_path=state_path,
        config_hash_value=hash_config(config_obj),
        reset_progress=reset_progress,
        ignore_config_change=ignore_config_change,
        mode_name="sft_dataset",
    )

    if reset_progress:
        for p in (cache_path, badline_path):
            if p.exists():
                p.unlink()

    state.setdefault("input_paths", [str(path.resolve()) for path in input_paths])
    state.setdefault("processed_lines", 0)
    state.setdefault("kept_samples", 0)
    state.setdefault("invalid_lines", 0)
    state.setdefault("blank_lines", 0)
    state.setdefault("filtered_lines", 0)
    state.setdefault("filter_reasons", {})
    state.setdefault("selection_stats", {})

    expected_input_paths = [str(path.resolve()) for path in input_paths]
    if expected_input_paths != state.get("input_paths"):
        raise RuntimeError("input_paths differ from saved state. Use --reset_progress.")

    rows_by_uid, scan_stats = _read_distilled_rows(
        input_paths=input_paths,
        badline_path=badline_path,
        reset_progress=reset_progress,
    )
    selected_rows, selection_stats = select_preferred_distilled_rows(
        rows_by_uid=rows_by_uid,
        compression_ratio_threshold=compression_ratio_threshold,
    )

    commit_count = 0
    for row in selected_rows:
        commit_count += 1
        record, reason = build_sft_record(
            row=row,
            token_counter=token_counter,
            compression_ratio_threshold=compression_ratio_threshold,
            max_distilled_cot_tokens=max_distilled_cot_tokens,
            latent_pad_token=latent_pad_token,
            latent_min=latent_min,
            latent_max=latent_max,
            cot_loss_weight=cot_loss_weight,
            answer_loss_weight=answer_loss_weight,
            state_align_loss_weight=state_align_loss_weight,
        )
        if record is None:
            state["filtered_lines"] += 1
            _increment_reason(state["filter_reasons"], reason)
        else:
            append_jsonl(cache_path, record)
            state["kept_samples"] += 1

        if commit_count >= commit_every:
            state["updated_at"] = utcnow_iso()
            atomic_write_json(state_path, state)
            commit_count = 0

    state["processed_lines"] = safe_int(scan_stats.get("processed_lines"), 0)
    state["invalid_lines"] = safe_int(scan_stats.get("invalid_lines"), 0)
    state["blank_lines"] = safe_int(scan_stats.get("blank_lines"), 0)
    state["selection_stats"] = {
        **scan_stats,
        **selection_stats,
    }
    state["selected_rows"] = safe_int(selection_stats.get("selected_rows"), 0)
    state["filtered_lines"] += (
        safe_int(selection_stats.get("single_source_filtered_ratio"), 0)
        + safe_int(selection_stats.get("dropped_both_ratio_ge_threshold"), 0)
        + safe_int(selection_stats.get("dropped_original_not_validated"), 0)
        + safe_int(selection_stats.get("dropped_other_duplicate_case"), 0)
    )

    state["updated_at"] = utcnow_iso()
    atomic_write_json(state_path, state)
    return state


def collect_goose_jsonl_files(goose_dir: Path) -> List[Path]:
    data_dir = goose_dir / "data"
    if not data_dir.exists():
        return []
    return sorted(p for p in data_dir.glob("*-train.jsonl") if p.is_file())


def process_goose_rl_stream(
    goose_dir: Path,
    cache_path: Path,
    badline_path: Path,
    state_path: Path,
    latent_min_steps: int,
    commit_every: int,
    reset_progress: bool,
    ignore_config_change: bool,
) -> Dict[str, Any]:
    ensure_dir(cache_path.parent)
    ensure_dir(state_path.parent)

    files = collect_goose_jsonl_files(goose_dir)
    config_obj = {
        "goose_dir": str(goose_dir.resolve()),
        "files": [str(p.resolve()) for p in files],
        "latent_min_steps": latent_min_steps,
    }

    state = load_or_init_state(
        state_path=state_path,
        config_hash_value=hash_config(config_obj),
        reset_progress=reset_progress,
        ignore_config_change=ignore_config_change,
        mode_name="rl_stream",
    )

    if reset_progress:
        for p in (cache_path, badline_path):
            if p.exists():
                p.unlink()

    state.setdefault("goose_dir", str(goose_dir.resolve()))
    state.setdefault("files", {})
    state.setdefault("processed_lines", 0)
    state.setdefault("kept_samples", 0)
    state.setdefault("invalid_lines", 0)
    state.setdefault("filtered_lines", 0)
    state.setdefault("filter_reasons", {})

    if str(goose_dir.resolve()) != state.get("goose_dir"):
        raise RuntimeError("goose_dir differs from saved state. Use --reset_progress.")

    for file_path in files:
        file_key = str(file_path.resolve())
        file_state = state["files"].get(file_key)
        if file_state is None:
            file_state = {
                "offset": 0,
                "line_no": 0,
                "processed_lines": 0,
                "kept_samples": 0,
            }
            state["files"][file_key] = file_state

        size_now = file_path.stat().st_size
        if safe_int(file_state.get("offset"), 0) > size_now:
            raise RuntimeError(
                f"Saved RL offset exceeds file size for {file_key}. "
                "Input may be truncated/replaced. Use --reset_progress."
            )

        commit_count = 0
        with file_path.open("r", encoding="utf-8") as fin:
            fin.seek(safe_int(file_state.get("offset"), 0))

            while True:
                line_start = fin.tell()
                line = fin.readline()
                if not line:
                    break

                file_state["line_no"] = safe_int(file_state.get("line_no"), 0) + 1
                file_state["processed_lines"] = safe_int(file_state.get("processed_lines"), 0) + 1
                state["processed_lines"] = safe_int(state.get("processed_lines"), 0) + 1
                commit_count += 1

                stripped = line.strip()
                if not stripped:
                    file_state["offset"] = fin.tell()
                    continue

                try:
                    row = json.loads(stripped)
                except Exception:
                    state["invalid_lines"] = safe_int(state.get("invalid_lines"), 0) + 1
                    append_jsonl(
                        badline_path,
                        {
                            "file": file_key,
                            "offset": line_start,
                            "line_no": file_state["line_no"],
                            "line_preview": stripped[:1000],
                            "error": "json_decode_error",
                        },
                    )
                    file_state["offset"] = fin.tell()
                    if commit_count >= commit_every:
                        state["updated_at"] = utcnow_iso()
                        atomic_write_json(state_path, state)
                        commit_count = 0
                    continue

                record, reason = build_goose_rl_record(
                    row=row,
                    source_file=file_key,
                    line_no=safe_int(file_state.get("line_no"), 0),
                    latent_min_steps=latent_min_steps,
                )

                if record is None:
                    state["filtered_lines"] = safe_int(state.get("filtered_lines"), 0) + 1
                    state["filter_reasons"][reason] = safe_int(state["filter_reasons"].get(reason), 0) + 1
                else:
                    append_jsonl(cache_path, record)
                    state["kept_samples"] = safe_int(state.get("kept_samples"), 0) + 1
                    file_state["kept_samples"] = safe_int(file_state.get("kept_samples"), 0) + 1

                file_state["offset"] = fin.tell()

                if commit_count >= commit_every:
                    state["updated_at"] = utcnow_iso()
                    atomic_write_json(state_path, state)
                    commit_count = 0

    state["updated_at"] = utcnow_iso()
    atomic_write_json(state_path, state)
    return state


def dedupe_by_record_id(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        rid = str(rec.get("record_id", "") or "")
        if not rid:
            rid = hashlib.md5(json.dumps(rec, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        mapping[rid] = rec
    return list(mapping.values())


def finalize_sft_outputs(cache_path: Path, output_dir: Path) -> Dict[str, Any]:
    rows = read_jsonl(cache_path)
    rows = dedupe_by_record_id(rows)

    rows.sort(
        key=lambda x: (
            safe_int(x.get("difficulty_rank"), 9),
            safe_int(x.get("n_latent_steps"), 10**9),
            safe_int(x.get("distilled_cot_tokens"), 10**9),
        )
    )

    for idx, rec in enumerate(rows):
        rec["curriculum_index"] = idx

    sft_jsonl = output_dir / "sft_train.jsonl"
    sft_parquet = output_dir / "sft_train.parquet"

    n = write_jsonl(sft_jsonl, rows)
    ok, engine = maybe_build_parquet(rows, sft_parquet)

    difficulties = {"easy": 0, "medium": 0, "hard": 0}
    for rec in rows:
        d = str(rec.get("difficulty", ""))
        if d in difficulties:
            difficulties[d] += 1

    summary = {
        "samples": n,
        "jsonl": str(sft_jsonl),
        "parquet": str(sft_parquet) if ok else None,
        "parquet_status": engine,
        "difficulty_distribution": difficulties,
        "n_latent_min": min((safe_int(r.get("n_latent_steps"), 0) for r in rows), default=0),
        "n_latent_max": max((safe_int(r.get("n_latent_steps"), 0) for r in rows), default=0),
    }
    return summary


def finalize_rl_outputs(cache_path: Path, output_dir: Path) -> Dict[str, Any]:
    rows = read_jsonl(cache_path)
    rows = dedupe_by_record_id(rows)

    rl_jsonl = output_dir / "rl_train.jsonl"
    rl_parquet = output_dir / "rl_train.parquet"

    n = write_jsonl(rl_jsonl, rows)
    ok, engine = maybe_build_parquet(rows, rl_parquet)

    summary = {
        "samples": n,
        "jsonl": str(rl_jsonl),
        "parquet": str(rl_parquet) if ok else None,
        "parquet_status": engine,
    }
    return summary


def preprocess_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    progress_dir = output_dir / "_progress"
    cache_dir = output_dir / "_cache"
    ensure_dir(progress_dir)
    ensure_dir(cache_dir)

    token_counter = TokenCounter(args.tokenizer)
    if token_counter.tokenizer is not None:
        validate_latent_think_tokenizer_contract(token_counter.tokenizer)

    report: Dict[str, Any] = {
        "started_at": utcnow_iso(),
        "tokenizer": args.tokenizer,
        "token_counter_mode": token_counter.mode,
        "mode": args.mode,
    }

    if args.mode in ("sft", "both"):
        distilled_input_paths = validate_distilled_input_paths(parse_distilled_input_paths(args.input))
        report["sft_input_paths"] = [str(path) for path in distilled_input_paths]
        sft_state = process_distilled_sft_dataset(
            input_paths=distilled_input_paths,
            cache_path=cache_dir / "sft_samples.jsonl",
            badline_path=progress_dir / "sft_badlines.jsonl",
            state_path=progress_dir / "sft_state.json",
            token_counter=token_counter,
            compression_ratio_threshold=args.compression_ratio_threshold,
            max_distilled_cot_tokens=args.max_distilled_cot_tokens,
            latent_pad_token=args.latent_pad_token,
            latent_min=args.latent_min,
            latent_max=args.latent_max,
            cot_loss_weight=args.cot_loss_weight,
            answer_loss_weight=args.answer_loss_weight,
            state_align_loss_weight=args.state_align_loss_weight,
            commit_every=args.commit_every,
            reset_progress=args.reset_progress,
            ignore_config_change=args.ignore_config_change,
        )
        sft_summary = finalize_sft_outputs(cache_path=cache_dir / "sft_samples.jsonl", output_dir=output_dir)
        report["sft_state"] = sft_state
        report["sft_summary"] = sft_summary

    if args.mode in ("rl", "both"):
        rl_state = process_goose_rl_stream(
            goose_dir=Path(args.goose_dir).resolve(),
            cache_path=cache_dir / "rl_samples.jsonl",
            badline_path=progress_dir / "rl_badlines.jsonl",
            state_path=progress_dir / "rl_state.json",
            latent_min_steps=args.latent_min,
            commit_every=args.commit_every,
            reset_progress=args.reset_progress,
            ignore_config_change=args.ignore_config_change,
        )
        rl_summary = finalize_rl_outputs(cache_path=cache_dir / "rl_samples.jsonl", output_dir=output_dir)
        report["rl_state"] = rl_state
        report["rl_summary"] = rl_summary

    report["finished_at"] = utcnow_iso()
    report_path = output_dir / "preprocess_report.json"
    atomic_write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refactored preprocess for latent reasoning SFT/RL data")

    parser.add_argument(
        "--input",
        default=DEFAULT_DISTILLED_INPUT,
        help="One distilled JSONL path or two comma-separated distilled JSONL paths",
    )
    parser.add_argument("--goose_dir", default=DEFAULT_GOOSE_DIR, help="GooseReason-0.7M root directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for processed data")

    parser.add_argument("--mode", default="both", choices=["sft", "rl", "both"])
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-14B", help="Tokenizer name/path for latent token count")

    # Requirement #2
    parser.add_argument(
        "--compression_ratio_threshold",
        type=float,
        default=1.5,
        help="Keep only samples with token_stats.compression_ratio_vs_primary_output < threshold",
    )
    parser.add_argument(
        "--max_distilled_cot_tokens",
        type=int,
        default=32768,
        help="Keep only samples with token_stats.distilled_cot_tokens < threshold",
    )

    # Requirement #3/#7
    parser.add_argument("--latent_pad_token", default="<|endoftext|>", help="Pad token inside <latent_think>")
    parser.add_argument("--latent_min", type=int, default=1, help="Minimum latent token count")
    parser.add_argument("--latent_max", type=int, default=128, help="Maximum latent token count")

    # Requirement #8
    parser.add_argument("--cot_loss_weight", type=float, default=0.25, help="Loss weight for <think> CoT part")
    parser.add_argument("--answer_loss_weight", type=float, default=1.0, help="Loss weight for final answer part")

    # Requirement #5/#6
    parser.add_argument(
        "--state_align_loss_weight",
        type=float,
        default=1.0,
        help="Auxiliary state alignment loss weight metadata",
    )

    # Requirement #1 (resumable progress)
    parser.add_argument("--commit_every", type=int, default=200, help="Persist progress every N lines")
    parser.add_argument(
        "--reset_progress",
        action="store_true",
        help="Reset existing progress/cache and rebuild from scratch",
    )
    parser.add_argument(
        "--ignore_config_change",
        action="store_true",
        help="Allow resume even if preprocess config changed",
    )

    return parser


def print_report(report: Dict[str, Any]) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    result = preprocess_dataset(args)
    print_report(result)
