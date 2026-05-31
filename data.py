import ast
import json
import os
import re
from typing import Any, Dict, Iterable, Optional

from datasets import load_dataset

from utils import extract_gold, normalize_answer


PUBLIC_DATASETS: Dict[str, tuple[str | None, str | None]] = {
    "gsm8k": ("openai/gsm8k", "main"),
    "aime2024": ("HuggingFaceH4/aime_2024", None),
    "aime2025": ("math-ai/AIME_2025", None),
    "math500": ("HuggingFaceH4/MATH-500", None),
    "gpqa_diamond": ("Idavidrein/gpqa", "gpqa_diamond"),
    "arc": ("ai2_arc", None),
    "commonsense_qa": ("tau/commonsense_qa", None),
    "mbppplus": ("evalplus/mbppplus", None),
    "humanevalplus": ("evalplus/humanevalplus", None),
    "prosqa": (None, None),
    "dolci_think_sft_32b_sampled": (None, None),
}


def _dataset_env_key(task: str) -> str:
    return "DATASET_" + re.sub(r"[^A-Za-z0-9]+", "_", task).upper()


def _resolve_dataset(task: str, *, required: bool = False) -> tuple[str, str | None]:
    default_name, default_config = PUBLIC_DATASETS[task]
    env_key = _dataset_env_key(task)
    raw_name = os.getenv(env_key, default_name or "").strip()
    config = os.getenv(f"{env_key}_CONFIG", default_config or "").strip() or None
    if "::" in raw_name:
        raw_name, inline_config = raw_name.split("::", 1)
        config = inline_config or config
    if required and not raw_name:
        raise ValueError(
            f"Dataset for '{task}' is not bundled. Set {env_key} to a local path or Hugging Face dataset id."
        )
    if not raw_name:
        raise ValueError(f"No dataset configured for '{task}'. Set {env_key}.")
    return raw_name, config


def _load_dataset(task: str, *, split: str, cache_dir: Optional[str] = None, required: bool = False, config: str | None = None):
    dataset_name, dataset_config = _resolve_dataset(task, required=required)
    dataset_config = config if config is not None else dataset_config
    if dataset_config:
        return load_dataset(dataset_name, dataset_config, split=split, cache_dir=cache_dir)
    return load_dataset(dataset_name, split=split, cache_dir=cache_dir)


def _first_present(item: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return default


def extract_content_outside_think(text):
    """Split text inside <think> blocks from the visible response."""
    inside_pattern = r"<think>(.*?)</think>"
    inside_content = re.findall(inside_pattern, text, flags=re.DOTALL)
    outside_pattern = r"<think>.*?</think>"
    outside_content_raw = re.split(outside_pattern, text, flags=re.DOTALL)
    cleaned_parts = (part.strip() for part in outside_content_raw if part and part.strip())
    outside_content_string = " ".join(cleaned_parts)
    return inside_content, outside_content_string


def _normalize_distilled_ground_truth(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return " || ".join(cleaned)

    text = str(value).strip()
    if not text:
        return ""

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
    except TypeError:
        return text

    if isinstance(parsed, list):
        cleaned = [str(item).strip() for item in parsed if str(item).strip()]
        return " || ".join(cleaned)
    return str(parsed).strip()


def load_distilled_latent_reasoning(
    data_path: str,
    with_insight: bool = False,
) -> Iterable[Dict]:
    with open(data_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            question = str(item.get("question", "")).strip()

            if not question:
                continue

            stage1 = item.get("stage1", {}) or {}
            stage2 = item.get("stage2", {}) or {}
            gold = _normalize_distilled_ground_truth(item.get("ground_truth", ""))
            solution = str(stage2.get("answer", "")).strip() or gold

            correct_insight = stage1.get("correct_insight", "")

            if with_insight and correct_insight != "":
                question += f"""
Here are some insights to solve the question:

{correct_insight}

Please follow it to solve the question.
"""

            yield {
                "uid": item.get("uid", ""),
                "question": question,
                "solution": solution,
                "gold": gold,
                "answer_type": "distilled_reasoning",
                "task_summary": stage1.get("task_summary", ""),
                "correct_insight": stage1.get("correct_insight", ""),
                "incorrect_insights": stage1.get("incorrect_insights", []),
                "distilled_cot": stage2.get("distilled_cot", ""),
                "selected_insight_type": stage2.get("selected_insight_type", ""),
                "selected_insight_index": stage2.get("selected_insight_index", None),
                "dataset_source": item.get("dataset_source", ""),
                "original_dataset": item.get("original_dataset", ""),
                "metadata": item.get("metadata", {}),
            }


def load_gsm8k(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = _load_dataset("gsm8k", split=split, cache_dir=cache_dir)
    for item in ds:
        question = item["question"].strip()
        solution = item["answer"]
        gold = normalize_answer(extract_gold(solution))
        yield {"question": question, "solution": solution, "gold": gold}


def load_aime2025(split: str = "train", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = _load_dataset("aime2025", split=split, cache_dir=cache_dir)
    for item in ds:
        problem = str(_first_present(item, "problem", "question", "Question")).strip()
        answer = str(_first_present(item, "answer", "Answer", "final_answer")).strip()
        yield {"question": problem, "solution": answer, "gold": normalize_answer(answer)}


def load_math500(split: str = "train", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = _load_dataset("math500", split=split, cache_dir=cache_dir)
    for item in ds:
        problem = str(_first_present(item, "problem", "question")).strip()
        answer = str(_first_present(item, "answer", "solution")).strip()
        yield {"question": problem, "solution": answer, "gold": normalize_answer(answer)}


def load_aime2024(split: str = "train", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = _load_dataset("aime2024", split=split, cache_dir=cache_dir)
    for item in ds:
        problem = str(_first_present(item, "problem", "question", "Question")).strip()
        answer = str(_first_present(item, "answer", "Answer", "final_answer")).strip()
        yield {"question": problem, "solution": answer, "gold": normalize_answer(answer)}


def load_gpqa_diamond(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = _load_dataset("gpqa_diamond", split=split, cache_dir=cache_dir)
    for item in ds:
        question = str(_first_present(item, "question", "Question")).strip()
        answer = str(_first_present(item, "answer", "Correct Answer", "correct_answer")).strip()
        incorrect = [
            str(_first_present(item, f"Incorrect Answer {idx}", f"incorrect_answer_{idx}", default="")).strip()
            for idx in range(1, 4)
        ]
        choices = [answer] + [choice for choice in incorrect if choice]
        if choices and not any(label in question.lower() for label in ["a:", "a)"]):
            labels = ["A", "B", "C", "D"]
            question = question + "\n" + "\n".join(f"{label}: {choice}" for label, choice in zip(labels, choices))
            answer = "A"
        yield {"question": question, "solution": answer, "gold": normalize_answer(answer)}


def _load_arc(split: str, config: str, cache_dir: Optional[str]) -> Iterable[Dict]:
    ds = _load_dataset("arc", split=split, cache_dir=cache_dir, config=config)
    for item in ds:
        stem = item["question"].strip()
        choices = item["choices"]
        labels = choices["label"]
        texts = choices["text"]
        label_map = {"1": "a", "2": "b", "3": "c", "4": "d"}

        def map_label(label_value: str) -> str:
            s = str(label_value).strip()
            if s in label_map:
                return label_map[s]
            return s.lower()

        formatted_choices = {}
        mapped_order = []
        for label, text in zip(labels, texts):
            mlabel = map_label(label)
            formatted_choices[mlabel] = text.strip()
            mapped_order.append(mlabel)

        ordered_lines = [f"{lab}: {formatted_choices[lab]}" for lab in mapped_order]
        question = stem + "\n" + "\n".join(ordered_lines)
        raw_answer = item.get("answerKey", "").strip()
        mapped_answer = map_label(raw_answer) if raw_answer else ""
        yield {"question": question, "solution": mapped_answer, "gold": normalize_answer(mapped_answer)}


def load_arc_easy(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    yield from _load_arc(split=split, config="ARC-Easy", cache_dir=cache_dir)


def load_arc_challenge(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    yield from _load_arc(split=split, config="ARC-Challenge", cache_dir=cache_dir)


def load_prosqa(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = _load_dataset("prosqa", split=split, cache_dir=cache_dir, required=True)
    for item in ds:
        question = str(_first_present(item, "question", "prompt")).strip()
        answer = str(_first_present(item, "answer", "gold", "label")).strip()
        yield {"question": question, "solution": answer, "gold": normalize_answer(answer)}


def load_commonsense_qa(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = _load_dataset("commonsense_qa", split=split, cache_dir=cache_dir)
    for item in ds:
        stem = item["question"].strip()
        choices = item["choices"]
        labels = choices["label"]
        texts = choices["text"]
        label_map = {"1": "a", "2": "b", "3": "c", "4": "d", "5": "e"}

        def map_label(label_value: str) -> str:
            s = str(label_value).strip()
            if s in label_map:
                return label_map[s]
            return s.lower()

        formatted_choices = {}
        mapped_order = []
        for label, text in zip(labels, texts):
            mlabel = map_label(label)
            formatted_choices[mlabel] = text.strip()
            mapped_order.append(mlabel)

        ordered_lines = [f"{lab}: {formatted_choices[lab]}" for lab in mapped_order]
        question = stem + "\n" + "\n".join(ordered_lines)
        raw_answer = item.get("answerKey", "").strip()
        mapped_answer = map_label(raw_answer) if raw_answer else ""
        yield {"question": question, "solution": mapped_answer, "gold": normalize_answer(mapped_answer)}


def load_winogrande(
    split: str = "validation",
    subset: str = "winogrande_debiased",
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:
    ds = load_dataset("allenai/winogrande", subset, split=split, cache_dir=cache_dir)
    for item in ds:
        ask_str = "Pickout proper choice that fits the _ in the following sentence:"
        sentence = item["sentence"].strip()
        option1 = str(item["option1"]).strip()
        option2 = str(item["option2"]).strip()
        question = f"{ask_str}\n{sentence}\n1: {option1}\n2: {option2}"
        answer = str(item["answer"])
        yield {"question": question, "solution": answer, "gold": normalize_answer(answer)}


def load_mbppplus(
    split: str = "test",
    subset: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:
    ds = _load_dataset("mbppplus", split=split, cache_dir=cache_dir, config=subset)
    for item in ds:
        tests = item.get("test_list") or []
        test_preview = "\n".join(str(test) for test in tests[:3])
        question = f"""Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```python\nYOUR_PYTHON_CODE\n```:
{item["prompt"]}
Your answer will be tested on test cases like:
{test_preview}
"""
        answer = str(item.get("test", ""))
        yield {"question": question, "solution": answer, "gold": answer}


def load_humanevalplus(
    split: str = "test",
    subset: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:
    ds = _load_dataset("humanevalplus", split=split, cache_dir=cache_dir, config=subset)
    for item in ds:
        question = f"""Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```python\nYOUR_PYTHON_CODE\n```:
{item["prompt"]}
"""
        raw_answer = str(item.get("test", ""))
        entry_point = item.get("entry_point", "candidate")
        answer = raw_answer.replace("candidate", entry_point)
        answer += f"\n\ncheck({entry_point})"
        yield {"question": question, "solution": answer, "gold": answer}


def load_Dolci_Think_SFT_32B_sampled(
    split: str = "train",
    subset: str | None = None,
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:
    ds = _load_dataset("dolci_think_sft_32b_sampled", split=split, cache_dir=cache_dir, required=True, config=subset)
    for item in ds:
        question = ""
        raw_answer = ""
        for message in item["messages"]:
            if message["role"] == "user":
                question = message["content"].strip()
            elif message["role"] == "assistant":
                raw_answer = message["content"].strip()

        _, answer = extract_content_outside_think(raw_answer)
        answer = answer.strip()
        yield {"question": question, "solution": answer, "gold": answer}


def load_medqa(split=None, subset=None, cache_dir=None):
    data_file = os.getenv("MEDQA_DATA_FILE", "./data/medqa.json")
    ds = load_dataset("json", data_files=data_file, split="train")
    for item in ds:
        question = item["query"]
        raw_answer = str(item["answer"])
        choice_map = {"0": "A", "1": "B", "2": "C", "3": "D"}

        answer = ""
        for idx, op in enumerate(item["options"]):
            if raw_answer in op:
                answer = choice_map[str(idx)].lower()
                break

        yield {"question": question, "solution": answer, "gold": normalize_answer(answer)}
