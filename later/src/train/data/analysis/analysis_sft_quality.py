import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from later.src.utils.utils import ensure_latent_think_special_tokens, safe_finite_float

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it first, e.g. `pip install matplotlib`."
    ) from exc

try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
except ModuleNotFoundError as exc:
    raise SystemExit(
        "transformers is required for token counting. Install it first, e.g. `pip install transformers`."
    ) from exc


DEFAULT_INPUT_PATH = (
    "data/processed_mix_ds_full/sft_train.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    "later/src/train/data/analysis/output"
)
DEFAULT_TOKENIZER_PATH = "Qwen/Qwen3-14B"


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        for key in ("content", "text", "output", "answer", "reasoning_content", "solution", "response"):
            candidate = _to_text(value.get(key))
            if candidate:
                return candidate
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    if isinstance(value, list):
        parts = [_to_text(item) for item in value]
        return "\n".join([part for part in parts if part]).strip()
    return str(value)


def _extract_source_outputs(outputs: Any) -> List[str]:
    if outputs is None:
        return []
    if not isinstance(outputs, list):
        outputs = [outputs]

    text_outputs: List[str] = []
    for item in outputs:
        text = _to_text(item)
        if text:
            text_outputs.append(text)
    return text_outputs


def _to_float(value: Any) -> Optional[float]:
    return safe_finite_float(value)


def _float_or_nan(value: Any) -> float:
    parsed = _to_float(value)
    return np.nan if parsed is None else float(parsed)


def _to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def detect_record_schema(record: Dict[str, Any]) -> str:
    processed_sft_markers = {
        "assistant_cot",
        "assistant_answer",
        "compression_ratio_vs_primary_output",
        "stage2_is_correct",
        "record_id",
    }
    if any(key in record for key in processed_sft_markers):
        return "processed_sft"
    return "distilled_pipeline"


def _json_load_line(line: str, use_json_repair: bool) -> Dict[str, Any]:
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        if not use_json_repair:
            raise
        from json_repair import repair_json

        parsed = repair_json(line, return_objects=True, ensure_ascii=False, skip_json_loads=True)

    if not isinstance(parsed, dict):
        raise ValueError(f"Each JSONL row must be a dict, but got {type(parsed).__name__}.")
    return parsed


class TokenCounter:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer
        self.cache: Dict[str, int] = {}

    def count(self, text: str) -> int:
        clean = (text or "").strip()
        if not clean:
            return 0
        cached = self.cache.get(clean)
        if cached is not None:
            return cached
        token_ids = self.tokenizer(clean, add_special_tokens=False)["input_ids"]
        count = int(len(token_ids))
        self.cache[clean] = count
        return count


def _base_metric_row(
    *,
    record_schema: str,
    line_no: int,
    uid: str,
    dataset_source: str,
    original_dataset: str,
    is_correct: Optional[bool],
    score: Optional[float],
    ground_truth_tokens: float,
    insight_tokens: float,
    source_cot_tokens: float,
    distilled_cot_tokens: float,
    cot_compression_ratio: float,
    difficulty: str = "",
    difficulty_rank: float = np.nan,
    n_latent_steps: float = np.nan,
    latent_backprop_strategy: str = "",
    state_align_enabled: str = "",
    state_align_target: str = "",
    mask_prompt_loss: str = "",
    mask_system_loss: str = "",
    latent_loss_weight: float = np.nan,
    cot_loss_weight: float = np.nan,
    answer_loss_weight: float = np.nan,
) -> Dict[str, Any]:
    return {
        "line_no": line_no,
        "uid": uid,
        "record_schema": record_schema,
        "dataset_source": dataset_source,
        "original_dataset": original_dataset,
        "is_correct": is_correct,
        "is_correct_int": np.nan if is_correct is None else int(is_correct),
        "score": score,
        "ground_truth_tokens": ground_truth_tokens,
        "insight_tokens": insight_tokens,
        "source_cot_tokens": source_cot_tokens,
        "distilled_cot_tokens": distilled_cot_tokens,
        "cot_compression_ratio": cot_compression_ratio,
        "difficulty": difficulty,
        "difficulty_rank": difficulty_rank,
        "n_latent_steps": n_latent_steps,
        "latent_backprop_strategy": latent_backprop_strategy,
        "state_align_enabled": state_align_enabled,
        "state_align_target": state_align_target,
        "mask_prompt_loss": mask_prompt_loss,
        "mask_system_loss": mask_system_loss,
        "latent_loss_weight": latent_loss_weight,
        "cot_loss_weight": cot_loss_weight,
        "answer_loss_weight": answer_loss_weight,
    }


def _extract_processed_sft_metrics(record: Dict[str, Any], line_no: int, token_counter: TokenCounter) -> Dict[str, Any]:
    ground_truth = _to_text(record.get("ground_truth"))
    insight = _to_text(record.get("correct_insight")) or _to_text(record.get("selected_insight_text"))
    distilled_cot = _to_text(record.get("assistant_cot"))

    distilled_cot_tokens = _to_float(record.get("distilled_cot_tokens"))
    if distilled_cot_tokens is None or distilled_cot_tokens <= 0:
        distilled_cot_tokens = float(token_counter.count(distilled_cot))

    cot_compression_ratio = _to_float(record.get("compression_ratio_vs_primary_output"))
    if cot_compression_ratio is None:
        cot_compression_ratio = np.nan

    if cot_compression_ratio and cot_compression_ratio > 0:
        source_cot_tokens = float(distilled_cot_tokens) / float(cot_compression_ratio)
    else:
        source_cot_tokens = np.nan

    return _base_metric_row(
        record_schema="processed_sft",
        line_no=line_no,
        uid=_to_text(record.get("record_id")) or _to_text(record.get("source_uid")) or _to_text(record.get("uid")),
        dataset_source=_to_text(record.get("dataset_source")),
        original_dataset=_to_text(record.get("original_dataset")),
        is_correct=_to_bool(record.get("stage2_is_correct")),
        score=_to_float(record.get("score")),
        ground_truth_tokens=float(token_counter.count(ground_truth)),
        insight_tokens=float(token_counter.count(insight)),
        source_cot_tokens=float(source_cot_tokens),
        distilled_cot_tokens=float(distilled_cot_tokens),
        cot_compression_ratio=float(cot_compression_ratio),
        difficulty=_to_text(record.get("difficulty")),
        difficulty_rank=_float_or_nan(record.get("difficulty_rank")),
        n_latent_steps=_float_or_nan(record.get("n_latent_steps")),
        latent_backprop_strategy=_to_text(record.get("latent_backprop_strategy")),
        state_align_enabled=_to_text(record.get("state_align_enabled")),
        state_align_target=_to_text(record.get("state_align_target")),
        mask_prompt_loss=_to_text(record.get("mask_prompt_loss")),
        mask_system_loss=_to_text(record.get("mask_system_loss")),
        latent_loss_weight=_float_or_nan(record.get("latent_loss_weight")),
        cot_loss_weight=_float_or_nan(record.get("cot_loss_weight")),
        answer_loss_weight=_float_or_nan(record.get("answer_loss_weight")),
    )


def _extract_distilled_pipeline_metrics(record: Dict[str, Any], line_no: int, token_counter: TokenCounter) -> Dict[str, Any]:
    stage1 = record.get("stage1") or {}
    stage2 = record.get("stage2") or {}
    validation = stage2.get("validation") or {}
    token_stats = record.get("token_stats") or {}

    ground_truth = _to_text(record.get("ground_truth"))
    insight = _to_text(stage1.get("correct_insight"))
    distilled_cot = _to_text(stage2.get("distilled_cot"))

    source_outputs = _extract_source_outputs(record.get("source_outputs"))
    source_primary = source_outputs[0] if source_outputs else ""

    source_cot_tokens = _to_float(token_stats.get("source_primary_output_tokens"))
    if source_cot_tokens is None or source_cot_tokens <= 0:
        source_cot_tokens = float(token_counter.count(source_primary))

    distilled_cot_tokens = float(token_counter.count(distilled_cot))
    ground_truth_tokens = float(token_counter.count(ground_truth))
    insight_tokens = float(token_counter.count(insight))

    if source_cot_tokens > 0:
        cot_compression_ratio = distilled_cot_tokens / source_cot_tokens
    else:
        cot_compression_ratio = np.nan

    is_correct = _to_bool(validation.get("is_correct"))

    return _base_metric_row(
        record_schema="distilled_pipeline",
        line_no=line_no,
        uid=_to_text(record.get("uid")),
        dataset_source=_to_text(record.get("dataset_source")),
        original_dataset=_to_text(record.get("original_dataset")),
        is_correct=is_correct,
        score=_to_float(validation.get("score")),
        ground_truth_tokens=ground_truth_tokens,
        insight_tokens=insight_tokens,
        source_cot_tokens=source_cot_tokens,
        distilled_cot_tokens=distilled_cot_tokens,
        cot_compression_ratio=cot_compression_ratio,
    )


def _extract_metrics(record: Dict[str, Any], line_no: int, token_counter: TokenCounter) -> Dict[str, Any]:
    record_schema = detect_record_schema(record)
    if record_schema == "processed_sft":
        return _extract_processed_sft_metrics(record, line_no, token_counter)
    return _extract_distilled_pipeline_metrics(record, line_no, token_counter)


def load_metrics_dataframe(
    input_path: Path,
    max_samples: Optional[int],
    use_json_repair: bool,
    token_counter: TokenCounter,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    bad_rows: List[Dict[str, Any]] = []
    total_lines = 0
    empty_lines = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            total_lines += 1
            stripped = line.strip()
            if not stripped:
                empty_lines += 1
                continue

            if max_samples is not None and len(rows) >= max_samples:
                break

            try:
                record = _json_load_line(stripped, use_json_repair=use_json_repair)
                rows.append(_extract_metrics(record, line_no, token_counter))
            except Exception as exc:  # noqa: BLE001
                if len(bad_rows) < 20:
                    bad_rows.append({"line_no": line_no, "error": f"{type(exc).__name__}: {exc}"})

    df = pd.DataFrame(rows)
    meta = {
        "input_path": str(input_path),
        "total_lines": total_lines,
        "empty_lines": empty_lines,
        "parsed_rows": len(rows),
        "bad_rows": len(bad_rows),
        "bad_row_examples": bad_rows,
    }
    return df, meta


def _stats(series: pd.Series) -> Dict[str, Any]:
    clean = series.dropna()
    if clean.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "min": None,
            "max": None,
        }

    return {
        "count": int(clean.shape[0]),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p25": float(clean.quantile(0.25)),
        "p75": float(clean.quantile(0.75)),
        "p95": float(clean.quantile(0.95)),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def categorical_distribution(df: pd.DataFrame, column: str, top_k: int = 50) -> List[Dict[str, Any]]:
    if column not in df.columns or df.empty:
        return []

    series = df[column].fillna("").astype(str)
    series = series.where(series.str.len() > 0, "<missing>")
    counts = series.value_counts(dropna=False).head(top_k)
    total = int(len(df))
    return [
        {
            "value": str(value),
            "count": int(count),
            "percentage": float(count / total) if total > 0 else None,
        }
        for value, count in counts.items()
    ]


def grouped_quality_summary(df: pd.DataFrame, group_col: str, top_k: int = 50) -> List[Dict[str, Any]]:
    if group_col not in df.columns or df.empty:
        return []

    work = df.copy()
    work[group_col] = work[group_col].fillna("").astype(str)
    work[group_col] = work[group_col].where(work[group_col].str.len() > 0, "<missing>")
    total = int(len(work))

    rows: List[Dict[str, Any]] = []
    for group_value, group_df in work.groupby(group_col, dropna=False):
        acc = group_df["is_correct_int"].dropna()
        ratio = group_df["cot_compression_ratio"].dropna()
        rows.append(
            {
                "group": str(group_value),
                "count": int(len(group_df)),
                "percentage": float(len(group_df) / total) if total > 0 else None,
                "accuracy": float(acc.mean()) if not acc.empty else None,
                "cot_compression_ratio_mean": float(ratio.mean()) if not ratio.empty else None,
                "cot_compression_ratio_median": float(ratio.median()) if not ratio.empty else None,
                "source_cot_tokens_median": float(group_df["source_cot_tokens"].dropna().median())
                if not group_df["source_cot_tokens"].dropna().empty
                else None,
                "distilled_cot_tokens_median": float(group_df["distilled_cot_tokens"].dropna().median())
                if not group_df["distilled_cot_tokens"].dropna().empty
                else None,
                "insight_tokens_median": float(group_df["insight_tokens"].dropna().median())
                if not group_df["insight_tokens"].dropna().empty
                else None,
            }
        )

    rows.sort(key=lambda row: row["count"], reverse=True)
    return rows[:top_k]


def build_data_composition_summary(df: pd.DataFrame) -> Dict[str, Any]:
    training_config_fields = [
        "latent_backprop_strategy",
        "state_align_enabled",
        "state_align_target",
        "mask_prompt_loss",
        "mask_system_loss",
        "latent_loss_weight",
        "cot_loss_weight",
        "answer_loss_weight",
    ]
    return {
        "record_schema_distribution": categorical_distribution(df, "record_schema"),
        "dataset_source_distribution": categorical_distribution(df, "dataset_source"),
        "original_dataset_distribution": categorical_distribution(df, "original_dataset"),
        "difficulty_distribution": categorical_distribution(df, "difficulty"),
        "difficulty_rank_distribution": categorical_distribution(df, "difficulty_rank"),
        "n_latent_steps_distribution": categorical_distribution(df, "n_latent_steps"),
        "training_config_distribution": {
            field: categorical_distribution(df, field) for field in training_config_fields
        },
    }


def build_grouped_quality_summary(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "by_dataset_source": grouped_quality_summary(df, "dataset_source"),
        "by_original_dataset": grouped_quality_summary(df, "original_dataset"),
        "by_difficulty": grouped_quality_summary(df, "difficulty"),
        "by_n_latent_steps": grouped_quality_summary(df, "n_latent_steps"),
        "by_record_schema": grouped_quality_summary(df, "record_schema"),
    }


def _accuracy_by_bins(df: pd.DataFrame, value_col: str, bins: int) -> pd.DataFrame:
    subset = df[[value_col, "is_correct_int"]].dropna()
    if subset.empty or subset[value_col].nunique() < 2:
        return pd.DataFrame()

    q = max(2, min(bins, int(subset[value_col].nunique())))
    binned = pd.qcut(subset[value_col], q=q, duplicates="drop")
    work = subset.loc[binned.index].copy()
    work["bin"] = binned

    result = (
        work.groupby("bin", observed=True)
        .agg(
            x_median=(value_col, "median"),
            x_min=(value_col, "min"),
            x_max=(value_col, "max"),
            accuracy=("is_correct_int", "mean"),
            count=("is_correct_int", "size"),
        )
        .reset_index(drop=True)
    )
    return result


def write_json_data(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_dataframe_csv(path: Path, df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
    if columns is not None:
        export_df = df.reindex(columns=columns)
    else:
        export_df = df
    export_df.to_csv(path, index=False)


def series_histogram_data(series: pd.Series, bins: int, clip_quantile: Optional[float] = None) -> Dict[str, Any]:
    clean = series.dropna()
    if clean.empty:
        return {
            "count": 0,
            "visible_count": 0,
            "clip_quantile": clip_quantile,
            "clip_max": None,
            "median": None,
            "bins": [],
        }

    clip_max = None
    visible = clean
    if clip_quantile is not None:
        clip_max = float(clean.quantile(clip_quantile))
        if clip_max > 0:
            visible = clean[clean <= clip_max]

    counts, edges = np.histogram(visible.to_numpy(dtype=float), bins=bins)
    rows = [
        {
            "bin_left": float(edges[idx]),
            "bin_right": float(edges[idx + 1]),
            "count": int(count),
        }
        for idx, count in enumerate(counts)
    ]
    return {
        "count": int(clean.shape[0]),
        "visible_count": int(visible.shape[0]),
        "clip_quantile": clip_quantile,
        "clip_max": clip_max,
        "median": float(clean.median()),
        "bins": rows,
    }


def scatter_records(df: pd.DataFrame, columns: List[str]) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    return _to_jsonable(df.reindex(columns=columns).to_dict(orient="records"))


def _save_placeholder_plot(output_path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_token_length_distributions(df: pd.DataFrame, output_path: Path) -> None:
    fields = [
        ("ground_truth_tokens", "ground_truth tokens"),
        ("insight_tokens", "insight tokens"),
        ("source_cot_tokens", "source cot tokens (before compression)"),
        ("distilled_cot_tokens", "distilled cot tokens (after compression)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (col, title) in zip(axes.flatten(), fields):
        series = df[col].dropna()
        if series.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            continue

        clip_max = float(series.quantile(0.99))
        visible = series[series <= clip_max] if clip_max > 0 else series
        ax.hist(visible, bins=40, color="#4C72B0", alpha=0.85)
        ax.axvline(series.median(), color="#D62728", linestyle="--", linewidth=1.5, label="median")
        ax.set_title(title)
        ax.set_xlabel("tokens")
        ax.set_ylabel("count")
        ax.legend(loc="upper right")

    fig.suptitle("Token Length Distributions (99th percentile clipped)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cot_ratio_vs_accuracy(df: pd.DataFrame, output_path: Path, bins: int) -> None:
    subset = df[["cot_compression_ratio", "is_correct_int"]].dropna()
    if subset.empty:
        _save_placeholder_plot(
            output_path,
            "CoT Compression Ratio vs Accuracy",
            "No valid samples with both compression ratio and correctness.",
        )
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    rng = np.random.default_rng(42)
    jitter = rng.normal(0.0, 0.03, size=len(subset))
    axes[0].scatter(
        subset["cot_compression_ratio"],
        subset["is_correct_int"] + jitter,
        s=12,
        alpha=0.35,
        color="#1F77B4",
    )
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].set_yticks([0.0, 1.0])
    axes[0].set_yticklabels(["incorrect", "correct"])
    axes[0].set_xlabel("cot compression ratio")
    axes[0].set_title("Sample-level relation")

    table = _accuracy_by_bins(subset, "cot_compression_ratio", bins=bins)
    if table.empty:
        axes[1].text(0.5, 0.5, "Not enough bins", ha="center", va="center")
        axes[1].set_title("Binned accuracy")
    else:
        axes[1].plot(table["x_median"], table["accuracy"], marker="o", linewidth=2, color="#2CA02C")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_xlabel("cot compression ratio (bin median)")
        axes[1].set_ylabel("accuracy")
        axes[1].set_title("Binned ratio vs accuracy")
        for _, row in table.iterrows():
            axes[1].annotate(
                str(int(row["count"])),
                (row["x_median"], row["accuracy"]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
            )

    fig.suptitle("CoT Compression Ratio and Correctness")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_insight_cot_ratio_relations(df: pd.DataFrame, output_path: Path) -> None:
    need_cols = ["insight_tokens", "source_cot_tokens", "distilled_cot_tokens", "cot_compression_ratio"]
    subset = df[need_cols + ["is_correct_int"]].dropna(subset=need_cols)
    if subset.empty:
        _save_placeholder_plot(
            output_path,
            "Insight vs CoT Length and Compression",
            "No valid samples for relation plots.",
        )
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sc0 = axes[0].scatter(
        subset["insight_tokens"],
        subset["source_cot_tokens"],
        c=subset["cot_compression_ratio"],
        cmap="viridis",
        s=12,
        alpha=0.45,
    )
    axes[0].set_xlabel("insight tokens")
    axes[0].set_ylabel("source cot tokens")
    axes[0].set_title("Insight vs Source CoT")
    fig.colorbar(sc0, ax=axes[0], label="compression ratio")

    sc1 = axes[1].scatter(
        subset["insight_tokens"],
        subset["distilled_cot_tokens"],
        c=subset["cot_compression_ratio"],
        cmap="viridis",
        s=12,
        alpha=0.45,
    )
    axes[1].set_xlabel("insight tokens")
    axes[1].set_ylabel("distilled cot tokens")
    axes[1].set_title("Insight vs Distilled CoT")
    fig.colorbar(sc1, ax=axes[1], label="compression ratio")

    colors = np.where(subset["is_correct_int"].fillna(0) >= 0.5, "#2CA02C", "#D62728")
    axes[2].scatter(
        subset["insight_tokens"],
        subset["cot_compression_ratio"],
        c=colors,
        s=12,
        alpha=0.45,
    )
    axes[2].set_xlabel("insight tokens")
    axes[2].set_ylabel("cot compression ratio")
    axes[2].set_title("Insight vs Compression Ratio")

    fig.suptitle("Relations: Insight Length, Pre/Post CoT Lengths, Compression")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ratio_distribution(
    df: pd.DataFrame,
    output_path: Path,
    normal_ratio_threshold: float,
    abnormal_ratio_threshold: float,
) -> None:
    ratio = df["cot_compression_ratio"].dropna()
    if ratio.empty:
        _save_placeholder_plot(output_path, "Compression Ratio Distribution", "No compression ratio values.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(ratio, bins=60, color="#4C72B0", alpha=0.9)
    axes[0].axvline(normal_ratio_threshold, color="#2CA02C", linestyle="--", label=f"normal<{normal_ratio_threshold}")
    axes[0].axvline(abnormal_ratio_threshold, color="#D62728", linestyle="--", label=f"abnormal>{abnormal_ratio_threshold}")
    axes[0].set_xlabel("cot compression ratio")
    axes[0].set_ylabel("count")
    axes[0].set_title("Overall ratio distribution")
    axes[0].legend(loc="upper right")

    correct_ratio = df.loc[df["is_correct_int"] == 1, "cot_compression_ratio"].dropna()
    incorrect_ratio = df.loc[df["is_correct_int"] == 0, "cot_compression_ratio"].dropna()
    if not correct_ratio.empty:
        axes[1].hist(correct_ratio, bins=50, alpha=0.6, label="correct", color="#2CA02C")
    if not incorrect_ratio.empty:
        axes[1].hist(incorrect_ratio, bins=50, alpha=0.6, label="incorrect", color="#D62728")
    axes[1].axvline(normal_ratio_threshold, color="#2CA02C", linestyle="--")
    axes[1].axvline(abnormal_ratio_threshold, color="#D62728", linestyle="--")
    axes[1].set_xlabel("cot compression ratio")
    axes[1].set_ylabel("count")
    axes[1].set_title("Ratio distribution by correctness")
    axes[1].legend(loc="upper right")

    fig.suptitle("CoT Compression Ratio Distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def identify_abnormal_cases(df: pd.DataFrame, threshold: float, top_k: int) -> pd.DataFrame:
    abnormal = df[df["cot_compression_ratio"] > threshold].copy()
    if abnormal.empty:
        return abnormal
    abnormal = abnormal.sort_values("cot_compression_ratio", ascending=False)
    return abnormal.head(top_k)


def plot_abnormal_cases(abnormal_df: pd.DataFrame, output_path: Path, threshold: float) -> None:
    if abnormal_df.empty:
        _save_placeholder_plot(
            output_path,
            "Abnormal High Compression Cases",
            f"No cases with cot compression ratio > {threshold}",
        )
        return

    labels = [f"{str(uid)[:8]}@L{int(line_no)}" for uid, line_no in zip(abnormal_df["uid"], abnormal_df["line_no"])]
    ratios = abnormal_df["cot_compression_ratio"].to_numpy()
    colors = np.where(abnormal_df["is_correct_int"].fillna(0) >= 0.5, "#2CA02C", "#D62728")

    fig, ax = plt.subplots(figsize=(12, 6))
    y = np.arange(len(labels))
    ax.barh(y, ratios, color=colors, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(threshold, color="#000000", linestyle="--", linewidth=1.2, label=f"threshold={threshold}")
    ax.set_xlabel("cot compression ratio")
    ax.set_title("Top Abnormal High Compression Ratio Cases")
    ax.legend(loc="lower right")

    for i, ratio in enumerate(ratios):
        ax.text(ratio, i, f" {ratio:.2f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_normal_ratio_metrics(df: pd.DataFrame, output_path: Path, threshold: float) -> None:
    normal = df[df["cot_compression_ratio"] < threshold].copy()
    if normal.empty:
        _save_placeholder_plot(
            output_path,
            "Metrics Under Normal Compression Ratio",
            f"No samples where cot compression ratio < {threshold}",
        )
        return

    fields = [
        ("ground_truth_tokens", "ground_truth tokens"),
        ("insight_tokens", "insight tokens"),
        ("source_cot_tokens", "source cot tokens"),
        ("distilled_cot_tokens", "distilled cot tokens"),
    ]

    acc = normal["is_correct_int"].dropna().mean()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (col, title) in zip(axes.flatten(), fields):
        series = normal[col].dropna()
        if series.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            continue

        clip_max = float(series.quantile(0.99))
        visible = series[series <= clip_max] if clip_max > 0 else series
        ax.hist(visible, bins=40, color="#59A14F", alpha=0.85)
        ax.axvline(series.median(), color="#D62728", linestyle="--", linewidth=1.5, label="median")
        ax.set_title(title)
        ax.set_xlabel("tokens")
        ax.set_ylabel("count")
        ax.legend(loc="upper right")

    fig.suptitle(
        f"Metrics in Normal Compression Region (ratio < {threshold}, n={len(normal)}, acc={acc:.4f})"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _histogram_rows(field: str, histogram: Dict[str, Any], group: Optional[str] = None) -> List[Dict[str, Any]]:
    rows = []
    for row in histogram["bins"]:
        payload = {
            "field": field,
            "bin_left": row["bin_left"],
            "bin_right": row["bin_right"],
            "count": row["count"],
            "median": histogram["median"],
            "clip_max": histogram["clip_max"],
        }
        if group is not None:
            payload["group"] = group
        rows.append(payload)
    return rows


def _write_histogram_exports(
    plot_data_dir: Path,
    stem: str,
    payload: Dict[str, Any],
    rows: List[Dict[str, Any]],
    columns: List[str],
) -> None:
    write_json_data(plot_data_dir / f"{stem}.json", payload)
    write_dataframe_csv(plot_data_dir / f"{stem}.csv", pd.DataFrame(rows), columns)


def export_token_length_distribution_data(df: pd.DataFrame, plot_data_dir: Path) -> None:
    fields = [
        ("ground_truth_tokens", "ground_truth tokens"),
        ("insight_tokens", "insight tokens"),
        ("source_cot_tokens", "source cot tokens (before compression)"),
        ("distilled_cot_tokens", "distilled cot tokens (after compression)"),
    ]
    payload = {"fields": {}}
    rows: List[Dict[str, Any]] = []
    for col, label in fields:
        histogram = series_histogram_data(df[col], bins=40, clip_quantile=0.99)
        payload["fields"][col] = {
            "label": label,
            "stats": _stats(df[col]),
            **histogram,
        }
        rows.extend(_histogram_rows(col, histogram))

    _write_histogram_exports(
        plot_data_dir,
        "token_length_distributions",
        payload,
        rows,
        ["field", "bin_left", "bin_right", "count", "median", "clip_max"],
    )


def export_cot_ratio_vs_accuracy_data(df: pd.DataFrame, plot_data_dir: Path, bins: int) -> None:
    scatter_cols = ["line_no", "uid", "record_schema", "cot_compression_ratio", "is_correct_int"]
    scatter_df = df[scatter_cols].dropna(subset=["cot_compression_ratio", "is_correct_int"]).copy()
    binned_df = _accuracy_by_bins(df, "cot_compression_ratio", bins=bins)

    payload = {
        "scatter": {
            "columns": scatter_cols,
            "count": int(len(scatter_df)),
            "records": scatter_records(scatter_df, scatter_cols),
        },
        "binned_accuracy": {
            "columns": ["x_median", "x_min", "x_max", "accuracy", "count"],
            "count": int(len(binned_df)),
            "records": scatter_records(binned_df, ["x_median", "x_min", "x_max", "accuracy", "count"]),
        },
    }
    write_json_data(plot_data_dir / "cot_compression_ratio_vs_accuracy.json", payload)
    write_dataframe_csv(plot_data_dir / "cot_compression_ratio_vs_accuracy_scatter.csv", scatter_df, scatter_cols)
    write_dataframe_csv(
        plot_data_dir / "cot_compression_ratio_vs_accuracy_bins.csv",
        binned_df,
        ["x_median", "x_min", "x_max", "accuracy", "count"],
    )


def export_insight_cot_ratio_relation_data(df: pd.DataFrame, plot_data_dir: Path) -> None:
    columns = [
        "line_no",
        "uid",
        "record_schema",
        "insight_tokens",
        "source_cot_tokens",
        "distilled_cot_tokens",
        "cot_compression_ratio",
        "is_correct_int",
    ]
    subset = df[columns].dropna(subset=["insight_tokens", "source_cot_tokens", "distilled_cot_tokens", "cot_compression_ratio"])
    payload = {
        "columns": columns,
        "count": int(len(subset)),
        "panels": [
            {
                "name": "insight_vs_source_cot",
                "x": "insight_tokens",
                "y": "source_cot_tokens",
                "color": "cot_compression_ratio",
            },
            {
                "name": "insight_vs_distilled_cot",
                "x": "insight_tokens",
                "y": "distilled_cot_tokens",
                "color": "cot_compression_ratio",
            },
            {
                "name": "insight_vs_compression_ratio",
                "x": "insight_tokens",
                "y": "cot_compression_ratio",
                "color": "is_correct_int",
            },
        ],
        "records": scatter_records(subset, columns),
    }
    write_json_data(plot_data_dir / "insight_vs_cot_and_ratio.json", payload)
    write_dataframe_csv(plot_data_dir / "insight_vs_cot_and_ratio.csv", subset, columns)


def export_ratio_distribution_data(
    df: pd.DataFrame,
    plot_data_dir: Path,
    normal_ratio_threshold: float,
    abnormal_ratio_threshold: float,
) -> None:
    groups = {
        "overall": df["cot_compression_ratio"].dropna(),
        "correct": df.loc[df["is_correct_int"] == 1, "cot_compression_ratio"].dropna(),
        "incorrect": df.loc[df["is_correct_int"] == 0, "cot_compression_ratio"].dropna(),
    }
    payload = {
        "normal_ratio_threshold": float(normal_ratio_threshold),
        "abnormal_ratio_threshold": float(abnormal_ratio_threshold),
        "groups": {},
    }
    rows: List[Dict[str, Any]] = []
    for group_name, series in groups.items():
        histogram = series_histogram_data(series, bins=60 if group_name == "overall" else 50)
        payload["groups"][group_name] = histogram
        for row in histogram["bins"]:
            rows.append({"group": group_name, **row})

    _write_histogram_exports(
        plot_data_dir,
        "cot_compression_ratio_distribution",
        payload,
        rows,
        ["group", "bin_left", "bin_right", "count"],
    )


def export_abnormal_case_data(abnormal_df: pd.DataFrame, plot_data_dir: Path) -> None:
    if abnormal_df.empty:
        export_df = pd.DataFrame(
            columns=[
                "label",
                "uid",
                "line_no",
                "record_schema",
                "dataset_source",
                "original_dataset",
                "is_correct_int",
                "score",
                "cot_compression_ratio",
                "insight_tokens",
                "source_cot_tokens",
                "distilled_cot_tokens",
            ]
        )
    else:
        export_df = abnormal_df.copy()
        export_df["label"] = [
            f"{str(uid)[:8]}@L{int(line_no)}" for uid, line_no in zip(export_df["uid"], export_df["line_no"])
        ]
        export_df = export_df[
            [
                "label",
                "uid",
                "line_no",
                "record_schema",
                "dataset_source",
                "original_dataset",
                "is_correct_int",
                "score",
                "cot_compression_ratio",
                "insight_tokens",
                "source_cot_tokens",
                "distilled_cot_tokens",
            ]
        ]

    payload = {
        "count": int(len(export_df)),
        "records": scatter_records(export_df, list(export_df.columns)),
    }
    write_json_data(plot_data_dir / "abnormal_high_compression_cases.json", payload)
    write_dataframe_csv(plot_data_dir / "abnormal_high_compression_cases.csv", export_df)


def export_normal_ratio_metrics_data(df: pd.DataFrame, plot_data_dir: Path, threshold: float) -> None:
    normal = df[df["cot_compression_ratio"] < threshold].copy()
    fields = [
        ("ground_truth_tokens", "ground_truth tokens"),
        ("insight_tokens", "insight tokens"),
        ("source_cot_tokens", "source cot tokens"),
        ("distilled_cot_tokens", "distilled cot tokens"),
    ]
    payload = {
        "threshold": float(threshold),
        "rows": int(len(normal)),
        "accuracy": float(normal["is_correct_int"].dropna().mean()) if not normal["is_correct_int"].dropna().empty else None,
        "fields": {},
    }
    rows: List[Dict[str, Any]] = []
    for col, label in fields:
        histogram = series_histogram_data(normal[col], bins=40, clip_quantile=0.99)
        payload["fields"][col] = {
            "label": label,
            "stats": _stats(normal[col]),
            **histogram,
        }
        rows.extend(_histogram_rows(col, histogram))

    _write_histogram_exports(
        plot_data_dir,
        "normal_ratio_metrics",
        payload,
        rows,
        ["field", "bin_left", "bin_right", "count", "median", "clip_max"],
    )


def export_plot_data_for_split(
    *,
    split_df: pd.DataFrame,
    split_dir: Path,
    bins: int,
    normal_ratio_threshold: float,
    abnormal_ratio_threshold: float,
    abnormal_top_k: int,
) -> Dict[str, Any]:
    plot_data_dir = split_dir / "plot_data"
    plot_data_dir.mkdir(parents=True, exist_ok=True)

    export_token_length_distribution_data(split_df, plot_data_dir)
    export_cot_ratio_vs_accuracy_data(split_df, plot_data_dir, bins=bins)
    export_insight_cot_ratio_relation_data(split_df, plot_data_dir)
    export_ratio_distribution_data(
        split_df,
        plot_data_dir,
        normal_ratio_threshold=normal_ratio_threshold,
        abnormal_ratio_threshold=abnormal_ratio_threshold,
    )
    abnormal_df = identify_abnormal_cases(split_df, threshold=abnormal_ratio_threshold, top_k=abnormal_top_k)
    export_abnormal_case_data(abnormal_df, plot_data_dir)
    export_normal_ratio_metrics_data(split_df, plot_data_dir, threshold=normal_ratio_threshold)

    return {
        "plot_data_dir": str(plot_data_dir),
        "files": sorted(path.name for path in plot_data_dir.iterdir() if path.is_file()),
    }


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _sanitize_filename(text: str, max_len: int = 80) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    safe = safe.strip("_")
    if not safe:
        safe = "unknown"
    return safe[:max_len]


def _trim_record_for_qc(record: Dict[str, Any]) -> Dict[str, Any]:
    pruned = copy.deepcopy(record)

    stage1 = pruned.get("stage1")
    if isinstance(stage1, dict):
        stage1.pop("messages", None)
        stage1.pop("reasoning_content", None)
        stage1.pop("content", None)

    stage2 = pruned.get("stage2")
    if isinstance(stage2, dict):
        stage2.pop("messages", None)

    return pruned


def export_normal_ratio_random_samples(
    *,
    df: pd.DataFrame,
    input_path: Path,
    output_dir: Path,
    use_json_repair: bool,
    normal_ratio_threshold: float,
    sample_count: int,
    sample_seed: int,
) -> Dict[str, Any]:
    target_dir = output_dir / "normal_ratio_random_samples"
    target_dir.mkdir(parents=True, exist_ok=True)

    normal_df = df[df["cot_compression_ratio"] < normal_ratio_threshold].copy()
    if normal_df.empty:
        return {
            "requested": int(sample_count),
            "candidates": 0,
            "saved": 0,
            "output_dir": str(target_dir),
            "sample_seed": int(sample_seed),
        }

    actual_n = min(int(sample_count), int(len(normal_df)))
    sampled = normal_df.sample(n=actual_n, random_state=sample_seed, replace=False)
    sampled = sampled[["line_no", "uid", "cot_compression_ratio", "is_correct"]].reset_index(drop=True)

    selected_line_nos = [int(v) for v in sampled["line_no"].tolist()]
    selected_set = set(selected_line_nos)
    remaining_line_nos = set(selected_line_nos)
    rank_map = {line_no: idx + 1 for idx, line_no in enumerate(selected_line_nos)}
    meta_map = {
        int(row.line_no): {
            "uid": _to_text(row.uid),
        }
        for row in sampled.itertuples(index=False)
    }

    saved = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line_no not in selected_set:
                continue

            stripped = line.strip()
            if not stripped:
                continue

            try:
                record = _json_load_line(stripped, use_json_repair=use_json_repair)
            except Exception:
                continue

            pruned = _trim_record_for_qc(record)
            extra = meta_map.get(line_no, {})

            uid = _sanitize_filename(_to_text(extra.get("uid")) or _to_text(pruned.get("uid")))
            rank = rank_map.get(line_no, saved + 1)
            filename = f"{rank:03d}_line{line_no}_{uid}.json"
            (target_dir / filename).write_text(json.dumps(pruned, ensure_ascii=False, indent=4), encoding="utf-8")
            saved += 1
            remaining_line_nos.discard(line_no)

            if not remaining_line_nos:
                break

    return {
        "requested": int(sample_count),
        "candidates": int(len(normal_df)),
        "saved": int(saved),
        "output_dir": str(target_dir),
        "sample_seed": int(sample_seed),
    }


def build_summary(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    split_name: str,
    split_filter: str,
    normal_ratio_threshold: float,
    abnormal_ratio_threshold: float,
    abnormal_top_k: int,
    normal_sample_export: Dict[str, Any],
) -> Dict[str, Any]:
    ratio_series = df["cot_compression_ratio"].dropna()
    valid_acc = df["is_correct_int"].dropna()

    normal_df = df[df["cot_compression_ratio"] < normal_ratio_threshold].copy()
    abnormal_top_df = identify_abnormal_cases(df, abnormal_ratio_threshold, abnormal_top_k)

    summary = {
        "meta": meta,
        "overall": {
            "split_name": split_name,
            "split_filter": split_filter,
            "rows": int(len(df)),
            "rows_with_ratio": int(ratio_series.shape[0]),
            "rows_with_accuracy": int(valid_acc.shape[0]),
            "overall_accuracy": float(valid_acc.mean()) if not valid_acc.empty else None,
            "normal_ratio_threshold": float(normal_ratio_threshold),
            "abnormal_ratio_threshold": float(abnormal_ratio_threshold),
        },
        "token_stats": {
            "ground_truth_tokens": _stats(df["ground_truth_tokens"]),
            "insight_tokens": _stats(df["insight_tokens"]),
            "source_cot_tokens": _stats(df["source_cot_tokens"]),
            "distilled_cot_tokens": _stats(df["distilled_cot_tokens"]),
        },
        "compression_ratio_stats": _stats(df["cot_compression_ratio"]),
        "data_composition": build_data_composition_summary(df),
        "grouped_quality": build_grouped_quality_summary(df),
        "normal_ratio_region": {
            "rows": int(len(normal_df)),
            "accuracy": float(normal_df["is_correct_int"].dropna().mean())
            if not normal_df["is_correct_int"].dropna().empty
            else None,
            "ground_truth_tokens": _stats(normal_df["ground_truth_tokens"]),
            "insight_tokens": _stats(normal_df["insight_tokens"]),
            "source_cot_tokens": _stats(normal_df["source_cot_tokens"]),
            "distilled_cot_tokens": _stats(normal_df["distilled_cot_tokens"]),
        },
        "abnormal_high_ratio_top_cases": abnormal_top_df[
            [
                "uid",
                "line_no",
                "dataset_source",
                "original_dataset",
                "is_correct",
                "score",
                "cot_compression_ratio",
                "insight_tokens",
                "source_cot_tokens",
                "distilled_cot_tokens",
            ]
        ].to_dict(orient="records"),
        "normal_ratio_random_samples_export": normal_sample_export,
    }
    return _to_jsonable(summary)


def write_summary_markdown(summary: Dict[str, Any], output_path: Path) -> None:
    overall = summary["overall"]
    meta = summary["meta"]
    token_stats = summary["token_stats"]
    ratio_stats = summary["compression_ratio_stats"]
    data_composition = summary.get("data_composition", {})
    grouped_quality = summary.get("grouped_quality", {})
    normal_region = summary["normal_ratio_region"]
    abnormal_cases = summary["abnormal_high_ratio_top_cases"]
    sample_export = summary["normal_ratio_random_samples_export"]

    lines = [
        "# Distilled SFT Token-Based Quality Analysis",
        "",
        "## Data Loading",
        f"- input_path: {meta['input_path']}",
        f"- total_lines: {meta['total_lines']}",
        f"- empty_lines: {meta['empty_lines']}",
        f"- parsed_rows: {meta['parsed_rows']}",
        f"- bad_rows: {meta['bad_rows']}",
        "",
        "## Overall",
        f"- split_name: {overall['split_name']}",
        f"- split_filter: {overall['split_filter']}",
        f"- rows: {overall['rows']}",
        f"- rows_with_ratio: {overall['rows_with_ratio']}",
        f"- rows_with_accuracy: {overall['rows_with_accuracy']}",
        f"- overall_accuracy: {overall['overall_accuracy']}",
        f"- normal_ratio_threshold: {overall['normal_ratio_threshold']}",
        f"- abnormal_ratio_threshold: {overall['abnormal_ratio_threshold']}",
        "",
        "## Token Stats",
    ]

    for key, stats in token_stats.items():
        lines.append(
            f"- {key}: count={stats['count']}, mean={stats['mean']}, median={stats['median']}, "
            f"p25={stats['p25']}, p75={stats['p75']}, p95={stats['p95']}"
        )

    lines.extend(
        [
            "",
            "## Compression Ratio Stats",
            f"- count={ratio_stats['count']}, mean={ratio_stats['mean']}, median={ratio_stats['median']}, "
            f"p25={ratio_stats['p25']}, p75={ratio_stats['p75']}, p95={ratio_stats['p95']}",
            "",
            "## Data Composition",
            "",
            "### Dataset Source Distribution (top 10)",
        ]
    )

    for item in data_composition.get("dataset_source_distribution", [])[:10]:
        lines.append(f"- {item['value']}: count={item['count']}, percentage={item['percentage']}")

    lines.extend(["", "### Original Dataset Distribution (top 10)"])
    for item in data_composition.get("original_dataset_distribution", [])[:10]:
        lines.append(f"- {item['value']}: count={item['count']}, percentage={item['percentage']}")

    lines.extend(["", "### Difficulty Distribution"])
    for item in data_composition.get("difficulty_distribution", [])[:10]:
        lines.append(f"- {item['value']}: count={item['count']}, percentage={item['percentage']}")

    lines.extend(["", "### Latent Step Distribution (top 10)"])
    for item in data_composition.get("n_latent_steps_distribution", [])[:10]:
        lines.append(f"- {item['value']}: count={item['count']}, percentage={item['percentage']}")

    lines.extend(["", "### Grouped Quality by Dataset Source (top 10)",])
    for item in grouped_quality.get("by_dataset_source", [])[:10]:
        lines.append(
            f"- {item['group']}: count={item['count']}, accuracy={item['accuracy']}, "
            f"ratio_median={item['cot_compression_ratio_median']}, distilled_cot_tokens_median={item['distilled_cot_tokens_median']}"
        )

    lines.extend(
        [
            "",
            f"## Normal Ratio Region (ratio < {overall['normal_ratio_threshold']})",
            f"- rows: {normal_region['rows']}",
            f"- accuracy: {normal_region['accuracy']}",
            "",
            "## Normal Ratio Random Sample Export",
            f"- requested: {sample_export['requested']}",
            f"- candidates: {sample_export['candidates']}",
            f"- saved: {sample_export['saved']}",
            f"- sample_seed: {sample_export['sample_seed']}",
            f"- output_dir: {sample_export['output_dir']}",
            "",
            "## Abnormal High Ratio Top Cases",
        ]
    )

    if not abnormal_cases:
        lines.append("- No abnormal cases above threshold.")
    else:
        for case in abnormal_cases:
            lines.append(
                f"- uid={case['uid']}, line={case['line_no']}, ratio={case['cot_compression_ratio']}, "
                f"correct={case['is_correct']}, source_cot_tokens={case['source_cot_tokens']}, "
                f"distilled_cot_tokens={case['distilled_cot_tokens']}, insight_tokens={case['insight_tokens']}"
            )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(
    input_path: Path,
    output_dir: Path,
    tokenizer_path: str,
    max_samples: Optional[int],
    bins: int,
    use_json_repair: bool,
    normal_ratio_threshold: float,
    abnormal_ratio_threshold: float,
    abnormal_top_k: int,
    normal_sample_count: int,
    normal_sample_seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    ensure_latent_think_special_tokens(tokenizer)
    token_counter = TokenCounter(tokenizer)

    df, meta = load_metrics_dataframe(
        input_path=input_path,
        max_samples=max_samples,
        use_json_repair=use_json_repair,
        token_counter=token_counter,
    )
    if df.empty:
        raise RuntimeError("No valid rows were parsed from the input JSONL.")

    split_frames: Dict[str, pd.DataFrame] = {
        "all": df.copy(),
        "normal": df[df["cot_compression_ratio"] < normal_ratio_threshold].copy(),
        "abnormal": df[df["cot_compression_ratio"] >= normal_ratio_threshold].copy(),
    }
    split_filters: Dict[str, str] = {
        "all": "all samples",
        "normal": f"cot_compression_ratio < {normal_ratio_threshold}",
        "abnormal": f"cot_compression_ratio >= {normal_ratio_threshold}",
    }

    split_summaries: Dict[str, Dict[str, Any]] = {}

    for split_name, split_df in split_frames.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        if split_df.empty:
            _save_placeholder_plot(
                split_dir / "token_length_distributions.png",
                f"{split_name}: Token Length Distributions",
                "No data in this split.",
            )
            _save_placeholder_plot(
                split_dir / "cot_compression_ratio_vs_accuracy.png",
                f"{split_name}: CoT Compression Ratio vs Accuracy",
                "No data in this split.",
            )
            _save_placeholder_plot(
                split_dir / "insight_vs_cot_and_ratio.png",
                f"{split_name}: Insight vs CoT and Ratio",
                "No data in this split.",
            )
            _save_placeholder_plot(
                split_dir / "cot_compression_ratio_distribution.png",
                f"{split_name}: CoT Compression Ratio Distribution",
                "No data in this split.",
            )
            _save_placeholder_plot(
                split_dir / "abnormal_high_compression_cases.png",
                f"{split_name}: Abnormal High Compression Cases",
                "No data in this split.",
            )
            _save_placeholder_plot(
                split_dir / "normal_ratio_metrics.png",
                f"{split_name}: Normal Ratio Metrics",
                "No data in this split.",
            )
        else:
            plot_token_length_distributions(split_df, split_dir / "token_length_distributions.png")
            plot_cot_ratio_vs_accuracy(split_df, split_dir / "cot_compression_ratio_vs_accuracy.png", bins=bins)
            plot_insight_cot_ratio_relations(split_df, split_dir / "insight_vs_cot_and_ratio.png")
            plot_ratio_distribution(
                split_df,
                split_dir / "cot_compression_ratio_distribution.png",
                normal_ratio_threshold=normal_ratio_threshold,
                abnormal_ratio_threshold=abnormal_ratio_threshold,
            )

            abnormal_df = identify_abnormal_cases(split_df, threshold=abnormal_ratio_threshold, top_k=abnormal_top_k)
            plot_abnormal_cases(
                abnormal_df,
                split_dir / "abnormal_high_compression_cases.png",
                threshold=abnormal_ratio_threshold,
            )
            plot_normal_ratio_metrics(
                split_df,
                split_dir / "normal_ratio_metrics.png",
                threshold=normal_ratio_threshold,
            )

        if split_name == "normal":
            normal_sample_export = export_normal_ratio_random_samples(
                df=split_df,
                input_path=input_path,
                output_dir=split_dir,
                use_json_repair=use_json_repair,
                normal_ratio_threshold=normal_ratio_threshold,
                sample_count=normal_sample_count,
                sample_seed=normal_sample_seed,
            )
        else:
            normal_sample_export = {
                "requested": 0,
                "candidates": int(len(split_df)),
                "saved": 0,
                "output_dir": str(split_dir / "normal_ratio_random_samples"),
                "sample_seed": normal_sample_seed,
                "note": "QC random sample export is only generated for split=normal.",
            }

        summary = build_summary(
            df=split_df,
            meta=meta,
            split_name=split_name,
            split_filter=split_filters[split_name],
            normal_ratio_threshold=normal_ratio_threshold,
            abnormal_ratio_threshold=abnormal_ratio_threshold,
            abnormal_top_k=abnormal_top_k,
            normal_sample_export=normal_sample_export,
        )
        plot_data_export = export_plot_data_for_split(
            split_df=split_df,
            split_dir=split_dir,
            bins=bins,
            normal_ratio_threshold=normal_ratio_threshold,
            abnormal_ratio_threshold=abnormal_ratio_threshold,
            abnormal_top_k=abnormal_top_k,
        )
        summary["plot_data_dir"] = plot_data_export["plot_data_dir"]
        summary["plot_data_files"] = plot_data_export["files"]
        (split_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_summary_markdown(summary, split_dir / "summary.md")
        split_summaries[split_name] = summary

    index_summary = {
        "meta": meta,
        "split_filters": split_filters,
        "split_rows": {name: int(len(frame)) for name, frame in split_frames.items()},
        "split_output_dirs": {name: str(output_dir / name) for name in split_frames},
        "split_plot_data_dirs": {name: str(output_dir / name / "plot_data") for name in split_frames},
    }
    (output_dir / "summary_index.json").write_text(
        json.dumps(index_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Analysis complete. Output directory: {output_dir}")
    print(f"Parsed rows: {meta['parsed_rows']} | Bad rows: {meta['bad_rows']} | Total lines: {meta['total_lines']}")
    print(
        "Split rows: "
        + ", ".join(f"{name}={index_summary['split_rows'][name]}" for name in ["all", "normal", "abnormal"])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Token-based quality analysis for distilled SFT data. "
            "Focuses on token length distributions, compression ratio behavior, and correctness relations."
        )
    )
    parser.add_argument("--input_path", type=str, default=DEFAULT_INPUT_PATH, help="Path to distilled JSONL data file.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save plots and summary.")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=DEFAULT_TOKENIZER_PATH,
        help="Tokenizer path for token counting.",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on parsed valid samples.")
    parser.add_argument("--bins", type=int, default=10, help="Quantile bin count for ratio-accuracy analysis.")
    parser.add_argument(
        "--normal_ratio_threshold",
        type=float,
        default=1.5,
        help="Threshold for normal compression ratio region.",
    )
    parser.add_argument(
        "--abnormal_ratio_threshold",
        type=float,
        default=1.5,
        help="Threshold for abnormal high compression ratio cases.",
    )
    parser.add_argument(
        "--abnormal_top_k",
        type=int,
        default=20,
        help="How many abnormal high-ratio cases to include in summary/plot.",
    )
    parser.add_argument(
        "--normal_sample_count",
        type=int,
        default=100,
        help="Random sample size from normal compression ratio region for QC JSON export.",
    )
    parser.add_argument(
        "--normal_sample_seed",
        type=int,
        default=42,
        help="Random seed for normal compression ratio QC sampling.",
    )
    parser.add_argument(
        "--use_json_repair",
        action="store_true",
        help="Use json_repair as fallback when a JSONL row is malformed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(
        input_path=Path(args.input_path),
        output_dir=Path(args.output_dir),
        tokenizer_path=args.tokenizer_path,
        max_samples=args.max_samples,
        bins=max(2, args.bins),
        use_json_repair=args.use_json_repair,
        normal_ratio_threshold=args.normal_ratio_threshold,
        abnormal_ratio_threshold=args.abnormal_ratio_threshold,
        abnormal_top_k=max(1, args.abnormal_top_k),
        normal_sample_count=max(1, args.normal_sample_count),
        normal_sample_seed=args.normal_sample_seed,
    )


if __name__ == "__main__":
    main()
