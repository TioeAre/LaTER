import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from transformers import AutoTokenizer


# =========================
# Editable config
# =========================
LATENT_DATA_DIR = (
    ""
)
COT_DATA_DIR = (
    ""
    ""
)
TOKENIZER_NAME = "models/Qwen3-14B"
OUTPUT_DIR = (
    ""
    ""
)

MAX_FILES: Optional[int] = 30
MAX_SAMPLES: Optional[int] = None
MIN_TOKENS_PER_SENTENCE = 3
NUM_PROGRESS_BINS = 24
EXPORT_SENTENCE_MAX_ENTROPY = True
DRAW_SENTENCE_MAX_ENTROPY = True
BOXPLOT_SORT_BY = "max"


SENTENCE_BOUNDARY_RE = re.compile(r".*?(?:[.!?;。！？；]+(?:['\"\)\]]*)|\n{2,}|$)", re.DOTALL)


def list_pt_files(data_dir: str, max_files: Optional[int]) -> List[Path]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    files = [path for path in root.iterdir() if path.suffix == ".pt"]
    try:
        files.sort(key=lambda path: int(path.stem.split("_")[1]))
    except (IndexError, ValueError):
        files.sort()

    if max_files is not None:
        files = files[:max_files]
    return files


def extract_batch_base_index(pt_path: Path) -> int:
    try:
        return int(pt_path.stem.split("_")[1])
    except (IndexError, ValueError):
        return 0


def load_tokenizer(name: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, use_fast=True, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required because offset mappings are needed for sentence alignment.")
    return tokenizer


def split_sentences(text: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    for match in SENTENCE_BOUNDARY_RE.finditer(text):
        start, end = match.span()
        chunk = text[start:end]
        if not chunk:
            continue
        stripped = chunk.strip()
        if not stripped:
            continue
        left_trim = len(chunk) - len(chunk.lstrip())
        right_trim = len(chunk.rstrip())
        spans.append((start + left_trim, start + right_trim, stripped))
    return spans


def tokenize_with_offsets(tokenizer, text: str) -> Tuple[List[Tuple[int, int]], List[int]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = []
    keep_indices = []
    for idx, offset in enumerate(encoded["offset_mapping"]):
        start, end = int(offset[0]), int(offset[1])
        if end <= start:
            continue
        offsets.append((start, end))
        keep_indices.append(idx)
    return offsets, keep_indices


def align_lengths(
    entropies: Sequence[float],
    type_masks: Sequence[int],
    offsets: Sequence[Tuple[int, int]],
) -> Tuple[List[float], List[int], List[Tuple[int, int]]]:
    min_len = min(len(entropies), len(type_masks), len(offsets))
    if min_len == 0:
        return [], [], []

    if not (len(entropies) == len(type_masks) == len(offsets)):
        logger.warning(
            "Length mismatch before trimming: entropies={} type_masks={} offsets={}. Trim to min_len={}",
            len(entropies),
            len(type_masks),
            len(offsets),
            min_len,
        )

    return (
        [float(x) for x in entropies[:min_len]],
        [int(x) for x in type_masks[:min_len]],
        list(offsets[:min_len]),
    )


def build_sentence_records(
    text: str,
    entropies: Sequence[float],
    type_masks: Sequence[int],
    offsets: Sequence[Tuple[int, int]],
    group: str,
) -> List[Dict]:
    sentence_spans = split_sentences(text)
    if not sentence_spans:
        return []

    target_mask_value = 0 if group == "latent" else 1
    sentence_records: List[Dict] = []

    for start_char, end_char, sentence_text in sentence_spans:
        token_indices = []
        for token_idx, ((token_start, token_end), token_type) in enumerate(zip(offsets, type_masks)):
            if token_type != target_mask_value:
                continue
            if token_end <= start_char or token_start >= end_char:
                continue
            token_indices.append(token_idx)

        if len(token_indices) < MIN_TOKENS_PER_SENTENCE:
            continue

        token_entropies = [float(entropies[idx]) for idx in token_indices]
        sentence_records.append(
            {
                "sentence_text": sentence_text,
                "start_char": start_char,
                "end_char": end_char,
                "start_token_idx": token_indices[0],
                "end_token_idx": token_indices[-1],
                "num_tokens": len(token_indices),
                "mean_entropy": float(np.mean(token_entropies)),
                "max_entropy": float(np.max(token_entropies)),
            }
        )

    if not sentence_records:
        return []

    total_sentences = len(sentence_records)
    for idx, record in enumerate(sentence_records):
        progress = 0.5 if total_sentences == 1 else idx / max(1, total_sentences - 1)
        record["sentence_index"] = idx
        record["progress"] = float(progress)

    return sentence_records


def summarize_token_boxplot(sample_id: int, token_entropies: Sequence[float], source_file: str) -> Dict:
    values = np.asarray(token_entropies, dtype=np.float64)
    return {
        "sample_id": sample_id,
        "source_file": source_file,
        "count": int(values.size),
        "min": float(np.min(values)),
        "q1": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "q3": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def collect_group_samples(group: str, data_dir: str, tokenizer, max_files: Optional[int], max_samples: Optional[int]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    samples: List[Dict] = []
    boxplot_stats: List[Dict] = []
    skipped: List[Dict] = []
    processed = 0

    for pt_path in list_pt_files(data_dir, max_files):
        logger.info(f"[{group}] Loading {pt_path}")
        data = torch.load(pt_path, map_location="cpu")
        texts_batch = data.get("texts", [])
        entropies_batch = data.get("entropies", [])
        type_masks_batch = data.get("type_masks", [])
        batch_size = min(len(texts_batch), len(entropies_batch), len(type_masks_batch))
        batch_base_index = extract_batch_base_index(pt_path)

        for batch_offset in range(batch_size):
            sample_id = batch_base_index + batch_offset
            text = str(texts_batch[batch_offset])
            entropies = entropies_batch[batch_offset]
            type_masks = type_masks_batch[batch_offset]

            try:
                offsets, _ = tokenize_with_offsets(tokenizer, text)
                entropies_aligned, masks_aligned, offsets_aligned = align_lengths(entropies, type_masks, offsets)
                if not entropies_aligned:
                    raise ValueError("Empty aligned entropy sequence.")

                sentence_records = build_sentence_records(
                    text=text,
                    entropies=entropies_aligned,
                    type_masks=masks_aligned,
                    offsets=offsets_aligned,
                    group=group,
                )
                if not sentence_records:
                    raise ValueError("No valid sentences after sentence split and token filtering.")

                sample_payload = {
                    "sample_id": sample_id,
                    "group": group,
                    "source_file": str(pt_path),
                    "text_length": len(text),
                    "num_aligned_tokens": len(entropies_aligned),
                    "num_sentences": len(sentence_records),
                    "sentences": sentence_records,
                }
                samples.append(sample_payload)

                if group == "latent":
                    latent_token_entropies = [
                        float(ent) for ent, mask in zip(entropies_aligned, masks_aligned) if int(mask) == 0
                    ]
                    if latent_token_entropies:
                        boxplot_stats.append(
                            summarize_token_boxplot(
                                sample_id=sample_id,
                                token_entropies=latent_token_entropies,
                                source_file=str(pt_path),
                            )
                        )

                processed += 1
                if max_samples is not None and processed >= max_samples:
                    return samples, boxplot_stats, skipped
            except Exception as exc:
                logger.warning(f"[{group}] Skip sample {sample_id} from {pt_path.name}: {exc}")
                skipped.append(
                    {
                        "group": group,
                        "sample_id": sample_id,
                        "source_file": str(pt_path),
                        "reason": str(exc),
                    }
                )

    return samples, boxplot_stats, skipped


def summarize_matrix(values: np.ndarray) -> Dict[str, List[float]]:
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "p25": np.percentile(values, 25, axis=0).tolist(),
        "p50": np.percentile(values, 50, axis=0).tolist(),
        "p75": np.percentile(values, 75, axis=0).tolist(),
    }


def build_progress_curve(samples: List[Dict], metric_name: str, num_bins: int) -> Dict:
    progress_axis = np.linspace(0.0, 1.0, num_bins)
    groups: Dict[str, Dict] = {}

    for group_name in sorted({sample["group"] for sample in samples}):
        group_samples = [sample for sample in samples if sample["group"] == group_name]
        per_sample_curves = []
        for sample in group_samples:
            sentence_progress = np.asarray([row["progress"] for row in sample["sentences"]], dtype=np.float64)
            sentence_values = np.asarray([row[metric_name] for row in sample["sentences"]], dtype=np.float64)
            if sentence_values.size == 1:
                curve = np.repeat(sentence_values[0], num_bins)
            else:
                curve = np.interp(progress_axis, sentence_progress, sentence_values)
            per_sample_curves.append(curve)

        curve_matrix = np.vstack(per_sample_curves)
        groups[group_name] = {
            "count": len(group_samples),
            "progress_axis": progress_axis.tolist(),
            "curve": summarize_matrix(curve_matrix),
        }

    return groups


def build_sentence_summary(samples: List[Dict]) -> Dict:
    return {
        "config": {
            "latent_data_dir": LATENT_DATA_DIR,
            "cot_data_dir": COT_DATA_DIR,
            "tokenizer_name": TOKENIZER_NAME,
            "output_dir": OUTPUT_DIR,
            "max_files": MAX_FILES,
            "max_samples": MAX_SAMPLES,
            "min_tokens_per_sentence": MIN_TOKENS_PER_SENTENCE,
            "num_progress_bins": NUM_PROGRESS_BINS,
        },
        "groups": {
            group: {
                "count": payload["count"],
                "progress_axis": payload["progress_axis"],
                "mean_entropy_curve": payload["curve"],
                "max_entropy_curve": build_progress_curve(samples, "max_entropy", NUM_PROGRESS_BINS)[group]["curve"],
            }
            for group, payload in build_progress_curve(samples, "mean_entropy", NUM_PROGRESS_BINS).items()
        },
    }


def ensure_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_sentence_samples_csv(path: Path, samples: List[Dict]) -> None:
    fieldnames = [
        "group",
        "sample_id",
        "source_file",
        "sentence_index",
        "progress",
        "num_tokens",
        "mean_entropy",
        "max_entropy",
        "start_token_idx",
        "end_token_idx",
        "start_char",
        "end_char",
        "sentence_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            for sentence in sample["sentences"]:
                row = {
                    "group": sample["group"],
                    "sample_id": sample["sample_id"],
                    "source_file": sample["source_file"],
                    **sentence,
                }
                writer.writerow(row)


def write_sentence_summary_csv(path: Path, summary: Dict) -> None:
    fieldnames = ["group", "metric", "progress_bin", "progress", "mean", "p25", "p50", "p75", "count"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for group_name, payload in summary["groups"].items():
            for metric_name in ["mean_entropy_curve", "max_entropy_curve"]:
                curve = payload[metric_name]
                for idx, progress in enumerate(payload["progress_axis"]):
                    writer.writerow(
                        {
                            "group": group_name,
                            "metric": metric_name,
                            "progress_bin": idx,
                            "progress": progress,
                            "mean": curve["mean"][idx],
                            "p25": curve["p25"][idx],
                            "p50": curve["p50"][idx],
                            "p75": curve["p75"][idx],
                            "count": payload["count"],
                        }
                    )


def write_boxplot_raw_csv(path: Path, boxplot_values: List[Dict]) -> None:
    fieldnames = ["sample_id", "token_entropy"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in boxplot_values:
            sample_id = item["sample_id"]
            for entropy in item["token_entropies"]:
                writer.writerow({"sample_id": sample_id, "token_entropy": entropy})


def plot_entropy_trend(summary: Dict, metric_name: str, path: Path, title: str, y_label: str) -> None:
    colors = {"latent": "#1f77b4", "cot": "#d62728"}
    fig, ax = plt.subplots(figsize=(10, 6))

    for group_name, payload in summary["groups"].items():
        x = payload["progress_axis"]
        curve = payload[metric_name]
        ax.plot(x, curve["mean"], color=colors[group_name], linewidth=2.5, label=f"{group_name} mean")
        ax.plot(x, curve["p50"], color=colors[group_name], linestyle="--", linewidth=1.4, label=f"{group_name} median")
        ax.fill_between(x, curve["p25"], curve["p75"], color=colors[group_name], alpha=0.18)

    ax.set_title(title)
    ax.set_xlabel("Relative Sentence Progress")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_latent_boxplot(boxplot_stats: List[Dict], path: Path, title: str, sort_by: str) -> None:
    if not boxplot_stats:
        return

    sorted_stats = sorted(boxplot_stats, key=lambda item: item.get(sort_by, item["max"]))
    labels = [str(item["sample_id"]) for item in sorted_stats]
    stats = [
        {
            "label": str(item["sample_id"]),
            "whislo": item["min"],
            "q1": item["q1"],
            "med": item["median"],
            "q3": item["q3"],
            "whishi": item["max"],
            "fliers": [],
        }
        for item in sorted_stats
    ]

    fig_width = max(12, len(sorted_stats) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    ax.bxp(stats, showfliers=False, patch_artist=True)
    for patch in ax.artists:
        patch.set_facecolor("#7fb3d5")
        patch.set_alpha(0.55)

    ax.set_title(title)
    ax.set_xlabel(f"Sample ID (sorted by {sort_by} entropy)")
    ax.set_ylabel("Token Entropy")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_analysis() -> None:
    output_dir = ensure_output_dir(OUTPUT_DIR)
    tokenizer = load_tokenizer(TOKENIZER_NAME)

    latent_samples, latent_boxplot_stats, latent_skipped = collect_group_samples(
        group="latent",
        data_dir=LATENT_DATA_DIR,
        tokenizer=tokenizer,
        max_files=MAX_FILES,
        max_samples=MAX_SAMPLES,
    )
    cot_samples, _, cot_skipped = collect_group_samples(
        group="cot",
        data_dir=COT_DATA_DIR,
        tokenizer=tokenizer,
        max_files=MAX_FILES,
        max_samples=MAX_SAMPLES,
    )

    all_samples = latent_samples + cot_samples
    all_skipped = latent_skipped + cot_skipped
    if not all_samples:
        raise RuntimeError("No valid sentence entropy samples were collected.")

    boxplot_raw_values = []
    for pt_path in list_pt_files(LATENT_DATA_DIR, MAX_FILES):
        data = torch.load(pt_path, map_location="cpu")
        batch_base_index = extract_batch_base_index(pt_path)
        batch_size = min(len(data.get("texts", [])), len(data.get("entropies", [])), len(data.get("type_masks", [])))
        for batch_offset in range(batch_size):
            sample_id = batch_base_index + batch_offset
            entropies = data["entropies"][batch_offset]
            masks = data["type_masks"][batch_offset]
            values = [float(ent) for ent, mask in zip(entropies, masks) if int(mask) == 0]
            if values:
                boxplot_raw_values.append({"sample_id": sample_id, "token_entropies": values})

    sentence_summary = build_sentence_summary(all_samples)
    sentence_samples_payload = {
        "config": sentence_summary["config"],
        "samples": all_samples,
        "skipped_samples": all_skipped,
    }
    boxplot_summary = {
        "config": sentence_summary["config"],
        "samples": sorted(latent_boxplot_stats, key=lambda item: item.get(BOXPLOT_SORT_BY, item["max"])),
        "sort_by": BOXPLOT_SORT_BY,
    }

    write_json(output_dir / "sentence_entropy_samples.json", sentence_samples_payload)
    write_json(output_dir / "sentence_entropy_summary.json", sentence_summary)
    write_json(output_dir / "latent_token_boxplot_summary.json", boxplot_summary)
    write_sentence_samples_csv(output_dir / "sentence_entropy_samples.csv", all_samples)
    write_sentence_summary_csv(output_dir / "sentence_entropy_summary.csv", sentence_summary)
    write_boxplot_raw_csv(output_dir / "latent_token_boxplot_samples.csv", boxplot_raw_values)

    plot_entropy_trend(
        sentence_summary,
        metric_name="mean_entropy_curve",
        path=output_dir / "sentence_mean_entropy_trend.png",
        title="Sentence-level Entropy Trend: Latent Reasoning vs CoT",
        y_label="Mean Sentence Entropy",
    )
    if DRAW_SENTENCE_MAX_ENTROPY and EXPORT_SENTENCE_MAX_ENTROPY:
        plot_entropy_trend(
            sentence_summary,
            metric_name="max_entropy_curve",
            path=output_dir / "sentence_max_entropy_trend.png",
            title="Sentence-level Max Entropy Trend: Latent Reasoning vs CoT",
            y_label="Max Sentence Entropy",
        )
    plot_latent_boxplot(
        latent_boxplot_stats,
        path=output_dir / "latent_token_entropy_boxplot.png",
        title="Latent Reasoning Token Entropy Distribution per Sample",
        sort_by=BOXPLOT_SORT_BY,
    )

    logger.info(f"Finished entropy analysis. Outputs saved to {output_dir}")
    logger.info(
        "Valid samples: latent={}, cot={}, skipped={}, latent_boxplots={}",
        len(latent_samples),
        len(cot_samples),
        len(all_skipped),
        len(latent_boxplot_stats),
    )


if __name__ == "__main__":
    run_analysis()
