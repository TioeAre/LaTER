import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from loguru import logger


# =========================
# Editable config
# =========================
LATENT_DATA_DIR = (
    ""
)
COT_DATA_DIR = (
    ""
)
OUTPUT_DIR = (
    ""
    ""
)

MAX_FILES: Optional[int] = 30
MAX_SAMPLES: Optional[int] = None
MIN_STEPS = 8
CENTER_EMBEDDINGS = True
PLOT_RAW_SINGULAR_VALUES = True
PLOT_NORMALIZED_SINGULAR_VALUES = True
PLOT_CUMULATIVE_ENERGY = True


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


def stack_hidden_states(hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
    tensor = torch.stack([state.squeeze().float().cpu() for state in hidden_states])
    if tensor.ndim != 2:
        raise ValueError(f"Expected hidden states to stack into 2D array, got shape {tuple(tensor.shape)}")
    return tensor


def select_sample_matrix(group: str, hidden_states: Sequence[torch.Tensor], type_masks: Sequence[int]) -> Optional[torch.Tensor]:
    if len(hidden_states) != len(type_masks):
        raise ValueError(f"Hidden state length {len(hidden_states)} does not match type mask length {len(type_masks)}")

    if group == "latent":
        selected_indices = [idx for idx, mask in enumerate(type_masks) if mask == 0]
    elif group == "cot":
        selected_indices = [idx for idx, mask in enumerate(type_masks) if mask == 1]
    else:
        raise ValueError(f"Unknown group: {group}")

    if len(selected_indices) < MIN_STEPS:
        return None

    selected_states = [hidden_states[idx] for idx in selected_indices]
    return stack_hidden_states(selected_states)


def compute_svd_metrics(matrix: torch.Tensor, center_embeddings: bool) -> Dict[str, List[float]]:
    working_matrix = matrix.to(dtype=torch.float32)
    if center_embeddings:
        working_matrix = working_matrix - working_matrix.mean(dim=0, keepdim=True)

    num_steps, hidden_dim = working_matrix.shape
    if num_steps <= hidden_dim:
        gram = working_matrix @ working_matrix.transpose(0, 1)
    else:
        gram = working_matrix.transpose(0, 1) @ working_matrix

    eigenvalues = torch.linalg.eigvalsh(gram)
    eigenvalues = torch.flip(eigenvalues, dims=[0])
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    singular_values = torch.sqrt(eigenvalues).cpu().numpy().astype(np.float64, copy=False)

    singular_sum = float(singular_values.sum())
    if singular_sum > 0:
        normalized_singular_values = singular_values / singular_sum
    else:
        normalized_singular_values = np.zeros_like(singular_values)

    squared = singular_values ** 2
    squared_sum = float(squared.sum())
    if squared_sum > 0:
        cumulative_energy = np.cumsum(squared) / squared_sum
    else:
        cumulative_energy = np.zeros_like(singular_values)

    return {
        "singular_values": singular_values.tolist(),
        "normalized_singular_values": normalized_singular_values.tolist(),
        "cumulative_energy": cumulative_energy.tolist(),
    }


def collect_group_samples(group: str, data_dir: str, max_files: Optional[int], max_samples: Optional[int]) -> Tuple[List[Dict], List[Dict]]:
    samples: List[Dict] = []
    skipped: List[Dict] = []
    processed = 0

    for pt_path in list_pt_files(data_dir, max_files):
        logger.info(f"[{group}] Loading {pt_path}")
        data = torch.load(pt_path, map_location="cpu")
        hidden_states_batch = data.get("hidden_states", [])
        type_masks_batch = data.get("type_masks", [])

        batch_base_index = extract_batch_base_index(pt_path)
        batch_size = min(len(hidden_states_batch), len(type_masks_batch))

        for batch_offset in range(batch_size):
            sample_id = batch_base_index + batch_offset
            hidden_states = hidden_states_batch[batch_offset]
            type_masks = type_masks_batch[batch_offset]

            try:
                sample_matrix = select_sample_matrix(group, hidden_states, type_masks)
                if sample_matrix is None:
                    skipped.append(
                        {
                            "group": group,
                            "sample_id": sample_id,
                            "source_file": str(pt_path),
                            "reason": f"effective_steps_below_min_steps({MIN_STEPS})",
                            "original_num_steps": len(hidden_states),
                        }
                    )
                    continue

                metrics = compute_svd_metrics(sample_matrix, center_embeddings=CENTER_EMBEDDINGS)
                samples.append(
                    {
                        "sample_id": sample_id,
                        "group": group,
                        "source_file": str(pt_path),
                        "original_num_steps": len(hidden_states),
                        "num_steps": int(sample_matrix.shape[0]),
                        "hidden_dim": int(sample_matrix.shape[1]),
                        "rank": int(min(sample_matrix.shape)),
                        "center_embeddings": CENTER_EMBEDDINGS,
                        **metrics,
                    }
                )
                processed += 1
                if max_samples is not None and processed >= max_samples:
                    return samples, skipped
            except Exception as exc:
                logger.warning(f"[{group}] Skip sample {sample_id} from {pt_path.name}: {exc}")
                skipped.append(
                    {
                        "group": group,
                        "sample_id": sample_id,
                        "source_file": str(pt_path),
                        "reason": str(exc),
                        "original_num_steps": len(hidden_states),
                    }
                )

    return samples, skipped


def build_summary(samples: List[Dict]) -> Dict:
    if not samples:
        raise ValueError("No valid samples collected, cannot build summary.")

    rank_limit = min(len(sample["singular_values"]) for sample in samples)
    groups: Dict[str, Dict] = {}

    for group_name in sorted({sample["group"] for sample in samples}):
        group_samples = [sample for sample in samples if sample["group"] == group_name]
        singular_matrix = np.asarray([sample["singular_values"][:rank_limit] for sample in group_samples], dtype=np.float64)
        normalized_matrix = np.asarray(
            [sample["normalized_singular_values"][:rank_limit] for sample in group_samples], dtype=np.float64
        )
        cumulative_matrix = np.asarray([sample["cumulative_energy"][:rank_limit] for sample in group_samples], dtype=np.float64)

        groups[group_name] = {
            "count": len(group_samples),
            "common_rank": rank_limit,
            "rank_axis": list(range(1, rank_limit + 1)),
            "singular_values": summarize_matrix(singular_matrix),
            "normalized_singular_values": summarize_matrix(normalized_matrix),
            "cumulative_energy": summarize_matrix(cumulative_matrix),
        }

    return {
        "config": {
            "latent_data_dir": LATENT_DATA_DIR,
            "cot_data_dir": COT_DATA_DIR,
            "output_dir": OUTPUT_DIR,
            "max_files": MAX_FILES,
            "max_samples": MAX_SAMPLES,
            "min_steps": MIN_STEPS,
            "center_embeddings": CENTER_EMBEDDINGS,
        },
        "groups": groups,
    }


def summarize_matrix(values: np.ndarray) -> Dict[str, List[float]]:
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "p25": np.percentile(values, 25, axis=0).tolist(),
        "p50": np.percentile(values, 50, axis=0).tolist(),
        "p75": np.percentile(values, 75, axis=0).tolist(),
    }


def ensure_output_dir(output_dir: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def write_json(output_path: Path, payload: Dict) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_samples_csv(output_path: Path, samples: List[Dict]) -> None:
    fieldnames = [
        "group",
        "sample_id",
        "source_file",
        "original_num_steps",
        "num_steps",
        "hidden_dim",
        "rank_limit",
        "rank",
        "singular_value",
        "normalized_singular_value",
        "cumulative_energy",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            rank_limit = len(sample["singular_values"])
            for rank_idx, (singular_value, normalized_value, cumulative_energy) in enumerate(
                zip(
                    sample["singular_values"],
                    sample["normalized_singular_values"],
                    sample["cumulative_energy"],
                ),
                start=1,
            ):
                writer.writerow(
                    {
                        "group": sample["group"],
                        "sample_id": sample["sample_id"],
                        "source_file": sample["source_file"],
                        "original_num_steps": sample["original_num_steps"],
                        "num_steps": sample["num_steps"],
                        "hidden_dim": sample["hidden_dim"],
                        "rank_limit": rank_limit,
                        "rank": rank_idx,
                        "singular_value": singular_value,
                        "normalized_singular_value": normalized_value,
                        "cumulative_energy": cumulative_energy,
                    }
                )


def write_summary_csv(output_path: Path, summary: Dict) -> None:
    fieldnames = ["group", "metric", "rank", "mean", "p25", "p50", "p75", "count"]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for group_name, group_payload in summary["groups"].items():
            for metric_name in ["singular_values", "normalized_singular_values", "cumulative_energy"]:
                metric_payload = group_payload[metric_name]
                for rank_idx, (mean, p25, p50, p75) in enumerate(
                    zip(
                        metric_payload["mean"],
                        metric_payload["p25"],
                        metric_payload["p50"],
                        metric_payload["p75"],
                    ),
                    start=1,
                ):
                    writer.writerow(
                        {
                            "group": group_name,
                            "metric": metric_name,
                            "rank": rank_idx,
                            "mean": mean,
                            "p25": p25,
                            "p50": p50,
                            "p75": p75,
                            "count": group_payload["count"],
                        }
                    )


def plot_metric(summary: Dict, metric_name: str, output_path: Path, title: str, y_label: str, y_log_scale: bool = False) -> None:
    colors = {"latent": "#1f77b4", "cot": "#d62728"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for group_name, group_payload in summary["groups"].items():
        rank_axis = group_payload["rank_axis"]
        metric_payload = group_payload[metric_name]
        color = colors.get(group_name, None)

        ax.plot(rank_axis, metric_payload["mean"], color=color, linewidth=2.5, label=f"{group_name} mean")
        ax.fill_between(rank_axis, metric_payload["p25"], metric_payload["p75"], color=color, alpha=0.20)
        ax.plot(rank_axis, metric_payload["p50"], color=color, linewidth=1.5, linestyle="--", label=f"{group_name} median")

    ax.set_title(title)
    ax.set_xlabel("Rank")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    if y_log_scale:
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_individual_curves(samples: List[Dict], metric_name: str, output_path: Path, title: str, y_label: str, y_log_scale: bool = False) -> None:
    colors = {"latent": "#1f77b4", "cot": "#d62728"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for sample in samples:
        values = sample[metric_name]
        rank_axis = list(range(1, len(values) + 1))
        ax.plot(rank_axis, values, color=colors.get(sample["group"], "gray"), alpha=0.10, linewidth=1.0)

    ax.set_title(title)
    ax.set_xlabel("Rank")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", alpha=0.4)
    if y_log_scale:
        positive_min = min(
            (value for sample in samples for value in sample[metric_name] if value > 0),
            default=None,
        )
        if positive_min is not None and not math.isclose(positive_min, 0.0):
            ax.set_yscale("log")

    handles = [
        Line2D([0], [0], color=colors["latent"], linewidth=2, label="latent samples"),
        Line2D([0], [0], color=colors["cot"], linewidth=2, label="cot samples"),
    ]
    ax.legend(handles=handles)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_analysis() -> None:
    output_dir = ensure_output_dir(OUTPUT_DIR)

    latent_samples, latent_skipped = collect_group_samples("latent", LATENT_DATA_DIR, MAX_FILES, MAX_SAMPLES)
    cot_samples, cot_skipped = collect_group_samples("cot", COT_DATA_DIR, MAX_FILES, MAX_SAMPLES)

    all_samples = latent_samples + cot_samples
    all_skipped = latent_skipped + cot_skipped
    if not all_samples:
        raise RuntimeError("No valid samples were collected from either latent or cot inputs.")

    summary = build_summary(all_samples)
    samples_payload = {
        "config": summary["config"],
        "samples": all_samples,
        "skipped_samples": all_skipped,
    }

    write_json(output_dir / "svd_samples.json", samples_payload)
    write_json(output_dir / "svd_summary.json", summary)
    write_samples_csv(output_dir / "svd_samples.csv", all_samples)
    write_summary_csv(output_dir / "svd_summary.csv", summary)

    if PLOT_RAW_SINGULAR_VALUES:
        plot_individual_curves(
            all_samples,
            metric_name="singular_values",
            output_path=output_dir / "svd_individual_singular_values.png",
            title="Per-sample Singular Value Curves",
            y_label="Singular Value",
            y_log_scale=True,
        )
        plot_metric(
            summary,
            metric_name="singular_values",
            output_path=output_dir / "svd_summary_singular_values.png",
            title="Mean Singular Value Curves with Interquartile Bands",
            y_label="Singular Value",
            y_log_scale=True,
        )

    if PLOT_NORMALIZED_SINGULAR_VALUES:
        plot_individual_curves(
            all_samples,
            metric_name="normalized_singular_values",
            output_path=output_dir / "svd_individual_normalized_singular_values.png",
            title="Per-sample Normalized Singular Value Curves",
            y_label="Normalized Singular Value",
            y_log_scale=True,
        )
        plot_metric(
            summary,
            metric_name="normalized_singular_values",
            output_path=output_dir / "svd_summary_normalized_singular_values.png",
            title="Mean Normalized Singular Value Curves with Interquartile Bands",
            y_label="Normalized Singular Value",
            y_log_scale=True,
        )

    if PLOT_CUMULATIVE_ENERGY:
        plot_individual_curves(
            all_samples,
            metric_name="cumulative_energy",
            output_path=output_dir / "svd_individual_cumulative_energy.png",
            title="Per-sample Cumulative Energy Curves",
            y_label="Cumulative Energy",
            y_log_scale=False,
        )
        plot_metric(
            summary,
            metric_name="cumulative_energy",
            output_path=output_dir / "svd_summary_cumulative_energy.png",
            title="Mean Cumulative Energy Curves with Interquartile Bands",
            y_label="Cumulative Energy",
            y_log_scale=False,
        )

    logger.info(f"Finished SVD analysis. Outputs saved to {output_dir}")
    logger.info(f"Valid samples: latent={len(latent_samples)}, cot={len(cot_samples)}, skipped={len(all_skipped)}")


if __name__ == "__main__":
    run_analysis()
