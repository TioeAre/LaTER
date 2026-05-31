"""Sample Dolci-Think-SFT-32B by source field and save with original parquet schema.

This script reads all parquet shards under:
  data/external/Dolci-Think-SFT-32B/data

It performs stratified sampling by the `source` column while preserving the
original source distribution. It supports excluding already sampled ids so we
can extend an existing 20k sample with a new 180k complement safely.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

import pandas as pd
from loguru import logger

from later.src.utils.utils import normalize_group_key


DEFAULT_INPUT_DIR = "data/external/Dolci-Think-SFT-32B"
DEFAULT_OUTPUT_DIR = "data/external/Dolci-Think-SFT-32B_sampled"

SampleRef = Tuple[str, int, str]


def _get_parquet_files(data_dir: Path) -> List[Path]:
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    return parquet_files


def _read_parquet_safe(file_path: Path, columns: List[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_parquet(file_path, columns=columns)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Skip corrupt parquet: {file_path} | {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def _normalize_source_key(value: Any) -> str:
    return normalize_group_key(value)


def _load_excluded_ids(exclude_dir: Optional[str]) -> Set[str]:
    if not exclude_dir:
        return set()

    exclude_data_dir = Path(exclude_dir) / "data"
    if not exclude_data_dir.exists():
        raise FileNotFoundError(f"Exclude sampled data directory not found: {exclude_data_dir}")

    excluded_ids: Set[str] = set()
    for file_path in sorted(exclude_data_dir.glob("*.parquet")):
        df_ids = _read_parquet_safe(file_path, columns=["id"])
        if df_ids.empty and "id" not in df_ids.columns:
            continue
        if "id" not in df_ids.columns:
            logger.error(f"Skip exclude file without id column: {file_path}")
            continue
        excluded_ids.update(df_ids["id"].astype(str).tolist())

    logger.info(f"Loaded {len(excluded_ids)} excluded ids from {exclude_data_dir}")
    return excluded_ids


def _filter_excluded_rows(df: pd.DataFrame, excluded_ids: Set[str]) -> pd.DataFrame:
    if not excluded_ids:
        return df
    if "id" not in df.columns:
        raise KeyError("exclude_dir requires an id column in every readable parquet shard")
    mask = ~df["id"].astype(str).isin(excluded_ids)
    return df.loc[mask].copy()


def _count_source_distribution(
    parquet_files: List[Path],
    excluded_ids: Optional[Set[str]] = None,
) -> Tuple[Dict[str, int], int, Set[Path]]:
    excluded_ids = excluded_ids or set()
    source_counts: Dict[str, int] = {}
    total_rows = 0
    bad_files: Set[Path] = set()

    for file_path in parquet_files:
        df = _read_parquet_safe(file_path)
        if df.empty:
            bad_files.add(file_path)
            continue
        if "source" not in df.columns:
            logger.error(f"Skip file without source column: {file_path}")
            bad_files.add(file_path)
            continue
        try:
            df = _filter_excluded_rows(df, excluded_ids)
        except KeyError:
            logger.error(f"Skip file without id column while exclude_dir is enabled: {file_path}")
            bad_files.add(file_path)
            continue
        if df.empty:
            continue

        keys = df["source"].map(_normalize_source_key)
        vc = keys.value_counts(dropna=False)
        for source_key, count in vc.items():
            source_counts[str(source_key)] = source_counts.get(str(source_key), 0) + int(count)
            total_rows += int(count)

    return source_counts, total_rows, bad_files


def _build_source_alloc(source_counts: Dict[str, int], target_size: int) -> Dict[str, int]:
    total_rows = sum(source_counts.values())
    if total_rows == 0:
        raise ValueError("No readable rows found from input parquet files")
    if target_size >= total_rows:
        return dict(source_counts)

    alloc: Dict[str, int] = {}
    remainders: Dict[str, float] = {}
    assigned = 0
    for source_name in sorted(source_counts.keys()):
        size = source_counts[source_name]
        exact = target_size * (size / total_rows)
        base = int(math.floor(exact))
        alloc[source_name] = min(base, size)
        remainders[source_name] = exact - base
        assigned += alloc[source_name]

    leftover = target_size - assigned
    if leftover > 0:
        for source_name, _ in sorted(remainders.items(), key=lambda item: item[1], reverse=True):
            if leftover <= 0:
                break
            capacity = source_counts[source_name] - alloc[source_name]
            if capacity <= 0:
                continue
            alloc[source_name] += 1
            leftover -= 1

    return alloc


def _sample_reference_rows_by_source(
    parquet_files: List[Path],
    alloc: Dict[str, int],
    seed: int,
    bad_files: Set[Path],
    excluded_ids: Optional[Set[str]] = None,
) -> List[SampleRef]:
    excluded_ids = excluded_ids or set()
    rng = random.Random(seed)
    seen_per_source: Dict[str, int] = {key: 0 for key in alloc.keys()}
    samples_per_source: Dict[str, List[Tuple[str, int]]] = {key: [] for key in alloc.keys()}

    for file_path in parquet_files:
        if file_path in bad_files:
            continue
        df = _read_parquet_safe(file_path)
        if df.empty:
            continue
        if "source" not in df.columns:
            logger.error(f"Skip file without source column in pass2: {file_path}")
            continue

        df = df.reset_index(drop=True)
        df["_raw_row_idx"] = df.index
        try:
            df = _filter_excluded_rows(df, excluded_ids)
        except KeyError:
            logger.error(f"Skip file without id column in pass2 while exclude_dir is enabled: {file_path}")
            continue
        if df.empty:
            continue

        df["_source_key"] = df["source"].map(_normalize_source_key)

        for source_key, group_df in df.groupby("_source_key", dropna=False, sort=False):
            source_key = str(source_key)
            k = alloc.get(source_key, 0)
            if k <= 0:
                continue
            reservoir = samples_per_source[source_key]
            seen = seen_per_source[source_key]
            for row_idx in group_df["_raw_row_idx"].tolist():
                seen += 1
                ref = (file_path.name, int(row_idx))
                if len(reservoir) < k:
                    reservoir.append(ref)
                else:
                    j = rng.randint(1, seen)
                    if j <= k:
                        reservoir[j - 1] = ref
            seen_per_source[source_key] = seen

    all_refs: List[SampleRef] = []
    for source_key in sorted(samples_per_source.keys()):
        for file_name, row_idx in samples_per_source[source_key]:
            all_refs.append((file_name, row_idx, source_key))
    rng.shuffle(all_refs)
    return all_refs


def _json_ready(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, list, dict)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # noqa: BLE001
            pass
    return value


def _write_sampled_shards_from_refs(
    parquet_files: List[Path],
    sampled_refs: List[SampleRef],
    output_root: Path,
    num_output_shards: int,
) -> int:
    output_data_dir = output_root / "data"
    output_data_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in output_data_dir.glob("*.parquet"):
        stale_file.unlink()

    temp_dir = output_root / "_tmp_shards_jsonl"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    file_map = {path.name: path for path in parquet_files}
    shard_size = max(math.ceil(len(sampled_refs) / max(num_output_shards, 1)), 1)
    requests_by_file: DefaultDict[str, List[Tuple[int, int, int]]] = defaultdict(list)
    for global_order, (file_name, row_idx, _) in enumerate(sampled_refs):
        shard_idx = min(global_order // shard_size, max(num_output_shards - 1, 0))
        requests_by_file[file_name].append((row_idx, shard_idx, global_order))

    for file_name, requests in requests_by_file.items():
        file_path = file_map[file_name]
        df = _read_parquet_safe(file_path)
        if df.empty:
            logger.error(f"Failed to materialize selected rows from {file_path}; output may be short")
            continue
        df = df.reset_index(drop=True)
        requests_by_shard: DefaultDict[int, List[Tuple[int, int]]] = defaultdict(list)
        for row_idx, shard_idx, order in requests:
            if 0 <= row_idx < len(df):
                requests_by_shard[shard_idx].append((row_idx, order))

        for shard_idx, items in requests_by_shard.items():
            shard_df = df.iloc[[row_idx for row_idx, _ in items]].copy()
            shard_df["__order"] = [order for _, order in items]
            shard_path = temp_dir / f"shard-{shard_idx:05d}.jsonl"
            with shard_path.open("a", encoding="utf-8") as handle:
                for record in shard_df.to_dict(orient="records"):
                    payload = {key: _json_ready(value) for key, value in record.items()}
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    written = 0
    for shard_idx in range(max(num_output_shards, 1)):
        temp_jsonl = temp_dir / f"shard-{shard_idx:05d}.jsonl"
        if not temp_jsonl.exists():
            continue
        rows: List[Dict[str, Any]] = []
        with temp_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        if not rows:
            continue
        rows.sort(key=lambda row: int(row.get("__order", 0)))
        for row in rows:
            row.pop("__order", None)
        shard_df = pd.DataFrame(rows)
        shard_name = f"train-{shard_idx:05d}-of-{max(num_output_shards, 1):05d}.parquet"
        shard_df.to_parquet(output_data_dir / shard_name, index=False)
        written += len(shard_df)

    shutil.rmtree(temp_dir, ignore_errors=True)
    return written


def sample_and_save_dolci_sft_dataset(
    input_dir: str = DEFAULT_INPUT_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    target_size: int = 20_000,
    seed: int = 42,
    num_output_shards: int = 4,
    exclude_dir: Optional[str] = None,
) -> None:
    input_data_dir = Path(input_dir) / "data"
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    parquet_files = _get_parquet_files(input_data_dir)
    excluded_ids = _load_excluded_ids(exclude_dir)

    source_counts, total_rows, bad_files = _count_source_distribution(parquet_files, excluded_ids=excluded_ids)
    alloc = _build_source_alloc(source_counts, target_size=target_size)
    sampled_refs = _sample_reference_rows_by_source(
        parquet_files=parquet_files,
        alloc=alloc,
        seed=seed,
        bad_files=bad_files,
        excluded_ids=excluded_ids,
    )
    if len(sampled_refs) > target_size:
        sampled_refs = sampled_refs[:target_size]

    input_readme = Path(input_dir) / "README.md"
    if input_readme.exists():
        readme_out = output_root / "README.md"
        if not readme_out.exists():
            readme_out.write_text(input_readme.read_text(encoding="utf-8"), encoding="utf-8")

    written = _write_sampled_shards_from_refs(
        parquet_files=parquet_files,
        sampled_refs=sampled_refs,
        output_root=output_root,
        num_output_shards=max(num_output_shards, 1),
    )

    stats = Counter(source_key for _, _, source_key in sampled_refs)
    stats_df = pd.DataFrame(
        [{"source_key": source_key, "count": count} for source_key, count in stats.items()]
    ).sort_values(by="count", ascending=False)
    stats_df.to_csv(output_root / "sampling_stats_by_source.csv", index=False)

    print(f"Readable rows after exclusion: {total_rows}")
    print(f"Excluded existing ids: {len(excluded_ids)}")
    print(f"Corrupt files skipped: {len(bad_files)}")
    print(f"Sampled rows: {len(sampled_refs)}")
    print(f"Written rows: {written}")
    print(f"Saved to: {output_root / 'data'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample Dolci-Think-SFT-32B by source")
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target_size", type=int, default=20_000, help="Target total sampled rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_output_shards", type=int, default=4)
    parser.add_argument("--exclude_dir", default=None, help="Optional existing sampled dataset root to exclude by id")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    sample_and_save_dolci_sft_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_size=args.target_size,
        seed=args.seed,
        num_output_shards=args.num_output_shards,
        exclude_dir=args.exclude_dir,
    )


if __name__ == "__main__":
    main()
