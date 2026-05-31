from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import pyarrow.parquet as pq


DEFAULT_BASE_DIR = "data/external/Dolci-Think-SFT-32B_sampled"
DEFAULT_OUTPUT_DIR = "data/external/Dolci-Think-SFT-32B_sampled_200k"


def _get_parquet_files(root: Path) -> List[Path]:
    data_dir = root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards found in {data_dir}")
    return files


def _row_count(file_path: Path) -> int:
    return int(pq.ParquetFile(file_path).metadata.num_rows)


def _iter_ids(file_path: Path) -> Iterable[str]:
    df = pd.read_parquet(file_path, columns=["id"])
    if "id" not in df.columns:
        raise KeyError(f"Missing id column in {file_path}")
    for value in df["id"].astype(str).tolist():
        yield value


def _validate_unique_ids(named_files: Sequence[Tuple[str, Path]]) -> Dict[str, int]:
    seen: Dict[str, str] = {}
    duplicate_ids: Dict[str, int] = {}
    for alias, file_path in named_files:
        for sample_id in _iter_ids(file_path):
            previous = seen.get(sample_id)
            if previous is None:
                seen[sample_id] = alias
                continue
            duplicate_ids[sample_id] = duplicate_ids.get(sample_id, 1) + 1
    return {
        "unique_ids": len(seen),
        "duplicate_id_values": len(duplicate_ids),
        "duplicate_rows": sum(count - 1 for count in duplicate_ids.values()),
    }


def _safe_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"Destination already exists: {dst}")
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def assemble_sampled_dataset(
    *,
    base_dir: str,
    additional_dirs: Sequence[str],
    output_dir: str,
    mode: str,
    validate_ids: bool,
) -> Dict[str, object]:
    if not additional_dirs:
        raise ValueError("At least one additional_dir is required")

    base_root = Path(base_dir).resolve()
    output_root = Path(output_dir).resolve()
    output_data_dir = output_root / "data"

    if output_data_dir.exists() and any(output_data_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_data_dir}")
    output_data_dir.mkdir(parents=True, exist_ok=True)

    base_files = _get_parquet_files(base_root)
    additional_roots = [Path(path).resolve() for path in additional_dirs]
    additional_file_groups = [_get_parquet_files(root) for root in additional_roots]

    named_files: List[Tuple[str, Path]] = []
    for file_path in base_files:
        named_files.append((file_path.name, file_path))

    for group_idx, files in enumerate(additional_file_groups, start=1):
        for file_path in files:
            alias = f"inc{group_idx:02d}-{file_path.name}"
            named_files.append((alias, file_path))

    if len({alias for alias, _ in named_files}) != len(named_files):
        raise RuntimeError("Alias collision detected while assembling sampled dataset")

    id_summary = None
    if validate_ids:
        id_summary = _validate_unique_ids(named_files)
        if id_summary["duplicate_id_values"] != 0:
            raise RuntimeError(f"Found duplicate ids across assembled sampled data: {id_summary}")

    total_rows = 0
    file_records: List[Dict[str, object]] = []
    for alias, src in named_files:
        dst = output_data_dir / alias
        _safe_link_or_copy(src, dst, mode=mode)
        rows = _row_count(src)
        total_rows += rows
        file_records.append({
            "alias": alias,
            "source_path": str(src),
            "output_path": str(dst),
            "rows": rows,
        })

    readme_path = base_root / "README.md"
    if readme_path.exists():
        dst_readme = output_root / "README.md"
        if not dst_readme.exists():
            shutil.copy2(readme_path, dst_readme)

    manifest = {
        "base_dir": str(base_root),
        "additional_dirs": [str(path) for path in additional_roots],
        "output_dir": str(output_root),
        "mode": mode,
        "total_shards": len(file_records),
        "total_rows": total_rows,
        "validate_ids": validate_ids,
        "id_summary": id_summary,
        "files": file_records,
    }
    manifest_path = output_root / "assembly_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assemble an incremental sampled dataset for distillation resume")
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--additional_dir", action="append", dest="additional_dirs", default=[])
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--skip_validate_ids", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    manifest = assemble_sampled_dataset(
        base_dir=args.base_dir,
        additional_dirs=args.additional_dirs,
        output_dir=args.output_dir,
        mode=args.mode,
        validate_ids=not args.skip_validate_ids,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
