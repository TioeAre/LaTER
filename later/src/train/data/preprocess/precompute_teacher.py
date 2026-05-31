from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from later.src.train.dataset import (
    LatentSFTCollator,
    LatentSFTDataset,
    build_teacher_reference_spans,
    truncate_teacher_reference_ids,
)
from later.src.train.utils import ASSISTANT_PREFIX, find_subsequence, get_token_constants, load_yaml
from later.src.train.utils import PrecomputedTeacherCache
from later.src.utils.utils import (
    ensure_latent_think_special_tokens,
    get_registered_token_id,
    validate_latent_think_tokenizer_contract,
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bool_config(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _torch_dtype_from_config(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    text = str(value or "bfloat16").strip()
    if not hasattr(torch, text):
        raise ValueError(f"Unsupported torch dtype: {text}")
    resolved = getattr(torch, text)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"Resolved object is not a torch dtype: {text}")
    return resolved


def _numpy_prob_dtype_from_config(value: Any) -> np.dtype[Any]:
    text = str(value or "float16").strip().lower()
    mapping = {
        "float16": np.float16,
        "float32": np.float32,
    }
    if text not in mapping:
        raise ValueError(f"Unsupported precomputed_teacher_prob_dtype={text}. Supported values are: float16, float32.")
    return np.dtype(mapping[text])


def _load_processed_frame(train_data_path: str | Path) -> pd.DataFrame:
    path = Path(train_data_path)
    if not path.exists():
        raise FileNotFoundError(f"train_data not found: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                rows.append(json.loads(text))
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported train_data format: {path}")


def _resolve_runtime_values(config: Dict[str, Any]) -> Dict[str, Any]:
    teacher_model_path = str(config.get("teacher_model_path") or config["model_path"])
    train_data = str(config["train_data"])
    output_dir = str(config.get("precomputed_teacher_dir") or config.get("teacher_cache_dir") or "").strip()
    if not output_dir:
        raise ValueError("Config must set precomputed_teacher_dir or teacher_cache_dir for offline teacher cache.")

    topk = int(config.get("precomputed_teacher_topk", config.get("kl_topk_dim", 32)) or 32)
    batch_size = max(int(config.get("teacher_precompute_batch_size", 1) or 1), 1)
    num_workers = max(int(config.get("teacher_precompute_num_workers", config.get("num_workers", 0)) or 0), 0)
    rows_per_shard = max(int(config.get("teacher_precompute_rows_per_shard", 500000) or 500000), 1)
    teacher_max_length = int(config.get("teacher_max_length", config.get("max_length", 0)) or 0)
    if teacher_max_length <= 0:
        raise ValueError("teacher_max_length/max_length must be a positive integer.")

    trust_remote_code = _bool_config(config.get("trust_remote_code", True), default=True)
    teacher_attn_implementation = str(
        config.get("teacher_attn_implementation", config.get("attn_implementation", "sdpa"))
    ).strip()
    validate_metadata = _bool_config(config.get("precomputed_teacher_validate_metadata", True), default=True)
    prob_dtype = _numpy_prob_dtype_from_config(config.get("precomputed_teacher_prob_dtype", "float16"))
    teacher_torch_dtype = _torch_dtype_from_config(config.get("torch_dtype", "bfloat16"))
    projection_chunk_size = max(int(config.get("supervised_logits_chunk_size", 64) or 64), 1)
    max_train_samples = int(config.get("max_train_samples", 0) or 0)
    kl_temperature = float(config.get("kl_temperature", 1.0))

    return {
        "teacher_model_path": teacher_model_path,
        "train_data": train_data,
        "output_dir": output_dir,
        "topk": topk,
        "teacher_max_length": teacher_max_length,
        "teacher_precompute_batch_size": batch_size,
        "teacher_precompute_num_workers": num_workers,
        "teacher_precompute_rows_per_shard": rows_per_shard,
        "teacher_attn_implementation": teacher_attn_implementation,
        "validate_metadata": validate_metadata,
        "prob_dtype": prob_dtype,
        "teacher_torch_dtype": teacher_torch_dtype,
        "trust_remote_code": trust_remote_code,
        "projection_chunk_size": projection_chunk_size,
        "max_train_samples": max_train_samples,
        "kl_temperature": kl_temperature,
    }


def _init_distributed() -> Dict[str, Any]:
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    rank = int(dist.get_rank()) if dist.is_available() and dist.is_initialized() else env_rank
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank % max(torch.cuda.device_count(), 1))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if env_world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        init_kwargs: Dict[str, Any] = {}
        if device.type == "cuda":
            init_kwargs["device_id"] = device
        try:
            dist.init_process_group(backend=backend, **init_kwargs)
        except TypeError:
            dist.init_process_group(backend=backend)

    world_size = int(dist.get_world_size()) if dist.is_available() and dist.is_initialized() else env_world_size
    rank = int(dist.get_rank()) if dist.is_available() and dist.is_initialized() else env_rank

    return {
        "rank": int(rank),
        "world_size": int(world_size),
        "local_rank": int(local_rank),
        "device": device,
    }


def _destroy_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        try:
            if torch.cuda.is_available():
                dist.barrier(device_ids=[torch.cuda.current_device()])
            else:
                dist.barrier()
        except TypeError:
            dist.barrier()
        dist.destroy_process_group()


def _load_teacher_tokenizer(config: Dict[str, Any], runtime: Dict[str, Any]) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        str(config["model_path"]),
        trust_remote_code=bool(runtime["trust_remote_code"]),
        use_fast=True,
    )
    ensure_latent_think_special_tokens(tokenizer)
    validate_latent_think_tokenizer_contract(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "<|endoftext|>"
    return tokenizer


def _load_teacher_model(runtime: Dict[str, Any]) -> Any:
    load_kwargs: Dict[str, Any] = {
        "torch_dtype": runtime["teacher_torch_dtype"],
        "trust_remote_code": bool(runtime["trust_remote_code"]),
    }
    attn_impl = str(runtime["teacher_attn_implementation"]).strip()
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl

    try:
        model = AutoModelForCausalLM.from_pretrained(str(runtime["teacher_model_path"]), **load_kwargs)
        effective_attn = attn_impl or "default"
    except Exception:
        if attn_impl == "flash_attention_2":
            fallback_kwargs = dict(load_kwargs)
            fallback_kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(str(runtime["teacher_model_path"]), **fallback_kwargs)
            effective_attn = "sdpa"
        else:
            raise
    model.eval()
    return model, effective_attn


def _build_dataset_and_collator(
    config: Dict[str, Any], tokenizer: Any, runtime: Dict[str, Any]
) -> tuple[LatentSFTDataset, LatentSFTCollator]:
    frame = _load_processed_frame(runtime["train_data"])
    max_train_samples = int(runtime["max_train_samples"])
    if max_train_samples > 0:
        frame = frame.iloc[:max_train_samples].reset_index(drop=True)

    dataset = LatentSFTDataset(
        frame=frame,
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
        seed=int(config["seed"]),
        lazy=False,
        cot_ce_loss_weight=float(config.get("cot_ce_loss_weight", 0.3)),
        latent_start_ce_loss_weight=float(config.get("latent_start_ce_loss_weight", 1.0)),
        latent_end_ce_loss_weight=float(config.get("latent_end_ce_loss_weight", 1.0)),
    )
    collator = LatentSFTCollator(tokenizer=tokenizer, config=config)
    return dataset, collator


@dataclass
class TeacherLayout:
    assistant_prefix_start: int
    assistant_content_start: int
    im_end: int
    row_count: int
    source_start: int
    source_end: int


def _build_teacher_layout(teacher_ids: Sequence[int], tokenizer: Any, token_constants: Dict[str, int]) -> TeacherLayout:
    teacher_spans = build_teacher_reference_spans(teacher_ids, tokenizer, token_constants)
    row_count = int(teacher_spans.im_end - teacher_spans.assistant_content_start)
    if row_count <= 0:
        raise ValueError(
            f"Teacher assistant content must contain at least one visible token. "
            f"assistant_content_start={teacher_spans.assistant_content_start}, im_end={teacher_spans.im_end}"
        )
    source_start = int(teacher_spans.assistant_content_start - 1)
    source_end = int(teacher_spans.im_end - 1)
    if source_end <= source_start or (source_end - source_start) != row_count:
        raise ValueError(
            "Invalid teacher source slice for KL cache. "
            f"source_start={source_start}, source_end={source_end}, row_count={row_count}"
        )
    return TeacherLayout(
        assistant_prefix_start=int(teacher_spans.assistant_prefix_start),
        assistant_content_start=int(teacher_spans.assistant_content_start),
        im_end=int(teacher_spans.im_end),
        row_count=int(row_count),
        source_start=int(source_start),
        source_end=int(source_end),
    )


def _batched(iterable: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(iterable), batch_size):
        yield iterable[start : start + batch_size]


def _resolve_base_model(model: Any) -> Any:
    prefix = getattr(model, "base_model_prefix", "model")
    candidate = getattr(model, prefix, None)
    if candidate is None:
        candidate = getattr(model, "model", None)
    if candidate is None:
        raise ValueError("Teacher model does not expose a callable base model for hidden-state extraction.")
    return candidate


def _compute_sparse_topk_from_hidden(
    hidden_states: torch.Tensor,
    lm_head: Any,
    topk: int,
    temperature: float,
    projection_chunk_size: int,
    prob_dtype: np.dtype[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rows = int(hidden_states.size(0))
    if rows <= 0:
        return (
            np.zeros((0, topk), dtype=np.int32),
            np.zeros((0, topk), dtype=prob_dtype),
            np.zeros((0,), dtype=prob_dtype),
            0.0,
        )

    ids_chunks: List[np.ndarray] = []
    probs_chunks: List[np.ndarray] = []
    tail_chunks: List[np.ndarray] = []
    captured_mass_sum = 0.0
    chunk_size = max(int(projection_chunk_size), 1)

    for start in range(0, rows, chunk_size):
        end = min(start + chunk_size, rows)
        logits = lm_head(hidden_states[start:end]).float() / float(max(temperature, 1.0e-8))
        vocab_size = int(logits.size(-1))
        keep_topk = min(int(topk), vocab_size)
        topk_logits, topk_ids = torch.topk(logits, k=keep_topk, dim=-1)
        log_denom = torch.logsumexp(logits, dim=-1, keepdim=True)
        topk_probs = (topk_logits - log_denom).exp()
        tail = torch.clamp(1.0 - topk_probs.sum(dim=-1), min=0.0)
        captured_mass_sum += float(topk_probs.sum().detach().item())

        ids_np = topk_ids.detach().cpu().to(torch.int32).numpy()
        probs_np = topk_probs.detach().cpu().numpy().astype(prob_dtype, copy=False)
        tail_np = tail.detach().cpu().numpy().astype(prob_dtype, copy=False)

        if keep_topk < int(topk):
            ids_pad = np.zeros((ids_np.shape[0], int(topk)), dtype=np.int32)
            probs_pad = np.zeros((probs_np.shape[0], int(topk)), dtype=prob_dtype)
            ids_pad[:, :keep_topk] = ids_np
            probs_pad[:, :keep_topk] = probs_np
            ids_np = ids_pad
            probs_np = probs_pad

        ids_chunks.append(ids_np)
        probs_chunks.append(probs_np)
        tail_chunks.append(tail_np)

    return (
        np.concatenate(ids_chunks, axis=0),
        np.concatenate(probs_chunks, axis=0),
        np.concatenate(tail_chunks, axis=0),
        float(captured_mass_sum),
    )


class RankShardWriter:
    def __init__(self, root: Path, rank: int, topk: int, prob_dtype: np.dtype[Any], rows_per_shard: int) -> None:
        self.root = root
        self.rank = int(rank)
        self.topk = int(topk)
        self.prob_dtype = prob_dtype
        self.rows_per_shard = int(rows_per_shard)

        self.entries: Dict[str, Dict[str, int]] = {}
        self.current_ids: List[np.ndarray] = []
        self.current_probs: List[np.ndarray] = []
        self.current_tail: List[np.ndarray] = []
        self.current_rows = 0
        self.local_shard_index = 0
        self.total_rows = 0
        self.captured_mass_sum = 0.0

    def _flush_current_shard(self) -> None:
        if self.current_rows <= 0:
            return
        shard_path = self.root / f"rank_{self.rank:05d}_shard_{self.local_shard_index:05d}"
        ids = np.concatenate(self.current_ids, axis=0).astype(np.int32, copy=False)
        probs = np.concatenate(self.current_probs, axis=0).astype(self.prob_dtype, copy=False)
        tail = np.concatenate(self.current_tail, axis=0).astype(self.prob_dtype, copy=False)
        np.save(str(shard_path) + "_ids.npy", ids)
        np.save(str(shard_path) + "_probs.npy", probs)
        np.save(str(shard_path) + "_tail.npy", tail)
        self.local_shard_index += 1
        self.current_ids = []
        self.current_probs = []
        self.current_tail = []
        self.current_rows = 0

    def add_record(
        self,
        record_id: str,
        ids: np.ndarray,
        probs: np.ndarray,
        tail: np.ndarray,
        captured_mass_sum: float,
    ) -> None:
        row_count = int(ids.shape[0])
        if row_count != int(probs.shape[0]) or row_count != int(tail.shape[0]):
            raise ValueError(
                f"Inconsistent shard row counts for record_id={record_id}: "
                f"ids={ids.shape}, probs={probs.shape}, tail={tail.shape}"
            )
        if row_count <= 0:
            raise ValueError(f"Refusing to write empty teacher cache entry for record_id={record_id}")
        if self.current_rows > 0 and (self.current_rows + row_count) > self.rows_per_shard:
            self._flush_current_shard()

        shard_id = self.rank * 100000 + self.local_shard_index
        row_start = int(self.current_rows)
        self.entries[str(record_id)] = {
            "shard_id": int(shard_id),
            "row_start": int(row_start),
            "row_count": int(row_count),
        }
        self.current_ids.append(ids.astype(np.int32, copy=False))
        self.current_probs.append(probs.astype(self.prob_dtype, copy=False))
        self.current_tail.append(tail.astype(self.prob_dtype, copy=False))
        self.current_rows += row_count
        self.total_rows += row_count
        self.captured_mass_sum += float(captured_mass_sum)

    def finalize(self) -> None:
        self._flush_current_shard()
        payload = {
            "entries": self.entries,
            "rank": int(self.rank),
            "topk": int(self.topk),
            "probs_dtype": str(self.prob_dtype),
            "captured_mass_sum": float(self.captured_mass_sum),
            "row_count": int(self.total_rows),
            "num_shards": int(self.local_shard_index),
        }
        torch.save(payload, self.root / f"rank_{self.rank:05d}_index.pt")


def _merge_rank_indexes(
    root: Path,
    dataset: LatentSFTDataset,
    world_size: int,
    metadata: Dict[str, Any],
) -> None:
    merged_entries: Dict[str, Dict[str, int]] = {}
    total_rows = 0
    total_captured_mass = 0.0

    for rank in range(int(world_size)):
        rank_index_path = root / f"rank_{rank:05d}_index.pt"
        if not rank_index_path.exists():
            raise FileNotFoundError(f"Missing per-rank index file: {rank_index_path}")
        rank_index = torch.load(rank_index_path, map_location="cpu")
        entries = rank_index.get("entries", {})
        merged_entries.update(entries)
        total_rows += int(rank_index.get("row_count", 0))
        total_captured_mass += float(rank_index.get("captured_mass_sum", 0.0))

    ordered_record_ids = [str(sample["record_id"]) for sample in dataset.samples]
    missing = [record_id for record_id in ordered_record_ids if record_id not in merged_entries]
    if missing:
        raise ValueError(f"Merged teacher cache index is missing record_ids, first few={missing[:8]}")

    shard_ids = torch.tensor(
        [int(merged_entries[record_id]["shard_id"]) for record_id in ordered_record_ids],
        dtype=torch.int32,
    )
    row_starts = torch.tensor(
        [int(merged_entries[record_id]["row_start"]) for record_id in ordered_record_ids],
        dtype=torch.int64,
    )
    row_counts = torch.tensor(
        [int(merged_entries[record_id]["row_count"]) for record_id in ordered_record_ids],
        dtype=torch.int64,
    )
    torch.save(
        {
            "record_ids": ordered_record_ids,
            "shard_ids": shard_ids,
            "row_starts": row_starts,
            "row_counts": row_counts,
        },
        root / "index.pt",
    )

    metadata["num_examples"] = int(len(ordered_record_ids))
    metadata["dataset_size"] = int(len(ordered_record_ids))
    metadata["total_sparse_rows"] = int(total_rows)
    metadata["mean_topk_captured_mass"] = float(total_captured_mass / max(total_rows, 1))
    with (root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _validate_first_sample_alignment(dataset: LatentSFTDataset, collator: LatentSFTCollator, tokenizer: Any) -> None:
    if len(dataset) <= 0:
        raise ValueError("Teacher precompute dataset is empty.")
    sample = dataset[0]
    batch = collator([sample])
    pair_count = int(batch["teacher_kl_pair_mask"][0].sum().item())
    layout = _build_teacher_layout(sample["teacher_ids"], tokenizer=tokenizer, token_constants=dataset.token_constants)
    if pair_count != int(layout.row_count):
        raise ValueError(
            "Teacher cache row-count contract mismatch on first sample. "
            f"record_id={sample['record_id']}, pair_count={pair_count}, row_count={layout.row_count}"
        )


def _validate_written_cache(root: Path, dataset: LatentSFTDataset) -> None:
    cache = PrecomputedTeacherCache(root)
    sample = dataset[0]
    ids, probs, tail = cache.get(str(sample["record_id"]))
    if int(ids.shape[0]) <= 0:
        raise ValueError("Written teacher cache entry is empty for the first sample.")
    if int(ids.shape[0]) != int(probs.shape[0]) or int(ids.shape[0]) != int(tail.shape[0]):
        raise ValueError("Written teacher cache arrays have inconsistent shapes for the first sample.")


def _prepare_signature(
    config: Dict[str, Any],
    runtime: Dict[str, Any],
    tokenizer: Any,
    effective_teacher_attn_implementation: str,
) -> Dict[str, Any]:
    latent_start_id = get_registered_token_id(tokenizer, "<latent_think>")
    latent_end_id = get_registered_token_id(tokenizer, "</latent_think>")
    return {
        "model_path": str(config["model_path"]),
        "teacher_model_path": str(runtime["teacher_model_path"]),
        "train_data": str(runtime["train_data"]),
        "teacher_max_length": int(runtime["teacher_max_length"]),
        "teacher_attn_implementation": str(effective_teacher_attn_implementation),
        "kl_temperature": float(runtime["kl_temperature"]),
        "trust_remote_code": bool(runtime["trust_remote_code"]),
        "teacher_source_field": "state_align_reference_messages",
        "latent_start_token": "<latent_think>",
        "latent_start_token_id": int(latent_start_id) if latent_start_id is not None else None,
        "latent_end_token": "</latent_think>",
        "latent_end_token_id": int(latent_end_id) if latent_end_id is not None else None,
    }


def _print_rank0(message: str, rank: int) -> None:
    if int(rank) == 0:
        print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline precompute for teacher KL cache")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    runtime = _resolve_runtime_values(config)
    dist_info = _init_distributed()
    rank = int(dist_info["rank"])
    world_size = int(dist_info["world_size"])
    device = dist_info["device"]

    output_dir = Path(runtime["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer = _load_teacher_tokenizer(config=config, runtime=runtime)
        token_constants = get_token_constants(tokenizer)
        dataset, collator = _build_dataset_and_collator(config=config, tokenizer=tokenizer, runtime=runtime)
        _validate_first_sample_alignment(dataset=dataset, collator=collator, tokenizer=tokenizer)

        teacher_model, effective_teacher_attn_implementation = _load_teacher_model(runtime=runtime)
        teacher_model.to(device)
        base_model = _resolve_base_model(teacher_model)
        lm_head = teacher_model.get_output_embeddings()
        if lm_head is None:
            raise ValueError("Teacher model does not expose output embeddings / lm_head.")

        writer = RankShardWriter(
            root=output_dir,
            rank=rank,
            topk=int(runtime["topk"]),
            prob_dtype=runtime["prob_dtype"],
            rows_per_shard=int(runtime["teacher_precompute_rows_per_shard"]),
        )
        truncated_example_count = 0
        truncated_prompt_token_count = 0

        assigned_indices = list(range(rank, len(dataset), world_size))
        _print_rank0(
            (
                f"[teacher_precompute] starting with {len(dataset)} examples, world_size={world_size}, "
                f"topk={runtime['topk']}, batch_size={runtime['teacher_precompute_batch_size']}, "
                f"num_workers={runtime['teacher_precompute_num_workers']}, output_dir={output_dir}"
            ),
            rank=rank,
        )

        with torch.no_grad():
            for batch_indices in _batched(assigned_indices, int(runtime["teacher_precompute_batch_size"])):
                samples = [dataset[int(index)] for index in batch_indices]
                teacher_ids_list: List[List[int]] = []
                layouts: List[TeacherLayout] = []
                for sample in samples:
                    raw_teacher_ids = list(sample["teacher_ids"])
                    raw_spans = build_teacher_reference_spans(raw_teacher_ids, tokenizer, token_constants)
                    teacher_ids = truncate_teacher_reference_ids(
                        token_ids=raw_teacher_ids,
                        spans=raw_spans,
                        max_length=int(runtime["teacher_max_length"]),
                    )
                    dropped_prompt_tokens = max(len(raw_teacher_ids) - len(teacher_ids), 0)
                    if dropped_prompt_tokens > 0:
                        truncated_example_count += 1
                        truncated_prompt_token_count += int(dropped_prompt_tokens)
                    teacher_ids_list.append(teacher_ids)
                    layouts.append(
                        _build_teacher_layout(
                            teacher_ids=teacher_ids,
                            tokenizer=tokenizer,
                            token_constants=token_constants,
                        )
                    )

                max_teacher_len = max(len(ids) for ids in teacher_ids_list)
                batch_size = len(samples)
                input_ids = torch.full(
                    (batch_size, max_teacher_len),
                    int(tokenizer.pad_token_id),
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = torch.zeros((batch_size, max_teacher_len), dtype=torch.long, device=device)
                for row_idx, teacher_ids in enumerate(teacher_ids_list):
                    seq_len = len(teacher_ids)
                    input_ids[row_idx, :seq_len] = torch.tensor(teacher_ids, dtype=torch.long, device=device)
                    attention_mask[row_idx, :seq_len] = 1

                outputs = base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                hidden_states = outputs.last_hidden_state

                for row_idx, sample in enumerate(samples):
                    layout = layouts[row_idx]
                    source_hidden = hidden_states[row_idx, layout.source_start : layout.source_end, :]
                    row_count = int(source_hidden.size(0))
                    if row_count != int(layout.row_count):
                        raise ValueError(
                            "Hidden-state teacher row count mismatch. "
                            f"record_id={sample['record_id']}, source_hidden_rows={row_count}, "
                            f"expected_rows={layout.row_count}"
                        )
                    ids_np, probs_np, tail_np, captured_mass_sum = _compute_sparse_topk_from_hidden(
                        hidden_states=source_hidden,
                        lm_head=lm_head,
                        topk=int(runtime["topk"]),
                        temperature=float(runtime["kl_temperature"]),
                        projection_chunk_size=int(runtime["projection_chunk_size"]),
                        prob_dtype=runtime["prob_dtype"],
                    )
                    if int(ids_np.shape[0]) != int(layout.row_count):
                        raise ValueError(
                            "Offline teacher cache row_count mismatch after top-k extraction. "
                            f"record_id={sample['record_id']}, cache_rows={ids_np.shape[0]}, "
                            f"expected_rows={layout.row_count}"
                        )
                    writer.add_record(
                        record_id=str(sample["record_id"]),
                        ids=ids_np,
                        probs=probs_np,
                        tail=tail_np,
                        captured_mass_sum=float(captured_mass_sum),
                    )

        writer.finalize()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if rank == 0:
            metadata = {
                "version": 1,
                "created_at_utc": _utcnow_iso(),
                "topk": int(runtime["topk"]),
                "rows_per_shard": int(runtime["teacher_precompute_rows_per_shard"]),
                "probs_dtype": str(runtime["prob_dtype"]),
                "teacher_dtype": str(next(teacher_model.parameters()).dtype).replace("torch.", ""),
                "world_size": int(world_size),
                "source_config_path": str(Path(args.config).resolve()),
                "teacher_source_field": "state_align_reference_messages",
                "truncated_prompt_examples": int(truncated_example_count),
                "truncated_prompt_tokens": int(truncated_prompt_token_count),
                "signature": _prepare_signature(
                    config=config,
                    runtime=runtime,
                    tokenizer=tokenizer,
                    effective_teacher_attn_implementation=effective_teacher_attn_implementation,
                ),
            }
            _merge_rank_indexes(
                root=output_dir,
                dataset=dataset,
                world_size=world_size,
                metadata=metadata,
            )
            if bool(runtime["validate_metadata"]):
                _validate_written_cache(root=output_dir, dataset=dataset)
            _print_rank0("[teacher_precompute] complete", rank=rank)
    finally:
        _destroy_distributed()


if __name__ == "__main__":
    main()
