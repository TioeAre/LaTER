from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import Dataset

from later.src.train.dataset import load_sft_split
from later.src.train.utils import (
    ASSISTANT_PREFIX,
    IM_END,
    USER_PREFIX,
    build_position_ids,
    find_subsequence,
    render_manual_chat,
    render_prompt_only,
    resolve_context_parallel_padding_multiple,
)


@dataclass
class NoLatentSampleSpans:
    assistant_prefix_start: int
    assistant_content_start: int
    think_start: int
    think_end: int
    answer_start: int
    im_end: int


def get_no_latent_token_constants(tokenizer: Any) -> Dict[str, int]:
    required = {
        "think_start_id": tokenizer.convert_tokens_to_ids("<think>"),
        "think_end_id": tokenizer.convert_tokens_to_ids("</think>"),
        "im_start_id": tokenizer.convert_tokens_to_ids("<|im_start|>"),
        "im_end_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "eot_id": tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    }
    missing = [name for name, token_id in required.items() if token_id is None or int(token_id) < 0]
    if missing:
        raise ValueError(f"Tokenizer is missing required no-latent special tokens: {missing}")

    think_start_id = int(required["think_start_id"])
    think_end_id = int(required["think_end_id"])
    if list(tokenizer.encode("<think>", add_special_tokens=False)) != [think_start_id]:
        raise ValueError("Tokenizer must encode <think> as one dedicated token for no-latent SFT")
    if list(tokenizer.encode("</think>", add_special_tokens=False)) != [think_end_id]:
        raise ValueError("Tokenizer must encode </think> as one dedicated token for no-latent SFT")
    return {name: int(value) for name, value in required.items()}


def build_no_latent_spans(
    token_ids: Sequence[int],
    tokenizer: Any,
    token_constants: Dict[str, int],
) -> NoLatentSampleSpans:
    think_start_id = int(token_constants["think_start_id"])
    think_end_id = int(token_constants["think_end_id"])
    if list(token_ids).count(think_start_id) != 1:
        raise ValueError("Expected exactly one <think> token in no-latent sample")
    if list(token_ids).count(think_end_id) != 1:
        raise ValueError("Expected exactly one </think> token in no-latent sample")

    assistant_prefix_ids = tokenizer.encode(ASSISTANT_PREFIX, add_special_tokens=False)
    assistant_prefix_start = find_subsequence(token_ids, assistant_prefix_ids)
    if assistant_prefix_start < 0:
        raise ValueError("Assistant prefix not found in no-latent sample")
    assistant_content_start = assistant_prefix_start + len(assistant_prefix_ids)
    think_start = list(token_ids).index(think_start_id)
    think_end = list(token_ids).index(think_end_id)
    if not (assistant_content_start <= think_start < think_end):
        raise ValueError(
            "Invalid no-latent boundary order: "
            f"assistant_content_start={assistant_content_start}, think_start={think_start}, think_end={think_end}"
        )
    im_end = len(token_ids) - 1 - list(reversed(token_ids)).index(int(token_constants["im_end_id"]))
    answer_start = think_end + 1
    return NoLatentSampleSpans(
        assistant_prefix_start=assistant_prefix_start,
        assistant_content_start=assistant_content_start,
        think_start=think_start,
        think_end=think_end,
        answer_start=answer_start,
        im_end=im_end,
    )


def truncate_no_latent_ids(
    token_ids: List[int],
    spans: NoLatentSampleSpans,
    max_length: int,
) -> List[int]:
    if len(token_ids) <= max_length:
        return token_ids
    prompt_prefix = token_ids[: spans.assistant_prefix_start]
    assistant_head = token_ids[spans.assistant_prefix_start : spans.think_start + 1]
    assistant_tail = token_ids[spans.think_end :]
    mandatory = assistant_head + assistant_tail
    if len(mandatory) <= max_length:
        available_prefix = max_length - len(mandatory)
        return prompt_prefix[-available_prefix:] + mandatory
    if len(assistant_head) >= max_length:
        return assistant_head[: max_length]
    tail_budget = max_length - len(assistant_head)
    tail_start = max(spans.think_end, len(token_ids) - tail_budget)
    return assistant_head + token_ids[tail_start:]


class NoLatentSFTDataset(Dataset):
    def __init__(
        self,
        frame: Any,
        tokenizer: Any,
        max_length: int,
        seed: int,
        lazy: bool = False,
        cot_ce_loss_weight: float | None = None,
        im_end_ce_loss_weight: float = 1.0,
    ) -> None:
        del seed
        self.frame = frame
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.lazy = bool(lazy)
        self.cot_ce_loss_weight = cot_ce_loss_weight
        self.im_end_ce_loss_weight = float(im_end_ce_loss_weight)
        self.token_constants = get_no_latent_token_constants(tokenizer)
        self.skipped_rows = 0
        self.curriculum_sort_keys: List[List[int]] = []
        self.valid_indices: List[int] = list(range(len(frame)))
        self.samples: List[Dict[str, Any]] = []
        self._warned_string_reference = False
        if self.lazy:
            for i in range(len(frame)):
                try:
                    self.curriculum_sort_keys.append([int(v) for v in frame.iloc[i]["curriculum_sort_key"]])
                except Exception:
                    self.curriculum_sort_keys.append([0, 0, 0])
        else:
            self.valid_indices = []
            for i in range(len(frame)):
                try:
                    sample = self._process_row(frame.iloc[i].to_dict())
                    self.samples.append(sample)
                    self.valid_indices.append(i)
                    self.curriculum_sort_keys.append(list(sample["curriculum_sort_key"]))
                except Exception:
                    self.skipped_rows += 1

    def _encode_prompt_only_ids(self, user_content: str) -> List[int]:
        token_ids: List[int] = []
        token_ids.extend(self.tokenizer.encode(USER_PREFIX, add_special_tokens=False))
        token_ids.extend(self.tokenizer.encode(user_content, add_special_tokens=False))
        token_ids.extend(self.tokenizer.encode(IM_END, add_special_tokens=False))
        token_ids.extend(self.tokenizer.encode(ASSISTANT_PREFIX, add_special_tokens=False))
        return token_ids

    def _process_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        user_content = str(row["messages"][0]["content"])
        teacher_messages = row["state_align_reference_messages"]
        if not self._warned_string_reference:
            warnings.warn(
                "No-latent SFT uses state_align_reference_messages as teacher-forced supervision text.",
                stacklevel=2,
            )
            self._warned_string_reference = True
        teacher_text = render_manual_chat(teacher_messages)
        token_ids = self.tokenizer.encode(teacher_text, add_special_tokens=False)
        spans = build_no_latent_spans(token_ids, self.tokenizer, self.token_constants)
        if len(token_ids) > self.max_length:
            token_ids = truncate_no_latent_ids(token_ids, spans, self.max_length)
            spans = build_no_latent_spans(token_ids, self.tokenizer, self.token_constants)
        if spans.answer_start >= spans.im_end:
            raise ValueError("Truncated no-latent sample lost the answer region")

        prompt_only_ids = self._encode_prompt_only_ids(user_content)
        labels = list(token_ids)
        loss_weights = [0.0] * len(token_ids)
        prompt_mask = [False] * len(token_ids)
        cot_mask = [False] * len(token_ids)
        answer_mask = [False] * len(token_ids)
        teacher_kl_mask = [False] * len(token_ids)

        for idx in range(spans.assistant_content_start):
            labels[idx] = -100
            prompt_mask[idx] = True

        cot_branch_weight = (
            float(self.cot_ce_loss_weight)
            if self.cot_ce_loss_weight is not None
            else float(row["cot_loss_weight"])
        )
        for idx in range(spans.think_start, spans.answer_start):
            cot_mask[idx] = True
            loss_weights[idx] = 0.0
            if idx > 0:
                teacher_kl_mask[idx - 1] = True

        for idx in range(spans.answer_start, spans.im_end):
            answer_mask[idx] = True
            loss_weights[idx] = float(row["answer_loss_weight"])
            if idx > 0:
                teacher_kl_mask[idx - 1] = True

        loss_weights[spans.im_end] = float(self.im_end_ce_loss_weight)
        return {
            "record_id": str(row["record_id"]),
            "source_uid": str(row.get("source_uid", row["record_id"])),
            "text": teacher_text,
            "token_ids": token_ids,
            "prompt_only_ids": prompt_only_ids,
            "labels": labels,
            "loss_weights": loss_weights,
            "cot_branch_weight": cot_branch_weight,
            "prompt_mask": prompt_mask,
            "cot_mask": cot_mask,
            "answer_mask": answer_mask,
            "teacher_kl_mask": teacher_kl_mask,
            "spans": spans,
            "difficulty_rank": int(row["difficulty_rank"]),
            "curriculum_sort_key": [int(v) for v in row["curriculum_sort_key"]],
            "assistant_answer": str(row["assistant_answer"]),
        }

    def __len__(self) -> int:
        if self.lazy:
            return len(self.valid_indices)
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.lazy:
            sample = None
            max_tries = min(len(self.valid_indices), 32)
            for offset in range(max_tries):
                row_index = self.valid_indices[(index + offset) % len(self.valid_indices)]
                try:
                    sample = dict(self._process_row(self.frame.iloc[row_index].to_dict()))
                    break
                except Exception:
                    continue
            if sample is None:
                raise ValueError("Unable to build a valid no-latent sample after multiple lazy-dataset retries")
        else:
            sample = dict(self.samples[index])
        sample["index"] = index
        return sample

    def get_curriculum_sort_key(self, index: int) -> List[int]:
        return list(self.curriculum_sort_keys[index])


class NoLatentSFTCollator:
    def __init__(self, tokenizer: Any, config: Dict[str, Any]):
        self.tokenizer = tokenizer
        self.pad_token_id = int(tokenizer.pad_token_id)
        self.config = config

    def set_stage(self, train_stage: int, progress: float) -> None:
        del train_stage, progress
        return

    @staticmethod
    def _build_effective_token_mappings(
        valid_token_mask: Sequence[bool],
        labels: Sequence[int],
        teacher_kl_mask: Sequence[bool],
    ) -> tuple[List[int], List[int], List[int], List[int]]:
        valid_positions = [idx for idx, is_valid in enumerate(valid_token_mask) if is_valid]
        loss_source_positions: List[int] = []
        loss_target_positions: List[int] = []
        kl_source_positions: List[int] = []
        kl_target_positions: List[int] = []
        for pair_index in range(1, len(valid_positions)):
            source_pos = int(valid_positions[pair_index - 1])
            target_pos = int(valid_positions[pair_index])
            if int(labels[target_pos]) != -100:
                loss_source_positions.append(source_pos)
                loss_target_positions.append(target_pos)
            if bool(teacher_kl_mask[source_pos]):
                kl_source_positions.append(source_pos)
                kl_target_positions.append(target_pos)
        return loss_source_positions, loss_target_positions, kl_source_positions, kl_target_positions

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        batch_size = len(batch)
        cp_size = int(self.config.get("context_parallel_size", 1) or 1)
        if cp_size > 1 and batch_size != 1:
            raise ValueError(
                "No-latent context parallel training requires collator batch_size=1 "
                f"because all CP ranks cooperate on one sample, got batch_size={batch_size}."
            )
        max_seq_len = max((len(sample["token_ids"]) for sample in batch), default=0)
        cp_padding_multiple = int(resolve_context_parallel_padding_multiple(self.config))
        if cp_padding_multiple > 1 and max_seq_len > 0:
            remainder = int(max_seq_len % cp_padding_multiple)
            if remainder != 0:
                max_seq_len += int(cp_padding_multiple - remainder)
        input_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)
        loss_weights = torch.zeros((batch_size, max_seq_len), dtype=torch.float32)
        cot_branch_weight = torch.zeros((batch_size,), dtype=torch.float32)
        prompt_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
        cot_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
        answer_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
        teacher_kl_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
        valid_token_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
        position_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        think_start_positions = torch.zeros((batch_size,), dtype=torch.long)
        think_end_positions = torch.zeros((batch_size,), dtype=torch.long)
        max_loss_pairs = 0
        max_kl_pairs = 0
        prepared: List[Dict[str, Any]] = []

        for sample in batch:
            seq_len = len(sample["token_ids"])
            valid_mask = [True] * seq_len
            (
                loss_source_positions,
                loss_target_positions,
                kl_source_positions,
                kl_target_positions,
            ) = self._build_effective_token_mappings(
                valid_token_mask=valid_mask,
                labels=sample["labels"],
                teacher_kl_mask=sample["teacher_kl_mask"],
            )
            max_loss_pairs = max(max_loss_pairs, len(loss_target_positions))
            max_kl_pairs = max(max_kl_pairs, len(kl_source_positions))
            prepared.append(
                {
                    "record_id": sample["record_id"],
                    "prompt_only_ids": sample["prompt_only_ids"],
                    "assistant_answer": sample["assistant_answer"],
                    "spans": {
                        "assistant_prefix_start": int(sample["spans"].assistant_prefix_start),
                        "assistant_content_start": int(sample["spans"].assistant_content_start),
                        "think_start": int(sample["spans"].think_start),
                        "think_end": int(sample["spans"].think_end),
                        "answer_start": int(sample["spans"].answer_start),
                        "im_end": int(sample["spans"].im_end),
                    },
                    "token_ids": sample["token_ids"],
                    "labels": sample["labels"],
                    "loss_weights": sample["loss_weights"],
                    "cot_branch_weight": float(sample["cot_branch_weight"]),
                    "prompt_mask": sample["prompt_mask"],
                    "cot_mask": sample["cot_mask"],
                    "answer_mask": sample["answer_mask"],
                    "teacher_kl_mask": sample["teacher_kl_mask"],
                    "loss_source_positions": loss_source_positions,
                    "loss_target_positions": loss_target_positions,
                    "kl_source_positions": kl_source_positions,
                    "kl_target_positions": kl_target_positions,
                }
            )

        loss_source_positions = torch.full((batch_size, max_loss_pairs), -1, dtype=torch.long)
        loss_target_positions = torch.full((batch_size, max_loss_pairs), -1, dtype=torch.long)
        loss_pair_mask = torch.zeros((batch_size, max_loss_pairs), dtype=torch.bool)
        teacher_kl_source_positions = torch.full((batch_size, max_kl_pairs), -1, dtype=torch.long)
        teacher_kl_target_positions = torch.full((batch_size, max_kl_pairs), -1, dtype=torch.long)
        teacher_kl_pair_mask = torch.zeros((batch_size, max_kl_pairs), dtype=torch.bool)

        for row_idx, sample in enumerate(prepared):
            seq_len = len(sample["token_ids"])
            input_ids[row_idx, :seq_len] = torch.tensor(sample["token_ids"], dtype=torch.long)
            labels[row_idx, :seq_len] = torch.tensor(sample["labels"], dtype=torch.long)
            loss_weights[row_idx, :seq_len] = torch.tensor(sample["loss_weights"], dtype=torch.float32)
            cot_branch_weight[row_idx] = float(sample["cot_branch_weight"])
            prompt_mask[row_idx, :seq_len] = torch.tensor(sample["prompt_mask"], dtype=torch.bool)
            cot_mask[row_idx, :seq_len] = torch.tensor(sample["cot_mask"], dtype=torch.bool)
            answer_mask[row_idx, :seq_len] = torch.tensor(sample["answer_mask"], dtype=torch.bool)
            teacher_kl_mask[row_idx, :seq_len] = torch.tensor(sample["teacher_kl_mask"], dtype=torch.bool)
            valid_token_mask[row_idx, :seq_len] = True
            attention_mask[row_idx, :seq_len] = 1
            position_ids[row_idx] = build_position_ids(valid_token_mask[row_idx])
            think_start_positions[row_idx] = int(sample["spans"]["think_start"])
            think_end_positions[row_idx] = int(sample["spans"]["think_end"])
            if sample["loss_target_positions"]:
                pair_len = len(sample["loss_target_positions"])
                loss_source_positions[row_idx, :pair_len] = torch.tensor(sample["loss_source_positions"], dtype=torch.long)
                loss_target_positions[row_idx, :pair_len] = torch.tensor(sample["loss_target_positions"], dtype=torch.long)
                loss_pair_mask[row_idx, :pair_len] = True
            if sample["kl_source_positions"]:
                pair_len = len(sample["kl_source_positions"])
                teacher_kl_source_positions[row_idx, :pair_len] = torch.tensor(
                    sample["kl_source_positions"], dtype=torch.long
                )
                teacher_kl_target_positions[row_idx, :pair_len] = torch.tensor(
                    sample["kl_target_positions"], dtype=torch.long
                )
                teacher_kl_pair_mask[row_idx, :pair_len] = True

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_weights": loss_weights,
            "cot_branch_weight": cot_branch_weight,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prompt_mask": prompt_mask,
            "cot_mask": cot_mask,
            "answer_mask": answer_mask,
            "teacher_kl_mask": teacher_kl_mask,
            "valid_token_mask": valid_token_mask,
            "think_start_positions": think_start_positions,
            "think_end_positions": think_end_positions,
            "record_ids": [sample["record_id"] for sample in prepared],
            "prompt_only_ids": [sample["prompt_only_ids"] for sample in prepared],
            "assistant_answers": [sample["assistant_answer"] for sample in prepared],
            "spans": [sample["spans"] for sample in prepared],
            "loss_source_positions": loss_source_positions,
            "loss_target_positions": loss_target_positions,
            "loss_pair_mask": loss_pair_mask,
            "teacher_kl_source_positions": teacher_kl_source_positions,
            "teacher_kl_target_positions": teacher_kl_target_positions,
            "teacher_kl_pair_mask": teacher_kl_pair_mask,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="No-latent dataset/collator smoke test")
    parser.add_argument("--config", default="later/src/config/sft_no_latent_config.yaml")
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from later.src.train.utils import load_yaml

    config = load_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "<|endoftext|>"
    train_df, _ = load_sft_split(config["train_data"], float(config["val_ratio"]))
    dataset = NoLatentSFTDataset(
        frame=train_df.iloc[: max(int(args.batch_size), 2)].reset_index(drop=True),
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
        seed=int(config["seed"]),
        lazy=False,
        cot_ce_loss_weight=float(config.get("cot_ce_loss_weight", 0.25)),
        im_end_ce_loss_weight=float(config.get("im_end_ce_loss_weight", 1.0)),
    )
    collator = NoLatentSFTCollator(tokenizer=tokenizer, config=config)
    batch = collator([dataset[i] for i in range(min(len(dataset), int(args.batch_size)))])
    print(
        {
            "batch_size": int(batch["input_ids"].size(0)),
            "seq_len": int(batch["input_ids"].size(1)),
            "loss_pair_slots": int(batch["loss_pair_mask"].size(1)),
            "teacher_kl_pair_slots": int(batch["teacher_kl_pair_mask"].size(1)),
        }
    )
