from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from later.src.train.utils import (
    ASSISTANT_PREFIX,
    IM_END,
    USER_PREFIX,
    build_position_ids,
    find_subsequence,
    get_token_constants,
    render_manual_chat,
    render_prompt_only,
    render_student_messages,
    resolve_context_parallel_padding_multiple,
)
from later.src.utils.utils import get_registered_token_id


@dataclass
class SampleSpans:
    assistant_prefix_start: int
    assistant_content_start: int
    latent_start: int
    latent_end: int
    think_start: int
    think_end: int
    answer_start: int
    im_end: int


@dataclass
class TeacherReferenceSpans:
    assistant_prefix_start: int
    assistant_content_start: int
    think_start: int
    think_end: int
    answer_start: int
    im_end: int


def load_sft_split(data_path: str, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(data_path)
    if "curriculum_index" in frame.columns:
        frame = frame.sort_values("curriculum_index").reset_index(drop=True)
    val_size = max(1, int(len(frame) * val_ratio))
    train_df = frame.iloc[:-val_size].reset_index(drop=True)
    val_df = frame.iloc[-val_size:].reset_index(drop=True)
    return train_df, val_df


def load_sft_frame(data_path: str) -> pd.DataFrame:
    frame = pd.read_parquet(data_path)
    if "curriculum_index" in frame.columns:
        frame = frame.sort_values("curriculum_index").reset_index(drop=True)
    return frame.reset_index(drop=True)


def build_spans(
    token_ids: Sequence[int],
    tokenizer: Any,
    token_constants: Dict[str, int],
) -> SampleSpans:
    latent_start_id = int(token_constants["latent_start_id"])
    latent_end_id = int(token_constants["latent_end_id"])
    think_start_id = int(token_constants["think_start_id"])
    think_end_id = int(token_constants["think_end_id"])
    if list(token_ids).count(latent_start_id) != 1:
        raise ValueError("Expected exactly one <latent_think> token in tokenized sample")
    if list(token_ids).count(latent_end_id) != 1:
        raise ValueError("Expected exactly one </latent_think> token in tokenized sample")
    if list(token_ids).count(think_start_id) != 1:
        raise ValueError("Expected exactly one <think> token in tokenized sample")
    if list(token_ids).count(think_end_id) != 1:
        raise ValueError("Expected exactly one </think> token in tokenized sample")

    assistant_prefix_ids = tokenizer.encode(ASSISTANT_PREFIX, add_special_tokens=False)
    assistant_prefix_start = find_subsequence(token_ids, assistant_prefix_ids)
    if assistant_prefix_start < 0:
        raise ValueError("Assistant prefix not found in tokenized sample")
    assistant_content_start = assistant_prefix_start + len(assistant_prefix_ids)
    latent_start = token_ids.index(latent_start_id)
    latent_end = token_ids.index(latent_end_id)
    think_start = token_ids.index(think_start_id)
    think_end = token_ids.index(think_end_id)
    if not (assistant_content_start <= latent_start < latent_end < think_start < think_end):
        raise ValueError(
            "Invalid assistant boundary order for latent/think tokens: "
            f"assistant_content_start={assistant_content_start}, latent_start={latent_start}, "
            f"latent_end={latent_end}, think_start={think_start}, think_end={think_end}"
        )
    im_end = len(token_ids) - 1 - list(reversed(token_ids)).index(token_constants["im_end_id"])
    answer_start = think_end + 1
    return SampleSpans(
        assistant_prefix_start=assistant_prefix_start,
        assistant_content_start=assistant_content_start,
        latent_start=latent_start,
        latent_end=latent_end,
        think_start=think_start,
        think_end=think_end,
        answer_start=answer_start,
        im_end=im_end,
    )

def build_teacher_reference_spans(
    token_ids: Sequence[int],
    tokenizer: Any,
    token_constants: Dict[str, int],
) -> TeacherReferenceSpans:
    think_start_id = int(token_constants["think_start_id"])
    think_end_id = int(token_constants["think_end_id"])
    if list(token_ids).count(think_start_id) != 1:
        raise ValueError("Expected exactly one <think> token in teacher reference")
    if list(token_ids).count(think_end_id) != 1:
        raise ValueError("Expected exactly one </think> token in teacher reference")

    assistant_prefix_ids = tokenizer.encode(ASSISTANT_PREFIX, add_special_tokens=False)
    assistant_prefix_start = find_subsequence(token_ids, assistant_prefix_ids)
    if assistant_prefix_start < 0:
        raise ValueError("Assistant prefix not found in teacher reference")
    assistant_content_start = assistant_prefix_start + len(assistant_prefix_ids)
    think_start = token_ids.index(think_start_id)
    think_end = token_ids.index(think_end_id)
    if not (assistant_content_start <= think_start < think_end):
        raise ValueError(
            "Invalid teacher reference boundary order: "
            f"assistant_content_start={assistant_content_start}, think_start={think_start}, think_end={think_end}"
        )
    im_end = len(token_ids) - 1 - list(reversed(token_ids)).index(token_constants["im_end_id"])
    answer_start = think_end + 1
    return TeacherReferenceSpans(
        assistant_prefix_start=assistant_prefix_start,
        assistant_content_start=assistant_content_start,
        think_start=think_start,
        think_end=think_end,
        answer_start=answer_start,
        im_end=im_end,
    )


def truncate_teacher_reference_ids(
    token_ids: List[int],
    spans: TeacherReferenceSpans,
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



class LatentSFTDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: Any,
        max_length: int,
        seed: int,
        lazy: bool = False,
        cot_ce_loss_weight: float | None = None,
        latent_start_ce_loss_weight: float = 1.0,
        latent_end_ce_loss_weight: float = 1.0,
        im_end_ce_loss_weight: float = 1.0,
    ) -> None:
        self.frame = frame
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed
        self.lazy = lazy
        self.cot_ce_loss_weight = cot_ce_loss_weight
        self.latent_start_ce_loss_weight = float(latent_start_ce_loss_weight)
        self.latent_end_ce_loss_weight = float(latent_end_ce_loss_weight)
        self.im_end_ce_loss_weight = float(im_end_ce_loss_weight)
        self.token_constants = get_token_constants(tokenizer)
        self.skipped_rows = 0
        self._warned_string_reencode_fallback = False
        self.curriculum_sort_keys: List[List[int]] = []
        self.valid_indices: List[int] = list(range(len(frame)))
        self.samples: List[Dict[str, Any]] = []
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

    def _encode_latent_placeholder_ids(self, row: Dict[str, Any]) -> List[int]:
        latent_pad_token = str(row.get("latent_pad_token", "<|endoftext|>"))
        latent_steps = max(int(row.get("n_latent_steps", 0) or 0), 0)
        if latent_steps <= 0:
            return []
        registered_id = get_registered_token_id(self.tokenizer, latent_pad_token)
        if registered_id is not None:
            return [int(registered_id)] * latent_steps
        return self.tokenizer.encode(latent_pad_token * latent_steps, add_special_tokens=False)

    @staticmethod
    def _has_structured_student_fields(row: Dict[str, Any]) -> bool:
        required_fields = ("assistant_cot", "assistant_answer", "n_latent_steps", "latent_pad_token")
        return all(field in row and row.get(field) is not None for field in required_fields)

    def _build_structured_student_ids(self, row: Dict[str, Any], user_content: str) -> List[int]:
        token_ids: List[int] = []
        token_ids.extend(self.tokenizer.encode(USER_PREFIX, add_special_tokens=False))
        token_ids.extend(self.tokenizer.encode(user_content, add_special_tokens=False))
        token_ids.extend(self.tokenizer.encode(IM_END, add_special_tokens=False))
        token_ids.extend(self.tokenizer.encode(ASSISTANT_PREFIX, add_special_tokens=False))
        token_ids.append(int(self.token_constants["latent_start_id"]))
        token_ids.extend(self._encode_latent_placeholder_ids(row))
        token_ids.append(int(self.token_constants["latent_end_id"]))
        token_ids.append(int(self.token_constants["think_start_id"]))
        token_ids.extend(self.tokenizer.encode(str(row["assistant_cot"]), add_special_tokens=False))
        token_ids.append(int(self.token_constants["think_end_id"]))
        token_ids.extend(self.tokenizer.encode(str(row["assistant_answer"]), add_special_tokens=False))
        token_ids.extend(self.tokenizer.encode(IM_END, add_special_tokens=False))
        return token_ids

    def _truncate(self, token_ids: List[int], spans: SampleSpans) -> List[int]:
        """超过 max_length 时截断"""
        if len(token_ids) <= self.max_length:
            return token_ids  # 直接返回
        prompt_prefix = token_ids[: spans.assistant_prefix_start]
        assistant_head = token_ids[spans.assistant_prefix_start : spans.think_start + 1]
        assistant_tail = token_ids[spans.think_end :]
        mandatory = assistant_head + assistant_tail
        # 一部分 prompt + assistant
        if len(mandatory) <= self.max_length:
            available_prefix = self.max_length - len(mandatory)
            return prompt_prefix[-available_prefix:] + mandatory
        # 保留 assistant + latent token
        if len(assistant_head) >= self.max_length:
            return assistant_head[: self.max_length]
        # 保留 <think> 外的答案的
        tail_budget = self.max_length - len(assistant_head)
        tail_start = max(spans.think_end, len(token_ids) - tail_budget)
        return assistant_head + token_ids[tail_start:]

    def _process_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        user_content = str(row["messages"][0]["content"])
        assistant_content = str(row["messages"][1]["content"])
        student_text = render_student_messages(user_content=user_content, assistant_content=assistant_content)
        teacher_text = render_manual_chat(row["state_align_reference_messages"])
        prompt_only_text = render_prompt_only(user_content)

        if self._has_structured_student_fields(row):
            student_ids = self._build_structured_student_ids(row=row, user_content=user_content)
        else:
            if not self._warned_string_reencode_fallback:
                warnings.warn(
                    "Falling back to string re-encoding for SFT sample construction because structured latent fields "
                    "are missing. This path is less robust for latent special-token boundaries.",
                    stacklevel=2,
                )
                self._warned_string_reencode_fallback = True
            student_ids = self.tokenizer.encode(student_text, add_special_tokens=False)
        spans = build_spans(student_ids, self.tokenizer, self.token_constants)
        if len(student_ids) > self.max_length:
            student_ids = self._truncate(student_ids, spans)
            spans = build_spans(student_ids, self.tokenizer, self.token_constants)
        if spans.answer_start >= spans.im_end:
            raise ValueError("Truncated sample lost the answer region")
        teacher_ids = self.tokenizer.encode(teacher_text, add_special_tokens=False)
        teacher_spans = build_teacher_reference_spans(teacher_ids, self.tokenizer, self.token_constants)
        if len(teacher_ids) > self.max_length:
            teacher_ids = truncate_teacher_reference_ids(teacher_ids, teacher_spans, self.max_length)
            teacher_spans = build_teacher_reference_spans(teacher_ids, self.tokenizer, self.token_constants)
        if teacher_spans.answer_start >= teacher_spans.im_end:
            raise ValueError("Truncated teacher reference lost the answer region")
        prompt_only_ids = self._encode_prompt_only_ids(user_content)

        labels = list(student_ids)
        loss_weights = [0.0] * len(student_ids)
        prompt_mask = [False] * len(student_ids)
        latent_internal_mask = [False] * len(student_ids)
        latent_boundary_mask = [False] * len(student_ids)
        cot_mask = [False] * len(student_ids)
        answer_mask = [False] * len(student_ids)
        teacher_kl_mask = [False] * len(student_ids)

        for idx in range(spans.assistant_content_start):
            labels[idx] = -100
            prompt_mask[idx] = True

        for idx in range(spans.latent_start + 1, spans.latent_end):
            labels[idx] = -100
            latent_internal_mask[idx] = True

        latent_boundary_mask[spans.latent_start] = True
        latent_boundary_mask[spans.latent_end] = True
        latent_start_weight = float(row.get("latent_start_ce_loss_weight", self.latent_start_ce_loss_weight))
        latent_end_weight = float(row.get("latent_end_ce_loss_weight", self.latent_end_ce_loss_weight))
        loss_weights[spans.latent_start] = float(latent_start_weight)
        loss_weights[spans.latent_end] = float(latent_end_weight)

        cot_branch_weight = (
            float(self.cot_ce_loss_weight)
            if self.cot_ce_loss_weight is not None
            else float(row["cot_loss_weight"])
        )
        for idx in range(spans.think_start, spans.answer_start):
            cot_mask[idx] = True
            loss_weights[idx] = 0.0
            if not bool(row.get("skip_teacher_kl", False)):
                teacher_kl_mask[idx - 1] = idx > 0

        for idx in range(spans.answer_start, spans.im_end):
            answer_mask[idx] = True
            loss_weights[idx] = float(row["answer_loss_weight"])
            if not bool(row.get("skip_teacher_kl", False)):
                teacher_kl_mask[idx - 1] = idx > 0

        loss_weights[spans.im_end] = float(self.im_end_ce_loss_weight)

        latent_length = spans.latent_end - spans.latent_start - 1
        if latent_length < 0:
            raise ValueError("Invalid latent span length")

        return {
            "record_id": str(row["record_id"]),
            "source_uid": str(row.get("source_uid", row["record_id"])),
            "text": student_text,
            "token_ids": student_ids,
            "prompt_only_ids": prompt_only_ids,
            "teacher_ids": teacher_ids,
            "labels": labels,  # user prompt and latent tokens (except boundaries) are masked to -100
            "loss_weights": loss_weights,
            "cot_branch_weight": cot_branch_weight,
            "prompt_mask": prompt_mask,
            "latent_internal_mask": latent_internal_mask,
            "latent_boundary_mask": latent_boundary_mask,
            "cot_mask": cot_mask,
            "answer_mask": answer_mask,  # metric mask excludes <|im_end|>; CE now supervises it separately
            "teacher_kl_mask": teacher_kl_mask,
            "teacher_target_start": spans.latent_end,
            "spans": spans,
            "difficulty_rank": int(row["difficulty_rank"]),
            "n_latent_steps": int(row["n_latent_steps"]),
            "latent_length": latent_length,
            "curriculum_sort_key": [int(v) for v in row["curriculum_sort_key"]],
            "stage2_is_correct": bool(row["stage2_is_correct"]),
            "assistant_answer": str(row["assistant_answer"]),
            "benchmark_name": str(row.get("benchmark_name", "train")),
            "benchmark_split": str(row.get("benchmark_split", "train")),
            "skip_teacher_kl": bool(row.get("skip_teacher_kl", False)),
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
                raise ValueError("Unable to build a valid sample after multiple lazy-dataset retries")
        else:
            sample = dict(self.samples[index])
        sample["index"] = index
        return sample

    def get_curriculum_sort_key(self, index: int) -> List[int]:
        return list(self.curriculum_sort_keys[index])


class LatentSFTCollator:
    def __init__(self, tokenizer: Any, config: Dict[str, Any]):
        self.tokenizer = tokenizer
        self.pad_token_id = int(tokenizer.pad_token_id)
        self.config = config
        self.train_stage = int(config["train_stage"])
        self.curriculum_progress = 0.0
        self._shared_train_stage = mp.Value("i", int(self.train_stage))
        self._shared_curriculum_progress = mp.Value("d", float(self.curriculum_progress))
        self.rng = random.Random(int(config["seed"]))

    def set_stage(self, train_stage: int, progress: float) -> None:
        self.train_stage = int(train_stage)
        self.curriculum_progress = float(progress)
        with self._shared_train_stage.get_lock():
            self._shared_train_stage.value = int(self.train_stage)
        with self._shared_curriculum_progress.get_lock():
            self._shared_curriculum_progress.value = float(self.curriculum_progress)

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

    def _should_apply_stage2_restriction(self, sample: Dict[str, Any]) -> bool:
        if int(self.config.get("context_parallel_size", 1) or 1) > 1:
            return False
        stage = int(self._shared_train_stage.value)
        if stage < 2:
            return False
        if sample["difficulty_rank"] != 0:
            return False
        cot_length = sample["spans"].answer_start - sample["spans"].think_start
        if cot_length >= int(self.config["stage2_easy_cot_max_tokens"]):
            return False
        return self.rng.random() < float(self.config["stage2_easy_sample_prob"])

    def _call_cp_single(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if len(batch) != 1:
            raise ValueError(f"Latent CP training requires global batch size 1, got batch_size={len(batch)}")
        sample = batch[0]
        seq_len = len(sample["token_ids"])
        padded_len = int(seq_len)
        required_multiple = int(resolve_context_parallel_padding_multiple(self.config))
        if required_multiple > 1:
            remainder = padded_len % required_multiple
            if remainder:
                padded_len += required_multiple - remainder

        def tensor_long(values: Sequence[int], fill: int) -> torch.Tensor:
            out = torch.full((1, padded_len), int(fill), dtype=torch.long)
            out[0, :seq_len] = torch.tensor(values, dtype=torch.long)
            return out

        def tensor_bool(values: Sequence[bool]) -> torch.Tensor:
            out = torch.zeros((1, padded_len), dtype=torch.bool)
            out[0, :seq_len] = torch.tensor(values, dtype=torch.bool)
            return out

        def tensor_float(values: Sequence[float]) -> torch.Tensor:
            out = torch.zeros((1, padded_len), dtype=torch.float32)
            out[0, :seq_len] = torch.tensor(values, dtype=torch.float32)
            return out

        valid_token_mask_list = [True] * seq_len + [False] * (padded_len - seq_len)
        latent_positions_list = list(range(sample["spans"].latent_start + 1, sample["spans"].latent_end))
        latent_length = len(latent_positions_list)
        (
            loss_source_positions_list,
            loss_target_positions_list,
            kl_source_positions_list,
            kl_target_positions_list,
        ) = self._build_effective_token_mappings(
            valid_token_mask=valid_token_mask_list,
            labels=list(sample["labels"]) + [-100] * (padded_len - seq_len),
            teacher_kl_mask=list(sample["teacher_kl_mask"]) + [False] * (padded_len - seq_len),
        )

        input_ids = torch.full((1, padded_len), self.pad_token_id, dtype=torch.long)
        input_ids[0, :seq_len] = torch.tensor(sample["token_ids"], dtype=torch.long)
        labels = tensor_long(sample["labels"], -100)
        loss_weights = tensor_float(sample["loss_weights"])
        cot_branch_weight = torch.tensor([float(sample["cot_branch_weight"])], dtype=torch.float32)
        attention_mask = torch.zeros((1, padded_len), dtype=torch.long)
        attention_mask[0, :seq_len] = 1
        valid_token_mask = attention_mask.to(torch.bool)
        position_ids = torch.arange(padded_len, dtype=torch.long).unsqueeze(0)

        latent_positions = torch.full((1, max(latent_length, 0)), -1, dtype=torch.long)
        latent_slot_mask = torch.zeros((1, max(latent_length, 0)), dtype=torch.bool)
        if latent_length > 0:
            latent_positions[0, :latent_length] = torch.tensor(latent_positions_list, dtype=torch.long)
            latent_slot_mask[0, :latent_length] = True

        max_loss_pairs = len(loss_source_positions_list)
        loss_source_positions = torch.full((1, max_loss_pairs), -1, dtype=torch.long)
        loss_target_positions = torch.full((1, max_loss_pairs), -1, dtype=torch.long)
        loss_pair_mask = torch.zeros((1, max_loss_pairs), dtype=torch.bool)
        if max_loss_pairs > 0:
            loss_source_positions[0] = torch.tensor(loss_source_positions_list, dtype=torch.long)
            loss_target_positions[0] = torch.tensor(loss_target_positions_list, dtype=torch.long)
            loss_pair_mask[0] = True

        max_kl_pairs = len(kl_source_positions_list)
        teacher_kl_source_positions = torch.full((1, max_kl_pairs), -1, dtype=torch.long)
        teacher_kl_target_positions = torch.full((1, max_kl_pairs), -1, dtype=torch.long)
        teacher_kl_pair_mask = torch.zeros((1, max_kl_pairs), dtype=torch.bool)
        if max_kl_pairs > 0:
            teacher_kl_source_positions[0] = torch.tensor(kl_source_positions_list, dtype=torch.long)
            teacher_kl_target_positions[0] = torch.tensor(kl_target_positions_list, dtype=torch.long)
            teacher_kl_pair_mask[0] = True

        spans = {
            "assistant_prefix_start": int(sample["spans"].assistant_prefix_start),
            "assistant_content_start": int(sample["spans"].assistant_content_start),
            "latent_start": int(sample["spans"].latent_start),
            "latent_end": int(sample["spans"].latent_end),
            "think_start": int(sample["spans"].think_start),
            "think_end": int(sample["spans"].think_end),
            "answer_start": int(sample["spans"].answer_start),
            "im_end": int(sample["spans"].im_end),
        }
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_weights": loss_weights,
            "cot_branch_weight": cot_branch_weight,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prompt_mask": tensor_bool(sample["prompt_mask"]),
            "latent_internal_mask": tensor_bool(sample["latent_internal_mask"]),
            "latent_boundary_mask": tensor_bool(sample["latent_boundary_mask"]),
            "cot_mask": tensor_bool(sample["cot_mask"]),
            "answer_mask": tensor_bool(sample["answer_mask"]),
            "teacher_kl_mask": tensor_bool(sample["teacher_kl_mask"]),
            "valid_token_mask": valid_token_mask,
            "latent_pad_mask": torch.zeros((1, padded_len), dtype=torch.bool),
            "teacher_target_start": torch.tensor([int(sample["teacher_target_start"])], dtype=torch.long),
            "record_ids": [sample["record_id"]],
            "prompt_only_ids": [sample["prompt_only_ids"]],
            "assistant_answers": [sample["assistant_answer"]],
            "benchmark_names": [sample["benchmark_name"]],
            "benchmark_splits": [sample["benchmark_split"]],
            "skip_teacher_kl_flags": [bool(sample["skip_teacher_kl"])],
            "spans": [spans],
            "curriculum_sort_keys": [sample["curriculum_sort_key"]],
            "latent_positions": latent_positions,
            "latent_slot_mask": latent_slot_mask,
            "latent_lengths": torch.tensor([latent_length], dtype=torch.long),
            "latent_start_positions": torch.tensor([spans["latent_start"]], dtype=torch.long),
            "latent_end_positions": torch.tensor([spans["latent_end"]], dtype=torch.long),
            "loss_source_positions": loss_source_positions,
            "loss_target_positions": loss_target_positions,
            "loss_pair_mask": loss_pair_mask,
            "teacher_kl_source_positions": teacher_kl_source_positions,
            "teacher_kl_target_positions": teacher_kl_target_positions,
            "teacher_kl_pair_mask": teacher_kl_pair_mask,
        }

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if int(self.config.get("context_parallel_size", 1) or 1) > 1:
            return self._call_cp_single(batch)

        assistant_targets = [
            sample["spans"].assistant_prefix_start for sample in batch
        ]  # position pf <|im_start|>assistant
        think_targets = [sample["spans"].think_start for sample in batch]
        aligned_assistant = max(assistant_targets)
        aligned_think = max(
            aligned_assistant + (t - a) for a, t in zip(assistant_targets, think_targets)
        )  # (t-a)=tokens of latent_reasoning, padding to <|im_start|>assistant + tokens to <think>

        # print(f"aligned_assistant={aligned_assistant}, aligned_think={aligned_think}")
        padded_samples = []
        max_seq_len = 0
        max_latent_internal = 0
        for sample in batch:
            left_pad = aligned_assistant - sample["spans"].assistant_prefix_start
            think_after_left = left_pad + sample["spans"].think_start
            middle_pad = aligned_think - think_after_left
            token_ids = [self.pad_token_id] * left_pad
            token_ids.extend(sample["token_ids"][: sample["spans"].think_start])
            token_ids.extend([self.pad_token_id] * middle_pad)
            token_ids.extend(sample["token_ids"][sample["spans"].think_start :])

            labels = [-100] * left_pad
            labels.extend(sample["labels"][: sample["spans"].think_start])
            labels.extend([-100] * middle_pad)
            labels.extend(sample["labels"][sample["spans"].think_start :])

            def pad_bool(name: str) -> List[bool]:
                values = [False] * left_pad
                values.extend(sample[name][: sample["spans"].think_start])
                values.extend([False] * middle_pad)
                values.extend(sample[name][sample["spans"].think_start :])
                return values

            def pad_float(name: str) -> List[float]:
                values = [0.0] * left_pad
                values.extend(sample[name][: sample["spans"].think_start])
                values.extend([0.0] * middle_pad)
                values.extend(sample[name][sample["spans"].think_start :])
                return values

            spans = {
                "assistant_prefix_start": left_pad + sample["spans"].assistant_prefix_start,
                "assistant_content_start": left_pad + sample["spans"].assistant_content_start,
                "latent_start": left_pad + sample["spans"].latent_start,
                "latent_end": left_pad + sample["spans"].latent_end,
                "think_start": aligned_think,
                "think_end": left_pad + middle_pad + sample["spans"].think_end,
                "answer_start": left_pad + middle_pad + sample["spans"].answer_start,
                "im_end": left_pad + middle_pad + sample["spans"].im_end,
            }
            valid_token_mask = torch.tensor(
                [False] * left_pad
                + [True] * sample["spans"].think_start
                + [False] * middle_pad
                + [True] * (len(sample["token_ids"]) - sample["spans"].think_start)
            )
            latent_positions = list(
                range(spans["latent_start"] + 1, spans["latent_end"])
            )  # don't include <latent_think> and </latent_think>
            latent_length = len(latent_positions)
            latent_slot_mask = [True] * latent_length
            latent_pad_mask = (
                [False] * left_pad
                + [False] * sample["spans"].think_start
                + [True] * middle_pad
                + [False] * (len(sample["token_ids"]) - sample["spans"].think_start)
            )
            padded_teacher_kl_mask = pad_bool("teacher_kl_mask")
            (
                loss_source_positions,
                loss_target_positions,
                kl_source_positions,
                kl_target_positions,
            ) = self._build_effective_token_mappings(
                valid_token_mask=valid_token_mask.tolist(),
                labels=labels,
                teacher_kl_mask=padded_teacher_kl_mask,
            )
            max_latent_internal = max(max_latent_internal, latent_length)
            max_seq_len = max(max_seq_len, len(token_ids))
            padded_samples.append(
                {
                    "record_id": sample["record_id"],
                    "token_ids": token_ids,
                    "labels": labels,
                    "loss_weights": pad_float("loss_weights"),
                    "cot_branch_weight": float(sample["cot_branch_weight"]),
                    "prompt_mask": pad_bool("prompt_mask"),
                    "latent_internal_mask": pad_bool("latent_internal_mask"),
                    "latent_boundary_mask": pad_bool("latent_boundary_mask"),
                    "cot_mask": pad_bool("cot_mask"),
                    "answer_mask": pad_bool("answer_mask"),
                    "teacher_kl_mask": padded_teacher_kl_mask,
                    "valid_token_mask": valid_token_mask.tolist(),
                    "latent_pad_mask": latent_pad_mask,
                    "teacher_target_start": left_pad + sample["teacher_target_start"],
                    "spans": spans,
                    "prompt_only_ids": sample["prompt_only_ids"],
                    "assistant_answer": sample["assistant_answer"],
                    "benchmark_name": sample["benchmark_name"],
                    "benchmark_split": sample["benchmark_split"],
                    "skip_teacher_kl": bool(sample["skip_teacher_kl"]),
                    "stage2_restricted": self._should_apply_stage2_restriction(sample),
                    "curriculum_sort_key": sample["curriculum_sort_key"],
                    "latent_positions": latent_positions,
                    "latent_length": latent_length,
                    "latent_slot_mask": latent_slot_mask,
                    "loss_source_positions": loss_source_positions,
                    "loss_target_positions": loss_target_positions,
                    "kl_source_positions": kl_source_positions,
                    "kl_target_positions": kl_target_positions,
                }
            )

        batch_size = len(padded_samples)
        input_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)
        loss_weights = torch.zeros((batch_size, max_seq_len), dtype=torch.float32)
        cot_branch_weight = torch.zeros((batch_size,), dtype=torch.float32)
        prompt_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
        latent_internal_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
        latent_boundary_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
        cot_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
        answer_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
        teacher_kl_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
        valid_token_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
        latent_pad_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
        position_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        latent_positions = torch.full((batch_size, max_latent_internal), -1, dtype=torch.long)
        latent_slot_mask = torch.zeros((batch_size, max_latent_internal), dtype=torch.bool)
        latent_lengths = torch.zeros((batch_size,), dtype=torch.long)
        latent_start_positions = torch.zeros((batch_size,), dtype=torch.long)
        latent_end_positions = torch.zeros((batch_size,), dtype=torch.long)
        max_loss_pairs = max((len(sample["loss_target_positions"]) for sample in padded_samples), default=0)
        max_kl_pairs = max((len(sample["kl_source_positions"]) for sample in padded_samples), default=0)
        loss_source_positions = torch.full((batch_size, max_loss_pairs), -1, dtype=torch.long)
        loss_target_positions = torch.full((batch_size, max_loss_pairs), -1, dtype=torch.long)
        loss_pair_mask = torch.zeros((batch_size, max_loss_pairs), dtype=torch.bool)
        teacher_kl_source_positions = torch.full((batch_size, max_kl_pairs), -1, dtype=torch.long)
        teacher_kl_target_positions = torch.full((batch_size, max_kl_pairs), -1, dtype=torch.long)
        teacher_kl_pair_mask = torch.zeros((batch_size, max_kl_pairs), dtype=torch.bool)
        use_custom_attention_bias = bool(self.config.get("use_custom_attention_bias", False))
        build_attention_bias = use_custom_attention_bias and any(
            sample["stage2_restricted"] for sample in padded_samples
        )
        attention_bias = None
        if build_attention_bias:
            attention_bias = torch.full(
                (batch_size, 1, max_seq_len, max_seq_len), fill_value=-1.0e9, dtype=torch.float32
            )
        teacher_target_start = torch.zeros((batch_size,), dtype=torch.long)

        for row_idx, sample in enumerate(padded_samples):
            seq_len = len(sample["token_ids"])
            input_ids[row_idx, :seq_len] = torch.tensor(sample["token_ids"], dtype=torch.long)
            labels[row_idx, :seq_len] = torch.tensor(sample["labels"], dtype=torch.long)
            loss_weights[row_idx, :seq_len] = torch.tensor(sample["loss_weights"], dtype=torch.float32)
            cot_branch_weight[row_idx] = float(sample["cot_branch_weight"])
            prompt_mask[row_idx, :seq_len] = torch.tensor(sample["prompt_mask"], dtype=torch.bool)
            latent_internal_mask[row_idx, :seq_len] = torch.tensor(sample["latent_internal_mask"], dtype=torch.bool)
            latent_boundary_mask[row_idx, :seq_len] = torch.tensor(sample["latent_boundary_mask"], dtype=torch.bool)
            cot_mask[row_idx, :seq_len] = torch.tensor(sample["cot_mask"], dtype=torch.bool)
            answer_mask[row_idx, :seq_len] = torch.tensor(sample["answer_mask"], dtype=torch.bool)
            teacher_kl_mask[row_idx, :seq_len] = torch.tensor(sample["teacher_kl_mask"], dtype=torch.bool)
            valid_token_mask[row_idx, :seq_len] = torch.tensor(sample["valid_token_mask"], dtype=torch.bool)
            latent_pad_mask[row_idx, :seq_len] = torch.tensor(sample["latent_pad_mask"], dtype=torch.bool)
            attention_mask[row_idx, :seq_len] = valid_token_mask[row_idx, :seq_len].to(torch.long)
            teacher_target_start[row_idx] = int(sample["teacher_target_start"])
            position_ids[row_idx] = build_position_ids(valid_token_mask[row_idx])
            latent_lengths[row_idx] = int(sample["latent_length"])
            latent_start_positions[row_idx] = int(sample["spans"]["latent_start"])
            latent_end_positions[row_idx] = int(sample["spans"]["latent_end"])

            if sample["latent_positions"]:
                length = len(sample["latent_positions"])
                latent_positions[row_idx, :length] = torch.tensor(sample["latent_positions"], dtype=torch.long)
                latent_slot_mask[row_idx, :length] = True
            if sample["loss_target_positions"]:
                pair_len = len(sample["loss_target_positions"])
                loss_source_positions[row_idx, :pair_len] = torch.tensor(
                    sample["loss_source_positions"], dtype=torch.long
                )
                loss_target_positions[row_idx, :pair_len] = torch.tensor(
                    sample["loss_target_positions"], dtype=torch.long
                )
                loss_pair_mask[row_idx, :pair_len] = True
            if sample["kl_source_positions"]:
                kl_pair_len = len(sample["kl_source_positions"])
                teacher_kl_source_positions[row_idx, :kl_pair_len] = torch.tensor(
                    sample["kl_source_positions"], dtype=torch.long
                )
                teacher_kl_target_positions[row_idx, :kl_pair_len] = torch.tensor(
                    sample["kl_target_positions"], dtype=torch.long
                )
                teacher_kl_pair_mask[row_idx, :kl_pair_len] = True

            if attention_bias is not None:
                visible = valid_token_mask[row_idx]
                for tgt in range(max_seq_len):
                    if not visible[tgt]:
                        continue
                    allowed = visible.clone()
                    allowed[tgt + 1 :] = False
                    if (
                        sample["stage2_restricted"]
                        and sample["spans"]["think_start"] <= tgt < sample["spans"]["answer_start"]
                    ):
                        prompt_region = torch.zeros_like(allowed)
                        prompt_region[: sample["spans"]["assistant_content_start"]] = True
                        allowed &= ~prompt_region
                    attention_bias[row_idx, 0, tgt, allowed] = 0.0

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "loss_weights": loss_weights,
            "cot_branch_weight": cot_branch_weight,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prompt_mask": prompt_mask,
            "latent_internal_mask": latent_internal_mask,
            "latent_boundary_mask": latent_boundary_mask,
            "cot_mask": cot_mask,
            "answer_mask": answer_mask,
            "teacher_kl_mask": teacher_kl_mask,
            "valid_token_mask": valid_token_mask,
            "latent_pad_mask": latent_pad_mask,
            "teacher_target_start": teacher_target_start,
            "record_ids": [sample["record_id"] for sample in padded_samples],
            "prompt_only_ids": [sample["prompt_only_ids"] for sample in padded_samples],
            "assistant_answers": [sample["assistant_answer"] for sample in padded_samples],
            "benchmark_names": [sample["benchmark_name"] for sample in padded_samples],
            "benchmark_splits": [sample["benchmark_split"] for sample in padded_samples],
            "skip_teacher_kl_flags": [bool(sample["skip_teacher_kl"]) for sample in padded_samples],
            "spans": [sample["spans"] for sample in padded_samples],
            "curriculum_sort_keys": [sample["curriculum_sort_key"] for sample in padded_samples],
            "latent_positions": latent_positions,
            "latent_slot_mask": latent_slot_mask,
            "latent_lengths": latent_lengths,
            "latent_start_positions": latent_start_positions,
            "latent_end_positions": latent_end_positions,
            "loss_source_positions": loss_source_positions,
            "loss_target_positions": loss_target_positions,
            "loss_pair_mask": loss_pair_mask,
            "teacher_kl_source_positions": teacher_kl_source_positions,
            "teacher_kl_target_positions": teacher_kl_target_positions,
            "teacher_kl_pair_mask": teacher_kl_pair_mask,
        }
        if attention_bias is not None:
            result["attention_bias"] = attention_bias
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal dataset/collator smoke test")
    parser.add_argument("--config", default="later/src/config/sft_config.yaml")
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--max_train_samples", type=int, default=64)
    parser.add_argument("--max_val_samples", type=int, default=64)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from later.src.train.utils import load_yaml
    from later.src.utils.utils import ensure_latent_think_special_tokens

    config = load_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True, use_fast=True)
    ensure_latent_think_special_tokens(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "<|endoftext|>"

    train_df, val_df = load_sft_split(config["train_data"], float(config["val_ratio"]))
    if args.max_train_samples > 0:
        train_df = train_df.iloc[: args.max_train_samples].reset_index(drop=True)
    if args.max_val_samples > 0:
        val_df = val_df.iloc[: args.max_val_samples].reset_index(drop=True)

    train_dataset = LatentSFTDataset(
        frame=train_df,
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
        seed=int(config["seed"]),
        lazy=bool(config.get("lazy_dataset", False)),
        cot_ce_loss_weight=float(config.get("cot_ce_loss_weight", 0.3)),
        im_end_ce_loss_weight=float(config.get("im_end_ce_loss_weight", 1.0)),
    )
    val_dataset = LatentSFTDataset(
        frame=val_df,
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
        seed=int(config["seed"]),
        lazy=bool(config.get("lazy_dataset", False)),
        cot_ce_loss_weight=float(config.get("cot_ce_loss_weight", 0.3)),
        im_end_ce_loss_weight=float(config.get("im_end_ce_loss_weight", 1.0)),
    )
    collator = LatentSFTCollator(tokenizer=tokenizer, config=config)

    def run_one(split_name: str, dataset: LatentSFTDataset) -> None:
        if len(dataset) == 0:
            raise ValueError(f"{split_name} dataset is empty")
        n = min(int(args.batch_size), len(dataset))
        batch = collator([dataset[i] for i in range(n)])
        assert batch["input_ids"].shape[0] == n
        assert batch["labels"].shape == batch["input_ids"].shape
        assert batch["attention_mask"].shape == batch["input_ids"].shape
        assert batch["valid_token_mask"].shape == batch["input_ids"].shape
        assert torch.equal(batch["attention_mask"].to(torch.bool), batch["valid_token_mask"])
        print(
            f"[{split_name}] ok: batch_size={n}, padded_seq_len={batch['input_ids'].shape[1]}, keys={len(batch.keys())}"
        )

    if args.split in {"train", "both"}:
        run_one("train", train_dataset)
    if args.split in {"val", "both"}:
        collator.set_stage(train_stage=2, progress=1.0)
        run_one("val", val_dataset)

    print("Smoke test passed.")
