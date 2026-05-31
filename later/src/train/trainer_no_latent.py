from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Iterator, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.nn import functional as dist_nn_functional
from torch.utils.data import DataLoader, WeightedRandomSampler

from later.src.train.losses import compute_inner_cot_mask
from later.src.train.trainer import LatentSFTTrainer
from later.src.train.utils import (
    curriculum_weight,
    mean_or_zero,
    resolve_context_parallel_padding_multiple,
)


class _DeviceLoader:
    def __init__(self, loader: DataLoader, device: torch.device) -> None:
        self.loader = loader
        self.device = device

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for batch in self.loader:
            yield {
                key: value.to(device=self.device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }

    def __len__(self) -> int:
        return len(self.loader)


class NoLatentSFTTrainer(LatentSFTTrainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if str(self.config.get("distributed_backend", "")).lower() != "fsdp":
            raise ValueError("No-latent trainer is CP-only and requires distributed_backend=fsdp.")
        if self._context_parallel_size() <= 1:
            raise ValueError("No-latent trainer is CP-only and requires context_parallel_size > 1.")
        self.stage1_trainable_mode = "full"
        self.stage2_trainable_mode = "full"
        if self._current_trainable_mode is None:
            self._current_trainable_mode = "full"
        self._cp_forward_debug_logged = False
        self._cp_generation_debug_logged = False
        self._cp_kl_zero_warning_logged = False

    def _context_parallel_size(self) -> int:
        return max(int(self.config.get("context_parallel_size", 1) or 1), 1)

    def _context_parallel_enabled(self) -> bool:
        return True

    def _effective_global_batch_size(self, num_processes: int) -> int:
        del num_processes
        return 1

    def _compute_train_stage(self, global_step: int, total_train_steps: int) -> tuple[int, float]:
        progress = float(global_step) / float(max(int(total_train_steps), 1))
        progress = min(max(progress, 0.0), 1.0)
        train_stage = 2 if progress >= float(self.config.get("stage2_start_fraction", 0.0)) else 1
        if int(self.config.get("train_stage", 2)) == 2:
            train_stage = 2
        return int(train_stage), float(progress)

    def _prepare_dataloader(self, loader: DataLoader, purpose: str) -> DataLoader | _DeviceLoader:
        self.logger.info(
            {
                "tag": "cp_dataloader_unsharded",
                "purpose": str(purpose),
                "context_parallel_size": int(self._context_parallel_size()),
                "dataloader_prepared": False,
                "raw_loader_len": int(len(loader)),
                "effective_global_batch_size": 1,
            }
        )
        return _DeviceLoader(loader=loader, device=self.accelerator.device)

    def _sampler_for_epoch(self, epoch_index: int) -> WeightedRandomSampler:
        total_epochs = max(int(self.config["num_epochs"]), 1)
        progress = 0.0 if total_epochs == 1 else float(epoch_index / (total_epochs - 1))
        weights = [
            curriculum_weight(
                sort_key=self.train_dataset.get_curriculum_sort_key(index),
                progress=progress,
                power=float(self.config["curriculum_power"]),
            )
            for index in range(len(self.train_dataset))
        ]
        generator = torch.Generator()
        generator.manual_seed(int(self.config.get("seed", 0)) + int(epoch_index))
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=False,
            generator=generator,
        )

    def _assert_same_record_id_across_ranks(self, batch: Dict[str, Any], phase: str) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        record_ids = list(batch.get("record_ids", []))
        local_record_id = str(record_ids[0]) if record_ids else ""
        gathered: List[Any] = [None for _ in range(int(dist.get_world_size()))]
        dist.all_gather_object(gathered, local_record_id)
        unique_ids = sorted({str(item) for item in gathered})
        if len(unique_ids) != 1:
            raise RuntimeError(
                "No-latent CP requires every rank's DataLoader to produce the same sample before forward. "
                f"phase={phase}, gathered_record_ids={gathered}"
            )

    @staticmethod
    def _autograd_all_reduce_sum(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        if not (dist.is_available() and dist.is_initialized()):
            return tensor
        return dist_nn_functional.all_reduce(tensor, op=dist.ReduceOp.SUM)

    @staticmethod
    def _to_local_tensor(value: torch.Tensor) -> torch.Tensor:
        if hasattr(value, "to_local"):
            return value.to_local()
        return value

    def _validate_cp_batch(self, batch: Dict[str, Any]) -> None:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        if int(input_ids.size(0)) != 1:
            raise ValueError(f"No-latent CP requires global batch size 1, got batch_shape={list(input_ids.shape)}")
        cp_size = self._context_parallel_size()
        padding_factor = max(int(self.config.get("context_parallel_padding_factor", 2) or 2), 1)
        required_multiple = int(resolve_context_parallel_padding_multiple(self.config))
        if int(input_ids.size(1)) % required_multiple != 0:
            raise ValueError(
                "No-latent CP requires sequence length divisible by the Accelerate CP padding multiple. "
                f"seq_len={int(input_ids.size(1))}, context_parallel_size={cp_size}, "
                f"context_parallel_padding_factor={padding_factor}, required_multiple={required_multiple}."
            )
        mask = attention_mask.to(torch.long)
        valid_len = int(mask[0].sum().item())
        if valid_len < 0 or valid_len > int(mask.size(1)):
            raise ValueError(f"Invalid attention mask valid_len={valid_len}, seq_len={int(mask.size(1))}")
        if bool((mask[0, :valid_len] != 1).any().item()) or bool((mask[0, valid_len:] != 0).any().item()):
            raise ValueError("No-latent CP only supports right padding: attention_mask must be prefix-ones suffix-zeros.")

    def _batch_diagnostics(self, batch: Dict[str, Any], global_step: int, epoch_index: int) -> Dict[str, Any]:
        seq_len = int(batch["input_ids"].shape[1])
        attention_mask = batch["attention_mask"].detach().to(torch.long).cpu()
        token_lengths_tensor = attention_mask.sum(dim=1).to(torch.long)
        think_start_positions = batch["think_start_positions"].detach().to(torch.long).cpu()
        think_end_positions = batch["think_end_positions"].detach().to(torch.long).cpu()
        cot_lengths_tensor = (think_end_positions - think_start_positions - 1).clamp_min(0)
        diag: Dict[str, Any] = {
            "global_step": int(global_step),
            "epoch_index": int(epoch_index),
            "micro_batch_size": int(batch["input_ids"].shape[0]),
            "padded_seq_len": seq_len,
            "token_lengths": token_lengths_tensor.tolist(),
            "max_token_length": int(token_lengths_tensor.max().item()) if token_lengths_tensor.numel() > 0 else 0,
            "cot_lengths": cot_lengths_tensor.tolist(),
            "max_cot_length": int(cot_lengths_tensor.max().item()) if cot_lengths_tensor.numel() > 0 else 0,
            "think_start_positions": think_start_positions.tolist(),
            "think_end_positions": think_end_positions.tolist(),
            "record_ids": list(batch.get("record_ids", []))[:4],
            "teacher_length_mismatch_count": int(self.teacher_length_mismatch_count),
        }
        loss_pair_mask = batch.get("loss_pair_mask")
        if torch.is_tensor(loss_pair_mask):
            loss_pairs_tensor = loss_pair_mask.detach().to(torch.long).cpu().sum(dim=1)
            diag["loss_pairs"] = loss_pairs_tensor.tolist()
            diag["max_loss_pairs"] = int(loss_pairs_tensor.max().item()) if loss_pairs_tensor.numel() > 0 else 0
        teacher_kl_pair_mask = batch.get("teacher_kl_pair_mask")
        if torch.is_tensor(teacher_kl_pair_mask):
            kl_pairs_tensor = teacher_kl_pair_mask.detach().to(torch.long).cpu().sum(dim=1)
            diag["teacher_kl_pairs"] = kl_pairs_tensor.tolist()
            diag["max_teacher_kl_pairs"] = int(kl_pairs_tensor.max().item()) if kl_pairs_tensor.numel() > 0 else 0
        if torch.cuda.is_available() and self.enable_forward_memory_breakdown_log:
            diag.update(self._safe_current_cuda_memory_gib())
        return diag

    @staticmethod
    def _safe_current_cuda_memory_gib() -> Dict[str, float]:
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        return {
            "allocated_gib": float(allocated),
            "reserved_gib": float(reserved),
            "max_allocated_gib": float(max_allocated),
            "max_reserved_gib": float(max_reserved),
        }

    @staticmethod
    def _build_cp_local_position_runs(local_positions: torch.Tensor) -> List[tuple[int, int, int, int]]:
        if local_positions.dim() != 1:
            raise ValueError(f"Expected rank-1 local_positions, got shape={list(local_positions.shape)}")
        if int(local_positions.numel()) <= 0:
            return []
        if int(local_positions.numel()) == 1:
            pos = int(local_positions[0].item())
            return [(0, 1, pos, pos + 1)]

        diffs = local_positions[1:] - local_positions[:-1]
        break_points = torch.nonzero(diffs != 1, as_tuple=False).flatten() + 1
        starts = torch.cat(
            [
                torch.zeros((1,), device=local_positions.device, dtype=torch.long),
                break_points.to(dtype=torch.long),
            ],
            dim=0,
        )
        ends = torch.cat(
            [
                break_points.to(dtype=torch.long),
                torch.tensor([int(local_positions.numel())], device=local_positions.device, dtype=torch.long),
            ],
            dim=0,
        )

        runs: List[tuple[int, int, int, int]] = []
        for start_tensor, end_tensor in zip(starts, ends):
            local_start = int(start_tensor.item())
            local_end = int(end_tensor.item())
            if local_end <= local_start:
                continue
            global_start = int(local_positions[local_start].item())
            global_end = int(local_positions[local_end - 1].item()) + 1
            expected_len = int(global_end - global_start)
            actual_len = int(local_end - local_start)
            if expected_len != actual_len:
                raise RuntimeError(
                    "Invalid CP position run: expected each run to be contiguous with stride 1, "
                    f"local_start={local_start}, local_end={local_end}, "
                    f"global_start={global_start}, global_end={global_end}."
                )
            runs.append((local_start, local_end, global_start, global_end))
        return runs

    @staticmethod
    def _map_global_positions_to_cp_local_indices(
        source_positions: torch.Tensor,
        pair_valid: torch.Tensor,
        runs: List[tuple[int, int, int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        matched = torch.zeros_like(pair_valid, dtype=torch.bool)
        local_indices = torch.full_like(source_positions, fill_value=-1, dtype=torch.long)
        for local_start, _local_end, global_start, global_end in runs:
            in_run = (
                pair_valid
                & (source_positions >= int(global_start))
                & (source_positions < int(global_end))
            )
            if bool(in_run.any().item()):
                local_indices[in_run] = int(local_start) + (source_positions[in_run] - int(global_start))
                matched = matched | in_run
        return matched, local_indices

    def _maybe_log_cp_position_runs(
        self,
        local_positions: torch.Tensor,
        runs: List[tuple[int, int, int, int]],
    ) -> None:
        if not self._context_parallel_enabled() or self._cp_forward_debug_logged:
            return
        rank = int(dist.get_rank()) if (dist.is_available() and dist.is_initialized()) else 0
        self.logger.info(
            {
                "tag": "cp_position_runs_debug",
                "rank": int(rank),
                "local_seq_len": int(local_positions.numel()),
                "position_runs": [list(run) for run in runs],
                "position_head": local_positions[:8].detach().cpu().tolist(),
                "position_tail": local_positions[-8:].detach().cpu().tolist(),
            }
        )
        self._cp_forward_debug_logged = True

    def _compute_cp_teacher_kl_from_local_logits(
        self,
        local_logits: torch.Tensor,
        position_runs: List[tuple[int, int, int, int]],
        batch: Dict[str, Any],
        zero_anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_kl_numer = zero_anchor.float()
        local_kl_count = torch.zeros((), device=local_logits.device, dtype=torch.float32)
        source_positions_tensor = batch.get("teacher_kl_source_positions")
        pair_mask_tensor = batch.get("teacher_kl_effective_mask", batch.get("teacher_kl_pair_mask"))
        topk_ids_tensor = batch.get("teacher_kl_topk_ids")
        topk_probs_tensor = batch.get("teacher_kl_topk_probs")
        tail_tensor = batch.get("teacher_kl_tail")
        required = [
            source_positions_tensor,
            pair_mask_tensor,
            topk_ids_tensor,
            topk_probs_tensor,
            tail_tensor,
        ]
        if not all(torch.is_tensor(value) for value in required):
            return local_kl_numer, local_kl_count

        device = local_logits.device
        source_positions = source_positions_tensor[0].to(device=device, dtype=torch.long)
        pair_valid = pair_mask_tensor[0].to(device=device, dtype=torch.bool) & (source_positions >= 0)
        in_local, local_indices = self._map_global_positions_to_cp_local_indices(
            source_positions=source_positions,
            pair_valid=pair_valid,
            runs=position_runs,
        )
        if not bool(in_local.any().item()):
            return local_kl_numer, local_kl_count

        local_indices_kept = local_indices[in_local].to(device=device, dtype=torch.long)
        topk_ids = topk_ids_tensor[0, in_local, :].to(device=device, dtype=torch.long)
        teacher_probs = topk_probs_tensor[0, in_local, :].to(device=device, dtype=torch.float32)
        teacher_tail = tail_tensor[0, in_local].to(device=device, dtype=torch.float32)
        usable = int(local_indices_kept.numel())
        if usable <= 0:
            return local_kl_numer, local_kl_count

        chunk_size = int(self.config.get("teacher_kl_compute_chunk_size", 64) or 64)
        chunk_size = max(int(chunk_size), 1)
        kl_temperature = float(self.config.get("kl_temperature", 1.0))
        if kl_temperature <= 0.0:
            raise ValueError(f"kl_temperature must be > 0, got {kl_temperature}")

        for start in range(0, usable, chunk_size):
            end = min(start + chunk_size, usable)
            current_logits = local_logits[0, local_indices_kept[start:end], :].float() / kl_temperature
            current_log_denom = torch.logsumexp(current_logits, dim=-1)
            current_topk_logits = torch.gather(current_logits, dim=-1, index=topk_ids[start:end])
            student_topk_log_probs = current_topk_logits - current_log_denom.unsqueeze(-1)
            student_topk_probs = student_topk_log_probs.exp()
            student_tail_probs = torch.clamp(1.0 - student_topk_probs.sum(dim=-1), min=1.0e-8)

            teacher_topk = torch.clamp(teacher_probs[start:end], min=0.0)
            teacher_tail_mass = torch.clamp(teacher_tail[start:end], min=0.0)
            teacher_norm = torch.clamp(teacher_topk.sum(dim=-1) + teacher_tail_mass, min=1.0e-8)
            teacher_topk = teacher_topk / teacher_norm.unsqueeze(-1)
            teacher_tail_mass = teacher_tail_mass / teacher_norm

            kl_topk = teacher_topk * (
                torch.log(torch.clamp(teacher_topk, min=1.0e-8)) - student_topk_log_probs
            )
            kl_tail = teacher_tail_mass * (
                torch.log(torch.clamp(teacher_tail_mass, min=1.0e-8)) - torch.log(student_tail_probs)
            )
            local_kl_numer = local_kl_numer + kl_topk.sum() + kl_tail.sum()
            local_kl_count = local_kl_count + float(end - start)

        return local_kl_numer, local_kl_count

    def _forward_no_latent_cp_dense_ce(self, batch: Dict[str, Any]) -> SimpleNamespace:
        input_ids = batch["input_ids"]
        position_ids = batch["position_ids"]
        labels = batch["labels"]
        loss_weights = batch["loss_weights"]
        loss_source_positions = batch["loss_source_positions"]
        loss_target_positions = batch["loss_target_positions"]
        loss_pair_mask = batch["loss_pair_mask"].to(torch.bool)
        answer_mask = batch["answer_mask"].to(torch.bool)
        cot_mask = batch["cot_mask"].to(torch.bool)
        inner_cot_mask = compute_inner_cot_mask(cot_mask)
        device = input_ids.device

        cp_input_ids = input_ids
        cp_position_ids = position_ids
        with self.accelerator.maybe_context_parallel(
            buffers=[cp_input_ids, cp_position_ids],
            buffer_seq_dims=[1, 1],
            no_restore_buffers={cp_input_ids, cp_position_ids},
        ):
            outputs = self.model(
                input_ids=cp_input_ids,
                attention_mask=None,
                position_ids=cp_position_ids,
                use_cache=False,
                return_dict=True,
            )
            local_logits = self._to_local_tensor(outputs.logits)
            local_position_ids = self._to_local_tensor(cp_position_ids).to(device=device, dtype=torch.long)

            if int(local_position_ids.size(0)) != 1:
                raise ValueError(
                    "No-latent CP dense CE currently requires global batch size 1, "
                    f"got local_position_ids_shape={list(local_position_ids.shape)}"
                )
            local_positions = local_position_ids[0]
            if int(local_positions.numel()) <= 0:
                raise ValueError("No-latent CP received an empty local sequence shard.")
            position_runs = self._build_cp_local_position_runs(local_positions)
            self._maybe_log_cp_position_runs(local_positions=local_positions, runs=position_runs)

            source_positions = loss_source_positions[0].to(device=device, dtype=torch.long)
            target_positions = loss_target_positions[0].to(device=device, dtype=torch.long)
            pair_valid = loss_pair_mask[0].to(device=device)
            in_local, local_indices = self._map_global_positions_to_cp_local_indices(
                source_positions=source_positions,
                pair_valid=pair_valid & (source_positions >= 0),
                runs=position_runs,
            )
            if bool(in_local.any().item()):
                local_indices_kept = local_indices[in_local].to(torch.long)
                target_positions_kept = target_positions[in_local].clamp(
                    min=0,
                    max=max(int(labels.size(1)) - 1, 0),
                )
                target_labels = labels[0, target_positions_kept].to(device=device, dtype=torch.long)
                target_weights = loss_weights[0, target_positions_kept].to(device=device, dtype=torch.float32)
                keep = target_labels != -100
            else:
                local_indices_kept = torch.empty((0,), device=device, dtype=torch.long)
                target_positions_kept = torch.empty((0,), device=device, dtype=torch.long)
                target_labels = torch.empty((0,), device=device, dtype=torch.long)
                target_weights = torch.empty((0,), device=device, dtype=torch.float32)
                keep = torch.empty((0,), device=device, dtype=torch.bool)

            zero_anchor = local_logits.sum() * 0.0
            local_cot_numer = zero_anchor.float()
            local_cot_count = torch.zeros((), device=device, dtype=torch.float32)
            local_non_cot_numer = zero_anchor.float()
            local_non_cot_denom = torch.zeros((), device=device, dtype=torch.float32)
            local_answer_numer = zero_anchor.float()
            local_answer_denom = torch.zeros((), device=device, dtype=torch.float32)
            cot_weight = batch["cot_branch_weight"][0].to(device=device, dtype=torch.float32)
            local_kl_numer, local_kl_count = self._compute_cp_teacher_kl_from_local_logits(
                local_logits=local_logits,
                position_runs=position_runs,
                batch=batch,
                zero_anchor=zero_anchor,
            )

            if bool(keep.any().item()):
                selected_logits = local_logits[0, local_indices_kept[keep], :].float()
                token_losses = F.cross_entropy(selected_logits, target_labels[keep], reduction="none")
                selected_target_positions = target_positions_kept[keep]
                cot_keep = inner_cot_mask[0, selected_target_positions].to(device=device)
                non_cot_keep = ~cot_keep
                if bool(cot_keep.any().item()):
                    local_cot_numer = local_cot_numer + token_losses[cot_keep].sum()
                    local_cot_count = local_cot_count + cot_keep.to(torch.float32).sum()
                if bool(non_cot_keep.any().item()):
                    selected_weights = target_weights[keep][non_cot_keep]
                    local_non_cot_numer = local_non_cot_numer + (token_losses[non_cot_keep] * selected_weights).sum()
                    local_non_cot_denom = local_non_cot_denom + selected_weights.sum()
                answer_keep = answer_mask[0, selected_target_positions].to(device=device)
                if bool(answer_keep.any().item()):
                    local_answer_numer = local_answer_numer + token_losses[answer_keep].sum()
                    local_answer_denom = local_answer_denom + answer_keep.to(torch.float32).sum()

            global_cot_numer = self._autograd_all_reduce_sum(local_cot_numer)
            global_cot_count = self._autograd_all_reduce_sum(local_cot_count)
            global_non_cot_numer = self._autograd_all_reduce_sum(local_non_cot_numer)
            global_non_cot_denom = self._autograd_all_reduce_sum(local_non_cot_denom)
            global_answer_numer = self._autograd_all_reduce_sum(local_answer_numer)
            global_answer_denom = self._autograd_all_reduce_sum(local_answer_denom)
            global_kl_numer = self._autograd_all_reduce_sum(local_kl_numer)
            global_kl_count = self._autograd_all_reduce_sum(local_kl_count)
            local_chunk_len = int(local_positions.numel())
            del local_logits
            del outputs

        cot_ce = global_cot_numer / torch.clamp(global_cot_count, min=1.0)
        non_cot_ce = global_non_cot_numer / torch.clamp(global_non_cot_denom, min=1.0)
        ce_loss = cot_weight * cot_ce + non_cot_ce
        answer_ce = global_answer_numer / torch.clamp(global_answer_denom, min=1.0)
        kl_loss = global_kl_numer / torch.clamp(global_kl_count, min=1.0)
        return SimpleNamespace(
            ce_loss=ce_loss,
            cot_ce=cot_ce,
            non_cot_ce=non_cot_ce,
            answer_ce=answer_ce,
            kl_loss=kl_loss,
            ce_token_weight=global_non_cot_denom.detach(),
            cot_token_count=global_cot_count.detach(),
            answer_token_count=global_answer_denom.detach(),
            kl_positions=global_kl_count.detach(),
            chunked_forward_info={
                "chunk_count": int(self._context_parallel_size()),
                "forward_chunk_size": int(local_chunk_len),
                "max_chunk_active_tokens": int(batch["attention_mask"].sum(dim=1).max().item()),
                "gc_temporarily_disabled": 0,
                "past_key_values_detached": 0,
                "full_vocab_logits_materialized": 1,
                "context_parallel_size": int(self._context_parallel_size()),
                "teacher_kl_enabled": 1,
                "sparse_lm_head_enabled": 0,
            },
        )

    @staticmethod
    def _sample_from_logits(
        logits: torch.Tensor,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> int:
        if logits.dim() != 2:
            raise ValueError(f"_sample_from_logits expects rank-2 logits, got shape={list(logits.shape)}")
        if not bool(do_sample):
            return int(torch.argmax(logits, dim=-1).item())
        if float(temperature) <= 0.0:
            raise ValueError(f"temperature must be > 0 for sampling, got temperature={float(temperature)}")

        filtered_logits = logits.float() / float(temperature)
        vocab_size = int(filtered_logits.size(-1))

        if int(top_k) > 0 and int(top_k) < vocab_size:
            topk_values, _ = torch.topk(filtered_logits, k=int(top_k), dim=-1)
            kth_values = topk_values[:, -1].unsqueeze(-1)
            filtered_logits = filtered_logits.masked_fill(filtered_logits < kth_values, float("-inf"))

        if float(top_p) < 1.0:
            if float(top_p) <= 0.0:
                raise ValueError(f"top_p must be > 0 when enabled, got top_p={float(top_p)}")
            sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_remove = cumulative_probs > float(top_p)
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove_mask = torch.zeros_like(sorted_remove, dtype=torch.bool)
            remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_remove)
            filtered_logits = filtered_logits.masked_fill(remove_mask, float("-inf"))

        probs = torch.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return int(next_token.item())

    def _maybe_log_alignment_debug(
        self,
        batch: Dict[str, Any],
        metrics: Dict[str, Any],
        global_step: int,
        epoch_index: int,
        tag: str,
    ) -> None:
        del batch, metrics, global_step, epoch_index, tag
        return

    def _pad_token_ids_for_cp_generation(
        self,
        token_ids: List[int],
        pad_token_id: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        real_len = int(len(token_ids))
        if real_len <= 0:
            raise ValueError("CP eval generation requires a non-empty prompt/current sequence.")
        required_multiple = int(resolve_context_parallel_padding_multiple(self.config))
        padded_len = int(real_len)
        if required_multiple > 1:
            remainder = int(padded_len % required_multiple)
            if remainder != 0:
                padded_len += int(required_multiple - remainder)
        if required_multiple > 1 and padded_len % required_multiple != 0:
            raise RuntimeError(
                "Failed to pad CP eval generation input to the required multiple: "
                f"real_len={real_len}, padded_len={padded_len}, required_multiple={required_multiple}."
            )

        padded_ids = list(int(token_id) for token_id in token_ids)
        if padded_len > real_len:
            padded_ids.extend([int(pad_token_id)] * int(padded_len - real_len))
        input_ids = torch.tensor([padded_ids], device=device, dtype=torch.long)
        attention_mask = torch.zeros((1, padded_len), device=device, dtype=torch.long)
        attention_mask[:, :real_len] = 1
        position_ids = torch.arange(padded_len, device=device, dtype=torch.long).unsqueeze(0)
        return input_ids, attention_mask, position_ids, real_len

    def _maybe_log_cp_generation_padding(
        self,
        prompt_len: int,
        padded_len: int,
        attention_mask_sum: int,
    ) -> None:
        if self._cp_generation_debug_logged:
            return
        self._cp_generation_debug_logged = True
        if not self.accelerator.is_main_process:
            return
        padding_factor = max(int(self.config.get("context_parallel_padding_factor", 2) or 2), 1)
        self.logger.info(
            {
                "tag": "cp_eval_generation_padding",
                "prompt_len": int(prompt_len),
                "padded_len": int(padded_len),
                "required_multiple": int(resolve_context_parallel_padding_multiple(self.config)),
                "context_parallel_size": int(self._context_parallel_size()),
                "context_parallel_padding_factor": int(padding_factor),
                "attention_mask_sum": int(attention_mask_sum),
            }
        )

    @torch.no_grad()
    def _generate_cooperative_single_long_sample(
        self,
        prompt_ids: List[int],
        token_constants: Dict[str, int],
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> tuple[List[int], bool]:
        if max_new_tokens <= 0:
            return [], False

        device = self.accelerator.device
        eos_token_id = int(token_constants["im_end_id"])
        pad_token_id = int(getattr(self.tokenizer, "pad_token_id", eos_token_id) or eos_token_id)
        model_for_generate = self.model
        rank = int(dist.get_rank()) if (dist.is_available() and dist.is_initialized()) else 0
        generated: List[int] = []
        stopped_normally = False

        for _step_index in range(int(max_new_tokens)):
            current_token_ids = [int(token_id) for token_id in prompt_ids] + generated
            cp_input_ids, cp_attention_mask, cp_position_ids, real_len = self._pad_token_ids_for_cp_generation(
                token_ids=current_token_ids,
                pad_token_id=pad_token_id,
                device=device,
            )
            self._maybe_log_cp_generation_padding(
                prompt_len=int(len(prompt_ids)),
                padded_len=int(cp_input_ids.size(1)),
                attention_mask_sum=int(cp_attention_mask.sum().item()),
            )

            with self.accelerator.maybe_context_parallel(
                buffers=[cp_input_ids, cp_attention_mask, cp_position_ids],
                buffer_seq_dims=[1, 1, 1],
                no_restore_buffers={cp_input_ids, cp_attention_mask, cp_position_ids},
            ):
                outputs = model_for_generate(
                    input_ids=cp_input_ids,
                    attention_mask=cp_attention_mask,
                    position_ids=cp_position_ids,
                    use_cache=False,
                    return_dict=True,
                )
                local_logits = self._to_local_tensor(outputs.logits)
                local_position_ids = self._to_local_tensor(cp_position_ids).to(device=device, dtype=torch.long)
                if int(local_position_ids.size(0)) != 1:
                    raise ValueError(
                        "No-latent CP eval generation requires global batch size 1, "
                        f"got local_position_ids_shape={list(local_position_ids.shape)}"
                    )
                last_real_position = int(real_len - 1)
                owner_indices = torch.nonzero(
                    local_position_ids[0] == int(last_real_position),
                    as_tuple=False,
                ).flatten()
                if int(owner_indices.numel()) > 0:
                    local_next_logits = local_logits[:, int(owner_indices[0].item()), :].float()
                    owner_count = torch.ones((), device=device, dtype=torch.float32)
                else:
                    if int(local_logits.size(1)) <= 0:
                        raise RuntimeError("CP eval generation received an empty local logits shard.")
                    local_next_logits = torch.zeros_like(local_logits[:, 0, :], dtype=torch.float32)
                    owner_count = torch.zeros((), device=device, dtype=torch.float32)

                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(local_next_logits, op=dist.ReduceOp.SUM)
                    dist.all_reduce(owner_count, op=dist.ReduceOp.SUM)
                if float(owner_count.item()) <= 0.0:
                    raise RuntimeError(
                        "CP eval generation could not locate the final real token position on any rank: "
                        f"last_real_position={last_real_position}, real_len={real_len}, "
                        f"padded_len={int(cp_input_ids.size(1))}."
                    )
                next_logits = local_next_logits / torch.clamp(owner_count, min=1.0)

            if rank == 0:
                next_token = self._sample_from_logits(
                    next_logits,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                done = int(next_token == eos_token_id or len(generated) + 1 >= int(max_new_tokens))
            else:
                next_token = 0
                done = 0
            next_token = self._broadcast_int_from_rank0(int(next_token), device=device)
            done = self._broadcast_int_from_rank0(int(done), device=device)
            generated.append(int(next_token))
            if int(next_token) == eos_token_id:
                stopped_normally = True
            if int(done):
                break

        return generated, stopped_normally

    def _forward_loss(
        self,
        batch: Dict[str, Any],
        global_step: int = -1,
        epoch_index: int = -1,
        train_stage: int | None = None,
        train_progress: float | None = None,
    ) -> Dict[str, Any]:
        del global_step, epoch_index, train_stage, train_progress
        self._assert_same_record_id_across_ranks(batch=batch, phase="pre_forward")
        self._validate_cp_batch(batch)
        batch = self._attach_teacher_kl_batch(batch)
        batch = self._trim_teacher_kl_dim(batch)
        cp_outputs = self._forward_no_latent_cp_dense_ce(batch=batch)
        chunked_forward_info = getattr(cp_outputs, "chunked_forward_info", {})
        if float(cp_outputs.kl_positions.item()) <= 0.0 and not self._cp_kl_zero_warning_logged:
            self._cp_kl_zero_warning_logged = True
            if self.accelerator.is_main_process:
                self.logger.warning(
                    {
                        "tag": "cp_teacher_kl_zero_positions",
                        "record_ids": list(batch.get("record_ids", [])),
                        "teacher_kl_pair_slots": int(batch.get("teacher_kl_pair_mask").size(1))
                        if batch.get("teacher_kl_pair_mask") is not None
                        else 0,
                        "teacher_kl_effective_slots": int(
                            batch.get("teacher_kl_effective_mask").to(torch.long).sum().item()
                        )
                        if torch.is_tensor(batch.get("teacher_kl_effective_mask"))
                        else 0,
                    }
                )
        self.logger.info(
            {
                "tag": "loss_chunk_summary",
                "distributed_backend": str(self.distributed_backend),
                "loss_pair_slots": int(batch.get("loss_pair_mask").size(1))
                if batch.get("loss_pair_mask") is not None
                else 0,
                "teacher_kl_pair_slots": int(batch.get("teacher_kl_pair_mask").size(1))
                if batch.get("teacher_kl_pair_mask") is not None
                else 0,
                "chunk_count": int(chunked_forward_info.get("chunk_count", 1)),
                "forward_chunk_size": int(chunked_forward_info.get("forward_chunk_size", int(batch["input_ids"].size(1)))),
                "max_chunk_active_tokens": int(
                    chunked_forward_info.get(
                        "max_chunk_active_tokens",
                        int(batch["attention_mask"].sum(dim=1).max().item()),
                    )
                ),
                "gc_temporarily_disabled": int(chunked_forward_info.get("gc_temporarily_disabled", 0)),
                "past_key_values_detached": int(chunked_forward_info.get("past_key_values_detached", 0)),
                "full_vocab_logits_materialized": int(chunked_forward_info.get("full_vocab_logits_materialized", 1)),
                "context_parallel_size": int(chunked_forward_info.get("context_parallel_size", 1)),
                "teacher_kl_enabled": 1,
                "kl_loss_weight": float(self.config["kl_loss_weight"]),
                "kl_positions": float(cp_outputs.kl_positions.item()),
                "kl_loss": float(cp_outputs.kl_loss.detach().item()),
                "sparse_lm_head_enabled": 0,
                "ce_token_weight": float(cp_outputs.ce_token_weight.item()),
                "answer_token_count": float(cp_outputs.answer_token_count.item()),
            }
        )
        zero = batch["input_ids"].new_zeros((), dtype=torch.float32)
        total_loss = cp_outputs.ce_loss + float(self.config["kl_loss_weight"]) * cp_outputs.kl_loss
        outputs = SimpleNamespace(logits=None)
        return {
            "loss": total_loss,
            "halt_dense_loss": zero,
            "latent_stop_loss": zero,
            "stop_loss": zero,
            "halt_dense_loss_weight": zero,
            "early_exit_rank_loss": zero,
            "early_exit_argmax_violation_rate": zero,
            "early_exit_front_rank_loss": zero,
            "early_exit_nonfront_rank_loss": zero,
            "latent_end_soft_loss": zero,
            "other_end_hard_loss": zero,
            "latent_end_target_mean": zero,
            "latent_end_score_mean": zero,
            "latent_end_front_score_mean": zero,
            "latent_end_tail_score_mean": zero,
            "ce_loss": cp_outputs.ce_loss.detach(),
            "cot_ce": cp_outputs.cot_ce.detach(),
            "non_cot_ce": cp_outputs.non_cot_ce.detach(),
            "answer_ce": cp_outputs.answer_ce.detach(),
            "kl_loss": cp_outputs.kl_loss.detach(),
            "kl_positions": cp_outputs.kl_positions.detach(),
            "outputs": outputs,
            "ce_metrics": {
                "ce_loss": cp_outputs.ce_loss,
                "cot_ce": cp_outputs.cot_ce,
                "non_cot_ce": cp_outputs.non_cot_ce,
                "token_ce": torch.zeros_like(batch["loss_weights"], dtype=torch.float32),
                "weighted_token_ce": torch.zeros_like(batch["loss_weights"], dtype=torch.float32),
            },
            "halt_dense_metrics": {"halt_dense_loss": zero},
            "kl_stats": {},
        }

    @staticmethod
    def _count_generated_cot_tokens(token_ids: List[int], token_constants: Dict[str, int]) -> int:
        think_start_id = int(token_constants["think_start_id"])
        think_end_id = int(token_constants["think_end_id"])
        in_cot = False
        cot_tokens = 0
        for token_id in token_ids:
            if int(token_id) == think_start_id:
                in_cot = True
                continue
            if int(token_id) == think_end_id:
                in_cot = False
                continue
            if in_cot:
                cot_tokens += 1
        return int(cot_tokens)

    @torch.no_grad()
    def evaluate(self, global_step: int) -> None:
        val_loader = self._make_val_loader()
        val_loader = self._prepare_dataloader(val_loader, purpose="eval")
        self.model.eval()
        token_constants = self.val_dataset.token_constants
        rank = int(dist.get_rank()) if (dist.is_available() and dist.is_initialized()) else 0
        world_size = int(dist.get_world_size()) if (dist.is_available() and dist.is_initialized()) else 1

        try:
            num_val_batches_local = int(len(val_loader))
        except TypeError:
            num_val_batches_local = 0
        if num_val_batches_local <= 0:
            if self.accelerator.is_main_process:
                self.logger.warning({"tag": "eval_skipped", "reason": "empty_val_loader"})
            self.model.train()
            return

        cursor_device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        num_batches_tensor = torch.tensor([int(num_val_batches_local)], device=cursor_device, dtype=torch.long)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.MIN)
        num_val_batches = int(num_batches_tensor.item())
        if num_val_batches <= 0:
            if self.accelerator.is_main_process:
                self.logger.warning({"tag": "eval_skipped", "reason": "empty_val_loader_after_rank_min"})
            self.model.train()
            return

        if self.accelerator.is_main_process:
            base_batch_cursor = int(self._eval_batch_cursor % num_val_batches)
            next_batch_cursor = int(self._eval_batch_cursor + 1)
        else:
            base_batch_cursor = 0
            next_batch_cursor = 0
        base_batch_cursor = self._broadcast_int_from_rank0(base_batch_cursor, device=cursor_device)
        next_batch_cursor = self._broadcast_int_from_rank0(next_batch_cursor, device=cursor_device)
        self._eval_batch_cursor = int(next_batch_cursor)
        target_batch_index = int(base_batch_cursor % num_val_batches)

        selected_batch: Dict[str, Any] | None = None
        for batch_index, batch in enumerate(val_loader):
            if batch_index == target_batch_index:
                selected_batch = batch
                break
        if selected_batch is None:
            for batch in val_loader:
                selected_batch = batch
                break
        if selected_batch is None:
            if self.accelerator.is_main_process:
                self.logger.warning({"tag": "eval_skipped", "reason": "target_batch_not_found"})
            self.model.train()
            return
        self._assert_same_record_id_across_ranks(batch=selected_batch, phase="eval_selected")
        self._validate_cp_batch(selected_batch)

        metrics_forward = self._forward_loss(
            selected_batch,
            global_step=global_step,
            epoch_index=-1,
            train_progress=1.0,
        )
        local_forward = torch.tensor(
            [
                float(metrics_forward["loss"].item()),
                float(metrics_forward["ce_loss"].item()),
                float(metrics_forward["kl_loss"].item()),
                float(metrics_forward["answer_ce"].item()),
            ],
            device=selected_batch["input_ids"].device,
            dtype=torch.float32,
        )
        reduced_forward = self._all_reduce_mean_tensor(local_forward)

        batch_size = len(selected_batch["prompt_only_ids"])
        batch_size_tensor = torch.tensor([int(batch_size)], device=selected_batch["input_ids"].device, dtype=torch.long)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_size_tensor, op=dist.ReduceOp.MIN)
        effective_batch_size = int(batch_size_tensor.item())
        if effective_batch_size <= 0:
            if self.accelerator.is_main_process:
                self.logger.warning({"tag": "eval_skipped", "reason": "empty_eval_batch"})
            self.model.train()
            return

        if self.accelerator.is_main_process:
            base_sample_cursor = int(self._eval_sample_cursor % effective_batch_size)
            next_sample_cursor = int(self._eval_sample_cursor + 1)
        else:
            base_sample_cursor = 0
            next_sample_cursor = 0
        base_sample_cursor = self._broadcast_int_from_rank0(base_sample_cursor, device=selected_batch["input_ids"].device)
        next_sample_cursor = self._broadcast_int_from_rank0(next_sample_cursor, device=selected_batch["input_ids"].device)
        self._eval_sample_cursor = int(next_sample_cursor)

        sample_index = int(base_sample_cursor % effective_batch_size)

        prompt_ids = selected_batch["prompt_only_ids"][sample_index]
        prompt_token_ids = self._token_ids_to_list(prompt_ids)
        prompt_token_ids = self._broadcast_token_ids_from_rank0(
            token_ids=prompt_token_ids,
            device=selected_batch["input_ids"].device,
        )
        prompt_text = self.tokenizer.decode(prompt_token_ids, skip_special_tokens=False)
        gold_answer = str(selected_batch["assistant_answers"][sample_index])
        record_id = str(selected_batch["record_ids"][sample_index])
        sample_spans = selected_batch["spans"][sample_index]
        original_cot_count = max(int(sample_spans["think_end"]) - int(sample_spans["think_start"]) - 1, 0)
        original_total_count = int(selected_batch["attention_mask"][sample_index].sum().item())

        generated_token_ids, _ = self._generate_cooperative_single_long_sample(
            prompt_ids=prompt_token_ids,
            token_constants=token_constants,
            max_new_tokens=self._resolve_eval_max_new_tokens(global_step=global_step),
            do_sample=bool(self.config.get("eval_do_sample", False)),
            temperature=float(self.config.get("eval_temperature", 1.0)),
            top_p=float(self.config.get("eval_top_p", 1.0)),
            top_k=int(self.config.get("eval_top_k", 0)),
        )
        generated_token_ids = self._token_ids_to_list(generated_token_ids)
        decoded = self.tokenizer.decode(generated_token_ids, skip_special_tokens=False)
        generated_cot_tokens = self._count_generated_cot_tokens(generated_token_ids, token_constants)
        malformed_flag = float(("</think>" not in decoded) or ("<|im_end|>" not in decoded))
        local_generation = {
            "rank": int(rank),
            "world_size": int(world_size),
            "batch_index": int(target_batch_index),
            "sample_index": int(sample_index),
            "record_id": str(record_id),
            "prompt_text": str(prompt_text),
            "ground_truth": str(gold_answer),
            "generated_text": str(decoded),
            "prompt_token_ids": list(prompt_token_ids),
            "generated_token_ids": list(generated_token_ids),
            "original_cot_tokens": float(original_cot_count),
            "original_total_tokens": float(original_total_count),
            "generated_cot_tokens": float(generated_cot_tokens),
            "generated_total_tokens": float(len(generated_token_ids)),
            "exact_match": float(decoded == gold_answer),
            "malformed_rate": float(malformed_flag),
        }
        gathered_generation = [local_generation]

        def _mean_from_gathered(key: str) -> float:
            return mean_or_zero([float(item.get(key, 0.0)) for item in gathered_generation])

        metrics = {
            "val/loss": float(reduced_forward[0].item()),
            "val/ce_loss": float(reduced_forward[1].item()),
            "val/kl_loss": float(reduced_forward[2].item()),
            "val/answer_ce": float(reduced_forward[3].item()),
            "val/orig_cot_tokens": _mean_from_gathered("original_cot_tokens"),
            "val/orig_total_tokens": _mean_from_gathered("original_total_tokens"),
            "val/gen_cot_tokens": _mean_from_gathered("generated_cot_tokens"),
            "val/gen_total_tokens": _mean_from_gathered("generated_total_tokens"),
            "val/exact_match": _mean_from_gathered("exact_match"),
            "val/malformed_rate": _mean_from_gathered("malformed_rate"),
            "val/eval_batch_index": float(base_batch_cursor),
            "val/eval_sample_index": float(base_sample_cursor),
            "val/eval_world_size": float(world_size),
            "step": global_step,
        }
        if self.accelerator.is_main_process:
            detail = gathered_generation[0]
            self._log_eval_generated_text(
                global_step=global_step,
                batch_index=int(detail.get("batch_index", 0)),
                sample_index=int(detail.get("sample_index", 0)),
                record_id=str(detail.get("record_id", "")),
                generated_latent_tokens=0,
                generated_cot_tokens=int(detail.get("generated_cot_tokens", 0)),
                original_latent_tokens=0,
                original_cot_tokens=int(detail.get("original_cot_tokens", 0)),
                generated_text=str(detail.get("generated_text", "")),
            )
            self._log_eval_sample_text_bundle(
                global_step=global_step,
                batch_index=int(detail.get("batch_index", 0)),
                sample_index=int(detail.get("sample_index", 0)),
                record_id=str(detail.get("record_id", "")),
                generated_latent_tokens=0,
                generated_cot_tokens=int(detail.get("generated_cot_tokens", 0)),
                original_latent_tokens=0,
                original_cot_tokens=int(detail.get("original_cot_tokens", 0)),
                prompt_token_ids=list(detail.get("prompt_token_ids", [])),
                prompt_text=str(detail.get("prompt_text", "")),
                ground_truth=str(detail.get("ground_truth", "")),
                generated_token_ids=list(detail.get("generated_token_ids", [])),
                generated_text=str(detail.get("generated_text", "")),
            )
            self.logger.info(metrics)
            self._log_eval_scalars(metrics, global_step=global_step)
        self.model.train()
