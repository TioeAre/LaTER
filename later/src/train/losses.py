from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from later.src.train.utils import bytes_to_gib, tensor_nbytes


def compute_inner_cot_mask(cot_mask: torch.Tensor | None) -> torch.Tensor | None:
    if cot_mask is None:
        return None
    cot_mask_bool = cot_mask.to(dtype=torch.bool)
    if cot_mask_bool.dim() != 2:
        raise ValueError(f"cot_mask must be rank-2 [batch, seq_len], got {cot_mask_bool.dim()}")

    batch_size, seq_len = cot_mask_bool.shape
    if batch_size <= 0 or seq_len <= 0:
        return torch.zeros_like(cot_mask_bool, dtype=torch.bool)

    true_counts = cot_mask_bool.to(torch.long).sum(dim=1)
    has_inner = true_counts >= 3
    first_true = torch.argmax(cot_mask_bool.to(torch.long), dim=1)
    reversed_first_true = torch.argmax(torch.flip(cot_mask_bool, dims=[1]).to(torch.long), dim=1)
    last_true = (seq_len - 1) - reversed_first_true
    positions = torch.arange(seq_len, device=cot_mask_bool.device).unsqueeze(0).expand(batch_size, -1)
    return cot_mask_bool & has_inner.unsqueeze(1) & (positions > first_true.unsqueeze(1)) & (
        positions < last_true.unsqueeze(1)
    )


def _finalize_split_ce(
    token_ce: torch.Tensor,
    weighted_token_ce: torch.Tensor,
    supervised_mask: torch.Tensor,
    loss_weights: torch.Tensor,
    cot_mask: torch.Tensor | None,
    cot_branch_weight: torch.Tensor | None,
) -> Dict[str, torch.Tensor]:
    if cot_mask is None:
        cot_mask = torch.zeros_like(supervised_mask, dtype=torch.bool)
    else:
        cot_mask = cot_mask.to(device=supervised_mask.device, dtype=torch.bool)
        if cot_mask.shape != supervised_mask.shape:
            raise ValueError(
                "cot_mask shape does not match supervised positions: "
                f"cot_mask_shape={list(cot_mask.shape)} supervised_shape={list(supervised_mask.shape)}"
            )

    if cot_branch_weight is None:
        cot_branch_weight = torch.ones((token_ce.size(0),), device=token_ce.device, dtype=torch.float32)
    else:
        cot_branch_weight = cot_branch_weight.to(device=token_ce.device, dtype=torch.float32).view(-1)
        if int(cot_branch_weight.numel()) != int(token_ce.size(0)):
            raise ValueError(
                "cot_branch_weight batch dimension does not match token CE batch size: "
                f"cot_branch_weight_shape={list(cot_branch_weight.shape)} token_ce_shape={list(token_ce.shape)}"
            )

    inner_cot_mask = compute_inner_cot_mask(cot_mask)
    if inner_cot_mask is None:
        inner_cot_mask = torch.zeros_like(supervised_mask, dtype=torch.bool)

    cot_valid = supervised_mask & inner_cot_mask
    non_cot_valid = supervised_mask & (~inner_cot_mask)

    cot_token_count = cot_valid.to(torch.float32).sum()
    cot_ce = (token_ce * cot_valid.to(token_ce.dtype)).sum() / torch.clamp(cot_token_count, min=1.0)

    cot_weight_matrix = cot_branch_weight.unsqueeze(1).expand_as(token_ce)
    cot_scaled_numer = (token_ce * cot_valid.to(token_ce.dtype) * cot_weight_matrix).sum()
    cot_scaled_ce = cot_scaled_numer / torch.clamp(cot_token_count, min=1.0)

    non_cot_weights = loss_weights.to(device=token_ce.device, dtype=torch.float32) * non_cot_valid.to(torch.float32)
    non_cot_token_weight = non_cot_weights.sum()
    non_cot_ce = (token_ce * non_cot_weights.to(token_ce.dtype)).sum() / torch.clamp(non_cot_token_weight, min=1.0)

    weighted_token_ce = weighted_token_ce.clone()
    weighted_token_ce[cot_valid] = (token_ce * cot_weight_matrix)[cot_valid]

    cot_weight_mean = cot_branch_weight.mean() if int(cot_branch_weight.numel()) > 0 else token_ce.new_ones(())
    return {
        "ce_loss": cot_scaled_ce + non_cot_ce,
        "cot_ce": cot_ce,
        "cot_scaled_ce": cot_scaled_ce,
        "non_cot_ce": non_cot_ce,
        "token_ce": token_ce,
        "weighted_token_ce": weighted_token_ce,
        "cot_token_count": cot_token_count,
        "non_cot_token_weight": non_cot_token_weight,
        "cot_weight": cot_weight_mean,
    }


def compute_weighted_ce(
    logits: torch.Tensor | None,
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
    cot_mask: torch.Tensor | None = None,
    cot_branch_weight: torch.Tensor | None = None,
    loss_source_positions: torch.Tensor | None = None,
    loss_target_positions: torch.Tensor | None = None,
    loss_pair_mask: torch.Tensor | None = None,
    hidden_states: torch.Tensor | None = None,
    lm_head: Any | None = None,
    logits_chunk_size: int = 0,
    selected_target_logits: torch.Tensor | None = None,
    selected_log_denom: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    token_ce = labels.new_zeros(labels.shape, dtype=torch.float32)
    weighted_token_ce = labels.new_zeros(labels.shape, dtype=torch.float32)
    supervised_mask = torch.zeros(labels.shape, device=labels.device, dtype=torch.bool)

    if selected_target_logits is not None and selected_log_denom is not None:
        if loss_target_positions is None or loss_pair_mask is None:
            raise ValueError("compute_weighted_ce requires sparse target positions/mask for selected logits")

        batch_size = int(labels.size(0))
        pair_slots = int(loss_target_positions.size(1))
        row_ids = torch.arange(batch_size, device=labels.device).unsqueeze(1)
        total_weight = loss_weights.new_zeros((), dtype=torch.float32)
        chunk_size = pair_slots if int(logits_chunk_size) <= 0 else int(logits_chunk_size)
        chunk_size = max(chunk_size, 1)
        for start in range(0, pair_slots, chunk_size):
            end = min(start + chunk_size, pair_slots)
            target_pos_chunk = loss_target_positions[:, start:end]
            safe_target_cols = target_pos_chunk.clamp(min=0, max=max(int(labels.size(1)) - 1, 0))
            pair_rows_chunk = row_ids.expand(batch_size, end - start)
            gathered_labels = labels[pair_rows_chunk, safe_target_cols]
            gathered_weights = loss_weights[pair_rows_chunk, safe_target_cols].to(torch.float32)
            valid = loss_pair_mask[:, start:end] & (target_pos_chunk >= 0)
            keep = valid & (gathered_labels != -100)

            if not bool(keep.any().item()):
                continue

            rows = pair_rows_chunk[keep]
            cols = safe_target_cols[keep]
            logits_chunk = selected_target_logits[:, start:end]
            log_denom_chunk = selected_log_denom[:, start:end]
            token_loss = -(logits_chunk[keep].float() - log_denom_chunk[keep].float())
            token_ce[rows, cols] = token_loss
            weighted_token_ce[rows, cols] = token_loss * gathered_weights[keep]
            supervised_mask[rows, cols] = True
            total_weight = total_weight + gathered_weights[keep].sum()

        del total_weight
        return _finalize_split_ce(
            token_ce=token_ce,
            weighted_token_ce=weighted_token_ce,
            supervised_mask=supervised_mask,
            loss_weights=loss_weights,
            cot_mask=cot_mask,
            cot_branch_weight=cot_branch_weight,
        )

    if hidden_states is not None:
        if lm_head is None:
            raise ValueError("compute_weighted_ce requires lm_head when hidden_states is provided")
        if loss_source_positions is None or loss_target_positions is None or loss_pair_mask is None:
            raise ValueError("compute_weighted_ce requires sparse loss positions/mask when hidden_states is provided")

        batch_size = int(labels.size(0))
        pair_slots = int(loss_source_positions.size(1))
        pair_rows = torch.arange(batch_size, device=labels.device).unsqueeze(1).expand(batch_size, pair_slots)
        valid = loss_pair_mask & (loss_source_positions >= 0) & (loss_target_positions >= 0)

        safe_target_cols = loss_target_positions.clamp(min=0, max=max(int(labels.size(1)) - 1, 0))
        gathered_labels = labels[pair_rows, safe_target_cols]
        gathered_weights = loss_weights[pair_rows, safe_target_cols].to(torch.float32)
        keep = valid & (gathered_labels != -100)

        flat_hidden = hidden_states.reshape(batch_size * pair_slots, hidden_states.size(-1))
        flat_rows = pair_rows.reshape(-1)
        flat_target_cols = safe_target_cols.reshape(-1)
        flat_keep = keep.reshape(-1)
        flat_labels = gathered_labels.reshape(-1)
        flat_weights = gathered_weights.reshape(-1)
        safe_flat_labels = torch.where(
            flat_keep,
            flat_labels,
            torch.full_like(flat_labels, -100),
        )
        safe_flat_weights = torch.where(
            flat_keep,
            flat_weights,
            torch.zeros_like(flat_weights),
        )

        total_pairs = int(flat_hidden.size(0))
        chunk_size = total_pairs if int(logits_chunk_size) <= 0 else int(logits_chunk_size)
        chunk_size = max(chunk_size, 1)
        for start in range(0, total_pairs, chunk_size):
            end = min(start + chunk_size, total_pairs)
            chunk_logits = lm_head(flat_hidden[start:end])
            chunk_loss = F.cross_entropy(
                chunk_logits.float(),
                safe_flat_labels[start:end],
                reduction="none",
                ignore_index=-100,
            )
            chunk_keep = flat_keep[start:end]
            if bool(chunk_keep.any().item()):
                local_keep_idx = torch.nonzero(chunk_keep, as_tuple=False).view(-1)
                global_keep_idx = local_keep_idx + start
                rows = flat_rows[global_keep_idx]
                cols = flat_target_cols[global_keep_idx]
                selected_loss = chunk_loss[local_keep_idx]
                selected_weights = safe_flat_weights[global_keep_idx]
                token_ce[rows, cols] = selected_loss
                weighted_token_ce[rows, cols] = selected_loss * selected_weights
                supervised_mask[rows, cols] = True

        return _finalize_split_ce(
            token_ce=token_ce,
            weighted_token_ce=weighted_token_ce,
            supervised_mask=supervised_mask,
            loss_weights=loss_weights,
            cot_mask=cot_mask,
            cot_branch_weight=cot_branch_weight,
        )

    if logits is None:
        raise ValueError("compute_weighted_ce requires either sparse selected logits, hidden_states, or logits")

    if loss_source_positions is None or loss_target_positions is None or loss_pair_mask is None:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_weights = loss_weights[:, 1:].contiguous()

        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)
        weighted = token_loss * shift_weights
        token_ce[:, 1:] = token_loss
        weighted_token_ce[:, 1:] = weighted
        supervised_mask[:, 1:] = shift_labels != -100
        return _finalize_split_ce(
            token_ce=token_ce,
            weighted_token_ce=weighted_token_ce,
            supervised_mask=supervised_mask,
            loss_weights=loss_weights,
            cot_mask=cot_mask,
            cot_branch_weight=cot_branch_weight,
        )

    batch_index = torch.arange(logits.size(0), device=logits.device).unsqueeze(1).expand_as(loss_source_positions)
    valid = loss_pair_mask & (loss_source_positions >= 0) & (loss_target_positions >= 0)
    if bool(valid.any().item()):
        source_rows = batch_index[valid]
        source_cols = loss_source_positions[valid]
        target_cols = loss_target_positions[valid]
        selected_logits = logits[source_rows, source_cols, :]
        selected_labels = labels[source_rows, target_cols]
        selected_weights = loss_weights[source_rows, target_cols].to(torch.float32)
        keep = selected_labels != -100
        if bool(keep.any().item()):
            source_rows = source_rows[keep]
            source_cols = source_cols[keep]
            target_cols = target_cols[keep]
            selected_logits = selected_logits[keep]
            selected_labels = selected_labels[keep]
            selected_weights = selected_weights[keep]
            token_loss = F.cross_entropy(
                selected_logits.float(),
                selected_labels,
                reduction="none",
            )
            token_ce[source_rows, target_cols] = token_loss
            weighted_token_ce[source_rows, target_cols] = token_loss * selected_weights
            supervised_mask[source_rows, target_cols] = True

    return _finalize_split_ce(
        token_ce=token_ce,
        weighted_token_ce=weighted_token_ce,
        supervised_mask=supervised_mask,
        loss_weights=loss_weights,
        cot_mask=cot_mask,
        cot_branch_weight=cot_branch_weight,
    )


def compute_answer_ce(
    token_ce: torch.Tensor,
    answer_mask: torch.Tensor,
) -> torch.Tensor:
    if token_ce.shape == answer_mask.shape:
        mask = answer_mask.to(token_ce.dtype)
    else:
        mask = answer_mask[:, 1:].to(token_ce.dtype)
    denom = torch.clamp(mask.sum(), min=1.0)
    return (token_ce * mask).sum() / denom


def compute_early_exit_rank_loss(
    halt_dense_logits: torch.Tensor | None,
    halt_dense_best_allowed_logits: torch.Tensor | None,
    latent_internal_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    latent_pad_mask: torch.Tensor | None = None,
    front_fraction: float = 0.25,
    front_weight: float = 2.0,
    nonfront_weight: float = 1.0,
    margin: float = 0.0,
    soft_target_curve: str = "smoothstep",
    soft_target_temperature: float = 1.0,
    soft_target_power: float = 1.0,
    soft_loss_weight: float = 1.0,
    other_end_hard_loss_weight: float = 1.0,
    other_end_rank_margin: float = 0.0,
) -> Dict[str, torch.Tensor]:
    valid_mask = latent_internal_mask.to(torch.bool) & attention_mask.to(torch.bool)
    if latent_pad_mask is not None:
        if latent_pad_mask.shape != latent_internal_mask.shape:
            raise ValueError(
                "latent_pad_mask shape does not match latent_internal_mask: "
                f"latent_pad_mask_shape={list(latent_pad_mask.shape)}, latent_shape={list(latent_internal_mask.shape)}"
            )
        valid_mask = valid_mask & (~latent_pad_mask.to(torch.bool))

    zero = latent_internal_mask.new_zeros((), dtype=torch.float32)
    token_rank_loss = latent_internal_mask.new_zeros(latent_internal_mask.shape, dtype=torch.float32)
    position_weight = latent_internal_mask.new_zeros(latent_internal_mask.shape, dtype=torch.float32)
    front_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    argmax_violation_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    token_forbidden_minus_best_allowed = None
    latent_end_target = latent_internal_mask.new_zeros(latent_internal_mask.shape, dtype=torch.float32)
    latent_end_score = latent_internal_mask.new_zeros(latent_internal_mask.shape, dtype=torch.float32)
    token_latent_end_soft_loss = latent_internal_mask.new_zeros(latent_internal_mask.shape, dtype=torch.float32)
    token_other_end_hard_loss = latent_internal_mask.new_zeros(latent_internal_mask.shape, dtype=torch.float32)
    distance_to_latent_end = latent_internal_mask.new_zeros(latent_internal_mask.shape, dtype=torch.float32)
    progress_to_end = latent_internal_mask.new_zeros(latent_internal_mask.shape, dtype=torch.float32)
    latent_index_within_span = latent_internal_mask.new_full(latent_internal_mask.shape, -1, dtype=torch.long)

    if halt_dense_logits is None or halt_dense_best_allowed_logits is None:
        return {
            "halt_dense_loss": zero,
            "token_halt_dense_bce": token_rank_loss,
            "halt_dense_valid_mask": valid_mask,
            "token_rank_loss": token_rank_loss,
            "front_mask": front_mask,
            "position_weight": position_weight,
            "argmax_violation_mask": argmax_violation_mask,
            "front_loss": zero,
            "nonfront_loss": zero,
            "argmax_violation_rate": zero,
            "token_forbidden_minus_best_allowed": token_forbidden_minus_best_allowed,
            "latent_end_target": latent_end_target,
            "latent_end_score": latent_end_score,
            "token_latent_end_soft_loss": token_latent_end_soft_loss,
            "token_other_end_hard_loss": token_other_end_hard_loss,
            "distance_to_latent_end": distance_to_latent_end,
            "progress_to_end": progress_to_end,
            "latent_index_within_span": latent_index_within_span,
            "latent_end_soft_loss": zero,
            "other_end_hard_loss": zero,
            "latent_end_target_mean": zero,
            "latent_end_score_mean": zero,
            "latent_end_front_score_mean": zero,
            "latent_end_tail_score_mean": zero,
        }

    if halt_dense_logits.dim() != 3:
        raise ValueError(f"halt_dense_logits must be rank-3 [batch, seq_len, token_ids], got {halt_dense_logits.dim()}")
    if halt_dense_best_allowed_logits.dim() != 2:
        raise ValueError(
            "halt_dense_best_allowed_logits must be rank-2 [batch, seq_len], "
            f"got {halt_dense_best_allowed_logits.dim()}"
        )
    if latent_internal_mask.shape != halt_dense_logits.shape[:2]:
        raise ValueError(
            "latent_internal_mask shape does not match halt_dense_logits prefix: "
            f"mask_shape={list(latent_internal_mask.shape)}, logits_shape={list(halt_dense_logits.shape)}"
        )
    if attention_mask.shape != halt_dense_logits.shape[:2]:
        raise ValueError(
            "attention_mask shape does not match halt_dense_logits prefix: "
            f"mask_shape={list(attention_mask.shape)}, logits_shape={list(halt_dense_logits.shape)}"
        )
    if list(halt_dense_best_allowed_logits.shape) != list(halt_dense_logits.shape[:2]):
        raise ValueError(
            "halt_dense_best_allowed_logits shape does not match halt_dense_logits prefix: "
            f"best_allowed_shape={list(halt_dense_best_allowed_logits.shape)}, logits_shape={list(halt_dense_logits.shape)}"
        )
    if int(halt_dense_logits.size(-1)) <= 0:
        raise ValueError("halt_dense_logits must contain at least one forbidden token logit")

    valid_counts = valid_mask.to(torch.long).sum(dim=1)
    if bool((valid_counts > 0).any().item()):
        running_index = valid_mask.to(torch.long).cumsum(dim=1) - 1
        latent_index_within_span = torch.where(valid_mask, running_index, latent_index_within_span)
        front_fraction = min(max(float(front_fraction), 0.0), 1.0)
        front_counts = torch.ceil(valid_counts.to(torch.float32) * front_fraction).to(torch.long)
        front_counts = torch.where(
            valid_counts > 0,
            torch.clamp(front_counts, min=1),
            torch.zeros_like(front_counts),
        )
        front_mask = valid_mask & (running_index < front_counts.unsqueeze(1))
        max_index = torch.clamp(valid_counts - 1, min=0)
        distance_long = torch.clamp(max_index.unsqueeze(1) - running_index, min=0)
        distance_to_latent_end = torch.where(valid_mask, distance_long.to(torch.float32), distance_to_latent_end)
        denom_progress = torch.clamp(max_index.to(torch.float32), min=1.0)
        progress_raw = torch.where(
            valid_mask,
            running_index.to(torch.float32) / denom_progress.unsqueeze(1),
            progress_to_end,
        )
        progress_to_end = torch.where(
            valid_mask & (valid_counts.unsqueeze(1) == 1),
            torch.ones_like(progress_raw),
            progress_raw,
        )

    safe_progress = torch.clamp(progress_to_end, min=0.0, max=1.0)
    curve_name = str(soft_target_curve).lower()
    if curve_name == "sigmoid":
        temperature = max(float(soft_target_temperature), 1.0e-6)
        target_curve = torch.sigmoid((safe_progress - 0.5) / temperature)
    elif curve_name == "smoothstep":
        target_curve = safe_progress * safe_progress * (3.0 - 2.0 * safe_progress)
    else:
        raise ValueError(f"Unsupported latent_end_soft_target_curve: {soft_target_curve}")
    power = max(float(soft_target_power), 1.0e-6)
    latent_end_target = torch.where(valid_mask, target_curve.pow(power), latent_end_target)

    position_weight = torch.where(
        front_mask,
        torch.full_like(position_weight, float(front_weight)),
        torch.full_like(position_weight, float(nonfront_weight)),
    )
    position_weight = position_weight * valid_mask.to(torch.float32)

    forbidden_minus_best_allowed = halt_dense_logits.float() - halt_dense_best_allowed_logits.float().unsqueeze(-1)
    token_forbidden_minus_best_allowed = forbidden_minus_best_allowed
    latent_end_margin = forbidden_minus_best_allowed[..., 0]
    latent_end_score = torch.where(
        valid_mask,
        torch.sigmoid(latent_end_margin / max(float(soft_target_temperature), 1.0e-6)),
        latent_end_score,
    )
    token_latent_end_soft_loss = F.binary_cross_entropy(
        latent_end_score,
        latent_end_target,
        reduction="none",
    ) * valid_mask.to(torch.float32)

    if int(halt_dense_logits.size(-1)) > 1:
        other_end_margin = forbidden_minus_best_allowed[..., 1:]
        other_rank_terms = torch.relu(other_end_margin + float(other_end_rank_margin))
        token_other_end_hard_loss = other_rank_terms.mean(dim=-1)
    else:
        token_other_end_hard_loss = token_other_end_hard_loss

    token_rank_loss = (
        float(soft_loss_weight) * token_latent_end_soft_loss
        + float(other_end_hard_loss_weight) * token_other_end_hard_loss
    )
    argmax_violation_mask = valid_mask & (latent_end_margin >= 0.0)

    denom = torch.clamp(position_weight.sum(), min=1.0)
    front_weight_mask = front_mask.to(torch.float32) * position_weight
    nonfront_mask = valid_mask & (~front_mask)
    nonfront_weight_mask = nonfront_mask.to(torch.float32) * position_weight
    front_denom = torch.clamp(front_weight_mask.sum(), min=1.0)
    nonfront_denom = torch.clamp(nonfront_weight_mask.sum(), min=1.0)
    valid_weight_float = valid_mask.to(torch.float32)

    if not bool(valid_mask.any().item()):
        zero_anchor = (halt_dense_logits.sum() + halt_dense_best_allowed_logits.sum()) * 0.0
        halt_dense_loss = zero_anchor
        front_loss = halt_dense_loss
        nonfront_loss = halt_dense_loss
        argmax_violation_rate = halt_dense_loss
        latent_end_soft_loss = halt_dense_loss
        other_end_hard_loss = halt_dense_loss
        latent_end_target_mean = halt_dense_loss
        latent_end_score_mean = halt_dense_loss
        latent_end_front_score_mean = halt_dense_loss
        latent_end_tail_score_mean = halt_dense_loss
    else:
        halt_dense_loss = (token_rank_loss * position_weight).sum() / denom
        front_loss = (token_rank_loss * front_weight_mask).sum() / front_denom
        nonfront_loss = (token_rank_loss * nonfront_weight_mask).sum() / nonfront_denom
        argmax_violation_rate = (
            argmax_violation_mask.to(torch.float32) * valid_weight_float
        ).sum() / torch.clamp(valid_weight_float.sum(), min=1.0)
        latent_end_soft_loss = (token_latent_end_soft_loss * position_weight).sum() / denom
        other_end_hard_loss = (token_other_end_hard_loss * position_weight).sum() / denom
        latent_end_target_mean = (latent_end_target * valid_weight_float).sum() / torch.clamp(valid_weight_float.sum(), min=1.0)
        latent_end_score_mean = (latent_end_score * valid_weight_float).sum() / torch.clamp(valid_weight_float.sum(), min=1.0)
        latent_end_front_score_mean = (latent_end_score * front_mask.to(torch.float32)).sum() / torch.clamp(
            front_mask.to(torch.float32).sum(), min=1.0
        )
        latent_end_tail_score_mean = (latent_end_score * nonfront_mask.to(torch.float32)).sum() / torch.clamp(
            nonfront_mask.to(torch.float32).sum(), min=1.0
        )

    return {
        "halt_dense_loss": halt_dense_loss,
        "token_halt_dense_bce": token_rank_loss,
        "halt_dense_valid_mask": valid_mask,
        "token_rank_loss": token_rank_loss,
        "front_mask": front_mask,
        "position_weight": position_weight,
        "argmax_violation_mask": argmax_violation_mask,
        "front_loss": front_loss,
        "nonfront_loss": nonfront_loss,
        "argmax_violation_rate": argmax_violation_rate,
        "token_forbidden_minus_best_allowed": token_forbidden_minus_best_allowed,
        "latent_end_target": latent_end_target,
        "latent_end_score": latent_end_score,
        "token_latent_end_soft_loss": token_latent_end_soft_loss,
        "token_other_end_hard_loss": token_other_end_hard_loss,
        "distance_to_latent_end": distance_to_latent_end,
        "progress_to_end": progress_to_end,
        "latent_index_within_span": latent_index_within_span,
        "latent_end_soft_loss": latent_end_soft_loss,
        "other_end_hard_loss": other_end_hard_loss,
        "latent_end_target_mean": latent_end_target_mean,
        "latent_end_score_mean": latent_end_score_mean,
        "latent_end_front_score_mean": latent_end_front_score_mean,
        "latent_end_tail_score_mean": latent_end_tail_score_mean,
    }


def compute_teacher_kl(
    logits: torch.Tensor | None,
    batch: Dict[str, Any],
    teacher_cache: Any | None,
    kl_temperature: float,
    stats: Dict[str, Any] | None = None,
    hidden_states: torch.Tensor | None = None,
    lm_head: Any | None = None,
    logits_chunk_size: int = 0,
    selected_topk_logits: torch.Tensor | None = None,
    log_denom: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    kl_source_positions = batch.get("teacher_kl_source_positions")
    kl_pair_mask = batch.get("teacher_kl_pair_mask")
    teacher_effective_mask = batch.get("teacher_kl_effective_mask")
    teacher_topk_probs_batch = batch.get("teacher_kl_topk_probs")
    teacher_tail_batch = batch.get("teacher_kl_tail")

    valid_pair_mask = teacher_effective_mask if teacher_effective_mask is not None else kl_pair_mask
    if valid_pair_mask is not None and int(valid_pair_mask.to(torch.long).sum().item()) <= 0:
        zero = batch["input_ids"].new_zeros((), dtype=torch.float32)
        if stats is not None:
            stats.update(
                {
                    "student_logits_slice_gib": 0.0,
                    "teacher_cache_tensor_gib": 0.0,
                    "student_topk_tensor_gib": 0.0,
                    "kl_tensor_gib": 0.0,
                    "kl_positions": 0,
                    "missing_teacher_cache_sample_count": 0,
                    "missing_teacher_cache_record_ids_preview": [],
                    "skipped_kl_positions_due_to_missing_cache": 0,
                }
            )
        return zero, 0

    base_tensor = selected_topk_logits
    if base_tensor is None:
        base_tensor = hidden_states if hidden_states is not None else logits
    if base_tensor is None:
        base_tensor = log_denom
    if base_tensor is None:
        raise ValueError("compute_teacher_kl requires selected_topk_logits, hidden_states, logits, or log_denom")
    total_kl = base_tensor.new_zeros((), dtype=torch.float32)
    total_positions = 0
    collect_stats = stats is not None
    max_student_logits_bytes = 0
    max_teacher_cache_bytes = 0
    max_topk_prob_bytes = 0
    max_kl_tensor_bytes = 0
    missing_teacher_cache_sample_count = 0
    missing_teacher_cache_record_ids_preview: List[str] = []
    skipped_kl_positions_due_to_missing_cache = 0

    for batch_index, record_id in enumerate(batch["record_ids"]):
        if teacher_effective_mask is not None:
            valid_pairs = teacher_effective_mask[batch_index]
        elif kl_source_positions is not None and kl_pair_mask is not None:
            valid_pairs = kl_pair_mask[batch_index]
        else:
            valid_pairs = None
        if valid_pairs is not None and int(valid_pairs.to(torch.long).sum().item()) <= 0:
            continue

        teacher_ids = None
        teacher_probs = None
        teacher_tail = None
        if teacher_topk_probs_batch is not None and teacher_tail_batch is not None and valid_pairs is not None:
            teacher_probs_tensor = teacher_topk_probs_batch[batch_index, valid_pairs, :].to(torch.float32)
            teacher_tail_tensor = teacher_tail_batch[batch_index, valid_pairs].to(torch.float32)
            usable = int(teacher_probs_tensor.size(0))
            if selected_topk_logits is not None:
                student_selected_logits = selected_topk_logits[batch_index, valid_pairs, :]
                student_log_denom = log_denom[batch_index, valid_pairs] if log_denom is not None else None
                usable = min(usable, int(student_selected_logits.size(0)))
            elif hidden_states is not None:
                student_hidden = hidden_states[batch_index, valid_pairs, :]
                usable = min(usable, int(student_hidden.size(0)))
            else:
                if logits is None or kl_source_positions is None:
                    raise ValueError("Dense teacher KL fallback requires logits and kl_source_positions")
                source_positions = kl_source_positions[batch_index][valid_pairs]
                if source_positions.numel() <= 0:
                    continue
                student_logits = logits[batch_index, source_positions, :]
                usable = min(usable, int(student_logits.size(0)))
        else:
            if teacher_cache is None:
                raise ValueError("teacher_cache is required when batch teacher tensors are absent")
            if hasattr(teacher_cache, "get_optional"):
                maybe_teacher = teacher_cache.get_optional(record_id)
            else:
                try:
                    maybe_teacher = teacher_cache.get(record_id)
                except KeyError:
                    maybe_teacher = None
            if maybe_teacher is None:
                missing_teacher_cache_sample_count += 1
                if len(missing_teacher_cache_record_ids_preview) < 8:
                    missing_teacher_cache_record_ids_preview.append(str(record_id))
                if valid_pairs is not None:
                    skipped_kl_positions_due_to_missing_cache += int(valid_pairs.to(torch.long).sum().item())
                elif kl_source_positions is not None and kl_pair_mask is not None:
                    skipped_kl_positions_due_to_missing_cache += int(kl_pair_mask[batch_index].to(torch.long).sum().item())
                continue
            teacher_ids, teacher_probs, teacher_tail = maybe_teacher
            if logits is None:
                raise ValueError("Dense teacher KL fallback requires logits")
            shift_logits = logits[:, :-1, :]
            student_start = int(batch["teacher_target_start"][batch_index].item())
            student_logits = shift_logits[batch_index, student_start:]
            usable = min(student_logits.size(0), teacher_ids.shape[0])

        if usable <= 0:
            continue

        chunk_size = usable if int(logits_chunk_size) <= 0 else int(logits_chunk_size)
        chunk_size = max(chunk_size, 1)
        for start in range(0, usable, chunk_size):
            end = min(start + chunk_size, usable)
            if teacher_topk_probs_batch is not None and teacher_tail_batch is not None and valid_pairs is not None:
                teacher_probs_tensor = teacher_topk_probs_batch[batch_index, valid_pairs, :][start:end].to(torch.float32)
                teacher_tail_tensor = teacher_tail_batch[batch_index, valid_pairs][start:end].to(torch.float32)
                if selected_topk_logits is not None:
                    current_topk_logits = selected_topk_logits[batch_index, valid_pairs, :][start:end].float()
                    current_log_denom = log_denom[batch_index, valid_pairs][start:end].float()
                    student_topk_log_probs = current_topk_logits - current_log_denom.unsqueeze(-1)
                    student_topk_probs = student_topk_log_probs.exp()
                    student_tail_probs = torch.clamp(1.0 - student_topk_probs.sum(dim=-1), min=1.0e-8)
                    if collect_stats:
                        max_student_logits_bytes = max(
                            max_student_logits_bytes,
                            tensor_nbytes(current_topk_logits) + tensor_nbytes(current_log_denom),
                        )
                        max_teacher_cache_bytes = max(
                            max_teacher_cache_bytes,
                            tensor_nbytes(teacher_probs_tensor) + tensor_nbytes(teacher_tail_tensor),
                        )
                        max_topk_prob_bytes = max(
                            max_topk_prob_bytes,
                            tensor_nbytes(student_topk_log_probs)
                            + tensor_nbytes(student_topk_probs)
                            + tensor_nbytes(student_tail_probs),
                        )
                elif hidden_states is not None and kl_source_positions is not None and kl_pair_mask is not None:
                    current_logits = lm_head(student_hidden[start:end]).float() / kl_temperature
                    current_log_denom = torch.logsumexp(current_logits, dim=-1)
                    current_topk_logits = torch.gather(
                        current_logits,
                        dim=-1,
                        index=batch["teacher_kl_topk_ids"][batch_index, valid_pairs, :][start:end].to(
                            device=current_logits.device,
                            dtype=torch.long,
                        ),
                    )
                    student_topk_log_probs = current_topk_logits - current_log_denom.unsqueeze(-1)
                    student_topk_probs = student_topk_log_probs.exp()
                    student_tail_probs = torch.clamp(1.0 - student_topk_probs.sum(dim=-1), min=1.0e-8)
                else:
                    current_logits = student_logits[start:end].float() / kl_temperature
                    ids_tensor = batch["teacher_kl_topk_ids"][batch_index, valid_pairs, :][start:end].to(
                        device=current_logits.device,
                        dtype=torch.long,
                    )
                    current_log_denom = torch.logsumexp(current_logits, dim=-1)
                    current_topk_logits = torch.gather(current_logits, dim=-1, index=ids_tensor)
                    student_topk_log_probs = current_topk_logits - current_log_denom.unsqueeze(-1)
                    student_topk_probs = student_topk_log_probs.exp()
                    student_tail_probs = torch.clamp(1.0 - student_topk_probs.sum(dim=-1), min=1.0e-8)
            elif hidden_states is not None and kl_source_positions is not None and kl_pair_mask is not None:
                current_logits = lm_head(student_hidden[start:end]).float() / kl_temperature
                ids_tensor = torch.as_tensor(teacher_ids[start:end].copy(), device=base_tensor.device, dtype=torch.long)
                teacher_probs_tensor = torch.as_tensor(
                    teacher_probs[start:end].copy(), device=base_tensor.device, dtype=current_logits.dtype
                )
                teacher_tail_tensor = torch.as_tensor(
                    teacher_tail[start:end].copy(), device=base_tensor.device, dtype=current_logits.dtype
                )
                current_log_denom = torch.logsumexp(current_logits, dim=-1)
                student_topk_logits = torch.gather(current_logits, dim=-1, index=ids_tensor)
                student_topk_log_probs = student_topk_logits - current_log_denom.unsqueeze(-1)
                student_topk_probs = student_topk_log_probs.exp()
                student_tail_probs = torch.clamp(1.0 - student_topk_probs.sum(dim=-1), min=1.0e-8)
            else:
                current_logits = student_logits[start:end].float() / kl_temperature
                ids_tensor = torch.as_tensor(teacher_ids[start:end].copy(), device=base_tensor.device, dtype=torch.long)
                teacher_probs_tensor = torch.as_tensor(
                    teacher_probs[start:end].copy(), device=base_tensor.device, dtype=current_logits.dtype
                )
                teacher_tail_tensor = torch.as_tensor(
                    teacher_tail[start:end].copy(), device=base_tensor.device, dtype=current_logits.dtype
                )
                current_log_denom = torch.logsumexp(current_logits, dim=-1)
                student_topk_logits = torch.gather(current_logits, dim=-1, index=ids_tensor)
                student_topk_log_probs = student_topk_logits - current_log_denom.unsqueeze(-1)
                student_topk_probs = student_topk_log_probs.exp()
                student_tail_probs = torch.clamp(1.0 - student_topk_probs.sum(dim=-1), min=1.0e-8)

            if collect_stats:
                if teacher_topk_probs_batch is None or teacher_tail_batch is None or valid_pairs is None or selected_topk_logits is None:
                    max_student_logits_bytes = max(max_student_logits_bytes, tensor_nbytes(current_logits))
                    max_teacher_cache_bytes = max(
                        max_teacher_cache_bytes,
                        tensor_nbytes(ids_tensor) + tensor_nbytes(teacher_probs_tensor) + tensor_nbytes(teacher_tail_tensor),
                    )
                    max_topk_prob_bytes = max(
                        max_topk_prob_bytes,
                        tensor_nbytes(current_log_denom)
                        + tensor_nbytes(student_topk_logits)
                        + tensor_nbytes(student_topk_log_probs)
                        + tensor_nbytes(student_topk_probs)
                        + tensor_nbytes(student_tail_probs),
                    )

            teacher_topk = torch.clamp(teacher_probs_tensor, min=0.0)
            teacher_tail_mass = torch.clamp(teacher_tail_tensor, min=0.0)
            teacher_norm = torch.clamp(teacher_topk.sum(dim=-1) + teacher_tail_mass, min=1.0e-8)
            teacher_topk = teacher_topk / teacher_norm.unsqueeze(-1)
            teacher_tail_mass = teacher_tail_mass / teacher_norm

            kl_topk = teacher_topk * (
                torch.log(torch.clamp(teacher_topk, min=1.0e-8)) - student_topk_log_probs
            )
            kl_tail = teacher_tail_mass * (
                torch.log(torch.clamp(teacher_tail_mass, min=1.0e-8)) - torch.log(student_tail_probs)
            )
            if collect_stats:
                max_kl_tensor_bytes = max(
                    max_kl_tensor_bytes,
                    tensor_nbytes(teacher_topk)
                    + tensor_nbytes(teacher_tail_mass)
                    + tensor_nbytes(kl_topk)
                    + tensor_nbytes(kl_tail),
                )
            total_kl = total_kl + kl_topk.sum() + kl_tail.sum()
            total_positions += end - start

    if collect_stats and stats is not None:
        stats.update({
            "student_logits_slice_gib": bytes_to_gib(max_student_logits_bytes),
            "teacher_cache_tensor_gib": bytes_to_gib(max_teacher_cache_bytes),
            "student_topk_tensor_gib": bytes_to_gib(max_topk_prob_bytes),
            "kl_tensor_gib": bytes_to_gib(max_kl_tensor_bytes),
            "kl_positions": int(total_positions),
            "missing_teacher_cache_sample_count": int(missing_teacher_cache_sample_count),
            "missing_teacher_cache_record_ids_preview": list(missing_teacher_cache_record_ids_preview),
            "skipped_kl_positions_due_to_missing_cache": int(skipped_kl_positions_due_to_missing_cache),
        })

    if total_positions == 0:
        return base_tensor.new_zeros((), dtype=torch.float32), 0
    return total_kl / total_positions, total_positions
