from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class CompressionResult:
    reduced_embeddings: torch.Tensor
    kept_indices: torch.Tensor
    merged_indices: torch.Tensor
    target_indices: torch.Tensor


def _ensure_batched_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    if embeddings.ndim == 2:
        return embeddings.unsqueeze(0)
    if embeddings.ndim != 3:
        raise ValueError(f"Expected embeddings with 2 or 3 dims, got {embeddings.shape}")
    return embeddings


def _ensure_batched_attention(attention: torch.Tensor) -> torch.Tensor:
    if attention.ndim == 2:
        return attention.unsqueeze(0)
    if attention.ndim != 3:
        raise ValueError(f"Expected attention with 2 or 3 dims, got {attention.shape}")
    return attention


def _topk_safe(scores: torch.Tensor, k: int, largest: bool) -> torch.Tensor:
    if k <= 0:
        return torch.empty(scores.shape[0], 0, device=scores.device, dtype=torch.long)
    k = min(k, scores.shape[-1])
    return torch.topk(scores, k=k, dim=-1, largest=largest).indices


def _remaining_indices(token_count: int, merge_indices: torch.Tensor, start_index: int = 0) -> torch.Tensor:
    batch_size = merge_indices.shape[0]
    base = torch.arange(start_index, token_count, device=merge_indices.device)
    all_indices = base.unsqueeze(0).expand(batch_size, -1)
    relative_merge_indices = merge_indices - start_index
    mask = torch.ones_like(all_indices, dtype=torch.bool)
    mask.scatter_(1, relative_merge_indices, False)
    return all_indices[mask].view(batch_size, -1)


def _choose_targets_from_remaining(
    source_scores: torch.Tensor,
    remaining_indices: torch.Tensor,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    candidate_scores = source_scores.gather(
        2,
        remaining_indices.unsqueeze(1).expand(
            source_scores.shape[0], source_scores.shape[1], remaining_indices.shape[1]
        ),
    )
    topk = min(topk, remaining_indices.shape[1])
    target_scores, relative_indices = torch.topk(candidate_scores, k=topk, dim=-1)
    target_indices = remaining_indices.unsqueeze(1).expand(
        source_scores.shape[0], source_scores.shape[1], remaining_indices.shape[1]
    ).gather(2, relative_indices)
    return target_scores, target_indices


def _merge_token_block(
    embeddings: torch.Tensor,
    merge_indices: torch.Tensor,
    target_indices: torch.Tensor,
    target_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, token_count, hidden_size = embeddings.shape
    merge_count = merge_indices.shape[1]

    keep_mask = torch.ones(batch_size, token_count, device=embeddings.device, dtype=torch.bool)
    keep_mask.scatter_(1, merge_indices, False)
    kept_indices = keep_mask.nonzero(as_tuple=False).view(batch_size, token_count - merge_count, 2)[..., 1]

    reduced = embeddings.gather(
        1,
        kept_indices.unsqueeze(-1).expand(batch_size, token_count - merge_count, hidden_size),
    ).clone()

    if merge_count == 0:
        return reduced, kept_indices

    old_to_new = torch.full(
        (batch_size, token_count),
        -1,
        device=embeddings.device,
        dtype=torch.long,
    )
    new_positions = torch.arange(reduced.shape[1], device=embeddings.device).unsqueeze(0).expand(batch_size, -1)
    old_to_new.scatter_(1, kept_indices, new_positions)

    merge_sources = embeddings.gather(
        1,
        merge_indices.unsqueeze(-1).expand(batch_size, merge_count, hidden_size),
    )
    normalized_weights = target_weights / target_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    weighted_sources = merge_sources.unsqueeze(2) * normalized_weights.unsqueeze(-1)
    flat_sources = weighted_sources.reshape(batch_size, -1, hidden_size)

    flat_targets = target_indices.reshape(batch_size, -1)
    mapped_targets = old_to_new.gather(1, flat_targets)
    if (mapped_targets < 0).any():
        raise ValueError("Merge targets must refer to tokens that remain after compression")

    reduced.scatter_add_(
        1,
        mapped_targets.unsqueeze(-1).expand(batch_size, mapped_targets.shape[1], hidden_size),
        flat_sources,
    )
    return reduced, kept_indices


def reduce_visual_tokens(
    embeddings: torch.Tensor,
    attention: torch.Tensor,
    reduction_factor: int,
    include_class_token: bool = True,
    target_topk: int = 4,
) -> CompressionResult:
    embeddings = _ensure_batched_embeddings(embeddings)
    attention = _ensure_batched_attention(attention)

    if embeddings.shape[0] != attention.shape[0] or embeddings.shape[1] != attention.shape[1]:
        raise ValueError("Embeddings and attention must have matching batch/token dimensions")

    protected = 1 if include_class_token else 0
    token_count = embeddings.shape[1]
    candidate_count = token_count - protected
    if candidate_count <= 1:
        raise ValueError("Need at least two non-protected visual tokens to compress")

    merge_count = min(reduction_factor, candidate_count - 1)
    if merge_count <= 0:
        kept_indices = torch.arange(token_count, device=embeddings.device).unsqueeze(0).expand(embeddings.shape[0], -1)
        empty = torch.empty(embeddings.shape[0], 0, device=embeddings.device, dtype=torch.long)
        return CompressionResult(embeddings, kept_indices, empty, empty.unsqueeze(-1))

    patch_scores = attention[:, protected:, protected:].mean(dim=-1)
    if include_class_token:
        cls_scores = attention[:, 0, protected:]
    else:
        cls_scores = torch.zeros_like(patch_scores)
    scores = 0.35 * torch.nn.functional.normalize(cls_scores, dim=-1) - 0.65 * torch.nn.functional.normalize(
        patch_scores, dim=-1
    )
    merge_relative = _topk_safe(scores, merge_count, largest=False)
    merge_indices = merge_relative + protected

    candidate_scores = attention.gather(
        1,
        merge_indices.unsqueeze(-1).expand(-1, merge_count, attention.shape[-1]),
    ).clone()
    candidate_scores[..., :protected] = torch.finfo(candidate_scores.dtype).min
    candidate_scores.scatter_(
        -1,
        merge_indices.unsqueeze(1).expand(-1, merge_count, -1),
        torch.finfo(candidate_scores.dtype).min,
    )

    remaining_indices = _remaining_indices(token_count, merge_indices, start_index=protected)
    topk = min(target_topk, max(1, remaining_indices.shape[1]))
    target_scores, target_indices = _choose_targets_from_remaining(candidate_scores, remaining_indices, topk)
    target_weights = torch.softmax(target_scores, dim=-1)

    reduced_embeddings, kept_indices = _merge_token_block(
        embeddings,
        merge_indices,
        target_indices,
        target_weights,
    )
    return CompressionResult(reduced_embeddings, kept_indices, merge_indices, target_indices)


def reduce_language_tokens(
    embeddings: torch.Tensor,
    attention: torch.Tensor,
    visual_token_count: int,
    reduction_factor: int,
    target_topk: int = 4,
) -> CompressionResult:
    embeddings = _ensure_batched_embeddings(embeddings)
    attention = _ensure_batched_attention(attention)

    if visual_token_count <= 1 or visual_token_count >= embeddings.shape[1]:
        raise ValueError("visual_token_count must be between 2 and total_tokens - 1")
    if embeddings.shape[0] != attention.shape[0] or embeddings.shape[1] != attention.shape[1]:
        raise ValueError("Embeddings and attention must have matching batch/token dimensions")

    visual_embeddings = embeddings[:, :visual_token_count, :]
    text_embeddings = embeddings[:, visual_token_count:, :]

    merge_count = min(reduction_factor, visual_token_count - 1)
    if merge_count <= 0:
        kept_indices = torch.arange(embeddings.shape[1], device=embeddings.device).unsqueeze(0).expand(embeddings.shape[0], -1)
        empty = torch.empty(embeddings.shape[0], 0, device=embeddings.device, dtype=torch.long)
        return CompressionResult(embeddings, kept_indices, empty, empty.unsqueeze(-1))

    visual_to_visual = attention[:, :visual_token_count, :visual_token_count]
    visual_to_text = attention[:, :visual_token_count, visual_token_count:]
    text_to_visual = attention[:, visual_token_count:, :visual_token_count]

    direct_score = visual_to_visual.sum(dim=-1)
    cross_score = visual_to_text.mean(dim=-1)
    merge_scores = 0.6 * direct_score - 0.4 * cross_score
    merge_indices = _topk_safe(merge_scores, merge_count, largest=True)

    indirect_score = torch.einsum("bij,bjk->bik", visual_to_text, text_to_visual)
    correlate_scores = 0.6 * visual_to_visual + 0.4 * indirect_score
    diag = torch.arange(visual_token_count, device=embeddings.device)
    correlate_scores[:, diag, diag] = torch.finfo(correlate_scores.dtype).min
    correlate_scores.scatter_(
        -1,
        merge_indices.unsqueeze(1).expand(-1, merge_count, -1),
        torch.finfo(correlate_scores.dtype).min,
    )

    remaining_indices = _remaining_indices(visual_token_count, merge_indices)
    topk = min(target_topk, max(1, remaining_indices.shape[1]))
    source_scores = correlate_scores.gather(
        1,
        merge_indices.unsqueeze(-1).expand(-1, merge_count, visual_token_count),
    )
    target_scores, target_indices = _choose_targets_from_remaining(source_scores, remaining_indices, topk)
    target_weights = torch.softmax(target_scores, dim=-1)

    reduced_visual, kept_visual_indices = _merge_token_block(
        visual_embeddings,
        merge_indices,
        target_indices,
        target_weights,
    )
    reduced_embeddings = torch.cat([reduced_visual, text_embeddings], dim=1)

    text_indices = torch.arange(
        visual_token_count,
        embeddings.shape[1],
        device=embeddings.device,
    ).unsqueeze(0).expand(embeddings.shape[0], -1)
    kept_indices = torch.cat([kept_visual_indices, text_indices], dim=1)
    return CompressionResult(reduced_embeddings, kept_indices, merge_indices, target_indices)
