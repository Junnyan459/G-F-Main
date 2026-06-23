from __future__ import annotations

from typing import Sequence


def smooth_local_scores(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []

    smoothed: list[float] = []
    for index, score in enumerate(scores):
        prev_score = scores[index - 1] if index > 0 else score
        next_score = scores[index + 1] if index + 1 < len(scores) else score
        smoothed.append((0.2 * prev_score) + (0.6 * score) + (0.2 * next_score))
    return smoothed


def contiguous_ranges(indices: Sequence[int]) -> list[tuple[int, int]]:
    if not indices:
        return []

    ordered = sorted(indices)
    ranges: list[tuple[int, int]] = []
    start = ordered[0]
    end = start
    for index in ordered[1:]:
        if index == end + 1:
            end = index
            continue
        ranges.append((start, end))
        start = end = index
    ranges.append((start, end))
    return ranges


def _rank_indices(primary_scores: Sequence[float], secondary_scores: Sequence[float]) -> list[int]:
    return sorted(
        range(len(primary_scores)),
        key=lambda index: (primary_scores[index], secondary_scores[index], -index),
        reverse=True,
    )


def _neighbor_candidates(
    selected: set[int],
    total_count: int,
    smoothed_scores: Sequence[float],
    raw_scores: Sequence[float],
    max_distance: int,
) -> list[int]:
    candidates: list[tuple[int, int, float, float, int]] = []
    for index in range(total_count):
        if index in selected:
            continue
        min_distance = min(abs(index - selected_index) for selected_index in selected)
        if min_distance > max_distance:
            continue
        adjacency = int(index - 1 in selected) + int(index + 1 in selected)
        candidates.append((adjacency, -min_distance, smoothed_scores[index], raw_scores[index], -index))

    candidates.sort(reverse=True)
    return [-index for *_, index in candidates]


def select_ordered_indices(
    scores: Sequence[float],
    keep_count: int,
    *,
    preserve_neighbors: bool = True,
    neighbor_radius: int = 1,
) -> list[int]:
    total_count = len(scores)
    if keep_count <= 0 or total_count == 0:
        return []
    if keep_count >= total_count:
        return list(range(total_count))

    smoothed_scores = smooth_local_scores(scores)
    ranked_indices = _rank_indices(smoothed_scores, scores)

    neighbor_slots = 0
    if preserve_neighbors and keep_count > 1:
        neighbor_slots = min(max(1, keep_count // 3), keep_count - 1)
    seed_count = max(1, keep_count - neighbor_slots)

    selected = set(ranked_indices[:seed_count])

    for distance in range(1, max(1, neighbor_radius) + 1):
        for index in _neighbor_candidates(
            selected=selected,
            total_count=total_count,
            smoothed_scores=smoothed_scores,
            raw_scores=scores,
            max_distance=distance,
        ):
            if len(selected) >= keep_count:
                break
            selected.add(index)
        if len(selected) >= keep_count:
            break

    if len(selected) < keep_count and len(selected) >= 2:
        left = min(selected)
        right = max(selected)
        bridge_candidates = [
            index
            for index in range(left + 1, right)
            if index not in selected
        ]
        bridge_candidates.sort(
            key=lambda index: (smoothed_scores[index], scores[index], -index),
            reverse=True,
        )
        for index in bridge_candidates:
            selected.add(index)
            if len(selected) >= keep_count:
                break

    if len(selected) < keep_count:
        for index in ranked_indices:
            selected.add(index)
            if len(selected) >= keep_count:
                break

    return sorted(selected)


def enforce_protected_indices(
    selected_indices: Sequence[int],
    protected_indices: Sequence[int],
    scores: Sequence[float],
    keep_count: int,
) -> list[int]:
    if keep_count <= 0:
        return []

    total_count = len(scores)
    selected = {index for index in selected_indices if 0 <= index < total_count}
    protected = []
    for index in protected_indices:
        if 0 <= index < total_count and index not in protected:
            protected.append(index)

    for index in protected:
        if index in selected:
            continue
        if len(selected) < keep_count:
            selected.add(index)
            continue

        removable = sorted(
            (candidate for candidate in selected if candidate not in protected),
            key=lambda candidate: (scores[candidate], candidate),
        )
        if not removable:
            continue
        selected.remove(removable[0])
        selected.add(index)

    if len(selected) > keep_count:
        removable = sorted(
            (candidate for candidate in selected if candidate not in protected),
            key=lambda candidate: (scores[candidate], candidate),
        )
        while len(selected) > keep_count and removable:
            selected.remove(removable.pop(0))

    return sorted(selected)
