from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from glyph.retrieval import RetrievalResult, select_relevant_context, split_context_units
from scripts.word2png_function import text_to_images


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RENDER_CONFIG = REPO_ROOT / 'config' / 'config_en.json'


@dataclass
class ReadabilityPolicy:
    max_chars_per_page: int = 2200
    max_render_pages: int = 3
    min_font_size: int = 8
    max_font_size: int = 11
    min_line_height: int = 9
    max_line_height: int = 14
    min_margin_x: int = 6
    min_margin_y: int = 6
    min_horizontal_scale: float = 0.92


@dataclass
class QueryGuidedRenderArtifacts:
    image_paths: list[str]
    selected_text: str
    page_texts: list[str]
    retrieval: RetrievalResult
    render_config: dict
    readability_policy: ReadabilityPolicy
    metadata_path: str


def _load_render_config(
    *,
    config_path: Optional[str | Path] = None,
    config_dict: Optional[dict] = None,
) -> dict:
    if config_dict is not None:
        return dict(config_dict)
    config_source = Path(config_path) if config_path is not None else DEFAULT_RENDER_CONFIG
    return json.loads(config_source.read_text(encoding='utf-8'))


def _join_units_with_budget(units: list[str], max_chars: int) -> str:
    if max_chars <= 0:
        return ''

    selected: list[str] = []
    current_chars = 0
    for unit in units:
        normalized = unit.strip()
        if not normalized:
            continue
        added_chars = len(normalized) + (2 if selected else 0)
        if selected and current_chars + added_chars > max_chars:
            break
        if not selected and len(normalized) > max_chars:
            return normalized[:max_chars].rsplit(' ', 1)[0].strip() or normalized[:max_chars].strip()
        selected.append(normalized)
        current_chars += added_chars
    return '\n\n'.join(selected).strip()


def enforce_readability_budget(text: str, policy: ReadabilityPolicy) -> str:
    max_chars = policy.max_chars_per_page * policy.max_render_pages
    if len(text) <= max_chars:
        return text.strip()

    units = split_context_units(text)
    bounded = _join_units_with_budget(units, max_chars)
    if bounded:
        return bounded
    return text[:max_chars].rsplit(' ', 1)[0].strip() or text[:max_chars].strip()


def build_readability_render_config(base_config: dict, text: str, policy: ReadabilityPolicy) -> tuple[dict, dict]:
    config = dict(base_config)
    text_length = max(1, len(text))
    base_font_size = int(config.get('font-size', 9))
    base_line_height = int(config.get('line-height', base_font_size + 1))
    base_margin_x = int(config.get('margin-x', 10))
    base_margin_y = int(config.get('margin-y', 10))
    base_horizontal_scale = float(config.get('horizontal-scale', 1.0))

    estimated_pages = max(1, math.ceil(text_length / max(1, policy.max_chars_per_page)))
    target_pages = min(max(1, estimated_pages), max(1, policy.max_render_pages))
    compression_pressure = estimated_pages / max(1, target_pages)

    scale_factor = 1.0 / math.sqrt(max(1.0, compression_pressure))
    font_size = int(round(base_font_size * scale_factor))
    font_size = max(policy.min_font_size, min(policy.max_font_size, font_size))

    line_ratio = base_line_height / max(1, base_font_size)
    line_height = int(round(font_size * line_ratio))
    line_height = max(policy.min_line_height, min(policy.max_line_height, line_height))

    margin_shrink = min(4, max(0, estimated_pages - target_pages))
    margin_x = max(policy.min_margin_x, base_margin_x - margin_shrink)
    margin_y = max(policy.min_margin_y, base_margin_y - margin_shrink)

    horizontal_scale = max(policy.min_horizontal_scale, min(1.0, base_horizontal_scale * scale_factor))

    config['font-size'] = font_size
    config['line-height'] = line_height
    config['margin-x'] = margin_x
    config['margin-y'] = margin_y
    config['horizontal-scale'] = round(horizontal_scale, 3)
    config['auto-crop-width'] = True
    config['auto-crop-last-page'] = True

    summary = {
        'estimated_pages': estimated_pages,
        'target_pages': target_pages,
        'compression_pressure': round(compression_pressure, 4),
        'font_size': font_size,
        'line_height': line_height,
        'margin_x': margin_x,
        'margin_y': margin_y,
        'horizontal_scale': config['horizontal-scale'],
    }
    return config, summary


def partition_text_for_pages(text: str, page_count: int) -> list[str]:
    if page_count <= 0:
        return []
    units = [unit.strip() for unit in split_context_units(text) if unit.strip()]
    if not units:
        return [''] * page_count
    if page_count == 1:
        return ['\n\n'.join(units)]

    total_chars = sum(len(unit) for unit in units)
    target_chars = max(1, math.ceil(total_chars / page_count))
    pages: list[str] = []
    current_units: list[str] = []
    current_chars = 0

    for index, unit in enumerate(units):
        remaining_units = len(units) - index
        remaining_pages = page_count - len(pages)
        if current_units and current_chars >= target_chars and remaining_units >= remaining_pages:
            pages.append('\n\n'.join(current_units))
            current_units = [unit]
            current_chars = len(unit)
            continue

        current_units.append(unit)
        current_chars += len(unit)

    if current_units:
        pages.append('\n\n'.join(current_units))

    while len(pages) < page_count:
        pages.append('')
    if len(pages) > page_count:
        overflow = pages[page_count - 1:]
        pages = pages[: page_count - 1] + ['\n\n'.join(part for part in overflow if part)]
    return pages


def render_query_guided_pngs(
    *,
    context: str,
    question: str,
    output_dir: str | Path,
    unique_id: str,
    config_path: Optional[str | Path] = None,
    config_dict: Optional[dict] = None,
    retrieval_max_units: int = 6,
    retrieval_neighbor_window: int = 1,
    retrieval_max_sentences_per_unit: int = 5,
    readability_policy: Optional[ReadabilityPolicy] = None,
) -> QueryGuidedRenderArtifacts:
    policy = readability_policy or ReadabilityPolicy()
    retrieval = select_relevant_context(
        context=context,
        question=question,
        max_units=retrieval_max_units,
        neighbor_window=retrieval_neighbor_window,
        max_sentences_per_unit=retrieval_max_sentences_per_unit,
        max_chars=policy.max_chars_per_page * policy.max_render_pages,
    )
    selected_text = enforce_readability_budget(retrieval.selected_text, policy)

    base_config = _load_render_config(config_path=config_path, config_dict=config_dict)
    render_config, render_summary = build_readability_render_config(base_config, selected_text, policy)
    image_paths = text_to_images(
        text=selected_text,
        output_dir=str(output_dir),
        config_dict=render_config,
        unique_id=unique_id,
    )

    page_root = Path(image_paths[0]).parent if image_paths else Path(output_dir) / unique_id
    page_root.mkdir(parents=True, exist_ok=True)
    page_texts = partition_text_for_pages(selected_text, len(image_paths))
    metadata_path = page_root / 'page_texts.json'
    metadata_path.write_text(json.dumps(page_texts, ensure_ascii=False, indent=2), encoding='utf-8')

    report_path = page_root / 'query_guided_render_report.json'
    report_path.write_text(
        json.dumps(
            {
                'question': question,
                'selected_indices': retrieval.selected_indices,
                'total_units': retrieval.total_units,
                'retrieval_scores': retrieval.scores,
                'selected_text': selected_text,
                'readability_policy': asdict(policy),
                'render_summary': render_summary,
                'image_paths': image_paths,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )
    return QueryGuidedRenderArtifacts(
        image_paths=image_paths,
        selected_text=selected_text,
        page_texts=page_texts,
        retrieval=retrieval,
        render_config=render_config,
        readability_policy=policy,
        metadata_path=str(report_path),
    )
