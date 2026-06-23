#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import LlavaConfig, LlavaForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glyph import ReadabilityPolicy, render_query_guided_pngs
from glyph.inference import LlavaInferenceEngine
from glyph_ficoco import reduce_language_tokens, reduce_visual_tokens, select_ordered_indices


DEFAULT_RENDER_CONFIG = REPO_ROOT / 'config' / 'config_en.json'
DEFAULT_OUTPUT_DIR = Path('/tmp/glyph_llava_ficoco_pipeline')
DEFAULT_LLAVA_PATH = Path('/root/autodl-tmp/llava')

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
MULTI_DOC_MARKERS = (
    'Document ',
    'Passage ',
    'Title:',
    'Source:',
)


@dataclass
class PipelineArtifacts:
    rendered_images: list[str]
    selected_text: str
    visual_embedding_shape: tuple[int, ...]
    reduced_visual_shape: tuple[int, ...]
    refined_visual_shape: tuple[int, ...]
    chunk_count_before: int
    chunk_count_after: int
    selected_chunk_indices: list[int]
    generated_answer: str


def read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def _read_query(path: Path | None, query: str | None) -> str:
    if query:
        return query.strip()
    if path is not None:
        return read_text(path).strip()
    raise ValueError('A query must be provided with --query or --query-file')


def _chunk_overlap(lines_per_chunk: int) -> int:
    return max(2, min(8, lines_per_chunk // 5))


def _looks_like_boundary(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    return any(stripped.startswith(marker) for marker in MULTI_DOC_MARKERS)


def _segment_lines(lines: list[str]) -> list[list[str]]:
    segments: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if current and _looks_like_boundary(line):
            segments.append(current)
            current = [line]
            continue
        current.append(line)
    if current:
        segments.append(current)
    return segments


def _split_oversized_segment(segment: list[str], lines_per_chunk: int, overlap: int) -> list[list[str]]:
    if len(segment) <= lines_per_chunk:
        return [segment]

    windows: list[list[str]] = []
    start = 0
    while start < len(segment):
        end = min(len(segment), start + lines_per_chunk)
        windows.append(segment[start:end])
        if end >= len(segment):
            break
        start = max(start + 1, end - overlap)
    return windows


def chunk_text_for_ficoco(text: str, lines_per_chunk: int) -> list[str]:
    lines = text.splitlines()
    if not lines:
        return [text] if text else []

    overlap = _chunk_overlap(lines_per_chunk)
    segments = _segment_lines(lines)
    chunks: list[str] = []
    current: list[str] = []

    def flush_current() -> None:
        nonlocal current
        chunk = '\n'.join(current).strip()
        if chunk:
            chunks.append(chunk)
        current = []

    for segment in segments:
        for piece in _split_oversized_segment(segment, lines_per_chunk, overlap):
            if not current:
                current = piece.copy()
                continue
            if len(current) + len(piece) <= lines_per_chunk:
                current.extend(piece)
                continue
            flush_current()
            current = piece.copy()

    flush_current()
    return chunks


def pad_to_square(image: Image.Image, fill_rgb: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    canvas = Image.new('RGB', (side, side), fill_rgb)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def preprocess_for_llava(image_path: Path, image_size: int = 336) -> torch.Tensor:
    image = Image.open(image_path).convert('RGB')
    image = pad_to_square(image)
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)

    tensor = torch.tensor(list(image.getdata()), dtype=torch.float32).view(image_size, image_size, 3)
    tensor = tensor.permute(2, 0, 1) / 255.0
    mean = torch.tensor(CLIP_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(CLIP_STD, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def make_attention(embeddings: torch.Tensor) -> torch.Tensor:
    normalized = torch.nn.functional.normalize(embeddings, dim=-1)
    return normalized @ normalized.transpose(-1, -2)


def build_chunk_embeddings(text_chunks: list[str], hidden_size: int, dtype: torch.dtype) -> torch.Tensor:
    text_embeddings = torch.zeros(1, len(text_chunks), hidden_size, dtype=dtype)
    for index, chunk in enumerate(text_chunks):
        encoded = chunk.encode('utf-8', errors='ignore')
        if not encoded:
            continue
        raw = torch.tensor(list(encoded), dtype=dtype)
        features = torch.zeros(hidden_size, dtype=dtype)
        limit = min(hidden_size, raw.numel())
        features[:limit] = raw[:limit] / 255.0
        text_embeddings[0, index] = features
    return text_embeddings


def refine_visual_context_with_ficoco(
    reduced_visual_embeddings: torch.Tensor,
    text_chunks: list[str],
    language_reduction_factor: int,
) -> torch.Tensor:
    if not text_chunks:
        return reduced_visual_embeddings

    visual_count = reduced_visual_embeddings.shape[1]
    if visual_count <= 1:
        return reduced_visual_embeddings

    text_embeddings = build_chunk_embeddings(
        text_chunks=text_chunks,
        hidden_size=reduced_visual_embeddings.shape[-1],
        dtype=reduced_visual_embeddings.dtype,
    )
    multimodal_embeddings = torch.cat([reduced_visual_embeddings, text_embeddings], dim=1)
    multimodal_attention = make_attention(multimodal_embeddings)
    reduction_factor = min(language_reduction_factor, visual_count - 1)
    if reduction_factor <= 0:
        return reduced_visual_embeddings

    language_result = reduce_language_tokens(
        embeddings=multimodal_embeddings,
        attention=multimodal_attention,
        visual_token_count=visual_count,
        reduction_factor=reduction_factor,
    )
    return language_result.reduced_embeddings[:, : visual_count - reduction_factor, :]


def select_relevant_chunks(
    refined_visual_embeddings: torch.Tensor,
    text_chunks: list[str],
    chunk_reduction_factor: int,
) -> tuple[torch.Tensor, list[int]]:
    if not text_chunks:
        raise ValueError('No text chunks available for FiCoCo selection')

    hidden_size = refined_visual_embeddings.shape[-1]
    text_embeddings = build_chunk_embeddings(
        text_chunks=text_chunks,
        hidden_size=hidden_size,
        dtype=refined_visual_embeddings.dtype,
    )
    visual_summary = refined_visual_embeddings.mean(dim=1, keepdim=True)
    similarity = torch.nn.functional.cosine_similarity(
        text_embeddings,
        visual_summary.expand_as(text_embeddings),
        dim=-1,
    ).squeeze(0)
    keep_count = max(1, len(text_chunks) - max(0, chunk_reduction_factor))
    keep_indices = select_ordered_indices(
        similarity.tolist(),
        keep_count,
        preserve_neighbors=True,
        neighbor_radius=2,
    )
    return similarity, keep_indices


def load_llava_model(model_path: Path, device_map: str) -> LlavaForConditionalGeneration:
    config = LlavaConfig.from_pretrained(model_path, local_files_only=True)
    model_kwargs: dict[str, Any] = {
        'config': config,
        'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
        'low_cpu_mem_usage': True,
        'local_files_only': True,
    }
    if device_map == 'auto':
        model_kwargs['device_map'] = 'auto'
    model = LlavaForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
    if device_map != 'auto':
        model = model.to(device_map)
    model.eval()
    return model


def load_multi_page_image_features(
    rendered_images: list[str],
    model: LlavaForConditionalGeneration,
) -> torch.Tensor:
    page_tensors = [preprocess_for_llava(Path(image_path)) for image_path in rendered_images]
    image_tensor = torch.stack(page_tensors, dim=0)
    model_device = next(model.parameters()).device
    image_tensor = image_tensor.to(model_device, dtype=model.dtype)

    with torch.inference_mode():
        image_features = model.get_image_features(pixel_values=image_tensor)

    if image_features.ndim == 2:
        image_features = image_features.unsqueeze(0)
    return image_features


def aggregate_visual_features(image_features: torch.Tensor, page_keep_count: int) -> tuple[torch.Tensor, list[int]]:
    page_count = image_features.shape[0]
    if page_count == 1:
        return image_features, [0]

    page_summary = image_features.mean(dim=1)
    page_norm = torch.nn.functional.normalize(page_summary, dim=-1)
    page_scores = torch.matmul(page_norm, page_norm.mean(dim=0))
    keep_count = max(1, min(page_count, page_keep_count))
    selected_pages = select_ordered_indices(
        page_scores.tolist(),
        keep_count,
        preserve_neighbors=True,
        neighbor_radius=1,
    )
    return image_features[selected_pages], selected_pages


def run_pipeline(
    context: str,
    question: str,
    render_config: Path,
    output_dir: Path,
    llava_path: Path,
    visual_reduction_factor: int,
    language_reduction_factor: int,
    chunk_lines: int,
    chunk_reduction_factor: int,
    device_map: str,
    max_units: int,
    neighbor_window: int,
    max_sentences_per_unit: int,
    max_chars_per_page: int,
    max_render_pages: int,
    answer_model_path: str | None,
    answer_device: str,
    answer_max_new_tokens: int,
) -> tuple[PipelineArtifacts, dict[str, Any]]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    render_artifacts = render_query_guided_pngs(
        context=context,
        question=question,
        output_dir=output_dir,
        unique_id='glyph_llava_ficoco',
        config_path=render_config,
        retrieval_max_units=max_units,
        retrieval_neighbor_window=neighbor_window,
        retrieval_max_sentences_per_unit=max_sentences_per_unit,
        readability_policy=ReadabilityPolicy(
            max_chars_per_page=max_chars_per_page,
            max_render_pages=max_render_pages,
        ),
    )
    rendered_images = render_artifacts.image_paths
    if not rendered_images:
        raise RuntimeError('Glyph rendering did not produce any images')

    model = load_llava_model(llava_path, device_map=device_map)
    image_features = load_multi_page_image_features(rendered_images, model)
    page_keep_count = max(1, int(math.ceil(len(rendered_images) * 0.75)))
    selected_page_features, selected_pages = aggregate_visual_features(image_features, page_keep_count)

    flattened_features = selected_page_features.reshape(1, -1, selected_page_features.shape[-1])
    visual_attention = make_attention(flattened_features)
    effective_reduction = min(visual_reduction_factor, max(0, flattened_features.shape[1] - 1))
    visual_result = reduce_visual_tokens(
        embeddings=flattened_features,
        attention=visual_attention,
        reduction_factor=effective_reduction,
        include_class_token=False,
    )

    chunks = chunk_text_for_ficoco(render_artifacts.selected_text, lines_per_chunk=chunk_lines)
    refined_visual = refine_visual_context_with_ficoco(
        reduced_visual_embeddings=visual_result.reduced_embeddings.detach().cpu().float(),
        text_chunks=chunks,
        language_reduction_factor=language_reduction_factor,
    )
    scores, selected_chunk_indices = select_relevant_chunks(
        refined_visual_embeddings=refined_visual,
        text_chunks=chunks,
        chunk_reduction_factor=chunk_reduction_factor,
    )
    selected_chunks = [chunks[index] for index in selected_chunk_indices]
    compressed_text = '\n\n'.join(selected_chunks)

    answer_engine = LlavaInferenceEngine(
        model_path=answer_model_path,
        device=answer_device,
        enable_ficoco=True,
        ficoco_keep_ratio=0.5,
        ficoco_min_pages=1,
        ficoco_max_pages=max_render_pages,
        render_config_path=str(render_config),
        query_guided_render=False,
        enable_thinking=False,
    )
    generated_answer = answer_engine.generate(
        question=question,
        image_paths=rendered_images,
        temperature=0.0,
        top_p=None,
        num_beams=1,
        max_new_tokens=answer_max_new_tokens,
        unique_id='glyph_llava_ficoco_answer',
    )

    report = {
        'question': question,
        'retrieval_selected_indices': render_artifacts.retrieval.selected_indices,
        'retrieval_total_units': render_artifacts.retrieval.total_units,
        'retrieval_scores': render_artifacts.retrieval.scores,
        'selected_text': render_artifacts.selected_text,
        'page_texts': render_artifacts.page_texts,
        'rendered_images': rendered_images,
        'readability_policy': {
            'max_chars_per_page': max_chars_per_page,
            'max_render_pages': max_render_pages,
        },
        'selected_page_indices': selected_pages,
        'llava_model_path': str(llava_path),
        'answer_model_path': answer_engine.model_path,
        'visual_tokens_before': int(flattened_features.shape[1]),
        'visual_tokens_after': int(visual_result.reduced_embeddings.shape[1]),
        'visual_tokens_after_ficoco': int(refined_visual.shape[1]),
        'chunk_count_before': len(chunks),
        'chunk_count_after': len(selected_chunks),
        'selected_chunk_indices': selected_chunk_indices,
        'chunk_scores': [round(float(scores[index]), 6) for index in range(scores.numel())],
        'ficoco_selected_text': compressed_text,
        'generator_usage': answer_engine.last_usage,
        'generator_page_report': answer_engine.last_ficoco_report,
        'generated_answer': generated_answer,
    }
    artifacts = PipelineArtifacts(
        rendered_images=rendered_images,
        selected_text=render_artifacts.selected_text,
        visual_embedding_shape=tuple(flattened_features.shape),
        reduced_visual_shape=tuple(visual_result.reduced_embeddings.shape),
        refined_visual_shape=tuple(refined_visual.shape),
        chunk_count_before=len(chunks),
        chunk_count_after=len(selected_chunks),
        selected_chunk_indices=selected_chunk_indices,
        generated_answer=generated_answer,
    )
    return artifacts, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run query-guided Glyph rendering, LLaVA reading, CLIP encoding, FiCoCo compression, and answer generation.',
    )
    parser.add_argument('--input-file', type=Path, required=True)
    parser.add_argument('--query', type=str, default=None)
    parser.add_argument('--query-file', type=Path, default=None)
    parser.add_argument('--render-config', type=Path, default=DEFAULT_RENDER_CONFIG)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--llava-path', type=Path, default=DEFAULT_LLAVA_PATH)
    parser.add_argument('--answer-model-path', type=str, default=None)
    parser.add_argument('--answer-device', default='cuda')
    parser.add_argument('--device-map', default='auto')
    parser.add_argument('--visual-reduction-factor', type=int, default=64)
    parser.add_argument('--language-reduction-factor', type=int, default=32)
    parser.add_argument('--chunk-lines', type=int, default=30)
    parser.add_argument('--chunk-reduction-factor', type=int, default=1)
    parser.add_argument('--retrieval-max-units', type=int, default=6)
    parser.add_argument('--retrieval-neighbor-window', type=int, default=1)
    parser.add_argument('--retrieval-max-sentences-per-unit', type=int, default=5)
    parser.add_argument('--max-chars-per-page', type=int, default=2200)
    parser.add_argument('--max-render-pages', type=int, default=3)
    parser.add_argument('--answer-max-new-tokens', type=int, default=256)
    parser.add_argument('--report-file', type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    context = read_text(args.input_file)
    question = _read_query(args.query_file, args.query)
    artifacts, report = run_pipeline(
        context=context,
        question=question,
        render_config=args.render_config,
        output_dir=args.output_dir,
        llava_path=args.llava_path,
        visual_reduction_factor=args.visual_reduction_factor,
        language_reduction_factor=args.language_reduction_factor,
        chunk_lines=args.chunk_lines,
        chunk_reduction_factor=args.chunk_reduction_factor,
        device_map=args.device_map,
        max_units=args.retrieval_max_units,
        neighbor_window=args.retrieval_neighbor_window,
        max_sentences_per_unit=args.retrieval_max_sentences_per_unit,
        max_chars_per_page=args.max_chars_per_page,
        max_render_pages=args.max_render_pages,
        answer_model_path=args.answer_model_path,
        answer_device=args.answer_device,
        answer_max_new_tokens=args.answer_max_new_tokens,
    )

    print('[1/6] Query-guided text filtering complete')
    print(f"Selected text chars: {len(artifacts.selected_text)}")
    print('[2/6] Related content rendered to PNG')
    print(f'Rendered pages: {len(artifacts.rendered_images)}')
    print('[3/6] Readability-aware rendering applied')
    print(f"Selected pages: {report['selected_page_indices']}")
    print('[4/6] LLaVA read images + question and CLIP encoded visual features')
    print(f"Visual embeddings: {artifacts.visual_embedding_shape} -> {artifacts.reduced_visual_shape}")
    print('[5/6] OCR-aware / query-aware FiCoCo compression complete')
    print(
        f"Chunks kept: {artifacts.chunk_count_after}/{artifacts.chunk_count_before} "
        f"(indices={artifacts.selected_chunk_indices})"
    )
    print('[6/6] LLM answer generation complete')
    print(artifacts.generated_answer.strip())

    if args.report_file is not None:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Report written to {args.report_file}')


if __name__ == '__main__':
    main()
