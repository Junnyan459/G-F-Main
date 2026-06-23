from __future__ import annotations

import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Sequence

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from glyph_ficoco.selection import contiguous_ranges, enforce_protected_indices, select_ordered_indices
from glyph.pipeline import ReadabilityPolicy, render_query_guided_pngs


REPO_ROOT = Path(__file__).resolve().parents[1]


MULTI_DOC_KEYWORDS = (
    'given passages',
    'given documents',
    'following are given passages',
    'following are given documents',
    'based on the given passages',
    'based on the given documents',
)
MULTI_DOC_BUDGET_RATIO = 0.18
MULTI_DOC_MAX_PAGES = 3
NON_MULTI_DOC_MAX_PAGES = 2
MULTI_DOC_RETRIEVE_K = 3
NON_MULTI_DOC_RETRIEVE_K = 2
DEFAULT_RENDER_ROOT = Path('/tmp/glyph_query_guided_render')
STOPWORDS = {
    'the', 'a', 'an', 'of', 'to', 'and', 'or', 'in', 'on', 'for', 'with', 'by', 'is', 'are', 'was', 'were',
    'be', 'based', 'given', 'passages', 'documents', 'question', 'answer', 'only', 'from', 'which', 'who',
    'what', 'when', 'where', 'why', 'how', 'do', 'does', 'did', 'me', 'my', 'your', 'their', 'his', 'her',
}


def _candidate_model_paths(requested: Optional[str]) -> list[str]:
    candidates: list[str] = []
    if requested:
        candidates.append(str(requested))

    env_model_path = os.environ.get("GLYPH_MODEL_PATH")
    if env_model_path:
        candidates.append(env_model_path)

    candidates.extend(
        [
            str(Path('/root/autodl-tmp/Glyph')),
            str(REPO_ROOT / 'Model' / 'llava-v1.5'),
            str(Path('/root/autodl-tmp/llava')),
        ]
    )

    unique_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    return unique_candidates


def resolve_model_path(requested: Optional[str]) -> str:
    for candidate in _candidate_model_paths(requested):
        if Path(candidate).exists():
            return candidate
    if requested:
        return requested
    return str(Path('/root/autodl-tmp/Glyph'))


def _tokenize_query_terms(text: str) -> list[str]:
    terms = re.findall(r"[A-Za-z0-9']+", (text or '').lower())
    return [term for term in terms if len(term) > 2 and term not in STOPWORDS]


def _page_text_path(image_path: str) -> Optional[Path]:
    try:
        page_path = Path(image_path)
        return page_path.parent / 'page_texts.json'
    except TypeError:
        return None


class LlavaInferenceEngine:
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_base: Optional[str] = None,
        conv_mode: Optional[str] = None,
        load_8bit: bool = False,
        load_4bit: bool = False,
        device: str = 'cuda',
        enable_ficoco: bool = False,
        ficoco_keep_ratio: float = 0.5,
        ficoco_min_pages: int = 1,
        ficoco_max_pages: Optional[int] = None,
        ficoco_verbose: bool = False,
        enable_thinking: bool = True,
        render_config_path: Optional[str] = None,
        query_guided_render: bool = True,
        retrieval_max_units: int = 6,
        retrieval_neighbor_window: int = 1,
        retrieval_max_sentences_per_unit: int = 5,
        readability_max_chars_per_page: int = 2200,
        readability_max_render_pages: int = 3,
    ) -> None:
        del model_base
        del conv_mode

        self.model_path = resolve_model_path(model_path)
        self.load_8bit = load_8bit
        self.load_4bit = load_4bit
        self.device = device
        self.enable_ficoco = enable_ficoco
        self.ficoco_keep_ratio = ficoco_keep_ratio
        self.ficoco_min_pages = ficoco_min_pages
        self.ficoco_max_pages = ficoco_max_pages
        self.ficoco_verbose = ficoco_verbose
        self.enable_thinking = enable_thinking
        self.render_config_path = render_config_path
        self.query_guided_render = query_guided_render
        self.retrieval_max_units = retrieval_max_units
        self.retrieval_neighbor_window = retrieval_neighbor_window
        self.retrieval_max_sentences_per_unit = retrieval_max_sentences_per_unit
        self.readability_policy = ReadabilityPolicy(
            max_chars_per_page=readability_max_chars_per_page,
            max_render_pages=readability_max_render_pages,
        )
        self.local_files_only = Path(self.model_path).exists()
        self.last_usage: Optional[dict] = None
        self.last_render_report: Optional[dict] = None
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=self.local_files_only,
        )
        self.model = self._load_model()
        self.model.eval()
        self.last_ficoco_report: Optional[dict] = None

    def _model_dtype(self) -> torch.dtype:
        if self.device == 'cpu' or not torch.cuda.is_available():
            return torch.float32
        return torch.bfloat16

    def _load_model(self):
        model_kwargs = {
            'pretrained_model_name_or_path': self.model_path,
            'local_files_only': self.local_files_only,
        }
        if self.device == 'cpu' or not torch.cuda.is_available():
            model_kwargs['torch_dtype'] = torch.float32
        else:
            model_kwargs['torch_dtype'] = self._model_dtype()
            if self.load_4bit:
                model_kwargs['load_in_4bit'] = True
                model_kwargs['device_map'] = 'auto'
            elif self.load_8bit:
                model_kwargs['load_in_8bit'] = True
                model_kwargs['device_map'] = 'auto'
            elif self.device == 'cuda':
                model_kwargs['device_map'] = 'auto'
            else:
                model_kwargs['device_map'] = {'': self.device}

        model = AutoModelForImageTextToText.from_pretrained(**model_kwargs)
        if self.device == 'cpu' or not torch.cuda.is_available():
            model = model.to('cpu')
        return model

    def _unwrap_image_features(self, output):
        if isinstance(output, tuple):
            if not output:
                raise ValueError('Empty image feature output')
            return output[0]
        return output

    def _image_feature_mean(self, image_path: str) -> torch.Tensor:
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'url': str(image_path)},
                    {'type': 'text', 'text': ''},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors='pt',
        )
        pixel_values = inputs['pixel_values'].to(self.model.device)
        image_grid_thw = inputs['image_grid_thw'].to(self.model.device)
        with torch.inference_mode():
            features = self._unwrap_image_features(
                self.model.get_image_features(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )
            )
        return features.mean(dim=0).float().cpu()

    def _text_feature_mean(self, question: str) -> torch.Tensor:
        text_inputs = self.processor.tokenizer(
            question or '',
            return_tensors='pt',
            add_special_tokens=False,
        )
        input_ids = text_inputs['input_ids'].to(self.model.device)
        with torch.inference_mode():
            text_embeds = self.model.get_input_embeddings()(input_ids)
        return text_embeds.mean(dim=1).squeeze(0).float().cpu()

    def _is_multi_document_prompt(self, question: str) -> bool:
        normalized = (question or '').lower()
        return any(keyword in normalized for keyword in MULTI_DOC_KEYWORDS)

    def _protected_page_indices(self, total_pages: int, multi_document: bool) -> list[int]:
        if total_pages <= 1:
            return [0] if total_pages == 1 else []

        protected = {0}
        if multi_document and total_pages >= 3:
            protected.add(total_pages - 1)
        return sorted(protected)

    def _target_keep_count(self, total_pages: int, multi_document: bool) -> int:
        if multi_document:
            ratio_keep = int(math.ceil(total_pages * min(self.ficoco_keep_ratio, MULTI_DOC_BUDGET_RATIO)))
            keep_count = max(self.ficoco_min_pages, ratio_keep)
            return min(keep_count, MULTI_DOC_MAX_PAGES, total_pages)

        ratio_keep = int(math.ceil(total_pages * self.ficoco_keep_ratio))
        keep_count = max(self.ficoco_min_pages, ratio_keep)
        return min(keep_count, NON_MULTI_DOC_MAX_PAGES, total_pages)

    def _retrieve_k(self, multi_document: bool, total_pages: int) -> int:
        cap = MULTI_DOC_RETRIEVE_K if multi_document else NON_MULTI_DOC_RETRIEVE_K
        return min(total_pages, cap)

    def _load_page_texts(self, image_paths: Sequence[str]) -> list[str]:
        if not image_paths:
            return []
        page_text_file = _page_text_path(image_paths[0])
        if page_text_file is None or not page_text_file.exists():
            return [''] * len(image_paths)
        try:
            page_texts = json.loads(page_text_file.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return [''] * len(image_paths)
        if not isinstance(page_texts, list):
            return [''] * len(image_paths)
        return [str(page_texts[index]) if index < len(page_texts) else '' for index in range(len(image_paths))]

    def _lexical_page_scores(self, question: str, image_paths: Sequence[str]) -> list[float]:
        terms = _tokenize_query_terms(question)
        if not terms:
            return [0.0] * len(image_paths)

        page_texts = self._load_page_texts(image_paths)
        scores: list[float] = []
        for page_text in page_texts:
            lowered = page_text.lower()
            term_hits = sum(1 for term in terms if term in lowered)
            early_hits = sum(1 for term in terms if term in lowered[:600])
            scores.append(term_hits + (0.25 * early_hits))
        return scores

    def _retrieve_candidate_indices(self, question: str, image_paths: Sequence[str], multi_document: bool) -> list[int]:
        total_pages = len(image_paths)
        if total_pages <= 1:
            return list(range(total_pages))

        lexical_scores = self._lexical_page_scores(question, image_paths)
        if not any(lexical_scores):
            return list(range(total_pages))

        keep_count = self._retrieve_k(multi_document, total_pages)
        protected_indices = self._protected_page_indices(total_pages, multi_document)
        selected = select_ordered_indices(
            lexical_scores,
            max(keep_count, len(protected_indices)),
            preserve_neighbors=False,
            neighbor_radius=0,
        )
        return enforce_protected_indices(selected, protected_indices, lexical_scores, max(keep_count, len(protected_indices)))

    def _visual_scores_for_candidates(self, question: str, image_paths: Sequence[str], candidate_indices: Sequence[int]) -> list[float]:
        question_feature = self._text_feature_mean(question)
        question_feature = torch.nn.functional.normalize(question_feature, dim=0)
        scores: list[float] = []
        for index in candidate_indices:
            image_feature = self._image_feature_mean(image_paths[index])
            image_feature = torch.nn.functional.normalize(image_feature, dim=0)
            scores.append(float(torch.dot(image_feature, question_feature)))
        return scores

    def _build_selection_report(
        self,
        valid_paths: Sequence[str],
        scores: Sequence[float],
        selected_indices: Sequence[int],
        multi_document: bool,
        candidate_indices: Sequence[int],
    ) -> None:
        ranges = contiguous_ranges(selected_indices)
        self.last_ficoco_report = {
            'enabled': True,
            'multi_document_mode': multi_document,
            'total_pages': len(valid_paths),
            'candidate_indices': list(candidate_indices),
            'selected_indices': list(selected_indices),
            'selected_ranges': ranges,
            'selected_pages': [valid_paths[index] for index in selected_indices],
            'scores': list(scores),
            'coverage_ratio': round(len(selected_indices) / max(1, len(valid_paths)), 4),
        }

    def _select_relevant_pages(self, question: str, image_paths: Sequence[str]) -> list[str]:
        valid_paths = [str(path) for path in image_paths if path and os.path.exists(str(path))]
        if not self.enable_ficoco or len(valid_paths) <= 1:
            self.last_ficoco_report = None
            return valid_paths

        multi_document = self._is_multi_document_prompt(question)
        keep_count = self._target_keep_count(len(valid_paths), multi_document)
        if self.ficoco_max_pages is not None:
            keep_count = min(keep_count, max(1, self.ficoco_max_pages))

        protected_indices = self._protected_page_indices(len(valid_paths), multi_document)
        keep_count = max(keep_count, len(protected_indices))

        if keep_count >= len(valid_paths):
            selected_indices = list(range(len(valid_paths)))
            self._build_selection_report(valid_paths, [1.0] * len(valid_paths), selected_indices, multi_document, selected_indices)
            return valid_paths

        candidate_indices = self._retrieve_candidate_indices(question, valid_paths, multi_document)
        candidate_scores = self._visual_scores_for_candidates(question, valid_paths, candidate_indices)
        candidate_keep_count = min(keep_count, len(candidate_indices))

        local_selected = select_ordered_indices(
            candidate_scores,
            candidate_keep_count,
            preserve_neighbors=False,
            neighbor_radius=0,
        )
        full_scores = [0.0] * len(valid_paths)
        for index, score in zip(candidate_indices, candidate_scores):
            full_scores[index] = score
        selected_indices = sorted(candidate_indices[index] for index in local_selected)
        selected_indices = enforce_protected_indices(
            selected_indices,
            protected_indices,
            full_scores,
            keep_count,
        )
        selected_pages = [valid_paths[index] for index in selected_indices]
        self._build_selection_report(valid_paths, full_scores, selected_indices, multi_document, candidate_indices)

        if self.ficoco_verbose:
            candidate_str = ','.join(str(index) for index in candidate_indices)
            selected_str = ','.join(str(index) for index in selected_indices)
            print(
                f'[FiCoCo] candidates={candidate_str}; selected={selected_str}; '
                f'multi_document={multi_document}; keep={len(selected_pages)}/{len(valid_paths)}'
            )
        return selected_pages

    def _render_from_context(self, retrieval_query: str, context: str, unique_id: Optional[str]) -> list[str]:
        render_root = DEFAULT_RENDER_ROOT
        render_root.mkdir(parents=True, exist_ok=True)
        render_id = unique_id or f'render_{abs(hash((retrieval_query, context))) % 10**12}'
        render_target = render_root / render_id
        if render_target.exists():
            shutil.rmtree(render_target)

        artifacts = render_query_guided_pngs(
            context=context,
            question=retrieval_query,
            output_dir=render_root,
            unique_id=render_id,
            config_path=self.render_config_path,
            retrieval_max_units=self.retrieval_max_units,
            retrieval_neighbor_window=self.retrieval_neighbor_window,
            retrieval_max_sentences_per_unit=self.retrieval_max_sentences_per_unit,
            readability_policy=self.readability_policy,
        )
        self.last_render_report = {
            'selected_indices': artifacts.retrieval.selected_indices,
            'total_units': artifacts.retrieval.total_units,
            'retrieval_scores': artifacts.retrieval.scores,
            'selected_text': artifacts.selected_text,
            'page_texts': artifacts.page_texts,
            'render_config': artifacts.render_config,
            'metadata_path': artifacts.metadata_path,
            'image_paths': artifacts.image_paths,
        }
        return artifacts.image_paths

    def generate(
        self,
        question: str,
        context: Optional[str] = None,
        image_paths: Optional[Sequence[str]] = None,
        temperature: float = 0.0001,
        top_p: Optional[float] = None,
        num_beams: int = 1,
        max_new_tokens: int = 1024,
        unique_id: Optional[str] = None,
        retrieval_query: Optional[str] = None,
    ) -> str:
        retrieval_text = (retrieval_query or question or '').strip()
        valid_image_paths = list(image_paths or [])
        if self.query_guided_render and context and not valid_image_paths:
            valid_image_paths = self._render_from_context(retrieval_query=retrieval_text, context=context, unique_id=unique_id)
        else:
            self.last_render_report = None

        selected_image_paths = self._select_relevant_pages(retrieval_text, valid_image_paths)

        content = []
        for image_path in selected_image_paths:
            if image_path and os.path.exists(image_path):
                content.append({'type': 'image', 'url': str(image_path)})
        question_text = (question or '').strip()
        if not self.enable_thinking and not question_text.endswith('/nothink'):
            question_text = f'{question_text}\n/nothink'
        content.append({'type': 'text', 'text': question_text})
        messages = [{'role': 'user', 'content': content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors='pt',
        ).to(self.model.device)

        generation_kwargs = {
            'max_new_tokens': max_new_tokens,
            'num_beams': num_beams,
            'repetition_penalty': 1.1,
        }
        if temperature and temperature > 0:
            generation_kwargs['do_sample'] = True
            generation_kwargs['temperature'] = temperature
            if top_p is not None:
                generation_kwargs['top_p'] = top_p
        else:
            generation_kwargs['do_sample'] = False

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        prompt_length = inputs['input_ids'].shape[1]
        completion_tokens = generated_ids[0].shape[0] - prompt_length
        self.last_usage = {
            'prompt_tokens': int(prompt_length),
            'completion_tokens': int(completion_tokens),
            'total_tokens': int(prompt_length + completion_tokens),
        }
        return self.processor.decode(
            generated_ids[0][prompt_length:],
            skip_special_tokens=False,
        )
