from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

STOPWORDS = {
    'the', 'a', 'an', 'of', 'to', 'and', 'or', 'in', 'on', 'for', 'with', 'by', 'is', 'are', 'was', 'were',
    'be', 'based', 'given', 'passages', 'documents', 'question', 'answer', 'only', 'from', 'which', 'who',
    'what', 'when', 'where', 'why', 'how', 'do', 'does', 'did', 'me', 'my', 'your', 'their', 'his', 'her',
    'wife', 'husband', 'son', 'daughter', 'father', 'mother', 'child', 'children', 'born', 'birth', 'birthplace',
    'director', 'film', 'movie', 'work', 'works', 'worked', 'worked', 'located', 'location', 'grandchild', 'grandson', 'granddaughter',
}
DOC_HEADER_RE = re.compile(r'^(Passage|Document)\s+\d+\s*:?$', re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
WORD_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")
MULTI_DOC_QUERY_HINTS = ('given passages', 'given documents', 'passages', 'documents')



@dataclass
class RetrievalResult:
    selected_text: str
    selected_indices: list[int]
    total_units: int
    scores: list[float]


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize('NFKD', text or '')
    without_marks = ''.join(ch for ch in decomposed if not unicodedata.combining(ch))
    return without_marks.lower()


def _normalized_words(text: str) -> list[str]:
    return WORD_RE.findall(normalize_text(text))


def tokenize_query_terms(text: str) -> list[str]:
    terms = _normalized_words(text)
    return [
        term for term in terms
        if len(term) > 2 and term not in STOPWORDS
    ]


def query_phrases(text: str, min_size: int = 2, max_size: int = 3) -> list[str]:
    words = [word for word in _normalized_words(text) if word not in STOPWORDS]
    phrases: list[str] = []
    for size in range(min_size, max_size + 1):
        for start in range(0, max(0, len(words) - size + 1)):
            phrase_words = words[start:start + size]
            if any(len(word) > 2 for word in phrase_words):
                phrases.append(' '.join(phrase_words))
    return phrases


def matched_query_terms(question: str, unit: str) -> set[str]:
    terms = tokenize_query_terms(question)
    lowered = normalize_text(unit)
    return {term for term in terms if term in lowered}


def _clean_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _parse_unit(unit: str) -> tuple[str, str, list[str]]:
    lines = _clean_lines(unit)
    header = ''
    if lines and DOC_HEADER_RE.match(lines[0]):
        header = lines[0]
        lines = lines[1:]

    title = ''
    if lines and len(lines[0]) <= 140:
        title = lines[0]
        lines = lines[1:]
    return header, title, lines


def _format_unit(header: str, title: str, body: str) -> str:
    blocks = [part for part in (header, title, body.strip()) if part]
    return '\n'.join(blocks).strip()


def split_context_units(text: str) -> list[str]:
    if not text.strip():
        return []

    lines = text.splitlines()
    units: list[list[str]] = []
    current: list[str] = []
    saw_doc_headers = any(DOC_HEADER_RE.match(line.strip()) for line in lines)

    for line in lines:
        stripped = line.strip()
        if current and DOC_HEADER_RE.match(stripped):
            units.append(current)
            current = [line]
            continue
        if not saw_doc_headers and not stripped and current:
            units.append(current)
            current = []
            continue
        current.append(line)
    if current:
        units.append(current)

    merged = ['\n'.join(unit).strip() for unit in units if '\n'.join(unit).strip()]
    return merged or [text.strip()]


def score_unit(question: str, unit: str) -> float:
    terms = tokenize_query_terms(question)
    if not terms:
        return 0.0

    phrases = query_phrases(question)
    _header, title, body_lines = _parse_unit(unit)
    title_text = normalize_text(title)
    body_text = normalize_text(' '.join(body_lines))
    early_text = body_text[:900]

    title_term_hits = sum(1 for term in terms if term in title_text)
    body_term_hits = sum(1 for term in terms if term in body_text)
    early_hits = sum(1 for term in terms if term in early_text)
    title_phrase_hits = sum(1 for phrase in phrases if phrase in title_text)
    body_phrase_hits = sum(1 for phrase in phrases if phrase in body_text)
    exact_mentions = sum(body_text.count(term) for term in terms)

    return (
        (2.6 * title_term_hits)
        + body_term_hits
        + (0.35 * early_hits)
        + (2.8 * title_phrase_hits)
        + (1.35 * body_phrase_hits)
        + (0.12 * exact_mentions)
    )


def _truncate_unit_around_terms(unit: str, question: str, max_sentences: int) -> str:
    header, title, body_lines = _parse_unit(unit)
    body = ' '.join(body_lines)
    if not body:
        return _format_unit(header, title, '')

    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(body) if sentence.strip()]
    if len(sentences) <= max_sentences:
        return _format_unit(header, title, body)

    terms = tokenize_query_terms(question)
    if not terms:
        selected_indices = list(range(min(max_sentences, len(sentences))))
    else:
        scored = []
        for index, sentence in enumerate(sentences):
            lowered = normalize_text(sentence)
            hits = sum(1 for term in terms if term in lowered)
            exact = sum(lowered.count(term) for term in terms)
            score = hits + (0.15 * exact)
            if index < 2:
                score += 0.4
            scored.append((index, score))

        chosen = {0}
        ranked_sentences = sorted(scored, key=lambda item: (item[1], -item[0]), reverse=True)
        for index, score in ranked_sentences:
            if score <= 0 and len(chosen) >= min(2, max_sentences):
                break
            chosen.add(index)
            if len(chosen) >= max_sentences:
                break
        selected_indices = sorted(chosen)[:max_sentences]

    clipped_body = ' '.join(sentences[index] for index in selected_indices)
    return _format_unit(header, title, clipped_body)


def select_relevant_context(
    context: str,
    question: str,
    *,
    max_units: int = 6,
    neighbor_window: int = 1,
    max_sentences_per_unit: int = 5,
    max_chars: int | None = 2200,
) -> RetrievalResult:
    units = split_context_units(context)
    if not units:
        return RetrievalResult('', [], 0, [])
    if len(units) <= max_units:
        clipped_units = [_truncate_unit_around_terms(unit, question, max_sentences_per_unit) for unit in units]
        selected_text = '\n\n'.join(clipped_units)
        if max_chars is not None and len(selected_text) > max_chars:
            selected_text = selected_text[:max_chars].rsplit(' ', 1)[0]
        return RetrievalResult(selected_text, list(range(len(units))), len(units), [0.0] * len(units))

    scores = [score_unit(question, unit) for unit in units]
    ranked = sorted(range(len(units)), key=lambda index: (scores[index], -index), reverse=True)

    normalized_question = normalize_text(question)
    has_passage_headers = sum(1 for unit in units if DOC_HEADER_RE.match(_clean_lines(unit)[0]) if _clean_lines(unit)) >= 3
    is_multi_doc = has_passage_headers or any(hint in normalized_question for hint in MULTI_DOC_QUERY_HINTS)
    seed_target = min(max_units, len(units))
    if is_multi_doc and seed_target > 3:
        seed_target = max(3, min(seed_target - 2, 4))

    selected: list[int] = []
    covered_terms: set[str] = set()
    while len(selected) < seed_target:
        best_index = None
        best_key = None
        for index in ranked:
            if index in selected:
                continue
            unit_terms = matched_query_terms(question, units[index])
            new_terms = len(unit_terms - covered_terms)
            title_bonus = 1 if _parse_unit(units[index])[1] and unit_terms else 0
            diversity_bonus = 0.0 if not selected else min(0.4, 0.1 * abs(index - selected[0]))
            key = (new_terms, title_bonus, scores[index] + diversity_bonus, -index)
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        if best_index is None:
            break
        selected.append(best_index)
        covered_terms.update(matched_query_terms(question, units[best_index]))

    expanded = set(selected)
    effective_neighbor_window = neighbor_window
    if is_multi_doc:
        effective_neighbor_window = min(neighbor_window, 1)

    if effective_neighbor_window > 0:
        seed_scores = {index: scores[index] for index in selected}
        if seed_scores:
            max_seed_score = max(seed_scores.values())
            neighbor_candidates: list[tuple[int, float]] = []
            for index in selected:
                if is_multi_doc and seed_scores[index] < max_seed_score * 0.55:
                    continue
                start = max(0, index - effective_neighbor_window)
                end = min(len(units), index + effective_neighbor_window + 1)
                for probe in range(start, end):
                    if probe not in expanded:
                        proximity = -abs(probe - index)
                        neighbor_candidates.append((probe, proximity))
            deduped: dict[int, float] = {}
            for probe, proximity in neighbor_candidates:
                deduped[probe] = max(deduped.get(probe, float('-inf')), proximity)
            ranked_neighbors = sorted(
                deduped,
                key=lambda index: (scores[index], deduped[index], -index),
                reverse=True,
            )
            for index in ranked_neighbors:
                if len(expanded) >= max_units:
                    break
                expanded.add(index)

    minimum_keep = min(max_units, len(units))
    if len(expanded) < minimum_keep:
        for index in ranked:
            expanded.add(index)
            if len(expanded) >= minimum_keep:
                break

    ordered_indices = sorted(expanded)[:max_units]
    clipped_units = [_truncate_unit_around_terms(units[index], question, max_sentences_per_unit) for index in ordered_indices]
    selected_text = '\n\n'.join(clipped_units)
    if max_chars is not None and len(selected_text) > max_chars:
        selected_text = selected_text[:max_chars].rsplit(' ', 1)[0]
    return RetrievalResult(selected_text, ordered_indices, len(units), scores)
