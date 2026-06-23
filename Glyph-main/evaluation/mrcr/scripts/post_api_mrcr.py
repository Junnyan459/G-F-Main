#!/usr/bin/env python3
"""Run MRCR evaluation with the local Glyph model."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = REPO_ROOT / 'evaluation'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glyph.inference import LlavaInferenceEngine  # noqa: E402


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_file',
        default='/root/autodl-tmp/Data/mrcr/data/processed_2needle_0-128k.jsonl',
    )
    parser.add_argument('--output_dir', default='/root/autodl-tmp/Data/mrcr/results')
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--model-base', default=None)
    parser.add_argument('--conv-mode', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--load-8bit', action='store_true')
    parser.add_argument('--load-4bit', action='store_true')
    parser.add_argument('--enable-ficoco', action='store_true', help='Enable FiCoCo-style page filtering before generation')
    parser.add_argument('--ficoco-keep-ratio', type=float, default=0.5, help='Fraction of rendered pages to keep when FiCoCo-style filtering is enabled')
    parser.add_argument('--ficoco-min-pages', type=int, default=1, help='Minimum number of pages to keep when FiCoCo-style filtering is enabled')
    parser.add_argument('--ficoco-max-pages', type=int, default=None, help='Maximum number of pages to keep when FiCoCo-style filtering is enabled')
    parser.add_argument('--ficoco-verbose', action='store_true', help='Print selected page indices when FiCoCo-style filtering is enabled')
    parser.add_argument('--temperature', type=float, default=0.0001)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--pool_size', type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(args)


def read_jsonl(path: Path, limit: Optional[int] = None) -> List[dict]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def clean_prediction(text: str) -> str:
    text = (text or '').strip()
    if '</think>' in text:
        text = text.split('</think>')[-1].strip()
    elif '<think>' in text:
        prefix = text.split('<think>', 1)[0].strip()
        if not prefix:
            return ''
        text = prefix
    text = re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|[^|]+\|>', '', text)
    if text.endswith('<|user|>'):
        text = text[:-8].rstrip()
    return re.sub(r'[\x00-\x1f]', '\n', text).strip()


def debug_path_issue(path_str):
    if isinstance(path_str, str):
        return path_str.strip().replace('\n', '').replace('\r', '').replace('\t', '')
    return path_str


def existing_image_paths(item: dict) -> List[str]:
    paths = debug_path_issue(item.get('image_paths', []))
    if isinstance(paths, str) and os.path.isdir(paths):
        paths = [os.path.join(paths, name) for name in sorted(os.listdir(paths))]
    elif isinstance(paths, str):
        paths = [paths]
    elif isinstance(paths, list):
        paths = [debug_path_issue(path) for path in paths]
    else:
        paths = []
    return [str(path).strip() for path in paths if path and os.path.exists(str(path).strip())]


def grade_response(response: str, answer: str, random_string_to_prepend: str) -> float:
    if not response.startswith(random_string_to_prepend):
        return 0.0
    response_clean = response[len(random_string_to_prepend):].replace('<|user|>', '')
    answer_clean = answer[len(random_string_to_prepend):] if answer.startswith(random_string_to_prepend) else answer
    return float(SequenceMatcher(None, response_clean, answer_clean).ratio())


def extract_category_from_unique_id(unique_id: str) -> str:
    match = re.search(r'\d+needle_\d+k_\d+k', unique_id or '')
    return match.group(0) if match else 'unknown'


def build_engine(args: argparse.Namespace) -> LlavaInferenceEngine:
    return LlavaInferenceEngine(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device=args.device,
        enable_ficoco=args.enable_ficoco,
        ficoco_keep_ratio=args.ficoco_keep_ratio,
        ficoco_min_pages=args.ficoco_min_pages,
        ficoco_max_pages=args.ficoco_max_pages,
        ficoco_verbose=args.ficoco_verbose,
    )


def generate(engine: LlavaInferenceEngine, prompt: str, image_paths: Sequence[str], args: argparse.Namespace) -> str:
    return clean_prediction(
        engine.generate(
            question=prompt,
            image_paths=list(image_paths),
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
        )
    )


def evaluate_results(rows: List[dict]) -> dict:
    category_results: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        category_results[row['category']].append(row)

    evaluation_results: dict[str, dict] = {}
    total_score = 0.0
    total_count = 0
    for category, category_items in sorted(category_results.items()):
        category_scores = [item['score'] for item in category_items]
        avg_score = sum(category_scores) / len(category_scores)
        evaluation_results[category] = {
            'avg_score': avg_score,
            'count': len(category_items),
            'scores': category_scores,
            'items': category_items,
        }
        total_score += sum(category_scores)
        total_count += len(category_scores)

    if total_count > 0:
        evaluation_results['overall'] = {
            'avg_score': total_score / total_count,
            'total_count': total_count,
        }
    return evaluation_results


def build_detailed_results(evaluation: dict) -> dict:
    detailed_results = {}
    for category, category_data in evaluation.items():
        if category == 'overall' or 'items' not in category_data:
            continue
        detailed_results[category] = {
            'avg_score': category_data['avg_score'],
            'count': category_data['count'],
            'individual_scores': [
                {
                    'unique_id': item['unique_id'],
                    'prediction': item['prediction'],
                    'answer': item['answer'],
                    'score': item['score'],
                }
                for item in category_data['items']
            ],
        }
    return detailed_results


def main() -> None:
    args = parse_args()
    input_file = Path(args.input_file)
    if not input_file.exists():
        raise FileNotFoundError(f'Input file does not exist: {input_file}')

    rows = read_jsonl(input_file, limit=args.limit)
    engine = build_engine(args)
    results = []
    for index, item in enumerate(tqdm(rows, desc='MRCR')):
        prompt = str(item.get('question', '')).strip()
        image_paths = existing_image_paths(item)
        prediction = generate(engine, prompt, image_paths, args)
        answer = str(item.get('answer', ''))
        prefix = str(item.get('random_string_to_prepend', ''))
        results.append(
            {
                'index': index,
                'unique_id': item.get('unique_id', 'unknown'),
                'category': extract_category_from_unique_id(str(item.get('unique_id', ''))),
                'prediction': prediction,
                'answer': answer,
                'usage': 'local',
                'random_string_to_prepend': prefix,
                'score': grade_response(prediction, answer, prefix),
                'timestamp': time.time(),
            }
        )

    evaluation = evaluate_results(results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    append_jsonl(output_dir / 'predictions_with_scores.jsonl', results)

    with (output_dir / 'evaluation_by_category.json').open('w', encoding='utf-8') as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)

    with (output_dir / 'detailed_results_by_category.json').open('w', encoding='utf-8') as f:
        json.dump(build_detailed_results(evaluation), f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} predictions to {output_dir / 'predictions_with_scores.jsonl'}")
    print(f"Saved evaluation to {output_dir / 'evaluation_by_category.json'}")


if __name__ == '__main__':
    main()
