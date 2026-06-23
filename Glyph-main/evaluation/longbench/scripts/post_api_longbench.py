#!/usr/bin/env python3
"""Run LongBench predictions with the local query-guided Glyph pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = REPO_ROOT / 'evaluation'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glyph.inference import LlavaInferenceEngine  # noqa: E402

LONGBENCH_DATASETS = [
    'narrativeqa',
    'qasper',
    'multifieldqa_en',
    'multifieldqa_zh',
    'hotpotqa',
    '2wikimqa',
    'musique',
    'dureader',
    'gov_report',
    'qmsum',
    'multi_news',
    'vcsum',
    'trec',
    'triviaqa',
    'samsum',
    'lsht',
    'passage_count',
    'passage_retrieval_en',
    'passage_retrieval_zh',
    'lcc',
    'repobench-p',
]
LONGBENCH_E_DATASETS = [
    'qasper',
    'multifieldqa_en',
    'hotpotqa',
    '2wikimqa',
    'gov_report',
    'multi_news',
    'trec',
    'triviaqa',
    'samsum',
    'passage_count',
    'passage_retrieval_en',
    'lcc',
    'repobench-p',
]


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--e', action='store_true', help='Evaluate on LongBench-E')
    parser.add_argument('--use_image', action='store_true', help='Use pre-rendered image input instead of query-guided rendering')
    parser.add_argument('--input_dir', default=str(EVAL_ROOT / 'longbench' / 'rendered_images'))
    parser.add_argument('--output_dir', default=str(EVAL_ROOT / 'longbench' / 'pred'))
    parser.add_argument('--model_name', default='glyph_query_guided', help='Output subdirectory name')
    parser.add_argument('--output_file_suffix', default='_glyph_query_guided')
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--model-base', default=None)
    parser.add_argument('--conv-mode', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--load-8bit', action='store_true')
    parser.add_argument('--load-4bit', action='store_true')
    parser.add_argument('--enable-ficoco', action='store_true', help='Enable FiCoCo-style page filtering before generation')
    parser.add_argument('--ficoco-keep-ratio', type=float, default=0.5)
    parser.add_argument('--ficoco-min-pages', type=int, default=1)
    parser.add_argument('--ficoco-max-pages', type=int, default=None)
    parser.add_argument('--ficoco-verbose', action='store_true')
    parser.add_argument('--retrieval-max-units', type=int, default=6)
    parser.add_argument('--retrieval-neighbor-window', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--disable-thinking', action='store_true', default=True)
    parser.add_argument('--enable-thinking', dest='disable_thinking', action='store_false')
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--pool_size', type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(args)


def prediction_filename(dataset: str, suffix: str = '') -> str:
    return f'{dataset}{suffix}.jsonl'


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


def generate(
    engine: LlavaInferenceEngine,
    prompt: str,
    image_paths: Sequence[str],
    args: argparse.Namespace,
    *,
    context: str | None = None,
    unique_id: str | None = None,
    retrieval_query: str | None = None,
) -> str:
    return clean_prediction(
        engine.generate(
            question=prompt,
            context=context,
            image_paths=list(image_paths),
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            unique_id=unique_id,
            retrieval_query=retrieval_query,
        )
    )


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
        enable_thinking=not args.disable_thinking,
        query_guided_render=not args.use_image,
        retrieval_max_units=args.retrieval_max_units,
        retrieval_neighbor_window=args.retrieval_neighbor_window,
    )


def main() -> None:
    args = parse_args()
    prompt_file = EVAL_ROOT / 'longbench' / 'config' / 'dataset2vlmprompt.json'
    with prompt_file.open('r', encoding='utf-8') as f:
        dataset2prompt = json.load(f)

    output_root = Path(args.output_dir) / args.model_name
    output_root.mkdir(parents=True, exist_ok=True)
    engine = build_engine(args)

    datasets = LONGBENCH_E_DATASETS if args.e else LONGBENCH_DATASETS
    for dataset in datasets:
        input_file = Path(args.input_dir) / f'{dataset}.jsonl'
        output_file = output_root / prediction_filename(dataset, args.output_file_suffix)
        if not input_file.exists():
            print(f'Warning: input file does not exist: {input_file}')
            continue

        rows = read_jsonl(input_file, limit=args.limit)
        predictions = []
        for item in tqdm(rows, desc=f'LongBench/{dataset}'):
            prompt_key = item.get('dataset', dataset)
            if isinstance(prompt_key, str) and prompt_key.endswith('_e'):
                prompt_key = prompt_key[:-2]
            template = dataset2prompt.get(prompt_key, '{input}')

            raw_query = str(item.get('input', '')).strip()
            if args.use_image:
                prompt = template.replace('{input}', raw_query)
                image_paths = existing_image_paths(item)
                context = None
            else:
                context = str(item.get('context', '')).strip()
                prompt = template.replace('{input}', raw_query).replace('{context}', '')
                image_paths = []

            pred = generate(
                engine,
                prompt,
                image_paths,
                args,
                context=context,
                unique_id=str(item.get('_id') or item.get('id') or item.get('unique_id') or dataset),
                retrieval_query=raw_query,
            )
            row = {
                'pred': pred,
                'answers': item.get('answers', []),
                'all_classes': item.get('all_classes', []),
                'length': item.get('length', 0),
                'usage': engine.last_usage or {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
            }
            if engine.last_render_report is not None:
                row['render_report'] = engine.last_render_report
            if engine.last_ficoco_report is not None:
                row['ficoco_report'] = engine.last_ficoco_report
            predictions.append(row)

        append_jsonl(output_file, predictions)
        print(f'Saved {len(predictions)} predictions to {output_file}')


if __name__ == '__main__':
    main()
