#!/usr/bin/env python3
"""Run RULER evaluation with the local Glyph model."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = REPO_ROOT / 'evaluation'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glyph.inference import LlavaInferenceEngine  # noqa: E402


def string_match_part(preds, refs):
    recalls = [max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) for pred, ref in zip(preds, refs)]
    score = sum(recalls) / len(preds) * 100
    return round(score, 2), recalls


def string_match_all(preds, refs):
    recalls = [sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref) for pred, ref in zip(preds, refs)]
    score = sum(recalls) / len(preds) * 100
    return round(score, 2), recalls


tasks_base = {
    'niah': {
        'tokens_to_generate': 128,
        'template': """Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.
{context}
What are all the special magic {type_needle_v} for {query} mentioned in the provided text?""",
        'answer_prefix': """ The special magic {type_needle_v} for {query} mentioned in the provided text are""",
        'metric_fn': string_match_all,
    },
    'variable_tracking': {
        'tokens_to_generate': 30,
        'template': """Memorize and track the chain(s) of variable assignment hidden in the following text.

{context}
Question: Find all variables that are assigned the value {query} in the text above.""",
        'answer_prefix': """ Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables are assgined the value {query}, they are: """,
        'metric_fn': string_match_all,
    },
    'common_words_extraction': {
        'tokens_to_generate': 120,
        'template': """Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.
{context}
Question: What are the 10 most common words in the above list?""",
        'answer_prefix': """ Answer: The top 10 words that appear most often in the list are:""",
        'metric_fn': string_match_all,
    },
    'freq_words_extraction': {
        'tokens_to_generate': 50,
        'template': """Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words. {context}
Question: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text?""",
        'answer_prefix': """ Answer: According to the coded text above, the three most frequently appeared words are:""",
        'metric_fn': string_match_all,
    },
    'qa': {
        'tokens_to_generate': 32,
        'template': """Answer the question based on the given documents. Only give me the answer and do not output any other words.

The following are given documents.

{context}

Answer the question based on the given documents. Only give me the answer and do not output any other words.

Question: {query}""",
        'answer_prefix': """ Answer:""",
        'metric_fn': string_match_part,
    },
}


tasks_customized = {
    'niah_single_1': {'task': 'niah'},
    'niah_single_2': {'task': 'niah'},
    'niah_single_3': {'task': 'niah'},
    'niah_multikey_1': {'task': 'niah'},
    'niah_multikey_2': {'task': 'niah'},
    'niah_multikey_3': {'task': 'niah'},
    'niah_multivalue': {'task': 'niah'},
    'niah_multiquery': {'task': 'niah'},
    'vt': {'task': 'variable_tracking'},
    'cwe': {'task': 'common_words_extraction'},
    'fwe': {'task': 'freq_words_extraction'},
    'qa_1': {'task': 'qa'},
    'qa_2': {'task': 'qa'},
}


def postprocess_pred(predict_str: str):
    predict_str = predict_str.strip()
    np_pattern = re.compile(r'[\x00-\x1f]')
    predict_str = np_pattern.sub('\n', predict_str).strip()
    return predict_str


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/root/autodl-tmp/Data/ruler/data')
    parser.add_argument('--output_dir', default='/root/autodl-tmp/Data/ruler/results')
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
    parser.add_argument('--lengths', type=int, nargs='*', default=[4096, 8192, 16384, 32768, 65536, 126000])
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
    return postprocess_pred(text)


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


def evaluate_single_task(task_name, predictions, references):
    if task_name not in tasks_customized:
        print(f'警告：未知任务 {task_name}')
        return 0.0, []
    task_config = tasks_customized[task_name].copy()
    base_task = task_config['task']
    if base_task in tasks_base:
        task_config.update(tasks_base[base_task])
    processed_preds = [postprocess_pred(pred) for pred in predictions]
    if 'metric_fn' in task_config and callable(task_config['metric_fn']):
        score, recalls = task_config['metric_fn'](processed_preds, references)
        return score, recalls
    print(f'警告：任务 {task_name} 没有有效的评分函数')
    return 0.0, []


def process_data_item(data_item):
    prompt = str(data_item.get('question', '')).strip()
    image_paths = existing_image_paths(data_item)
    return prompt, image_paths


def process_single_item(engine: LlavaInferenceEngine, data_item: dict, index: int, args: argparse.Namespace):
    try:
        prompt, image_paths = process_data_item(data_item)
        pred_text = clean_prediction(
            engine.generate(
                question=prompt,
                image_paths=image_paths,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
            )
        )
        references = data_item.get('outputs', [])
        if isinstance(references, str):
            references = [references]
        return {
            'index': index,
            'task_name': data_item.get('task_name', 'unknown'),
            'prediction': pred_text,
            'references': references,
            'usage': 'local',
            'timestamp': time.time(),
        }
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def run_single_file(input_file: Path, output_dir: Path, engine: LlavaInferenceEngine, args: argparse.Namespace) -> None:
    data_list = read_jsonl(input_file, limit=args.limit)
    print(f'共有 {len(data_list)} 个数据项需要处理')

    results = []
    task_results = defaultdict(lambda: {'predictions': [], 'references': [], 'indices': []})
    for idx, data_item in enumerate(tqdm(data_list, desc=input_file.stem)):
        result = process_single_item(engine, data_item, idx, args)
        if result:
            results.append(result)
            task_name = result['task_name']
            task_results[task_name]['predictions'].append(result['prediction'])
            task_results[task_name]['references'].append(result['references'])
            task_results[task_name]['indices'].append(result['index'])

    results.sort(key=lambda x: x['index'])
    evaluation_results = {}
    summary_lines = []
    total_score = 0
    total_valid = 0
    print('\n开始评分...')
    for task_name, task_data in task_results.items():
        if task_name in ['niah_single_3', 'niah_multikey_3']:
            print(f'跳过任务: {task_name}')
            continue
        if len(task_data['predictions']) > 0:
            score, recalls = evaluate_single_task(
                task_name,
                task_data['predictions'],
                task_data['references'],
            )
            evaluation_results[task_name] = {
                'score': score,
                'valid': len(task_data['predictions']),
                'recalls': recalls,
            }
            total_score += score
            total_valid += len(task_data['predictions'])
            task_summary = f'任务 {task_name}: 得分 {score:.2f}, 有效样本 {len(task_data["predictions"])}'
            print(task_summary)
            summary_lines.append(task_summary)

    if len(evaluation_results) > 0:
        avg_score = total_score / len(evaluation_results)
        evaluation_results['average'] = {
            'score': avg_score,
            'valid': total_valid,
        }
        overall_summary = f'总体平均分: {avg_score:.2f} (有效样本总数 {total_valid})'
        print(f'\n{overall_summary}')
        summary_lines.append('')
        summary_lines.append(overall_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    append_jsonl(output_dir / 'predictions.jsonl', results)
    with (output_dir / 'evaluation.json').open('w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    if summary_lines:
        with (output_dir / 'evaluation_summary.txt').open('w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))

    print(f'\n最终结果已保存到 {output_dir}')
    print(f'预测结果: {output_dir / "predictions.jsonl"}')
    print(f'评分结果: {output_dir / "evaluation.json"}')


if __name__ == '__main__':
    args = parse_args()
    engine = build_engine(args)
    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)

    for lens in args.lengths:
        print(f'--- 开始处理长度: {lens} ---')
        output_dir = output_root / str(lens)
        if output_dir.exists():
            print(f'输出目录 {output_dir} 已存在，跳过处理。')
            continue

        jsonl_file = input_dir / f'final_dpi96_processed_ruler_all_tasks_{lens}.jsonl'
        if not jsonl_file.exists():
            print(f'输入文件 {jsonl_file} 不存在，跳过处理长度 {lens}。')
            continue

        run_single_file(jsonl_file, output_dir, engine, args)
        print(f'--- 长度 {lens} 处理完成 ---\n')
