import os
import json
import argparse
import numpy as np


from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def dataset_name_from_filename(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    if stem in dataset2metric:
        return stem
    for dataset in sorted(dataset2metric, key=len, reverse=True):
        if stem.startswith(f'{dataset}_'):
            return dataset
    return stem


import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='results', help="Model name (used as subfolder name)")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    # 新增输入输出路径参数
    parser.add_argument('--input_dir', type=str, default='/root/autodl-tmp/Data/longbench/pred', help="Input directory containing prediction files")
    parser.add_argument('--output_dir', type=str, default=None, help="Output directory for results (defaults to same as input_dir)")
    parser.add_argument('--result_filename', type=str, default='result.json', help="Output result filename")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

def parse_usage_token_count(value):
    if isinstance(value, dict):
        for key in ('prompt_tokens', 'input_tokens', 'total_tokens', 'completion_tokens'):
            token_value = value.get(key)
            try:
                return int(token_value)
            except (TypeError, ValueError):
                continue
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def calculate_token_stats(input_path, all_files):
    """统计各类token数"""
    # 定义每个数据集属于的类别
    categories = {
        "hotpotqa": "Multi-doc QA",
        "2wikimqa": "Multi-doc QA",
        "musique": "Multi-doc QA",
        "dureader": "Multi-doc QA",
        "multifieldqa_en": "Single-doc QA",
        "multifieldqa_zh": "Single-doc QA",
        "narrativeqa": "Single-doc QA",
        "qasper": "Single-doc QA",
        "gov_report": "Summarization",
        "qmsum": "Summarization",
        "multi_news": "Summarization",
        "vcsum": "Summarization",
        "triviaqa": "Few shot",
        "samsum": "Few shot",
        "trec": "Few shot",
        "lsht": "Few shot",
        "passage_retrieval_en": "Synthetic",
        "passage_count": "Synthetic",
        "passage_retrieval_zh": "Synthetic",
        "lcc": "Code",
        "repobench-p": "Code",
    }
    
    prompt_tokens_per_ds = {}
    prompt_tokens_per_category = {}
    total_prompt_tokens = 0
    total_records = 0
    records_with_usage = 0
    records_missing_usage = 0
    missing_usage_per_dataset = {}
    
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        
        dataset = dataset_name_from_filename(filename)
        input_file = os.path.join(input_path, filename)
        
        # 获取数据集所属类别
        category = categories.get(dataset, "Unknown")
        
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                token_count = parse_usage_token_count(data.get('usage'))
                total_records += 1
                if token_count is None:
                    records_missing_usage += 1
                    missing_usage_per_dataset[dataset] = missing_usage_per_dataset.get(dataset, 0) + 1
                    continue
                print(f"Dataset: {dataset}, Category: {category}, Tokens: {token_count}")
                prompt_tokens_per_ds.setdefault(dataset, []).append(token_count)
                prompt_tokens_per_category.setdefault(category, []).append(token_count)
                total_prompt_tokens += token_count
                records_with_usage += 1
    
    # 计算平均token数
    avg_prompt_tokens_per_ds = {ds: round(sum(tokens) / len(tokens), 2) if len(tokens) > 0 else 0
                                for ds, tokens in prompt_tokens_per_ds.items()}
    
    # 计算各类别的平均token数
    avg_prompt_tokens_per_category = {cat: round(sum(tokens) / len(tokens), 2) if len(tokens) > 0 else 0
                                     for cat, tokens in prompt_tokens_per_category.items()}
    
    overall_avg_tokens = round(total_prompt_tokens / records_with_usage, 2) if records_with_usage > 0 else 0
    
    return {
        'overall_avg_prompt_tokens': overall_avg_tokens,
        'avg_prompt_tokens_per_dataset': avg_prompt_tokens_per_ds,
        'avg_prompt_tokens_per_category': avg_prompt_tokens_per_category,
        'total_prompt_tokens': total_prompt_tokens,
        'total_records': total_records,
        'records_with_usage': records_with_usage,
        'records_missing_usage': records_missing_usage,
        'missing_usage_per_dataset': missing_usage_per_dataset
    }

def calculate_category_scores(scores):
    """按类别计算评估指标的平均值"""
    # 定义每个数据集属于的类别
    categories = {
        "hotpotqa": "Multi-doc QA",
        "2wikimqa": "Multi-doc QA",
        "musique": "Multi-doc QA",
        "dureader": "Multi-doc QA",
        "multifieldqa_en": "Single-doc QA",
        "multifieldqa_zh": "Single-doc QA",
        "narrativeqa": "Single-doc QA",
        "qasper": "Single-doc QA",
        "gov_report": "Summarization",
        "qmsum": "Summarization",
        "multi_news": "Summarization",
        "vcsum": "Summarization",
        "triviaqa": "Few shot",
        "samsum": "Few shot",
        "trec": "Few shot",
        "lsht": "Few shot",
        "passage_retrieval_en": "Synthetic",
        "passage_count": "Synthetic",
        "passage_retrieval_zh": "Synthetic",
        "lcc": "Code",
        "repobench-p": "Code",
    }
    
    scores_per_category = {}
    
    # 按类别收集分数
    for dataset, score in scores.items():
        category = categories.get(dataset, "Unknown")
        if category not in scores_per_category:
            scores_per_category[category] = []
        scores_per_category[category].append(score)
    
    # 计算每个类别的平均分数
    avg_scores_per_category = {}
    for category, score_list in scores_per_category.items():
        avg_scores_per_category[category] = round(sum(score_list) / len(score_list), 2)
    
    return avg_scores_per_category

def calculate_category_scores_e(scores):
    """按类别计算LongBench-E评估指标的平均值"""
    # 定义每个数据集属于的类别
    categories = {
        "hotpotqa": "Multi-doc QA",
        "2wikimqa": "Multi-doc QA",
        "musique": "Multi-doc QA",
        "dureader": "Multi-doc QA",
        "multifieldqa_en": "Single-doc QA",
        "multifieldqa_zh": "Single-doc QA",
        "narrativeqa": "Single-doc QA",
        "qasper": "Single-doc QA",
        "gov_report": "Summarization",
        "qmsum": "Summarization",
        "multi_news": "Summarization",
        "vcsum": "Summarization",
        "triviaqa": "Few shot",
        "samsum": "Few shot",
        "trec": "Few shot",
        "lsht": "Few shot",
        "passage_retrieval_en": "Synthetic",
        "passage_count": "Synthetic",
        "passage_retrieval_zh": "Synthetic",
        "lcc": "Code",
        "repobench-p": "Code",
    }
    
    scores_per_category = {"0-4k": {}, "4-8k": {}, "8k+": {}}
    
    # 按类别和长度范围收集分数
    for dataset, score_dict in scores.items():
        category = categories.get(dataset, "Unknown")
        for length_range in ["0-4k", "4-8k", "8k+"]:
            if category not in scores_per_category[length_range]:
                scores_per_category[length_range][category] = []
            if length_range in score_dict:
                scores_per_category[length_range][category].append(score_dict[length_range])
    
    # 计算每个类别在不同长度范围的平均分数
    avg_scores_per_category = {"0-4k": {}, "4-8k": {}, "8k+": {}}
    for length_range in ["0-4k", "4-8k", "8k+"]:
        for category, score_list in scores_per_category[length_range].items():
            if score_list:
                avg_scores_per_category[length_range][category] = round(sum(score_list) / len(score_list), 2)
    
    return avg_scores_per_category

if __name__ == '__main__':
    args = parse_args()
    
    # 检查必要参数
    if args.model is None:
        print("错误: 必须指定 --model 参数")
        exit(1)
    
    scores = dict()
    
    # 构建输入路径
    if args.e:
        input_path = os.path.join(args.input_dir + "_e", args.model)
    else:
        input_path = os.path.join(args.input_dir, args.model)
    
    # 构建输出路径
    if args.output_dir is None:
        output_path = input_path  # 默认输出到输入目录
    else:
        if args.e:
            output_path = os.path.join(args.output_dir + "_e", args.model)
        else:
            output_path = os.path.join(args.output_dir, args.model)
    
    # 检查输入路径是否存在
    if not os.path.exists(input_path):
        print(f"错误: 输入路径不存在: {input_path}")
        exit(1)
    
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"创建输出目录: {output_path}")
    
    all_files = os.listdir(input_path)
    print(f"输入路径: {input_path}")
    print(f"输出路径: {output_path}")
    print("Evaluating on:", all_files)
    
    # 统计token数
    print("\n正在统计Token使用情况...")
    token_stats = calculate_token_stats(input_path, all_files)
    
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = dataset_name_from_filename(filename)
        
        input_file = os.path.join(input_path, filename)
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
        
        if args.e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score
        print(f"Dataset {dataset}: {score}")
    
    # 计算各类别的评估指标平均值
    if args.e:
        category_scores = calculate_category_scores_e(scores)
    else:
        category_scores = calculate_category_scores(scores)
    
    # 输出结果文件，包含类别统计
    results_with_categories = {
        'dataset_scores': scores,
        'category_scores': category_scores
    }
    
    out_path = os.path.join(output_path, args.result_filename)
    with open(out_path, "w") as f:
        json.dump(results_with_categories, f, ensure_ascii=False, indent=4)
    
    print(f"\n评估完成，结果保存到: {out_path}")
    
    # 保存Token统计结果
    token_stat_path = os.path.join(output_path, 'token_stat.json')
    with open(token_stat_path, 'w', encoding='utf-8') as f:
        json.dump(token_stats, f, ensure_ascii=False, indent=4)
    print(f"Token统计结果已保存到: {token_stat_path}")
    
    # 打印总体结果摘要
    if args.e:
        print("\n=== LongBench-E 评估结果 ===")
        for dataset, score_dict in scores.items():
            print(f"{dataset}: {score_dict}")
        
        print("\n=== 各类别评估结果 (LongBench-E) ===")
        for length_range in ["0-4k", "4-8k", "8k+"]:
            print(f"\n{length_range}:")
            for category, score in category_scores[length_range].items():
                print(f"  {category}: {score}")
    else:
        print("\n=== LongBench 评估结果 ===")
        for dataset, score in scores.items():
            print(f"{dataset}: {score}")
        
        print("\n=== 各类别评估结果 ===")
        for category, score in category_scores.items():
            print(f"{category}: {score}")
    
    # 打印Token统计摘要
    print(f"\n=== Token使用统计 ===")
    print(f"总记录数: {token_stats['total_records']}")
    print(f"有Token统计的记录数: {token_stats['records_with_usage']}")
    print(f"缺失Token统计的记录数: {token_stats['records_missing_usage']}")
    print(f"总Token数: {token_stats['total_prompt_tokens']}")
    print(f"平均Token数: {token_stats['overall_avg_prompt_tokens']}")
    if token_stats['missing_usage_per_dataset']:
        print(f"缺失Token统计的数据集: {token_stats['missing_usage_per_dataset']}")

    print("\n各类别平均Token数:")
    for category, avg_tokens in token_stats['avg_prompt_tokens_per_category'].items():
        print(f"  {category}: {avg_tokens}")
    
    print("\n各数据集平均Token数:")
    for dataset, avg_tokens in token_stats['avg_prompt_tokens_per_dataset'].items():
        print(f"  {dataset}: {avg_tokens}")
