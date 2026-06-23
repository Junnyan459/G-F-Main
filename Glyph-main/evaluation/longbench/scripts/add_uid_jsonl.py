import os
import json
from pathlib import Path
from langdetect import detect, LangDetectException
import random
import argparse

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path('/root/autodl-tmp/Data/longbench')
REPO_ROOT = SCRIPT_DIR.parents[2]
CONFIG_DIR = REPO_ROOT / 'config'

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="为 LongBench 数据集添加 unique_id 和渲染配置。")
    parser.add_argument('--src-folder', type=str, default=str(DATA_ROOT / 'data'),
                        help="包含原始 .jsonl 数据集的源文件夹路径。")
    parser.add_argument('--dst-folder', type=str, default=str(DATA_ROOT / 'data' / 'uid_added'),
                        help="用于存储处理后文件的目标文件夹路径。")
    parser.add_argument('--chose-newline', action='store_true',
                        help="为指定的数据集列表启用特殊的换行符标记。")
    parser.add_argument('--all-newline', action='store_true',
                        help="为所有数据集启用特殊的换行符标记。")
    return parser.parse_args()

def main():
    """主执行函数"""
    args = parse_args()

    # --- 预加载配置文件 ---
    try:
        with (CONFIG_DIR / 'config_zh.json').open('r', encoding='utf-8') as f:
            config_zh = json.load(f)
        with (CONFIG_DIR / 'config_en.json').open('r', encoding='utf-8') as f:
            config_en = json.load(f)
    except FileNotFoundError as e:
        print(f"错误：找不到配置文件: {e}. 请确保路径 '{CONFIG_DIR}' 正确。")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"错误：配置文件JSON格式错误: {e}.")
        exit(1)

    os.makedirs(args.dst_folder, exist_ok=True)

    # 为 --chose-newline 定义数据集列表
    chose_newline_datasets = {
        'passage_retrieval_en', 'hotpotqa', 'narrativeqa', 'qmsum',
        'gov_report', 'vcsum', 'triviaqa', 'multi_news', 'dureader',
        'qasper', 'samsum', 'trec'
    }
    newline_value = '<font color="#FF0000"> \\n </font>'

    for fname in os.listdir(args.src_folder):
        if fname.endswith(".jsonl"):
            input_path = os.path.join(args.src_folder, fname)
            output_path = os.path.join(args.dst_folder, fname)
            base_name = fname.replace(".jsonl", "")
            
            with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
                for idx, line in enumerate(fin, 1):
                    try:
                        data = json.loads(line)
                        data["unique_id"] = f"{base_name}_{idx}"
                        
                        # --- 语言检测 ---
                        lang = 'en'
                        text_to_check = data.get("context", "")
                        if text_to_check:
                            try:
                                detected_lang = detect(text_to_check)
                                if detected_lang.startswith('zh'):
                                    lang = 'zh'
                            except LangDetectException:
                                lang = 'en'
                        
                        # --- 根据语言加载配置 ---
                        if lang == 'zh':
                            config = config_zh.copy()
                        else:
                            config = config_en.copy()

                        # --- 新增：根据命令行参数设置 newline-markup ---
                        if args.all_newline:
                            # --all-newline 优先级最高，对所有文件生效
                            config['newline-markup'] = newline_value
                        elif args.chose_newline:
                            # --chose-newline 只对列表中的数据集生效
                            if base_name in chose_newline_datasets:
                                config['newline-markup'] = newline_value
                            else:
                                config['newline-markup'] = None
                        else:
                            # 默认行为，不修改 newline-markup
                            config['newline-markup'] = None
                        
                        data["config"] = config
                        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                    except json.JSONDecodeError as e:
                        print(f"警告：跳过文件 {fname} 中的无效JSON行 {idx}: {e}")
                        continue
    print("处理完成。")

if __name__ == "__main__":
    main()