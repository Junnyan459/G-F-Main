#!/usr/bin/env python3
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import argparse
import io
import os
import json
import numpy as np
import gc
import re
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from xml.sax.saxutils import escape
import shutil
# ReportLab imports
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib import colors
import pdfplumber
# pdf2image import
from pdf2image import convert_from_bytes
import sys
import multiprocessing as mp
import random   

MULTI_DOC_HEADER_RE = re.compile(r'^(Passage|Document)\s+\d+\s*:?$', re.IGNORECASE)


def enhance_multidoc_readability(text):
    lines = text.split('\n')
    enhanced = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if MULTI_DOC_HEADER_RE.match(stripped):
            if enhanced and enhanced[-1] != '':
                enhanced.append('')
            enhanced.append(stripped.upper())
            enhanced.append('-' * min(24, max(12, len(stripped))))
            continue
        enhanced.append(line)
    return '\n'.join(enhanced)

SCRIPT_DIR = Path(__file__).resolve().parent

# 默认全局变量，将由 parse_args 设置，可被每个 item 的 config 覆盖
PAGE_SIZE = A4
MARGIN_X = 20
MARGIN_Y = 20
FONT_PATH = None
FONT_NAME = None
FONT_SIZE = 9
LINE_HEIGHT = None
PAGE_BG_COLOR = None
FONT_COLOR = None
PARA_BG_COLOR = None
PARA_BORDER_COLOR = None
FIRST_LINE_INDENT = 0
LEFT_INDENT = 0
RIGHT_INDENT = 0
ALIGNMENT = TA_JUSTIFY
SPACE_BEFORE = 0
SPACE_AFTER = 0
BORDER_WIDTH = 0
BORDER_PADDING = 0
HORIZONTAL_SCALE = 1.0
DPI = 72
AUTO_CROP_LAST_PAGE = True
AUTO_CROP_WIDTH = True
PROCESSES = 1
OUTPUT_DIR = None
JSON_PATH = None

NEWLINE_MARKUP = None  # 添加这一行

# 对齐映射
ALIGN_MAP = {
    "LEFT": TA_LEFT,
    "CENTER": TA_CENTER,
    "RIGHT": TA_RIGHT,
    "JUSTIFY": TA_JUSTIFY,
}

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate styled PDFs in parallel and export pages to PNGs via bash."
    )
    # 页面尺寸
    p.add_argument(
        "--page-size",
        type=str,
        default=None,
        help="页面尺寸，格式 width,height(pt)，例如 595,842；不传默认为 A4"
    )
    p.add_argument("--margin-x", type=float, default=20, help="左右边距 (pt)")
    p.add_argument("--margin-y", type=float, default=20, help="上下边距 (pt)")

    # 字体
    p.add_argument("--font-path", type=str, default=None, help="TTF 字体文件路径")
    p.add_argument("--font-size", type=float, default=9, help="字体大小 (pt)")
    p.add_argument(
        "--line-height",
        type=float,
        help="行高 (pt)，不传则为 font-size + 1"
    )

    # 颜色
    p.add_argument(
        "--page-bg-color",
        type=str,
        default="#FFFFFF",
        help="页面背景色 (#rrggbb)"
    )
    p.add_argument(
        "--font-color",
        type=str,
        default="#000000",
        help="字体颜色 (#rrggbb)"
    )
    p.add_argument(
        "--para-bg-color",
        type=str,
        default="#FFFFFF",
        help="段落背景色 (#rrggbb)"
    )
    p.add_argument(
        "--para-border-color",
        type=str,
        default="#FFFFFF",
        help="段落边框色 (#rrggbb)"
    )

    # 排版
    p.add_argument(
        "--first-line-indent",
        type=float,
        default=0,
        help="首行缩进 (pt)"
    )
    p.add_argument(
        "--left-indent",
        type=float,
        default=0,
        help="段落左缩进 (pt)"
    )
    p.add_argument(
        "--right-indent",
        type=float,
        default=0,
        help="段落右缩进 (pt)"
    )
    p.add_argument(
        "--alignment",
        choices=["LEFT","CENTER","RIGHT","JUSTIFY"],
        default="JUSTIFY",
        help="对齐方式"
    )
    p.add_argument("--space-before", type=float, default=0, help="段前距离 (pt)")
    p.add_argument("--space-after", type=float, default=0, help="段后距离 (pt)")
    p.add_argument(
        "--border-width",
        type=float,
        default=0,
        help="段落边框宽度 (pt)"
    )
    p.add_argument(
        "--border-padding",
        type=float,
        default=0,
        help="段落边框内边距 (pt)"
    )

    # 渲染 & PNG 输出
    p.add_argument(
        "--horizontal-scale",
        type=float,
        default=0.95,
        help="水平缩放比例"
    )
    p.add_argument("--dpi", type=int, default=72, help="PNG 分辨率 (dpi)")
    p.add_argument(
        "--auto-crop-last-page",
        action="store_true",
        help="对最后一页启用自适应裁剪"
    )
    p.add_argument(
        "--auto-crop-width",
        action="store_true",
        help="对宽启用自适应裁剪"
    )

    # 添加 newline-markup 参数
    p.add_argument(
        "--newline-markup",
        type=str,
        default=None,
        help="换行符的标记替换，例如 '<br/>' 或其他HTML标记"
    )

    return p.parse_args()


def process_one(item):
    if recover:
        _id = item.get('unique_id', 'UNKNOWN')
        existing_dir = os.path.join(OUTPUT_DIR, _id)
        if os.path.exists(existing_dir):
            print(f"Find existing dir for {_id}, skipping...")
            image_paths = []
            for name in sorted(os.listdir(existing_dir)):
                if not name.lower().endswith('.png'):
                    continue
                if not name.startswith('page_'):
                    continue
                image_paths.append(os.path.abspath(os.path.join(existing_dir, name)))
            item['image_paths'] = image_paths
            return item
    try:
        # 每个 item 可自带 config 覆盖全局渲染参数
        config = item.get('config', {}) or {}

        # 解析配置或使用全局默认
        page_size = (tuple(map(float, config['page-size'].split(',')))
                    if 'page-size' in config else PAGE_SIZE)
        margin_x = config.get('margin-x', MARGIN_X)
        margin_y = config.get('margin-y', MARGIN_Y)
        
        # --- 修改：调整字体文件的相对路径 ---
        font_path = config.get('font-path', FONT_PATH)
        if font_path and not os.path.isabs(font_path):
            # 如果是相对路径，我们假设它是相对于 config 目录的
            # 将其转换为相对于当前脚本目录的正确路径
            font_path = str((SCRIPT_DIR.parents[2] / 'config' / os.path.basename(font_path)).resolve())

        font_name = os.path.basename(font_path).split('.')[0]
        font_size = config.get('font-size', FONT_SIZE)
        line_height = config.get('line-height', None) or (font_size + 1)

        page_bg_color = (colors.HexColor(config['page-bg-color'])
                        if 'page-bg-color' in config else PAGE_BG_COLOR)
        font_color = (colors.HexColor(config['font-color'])
                    if 'font-color' in config else FONT_COLOR)
        para_bg_color = (colors.HexColor(config['para-bg-color'])
                        if 'para-bg-color' in config else PARA_BG_COLOR)
        para_border_color = (colors.HexColor(config['para-border-color'])
                            if 'para-border-color' in config else PARA_BORDER_COLOR)

        first_line_indent = config.get('first-line-indent', FIRST_LINE_INDENT)
        left_indent = config.get('left-indent', LEFT_INDENT)
        right_indent = config.get('right-indent', RIGHT_INDENT)
        alignment = ALIGN_MAP.get(config.get('alignment'), ALIGNMENT)
        space_before = config.get('space-before', SPACE_BEFORE)
        space_after = config.get('space-after', SPACE_AFTER)
        border_width = config.get('border-width', BORDER_WIDTH)
        border_padding = config.get('border-padding', BORDER_PADDING)

        horizontal_scale = config.get('horizontal-scale', HORIZONTAL_SCALE)
        dpi = config.get('dpi', DPI)
        auto_crop_last_page = config.get('auto-crop-last-page', AUTO_CROP_LAST_PAGE)
        auto_crop_width = config.get('auto-crop-width', AUTO_CROP_WIDTH)
        
        # 修改这部分逻辑
        newline_markup = config.get('newline-markup', NEWLINE_MARKUP)
        
        # # 如果您想保留随机逻辑，可以这样写：
        # if newline_markup is None:
        #     newline_markup = '<font color="#FF0000"> \\n </font>' if random.random() < 0.4 else None


        
        
        _id = item.get('unique_id', None) # 使用 unique_id 作为标识符
        assert _id
        assert font_path
        # 注册字体
        pdfmetrics.registerFont(TTFont(font_name, font_path))

        # --- 修改：读取 context 字段的文本 ---
        text = item.get('context', '')
        assert text

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=page_size,
            leftMargin=margin_x,
            rightMargin=margin_x,
            topMargin=margin_y,
            bottomMargin=margin_y,
        )

        # 本地绘制背景，使用 config 或默认值
        def draw_bg_local(canvas, d):
            canvas.saveState()
            canvas.setFillColor(page_bg_color)
            canvas.rect(0, 0, page_size[0], page_size[1], stroke=0, fill=1)
            canvas.restoreState()

        styles = getSampleStyleSheet()
        
        RE_CJK = re.compile(r'[\u4E00-\u9FFF]')
        if RE_CJK.search(text):
            custom = ParagraphStyle(
                name="Custom",
                parent=styles["Normal"],
                fontName=font_name,
                fontSize=font_size,
                leading=line_height,
                textColor=font_color,
                backColor=para_bg_color,
                borderColor=para_border_color,
                borderWidth=border_width,
                borderPadding=border_padding,
                firstLineIndent=first_line_indent,
                wordWrap="CJK",
                leftIndent=left_indent,
                rightIndent=right_indent,
                alignment=alignment,
                spaceBefore=space_before,
                spaceAfter=space_after,
            )
        else:
            # print(TA_LEFT)
            # print(alignment)
            custom = ParagraphStyle(
                name="Custom",
                parent=styles["Normal"],
                fontName=font_name,
                fontSize=font_size,
                leading=line_height,
                textColor=font_color,
                backColor=para_bg_color,
                borderColor=para_border_color,
                borderWidth=border_width,
                borderPadding=border_padding,
                firstLineIndent=first_line_indent,
                leftIndent=left_indent,
                rightIndent=right_indent,
                alignment=alignment,
                spaceBefore=space_before,
                spaceAfter=space_after,
            )
        story = []

        def replace_spaces(s):
            return re.sub(r' {2,}', lambda m: '&nbsp;'*len(m.group()), s)
        
        text = text.replace('\xad', '').replace('\u200b', '')
        if newline_markup:
            escaped = replace_spaces(escape(text))
            marked = escaped.replace('\t', '&nbsp;'*4).replace(
                '\n', newline_markup
            )
            story.append(Paragraph(marked, custom))
        else:
            processed_text = replace_spaces(escape(text)).replace('\n', '<br/>').replace('\t', '&nbsp;'*4)
            story.append(Paragraph(processed_text, custom))

        doc.build(
            story,
            onFirstPage=lambda c, d: draw_bg_local(c, d),
            onLaterPages=lambda c, d: draw_bg_local(c, d)
        )
        pdf_bytes = buf.getvalue()
        buf.close()

        # 提取页文本
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page_texts = [page.extract_text() for page in pdf.pages]
            num_pages = len(pdf.pages)
        
        out_root = os.path.join(OUTPUT_DIR, _id)
        os.makedirs(out_root, exist_ok=True)
        
        with open(os.path.join(out_root, 'page_texts.json'), 'w', encoding='utf-8') as f:
            json.dump(page_texts, f, indent=4, ensure_ascii=False)
        
        MAX_DPI = 300
        
        # --- 新增：用于存储图片路径的列表 ---
        image_paths = []

        for i in range(1, num_pages + 1):
            # 每次调用只转换一页
            images = convert_from_bytes(
                pdf_bytes,
                dpi=dpi,
                first_page=i,
                last_page=i
            )
            # DPI 过高降级重试逻辑
            if any(img.size == (1, 1) for img in images):
                print(f"Warning: requested dpi={dpi} too high for page {i}, retrying with dpi={MAX_DPI}", file=sys.stderr)
                images = convert_from_bytes(
                    pdf_bytes,
                    dpi=MAX_DPI,
                    first_page=i,
                    last_page=i
                )
            
            img = images[0] # images 列表里永远只有一张图片

            w, h = img.size
            if w == 1 or h == 1:
                assert False

            img = img.resize((int(w * horizontal_scale), h))
            tolerance = 5
            gray = np.array(img.convert("L"))
            bg_gray = np.median(gray[:2, :2])
            if auto_crop_width:
                mask = np.abs(gray - bg_gray) > tolerance
                cols = np.where(mask.any(axis=0))[0]
                if cols.size:
                    rightmost_col = cols[-1] + 1
                    right = min(img.width, rightmost_col + margin_x)
                    img = img.crop((0, 0, right, img.height))
            if auto_crop_last_page and i == num_pages:
                mask = np.abs(gray - bg_gray) > tolerance
                rows = np.where(mask.any(axis=1))[0]
                if rows.size:
                    last_row = rows[-1]
                    lower = min(img.height, last_row + margin_y)
                    img = img.crop((0, 0, img.width, lower))
            
            out_path = os.path.join(out_root, f"page_{i:03d}.png")
            img.save(out_path, 'PNG')
            image_paths.append(os.path.abspath(out_path)) # --- 新增：记录绝对路径 ---

            img.close()
            images.clear()
            del img
            del images
            
        del pdf_bytes
        del buf
        gc.collect()
        
        # --- 修改：返回带有图片路径的完整 item ---
        item['image_paths'] = image_paths
        return item
    except Exception as e:
        _id = item.get('unique_id', 'UNKNOWN')
        print(f"[ERROR] process_one_id={_id}, error={e}", file=sys.stderr)
        return None

def main():
    global PAGE_SIZE, MARGIN_X, MARGIN_Y, FONT_PATH, FONT_NAME, FONT_SIZE, LINE_HEIGHT
    global PAGE_BG_COLOR, FONT_COLOR, PARA_BG_COLOR, PARA_BORDER_COLOR
    global FIRST_LINE_INDENT, LEFT_INDENT, RIGHT_INDENT, ALIGNMENT
    global SPACE_BEFORE, SPACE_AFTER, BORDER_WIDTH, BORDER_PADDING
    global HORIZONTAL_SCALE, DPI, AUTO_CROP_LAST_PAGE, AUTO_CROP_WIDTH, PROCESSES, OUTPUT_DIR
    global NEWLINE_MARKUP

    args = parse_args()
    # 全局默认值
    if args.page_size:
        w, h = map(float, args.page_size.split(","))
        PAGE_SIZE = (w, h)
    MARGIN_X, MARGIN_Y = args.margin_x, args.margin_y
    FONT_PATH = args.font_path
    FONT_NAME = 'my_font'
    FONT_SIZE = args.font_size
    LINE_HEIGHT = args.line_height or (FONT_SIZE + 1)
    PAGE_BG_COLOR = colors.HexColor(args.page_bg_color)
    FONT_COLOR = colors.HexColor(args.font_color)
    PARA_BG_COLOR = colors.HexColor(args.para_bg_color)
    PARA_BORDER_COLOR = colors.HexColor(args.para_border_color)
    FIRST_LINE_INDENT = args.first_line_indent
    LEFT_INDENT = args.left_indent
    RIGHT_INDENT = args.right_indent
    ALIGNMENT = ALIGN_MAP[args.alignment]
    SPACE_BEFORE = args.space_before
    SPACE_AFTER = args.space_after
    BORDER_WIDTH = args.border_width
    BORDER_PADDING = args.border_padding
    HORIZONTAL_SCALE = args.horizontal_scale
    DPI = args.dpi
    AUTO_CROP_LAST_PAGE = args.auto_crop_last_page
    AUTO_CROP_WIDTH = args.auto_crop_width
    NEWLINE_MARKUP = args.newline_markup  # 添加这一行
    PROCESSES = 32

    # --- 修改：输入和输出文件夹路径 ---
    INPUT_DIR_PATH = '/root/autodl-tmp/Data/longbench/data/uid_added'
    FINAL_OUTPUT_DIR_PATH = '/root/autodl-tmp/Data/longbench/rendered_images'
    OUTPUT_DIR = os.path.join(FINAL_OUTPUT_DIR_PATH, 'pics') # 图片统一输出到此

    def ensure_empty_dir(dir_path):
        if os.path.isdir(dir_path) and not recover:
            print(f"Cleaning directory: {dir_path}")
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)

    ensure_empty_dir(FINAL_OUTPUT_DIR_PATH)
    ensure_empty_dir(OUTPUT_DIR)

    # 遍历输入文件夹中的所有 .jsonl 文件
    for filename in os.listdir(INPUT_DIR_PATH):
        if not filename.endswith(".jsonl"):
            continue

        input_jsonl_path = os.path.join(INPUT_DIR_PATH, filename)
        output_jsonl_path = os.path.join(FINAL_OUTPUT_DIR_PATH, filename)
        
        print(f"\nProcessing file: {input_jsonl_path}")

        # --- 读取 .jsonl 文件 ---
        data_to_process = []
        with open(input_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data_to_process.append(json.loads(line))
        
        if not data_to_process:
            print(f"No data found in {filename}, skipping.")
            continue

        # 获取已处理的unique_id集合，避免重复处理
        processed_ids = set()
        if recover and os.path.exists(output_jsonl_path):
            with open(output_jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON in {output_jsonl_path} on line {line_num}: {e}")
                        continue
                    image_paths = item.get('image_paths')
                    if isinstance(image_paths, str):
                        image_paths = [image_paths]
                    elif not isinstance(image_paths, list):
                        image_paths = []
                    valid_paths = [path for path in image_paths if path and os.path.exists(str(path).strip())]
                    if valid_paths:
                        processed_ids.add(item.get('unique_id'))
            print(f"Found {len(processed_ids)} already processed items with images in {output_jsonl_path}")
        
        # 过滤掉已处理的项目
        filtered_data = [item for item in data_to_process if item.get('unique_id') not in processed_ids]
        print(f"Total items: {len(data_to_process)}, Remaining items to process: {len(filtered_data)}")
        
        if not filtered_data:
            print(f"All items in {filename} have been processed. Skipping.")
            continue
        
        batch_size = 100
        batch_buffer = []
        
        with Pool(processes=PROCESSES) as pool:
            # 使用 imap_unordered 来获取处理结果
            for i, result_item in enumerate(tqdm(pool.imap_unordered(process_one, filtered_data, chunksize=1), total=len(filtered_data), desc=f"Rendering {filename}")):
                if result_item:
                    batch_buffer.append(result_item)
                    _id = result_item.get('unique_id', 'UNKNOWN')
                    count = len(result_item.get('image_paths', []))
                    tqdm.write(f"{_id}: generated {count} pages")
                    
                    # 每100条写入一次文件
                    if len(batch_buffer) >= batch_size:
                        print(f"\nWriting batch of {len(batch_buffer)} items to {output_jsonl_path}...")
                        with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                            for item in batch_buffer:
                                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        batch_buffer = []  # 清空缓冲区

        # 写入剩余的项目
        if batch_buffer:
            print(f"\nWriting final batch of {len(batch_buffer)} items to {output_jsonl_path}...")
            with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                for item in batch_buffer:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Finished processing {filename}.")

    print("\nAll files processed.")


if __name__ == '__main__':
    recover = True
    main()
