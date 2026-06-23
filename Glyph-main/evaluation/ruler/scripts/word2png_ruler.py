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
AUTO_CROP_LAST_PAGE = False
AUTO_CROP_WIDTH = False
PROCESSES = 1
OUTPUT_DIR = None
JSON_PATH = None
FINAL_JSONL_OUTPUT_PATH = None  # 新增：全局存储当前lens的JSONL输出路径

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

    return p.parse_args()


def process_one(item):
    # 新增：恢复模式下跳过已处理item（基于当前lens的输出目录）
    _id = item.get('unique_id', 'UNKNOWN')
    if recover:
        item_out_dir = os.path.join(OUTPUT_DIR, _id)
        if os.path.exists(item_out_dir):
            # 检查是否已有完整的图片（避免半截任务被误判）
            has_images = any(f.startswith("page_") and f.endswith(".png") for f in os.listdir(item_out_dir)) if os.path.isdir(item_out_dir) else False
            if has_images:
                print(f"[Lens {lens_current}] Skip existing item: {_id}")
                item['image_paths'] = [os.path.join(item_out_dir, f) for f in os.listdir(item_out_dir) if f.startswith("page_") and f.endswith(".png")]
                return item

    try:
        # 每个 item 可自带 config 覆盖全局渲染参数
        config = item.get('config', {}) or {}

        # 解析配置或使用全局默认
        page_size = (tuple(map(float, config['page-size'].split(',')))
                    if 'page-size' in config else PAGE_SIZE)
        margin_x = config.get('margin-x', MARGIN_X)
        margin_y = config.get('margin-y', MARGIN_Y)
        font_path = config.get('font-path', FONT_PATH)
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
        newline_markup = config.get('newline-markup', None)
        
        
        assert _id, f"Item missing 'unique_id': {item}"
        assert font_path, f"Item {_id} missing 'font-path' (global or config)"

        pdfmetrics.registerFont(TTFont(font_name, font_path))

        # 读取 context 字段的文本
        text = item.get('context', '')
        assert text, f"Item {_id} missing 'context' text"

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=page_size,
            leftMargin=margin_x,
            rightMargin=margin_x,
            topMargin=margin_y,
            bottomMargin=margin_y,
        )

        # 绘制页面背景
        def draw_bg_local(canvas, d):
            canvas.saveState()
            canvas.setFillColor(page_bg_color)
            canvas.rect(0, 0, page_size[0], page_size[1], stroke=0, fill=1)
            canvas.restoreState()

        styles = getSampleStyleSheet()
        RE_CJK = re.compile(r'[\u4E00-\u9FFF]')
        
        # 区分CJK文本和非CJK文本的样式
        if RE_CJK.search(text):
            custom_style = ParagraphStyle(
                name="CustomCJK",
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
            custom_style = ParagraphStyle(
                name="CustomNonCJK",
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
        # 处理空格和转义
        def replace_spaces(s):
            return re.sub(r' {2,}', lambda m: '&nbsp;'*len(m.group()), s)
        
        text_clean = text.replace('\xad', '').replace('\u200b', '')  # 清理软连字符和零宽空格
        if newline_markup:
            text_escaped = replace_spaces(escape(text_clean))
            text_marked = text_escaped.replace('\t', '&nbsp;'*4).replace('\n', newline_markup)
            story.append(Paragraph(text_marked, custom_style))
        else:
            # 将整个文本作为一个段落处理，用 <br/> 替换 \n
            processed_text = replace_spaces(escape(text)).replace('\n', '<br/>').replace('\t', '&nbsp;'*4)
            story.append(Paragraph(processed_text, custom_style))
            # for para in text.split("\n"):
            #     if para.strip():
            #         story.append(Paragraph(
            #             replace_spaces(escape(para)).replace('\t', '&nbsp;'*4),
            #             custom
            #         ))
            #     else:
            #         story.append(Paragraph("", custom))
        
        # # 将整个文本作为一个段落处理，用 <br/> 替换 \n
        # processed_text = replace_spaces(escape(text)).replace('\n', '<br/>').replace('\t', '&nbsp;'*4)
        # story.append(Paragraph(processed_text, custom))

        # 生成PDF
        doc.build(
            story,
            onFirstPage=draw_bg_local,
            onLaterPages=draw_bg_local
        )
        pdf_bytes = buf.getvalue()
        buf.close()

        # 提取PDF文本（用于验证）
        page_texts = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page_texts = [page.extract_text() for page in pdf.pages]
            num_pages = len(pdf.pages)
        
        # 创建当前item的输出目录
        item_out_dir = os.path.join(OUTPUT_DIR, _id)
        os.makedirs(item_out_dir, exist_ok=True)
        
        # 保存页面文本
        with open(os.path.join(item_out_dir, 'page_texts.json'), 'w', encoding='utf-8') as f:
            json.dump(page_texts, f, indent=4, ensure_ascii=False)
        
        MAX_DPI = 300  # 最高重试DPI
        image_paths = []

        # 逐页转换为PNG
        for page_idx in range(1, num_pages + 1):
            # 转换单页PDF为图片
            try:
                images = convert_from_bytes(
                    pdf_bytes,
                    dpi=dpi,
                    first_page=page_idx,
                    last_page=page_idx,
                    fmt="png",
                    thread_count=1  # 单页转换避免线程冲突
                )
            except Exception as e:
                # DPI过高时重试
                print(f"[Lens {lens_current}] Page {page_idx} of {_id} DPI {dpi} failed: {e}, retry with DPI {MAX_DPI}")
                images = convert_from_bytes(
                    pdf_bytes,
                    dpi=MAX_DPI,
                    first_page=page_idx,
                    last_page=page_idx,
                    fmt="png",
                    thread_count=1
                )
            
            img = images[0]
            # 检查图片有效性
            if img.size in [(1,1), (0,0)]:
                raise ValueError(f"Invalid image size for {_id} page {page_idx}: {img.size}")

            # 水平缩放
            if horizontal_scale != 1.0:
                new_width = int(img.width * horizontal_scale)
                img = img.resize((new_width, img.height), Image.Resampling.LANCZOS)

            # 自适应裁剪逻辑
            if auto_crop_width or (auto_crop_last_page and page_idx == num_pages):
                gray_img = np.array(img.convert("L"))
                bg_gray = np.median(gray_img[:2, :2])  # 取左上角2x2像素的中位数作为背景色
                tolerance = 5  # 颜色容差
                mask = np.abs(gray_img - bg_gray) > tolerance  # 非背景区域掩码

                # 宽度裁剪（所有页面）
                if auto_crop_width and mask.any():
                    cols = np.where(mask.any(axis=0))[0]  # 有内容的列
                    if cols.size:
                        crop_right = min(img.width, cols[-1] + 1 + int(margin_x/2))  # 保留少量边距
                        img = img.crop((0, 0, crop_right, img.height))

                # 最后一页高度裁剪
                if auto_crop_last_page and page_idx == num_pages and mask.any():
                    rows = np.where(mask.any(axis=1))[0]  # 有内容的行
                    if rows.size:
                        crop_bottom = min(img.height, rows[-1] + 1 + int(margin_y/2))  # 保留少量边距
                        img = img.crop((0, 0, img.width, crop_bottom))

            # 保存PNG
            img_out_path = os.path.join(item_out_dir, f"page_{page_idx:03d}.png")
            img.save(img_out_path, 'PNG', optimize=True)
            image_paths.append(os.path.abspath(img_out_path))

            # 释放内存
            img.close()
            del img
            images.clear()
            del images
        
        # 释放PDF相关内存
        del pdf_bytes
        del buf
        gc.collect()
        
        # 记录图片路径并返回
        item['image_paths'] = image_paths
        print(f"[Lens {lens_current}] Success: {_id} (pages: {len(image_paths)})")
        return item

    except Exception as e:
        error_msg = f"[Lens {lens_current}] Failed to process {_id}: {str(e)}"
        print(error_msg, file=sys.stderr)
        # 可选：保存错误日志
        with open(os.path.join(OUTPUT_DIR, "error_log.txt"), 'a', encoding='utf-8') as f:
            f.write(f"{error_msg}\n")
        return None


def main():
    # 全局变量初始化（会被循环覆盖）
    global PAGE_SIZE, MARGIN_X, MARGIN_Y, FONT_PATH, FONT_NAME, FONT_SIZE, LINE_HEIGHT
    global PAGE_BG_COLOR, FONT_COLOR, PARA_BG_COLOR, PARA_BORDER_COLOR
    global FIRST_LINE_INDENT, LEFT_INDENT, RIGHT_INDENT, ALIGNMENT
    global SPACE_BEFORE, SPACE_AFTER, BORDER_WIDTH, BORDER_PADDING
    global HORIZONTAL_SCALE, DPI, AUTO_CROP_LAST_PAGE, AUTO_CROP_WIDTH, PROCESSES
    global OUTPUT_DIR, JSON_PATH, FINAL_JSONL_OUTPUT_PATH, lens_current

    # -------------------------- 核心改动：定义要循环的lens列表 --------------------------
    lens_list = [4096, 8192, 16384, 32768, 65536, 126000]  # 目标lens值
    # -----------------------------------------------------------------------------------

    args = parse_args()
    # 全局渲染参数初始化（由命令行参数决定，所有lens共用）
    if args.page_size:
        w, h = map(float, args.page_size.split(","))
        PAGE_SIZE = (w, h)
    MARGIN_X, MARGIN_Y = args.margin_x, args.margin_y
    FONT_PATH = args.font_path
    FONT_NAME = 'my_font'  # 默认字体名（会被item config覆盖）
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
    PROCESSES = 32  # 进程数（可根据CPU核心数调整）

    # 目录清理工具函数
    def ensure_empty_dir(dir_path):
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)

    # -------------------------- 核心改动：循环处理每个lens --------------------------
    for lens_current in lens_list:
        print(f"\n" + "="*50)
        print(f"Starting processing for lens = {lens_current}")
        print("="*50)

        # 1. 构造当前lens的输入/输出路径（与原脚本路径格式保持一致）
        # benchmark eval
        JSON_PATH = f'/root/autodl-tmp/Data/ruler/data/dpi96_processed_ruler_all_tasks_{lens_current}.json'
        FINAL_JSONL_OUTPUT_PATH = f'/root/autodl-tmp/Data/ruler/data/final_dpi96_processed_ruler_all_tasks_{lens_current}.jsonl'
        OUTPUT_DIR = f'/root/autodl-tmp/Data/ruler/rendered_images/{lens_current}'

        # 2. 检查输入JSON文件是否存在
        if not os.path.exists(JSON_PATH):
            print(f"Warning: Input JSON file not found for lens {lens_current} -> {JSON_PATH}")
            print(f"Skipping lens {lens_current}...\n")
            continue

        # 3. 初始化输出目录（恢复模式保留目录，非恢复模式清空目录）
        if not recover:
            ensure_empty_dir(OUTPUT_DIR)
            print(f"Cleared output dir for lens {lens_current}: {OUTPUT_DIR}")
        else:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            print(f"Using existing output dir for lens {lens_current}: {OUTPUT_DIR}")

        # 4. 读取当前lens的待处理数据
        try:
            with open(JSON_PATH, 'r', encoding='utf-8') as f:
                data_to_process = json.load(f)
            print(f"Loaded {len(data_to_process)} items from {JSON_PATH}")
        except json.JSONDecodeError as e:
            print(f"Error reading JSON file for lens {lens_current}: {e}")
            print(f"Skipping lens {lens_current}...\n")
            continue

        # 5. 过滤已处理的item（基于JSONL输出文件）
        processed_ids = set()
        if recover and os.path.exists(FINAL_JSONL_OUTPUT_PATH):
            try:
                with open(FINAL_JSONL_OUTPUT_PATH, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            uid = item.get('unique_id')
                            if uid:
                                processed_ids.add(uid)
                        except json.JSONDecodeError:
                            print(f"Warning: Invalid JSON on line {line_num} of {FINAL_JSONL_OUTPUT_PATH}")
            except Exception as e:
                print(f"Warning: Failed to read processed items for lens {lens_current}: {e}")
            
            # 过滤已处理item
            data_to_process = [item for item in data_to_process if item.get('unique_id') not in processed_ids]
            print(f"Filtered out {len(processed_ids)} processed items, remaining: {len(data_to_process)}")

        # 6. 无待处理数据则跳过
        if not data_to_process:
            print(f"No items left to process for lens {lens_current}. Skipping...\n")
            continue

        # 7. 多进程处理当前lens的数据
        batch_size = 1000  # 批量写入阈值（避免频繁IO）
        batch_buffer = []

        with Pool(processes=PROCESSES) as pool:
            print(f"Starting {PROCESSES} processes for lens {lens_current}...")
            # 用imap_unordered提高效率（无序返回结果）
            for result_item in tqdm(
                pool.imap_unordered(process_one, data_to_process, chunksize=1),
                total=len(data_to_process),
                desc=f"Lens {lens_current} Processing"
            ):
                if result_item:
                    batch_buffer.append(result_item)

                    # 批量写入JSONL
                    if len(batch_buffer) >= batch_size:
                        print(f"\nWriting batch of {len(batch_buffer)} items to {FINAL_JSONL_OUTPUT_PATH}")
                        with open(FINAL_JSONL_OUTPUT_PATH, 'a', encoding='utf-8') as f:
                            for item in batch_buffer:
                                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        batch_buffer = []

            # 处理剩余数据
            if batch_buffer:
                print(f"\nWriting final batch of {len(batch_buffer)} items to {FINAL_JSONL_OUTPUT_PATH}")
                with open(FINAL_JSONL_OUTPUT_PATH, 'a', encoding='utf-8') as f:
                    for item in batch_buffer:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')

        # 8. 当前lens处理完成
        print(f"\n" + "="*50)
        print(f"Processing for lens {lens_current} completed!")
        print(f"Output PNG dir: {OUTPUT_DIR}")
        print(f"Output JSONL: {FINAL_JSONL_OUTPUT_PATH}")
        print("="*50 + "\n")

    # 所有lens处理完成
    print("All lens values processed!")


if __name__ == '__main__':
    recover = True  # 恢复模式：跳过已处理的item（True=启用，False=全量重新处理）
    main()