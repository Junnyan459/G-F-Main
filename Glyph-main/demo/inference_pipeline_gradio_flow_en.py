#!/usr/bin/env python3
"""
Text to Images + VLM Inference Gradio Service with Token Count and Multi-Model Comparison (Robust Streaming Version)

Usage:
    python text2img_vlm_gradio_with_tokens_streaming_robust.py
"""

import sys
import os
import json
import base64
import io
import math
import requests
import gradio as gr
from typing import List, Tuple, Dict, Any, Iterator
from PIL import Image
import threading
import queue
import time

# Attempt to import transformers, provide guidance if it fails
try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: The 'transformers' library is not installed. Please run 'pip install transformers sentencepiece'")
    sys.exit(1)

# ============================================================================
# General Setup
# ============================================================================

from scripts.word2png_function import text_to_images


# ============================================================================
# Configuration
# ============================================================================

CONFIG_EN_PATH = '../config/config_en.json'
CONFIG_CN_PATH = '../config/config_zh.json'
OUTPUT_DIR = './examples/output_images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

GLYPH_API_URL = "http://your_api_url:port/v1/chat/completions"
QWEN_API_URL = "http://your_api_url:port/v1/chat/completions"
MAX_PIXELS = 36000000

TOKENIZER_PATH = "" # if you have a local tokenizer path, set it here

# ============================================================================
# Story Loading, Tokenizer, and Image Encoding
# ============================================================================

def load_content(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File '{os.path.basename(file_path)}' not found."
    except Exception as e:
        return f"Error: Could not read file '{os.path.basename(file_path)}'."

# --- Unified Example Data Structure ---
all_examples = {
    "The Necklace (English)": {
        "content": load_content("./examples/case1.txt"),
        "question": "After taking on a massive debt to buy a replacement diamond necklace, what specific actions did the Loisels take and what sacrifices did they make to pay it off over the next ten years?"
    },
    "MultiDoc QA": {
        "content": load_content("./examples/case2.txt"),
        "question": "From which countries did the Norse originate?"
    },
    "Single Needle In A Haystack (NIAH)": {
        "content": load_content("./examples/case3.txt"),
        "question": "What is the special magic number for fretful-place mentioned in the provided text? The special magic number for fretful-place mentioned in the provided text is"
    },
    "Frequency Words Extraction (FWE)": {
        "content": load_content("./examples/case4.txt"),
        "question": "Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words. Question: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text? Answer: According to the coded text above, the three most frequently appeared words are:"
    }
}

# tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("zai-org/Glyph", trust_remote_code=True)

def count_text_tokens(text: str, tokenizer_instance) -> int:
    if not text or not tokenizer_instance: return 0
    return len(tokenizer_instance.encode(text))

def encode_image_with_max_pixels(image_path: str, max_pixels: int) -> str:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if w * h > max_pixels:
            scale = math.sqrt(max_pixels / (w * h))
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

# ============================================================================
# Model Inference Functions (Streaming Version - Corrected)
# ============================================================================

def vlm_inference_stream(question: str, encoded_images: List[Dict], api_url: str, question_token_count: int) -> Iterator[Any]:
    user_contents = encoded_images.copy()
    user_contents.append({'type': 'text', 'text': question.strip()})
    payload = {
        "model": "glyph", "messages": [{'role': 'user', 'content': user_contents}],
        "skip_special_tokens": False, "include_stop_str_in_output": False, "logprobs": False,
        "max_tokens": 8192, "top_p": 0.85, "top_k": -1, "temperature": 0.3, "repetition_penalty": 1.1,
        "stop_token_ids": [151329, 151348, 151336], "stream": True, "stream_options": {"include_usage": True}
    }
    headers = {'Content-Type': 'application/json'}
    usage_info = None
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=1200, stream=True)
        response.raise_for_status()
        thinking_started = False
        thinking_finished = False
        for chunk in response.iter_lines():
            if chunk:
                decoded_chunk = chunk.decode('utf-8')
                if decoded_chunk.startswith("data:"):
                    try:
                        data_str = decoded_chunk[5:].lstrip()
                        if data_str == "[DONE]": break
                        data = json.loads(data_str)
                        if 'usage' in data and data.get('usage'): usage_info = data['usage']
                        choices = data.get('choices', [])
                        if not choices: continue
                        delta = choices[0].get('delta', {})
                        if not delta: continue
                        reasoning_token = delta.get('reasoning_content')
                        answer_token = delta.get('content')
                        if reasoning_token:
                            if not thinking_started:
                                yield "<think>"; thinking_started = True
                            yield reasoning_token
                        if answer_token:
                            if thinking_started and not thinking_finished:
                                yield "</think>"; thinking_finished = True
                            yield answer_token
                    except json.JSONDecodeError: continue
        if thinking_started and not thinking_finished: yield "</think>"
        if usage_info:
            prompt_tokens = usage_info.get('prompt_tokens', 0)
            image_tokens = prompt_tokens - question_token_count
            yield {"type": "stats", "image_tokens": image_tokens}
    except requests.exceptions.RequestException as e: yield f"\n[VLM Inference Error: {e}]"
    except Exception as e: yield f"\n[Critical Error in VLM Stream Processing: {e}]"

def text_only_inference_stream(prompt: str, api_url: str) -> Iterator[str]:
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    payload = {"model": "Qwen3-8B", "messages": messages, "top_p": 0.8, "top_k": 50, "temperature": 0.3, "stream": True, "max_tokens": 8192}
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=1200, stream=True)
        response.raise_for_status()
        for chunk in response.iter_lines():
            if chunk:
                decoded_chunk = chunk.decode('utf-8')
                if decoded_chunk.startswith("data: "):
                    try:
                        data_str = decoded_chunk[6:]
                        if data_str.strip() == "[DONE]": break
                        data = json.loads(data_str)
                        content = data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                        if content: yield content
                    except json.JSONDecodeError: continue
    except requests.exceptions.RequestException as e: yield f"An error occurred during text model inference (URL: {api_url}): {str(e)}"
    except Exception as e: yield f"A critical error occurred in text stream: {str(e)}"

# ============================================================================
# Main Gradio Function (Generator with Queue)
# ============================================================================

_sentinel = object()

def stream_to_queue(stream: Iterator, q: queue.Queue, identifier):
    for item in stream:
        q.put((identifier, item))
    q.put((identifier, _sentinel))

def generate_and_ask(text_input, question_input, dpi, newline_choice) -> Iterator[Dict[gr.components.Component, Any]]:
    text = text_input.strip()
    question = question_input.strip()
    text_token_count = count_text_tokens(text, tokenizer)
    question_token_count = count_text_tokens(question, tokenizer)
    yield {
        text_token_output: text_token_count, question_token_output: question_token_count,
        qwen_output: "", vlm_output: "", image_output: None,
        image_token_output: "N/A (Streaming)", token_ratio_output: "N/A (Streaming)"
    }
    if not text or not question:
        yield {qwen_output: "Error: Text and question inputs cannot be empty."}
        return
    image_paths = []
    temp_config_path = None
    try:
        unique_id = f"req_{os.urandom(8).hex()}"
        contains_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
        base_config_path = CONFIG_CN_PATH if contains_chinese else CONFIG_EN_PATH
        with open(base_config_path, 'r', encoding='utf-8') as f: config_data = json.load(f)
        config_data['dpi'] = int(dpi)
        config_data['newline-markup'] = "<font color=\"#FF0000\"> \\n </font>" if newline_choice == "No" else "<br/>"
        temp_config_path = os.path.join(OUTPUT_DIR, f"{unique_id}_config.json")
        with open(temp_config_path, 'w', encoding='utf-8') as f: json.dump(config_data, f, ensure_ascii=False, indent=4)
        image_paths = text_to_images(text=text, output_dir=OUTPUT_DIR, config_path=temp_config_path, unique_id=unique_id)

        if not image_paths:
            yield {qwen_output: "Image generation failed."}
            return
        yield {image_output: image_paths}
    except Exception as e:
        yield {qwen_output: f"An error occurred during image generation: {e}"}
        return
    finally:
        if temp_config_path and os.path.exists(temp_config_path):
            try: os.remove(temp_config_path)
            except OSError as e: print(f"Failed to clean up temporary file: {e}")

    encoded_images_for_vlm = []
    try:
        for path in image_paths:
            encoded = encode_image_with_max_pixels(path, max_pixels=MAX_PIXELS)
            encoded_images_for_vlm.append({'type': 'image_url', 'image_url': {"url": f"data:image/png;base64,{encoded}"}})
    except Exception as e:
        yield {vlm_output: f"An error occurred during image encoding: {e}"}
        return
    
    try:
        q = queue.Queue()
        qwen_response, glyph_response = "", ""
        image_tokens, token_ratio = "N/A (Streaming)", "N/A (Streaming)"
        qwen_prompt = f"Please answer the question based on the following information.\n\n**Context**:\n```\n{text}\n```\n\n**Question**:\n{question}"
        qwen_stream = text_only_inference_stream(prompt=qwen_prompt, api_url=QWEN_API_URL)
        glyph_stream = vlm_inference_stream(question=question, encoded_images=encoded_images_for_vlm, api_url=GLYPH_API_URL, question_token_count=question_token_count)
        threading.Thread(target=stream_to_queue, args=(qwen_stream, q, "qwen")).start()
        threading.Thread(target=stream_to_queue, args=(glyph_stream, q, "glyph")).start()
        active_streams = 2
        while active_streams > 0:
            try:
                identifier, chunk = q.get(timeout=30)
                if chunk is _sentinel:
                    active_streams -= 1
                    continue
                if identifier == "qwen":
                    qwen_response += chunk
                elif identifier == "glyph":
                    if isinstance(chunk, str):
                        glyph_response += chunk
                    elif isinstance(chunk, dict) and chunk.get("type") == "stats":
                        img_toks = chunk.get("image_tokens", "Error")
                        image_tokens = img_toks
                        if isinstance(img_toks, int) and text_token_count > 0 and img_toks > 0:
                            token_ratio = f"{(text_token_count / img_toks):.2f}"
                        else:
                            token_ratio = "N/A"
                yield {
                    qwen_output: qwen_response, vlm_output: glyph_response,
                    image_token_output: image_tokens, token_ratio_output: token_ratio
                }
            except queue.Empty:
                glyph_response += "\n[Stream processing timed out]"
                qwen_response += "\n[Stream processing timed out]"
                yield {
                    qwen_output: qwen_response, vlm_output: glyph_response,
                    image_token_output: "Timeout", token_ratio_output: "Timeout"
                }
                break
    finally:
        # Clean up the generated image files after the stream is complete
        for path in image_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as e:
                    print(f"Failed to delete image file {path}: {e}")

# ============================================================================
# Create and Launch Gradio Service (UI part)
# ============================================================================
with gr.Blocks(theme=gr.themes.Soft(font=["Helvetica Neue", "Arial", "sans-serif"])) as demo:
    gr.Markdown(
        """
        # DEMO of Glyph
        ### 💡 **How to Use:**
        1.  **Select an Example (Optional):** Choose a pre-loaded example from the list below to auto-fill the text and question fields.
        2.  **Enter Custom Input:** Or, enter your own text in the **Input Text** box and a question in the **Input Question** box.
        3.  **Configure and Run:** Select the rendering **DPI** and click the **"Generate Image and Infer"** button.
        4.  **View Results:** The rendered images will appear first, followed by real-time streaming answers from Qwen3-8B (based on text) and Glyph (based on images).
        
        ❤️ If you like our project, please give us a star on <a href="https://github.com/thu-coai/Glyph" target="_blank">GitHub</a>!
        """
    )

    # --- [FIXED] Unified Example Loading Logic at the Top ---
    gr.Markdown("<h3 id='examples'>🚀 Quick Start: Select an Example</h3>")
    
    # Create a list of choices for the Radio selector from the unified dictionary keys
    example_choices = list(all_examples.keys())
    
    example_selector = gr.Radio(
        label="Select an Example Use Case",
        choices=example_choices,
        value=None
    )

    # Create the data for the table display
    static_table_data = [
        # ["The Necklace", "Document Q&A", "what specific actions did the Loisels take and what sacrifices did they make to pay it off over the next ten years?", "To pay for the replacement necklace, Mathilde and her husband, Monsieur Loisel, undertook a decade of grueling work and extreme saving."],
        ["The Necklace", "Document Q&A", "What specific actions did the Loisels ...", "To pay for the replacement necklace, Mathilde and her husband, Monsieur Loisel, undertook a decade of grueling work and extreme saving."],
        ["Example of MultiDoc QA", "Multi-Document Q&A", "From which countries did the Norse originate?", "Denmark, Iceland and Norway"],
        ["Example of NIAH", "Single Needle In A Haystack (NIAH)", "What is the special magic number for fretful-place?", "8971465"],
        ["Example of FWE", "Frequent Word Extraction", "What are the three most frequently appeared words in the coded text?", "The three most frequently appeared words are: 'tsamyl', 'kjsvgb', and 'ultalu'."]
    ]
    task_info_table = gr.DataFrame(headers=["Task Name", "Task Type", "Question", "Answer"], value=static_table_data, interactive=False)


    # --- Main UI Layout ---
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(label="Input Text (for image generation)", placeholder="Enter text here or select an example above...", lines=20)
            question_input = gr.Textbox(label="Input Question (about the text/images)", placeholder="Enter a question here or select an example above...", lines=4)
            gr.Markdown("### Rendering Parameters (affects Glyph model only)")
            with gr.Row():
                dpi_input = gr.Radio(label="DPI (Clarity)", choices=[72, 96], value=72)
                newline_input = gr.Radio(label="Use Newlines?", choices=["Yes", "No"], value="Yes")
        with gr.Column(scale=3):
            image_output = gr.Gallery(label="Generated Images for Glyph Model", show_label=True, elem_id="gallery", columns=4, object_fit="contain", height=425)
            with gr.Row():
                qwen_output = gr.Textbox(label="Qwen3-8B's Answer (from text)", lines=11, max_lines=11, interactive=False)
                vlm_output = gr.Textbox(label="Glyph's Answer (from images)", lines=11, max_lines=11, interactive=False)
            with gr.Row():
                text_token_output = gr.Textbox(label="Input Text Tokens", interactive=False)
                question_token_output = gr.Textbox(label="Input Question Tokens", interactive=False)
                image_token_output = gr.Textbox(label="Input Image Tokens (Glyph)", interactive=False, value="N/A (Streaming)")
                token_ratio_output = gr.Textbox(label="Token Compression Ratio", interactive=False, value="N/A (Streaming)")

    submit_btn = gr.Button("Generate Image and Infer", variant="primary")


    # --- [FIXED] Unified Event Handler for Examples ---
    def load_example(example_title: str) -> Tuple[str, str]:
        """Loads content and question from the unified 'all_examples' dictionary."""
        if not example_title or example_title not in all_examples:
            return "", ""
        
        example = all_examples[example_title]
        return example["content"], example["question"]

    # Connect the radio button to the handler function
    example_selector.change(
        fn=load_example,
        inputs=example_selector,
        outputs=[text_input, question_input]
    )
    
    # --- Submit button action ---
    inputs_list = [text_input, question_input, dpi_input, newline_input]
    outputs_list = [image_output, qwen_output, vlm_output, text_token_output, question_token_output, image_token_output, token_ratio_output]

    submit_btn.click(fn=generate_and_ask, inputs=inputs_list, outputs=outputs_list)

if __name__ == '__main__':
    print("Launching Gradio service (Robust Streaming Version)...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)