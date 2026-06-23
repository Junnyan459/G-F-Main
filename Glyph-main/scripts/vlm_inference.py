#!/usr/bin/env python3
"""
Simple VLM (Vision Language Model) Inference for ZhipuAI API

Usage:
    from vlm_inference import vlm_inference
    
    response = vlm_inference(
        question="What is in this image?",
        image_paths=["./image1.png", "./image2.png"]
    )
"""

import os
import json
import base64
import io
import math
import requests
from typing import List, Optional
from PIL import Image

# Clear proxy settings
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "glm-4.1v-thinking-flashx"  # 请确认该模型名称在当前API中是否准确
API_KEY = "4d86190163a341bba69de7cb7505efc7.fpjipPDXxDcoTDnB" 
API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MAX_PIXELS = 102400
# ============================================================================
# Helper Functions
# ============================================================================

def encode_image_with_max_pixels(image_path: str, max_pixels: int = 1000000) -> str:
    """Encode image to base64 string with pixel limit"""
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if w * h > max_pixels:
            scale = math.sqrt(max_pixels / (w * h))
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def vlm_inference(
    question: str,
    image_paths: Optional[List[str]] = None,
    api_url: str = API_URL,
    api_key: str = API_KEY,
    model_name: str = MODEL_NAME,
    max_pixels: int = MAX_PIXELS,
    max_tokens: int = 8192,
    temperature: float = 0.0001,
    top_p: float = 1.0
) -> Optional[str]:
    """
    Vision Language Model Inference for ZhipuAI API
    
    Args:
        question: Your question/prompt
        image_paths: List of image file paths (optional)
        api_url: API endpoint URL
        api_key: API authentication key
        model_name: Model name
        max_pixels: Maximum pixels for image encoding
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        
    Returns:
        str: Model response text, or None if error
    """
    # Clean input
    question = question.strip() if question else ""
    
    # Build message content
    user_contents = []
    
    # Add images if provided
    if image_paths:
        for image_path in image_paths:
            image_path = image_path.strip()
            if os.path.exists(image_path):
                try:
                    encoded = encode_image_with_max_pixels(image_path, max_pixels=max_pixels)
                    user_contents.append({
                        'type': 'image_url',
                        'image_url':{"url": f"data:image/png;base64,{encoded}"}
                    })
                except Exception as e:
                    print(f"Warning: Failed to encode image {image_path}: {e}")
            else:
                print(f"Warning: Image not found: {image_path}")
    
    # Add text question
    user_contents.append({'type': 'text', 'text': question})
    
    # Build request payload with only supported parameters
    messages = [{'role': 'user', 'content': user_contents}]
    
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    # Make API request
    try:
        response = requests.post(api_url, headers=headers,  json=payload,  # 使用 json 参数，requests 会自动处理 Content-Type 和编码
timeout=120
)
        
        # Check HTTP status
        if response.status_code != 200:
            print(f"HTTP error: {response.status_code}, Response: {response.text}")
            return None
        
        # Parse JSON response
        try:
            result = response.json()
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}, Raw response: {response.text}")
            return None
        
        # Validate response structure
        if 'choices' not in result or not result['choices']:
            print(f"Invalid response structure: {result}")
            return None
        
        if 'message' not in result['choices'][0] or 'content' not in result['choices'][0]['message']:
            print(f"Missing content in response: {result}")
            return None
        
        return result['choices'][0]['message']['content']
        
    except requests.exceptions.Timeout:
        print("Request timeout")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        return None


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == '__main__':
    # Example 1: Text only
    # response = vlm_inference(
    #     question="What is 2+2?"
    # )
    # print("Text-only response:")
    # print(response)
    # print()
    
    # Example 2: With images
    response = vlm_inference(
        question="Based on the story in the figures, what is the ending of wolf?",
        image_paths=[
            "/root/code/Glyph-main/scripts/output_images/test_001/page_001.png"
        ]
    )
    print("Image response:")
    print(response)

    