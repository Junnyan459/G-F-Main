#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from glyph_ficoco import reduce_language_tokens, reduce_visual_tokens
from scripts.word2png_function import text_to_images


DEFAULT_MODEL_PATH = Path("/root/autodl-tmp/Glyph")
DEFAULT_RENDER_CONFIG = REPO_ROOT / "config" / "config_en.json"
DEFAULT_STORY_IMAGE = REPO_ROOT / "assets" / "Little_Red_Riding_Hood.png"
DEFAULT_OUTPUT_DIR = Path("/tmp/glyph_ficoco_minimal_case")


def render_sample_text(output_dir: Path) -> list[str]:
    sample_text = (
        "Little Red Riding Hood walked through the forest.\n"
        "A wolf disguised himself as her grandmother.\n"
        "The hunter arrived and saved Little Red Riding Hood."
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return text_to_images(
        text=sample_text,
        output_dir=str(output_dir),
        config_path=str(DEFAULT_RENDER_CONFIG),
        unique_id="minimal_case",
    )


def run_local_glyph_inference(model_path: Path, image_path: Path, max_new_tokens: int) -> str:
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": str(image_path)},
                {"type": "text", "text": "Who pretended to be Little Red Riding Hood's grandmother?"},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=False,
    )


def run_ficoco_smoke_test(device: torch.device) -> None:
    torch.manual_seed(0)

    visual_embeddings = torch.randn(1, 17, 32, device=device)
    visual_attention = torch.randn(1, 17, 17, device=device)
    visual_result = reduce_visual_tokens(
        visual_embeddings,
        visual_attention,
        reduction_factor=4,
        include_class_token=True,
    )
    assert visual_result.reduced_embeddings.shape == (1, 13, 32)

    multimodal_embeddings = torch.randn(1, 20, 32, device=device)
    multimodal_attention = torch.randn(1, 20, 20, device=device)
    language_result = reduce_language_tokens(
        multimodal_embeddings,
        multimodal_attention,
        visual_token_count=8,
        reduction_factor=3,
    )
    assert language_result.reduced_embeddings.shape == (1, 17, 32)



def main() -> None:
    parser = argparse.ArgumentParser(description="Run the merged Glyph + FiCoCo minimal case.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()

    print(f"[1/3] Rendering sample text with Glyph renderer -> {args.output_dir}")
    rendered_images = render_sample_text(args.output_dir)
    if not rendered_images:
        raise RuntimeError("Glyph renderer did not produce any images")
    print(f"Rendered {len(rendered_images)} page(s)")
    print(f"First page: {rendered_images[0]}")

    print("[2/3] Running FiCoCo compression smoke test")
    smoke_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_ficoco_smoke_test(smoke_device)
    print(f"FiCoCo smoke test passed on {smoke_device}")

    if args.skip_model:
        print("[3/3] Skipped local Glyph model inference")
        return

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")

    print(f"[3/3] Running local Glyph model inference with {args.model_path}")
    answer = run_local_glyph_inference(
        model_path=args.model_path,
        image_path=DEFAULT_STORY_IMAGE,
        max_new_tokens=args.max_new_tokens,
    )
    print("Model output:")
    print(answer.strip())


if __name__ == "__main__":
    main()
