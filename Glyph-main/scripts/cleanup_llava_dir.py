#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_LLAVA_PATH = Path("/root/autodl-tmp/llava")

REQUIRED_FOR_TRANSFORMERS_INFERENCE = {
    "config.json",
    "generation_config.json",
    "mm_projector.bin",
    "pytorch_model-00001-of-00002.bin",
    "pytorch_model-00002-of-00002.bin",
    "pytorch_model.bin.index.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "tokenizer_config.json",
}

CANDIDATE_REMOVALS = {
    "README.md": "Documentation only.",
    "gitattributes": "Repository metadata only.",
    "model": "Original LLaVA source snapshot. Not needed by the transformers-based pipeline in this repo.",
}


def format_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def iter_cleanup_targets(base_dir: Path) -> list[tuple[Path, str]]:
    targets: list[tuple[Path, str]] = []
    for name, reason in CANDIDATE_REMOVALS.items():
        path = base_dir / name
        if path.exists():
            targets.append((path, reason))
    return targets


def delete_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run or apply safe cleanup for the local LLaVA directory.")
    parser.add_argument("--llava-path", type=Path, default=DEFAULT_LLAVA_PATH)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if not args.llava_path.exists():
        raise FileNotFoundError(f"LLaVA path does not exist: {args.llava_path}")

    print(f"LLaVA directory: {args.llava_path}")
    print("Required files kept for this repo's transformers-based pipeline:")
    for name in sorted(REQUIRED_FOR_TRANSFORMERS_INFERENCE):
        print(f"  KEEP  {name}")

    targets = iter_cleanup_targets(args.llava_path)
    if not targets:
        print("No cleanup candidates found.")
        return

    reclaimed_bytes = 0
    print("Cleanup candidates:")
    for path, reason in targets:
        size_bytes = format_size(path)
        reclaimed_bytes += size_bytes
        print(f"  DROP  {path.name}  ({size_bytes / (1024 ** 2):.2f} MiB)  {reason}")

    print(f"Estimated reclaimable space: {reclaimed_bytes / (1024 ** 2):.2f} MiB")
    if not args.apply:
        print("Dry run only. Re-run with --apply to delete the candidates above.")
        return

    for path, _ in targets:
        delete_path(path)
        print(f"Deleted {path}")


if __name__ == "__main__":
    main()
