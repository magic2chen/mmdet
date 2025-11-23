#!/usr/bin/env python3
"""Normalize COCO image paths between Windows and POSIX.

This script replaces backslashes in every image ``file_name`` entry with
forward slashes so that the Data can be consumed on Linux/macOS.

Usage:
    python tools/fix_coco_backslashes.py path/to/instances_train2017.json
    python tools/fix_coco_backslashes.py input.json --output output.json

If ``--output`` is omitted, the input file will be updated in place.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def normalize_file_names(data: Dict[str, Any]) -> int:
    """Replace backslashes with slashes in ``images[*]['file_name']``."""
    images = data.get("images", [])
    changed = 0
    for image in images:
        file_name = image.get("file_name")
        if isinstance(file_name, str):
            normalized = file_name.replace("\\", "/")
            if normalized != file_name:
                image["file_name"] = normalized
                changed += 1
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix backslashes in COCO JSON image file names."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the COCO annotation JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. Defaults to overwriting the input file.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help="Indent level for JSON dump. Defaults to compact output.",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    changed = normalize_file_names(data)

    output_path = args.output or args.input
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=args.indent, ensure_ascii=False)
        if args.indent is not None:
            f.write("\n")

    print(f"Processed {len(data.get('images', []))} images.")
    print(f"Updated {changed} file names.")
    print(f"Saved to {output_path}")

# python tools/fix_coco_backslashes.py Data/coco_dataset2_split/annotations/instances_train2017.json
# python tools/fix_coco_backslashes.py Data/coco_dataset2_split/annotations/instances_val2017.json
# python tools/fix_coco_backslashes.py Data/coco_dataset2_split/annotations/instances_test2017.json
if __name__ == "__main__":
    main()
