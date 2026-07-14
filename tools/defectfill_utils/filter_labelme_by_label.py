"""
从 labelme 目录中按“字段/关键字”筛选：
- 只要某个 shape 的 label 包含该关键字，就认为该图片+json 命中
- 按“实际标签名”分目录输出（同一张图若命中多个标签，会复制到多个目录）

用法:
  python filter_labelme_by_label.py --input_dir <labelme目录> --keyword 压伤
  可选: --output_dir <输出根目录>
"""
import json
import shutil
from pathlib import Path
from typing import Optional, Set
import argparse
from tqdm import tqdm


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF"]


def find_image_for_json(json_path: Path) -> Optional[Path]:
    """查找与 json 同名的图片文件（在同一目录下）。"""
    for ext in IMAGE_EXTS:
        p = json_path.with_suffix(ext)
        if p.is_file():
            return p
    return None


def safe_dirname(name: str) -> str:
    """Windows/Unix 都安全的目录名。"""
    bad = '<>:"/\\|?*'
    out = "".join("_" if c in bad else c for c in name.strip())
    return out if out else "_EMPTY_LABEL_"


def main():
    parser = argparse.ArgumentParser(description="按关键字筛选 labelme 图片与标注，并按标签名分目录输出")
    parser.add_argument("--input_dir", type=str, required=True, help="labelme 目录（含 .json 与对应图片，可含子目录）")
    parser.add_argument("--keyword", type=str, required=True, help="关键字/字段，如 压伤（匹配 label 包含该字段）")
    parser.add_argument("--output_dir", type=str, default=None, help="输出根目录，默认与 input_dir 同级的 <input_dir>_filtered")
    parser.add_argument("--case_sensitive", action="store_true", help="是否区分大小写（默认不区分）")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"输入目录不存在: {input_dir}")

    keyword = args.keyword.strip()
    if not keyword:
        raise SystemExit("--keyword 不能为空")

    output_base = Path(args.output_dir).resolve() if args.output_dir else input_dir.parent / f"{input_dir.name}_filtered"
    output_base.mkdir(parents=True, exist_ok=True)

    # 递归扫描所有 json
    json_files = sorted(input_dir.rglob("*.json"))
    if not json_files:
        print(f"未找到任何 json：{input_dir}")
        return

    def norm(s: str) -> str:
        return s if args.case_sensitive else s.lower()

    kw = norm(keyword)

    copied = 0
    hit_files = 0

    for json_path in tqdm(json_files, desc="Scanning labelme json", unit="file"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"跳过 {json_path}: {e}")
            continue

        shapes = data.get("shapes", [])
        if not isinstance(shapes, list) or not shapes:
            continue

        # 找出该 json 中所有“包含关键字”的实际标签名
        matched_labels: Set[str] = set()
        for s in shapes:
            lbl = str(s.get("label", "")).strip()
            if not lbl:
                continue
            if kw in norm(lbl):
                matched_labels.add(lbl)

        if not matched_labels:
            continue

        img_path = find_image_for_json(json_path)
        if not img_path:
            print(f"跳过（无对应图片）: {json_path}")
            continue

        hit_files += 1

        # 同一文件可能命中多个 label：分别复制到对应 label 目录
        for lbl in sorted(matched_labels):
            out_dir = output_base / safe_dirname(lbl)
            out_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(json_path, out_dir / json_path.name)
            shutil.copy2(img_path, out_dir / img_path.name)
            copied += 1

    print(f"命中 json 文件数: {hit_files}")
    print(f"复制次数（按命中标签计）: {copied}")
    print(f"输出目录: {output_base}")

# python filter_labelme_by_label.py --input_dir ../../data/ng --keyword 擦伤
if __name__ == "__main__":
    main()