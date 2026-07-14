"""
将 LabelMe 标注目录按「标签类别」拆分：
- 在指定输出根目录下，为每个出现过的标签名创建子目录
- 若某张 json 含某标签，则把该 json 与同名的图片复制到对应标签子目录
- 同一张图含多个不同标签时，会分别复制到多个标签目录（各一份）
- 若不同子目录存在同名 json，复制到同一标签目录时会自动加短后缀避免覆盖

用法:
  python labelme_split_by_category.py \
  --input_dir /mnt/d/0403/img_json \
  --output_dir /mnt/d/0403/img_json/split_by_category

可选:
  --skip_empty_label   忽略 label 为空的 shape

运行结束会打印：总复制次数、类别数，以及每个标签子目录下的「图片+json」对数量。
"""
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Optional, Set
import argparse
from tqdm import tqdm


IMAGE_EXTS = [
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
    ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF",
]


def find_image_for_json(json_path: Path) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        p = json_path.with_suffix(ext)
        if p.is_file():
            return p
    return None


def safe_dirname(name: str) -> str:
    bad = '<>:"/\\|?*'
    out = "".join("_" if c in bad else c for c in name.strip())
    return out if out else "_EMPTY_LABEL_"


def dest_pair_paths(out_dir: Path, json_path: Path, img_path: Path) -> tuple[Path, Path]:
    """若目标已存在（非同源重复），为 json/图片生成同名 stem 的唯一文件名对。"""
    j_dst = out_dir / json_path.name
    i_dst = out_dir / img_path.name
    if not j_dst.exists() and not i_dst.exists():
        return j_dst, i_dst
    # 已存在则可能来自另一路径；用源路径哈希区分
    h = hashlib.md5(str(json_path.resolve()).encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    stem = json_path.stem
    j_dst = out_dir / f"{stem}__{h}.json"
    i_dst = out_dir / f"{stem}__{h}{img_path.suffix}"
    n = 1
    while j_dst.exists() or i_dst.exists():
        j_dst = out_dir / f"{stem}__{h}_{n}.json"
        i_dst = out_dir / f"{stem}__{h}_{n}{img_path.suffix}"
        n += 1
    return j_dst, i_dst


def labels_in_json(data: dict, skip_empty: bool) -> Set[str]:
    shapes = data.get("shapes", [])
    if not isinstance(shapes, list):
        return set()
    out: Set[str] = set()
    for s in shapes:
        lbl = str(s.get("label", "")).strip()
        if not lbl and skip_empty:
            continue
        if not lbl:
            lbl = ""
        out.add(lbl)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="按标签类别拆分 LabelMe：每类一目录，复制对应 json 与图片到指定根目录下"
    )
    parser.add_argument("--input_dir", type=str, required=True, help="含 .json 与对应图片的 LabelMe 目录（可递归子目录）")
    parser.add_argument("--output_dir", type=str, required=True, help="输出根目录；其下将创建各标签名子目录")
    parser.add_argument(
        "--skip_empty_label",
        action="store_true",
        help="忽略无标签名的 shape；否则空标签会归入 _EMPTY_LABEL_ 目录",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_base = Path(args.output_dir).resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"输入目录不存在: {input_dir}")

    output_base.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.rglob("*.json"))
    if not json_files:
        print(f"未找到任何 json：{input_dir}")
        return

    copy_ops = 0
    skipped_no_image = 0
    skipped_bad_json = 0
    skipped_no_labels = 0
    # 每个输出子目录（标签类别）内复制的「图片+json」对数
    category_counts: Counter[str] = Counter()

    for json_path in tqdm(json_files, desc="Split by category", unit="file"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"跳过（无法解析）: {json_path} — {e}")
            skipped_bad_json += 1
            continue

        labels = labels_in_json(data, skip_empty=args.skip_empty_label)
        if not labels:
            skipped_no_labels += 1
            continue

        img_path = find_image_for_json(json_path)
        if not img_path:
            print(f"跳过（无对应图片）: {json_path}")
            skipped_no_image += 1
            continue

        for lbl in sorted(labels):
            dname = safe_dirname(lbl) if lbl else "_EMPTY_LABEL_"
            out_dir = output_base / dname
            out_dir.mkdir(parents=True, exist_ok=True)
            j_dst, i_dst = dest_pair_paths(out_dir, json_path, img_path)
            shutil.copy2(json_path, j_dst)
            shutil.copy2(img_path, i_dst)
            copy_ops += 1
            category_counts[dname] += 1

    print(f"扫描 json 数: {len(json_files)}")
    print(f"复制次数（按「文件×命中标签」计）: {copy_ops}")
    print(f"类别数（输出子目录数）: {len(category_counts)}")
    print(f"跳过（无图）: {skipped_no_image}, 跳过（坏 json）: {skipped_bad_json}, 跳过（无有效标签）: {skipped_no_labels}")
    print(f"输出根目录: {output_base}")
    print("\n各类别数量（该目录下复制的 图片+json 对数；同一张图多标签会多次计入）:")
    for dname, cnt in sorted(category_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {dname}: {cnt}")


if __name__ == "__main__":
    main()
