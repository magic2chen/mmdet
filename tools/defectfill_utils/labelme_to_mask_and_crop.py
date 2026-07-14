"""
Labelme 标注目录 → Mask 生成 / 缺陷块裁剪（二合一）

用法:
  Mask 生成（与 DefectFill 推理 1:1 配对兼容）:
    python labelme_to_mask_and_crop.py \
    --mode mask \
    --input_dir /mnt/d/0403/1/4A_FM7HRK000H00001EAH \
    --output_dir /mnt/d/0403/1/4A_FM7HRK000H00001EAH/output_mask \
    --object_class mhuashang

  缺陷块裁剪:
    python labelme_to_mask_and_crop.py \
    --mode crop \
    --input_dir <labelme目录> \
    --output_dir <输出目录> \
    [--patch_size 256]
"""
import os
import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from tqdm import tqdm

# ---------- 共用：从 labelme 解析形状 ----------

def create_mask_from_polygon(points: List, image_shape: Tuple[int, int]) -> np.ndarray:
    """从多边形点创建 mask。image_shape = (H, W)。"""
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def shapes_to_single_mask(shapes: List[Dict], image_height: int, image_width: int) -> Optional[np.ndarray]:
    """将 labelme shapes（polygon/rectangle）合并为一张二值 mask。"""
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    for shape in shapes:
        label = shape.get("label", "")
        points = shape.get("points", [])
        shape_type = shape.get("shape_type", "")
        if not points:
            continue
        if shape_type == "polygon":
            part = create_mask_from_polygon(points, (image_height, image_width))
            mask = cv2.bitwise_or(mask, part)
        elif shape_type == "rectangle" and len(points) >= 2:
            pt1 = (int(points[0][0]), int(points[0][1]))
            pt2 = (int(points[1][0]), int(points[1][1]))
            cv2.rectangle(mask, pt1, pt2, 255, -1)
    return mask if np.any(mask > 0) else None


def collect_labels_and_shapes(json_path: Path) -> Tuple[List[Dict], Dict[str, List]]:
    """返回 (all_shapes, labels_in_image)。labels_in_image: label -> [(points, shape_type), ...]"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    shapes = data.get("shapes", [])
    labels_in_image = {}
    for s in shapes:
        label = s.get("label", "")
        points = s.get("points", [])
        st = s.get("shape_type", "")
        if not label or not points:
            continue
        if label not in labels_in_image:
            labels_in_image[label] = []
        labels_in_image[label].append((points, st))
    return shapes, labels_in_image


def find_image_for_json(json_path: Path) -> Optional[Path]:
    """根据 json 路径查找同名的图片文件。"""
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".PNG", ".JPG"]:
        candidate = json_path.with_suffix(ext)
        if candidate.is_file():
            return candidate
    return None


# ---------- Mode: mask ----------
# 从 labelme 目录生成全图 mask，输出结构适配 DefectFill（test/good + test/masks，1:1 配对）


def run_mask_mode(
    input_dir: Path,
    output_dir: Path,
    object_class: str,
) -> None:
    """
    扫描 input_dir 下所有 .json，对每张图生成一张合并后的 mask（所有 label 合并为一块缺陷区域）。
    输出：
      output_dir / object_class / test / good /   -> 原图 PNG
      output_dir / object_class / test / masks /  -> {stem}_mask.png（仅对有标注的图生成，实现 1:1）
    无标注的图只放入 good，不生成 mask。
    """
    good_dir = output_dir / object_class / "test" / "good"
    masks_dir = output_dir / object_class / "test" / "masks"
    good_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        print(f"未在 {input_dir} 下找到 .json 文件")
        return

    good_count = 0
    mask_count = 0
    for json_path in tqdm(json_files, desc="Mask mode", unit="file"):
        image_path = find_image_for_json(json_path)
        if not image_path:
            print(f"跳过（无对应图片）: {json_path.name}")
            continue

        try:
            with Image.open(image_path) as img:
                w, h = img.size
            shapes, labels_in_image = collect_labels_and_shapes(json_path)

            if not shapes:
                # 无标注 → 仅复制到 good
                out_png = good_dir / (image_path.stem + ".png")
                with Image.open(image_path) as img:
                    if img.mode not in ("RGB", "RGBA", "L", "P"):
                        img = img.convert("RGB")
                    img.save(out_png, "PNG")
                good_count += 1
                continue

            # 有标注：合并所有形状为一张 mask
            mask = np.zeros((h, w), dtype=np.uint8)
            for label, shape_list in labels_in_image.items():
                for points, shape_type in shape_list:
                    if shape_type == "polygon":
                        part = create_mask_from_polygon(points, (h, w))
                        mask = cv2.bitwise_or(mask, part)
                    elif shape_type == "rectangle" and len(points) >= 2:
                        pt1 = (int(points[0][0]), int(points[0][1]))
                        pt2 = (int(points[1][0]), int(points[1][1]))
                        cv2.rectangle(mask, pt1, pt2, 255, -1)

            if not np.any(mask > 0):
                out_png = good_dir / (image_path.stem + ".png")
                with Image.open(image_path) as img:
                    if img.mode not in ("RGB", "RGBA", "L", "P"):
                        img = img.convert("RGB")
                    img.save(out_png, "PNG")
                good_count += 1
                continue

            # 保存原图到 good、mask 到 masks（DefectFill 1:1：xxx.png 对应 xxx_mask.png）
            out_img = good_dir / (image_path.stem + ".png")
            out_mask = masks_dir / (image_path.stem + "_mask.png")
            with Image.open(image_path) as img:
                if img.mode not in ("RGB", "RGBA", "L", "P"):
                    img = img.convert("RGB")
                img.save(out_img, "PNG")
            Image.fromarray(mask).save(out_mask)
            good_count += 1
            mask_count += 1
        except Exception as e:
            print(f"处理失败 {json_path.name}: {e}")

    print(f"Mask 模式完成: good 图片 {good_count}，生成 mask {mask_count}，输出 {output_dir / object_class / 'test'}")


# ---------- Mode: crop ----------
# 按缺陷外接框裁剪小图块，并写回裁剪后的 labelme json（可选保存裁剪后的 mask）


def get_bbox_from_points(points: List[List[float]]) -> Tuple[int, int, int, int]:
    pts = np.array(points)
    x_min = int(np.min(pts[:, 0]))
    y_min = int(np.min(pts[:, 1]))
    x_max = int(np.max(pts[:, 0]))
    y_max = int(np.max(pts[:, 1]))
    return x_min, y_min, x_max, y_max


def crop_region(defect_bbox: Tuple[int, int, int, int], img_w: int, img_h: int, patch_size: int) -> Tuple[int, int, int, int]:
    x_min, y_min, x_max, y_max = defect_bbox
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    half = patch_size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(img_w, cx + half)
    y2 = min(img_h, cy + half)
    if x2 - x1 < patch_size:
        if x1 == 0:
            x2 = min(img_w, x1 + patch_size)
        else:
            x1 = max(0, x2 - patch_size)
    if y2 - y1 < patch_size:
        if y1 == 0:
            y2 = min(img_h, y1 + patch_size)
        else:
            y1 = max(0, y2 - patch_size)
    return x1, y1, x2, y2


def adjust_points(points: List[List[float]], dx: int, dy: int) -> List[List[float]]:
    return [[p[0] - dx, p[1] - dy] for p in points]


def run_crop_mode(
    input_dir: Path,
    output_dir: Path,
    patch_size: int,
) -> None:
    """
    从大图中按 labelme 标注的缺陷外接框裁剪 patch_size x patch_size 的小图，
    并生成裁剪后的 labelme json（坐标已平移）。输出：output_dir/images/{label}/, output_dir/labels/{label}/。
    """
    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        print(f"未在 {input_dir} 下找到 .json 文件")
        return

    patch_counter: Dict[str, int] = {}
    total = 0
    for json_path in tqdm(json_files, desc="Crop mode", unit="file"):
        image_path = find_image_for_json(json_path)
        if not image_path:
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            with Image.open(image_path) as img:
                img_w, img_h = img.size
            shapes = data.get("shapes", [])
            if not shapes:
                continue
            _, labels_in_image = collect_labels_and_shapes(json_path)

            for label, shape_list in labels_in_image.items():
                all_x_min, all_y_min = float("inf"), float("inf")
                all_x_max, all_y_max = 0, 0
                for points, _ in shape_list:
                    xa, ya, xb, yb = get_bbox_from_points(points)
                    all_x_min = min(all_x_min, xa)
                    all_y_min = min(all_y_min, ya)
                    all_x_max = max(all_x_max, xb)
                    all_y_max = max(all_y_max, yb)
                defect_bbox = (int(all_x_min), int(all_y_min), int(all_x_max), int(all_y_max))
                x1, y1, x2, y2 = crop_region(defect_bbox, img_w, img_h, patch_size)

                crop_w, crop_h = x2 - x1, y2 - y1
                with Image.open(image_path) as img:
                    cropped = img.crop((x1, y1, x2, y2))
                if cropped.size[0] < patch_size or cropped.size[1] < patch_size:
                    new_img = Image.new("RGB", (patch_size, patch_size), (0, 0, 0))
                    new_img.paste(cropped, (0, 0))
                    cropped = new_img
                    scale_x, scale_y = 1.0, 1.0
                else:
                    cropped = cropped.resize((patch_size, patch_size), Image.Resampling.LANCZOS)
                    scale_x = patch_size / max(1, crop_w)
                    scale_y = patch_size / max(1, crop_h)

                patch_counter[label] = patch_counter.get(label, 0) + 1
                stem = f"{image_path.stem}_{label}_{patch_counter[label]:04d}"
                patch_filename = stem + ".png"
                (out_images / label).mkdir(parents=True, exist_ok=True)
                cropped.save(out_images / label / patch_filename)

                new_shapes = []
                for points, shape_type in shape_list:
                    adj = adjust_points(points, x1, y1)
                    scaled = [[p[0] * scale_x, p[1] * scale_y] for p in adj]
                    valid = [p for p in scaled if 0 <= p[0] < patch_size and 0 <= p[1] < patch_size]
                    if len(valid) >= 3:
                        new_shapes.append({
                            "label": label,
                            "points": valid,
                            "group_id": None,
                            "shape_type": shape_type,
                            "flags": {},
                        })
                if new_shapes:
                    new_data = {
                        "version": data.get("version", "5.0.0"),
                        "flags": {},
                        "shapes": new_shapes,
                        "imagePath": patch_filename,
                        "imageData": None,
                        "imageHeight": patch_size,
                        "imageWidth": patch_size,
                    }
                    (out_labels / label).mkdir(parents=True, exist_ok=True)
                    with open(out_labels / label / (stem + ".json"), "w", encoding="utf-8") as f:
                        json.dump(new_data, f, indent=2, ensure_ascii=False)
                total += 1
        except Exception as e:
            print(f"处理失败 {json_path.name}: {e}")

    print(f"Crop 模式完成: 共 {total} 个 patch，输出 {output_dir}")
    for label, c in sorted(patch_counter.items()):
        print(f"  {label}: {c}")


def main():
    parser = argparse.ArgumentParser(
        description="Labelme 标注目录 → Mask 生成（DefectFill 1:1）或缺陷块裁剪",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", type=str, choices=["mask", "crop"], default="mask",
                        help="mask=生成全图 mask（test/good + test/masks）；crop=按缺陷裁剪小图块")
    parser.add_argument("--input_dir", type=str, default=".",
                        help="包含 labelme .json 与对应图片的目录")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="mask 模式：输出根目录（其下生成 object_class/test/...）；crop 模式：patch 输出根目录")
    parser.add_argument("--object_class", type=str, default="object",
                        help="仅 mask 模式使用，用于生成 output_dir/object_class/test/good 与 test/masks")
    parser.add_argument("--patch_size", type=int, default=256,
                        help="仅 crop 模式使用，裁剪块边长")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_dir.is_dir():
        print(f"输入目录不存在: {input_dir}")
        return

    if args.mode == "mask":
        run_mask_mode(input_dir, output_dir, args.object_class)
    else:
        run_crop_mode(input_dir, output_dir, args.patch_size)

'''
python labelme_to_mask_and_crop.py \
--mode mask \
--input_dir ../../data/20260414-2/20260414-2/1A_FM7HSF000QN0001EAH \
--output_dir ../../data/20260414-2/20260414-2/1A_FM7HSF000QN0001EAH/masked \
--object_class phone
'''

'''
python labelme_to_mask_and_crop.py \
--mode mask \
--input_dir ../DATA/my_infer/phone/test/good \
--output_dir ../DATA/my_infer/phone/test/good/cashang_masked \
--object_class phone
'''


# python labelme_to_mask_and_crop.py --mode crop --input_dir /path/to/labelme_dir --output_dir /path/to/patches --patch_size 256
if __name__ == "__main__":
    main()
