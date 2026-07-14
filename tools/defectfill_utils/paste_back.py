"""
将 512x512 的推理结果粘贴回原始大图中
使用与 inference.py 相同的 smart_crop 逻辑来定位裁剪区域
"""
import os
import cv2
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


def smart_crop_get_coords(image, mask, base_size=512):
    """
    返回 smart_crop 的裁剪坐标（与 inference.py 中的逻辑完全一致）
    
    Returns:
        x1, y1, x2, y2: 裁剪区域的坐标
        crop_size: 裁剪前的尺寸（用于判断是否需要缩放）
    """
    h, w = image.shape[:2]
    
    # Find the Bounding Box of the defect
    y_indices, x_indices = np.where(mask > 0)
    
    if len(y_indices) == 0:
        # No defect? Return center crop 512
        cy, cx = h // 2, w // 2
        crop_size = base_size
    else:
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        min_x, max_x = np.min(x_indices), np.max(x_indices)
        
        defect_h = max_y - min_y
        defect_w = max_x - min_x
        
        # Center of the defect
        cy = min_y + defect_h // 2
        cx = min_x + defect_w // 2
        
        # Determine the Crop Size
        max_dim = max(defect_h, defect_w)
        padding = 50
        
        crop_size = max(base_size, max_dim + padding)
    
    # Calculate Crop Coordinates (Square Box)
    half_size = crop_size // 2
    x1 = cx - half_size
    y1 = cy - half_size
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    
    # Handle Edge Cases (Shift box if it goes out of bounds)
    if x1 < 0: x2 -= x1; x1 = 0
    if y1 < 0: y2 -= y1; y1 = 0
    if x2 > w: x1 -= (x2 - w); x2 = w
    if y2 > h: y1 -= (y2 - h); y2 = h
    
    # Double check we didn't shrink below image dims
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    
    return x1, y1, x2, y2, crop_size


def paste_back_to_original(original_image, generated_512, mask, base_size=512, blend_margin=10):
    """
    将 512x512 的生成结果粘贴回原始大图
    
    Args:
        original_image: 原始大图 (H, W, 3)
        generated_512: 生成的 512x512 图片 (512, 512, 3)
        mask: 原始大图的 mask (H, W)
        base_size: 裁剪基准尺寸
        blend_margin: 边缘融合宽度（像素）
        
    Returns:
        result_image: 粘贴后的完整图片
    """
    h, w = original_image.shape[:2]
    result_image = original_image.copy()
    
    # 获取裁剪坐标
    x1, y1, x2, y2, crop_size = smart_crop_get_coords(original_image, mask, base_size)
    
    # 如果裁剪区域被缩放过，需要将 512x512 的结果缩放回原始裁剪尺寸
    if crop_size != base_size:
        # 生成结果需要放大回原始裁剪尺寸
        generated_resized = cv2.resize(generated_512, (crop_size, crop_size), interpolation=cv2.INTER_LANCZOS4)
    else:
        generated_resized = generated_512
    
    # 获取实际的裁剪区域尺寸（可能因为边界而不是完整的 crop_size）
    actual_h = y2 - y1
    actual_w = x2 - x1
    
    # 如果实际裁剪区域小于 crop_size，需要裁剪生成结果
    if actual_h != crop_size or actual_w != crop_size:
        generated_resized = generated_resized[:actual_h, :actual_w]
    
    # 创建融合 mask（边缘羽化）
    if blend_margin > 0:
        blend_mask = np.ones((actual_h, actual_w), dtype=np.float32)
        
        # 对边缘进行羽化
        for i in range(blend_margin):
            alpha = i / blend_margin
            # 上边缘
            if i < actual_h:
                blend_mask[i, :] = alpha
            # 下边缘
            if actual_h - i - 1 >= 0:
                blend_mask[actual_h - i - 1, :] = np.minimum(blend_mask[actual_h - i - 1, :], alpha)
            # 左边缘
            if i < actual_w:
                blend_mask[:, i] = np.minimum(blend_mask[:, i], alpha)
            # 右边缘
            if actual_w - i - 1 >= 0:
                blend_mask[:, actual_w - i - 1] = np.minimum(blend_mask[:, actual_w - i - 1], alpha)
        
        blend_mask = blend_mask[:, :, np.newaxis]  # (H, W, 1)
        
        # 融合粘贴
        result_image[y1:y2, x1:x2] = (
            generated_resized * blend_mask + 
            result_image[y1:y2, x1:x2] * (1 - blend_mask)
        ).astype(np.uint8)
    else:
        # 直接粘贴（无融合）
        result_image[y1:y2, x1:x2] = generated_resized
    
    return result_image, (x1, y1, x2, y2)


def process_inference_results(inference_output_dir, data_dir, object_class, defect_type, 
                               output_dir, blend_margin=10, dilate_mask=False, mask_kernel_size=3):
    """
    批量处理推理结果，将 512x512 的图片粘贴回原始大图
    
    Args:
        inference_output_dir: 推理输出目录（包含 *_generated.png, *_original.png, *_mask.png）
        data_dir: 数据集根目录（用于找到原始大图）
        object_class: 物体类别
        defect_type: 缺陷类型
        output_dir: 输出目录
        blend_margin: 边缘融合宽度
        dilate_mask: 是否膨胀 mask（需要与推理时一致）
        mask_kernel_size: 膨胀核大小
    """
    import json
    
    # 推理结果目录
    inference_defect_dir = os.path.join(inference_output_dir, defect_type)
    
    if not os.path.exists(inference_defect_dir):
        print(f"Error: Inference directory not found: {inference_defect_dir}")
        return
    
    # 原始图片目录
    good_dir = os.path.join(data_dir, object_class, "test", "good")
    mask_dir = os.path.join(data_dir, object_class, "train", "defective_masks", defect_type)
    
    if not os.path.exists(good_dir):
        print(f"Error: Good images directory not found: {good_dir}")
        return
    
    if not os.path.exists(mask_dir):
        print(f"Error: Mask directory not found: {mask_dir}")
        return
    
    # 输出目录
    output_defect_dir = os.path.join(output_dir, defect_type)
    os.makedirs(output_defect_dir, exist_ok=True)
    
    # 读取 inference_log.json 获取映射关系（含 1:1 配对时的 input_mask）
    log_path = os.path.join(inference_output_dir, "inference_log.json")
    image_mask_mapping = {}
    mask_path_by_idx = {}
    
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            inference_log = json.load(f)
        results = inference_log.get("results", [])
        
        for result in results:
            idx = result.get("output_idx")
            img_path = result.get("input_image")
            m_path = result.get("input_mask")
            if idx is not None and img_path:
                image_mask_mapping[idx] = img_path
            if idx is not None and m_path:
                mask_path_by_idx[idx] = m_path
        
        print(f"Loaded {len(image_mask_mapping)} mappings from inference_log.json")
    else:
        print(f"Warning: inference_log.json not found at {log_path}")
        print("Will attempt to infer mappings from filenames...")
    
    # 获取所有生成的图片
    generated_files = sorted([f for f in os.listdir(inference_defect_dir) if f.endswith('_generated.png')])
    
    # 获取 mask 文件列表（用于推断）
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
    
    print(f"\nFound {len(generated_files)} generated images")
    print(f"Found {len(mask_files)} mask files")
    print(f"Processing with blend_margin={blend_margin}px\n")
    
    success_count = 0
    
    for gen_file in tqdm(generated_files, desc="Pasting back to original"):
        # 解析文件名获取索引
        idx_str = gen_file.split('_')[0]
        try:
            idx = int(idx_str)
        except ValueError:
            print(f"Warning: Cannot parse index from {gen_file}")
            continue
        
        # 读取生成的 512x512 图片
        gen_path = os.path.join(inference_defect_dir, gen_file)
        generated_512 = cv2.imread(gen_path)
        if generated_512 is None:
            print(f"Warning: Cannot read {gen_path}")
            continue
        generated_512 = cv2.cvtColor(generated_512, cv2.COLOR_BGR2RGB)
        
        # 获取原始图片路径（log 中可能是相对路径，与当前工作目录不一致时找不到）
        if idx in image_mask_mapping:
            original_image_path = image_mask_mapping[idx]
        else:
            print(f"Warning: Cannot find mapping for index {idx}, skipping...")
            continue
        if not os.path.exists(original_image_path):
            # 用 data_dir + object_class/test/good + 文件名 重试（解决推理与 paste_back 工作目录不同）
            fallback = os.path.join(os.path.abspath(data_dir), object_class, "test", "good", os.path.basename(original_image_path))
            if os.path.exists(fallback):
                original_image_path = fallback
        if not os.path.exists(original_image_path):
            print(f"Warning: Original image not found: {original_image_path}")
            continue
            
        original_image = cv2.imread(original_image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # 优先使用 log 中记录的 input_mask（1:1 配对推理时写入），否则按索引从 mask_dir 取
        if idx in mask_path_by_idx:
            mask_path = mask_path_by_idx[idx]
            if not os.path.exists(mask_path):
                for sub in ("test", "masks"), ("train", "defective_masks", defect_type):
                    fallback_m = os.path.join(os.path.abspath(data_dir), object_class, *sub, os.path.basename(mask_path))
                    if os.path.exists(fallback_m):
                        mask_path = fallback_m
                        break
        else:
            mask_idx = idx % len(mask_files)
            mask_path = os.path.join(mask_dir, mask_files[mask_idx])
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Cannot read mask: {mask_path}")
            continue
        
        # 应用膨胀（如果启用）
        if dilate_mask:
            k_size = mask_kernel_size if mask_kernel_size % 2 == 1 else mask_kernel_size + 1
            kernel = np.ones((k_size, k_size), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        # 粘贴回原图
        result_image, coords = paste_back_to_original(original_image, generated_512, mask, 
                                                       base_size=512, blend_margin=blend_margin)
        
        # 保存结果
        base_name = os.path.splitext(os.path.basename(original_image_path))[0]
        output_path = os.path.join(output_defect_dir, f"{idx:04d}_{base_name}_pasted.png")
        
        result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_bgr)
        
        success_count += 1
        
        # 创建对比图
        if args.create_comparison:
            comparison_path = os.path.join(output_defect_dir, f"{idx:04d}_{base_name}_comparison.png")
            original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
            comparison = np.hstack([original_bgr, result_bgr])
            h = comparison.shape[0]
            cv2.line(comparison, (original_bgr.shape[1], 0), (original_bgr.shape[1], h), (0, 255, 0), 3)
            cv2.imwrite(comparison_path, comparison)
    
    print(f"\n{'='*60}")
    print(f"Successfully processed {success_count}/{len(generated_files)} images")
    print(f"Results saved to: {output_defect_dir}")
    print(f"{'='*60}")


def paste_single_image(original_path, mask_path, generated_512_path, output_path, 
                       blend_margin=10, dilate_mask=False, mask_kernel_size=3):
    """
    处理单张图片：将 512x512 的生成结果粘贴回原始大图
    
    Args:
        original_path: 原始大图路径
        mask_path: 对应的 mask 路径
        generated_512_path: 生成的 512x512 图片路径
        output_path: 输出路径
        blend_margin: 边缘融合宽度（像素）
        dilate_mask: 是否膨胀 mask（需要与推理时一致）
        mask_kernel_size: 膨胀核大小
    """
    # 读取原始大图
    original_image = cv2.imread(original_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # 读取 mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # 应用膨胀（如果启用，需要与推理时一致）
    if dilate_mask:
        k_size = mask_kernel_size if mask_kernel_size % 2 == 1 else mask_kernel_size + 1
        kernel = np.ones((k_size, k_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    # 读取生成的 512x512 图片
    generated_512 = cv2.imread(generated_512_path)
    generated_512 = cv2.cvtColor(generated_512, cv2.COLOR_BGR2RGB)
    
    # 粘贴回原图
    result_image, coords = paste_back_to_original(original_image, generated_512, mask, 
                                                   base_size=512, blend_margin=blend_margin)
    
    # 保存结果
    result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)
    
    print(f"Saved: {output_path}")
    print(f"  Crop region: ({coords[0]}, {coords[1]}) to ({coords[2]}, {coords[3]})")
    
    return result_image, coords


def create_comparison_image(original_path, result_path, output_path):
    """
    创建对比图：原图 | 修复后
    """
    original = cv2.imread(original_path)
    result = cv2.imread(result_path)
    
    # 确保两张图片尺寸相同
    if original.shape != result.shape:
        print(f"Warning: Image size mismatch. Original: {original.shape}, Result: {result.shape}")
        return
    
    # 水平拼接
    comparison = np.hstack([original, result])
    
    # 添加分隔线
    h = comparison.shape[0]
    cv2.line(comparison, (original.shape[1], 0), (original.shape[1], h), (0, 255, 0), 3)
    
    cv2.imwrite(output_path, comparison)
    print(f"Comparison saved: {output_path}")

'''
python paste_back.py \
  --inference_dir ../infer_results/chengdu_huashang_05_26 \
  --data_dir ../DATA/my_infer \
  --object_class phone \
  --defect_type huashang \
  --output_dir ../infer_results/huashang_pasted_05_26 \
  --blend_margin 10 \
  --create_comparison

--inference_dir: 推理输出目录（包含 defect_type 子文件夹）
--data_dir: 数据集根目录
--object_class: 物体类别（如 'phone'）
--defect_type: 缺陷类型（如 'yashang'）
--output_dir: 输出目录

--blend_margin: 边缘融合宽度（像素），默认 10
--dilate_mask: 是否膨胀 mask（必须与推理时设置一致）
--mask_kernel_size: 膨胀核大小，默认 3
--create_comparison: 创建对比图（原图 | 修复后）
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paste 512x512 inference results back to original images")
    
    # 模式 1: 批量处理推理输出目录
    parser.add_argument("--inference_dir", type=str, help="Inference output directory (contains defect_type subfolders)")
    parser.add_argument("--data_dir", type=str, help="Dataset root directory")
    parser.add_argument("--object_class", type=str, help="Object class (e.g., 'phone')")
    parser.add_argument("--defect_type", type=str, help="Defect type (e.g., 'yashang')")
    
    # 模式 2: 处理单张图片
    parser.add_argument("--original", type=str, help="Original image path")
    parser.add_argument("--mask", type=str, help="Mask path")
    parser.add_argument("--generated", type=str, help="Generated 512x512 image path")
    
    # 通用参数
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--blend_margin", type=int, default=10, help="Edge blending margin in pixels (0 for no blending)")
    parser.add_argument("--dilate_mask", action="store_true", help="Dilate mask (must match inference settings)")
    parser.add_argument("--mask_kernel_size", type=int, default=3, help="Dilation kernel size")
    parser.add_argument("--create_comparison", action="store_true", help="Create side-by-side comparison images")
    
    args = parser.parse_args()
    
    # 模式 1: 批量处理
    if args.inference_dir and args.data_dir and args.object_class and args.defect_type:
        print(f"\n{'='*60}")
        print(f"Batch Mode: Processing inference results")
        print(f"{'='*60}\n")
        
        process_inference_results(
            inference_output_dir=args.inference_dir,
            data_dir=args.data_dir,
            object_class=args.object_class,
            defect_type=args.defect_type,
            output_dir=args.output_dir,
            blend_margin=args.blend_margin,
            dilate_mask=args.dilate_mask,
            mask_kernel_size=args.mask_kernel_size
        )
        
    # 模式 2: 单张图片处理
    elif args.original and args.mask and args.generated:
        print(f"\n{'='*60}")
        print(f"Single Image Mode")
        print(f"{'='*60}\n")
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(args.original))[0]
        output_path = os.path.join(args.output_dir, f"{base_name}_pasted.png")
        
        result_image, coords = paste_single_image(
            original_path=args.original,
            mask_path=args.mask,
            generated_512_path=args.generated,
            output_path=output_path,
            blend_margin=args.blend_margin,
            dilate_mask=args.dilate_mask,
            mask_kernel_size=args.mask_kernel_size
        )
        
        # 创建对比图
        if args.create_comparison:
            comparison_path = os.path.join(args.output_dir, f"{base_name}_comparison.png")
            create_comparison_image(args.original, output_path, comparison_path)
    
    else:
        print("Error: Please provide either:")
        print("  1. Batch mode: --inference_dir, --data_dir, --object_class, --defect_type")
        print("  2. Single mode: --original, --mask, --generated")
        parser.print_help()
