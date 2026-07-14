"""
根据labelme标签生成亮点效果
中心亮，边缘暗，随机性
"""

import json
import numpy as np
import cv2
from PIL import Image
import random


def load_labelme_polygons(json_path):
    """加载labelme JSON中的多边形坐标"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    polygons = []
    for shape in data.get('shapes', []):
        if shape['shape_type'] == 'polygon':
            points = np.array(shape['points'], dtype=np.int32)
            polygons.append(points)
    return polygons


def create_small_glow(mask, center, radius=20, intensity=1.0):
    """
    在mask区域内生成小亮斑
    中心亮，边缘暗，高斯衰减
    """
    h, w = mask.shape[:2]
    y, x = np.ogrid[:h, :w]

    # 计算到中心距离
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    # 高斯衰减
    sigma = radius / 2
    glow = intensity * np.exp(-dist**2 / (2 * sigma**2))

    # 应用到mask区域
    glow = glow * mask.astype(float)

    return glow


def process_image(json_path, output_dir='liangdian'):
    """处理单张图片"""
    import os
    # 读取图片（彩色模式用于叠加）
    base_path = json_path.replace('.json', '')
    image = cv2.imread(base_path + '.png', cv2.IMREAD_COLOR)
    if image is None:
        print(f"无法读取图片: {base_path}.png")
        return

    h, w = image.shape[:2]

    # 创建全黑输出（与原图同尺寸彩色）
    output = image.copy()

    # 加载多边形
    polygons = load_labelme_polygons(json_path)

    # 随机颜色/强度 - 最大亮度
    base_intensity = 1.0

    for poly in polygons:
        # 创建多边形mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)

        # 使用多边形中心作为亮点中心
        M = cv2.moments(poly)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            x, y, bw, bh = cv2.boundingRect(poly)
            cx = x + bw // 2
            cy = y + bh // 2

        # 亮斑半径与多边形边界框尺寸相当
        x, y, bw, bh = cv2.boundingRect(poly)
        radius = max(bw, bh) // 2
        radius = max(radius, 3)  # 最小半径

        # 微调中心位置，增加随机性
        cx += random.randint(-radius//3, radius//3)
        cy += random.randint(-radius//3, radius//3)

        # 生成亮斑（灰度）
        glow = create_small_glow(mask, (cx, cy), radius=radius, intensity=base_intensity)

        # 转换为彩色（白光）并叠加到输出
        glow_color = np.stack([glow, glow, glow], axis=-1)
        output = np.clip(output + glow_color, 0, 255).astype(np.uint8)

    # 保存结果到指定目录
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(base_path) + '_glow.png'
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, output)
    print(f"已保存: {output_path}")


if __name__ == '__main__':
    import glob
    # 处理目录下所有JSON文件
    json_files = sorted(glob.glob('*.json'))
    print(f"找到 {len(json_files)} 个JSON文件")
    for json_file in json_files:
        process_image(json_file)