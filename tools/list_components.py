#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""快速查看 MMDetection 中所有已注册的组件。

使用方法:
    python tools/list_components.py [--type transforms|datasets|all]
"""

import argparse
from mmdet.utils import register_all_modules
from mmdet.registry import TRANSFORMS, DATASETS, MODELS, METRICS, HOOKS, LOOPS


def list_transforms():
    """列出所有已注册的 Transform"""
    print("=" * 80)
    print("已注册的 Transform 组件:")
    print("=" * 80)
    
    transforms = sorted(TRANSFORMS._module_dict.keys())
    
    # 按类别分组
    categories = {
        '加载类': ['Load'],
        '几何变换': ['Resize', 'Crop', 'Flip', 'Rotate', 'Shear', 'Translate', 'Affine', 'Shift'],
        '颜色变换': ['Color', 'Brightness', 'Contrast', 'Equalize', 'Solarize', 'Posterize', 'Invert', 'AutoContrast'],
        '数据增强': ['Mosaic', 'MixUp', 'CopyPaste', 'CutOut', 'RandomErasing', 'InstaBoost'],
        '格式化': ['Pack', 'ToTensor', 'ImageToTensor', 'Transpose'],
        '其他': []
    }
    
    categorized = {cat: [] for cat in categories.keys()}
    categorized['其他'] = []
    
    for name in transforms:
        categorized_flag = False
        for cat, keywords in categories.items():
            if cat == '其他':
                continue
            if any(keyword in name for keyword in keywords):
                categorized[cat].append(name)
                categorized_flag = True
                break
        if not categorized_flag:
            categorized['其他'].append(name)
    
    for cat, items in categorized.items():
        if items:
            print(f"\n【{cat}】")
            for item in items:
                print(f"  - {item}")
    
    print(f"\n总计: {len(transforms)} 个 Transform")


def list_datasets():
    """列出所有已注册的 Dataset"""
    print("=" * 80)
    print("已注册的 Dataset 组件:")
    print("=" * 80)
    
    datasets = sorted(DATASETS._module_dict.keys())
    
    for name in datasets:
        print(f"  - {name}")
    
    print(f"\n总计: {len(datasets)} 个 Dataset")


def list_all():
    """列出所有已注册的组件"""
    print("=" * 80)
    print("MMDetection 已注册组件总览")
    print("=" * 80)
    
    print(f"\n【Transform】: {len(TRANSFORMS._module_dict)} 个")
    print(f"【Dataset】: {len(DATASETS._module_dict)} 个")
    print(f"【Model】: {len(MODELS._module_dict)} 个")
    print(f"【Metric】: {len(METRICS._module_dict)} 个")
    print(f"【Hook】: {len(HOOKS._module_dict)} 个")
    print(f"【Loop】: {len(LOOPS._module_dict)} 个")
    
    print("\n" + "=" * 80)
    print("查看详细信息:")
    print("  python tools/list_components.py --type transforms  # 查看所有 Transform")
    print("  python tools/list_components.py --type datasets    # 查看所有 Dataset")


def search_component(keyword, component_type='all'):
    """搜索包含关键词的组件"""
    print(f"=" * 80)
    print(f"搜索关键词: '{keyword}'")
    print("=" * 80)
    
    if component_type in ['transforms', 'all']:
        transforms = [name for name in TRANSFORMS._module_dict.keys() 
                     if keyword.lower() in name.lower()]
        if transforms:
            print(f"\n【Transform】找到 {len(transforms)} 个:")
            for name in sorted(transforms):
                print(f"  - {name}")
    
    if component_type in ['datasets', 'all']:
        datasets = [name for name in DATASETS._module_dict.keys() 
                   if keyword.lower() in name.lower()]
        if datasets:
            print(f"\n【Dataset】找到 {len(datasets)} 个:")
            for name in sorted(datasets):
                print(f"  - {name}")


def main():
    parser = argparse.ArgumentParser(description='查看 MMDetection 已注册组件')
    parser.add_argument(
        '--type',
        type=str,
        choices=['transforms', 'datasets', 'all'],
        default='all',
        help='要查看的组件类型')
    parser.add_argument(
        '--search',
        type=str,
        default=None,
        help='搜索包含关键词的组件')
    
    args = parser.parse_args()
    
    # 注册所有模块
    register_all_modules()
    
    if args.search:
        search_component(args.search, args.type)
    elif args.type == 'transforms':
        list_transforms()
    elif args.type == 'datasets':
        list_datasets()
    else:
        list_all()


# # 查看所有组件概览
# python tools/list_components.py

# # 查看所有 Transform
# python tools/list_components.py --type transforms

# # 查看所有 Dataset
# python tools/list_components.py --type datasets

# # 搜索包含关键词的组件
# python tools/list_components.py --search "Resize"
# python tools/list_components.py --search "Load" --type transforms
if __name__ == '__main__':
    main()

