# Copyright (c) OpenMMLab. All rights reserved.
"""MVTec AD dataset for DefectFill training and evaluation."""

import os
import os.path as osp
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from mmdet.registry import DATASETS


@DATASETS.register_module()
class MVTecDefectDataset(Dataset):
    """MVTec AD dataset for DefectFill training and evaluation.

    This dataset returns image-mask pairs for inpainting-based defect detection.
    It supports train/val/test splits and can filter by specific defect types.

    Args:
        root (str): Root directory of MVTec AD dataset.
        object_class (str): Object class name (e.g., 'bottle', 'phone').
        split (str): 'train', 'val', or 'test'.
        image_size (int): Target image resolution (default 512).
        defect_type (str, optional): Legacy single-type filter. Use
            ``defect_types`` for new code. Wrapped internally as
            ``defect_types=[defect_type]`` if provided.
        defect_types (list[str], optional): Subset of defect-type directory
            names to include (e.g. ``['yashang', 'myashang1']``). Takes
            precedence over ``defect_type``. ``None`` means "all types".
        dilate_mask (bool): Whether to dilate masks.
        mask_kernel_size (int): Kernel size for mask dilation (must be odd).
    """

    def __init__(
        self,
        root: str,
        object_class: str,
        split: str = 'train',
        image_size: int = 512,
        defect_type: Optional[str] = None,
        defect_types: Optional[List[str]] = None,
        dilate_mask: bool = False,
        mask_kernel_size: int = 3,
        limit: Optional[int] = None,
        **kwargs
    ):
        self.root = osp.abspath(osp.expanduser(root))
        self.object_class = object_class
        self.split = split
        self.image_size = image_size
        # Resolve to a single canonical list:
        #   defect_types is set        → use it as-is
        #   defect_type (singular) set → wrap as [defect_type]
        #   neither set                → None (no filter, all types)
        if defect_types is not None:
            self.target_defect_types: Optional[List[str]] = list(defect_types)
        elif defect_type is not None:
            self.target_defect_types = [defect_type]
        else:
            self.target_defect_types = None
        self.dilate_mask = dilate_mask
        self.mask_kernel_size = mask_kernel_size
        self.limit = limit

        # Load data list
        self.data_list = self.load_data_list()
        if self.limit is not None and self.limit > 0:
            self.data_list = self.data_list[:self.limit]

    def __len__(self):
        return len(self.data_list)

    def load_data_list(self) -> List[Dict]:
        """Load image and mask paths from MVTec AD directory structure.

        MVTec AD structure:
        - root/object_class/train/defective/{defect_type}/*.png
        - root/object_class/train/defective_masks/{defect_type}/*.png
        - root/object_class/test/good/*.png
        - root/object_class/val/defective/{defect_type}/*.png
        - root/object_class/val/defective_masks/{defect_type}/*.png
        - root/object_class/val/good/*.png

        For split='val': load BOTH defective/ (positive) and good/ (negative)
        from root/object_class/val/, so ROC-AUC can be computed.

        Returns:
            List of data info dicts with img_path, mask_path, is_defect, etc.
        """
        data_list = []

        if self.split == 'test':
            # Test split: only 'good' images (no defects)
            good_dir = self.root_path(self.object_class, 'test', 'good')
            if self._check_exists(good_dir):
                for fname in sorted(self._list_dir(good_dir)):
                    if self._is_image_file(fname):
                        data_list.append({
                            'img_path': self._join_path(good_dir, fname),
                            'mask_path': None,
                            'is_defect': False,
                            'object_class': self.object_class,
                            'split': 'test',
                        })
        elif self.split == 'val':
            # Val split: load both defective/ (positive) and good/ (negative)
            # so ROC-AUC has both label classes.
            val_root = self.root_path(self.object_class, 'val')

            # 1) Positive samples: defective/<defect_type>/*.png
            defective_dir = self._join_path(val_root, 'defective')
            if self._check_exists(defective_dir):
                defect_types_val = sorted(self._list_dir(defective_dir))
                if self.target_defect_types is not None:
                    defect_types_val = [
                        t for t in defect_types_val
                        if t in self.target_defect_types
                    ]
                for dtype in defect_types_val:
                    img_dir = self._join_path(defective_dir, dtype)
                    if not self._check_exists(img_dir):
                        continue
                    mask_dir = self._join_path(
                        val_root, 'defective_masks', dtype)

                    for fname in sorted(self._list_dir(img_dir)):
                        if not self._is_image_file(fname):
                            continue
                        base_name = self._get_base_name(fname)
                        mask_path = self._join_path(
                            mask_dir, f'{base_name}_mask.png')
                        if not self._check_exists(mask_path):
                            # Fallback: search any mask containing base_name
                            if self._check_exists(mask_dir):
                                candidates = [
                                    f for f in self._list_dir(mask_dir)
                                    if base_name in f and self._is_image_file(f)
                                ]
                                if candidates:
                                    mask_path = self._join_path(
                                        mask_dir, candidates[0])
                                else:
                                    continue
                            else:
                                continue
                        data_list.append({
                            'img_path': self._join_path(img_dir, fname),
                            'mask_path': mask_path,
                            'is_defect': True,
                            'object_class': self.object_class,
                            'defect_type': dtype,
                            'split': 'val',
                        })

            # 2) Negative samples: good/*.png
            good_dir = self._join_path(val_root, 'good')
            if self._check_exists(good_dir):
                for fname in sorted(self._list_dir(good_dir)):
                    if self._is_image_file(fname):
                        data_list.append({
                            'img_path': self._join_path(good_dir, fname),
                            'mask_path': None,
                            'is_defect': False,
                            'object_class': self.object_class,
                            'split': 'val',
                        })
        else:
            # Train/val split: defective images with masks
            defect_path = self.root_path(self.object_class, 'train', 'defective')

            if self._check_exists(defect_path):
                defect_types = [
                    d for d in self._list_dir(defect_path)
                    if self._check_exists(self._join_path(defect_path, d))
                ]

                if self.target_defect_types is not None:
                    defect_types = [
                        t for t in defect_types if t in self.target_defect_types
                    ]

                for dtype in defect_types:
                    img_dir = self._join_path(defect_path, dtype)
                    mask_dir = self.root_path(self.object_class, 'train', 'defective_masks', dtype)

                    if not (self._check_exists(img_dir) and
                            self._check_exists(mask_dir)):
                        continue

                    img_files = [f for f in self._list_dir(img_dir) if self._is_image_file(f)]

                    for fname in sorted(img_files):
                        base_name = self._get_base_name(fname)
                        mask_file = f"{base_name}_mask.png"
                        mask_path = self._join_path(mask_dir, mask_file)

                        if not self._check_exists(mask_path):
                            # Try to find mask with different naming pattern
                            possible_masks = [
                                f for f in self._list_dir(mask_dir)
                                if base_name in f
                            ]
                            if possible_masks:
                                mask_path = self._join_path(
                                    mask_dir, possible_masks[0])
                            else:
                                continue

                        data_list.append({
                            'img_path': self._join_path(img_dir, fname),
                            'mask_path': mask_path,
                            'is_defect': True,
                            'object_class': self.object_class,
                            'defect_type': dtype,
                            'split': self.split,
                        })

        return data_list

    def root_path(self, *parts) -> str:
        """Build path relative to dataset root."""
        return self._join_path(self.root, *parts)

    def _join_path(self, *parts) -> str:
        return '/'.join(parts)

    def _with_sep(self, path: str) -> str:
        """Ensure path uses forward slash for consistency."""
        return path.replace('\\', '/')

    def _check_exists(self, path: str) -> bool:
        """Check if path exists."""
        return os.path.exists(path)

    def _list_dir(self, path: str) -> List[str]:
        """List directory contents."""
        if not os.path.exists(path):
            return []
        return os.listdir(path)

    def _is_image_file(self, fname: str) -> bool:
        """Check if file is an image."""
        return fname.lower().endswith(('.png', '.jpg', '.jpeg'))

    def _get_base_name(self, fname: str) -> str:
        """Get base name without extension."""
        return os.path.splitext(fname)[0]

    def _generate_random_mask(self, h: int, w: int) -> np.ndarray:
        """Generate random rectangular masks for object loss training.

        Generates 30 random rectangles with size between 3% and 25% of image.
        This replicates the training logic for the object branch.

        Args:
            h: Image height
            w: Image width

        Returns:
            Mask array [H, W] in [0, 1]
        """
        mask = np.zeros((h, w), dtype=np.float32)

        for _ in range(30):
            min_size = int(min(h, w) * 0.03)
            max_size = int(min(h, w) * 0.25)

            rect_h = np.random.randint(min_size, max(1, max_size))
            rect_w = np.random.randint(min_size, max(1, max_size))
            y = np.random.randint(0, max(1, h - rect_h))
            x = np.random.randint(0, max(1, w - rect_w))

            mask[y:y+rect_h, x:x+rect_w] = 1.0

        return mask

    def __getitem__(self, index: int) -> Dict:
        """Get training/test sample.

        Returns:
            Dict with keys:
            - img: image tensor [3, H, W] normalized to [-1, 1]
            - mask: defect mask [1, H, W] in [0, 1]
            - background: I * (1-M) [3, H, W]
            - adjusted_mask: mask + alpha*(1-mask) for object loss
            - is_defect: bool tensor
            - object_class: str
        """
        item = self.data_list[index]

        # Load image
        img_path = item['img_path']
        if not os.path.exists(img_path):
            raise IOError(f"Image file does not exist: {img_path}")
        image = cv2.imread(img_path)
        if image is None:
            raise IOError(f"Failed to load image: {item['img_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Smart resizing: always resize to target size to ensure consistency
        h, w = image.shape[:2]
        if h != self.image_size or w != self.image_size:
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_CUBIC
            )

        # Load or generate mask
        if item['mask_path'] is not None:
            mask = cv2.imread(item['mask_path'], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise IOError(f"Failed to load mask: {item['mask_path']}")

            # Resize mask to match image size immediately after loading
            if mask.shape[:2] != (image.shape[0], image.shape[1]):
                mask = cv2.resize(
                    mask,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            # Dilate mask if enabled
            if self.dilate_mask:
                k_size = self.mask_kernel_size
                if k_size % 2 == 0:
                    k_size += 1
                kernel = np.ones((k_size, k_size), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)

            mask = mask.astype(np.float32) / 255.0
        else:
            mask = self._generate_random_mask(
                image.shape[0], image.shape[1])

        # Normalize to [-1, 1] float32
        image = image.astype(np.float32) / 127.5 - 1.0

        # Convert to tensor [C, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask).unsqueeze(0).contiguous()

        # Create background (masked image) - already normalized
        background = image * (1 - mask)

        # Adjusted mask for object loss (M' = M + alpha*(1-M))
        alpha = 0.3
        adjusted_mask = mask + alpha * (1 - mask)

        return {
            'img': image,
            'mask': mask,
            'background': background,
            'adjusted_mask': adjusted_mask,
            'is_defect': torch.tensor(item['is_defect'], dtype=torch.bool),
            'object_class': item['object_class'],
            'img_path': item['img_path'],         # 新增：让 test_step / generate 能落地文件名
            'defect_type': item.get('defect_type'),  # 新增：用于 inference_log.json 写 defect_type
        }