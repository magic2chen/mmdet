from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from mmdet.registry import DATASETS

# mean / std for torchvision style normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_default_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def build_penalty_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((2 * image_size, 2 * image_size)),
        transforms.RandomGrayscale(0.3),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


@DATASETS.register_module()
class EfficientADDataset(Dataset):
    """Dataset wrapper used for EfficientAD training and evaluation.

    Args:
        root (str): Root directory of the Data (e.g.
            ``mvtec_anomaly_detection``).
        dataset_type (str): Either ``mvtec_ad`` or ``mvtec_loco``.
        subdataset (str): Sub-Data name (e.g. ``bottle``).
        split (str): One of ``train``, ``val`` or ``test``.
        image_size (int): Target image resolution.
        train_split_ratio (float): Ratio of training samples for MVTec-AD
            (validation receives the remainder).
        imagenet_train_path (str): Optional path to ImageNet train images. If
            provided for the train split, penalty regularisation is enabled.
        seed (int): Random seed used for splitting the training set.
    """

    def __init__(self,
                 root: str,
                 dataset_type: str = 'mvtec_ad',
                 subdataset: str = 'bottle',
                 split: str = 'train',
                 image_size: int = 256,
                 train_split_ratio: float = 0.9,
                 imagenet_train_path: str = 'none',
                 seed: int = 42) -> None:
        assert dataset_type in {'mvtec_ad', 'mvtec_loco'}
        assert split in {'train', 'val', 'test'}
        self.dataset_type = dataset_type
        self.subdataset = subdataset
        self.split = split
        self.image_size = image_size
        self.train_split_ratio = train_split_ratio
        self.seed = seed

        self.default_transform = build_default_transform(image_size)
        self.augment_transform = transforms.RandomChoice([
            transforms.ColorJitter(brightness=0.2),
            transforms.ColorJitter(contrast=0.2),
            transforms.ColorJitter(saturation=0.2),
        ])

        self.data_list: List[Tuple[str, int]] = []
        dataset_root = os.path.join(root, subdataset)
        if not os.path.isdir(dataset_root):
            raise FileNotFoundError(f'Dataset path {dataset_root} not found.')

        self._prepare_split(dataset_root)

        # optional ImageNet Data for penalty regularisation
        self.penalty_enabled = (
            split == 'train' and imagenet_train_path != 'none'
            and os.path.isdir(imagenet_train_path))
        self.penalty_files: Optional[List[str]] = None
        if self.penalty_enabled:
            self.penalty_transform = build_penalty_transform(image_size)
            self.penalty_files = self._gather_image_files(imagenet_train_path)
            if not self.penalty_files:
                self.penalty_enabled = False

    def _gather_image_files(self, folder: str) -> List[str]:
        image_files: List[str] = []
        for root, _, files in os.walk(folder):
            for name in files:
                if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_files.append(os.path.join(root, name))
        return sorted(image_files)

    def _prepare_split(self, dataset_root: str) -> None:
        if self.split == 'test':
            subset_dir = os.path.join(dataset_root, 'test')
            self.data_list = self._collect_labelled_files(subset_dir)
            return

        if self.dataset_type == 'mvtec_ad':
            train_dir = os.path.join(dataset_root, 'train')
            files = self._collect_labelled_files(train_dir, labelled=False)
            rng = np.random.RandomState(self.seed)
            rng.shuffle(files)
            split_idx = int(len(files) * self.train_split_ratio)
            if self.split == 'train':
                selected = files[:split_idx]
            else:
                selected = files[split_idx:]
            # label 0 for normal samples
            self.data_list = [(path, 0) for path in selected]
        else:  # mvtec_loco provides explicit validation set
            subset_name = 'train' if self.split == 'train' else 'validation'
            subset_dir = os.path.join(dataset_root, subset_name)
            files = self._collect_labelled_files(subset_dir, labelled=False)
            self.data_list = [(path, 0) for path in files]

    def _collect_labelled_files(
            self,
            folder: str,
            labelled: bool = True) -> List:
        """Collect image files inside ``folder``.

        Args:
            folder (str): Directory to scan.
            labelled (bool): If ``True`` return ``(path, label)`` tuples,
                otherwise return a list of image paths.
        """
        results: List = []
        if not os.path.isdir(folder):
            return results

        for root, dirs, files in os.walk(folder):
            for file in files:
                if not file.lower().endswith(
                        ('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                path = os.path.join(root, file)
                if labelled:
                    defect_class = os.path.basename(os.path.dirname(path))
                    label = 0 if defect_class == 'good' else 1
                else:
                    results.append(path)
                    continue
                results.append((path, label))
        results.sort()
        return results

    def __len__(self) -> int:
        return len(self.data_list)

    def _load_image(self, path: str) -> Image.Image:
        with Image.open(path) as img:
            return img.convert('RGB')

    def _sample_penalty(self) -> Optional[torch.Tensor]:
        if not self.penalty_enabled or not self.penalty_files:
            return None
        penalty_path = random.choice(self.penalty_files)
        penalty_img = self._load_image(penalty_path)
        return self.penalty_transform(penalty_img)

    def __getitem__(self, index: int) -> Dict:
        path, label = self.data_list[index]
        image = self._load_image(path)

        if self.split == 'train':
            img_student = self.default_transform(image)
            augmented = self.augment_transform(image)
            img_autoencoder = self.default_transform(augmented)
            penalty_img = self._sample_penalty()
            if penalty_img is None:
                penalty_img = torch.zeros_like(img_student)
            return dict(
                img_student=img_student,
                img_autoencoder=img_autoencoder,
                img_penalty=penalty_img,
                use_penalty=self.penalty_enabled,
            )

        img_tensor = self.default_transform(image)
        orig_size = (image.height, image.width)

        return dict(
            img=img_tensor,
            label=torch.tensor(label, dtype=torch.int64),
            path=path,
            orig_size=orig_size,
        )
