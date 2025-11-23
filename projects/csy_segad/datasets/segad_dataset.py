# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from mmdet.registry import DATASETS


@DATASETS.register_module()
class SegADDataset(Dataset):
    """Dataset for SegAD that loads segmentation maps and anomaly maps.

    Args:
        data_root (str): Root directory containing the data.
        segm_path (str): Path to segmentation maps directory.
        an_path (str): Path to anomaly maps directory.
        models_list (List[str]): List of base anomaly detection model names.
        category (str): Category name (e.g., 'candle', 'pcb1').
        split (str): One of 'train' or 'test'.
        csv_file (str): Path to CSV file containing data information.
        num_components (int): Number of components in segmentation map.
    """

    def __init__(self,
                 data_root: str = '',
                 segm_path: str = '',
                 an_path: str = '',
                 models_list: List[str] = None,
                 category: str = 'candle',
                 split: str = 'train',
                 csv_file: str = '',
                 num_components: int = 2) -> None:
        super().__init__()
        self.data_root = data_root
        self.segm_path = segm_path
        self.an_path = an_path
        self.models_list = models_list if models_list else ['efficient_ad']
        self.category = category
        self.split = split
        self.csv_file = csv_file
        self.num_components = num_components

        # Load data from CSV file
        # Note: For SegAD, the training loop loads CSV files directly,
        # so this Data is mainly used for structure. CSV file can be empty
        # if it will be set later by the training loop.
        if csv_file and os.path.exists(csv_file):
            self.df = pd.read_csv(csv_file, index_col=0).reset_index()
            self.data_list = self._prepare_data_list()
        else:
            # Create empty dataframe if CSV doesn't exist
            # The training loop will set df directly
            self.df = pd.DataFrame()
            self.data_list = []

    def _prepare_data_list(self) -> List[Dict]:
        """Prepare data list from dataframe."""
        data_list = []
        for idx, row in self.df.iterrows():
            data_list.append({
                'an_map_path': row.get('an_map_path', ''),
                'filepath': row.get('filepath', ''),
                'label': int(row.get('label', 0)),
                'prediction_an_det': float(row.get('prediction_an_det', 0.0)),
            })
        return data_list

    def __len__(self) -> int:
        return len(self.data_list)

    def _load_segmentation_map(self, an_map_path: str,
                               label: int) -> np.ndarray:
        """Load segmentation map for a given anomaly map path."""
        segm_filename = os.path.basename(an_map_path)
        label_dir = 'bad' if label else 'good'
        # Try path without category first:
        # {segm_path}/{label_dir}/{filename}
        segm_filepath = os.path.join(self.segm_path, label_dir,
                                     segm_filename)
        if not os.path.exists(segm_filepath):
            # Try path with category:
            # {segm_path}/{category}/{label_dir}/{filename}
            segm_filepath = os.path.join(self.segm_path, self.category,
                                         label_dir, segm_filename)
        if not os.path.exists(segm_filepath):
            # Try with .npy extension if not present
            if not segm_filepath.endswith('.npy'):
                segm_filepath = segm_filepath + '.npy'
            # Try both paths again with .npy extension
            if not os.path.exists(segm_filepath):
                segm_filepath = os.path.join(self.segm_path, label_dir,
                                             segm_filename)
            if not os.path.exists(segm_filepath):
                path1 = os.path.join(self.segm_path, label_dir,
                                     segm_filename)
                path2 = os.path.join(self.segm_path, self.category,
                                     label_dir, segm_filename)
                raise FileNotFoundError(
                    f'Segmentation map not found. Tried:\n'
                    f'  - {path1}\n  - {path2}')
        return np.load(segm_filepath)

    def _load_anomaly_map(self, model: str, an_map_path: str,
                          label: int) -> np.ndarray:
        """Load anomaly map for a given model and path."""
        an_filename = os.path.basename(an_map_path)
        label_dir = 'bad' if label else 'good'
        # Try multiple path structures:
        # 1. {an_path}/{category}/{label_dir}/{filename}
        #    (direct, actual location)
        an_filepath = os.path.join(
            self.an_path, self.category, label_dir, an_filename)
        if not os.path.exists(an_filepath):
            # 2. {an_path}/{category}/anomaly_maps/{label_dir}/{filename}
            an_filepath = os.path.join(
                self.an_path, self.category, 'anomaly_maps',
                label_dir, an_filename)
        if not os.path.exists(an_filepath):
            # 3. {an_path}/{model}/{category}/anomaly_maps/
            #    {label_dir}/{filename}
            an_filepath = os.path.join(
                self.an_path, model, self.category, 'anomaly_maps',
                label_dir, an_filename)
        if not os.path.exists(an_filepath):
            # Try with .npy extension if not present
            if not an_filepath.endswith('.npy'):
                an_filepath = an_filepath + '.npy'
            # Try all paths again with .npy extension
            if not os.path.exists(an_filepath):
                an_filepath = os.path.join(
                    self.an_path, self.category, label_dir, an_filename)
            if not os.path.exists(an_filepath):
                path1 = os.path.join(self.an_path, self.category,
                                     label_dir, an_filename)
                path2 = os.path.join(self.an_path, self.category,
                                     'anomaly_maps', label_dir, an_filename)
                path3 = os.path.join(self.an_path, model, self.category,
                                     'anomaly_maps', label_dir, an_filename)
                raise FileNotFoundError(
                    f'Anomaly map not found. Tried:\n'
                    f'  - {path1}\n  - {path2}\n  - {path3}')
        return np.load(an_filepath)

    def __getitem__(self, index: int) -> Dict:
        """Get item from Data."""
        item = self.data_list[index]
        label = item['label']
        an_map_path = item['an_map_path']

        # Load segmentation map
        segm_map = self._load_segmentation_map(an_map_path, label)

        # Load anomaly maps for all models
        anomaly_maps = {}
        for model in self.models_list:
            anomaly_maps[model] = self._load_anomaly_map(
                model, an_map_path, label)

        return {
            'segm_map': segm_map,
            'anomaly_maps': anomaly_maps,
            'label': label,
            'prediction_an_det': item['prediction_an_det'],
            'filepath': item['filepath'],
            'an_map_path': an_map_path,
            'category': self.category,
        }
