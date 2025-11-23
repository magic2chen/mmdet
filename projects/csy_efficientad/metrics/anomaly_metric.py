from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import tifffile
from sklearn.metrics import roc_auc_score

from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS


@METRICS.register_module()
class AnomalyMetric(BaseMetric):
    """Compute ROC-AUC based on anomaly maps produced by EfficientAD.

    Args:
        save_dir (str, optional): If provided, anomaly maps will be written to
            this directory following the Data structure.
        map_low_quantile (float): Starting quantile used for normalising the
            student-teacher distance maps.
        map_high_quantile (float): Ending quantile used for normalising.
    """

    def __init__(self,
                 save_dir: str = '',
                 map_low_quantile: float = 0.9,
                 map_high_quantile: float = 0.995,
                 data_root: Optional[str] = None,
                 save_csv: bool = True,
                 save_format: str = 'npy',
                 csv_filename: Optional[str] = None) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.map_low_quantile = map_low_quantile
        self.map_high_quantile = map_high_quantile
        self.data_root = data_root
        self.save_csv = save_csv
        self.save_format = save_format  # 'npy' or 'tiff'
        self.csv_filename = csv_filename

        self.maps_st: List[torch.Tensor] = []
        self.maps_ae: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.paths: List[str] = []
        self.orig_sizes: List[Sequence[int]] = []
        self.an_map_paths: List[Optional[str]] = []

    def process(self, data_batch: Dict, data_samples: Sequence[Dict]) -> None:
        for sample in data_samples:
            self.maps_st.append(sample['map_st'])
            self.maps_ae.append(sample['map_ae'])
            self.labels.append(sample['label'])
            self.paths.append(sample['path'])
            self.orig_sizes.append(sample['orig_size'])

    def compute_metrics(self, results: List) -> Dict[str, float]:
        raise NotImplementedError  # pragma: no cover

    def evaluate(self, size: int = 0) -> Dict[str, float]:
        if not self.maps_st:
            return dict(auc=0.0)

        maps_st = torch.stack(self.maps_st)
        maps_ae = torch.stack(self.maps_ae)
        labels = np.asarray(self.labels, dtype=np.int32)

        q_st_start = torch.quantile(maps_st, self.map_low_quantile).item()
        q_st_end = torch.quantile(maps_st, self.map_high_quantile).item()
        q_ae_start = torch.quantile(maps_ae, self.map_low_quantile).item()
        q_ae_end = torch.quantile(maps_ae, self.map_high_quantile).item()

        eps = 1e-6
        scores: List[float] = []
        an_map_paths: List[Optional[str]] = []
        filepaths: List[str] = []

        for idx in range(len(self.maps_st)):
            map_st = self.maps_st[idx]
            map_ae = self.maps_ae[idx]

            norm_st = 0.1 * (map_st - q_st_start) / max(
                q_st_end - q_st_start, eps)
            norm_ae = 0.1 * (map_ae - q_ae_start) / max(
                q_ae_end - q_ae_start, eps)
            combined = 0.5 * norm_st + 0.5 * norm_ae

            # clamp to non-negative values
            combined = torch.clamp(combined, min=0.0)

            an_path = None
            if self.save_dir:
                an_path = self._save_map(idx, combined.numpy())

            score = float(torch.max(combined))
            scores.append(score)
            an_map_paths.append(an_path)

            # Build filepath for CSV (matching test_dianziyan.py format)
            path = self.paths[idx]
            if self.data_root:
                # Extract defect_class and image name from path
                # path might be like: /path/to/dianziyan/test/good/image.jpg
                # or: /path/to/dianziyan/train/good/image.jpg
                defect_class = os.path.basename(os.path.dirname(path))
                img_name = os.path.basename(path)
                # Build path like:
                # {data_root}/{split}/{defect_class}/{img_name}
                # We need to extract split from the path
                path_parts = path.split(os.sep)
                if 'test' in path_parts:
                    split = 'test'
                elif 'train' in path_parts:
                    split = 'train'
                elif 'val' in path_parts:
                    split = 'val'
                else:
                    split = 'test'  # default
                filepath = os.path.join(self.data_root, split, defect_class,
                                        img_name)
            else:
                # Use original path
                filepath = path
            filepaths.append(filepath)

        y_scores = np.asarray(scores)
        if len(np.unique(labels)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels, y_scores))

        # Save CSV file if requested
        if self.save_csv and self.save_dir:
            self._save_csv(filepaths, an_map_paths, labels, y_scores)

        # clear buffers for next evaluation
        self.maps_st.clear()
        self.maps_ae.clear()
        self.labels.clear()
        self.paths.clear()
        self.orig_sizes.clear()

        return dict(auc=auc)

    def _save_map(self, idx: int, anomaly_map: np.ndarray) -> Optional[str]:
        """Save anomaly map and return the path.

        Returns:
            Optional[str]: Path to saved anomaly map, or None if not saved.
        """
        if not self.save_dir:
            return None

        path = self.paths[idx]

        # Resize to fixed 256x256 as in test_dianziyan.py
        resized = torch.tensor(anomaly_map).unsqueeze(0).unsqueeze(0)
        resized = torch.nn.functional.interpolate(
            resized,
            size=(256, 256),  # Fixed size as in test_dianziyan.py
            mode='bilinear',
            align_corners=False)
        resized = resized.squeeze().numpy()

        defect_class = os.path.basename(os.path.dirname(path))
        img_name = os.path.splitext(os.path.basename(path))[0]
        target_dir = os.path.join(self.save_dir, defect_class)
        os.makedirs(target_dir, exist_ok=True)

        # Save in the requested format
        if self.save_format == 'npy':
            an_path = os.path.join(target_dir, f'{img_name}.npy')
            np.save(an_path, resized)
        else:  # tiff
            an_path = os.path.join(target_dir, f'{img_name}.tiff')
            tifffile.imwrite(an_path, resized)

        return an_path

    def _save_csv(self, filepaths: List[str],
                  an_map_paths: List[Optional[str]],
                  labels: np.ndarray, scores: np.ndarray) -> None:
        """Save results to CSV file.

        Args:
            filepaths: List of image file paths.
            an_map_paths: List of anomaly map paths.
            labels: Array of labels.
            scores: Array of prediction scores.
        """
        if not self.save_dir:
            return

        # Determine CSV filename
        if self.csv_filename:
            csv_filename = self.csv_filename
        else:
            # Default: save to parent directory with default name
            csv_filename = 'df_test.csv'

        # Create DataFrame
        df = pd.DataFrame({
            'filepath': filepaths,
            'an_map_path': an_map_paths,
            'label': labels.tolist(),
            'prediction_an_det': scores.tolist()
        })

        # Save CSV file
        # If save_dir is like "output_anomaly_maps/dianziyan/anomaly_maps",
        # save CSV to "output_anomaly_maps/dianziyan/"
        if 'anomaly_maps' in self.save_dir:
            # Save to parent directory (category level)
            csv_dir = os.path.dirname(self.save_dir)
            csv_path = os.path.join(csv_dir, csv_filename)
        else:
            # Save to same directory as save_dir
            csv_path = os.path.join(self.save_dir, csv_filename)
        csv_dir = (os.path.dirname(csv_path)
                   if os.path.dirname(csv_path) else '.')
        os.makedirs(csv_dir, exist_ok=True)
        df.to_csv(csv_path, index=False)
