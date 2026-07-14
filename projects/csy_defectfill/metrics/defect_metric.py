# Copyright (c) OpenMMLab. All rights reserved.
"""Anomaly detection metric for DefectFill."""

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric

from mmdet.registry import METRICS


@METRICS.register_module()
class DefectMetric(BaseMetric):
    """Anomaly detection metric for DefectFill.

    Computes ROC-AUC based on LPIPS-based anomaly scores.
    During inference, DefectFillDetector returns DetDataSample with
    anomaly_score field. This metric collects scores and computes
    the Area Under the ROC Curve.

    Args:
        save_dir (str, optional): Directory to save anomaly maps.
        save_format (str): 'npy' or 'tiff' format for saving maps.
    """

    def __init__(
        self,
        save_dir: str = '',
        save_format: str = 'npy',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.save_dir = save_dir
        self.save_format = save_format

        self.scores: List[float] = []
        self.labels: List[int] = []
        self.paths: List[str] = []

    def process(self, data_batch: Dict, data_samples: Sequence[Dict]) -> None:
        """Process a batch of results.

        Args:
            data_batch (Dict): Batch of input data containing img_path and label
            data_samples (Sequence[Dict]): List of DetDataSample with
                pred_images and anomaly_score fields
        """
        for sample in data_samples:
            # Extract anomaly score
            anomaly_score = sample.get('anomaly_score', None)
            if anomaly_score is not None:
                if torch.is_tensor(anomaly_score):
                    self.scores.append(anomaly_score.item())
                else:
                    self.scores.append(float(anomaly_score))
            else:
                self.scores.append(0.0)

            # Extract label (0=normal, 1=defect)
            label = sample.get('label', None)
            if label is not None:
                if torch.is_tensor(label):
                    self.labels.append(int(label.item()))
                else:
                    self.labels.append(int(label))
            else:
                self.labels.append(0)

            # Extract path for optional saving
            img_path = sample.get('img_path', '')
            self.paths.append(img_path)

    def evaluate(self, size: int = 0) -> Dict[str, float]:
        """Compute ROC-AUC metric.

        Args:
            size: Number of samples (not used, kept for API compatibility)

        Returns:
            Dict with 'auc' containing the ROC-AUC score
        """
        return self.compute_metrics()

    def compute_metrics(self, results: Optional[List] = None) -> Dict[str, float]:
        """Compute ROC-AUC metric.

        Args:
            results: mmengine 累积的 data_samples 列表（保留以兼容 BaseMetric
                接口；本指标已经从 self.scores / self.labels 累积数据，
                所以忽略此参数）。

        Returns:
            Dict with 'auc' containing the ROC-AUC score
        """
        if not self.scores:
            return dict(auc=0.0)

        y_scores = np.asarray(self.scores)
        y_labels = np.asarray(self.labels, dtype=np.int32)

        # Need both positive and negative samples for AUC
        if len(np.unique(y_labels)) < 2:
            auc = 0.5 if len(y_labels) > 0 else 0.0
        else:
            try:
                from sklearn.metrics import roc_auc_score
                auc = float(roc_auc_score(y_labels, y_scores))
            except Exception:
                auc = 0.0

        return dict(auc=auc)

    def _save_anomaly_map(
        self,
        anomaly_map: np.ndarray,
        path: str
    ) -> Optional[str]:
        """Save anomaly map to file.

        Args:
            anomaly_map: Anomaly map array
            path: Original image path (used to derive save name)

        Returns:
            Saved path if successful, None otherwise
        """
        if not self.save_dir:
            return None

        import os
        os.makedirs(self.save_dir, exist_ok=True)

        # Derive save name from original path
        base_name = os.path.splitext(os.path.basename(path))[0]
        if self.save_format == 'npy':
            save_path = os.path.join(self.save_dir, f"{base_name}_anomaly.npy")
            np.save(save_path, anomaly_map)
        elif self.save_format == 'tiff':
            save_path = os.path.join(self.save_dir, f"{base_name}_anomaly.tiff")
            try:
                import cv2
                cv2.imwrite(save_path, (anomaly_map * 255).astype(np.uint8))
            except Exception:
                save_path = None
        else:
            save_path = None

        return save_path