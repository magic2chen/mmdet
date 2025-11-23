# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS


@METRICS.register_module()
class SegADMetric(BaseMetric):
    """Compute ROC-AUC and FPR@95TPR for SegAD.

    Args:
        save_dir (str, optional): If provided, results will be saved to this
            directory.
    """

    def __init__(self, save_dir: str = '') -> None:
        super().__init__()
        self.save_dir = save_dir
        self.scores: List[float] = []
        self.labels: List[int] = []
        self.filepaths: List[str] = []

    def process(self, data_batch: Dict, data_samples: Sequence[Dict]) -> None:
        """Process a batch of data samples.

        Args:
            data_batch (Dict): Data batch.
            data_samples (Sequence[Dict]): Sequence of data samples.
        """
        for sample in data_samples:
            self.scores.append(sample['score'])
            self.labels.append(sample['label'])
            self.filepaths.append(sample.get('filepath', ''))

    def compute_metrics(self, results: List) -> Dict[str, float]:
        """Compute metrics from results.

        Args:
            results (List): List of results.

        Returns:
            Dict[str, float]: Computed metrics.
        """
        return self.evaluate()

    def evaluate(self, size: int = 0) -> Dict[str, float]:
        """Evaluate the model.

        Args:
            size (int): Size parameter (not used).

        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        if not self.scores:
            return dict(auroc=0.0, fpr95tpr=0.0)

        scores = np.asarray(self.scores, dtype=np.float32)
        labels = np.asarray(self.labels, dtype=np.int32)

        # Compute AUROC
        if len(np.unique(labels)) < 2:
            auroc = 0.0
        else:
            auroc = float(roc_auc_score(labels, scores))

        # Compute FPR@95TPR
        fpr95tpr = self._compute_fpr95tpr(scores, labels)

        # Save results if save_dir is provided
        if self.save_dir:
            self._save_results(scores, labels,
                               filepaths=self.filepaths)

        # Clear buffers for next evaluation
        self.scores.clear()
        self.labels.clear()
        self.filepaths.clear()

        return dict(auroc=auroc, fpr95tpr=fpr95tpr)

    def _compute_fpr95tpr(self, scores: np.ndarray,
                           labels: np.ndarray) -> float:
        """Compute FPR@95TPR (False Positive Rate at 95% True Positive Rate).

        Args:
            scores (np.ndarray): Anomaly scores.
            labels (np.ndarray): Ground truth labels.

        Returns:
            float: FPR@95TPR value.
        """
        if len(np.unique(labels)) < 2:
            return 0.0

        # Get threshold at 95% TPR
        positive_scores = scores[labels == 1]
        if len(positive_scores) == 0:
            return 0.0

        threshold = np.quantile(positive_scores, 0.05)

        # Compute FPR
        negative_scores = scores[labels == 0]
        if len(negative_scores) == 0:
            return 0.0

        fpr = (negative_scores >= threshold).mean()
        return float(fpr * 100.0)  # Convert to percentage

    def _save_results(self, scores: np.ndarray, labels: np.ndarray,
                      filepaths: List[str]) -> None:
        """Save results to file.

        Args:
            scores (np.ndarray): Anomaly scores.
            labels (np.ndarray): Ground truth labels.
            filepaths (List[str]): List of file paths.
        """
        if not self.save_dir:
            return

        os.makedirs(self.save_dir, exist_ok=True)
        import pandas as pd

        results_df = pd.DataFrame({
            'filepath': filepaths,
            'label': labels,
            'score': scores,
        })
        results_df.to_csv(
            os.path.join(self.save_dir, 'results.csv'), index=False)
