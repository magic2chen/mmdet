# Copyright (c) OpenMMLab. All rights reserved.
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from xgboost import XGBClassifier

from mmengine.model import BaseModule
from mmdet.registry import MODELS

from ..utils import FeatureExtractor


@MODELS.register_module()
class SegADModel(BaseModule):
    """SegAD model that uses XGBoost for anomaly detection.

    This model extracts features from segmentation maps and anomaly maps,
    then uses XGBoost to classify anomalies.

    Args:
        num_components (int): Number of components in segmentation map.
        models_list (List[str]): List of base anomaly detection model names.
        seed (int): Random seed for XGBoost.
        scale_pos_weight (float): Scale positive weight for XGBoost.
        xgb_params (Optional[Dict]): Additional parameters for XGBoost.
    """

    def __init__(self,
                 num_components: int = 2,
                 models_list: List[str] = None,
                 seed: int = 333,
                 scale_pos_weight: float = 1.0,
                 xgb_params: Optional[Dict] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_components = num_components
        self.models_list = models_list if models_list else ['efficient_ad']
        self.seed = seed
        self.scale_pos_weight = scale_pos_weight

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            num_components=num_components, models_list=self.models_list)

        # Initialize XGBoost classifier
        default_params = {
            'random_state': seed,
            'n_estimators': 10,
            'max_depth': 5,
            'num_parallel_tree': 200,
            'learning_rate': 0.3,
            'objective': 'binary:logitraw',
            'colsample_bynode': 0.6,
            'colsample_bytree': 0.6,
            'subsample': 0.6,
            'reg_alpha': 1.0,
            'scale_pos_weight': scale_pos_weight,
        }
        if xgb_params:
            default_params.update(xgb_params)
        self.xgb = XGBClassifier(**default_params)
        self.is_trained = False

        # Add a dummy parameter for optimizer (required by MMEngine)
        # This parameter won't be used but allows the optimizer to work
        # We set requires_grad=True so optimizer can find it, but it
        # won't affect training
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=True)

    def extract_features(self, segm_map: np.ndarray,
                         anomaly_maps: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract features from segmentation and anomaly maps.

        Args:
            segm_map (np.ndarray): Segmentation map.
            anomaly_maps (Dict[str, np.ndarray]): Dictionary of anomaly maps.

        Returns:
            np.ndarray: Extracted features as a 1D array.
        """
        features_dict = self.feature_extractor.extract_features(
            segm_map, anomaly_maps)
        # Convert to array in the correct order
        feature_array = np.array([
            features_dict[name]
            for name in self.feature_extractor.feature_names
        ])
        return feature_array

    def train_step(self, data_batch: Dict, optim_wrapper) -> Dict:
        """Training step for SegAD.

        Note: This is a placeholder. Actual training should be done
        using the custom training script since XGBoost doesn't follow
        the standard PyTorch training loop.

        Args:
            data_batch (Dict): Data batch.
            optim_wrapper: Optimizer wrapper (not used for XGBoost).

        Returns:
            Dict: Training results.
        """
        # XGBoost training is done separately via fit() method
        return {'loss': 0.0}

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit XGBoost model on features and labels.

        Args:
            features (np.ndarray): Feature array of shape (N, F).
            labels (np.ndarray): Label array of shape (N,).
        """
        self.xgb.fit(features, labels)
        self.is_trained = True

    def save_xgb_model(self, filepath: str) -> None:
        """Save XGBoost model to file.

        Args:
            filepath (str): Path to save the model.
        """
        if not self.is_trained:
            raise RuntimeError('Model must be trained before saving.')
        # Create directory if needed
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        # Save using pickle for compatibility
        with open(filepath, 'wb') as f:
            pickle.dump(self.xgb, f)

    def load_xgb_model(self, filepath: str) -> None:
        """Load XGBoost model from file.

        Args:
            filepath (str): Path to load the model from.
        """
        with open(filepath, 'rb') as f:
            self.xgb = pickle.load(f)
        self.is_trained = True

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict probability scores.

        Args:
            features (np.ndarray): Feature array of shape (N, F).

        Returns:
            np.ndarray: Probability scores of shape (N, 2).
        """
        if not self.is_trained:
            raise RuntimeError('Model must be trained before prediction.')
        return self.xgb.predict_proba(features)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict binary labels.

        Args:
            features (np.ndarray): Feature array of shape (N, F).

        Returns:
            np.ndarray: Binary predictions of shape (N,).
        """
        if not self.is_trained:
            raise RuntimeError('Model must be trained before prediction.')
        return self.xgb.predict(features)

    def val_step(self, data_batch: Dict, **kwargs) -> List[Dict]:
        """Validation step for SegAD.

        Args:
            data_batch (Dict): Data batch.

        Returns:
            List[Dict]: Validation results.
        """
        return self._infer_step(data_batch)

    def test_step(self, data_batch: Dict, **kwargs) -> List[Dict]:
        """Test step for SegAD.

        Args:
            data_batch (Dict): Data batch.

        Returns:
            List[Dict]: Test results.
        """
        return self._infer_step(data_batch)

    def _infer_step(self, data_batch: Dict) -> List[Dict]:
        """Inference step.

        Args:
            data_batch (Dict): Data batch.

        Returns:
            List[Dict]: Inference results.
        """
        segm_maps = data_batch['segm_map']
        anomaly_maps_list = data_batch['anomaly_maps']
        labels = data_batch['label']
        filepaths = data_batch['filepath']

        results = []
        batch_size = len(segm_maps) if isinstance(segm_maps, list) else 1

        if not isinstance(segm_maps, list):
            segm_maps = [segm_maps]
            anomaly_maps_list = [anomaly_maps_list]
            if not isinstance(labels, (list, np.ndarray)):
                labels = [labels]
            filepaths = ([filepaths]
                         if not isinstance(filepaths, list) else filepaths)

        for idx in range(batch_size):
            segm_map = segm_maps[idx]
            anomaly_maps = anomaly_maps_list[idx]
            label_val = (int(labels[idx])
                         if isinstance(labels, (list, np.ndarray))
                         else int(labels))
            filepath = filepaths[idx]

            # Extract features
            features = self.extract_features(segm_map, anomaly_maps)
            features = features.reshape(1, -1)

            # Predict
            if self.is_trained:
                proba = self.predict_proba(features)[0]
                score = proba[1]  # Probability of anomaly
            else:
                score = 0.0

            results.append({
                'score': float(score),
                'label': label_val,
                'filepath': filepath,
                'features': features[0],
            })

        return results

    def forward(self, *args, **kwargs):
        """Forward method (not used for SegAD)."""
        raise NotImplementedError(
            'SegADModel does not support direct forwarding; '
            'use fit() and predict_proba() methods instead.')
