# Copyright (c) OpenMMLab. All rights reserved.
import string
from typing import Dict, List

import numpy as np
from scipy.stats import kurtosis, skew


class FeatureExtractor:
    """Extract statistical features from anomaly maps based on
    segmentation maps.

    Args:
        num_components (int): Number of components in the segmentation map.
        models_list (List[str]): List of base anomaly detection model names.
    """

    def __init__(self, num_components: int, models_list: List[str]) -> None:
        self.num_components = num_components
        self.models_list = models_list
        self.components = self._get_components(num_components)
        self.feature_names = self._get_feature_names()

    def _get_components(self, num_components: int) -> List[str]:
        """Get component names (a, b, c, ...)."""
        return [string.ascii_lowercase[i] for i in range(num_components)]

    def _get_feature_names(self) -> List[str]:
        """Generate feature names based on components and models."""
        features = ["_q995", "_scewness", "_kurtosis", "_mean"]
        lst = [
            c + "_" + m for c in self.components for m in self.models_list
        ]
        list_features = [comp_model + f for comp_model in lst
                         for f in features]
        for model in self.models_list:
            list_features.append("an_det_score_" + model)
        return list_features

    @staticmethod
    def extract_features_from_part(part: str, model: str,
                                   selection: np.ndarray) -> Dict[str, float]:
        """Extract statistical features from a part of anomaly map.

        Args:
            part (str): Component name (e.g., 'a', 'b').
            model (str): Model name.
            selection (np.ndarray): Selected values from anomaly map.

        Returns:
            Dict[str, float]: Dictionary of extracted features.
        """
        if len(selection) > 0:
            return {
                f"{part}_{model}_q995": float(np.quantile(selection, 0.995)),
                f"{part}_{model}_scewness": float(skew(selection)),
                f"{part}_{model}_kurtosis": float(kurtosis(selection)),
                f"{part}_{model}_mean": float(selection.mean()),
            }
        else:
            return {
                f"{part}_{model}_q995": 0.0,
                f"{part}_{model}_scewness": 0.0,
                f"{part}_{model}_kurtosis": 0.0,
                f"{part}_{model}_mean": 0.0,
            }

    def extract_features(
            self, segm_map: np.ndarray,
            anomaly_maps: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Extract features from segmentation map and anomaly maps.

        Args:
            segm_map (np.ndarray): Segmentation map.
            anomaly_maps (Dict[str, np.ndarray]): Dictionary of anomaly
                maps for different models.

        Returns:
            Dict[str, float]: Dictionary of extracted features.
        """
        features = {}
        # Ensure segm_map is 2D
        if segm_map.ndim == 1:
            # Try to reshape to square if possible
            size = int(np.sqrt(segm_map.size))
            if size * size == segm_map.size:
                segm_map = segm_map.reshape(size, size)
            else:
                raise ValueError(
                    f'Cannot reshape segm_map from shape {segm_map.shape}')

        for model in self.models_list:
            anomaly_map = anomaly_maps[model]
            # Ensure anomaly_map is 2D and matches segm_map shape
            if anomaly_map.ndim == 1:
                # Try to reshape to square if possible
                size = int(np.sqrt(anomaly_map.size))
                if size * size == anomaly_map.size:
                    anomaly_map = anomaly_map.reshape(size, size)
                else:
                    raise ValueError(
                        f'Cannot reshape anomaly_map from shape '
                        f'{anomaly_map.shape}')

            # Ensure shapes match
            if anomaly_map.shape != segm_map.shape:
                # Resize anomaly_map to match segm_map
                from scipy.ndimage import zoom
                zoom_factors = (segm_map.shape[0] / anomaly_map.shape[0],
                                segm_map.shape[1] / anomaly_map.shape[1])
                anomaly_map = zoom(anomaly_map, zoom_factors, order=1)

            for j, part in enumerate(self.components):
                selection = anomaly_map[segm_map == j]
                part_features = self.extract_features_from_part(
                    part, model, selection)
                features.update(part_features)
            features[f"an_det_score_{model}"] = float(np.max(anomaly_map))
        return features
