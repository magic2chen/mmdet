# Copyright (c) OpenMMLab. All rights reserved.
"""Custom training loop for SegAD model using XGBoost."""
import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

from mmengine.logging import print_log
from mmengine.runner.loops import EpochBasedTrainLoop

from mmdet.registry import LOOPS

from ..models import SegADModel

# Categories and their number of components
# CATEGORIES = (
#     "candle", "capsules", "cashew", "chewinggum", "fryum",
#     "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
# )

# NUM_COMPONENTS = {
#     "candle": 2, "capsules": 2, "cashew": 2, "chewinggum": 2,
#     "fryum": 2, "pipe_fryum": 2, "macaroni1": 2, "macaroni2": 2,
#     "pcb1": 6, "pcb2": 8, "pcb3": 8, "pcb4": 6,
# }

CATEGORIES = (
    "dianziyan"
)

NUM_COMPONENTS = {
    "dianziyan": 2
}

# Seeds to reproduce results from the paper
SEEDS = [333, 576, 725, 823, 831, 902, 226, 598, 874, 589]


@LOOPS.register_module()
class SegADTrainLoop(EpochBasedTrainLoop):
    """Custom training loop for SegAD that uses XGBoost.

    This loop handles the special training procedure for SegAD:
    1. Load data from CSV files
    2. Split data for training and testing
    3. Extract features from segmentation and anomaly maps
    4. Train XGBoost classifier
    5. Evaluate and save results

    Args:
        runner (Runner): The runner instance.
        dataloader: Training dataloader (not used directly).
        max_epochs (int): Maximum epochs (not used for XGBoost).
        val_interval (int): Validation interval (not used).
        category (str): Category name to train on.
        models_list (List[str]): List of base anomaly detection models.
        bad_parts (int): Number of bad parts to use for training.
        segm_path (str): Path to segmentation maps.
        an_path (str): Path to anomaly maps.
        seeds (List[int], optional): Random seeds for training.
            Defaults to None (uses default seeds).
    """

    def __init__(self,
                 runner,
                 dataloader,
                 max_epochs: int = 1,
                 val_interval: int = 1,
                 category: str = 'candle',
                 models_list: List[str] = None,
                 bad_parts: int = 10,
                 segm_path: str = './data/visa_segm',
                 an_path: str = './data/anomaly_maps',
                 seeds: List[int] = None) -> None:
        super().__init__(runner, dataloader, max_epochs, val_interval)
        self.category = category
        self.models_list = models_list if models_list else ['efficient_ad']
        self.bad_parts = bad_parts
        self.segm_path = segm_path
        self.an_path = an_path
        self.seeds = seeds if seeds is not None else SEEDS

        if category not in CATEGORIES:
            raise ValueError(f"Category {category} not in {CATEGORIES}")

    def run(self) -> None:
        """Run the training loop."""
        self.runner.call_hook('before_train')
        self.runner.call_hook('before_train_epoch')

        # Run training for all seeds
        results = {}
        results_detailed = {}
        auroc_sum = 0
        fpr95tpr_sum = 0

        for seed in self.seeds:
            print_log(
                f'Training SegAD for {self.category} with seed {seed}...')
            metrics_dict = self._train_and_evaluate(seed)
            auroc_sum += metrics_dict['auroc']
            fpr95tpr_sum += metrics_dict['fpr95tpr']

            # Detailed auroc metrics per seed
            results_detailed[f"{self.category}_{seed}"] = (
                self.category, seed,
                round(metrics_dict['auroc'] * 100, 1))

        # Calculate mean metrics
        mean_auroc = auroc_sum / len(self.seeds) * 100
        mean_fpr95tpr = fpr95tpr_sum / len(self.seeds) * 100
        print_log(
            f"{self.category}, mean results for all seeds. "
            f"Cl. AUROC: {round(mean_auroc, 1)}, "
            f"FPR@95TPR: {round(mean_fpr95tpr, 1)}")
        results[self.category] = (round(mean_auroc, 1),
                                  round(mean_fpr95tpr, 1))

        # Save results
        work_dir = self.runner.work_dir
        os.makedirs(work_dir, exist_ok=True)
        pd.DataFrame.from_dict(results, orient="index").to_csv(
            os.path.join(work_dir, "results.csv"))
        pd.DataFrame.from_dict(results_detailed, orient="index").to_csv(
            os.path.join(work_dir, "results_detailed.csv"))

        print_log(f"Results saved to {work_dir}")

        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')

    def _train_and_evaluate(self, seed: int) -> dict:
        """Train and evaluate SegAD for a single seed.

        Args:
            seed (int): Random seed.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        # Load data
        # CSV files can be at different locations:
        # 1. {an_path}/df_training.csv (root level, saved by
        #    EfficientADTestLoop)
        # 2. {an_path}/{category}/df_training.csv
        # 3. {an_path}/{model}/{category}/df_training.csv
        csv_path = os.path.join(self.an_path, "df_training.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(self.an_path, self.category,
                                    "df_training.csv")
        if not os.path.exists(csv_path):
            # Try alternative path with model layer
            csv_path = os.path.join(self.an_path, self.models_list[0],
                                    self.category, "df_training.csv")
        if not os.path.exists(csv_path):
            path1 = os.path.join(self.an_path, 'df_training.csv')
            path2 = os.path.join(self.an_path, self.category,
                                 'df_training.csv')
            path3 = os.path.join(self.an_path, self.models_list[0],
                                 self.category, 'df_training.csv')
            raise FileNotFoundError(
                f"df_training.csv not found. Tried:\n"
                f"  - {path1}\n  - {path2}\n  - {path3}")
        df_training_all = pd.read_csv(csv_path, index_col=0).reset_index()

        csv_path = os.path.join(self.an_path, "df_test.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(self.an_path, self.category, "df_test.csv")
        if not os.path.exists(csv_path):
            # Try alternative path with model layer
            csv_path = os.path.join(self.an_path, self.models_list[0],
                                    self.category, "df_test.csv")
        if not os.path.exists(csv_path):
            path1 = os.path.join(self.an_path, 'df_test.csv')
            path2 = os.path.join(self.an_path, self.category, 'df_test.csv')
            path3 = os.path.join(self.an_path, self.models_list[0],
                                 self.category, 'df_test.csv')
            raise FileNotFoundError(
                f"df_test.csv not found. Tried:\n"
                f"  - {path1}\n  - {path2}\n  - {path3}")
        df_testing_all = pd.read_csv(csv_path, index_col=0).reset_index()

        # Calculate scale_pos_weight
        scale_pos_weight = len(df_training_all.index) / self.bad_parts
        num_comp_cls = NUM_COMPONENTS[self.category]

        # Split bad images from the test set for training and testing
        df_testing_bad = df_testing_all.loc[df_testing_all.label == 1]
        df_training_bad, df_testing_bad = train_test_split(
            df_testing_bad,
            test_size=len(df_testing_bad.index) - self.bad_parts,
            random_state=seed)
        df_training = pd.concat([
            df_training_all.loc[df_training_all.label == 0], df_training_bad
        ])
        df_testing = pd.concat([
            df_testing_all.loc[df_testing_all.label == 0], df_testing_bad
        ])

        # Initialize model
        model = SegADModel(
            num_components=num_comp_cls,
            models_list=self.models_list,
            seed=seed,
            scale_pos_weight=scale_pos_weight,
        )

        # Get Data class
        from ..datasets import SegADDataset

        # Create datasets
        train_dataset = SegADDataset(
            data_root='',
            segm_path=self.segm_path,
            an_path=self.an_path,
            models_list=self.models_list,
            category=self.category,
            split='train',
            csv_file='',  # We'll use the dataframe directly
            num_components=num_comp_cls,
        )
        train_dataset.df = df_training
        train_dataset.data_list = train_dataset._prepare_data_list()

        test_dataset = SegADDataset(
            data_root='',
            segm_path=self.segm_path,
            an_path=self.an_path,
            models_list=self.models_list,
            category=self.category,
            split='test',
            csv_file='',
            num_components=num_comp_cls,
        )
        test_dataset.df = df_testing
        test_dataset.data_list = test_dataset._prepare_data_list()

        # Extract features for training
        print_log(f'Extracting training features for {self.category} '
                  f'(seed {seed})...')
        train_features = []
        train_labels = []
        for idx in range(len(train_dataset)):
            item = train_dataset[idx]
            features = model.extract_features(
                item['segm_map'], item['anomaly_maps'])
            train_features.append(features)
            train_labels.append(item['label'])
        train_features = np.array(train_features)
        train_labels = np.array(train_labels)

        # Train XGBoost
        print_log(f'Training XGBoost for {self.category} (seed {seed})...')
        model.fit(train_features, train_labels)

        # Save XGBoost model
        work_dir = self.runner.work_dir
        os.makedirs(work_dir, exist_ok=True)
        xgb_model_path = os.path.join(
            work_dir, f'xgb_model_{self.category}_seed_{seed}.pkl')
        model.save_xgb_model(xgb_model_path)
        print_log(f'XGBoost model saved to {xgb_model_path}')

        # Extract features for testing
        print_log(f'Extracting test features for {self.category} '
                  f'(seed {seed})...')
        test_features = []
        test_labels = []
        for idx in range(len(test_dataset)):
            item = test_dataset[idx]
            features = model.extract_features(
                item['segm_map'], item['anomaly_maps'])
            test_features.append(features)
            test_labels.append(item['label'])
        test_features = np.array(test_features)
        test_labels = np.array(test_labels)

        # Predict
        predictions = model.predict_proba(test_features)[:, 1]
        df_testing = df_testing.copy()
        df_testing["final_score"] = predictions
        thr_accept = df_testing.loc[df_testing.label == 1,
                                    "final_score"].quantile(0.05)

        # Calculate metrics
        auroc = metrics.roc_auc_score(test_labels, predictions)
        fpr95tpr = (1 - (df_testing.loc[df_testing.label == 0,
                                        "final_score"] <
                         thr_accept).mean())

        return {
            'auroc': auroc,
            'fpr95tpr': fpr95tpr,
        }
