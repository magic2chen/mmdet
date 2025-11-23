# Copyright (c) OpenMMLab. All rights reserved.
"""Hook to save teacher, student, and autoencoder models separately."""
import os

import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class EfficientADSaveHook(Hook):
    """Hook to save teacher, student, and autoencoder models separately.

    This hook saves the models in the same format as the original EfficientAD
    project:
    - teacher_tmp.pth, student_tmp.pth, autoencoder_tmp.pth (every interval)
    - teacher_final.pth, student_final.pth, autoencoder_final.pth (at end)

    Args:
        output_dir (str): Base output directory. Models will be saved to
            {output_dir}/trainings/{Data}/{subdataset}/
        dataset (str): Dataset name (e.g., 'mvtec_ad').
        subdataset (str): Sub-Data name (e.g., 'bottle').
        interval (int): Save interval in iterations. Defaults to 1000.
    """

    def __init__(self,
                 output_dir: str = './output',
                 dataset: str = 'mvtec_ad',
                 subdataset: str = 'bottle',
                 interval: int = 1000) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.dataset = dataset
        self.subdataset = subdataset
        self.interval = interval

        # Create output directory
        self.train_output_dir = os.path.join(
            self.output_dir, 'trainings', self.dataset, self.subdataset)
        os.makedirs(self.train_output_dir, exist_ok=True)

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs=None) -> None:
        """Save models every interval iterations."""
        if runner.iter % self.interval == 0:
            self._save_models(runner, suffix='tmp')

    def after_train(self, runner: Runner) -> None:
        """Save final models after training."""
        self._save_models(runner, suffix='final')

    def _save_models(self, runner: Runner, suffix: str = 'tmp') -> None:
        """Save teacher, student, and autoencoder models.

        Args:
            runner (Runner): The runner instance.
            suffix (str): Suffix for the saved files ('tmp' or 'final').
        """
        model = runner.model
        if not hasattr(model, 'teacher') or not hasattr(model, 'student') or \
           not hasattr(model, 'autoencoder'):
            runner.logger.warning(
                'Model does not have teacher, student, or autoencoder '
                'attributes. Skipping model save.')
            return

        # Set models to eval mode for saving
        teacher_was_training = model.teacher.training
        student_was_training = model.student.training
        autoencoder_was_training = model.autoencoder.training

        model.teacher.eval()
        model.student.eval()
        model.autoencoder.eval()

        # Save models
        teacher_path = os.path.join(
            self.train_output_dir, f'teacher_{suffix}.pth')
        student_path = os.path.join(
            self.train_output_dir, f'student_{suffix}.pth')
        autoencoder_path = os.path.join(
            self.train_output_dir, f'autoencoder_{suffix}.pth')

        torch.save(model.teacher, teacher_path)
        torch.save(model.student, student_path)
        torch.save(model.autoencoder, autoencoder_path)

        runner.logger.info(
            f'Saved models to {self.train_output_dir} with suffix {suffix}')

        # Restore training mode
        if teacher_was_training:
            model.teacher.train()
        if student_was_training:
            model.student.train()
        if autoencoder_was_training:
            model.autoencoder.train()
