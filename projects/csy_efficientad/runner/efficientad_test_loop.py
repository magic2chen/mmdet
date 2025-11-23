# Copyright (c) OpenMMLab. All rights reserved.
"""Custom test loop for EfficientAD.

This loop generates both df_test.csv and df_training.csv automatically.
"""
from mmengine.logging import print_log
from mmengine.runner import TestLoop

from mmdet.registry import LOOPS


@LOOPS.register_module()
class EfficientADTestLoop(TestLoop):
    """Custom test loop for EfficientAD.

    This loop runs test evaluation first, then automatically runs validation
    evaluation to generate both df_test.csv and df_training.csv, matching
    the behavior of test_dianziyan.py.

    Args:
        runner: The runner instance.
        dataloader: Test dataloader.
        evaluator: Test evaluator.
    """

    def run(self) -> None:
        """Run test loop, then validation loop if configured."""
        # Run standard test evaluation (generates df_test.csv)
        super().run()

        # Check if runner has val_dataloader and val_evaluator configured
        # In test mode, runner might not have these initialized, so we check
        # the config instead
        has_val = False
        if hasattr(self.runner, 'val_dataloader') and hasattr(
                self.runner, 'val_evaluator'):
            has_val = (self.runner.val_dataloader is not None and
                       self.runner.val_evaluator is not None)
        elif hasattr(self.runner, 'cfg'):
            # Check config for val_dataloader and val_evaluator
            has_val = ('val_dataloader' in self.runner.cfg and
                       'val_evaluator' in self.runner.cfg)

        if has_val:
            # If val_dataloader/val_evaluator are not initialized, build them
            if (not hasattr(self.runner, 'val_dataloader') or
                    self.runner.val_dataloader is None):
                from mmengine.dataset import build_dataloader
                from mmengine.evaluator import build_evaluator
                if 'val_dataloader' in self.runner.cfg:
                    print_log('Building val_dataloader...')
                    self.runner.val_dataloader = build_dataloader(
                        self.runner.cfg.val_dataloader)
                if 'val_evaluator' in self.runner.cfg:
                    print_log('Building val_evaluator...')
                    # Copy config from test_evaluator to val_evaluator
                    # before building
                    val_eval_cfg = (
                        self.runner.cfg.val_evaluator.copy())
                    if 'test_evaluator' in self.runner.cfg:
                        test_eval_cfg = self.runner.cfg.test_evaluator
                        if isinstance(test_eval_cfg, list):
                            test_eval_cfg = (test_eval_cfg[0]
                                             if test_eval_cfg else {})
                        # Copy save_dir, data_root, save_format
                        if 'save_dir' in test_eval_cfg:
                            val_eval_cfg['save_dir'] = (
                                test_eval_cfg['save_dir'])
                        if 'data_root' in test_eval_cfg:
                            val_eval_cfg['data_root'] = (
                                test_eval_cfg['data_root'])
                        if 'save_format' in test_eval_cfg:
                            val_eval_cfg['save_format'] = (
                                test_eval_cfg['save_format'])
                    # Set csv_filename to df_training.csv
                    val_eval_cfg['csv_filename'] = 'df_training.csv'
                    self.runner.val_evaluator = build_evaluator(val_eval_cfg)
            print_log(
                'Running validation evaluation to generate df_training.csv...')

            # Temporarily replace test dataloader and evaluator with val ones
            original_dataloader = self.dataloader
            original_evaluator = self.evaluator

            self.dataloader = self.runner.val_dataloader
            self.evaluator = self.runner.val_evaluator

            # Copy configuration from test_evaluator config to val_evaluator
            # (save_dir, data_root, save_format, etc.)
            # Read from config instead of evaluator instance to get updated
            # values
            original_csv_filename = None

            # Get test evaluator config
            test_eval_cfg = None
            if (hasattr(self.runner, 'cfg') and
                    'test_evaluator' in self.runner.cfg):
                test_eval_cfg = self.runner.cfg.test_evaluator
                # If it's a list, get the first one
                if isinstance(test_eval_cfg, list):
                    test_eval_cfg = (test_eval_cfg[0]
                                     if test_eval_cfg else None)

            # Also try to get from test_evaluator instance (as fallback)
            test_evaluator = self.runner.test_evaluator
            test_metric = None
            if hasattr(test_evaluator, 'metrics') and test_evaluator.metrics:
                test_metric = test_evaluator.metrics[0]
            elif hasattr(test_evaluator, 'save_dir'):
                test_metric = test_evaluator

            # Get save_dir from config first, then from instance
            save_dir = None
            data_root = None
            save_format = None
            if test_eval_cfg:
                save_dir = test_eval_cfg.get('save_dir', '')
                data_root = test_eval_cfg.get('data_root', None)
                save_format = test_eval_cfg.get('save_format', 'npy')
            elif test_metric:
                if hasattr(test_metric, 'save_dir'):
                    save_dir = test_metric.save_dir
                if hasattr(test_metric, 'data_root'):
                    data_root = test_metric.data_root
                if hasattr(test_metric, 'save_format'):
                    save_format = test_metric.save_format

            if save_dir:
                # Copy config to val_evaluator
                val_metric = None
                if (hasattr(self.evaluator, 'metrics') and
                        self.evaluator.metrics):
                    val_metric = self.evaluator.metrics[0]
                elif hasattr(self.evaluator, 'save_dir'):
                    val_metric = self.evaluator

                if val_metric:
                    print_log(
                        f'Copying config: save_dir={save_dir}, '
                        f'data_root={data_root}, save_format={save_format}')
                    if hasattr(val_metric, 'save_dir'):
                        val_metric.save_dir = save_dir
                    if data_root and hasattr(val_metric, 'data_root'):
                        val_metric.data_root = data_root
                    if save_format and hasattr(val_metric, 'save_format'):
                        val_metric.save_format = save_format
                    # Set CSV filename to df_training.csv
                    if hasattr(val_metric, 'csv_filename'):
                        original_csv_filename = getattr(
                            val_metric, 'csv_filename', None)
                        val_metric.csv_filename = 'df_training.csv'
                        print_log('Set csv_filename to df_training.csv')
                else:
                    print_log(
                        'WARNING: Could not find val_metric to copy config')
            else:
                print_log(
                    f'WARNING: save_dir is empty. '
                    f'test_eval_cfg={test_eval_cfg}, '
                    f'test_metric={test_metric}')

            # Run validation evaluation
            try:
                # Call parent's run method to execute validation
                self.runner.call_hook('before_test')
                self.runner.call_hook('before_test_epoch')
                self.runner.model.eval()

                for idx, data_batch in enumerate(self.dataloader):
                    self.run_iter(idx, data_batch)

                # Compute metrics (this will save df_training.csv)
                metrics = self.evaluator.evaluate(
                    len(self.dataloader.dataset))
                self.runner.call_hook('after_test_epoch', metrics=metrics)
                self.runner.call_hook('after_test')
            finally:
                # Restore original filename if it was changed
                if original_csv_filename is not None:
                    if hasattr(self.evaluator, 'metrics'):
                        for metric in self.evaluator.metrics:
                            if hasattr(metric, 'csv_filename'):
                                metric.csv_filename = original_csv_filename
                                break

            # Restore original dataloader and evaluator
            self.dataloader = original_dataloader
            self.evaluator = original_evaluator

            print_log(
                'Validation evaluation completed. df_training.csv saved.')
