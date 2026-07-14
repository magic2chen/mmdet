# Copyright (c) OpenMMLab. All rights reserved.
"""CSY DefectFill project for MMDET integration.

CSY_DefectFill is a generative inpainting model using Stable Diffusion + LoRA
+ Textual Inversion for defect detection on MVTec AD dataset.

Module layout (since the 2026-07 refactor):
  - ``DefectFillCore``: SD 2 inpainting + LoRA + Textual Inversion, formerly
    known as ``DefectFillModel``. Lives at
    ``projects/csy_defectfill/models/defectfill_core.py`` (moved out of the
    external ``mmdet/DefectFill/`` package).
  - ``DefectFillDetector``: mmengine/MMDET-compatible wrapper with
    train_step / val_step / test_step.
  - ``MVTecDefectDataset``: data loader.
  - ``DefectMetric``: ROC-AUC evaluator.

External ``mmdet/DefectFill/`` is now reserved for data and SD weights only
(``DefectFill/DATA/`` and ``DefectFill/ck/``); no Python import from there.
"""

from projects.csy_defectfill.models import (
    AttentionStoreProcessor,
    DefectFillCore,
    DefectFillDetector,
    DefectFillLoss,
    USE_MODELSCOPE,
)
from projects.csy_defectfill.datasets import MVTecDefectDataset
from projects.csy_defectfill.metrics import DefectMetric

__all__ = [
    'AttentionStoreProcessor',
    'DefectFillCore',
    'DefectFillDetector',
    'DefectFillLoss',
    'MVTecDefectDataset',
    'DefectMetric',
    'USE_MODELSCOPE',
]