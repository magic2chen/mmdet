# Copyright (c) OpenMMLab. All rights reserved.
"""DefectFill data utilities.

Standalone CLI scripts for preparing DefectFill training/eval data, all
sitting next to ``mmdet/tools/`` so they can be invoked as:

    python -m tools.defectfill_utils.<script> --help

These were moved here from ``mmdet/DefectFill/tools/`` as part of the
2026-07 refactor (see ``projects/csy_defectfill/README.md``). The
``DefectFill/`` directory now only carries data and SD weights.

No script here imports from ``DefectFill/`` or ``projects.csy_defectfill``;
they are deliberately self-contained (stdlib + cv2 / numpy / PIL / tqdm)
so they remain usable on systems without the full mmdet stack installed.
"""

from . import (filter_labelme_by_label, generate_glow,
               labelme_split_by_category, labelme_to_mask_and_crop,
               paste_back)

__all__ = [
    'filter_labelme_by_label',
    'generate_glow',
    'labelme_split_by_category',
    'labelme_to_mask_and_crop',
    'paste_back',
]
