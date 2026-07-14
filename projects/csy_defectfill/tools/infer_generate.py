"""DefectFill2-style synthetic-defect generation, but driven by the
MMDET-wrapped ``DefectFillDetector``.

Mirrors ``DefectFill2/inference.py`` behaviour:

* 1:1 (good image, mask) pairing from the user's data root (DefectFill2 layout),
* smart_crop to 512x512 anchored on the mask,
* generate ``num_samples`` candidates per input, pick LPIPS-argmax,
* save ``*_generated.png`` / ``*_original.png`` / ``*_mask.png``,
* emit ``inference_log.json``.

Key difference from ``tools/test.py``: this script does NOT go through
``TestLoop`` (which only knows how to call ``test_step`` and aggregate
``anomaly_score``); instead it builds the model from the cfg, then calls
the new ``DefectFillDetector.generate_defect`` method directly.

Usage:
    python projects/csy_defectfill/tools/infer_generate.py \\
        projects/csy_defectfill/configs/defectfill_phone_infer.py \\
        work_dirs/defectfill_phone/iter_20000.pth \\
        --data-root DefectFill/DATA/my_infer/phone \\
        --object-class phone \\
        --defect-type yashang \\
        --output-dir work_dirs/defectfill_phone/infer_yashang \\
        --total-images 16 --num-samples 8 --steps 50
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image

# Reuse DefectFill2's smart_crop / pairing utilities verbatim so behaviour
# matches the original repo.
THIS = Path(__file__).resolve()
DF2_DIR = THIS.parents[3] / 'DefectFill2'
sys.path.insert(0, str(DF2_DIR))
from inference import (  # type: ignore
    smart_crop_dynamic,
    get_image_mask_pairs,
    count_available_resources,
    calculate_generation_plan,
)  # noqa: E402

from mmengine.config import Config  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402
from mmdet.registry import MODELS  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('config', help='Inference cfg, e.g. defectfill_phone_infer.py')
    ap.add_argument('checkpoint', help='Trained .pth to load')
    # ----- Data source -----
    ap.add_argument('--data-root',
                    help='Path to <object_class>/ directory (DefectFill2 layout: '
                         'expects <root>/test/good and <root>/train/defective_masks/<type>). '
                         'Optional if --good-dir and --mask-dir are given.')
    ap.add_argument('--good-dir',
                    help='Override path to good images directory. Bypasses the '
                         'DefectFill2 layout assumption.')
    ap.add_argument('--mask-dir',
                    help='Override path to defect-mask templates directory.')
    ap.add_argument('--object-class', required=True,
                    help='Used for pairing-mode fallback and for the prompt.')
    ap.add_argument('--defect-type', required=True,
                    help='Used for prompt and naming only.')
    # ----- Run / output -----
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--total-images', type=int, default=8)
    ap.add_argument('--num-samples', type=int, default=8)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--guidance-scale', type=float, default=2.0)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--dilate-mask', action='store_true')
    ap.add_argument('--mask-kernel-size', type=int, default=3)
    return ap.parse_args()


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise IOError(f'Failed to read {path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f'Failed to read mask {path}')
    return m


def main() -> None:
    args = parse_args()
    init_default_scope('mmdet')

    # ---- build model from cfg, then load checkpoint via the same hook
    #      that `tools/test.py` uses --------------------------------------
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained_model_path = os.path.abspath(cfg.model.pretrained_model_path) \
        if not os.path.isabs(cfg.model.pretrained_model_path) \
        else cfg.model.pretrained_model_path
    model = MODELS.build(cfg.model)
    state = torch.load(args.checkpoint, map_location='cpu')
    sd = state.get('state_dict', state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f'[load] {len(unexpected)} unexpected keys (should be 0 after the '
              f'load_state_dict trigger hook was added)')
    if missing and missing != ['_dummy_param']:
        print(f'[load] missing keys (non-dummy): {missing[:5]}...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    # ---- prep output dir + log -------------------------------------------
    out_root = Path(args.output_dir)
    defect_out = out_root / args.defect_type
    defect_out.mkdir(parents=True, exist_ok=True)
    inference_log = {
        'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        'checkpoint': os.path.abspath(args.checkpoint),
        'config': os.path.abspath(args.config),
        'object_class': args.object_class,
        'defect_type': args.defect_type,
        'data_root': os.path.abspath(args.data_root) if args.data_root else None,
        'good_dir': os.path.abspath(args.good_dir) if args.good_dir else None,
        'mask_dir': os.path.abspath(args.mask_dir) if args.mask_dir else None,
        'num_samples': args.num_samples,
        'steps': args.steps,
        'guidance_scale': args.guidance_scale,
        'results': [],
    }

    # ---- pairing -----------------------------------------------------------
    # Mode 1: --good-dir + --mask-dir supplied (bypass layout check)
    # Mode 2: 1:1 if test/good + test/masks exist
    # Mode 3: index-based from <root>/test/good + <root>/{train,test}/defective_masks/<type>
    plan: list = []
    if args.good_dir and args.mask_dir:
        good_files = sorted(
            [f for f in os.listdir(args.good_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        mask_files = sorted(
            [f for f in os.listdir(args.mask_dir)
             if f.lower().endswith(('.png', '.PNG'))])
        if not good_files or not mask_files:
            raise RuntimeError(
                f'--good-dir or --mask-dir is empty: '
                f'{args.good_dir} ({len(good_files)} files), '
                f'{args.mask_dir} ({len(mask_files)} files)')
        gen_plan = calculate_generation_plan(
            len(good_files), len(mask_files), target_total=args.total_images)
        plan = [(os.path.join(args.good_dir, good_files[g]),
                 os.path.join(args.mask_dir, mask_files[m]))
                for g, m, _ in gen_plan]
        print(f'Explicit mode: {len(plan)} pairs from '
              f'{args.good_dir} + {args.mask_dir}')
    else:
        if not args.data_root:
            raise RuntimeError(
                'Either --data-root or both --good-dir and --mask-dir are required.')
        pairs = get_image_mask_pairs(args.data_root, args.object_class)
        if pairs:
            plan = pairs[: args.total_images]
            print(f'Paired mode: {len(plan)} image-mask pairs from '
                  f'{args.data_root}/test/good + .../test/masks')
        else:
            num_good, num_masks, good_dir, mask_dir = count_available_resources(
                args.data_root, args.object_class, args.defect_type)
            if num_good == 0 or num_masks == 0:
                raise RuntimeError(
                    f'No usable image/mask under {args.data_root}: need test/good '
                    f'+ (test/masks or train/defective_masks/{args.defect_type}). '
                    f'Alternatively, pass --good-dir and --mask-dir explicitly.')
            good_files = sorted(
                [f for f in os.listdir(good_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            mask_files = sorted(
                [f for f in os.listdir(mask_dir)
                 if f.lower().endswith(('.png', '.PNG'))])
            gen_plan = calculate_generation_plan(
                len(good_files), len(mask_files), target_total=args.total_images)
            plan = [(os.path.join(good_dir, good_files[g]),
                     os.path.join(mask_dir, mask_files[m]))
                    for g, m, _ in gen_plan]
            print(f'Index-based mode: {len(plan)} pairs from {good_dir} + {mask_dir}')

    # ---- main loop -------------------------------------------------------
    for output_idx, (img_path, mask_path) in enumerate(plan):
        print(f'\n[{output_idx + 1}/{len(plan)}] {os.path.basename(img_path)}')
        try:
            image_np = load_image(img_path)
            mask_np = load_mask(mask_path)
        except IOError as e:
            print(f'  skip: {e}')
            continue

        if mask_np.shape[:2] != image_np.shape[:2]:
            mask_np = cv2.resize(
                mask_np, (image_np.shape[1], image_np.shape[0]),
                interpolation=cv2.INTER_NEAREST)
        if args.dilate_mask:
            k = args.mask_kernel_size if args.mask_kernel_size % 2 else args.mask_kernel_size + 1
            mask_np = cv2.dilate(mask_np, np.ones((k, k), np.uint8), iterations=1)

        crop_img, crop_mask = smart_crop_dynamic(image_np, mask_np, base_size=args.image_size)

        img_t = torch.from_numpy(crop_img.astype(np.float32) / 127.5 - 1.0) \
            .permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32)
        mask_t = torch.from_numpy(crop_mask.astype(np.float32) / 255.0) \
            .unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)

        best_image, best_idx, all_scores = model.generate_defect(
            image=img_t,
            mask=mask_t,
            object_class=args.object_class,
            num_samples=args.num_samples,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            batch_size=args.batch_size,
            seed=args.seed + output_idx,
        )

        save_image((best_image.float() + 1) / 2,
                   str(defect_out / f'{output_idx:04d}_generated.png'))
        save_image((img_t.float() + 1) / 2,
                   str(defect_out / f'{output_idx:04d}_original.png'))
        save_image(mask_t.float(),
                   str(defect_out / f'{output_idx:04d}_mask.png'))

        inference_log['results'].append({
            'output_idx': output_idx,
            'input_image': os.path.abspath(img_path),
            'input_mask': os.path.abspath(mask_path),
            'best_idx': int(best_idx),
            'best_lpips': float(all_scores[best_idx].item()),
            'all_lpips': [float(s.item()) for s in all_scores],
        })
        if output_idx % 4 == 0:
            torch.cuda.empty_cache()

    log_path = out_root / 'inference_log.json'
    with open(log_path, 'w') as f:
        json.dump(inference_log, f, indent=4)
    print(f'\nWrote {len(inference_log["results"])} samples to {defect_out}')
    print(f'Log saved to {log_path}')


if __name__ == '__main__':
    main()
