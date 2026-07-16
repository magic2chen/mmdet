#!/usr/bin/env python
"""Prepare bottle dataset for SegAD training.

This script:
1. Converts ground_truth masks to segmentation maps (.npy)
2. Generates anomaly maps using existing EfficientAD model
3. Creates CSV files for training/testing
"""
import os
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import torch

# Paths
DATA_ROOT = 'ck4efficientad/bottle'
SEG_OUTPUT = 'work_dirs/segad_bottle/segmentation_maps'
AN_OUTPUT = 'work_dirs/segad_bottle/anomaly_maps'
WORK_DIR = 'work_dirs/segad_bottle'

# EfficientAD model path
MODEL_PATH = 'work_dirs/efficientad_small/iter_30000.pth'
IMAGE_SIZE = 256  # EfficientAD image size


def convert_ground_truth_to_segmentation():
    """Convert bottle ground_truth masks to SegAD segmentation maps."""
    print("Converting ground_truth to segmentation maps...")

    os.makedirs(SEG_OUTPUT, exist_ok=True)
    os.makedirs(os.path.join(SEG_OUTPUT, 'good'), exist_ok=True)
    os.makedirs(os.path.join(SEG_OUTPUT, 'bad'), exist_ok=True)

    # Process train/good images - segmentation is 1 for entire image
    train_good = os.path.join(DATA_ROOT, 'train', 'good')
    if os.path.exists(train_good):
        for img_name in tqdm(os.listdir(train_good), desc='Train good segmentation'):
            if img_name.endswith('.png'):
                # For good images, segmentation is all 1s (one component)
                img = Image.open(os.path.join(train_good, img_name)).convert('L')
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                seg_map = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
                save_path = os.path.join(SEG_OUTPUT, 'good', img_name.replace('.png', '.npy'))
                np.save(save_path, seg_map)

    # Process test images
    for defect_type in ['good', 'broken_large', 'broken_small', 'contamination']:
        gt_dir = os.path.join(DATA_ROOT, 'ground_truth', defect_type)
        test_dir = os.path.join(DATA_ROOT, 'test', defect_type)
        out_dir = os.path.join(SEG_OUTPUT, 'bad' if defect_type != 'good' else 'good')
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(gt_dir):
            continue

        for mask_name in tqdm(os.listdir(gt_dir), desc=f'Test {defect_type} segmentation'):
            if not mask_name.endswith('_mask.png'):
                continue

            # Load mask and resize
            mask = Image.open(os.path.join(gt_dir, mask_name)).convert('L')
            mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE))
            mask_arr = np.array(mask)

            # Convert to segmentation: 0=background, 1=component
            # Mask is 0 for good areas, 255 for defect areas
            # For SegAD, we want: 0=background, 1=foreground component
            seg_map = (mask_arr > 127).astype(np.uint8)

            # Get corresponding test image name
            base_name = mask_name.replace('_mask.png', '.png')
            save_path = os.path.join(out_dir, base_name.replace('.png', '.npy'))
            np.save(save_path, seg_map)

    print(f"Segmentation maps saved to {SEG_OUTPUT}")


def generate_anomaly_maps():
    """Generate anomaly maps using existing EfficientAD model."""
    print("Generating anomaly maps with EfficientAD...")

    os.makedirs(AN_OUTPUT, exist_ok=True)
    os.makedirs(os.path.join(AN_OUTPUT, 'good'), exist_ok=True)
    os.makedirs(os.path.join(AN_OUTPUT, 'bad'), exist_ok=True)

    # Load EfficientAD model
    model = None
    try:
        import sys
        sys.path.insert(0, '.')
        from projects.csy_efficientad.models.efficientad import EfficientADModel
        from mmengine.config import Config

        cfg = Config.fromfile('projects/csy_efficientad/configs/efficientad_small.py')
        # cfg.model contains 'type' key which we don't need for constructor
        model_kwargs = {k: v for k, v in cfg.model.items() if k != 'type'}
        # Create model with default teacher checkpoint (we'll override teacher manually)
        model = EfficientADModel(**model_kwargs)

        # Load full model checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Extract teacher state dict and load into teacher
        teacher_state_dict = {k.replace('teacher.', ''): v
                              for k, v in state_dict.items()
                              if k.startswith('teacher.')}
        model.teacher.load_state_dict(teacher_state_dict, strict=True)
        model.eval()
        print(f"Loaded EfficientAD teacher from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading EfficientAD model: {e}")
        import traceback
        traceback.print_exc()
        print("Will use dummy anomaly maps for testing")
        model = None

    def process_images(image_dir, output_dir, label):
        """Process images and generate anomaly maps."""
        if not os.path.exists(image_dir):
            return

        for img_name in tqdm(os.listdir(image_dir), desc=f'Processing {label}'):
            if not img_name.endswith('.png'):
                continue

            img_path = os.path.join(image_dir, img_name)
            img = Image.open(img_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))

            if model is not None:
                # Convert to tensor
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                img_tensor = transform(img).unsqueeze(0)

                with torch.no_grad():
                    # EfficientADModel.forward returns combined anomaly map
                    anomaly_map = model(img_tensor)
                    # Ensure 2D output
                    if anomaly_map.dim() == 4:
                        anomaly_map = anomaly_map.squeeze(0).squeeze(0)
                    elif anomaly_map.dim() == 3:
                        anomaly_map = anomaly_map.squeeze(0)
                    anomaly_map = anomaly_map.cpu().numpy()
            else:
                # Dummy anomaly map for testing
                anomaly_map = np.random.rand(IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)

            save_path = os.path.join(output_dir, img_name.replace('.png', '.npy'))
            np.save(save_path, anomaly_map)

    # Process train/good
    process_images(os.path.join(DATA_ROOT, 'train', 'good'),
                   os.path.join(AN_OUTPUT, 'good'), 'train_good')

    # Process test images
    for defect_type in ['good', 'broken_large', 'broken_small', 'contamination']:
        img_dir = os.path.join(DATA_ROOT, 'test', defect_type)
        out_dir = os.path.join(AN_OUTPUT, 'bad' if defect_type != 'good' else 'good')
        process_images(img_dir, out_dir, f'test_{defect_type}')

    print(f"Anomaly maps saved to {AN_OUTPUT}")


def create_csv_files():
    """Create CSV files for SegAD training."""
    print("Creating CSV files...")

    # Process train
    train_data = []
    train_good_dir = os.path.join(SEG_OUTPUT, 'good')
    train_an_dir = os.path.join(AN_OUTPUT, 'good')

    for seg_name in os.listdir(train_good_dir):
        if not seg_name.endswith('.npy'):
            continue
        base_name = seg_name.replace('.npy', '')
        seg_path = os.path.join(train_good_dir, seg_name)
        an_path = os.path.join(train_an_dir, seg_name)

        if os.path.exists(an_path):
            train_data.append({
                'filepath': f'train/good/{base_name}.png',
                'an_map_path': f'bottle/good/{base_name}.npy',  # Include category
                'label': 0,
                'prediction_an_det': 0.0
            })

    df_train = pd.DataFrame(train_data)
    df_train.to_csv(os.path.join(WORK_DIR, 'df_training.csv'), index=False)
    print(f"Training samples: {len(df_train)}")

    # Process test
    test_data = []
    for defect_type in ['good', 'broken_large', 'broken_small', 'contamination']:
        seg_dir = os.path.join(SEG_OUTPUT, 'bad' if defect_type != 'good' else 'good')
        an_dir = os.path.join(AN_OUTPUT, 'bad' if defect_type != 'good' else 'good')
        label = 0 if defect_type == 'good' else 1

        for seg_name in os.listdir(seg_dir):
            if not seg_name.endswith('.npy'):
                continue
            base_name = seg_name.replace('.npy', '')
            seg_path = os.path.join(seg_dir, seg_name)
            an_path = os.path.join(an_dir, seg_name)

            if os.path.exists(an_path):
                test_data.append({
                    'filepath': f'test/{defect_type}/{base_name}.png',
                    'an_map_path': f'bottle/bad/{base_name}.npy' if label == 1 else f'bottle/good/{base_name}.npy',
                    'label': label,
                    'prediction_an_det': 0.0
                })

    df_test = pd.DataFrame(test_data)
    df_test.to_csv(os.path.join(WORK_DIR, 'df_test.csv'), index=False)
    print(f"Test samples: {len(df_test)} (good: {sum(df_test['label']==0)}, bad: {sum(df_test['label']==1)})")


if __name__ == '__main__':
    convert_ground_truth_to_segmentation()
    generate_anomaly_maps()
    create_csv_files()
    print("\nData preparation complete!")
    print(f"Data saved to: {WORK_DIR}")
