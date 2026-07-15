#!/usr/bin/env python
"""EfficientAD inference using mmdeploy inference_model API.

This script demonstrates the unified mmdeploy inference interface for ONNX and TensorRT.
No need to worry about backend-specific details - just change config and model files.

Usage:
    python projects/csy_efficientad/mmdeploy_inference.py --backend onnx
    python projects/csy_efficientad/mmdeploy_inference.py --backend tensorrt
"""
import argparse
import os


def to_linux_path(win_path):
    """Convert Windows path to Linux path for WSL."""
    if ':' in win_path:
        drive = win_path.split(':')[0].lower()
        rest = win_path.split(':')[1].replace('\\', '/')
        return f'/mnt/{drive}{rest}'
    return win_path


def get_args():
    parser = argparse.ArgumentParser(description='EfficientAD mmdeploy inference')
    parser.add_argument('--backend', type=str, default='onnx',
                        choices=['onnx', 'tensorrt'],
                        help='Backend to use: onnx or tensorrt')
    parser.add_argument('--img', type=str,
                        default='/mnt/d/csy/mmdet/ck4efficientad/bottle/test/good/000.png',
                        help='Path to input image')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: cpu for ONNX, cuda:0 for TensorRT)')
    return parser.parse_args()


def main():
    args = get_args()

    # Detect if running in WSL and convert paths
    is_wsl = os.path.exists('/mnt/d')

    def resolve_path(p):
        """Resolve path for current environment."""
        if is_wsl and ':' in p:
            return to_linux_path(p)
        return p

    # Base paths
    if is_wsl:
        base = '/mnt/d/csy/mmdet'
    else:
        base = 'D:/csy/mmdet'

    # Model config (same for both backends)
    model_cfg = resolve_path(f'{base}/projects/csy_efficientad/configs/efficientad_small.py')

    if args.backend == 'onnx':
        deploy_cfg = resolve_path(f'{base}/mmdeploy/configs/mmanomaly/anomaly_detection_onnxruntime_dynamic.py')
        backend_files = [resolve_path(f'{base}/deploy_efficientad/end2end.onnx')]
        device = args.device or 'cpu'
    else:  # tensorrt
        deploy_cfg = resolve_path(f'{base}/mmdeploy/configs/mmanomaly/anomaly_detection_tensorrt_dynamic.py')
        backend_files = [resolve_path(f'{base}/deploy_efficientad_trt/end2end.engine')]
        device = args.device or 'cuda:0'

    # Resolve input image path
    img_path = resolve_path(args.img)

    print(f'Backend: {args.backend}')
    print(f'Device: {device}')
    print(f'Model config: {model_cfg}')
    print(f'Deploy config: {deploy_cfg}')
    print(f'Backend file: {backend_files[0]}')
    print(f'Image: {img_path}')
    print()

    # Import mmdeploy
    from mmdeploy.apis import inference_model

    # Run inference
    result = inference_model(
        model_cfg=model_cfg,
        deploy_cfg=deploy_cfg,
        backend_files=backend_files,
        img=img_path,
        device=device
    )

    # Get anomaly map from result
    pred = result[0]
    if hasattr(pred, 'pred_anomaly_map'):
        anomaly_map = pred.pred_anomaly_map
    elif hasattr(pred, 'outputs'):
        anomaly_map = pred.outputs
    else:
        anomaly_map = getattr(pred, 'tensor', None)

    if anomaly_map is None:
        print('Error: Cannot find anomaly map in result')
        return

    score = anomaly_map.max().item()
    print(f'Anomaly score: {score:.4f}')

    # Interpretation
    if score < 1.0:
        label = 'Normal'
    else:
        label = 'Defective'
    print(f'Prediction: {label}')


if __name__ == '__main__':
    main()
