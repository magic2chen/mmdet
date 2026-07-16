#!/usr/bin/env python
"""SegAD inference using mmdeploy for EfficientAD + XGBoost pickle for final prediction.

This script demonstrates the two-stage deployment:
1. Stage 1: EfficientAD ONNX inference via mmdeploy
2. Stage 2: XGBoost pickle inference for final classification

Usage:
    python projects/csy_segad/segad_inference.py --img <image_path>
    python projects/csy_segad/segad_inference.py --img <image_path> --efficientad_onnx deploy_efficientad/end2end.onnx
"""
import argparse
import os
import numpy as np
from PIL import Image
import pickle
from scipy.stats import kurtosis, skew

# Paths
def get_default_paths():
    """Get default paths based on environment."""
    is_wsl = os.path.exists('/mnt/d')
    if is_wsl:
        base = '/mnt/d/csy/mmdet'
    else:
        base = 'D:/csy/mmdet'
    return base

def to_linux_path(win_path):
    """Convert Windows path to Linux path for WSL."""
    if ':' in win_path:
        drive = win_path.split(':')[0].lower()
        rest = win_path.split(':', 1)[1].replace('\\', '/')
        return f'/mnt/{drive}{rest}'
    return win_path


class FeatureExtractor:
    """Extract statistical features from anomaly maps based on segmentation maps."""

    def __init__(self, num_components, models_list):
        self.num_components = num_components
        self.models_list = models_list
        self.components = [chr(ord('a') + i) for i in range(num_components)]
        self.feature_names = self._get_feature_names()

    def _get_feature_names(self):
        features = ["_q995", "_scewness", "_kurtosis", "_mean"]
        lst = [c + "_" + m for c in self.components for m in self.models_list]
        list_features = [comp_model + f for comp_model in lst for f in features]
        for model in self.models_list:
            list_features.append("an_det_score_" + model)
        return list_features

    @staticmethod
    def extract_features_from_part(part, model, selection):
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

    def extract_features(self, segm_map, anomaly_maps):
        features = {}

        if segm_map.ndim == 1:
            size = int(np.sqrt(segm_map.size))
            if size * size == segm_map.size:
                segm_map = segm_map.reshape(size, size)

        for model in self.models_list:
            anomaly_map = anomaly_maps[model]

            if anomaly_map.ndim == 1:
                size = int(np.sqrt(anomaly_map.size))
                if size * size == anomaly_map.size:
                    anomaly_map = anomaly_map.reshape(size, size)

            if anomaly_map.shape != segm_map.shape:
                from scipy.ndimage import zoom
                zoom_factors = (segm_map.shape[0] / anomaly_map.shape[0],
                                segm_map.shape[1] / anomaly_map.shape[1])
                anomaly_map = zoom(anomaly_map, zoom_factors, order=1)

            for j, part in enumerate(self.components):
                selection = anomaly_map[segm_map == j]
                part_features = self.extract_features_from_part(part, model, selection)
                features.update(part_features)
            features[f"an_det_score_{model}"] = float(np.max(anomaly_map))

        # Return as ordered array matching feature_names
        return np.array([features[name] for name in self.feature_names], dtype=np.float32)


class SegADDeploy:
    """SegAD deployment: EfficientAD + XGBoost inference."""

    def __init__(self,
                 efficientad_dir,
                 efficientad_trt_dir,
                 xgboost_pkl,
                 segm_map_good_dir,
                 segm_map_bad_dir,
                 model_cfg,
                 deploy_cfg_onnx,
                 deploy_cfg_trt=None,
                 num_components=1,
                 models_list=None,
                 backend='onnx',
                 device='cuda:0'):
        """Initialize SegAD deployment.

        Args:
            efficientad_dir: Directory containing EfficientAD ONNX/engine files
            xgboost_pkl: Path to XGBoost pickle model
            segm_map_good_dir: Directory containing good segmentation maps
            segm_map_bad_dir: Directory containing bad segmentation maps
            model_cfg: Path to model config
            deploy_cfg_onnx: Path to ONNX deploy config
            deploy_cfg_trt: Path to TensorRT deploy config
            num_components: Number of components in segmentation map
            models_list: List of base models (default: ['efficient_ad'])
            backend: Backend type ('onnx' or 'tensorrt')
            device: Device for inference
        """
        self.num_components = num_components
        self.models_list = models_list or ['efficient_ad']
        self.backend = backend
        self.device = device
        self.segm_map_good_dir = segm_map_good_dir
        self.segm_map_bad_dir = segm_map_bad_dir

        # Feature extractor
        self.feature_extractor = FeatureExtractor(num_components, self.models_list)

        # Load XGBoost model
        print(f"Loading XGBoost model from {xgboost_pkl}...")
        with open(xgboost_pkl, 'rb') as f:
            self.xgboost = pickle.load(f)
        print("XGBoost model loaded successfully")

        # Determine backend file
        if backend == 'tensorrt':
            backend_file = os.path.join(efficientad_trt_dir, 'end2end.engine')
            deploy_cfg = deploy_cfg_trt or deploy_cfg_onnx
        else:
            backend_file = os.path.join(efficientad_dir, 'end2end.onnx')
            deploy_cfg = deploy_cfg_onnx

        if not os.path.exists(backend_file):
            raise FileNotFoundError(f"Backend file not found: {backend_file}")

        # Load EfficientAD via mmdeploy
        print(f"Loading EfficientAD model from {backend_file}...")
        from mmdeploy.apis import inference_model
        self.inference_model = inference_model
        self.model_cfg = model_cfg
        self.deploy_cfg = deploy_cfg
        self.backend_file = backend_file
        print(f"EfficientAD model loaded successfully (backend: {backend})")

    def load_segm_map(self, img_name, is_good):
        """Load segmentation map for given image name."""
        segm_dir = self.segm_map_good_dir if is_good else self.segm_map_bad_dir
        segm_path = os.path.join(segm_dir, img_name + '.npy')
        if not os.path.exists(segm_path):
            # Try the other directory
            alt_dir = self.segm_map_bad_dir if is_good else self.segm_map_good_dir
            segm_path = os.path.join(alt_dir, img_name + '.npy')
        if not os.path.exists(segm_path):
            raise FileNotFoundError(f"Segmentation map not found: {img_name}")
        return np.load(segm_path)

    def preprocess_image(self, img_path):
        """Preprocess image for EfficientAD."""
        from torchvision import transforms
        img = Image.open(img_path).convert('RGB').resize((256, 256))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(img).unsqueeze(0)

    def forward(self, img_path, img_name=None, is_good=True):
        """Run full SegAD inference.

        Args:
            img_path: Path to input image
            img_name: Name for loading segmentation map (default: extracted from filename)
            is_good: Whether the image is from good class (determines segm map directory)

        Returns:
            dict with keys: 'anomaly_score', 'prediction', 'anomaly_map', 'features'
        """
        if img_name is None:
            img_name = os.path.splitext(os.path.basename(img_path))[0]

        # Stage 1: EfficientAD inference via mmdeploy
        result = self.inference_model(
            model_cfg=self.model_cfg,
            deploy_cfg=self.deploy_cfg,
            backend_files=[self.backend_file],
            img=img_path,
            device=self.device
        )

        # Extract anomaly map from result
        pred = result[0]
        if hasattr(pred, 'pred_anomaly_map'):
            anomaly_map = pred.pred_anomaly_map
        elif hasattr(pred, 'outputs'):
            anomaly_map = pred.outputs
        else:
            anomaly_map = getattr(pred, 'tensor', None)

        if anomaly_map is None:
            raise ValueError("Cannot find anomaly map in EfficientAD result")

        # Convert to numpy
        if hasattr(anomaly_map, 'cpu'):
            anomaly_map = anomaly_map.cpu().numpy()
        elif not isinstance(anomaly_map, np.ndarray):
            anomaly_map = np.array(anomaly_map)

        # Ensure 2D
        if anomaly_map.ndim == 4:
            anomaly_map = anomaly_map.squeeze()
        if anomaly_map.ndim == 3:
            anomaly_map = anomaly_map.squeeze(0)

        anomaly_map = anomaly_map.astype(np.float32)

        # Stage 2: Load segmentation map and extract features
        segm_map = self.load_segm_map(img_name, is_good)

        # Extract features
        anomaly_maps = {'efficient_ad': anomaly_map}
        features = self.feature_extractor.extract_features(segm_map, anomaly_maps)
        features = features.reshape(1, -1)

        # Stage 3: XGBoost prediction
        raw_output = self.xgboost.predict_proba(features)[0]

        # XGBoost with binary:logitraw returns raw logits, not probabilities
        # Apply sigmoid to convert to probability
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        proba = np.array([sigmoid(-raw_output[1]), sigmoid(raw_output[1])])
        anomaly_score = float(proba[1])

        return {
            'img_path': img_path,
            'img_name': img_name,
            'anomaly_score': anomaly_score,
            'prediction': "Defective" if anomaly_score > 0.5 else "Normal",
            'anomaly_map': anomaly_map,
            'features': features[0],
            'probabilities': proba
        }

    def forward_batch(self, img_paths, is_good_list=None):
        """Run batch inference.

        Args:
            img_paths: List of image paths
            is_good_list: List of bools indicating if each image is good

        Returns:
            List of result dicts
        """
        if is_good_list is None:
            is_good_list = [True] * len(img_paths)

        results = []
        for i, (img_path, is_good) in enumerate(zip(img_paths, is_good_list)):
            print(f"[{i+1}/{len(img_paths)}] Processing {os.path.basename(img_path)}...")
            try:
                result = self.forward(img_path, is_good=is_good)
                results.append(result)
            except Exception as e:
                print(f"  Error: {e}")
                results.append({
                    'img_path': img_path,
                    'img_name': os.path.splitext(os.path.basename(img_path))[0],
                    'error': str(e)
                })

        return results


def get_args():
    parser = argparse.ArgumentParser(description='SegAD inference')
    base = get_default_paths()

    parser.add_argument('--img', type=str, default=None,
                        help='Path to input image (for single inference)')
    parser.add_argument('--img_dir', type=str, action='append', default=None,
                        help='Directory containing images (for batch inference). Can be specified multiple times.')
    parser.add_argument('--img_list', type=str, nargs='+', default=None,
                        help='List of image paths (for batch inference)')
    parser.add_argument('--backend', type=str, default='onnx',
                        choices=['onnx', 'tensorrt'],
                        help='Backend for EfficientAD: onnx or tensorrt')
    parser.add_argument('--efficientad_dir', type=str,
                        default=to_linux_path(f'{base}/deploy_efficientad'),
                        help='Directory containing EfficientAD ONNX models')
    parser.add_argument('--efficientad_trt_dir', type=str,
                        default=to_linux_path(f'{base}/deploy_efficientad_trt'),
                        help='Directory containing EfficientAD TensorRT engine')
    parser.add_argument('--xgboost_pkl', type=str,
                        default=to_linux_path(f'{base}/work_dirs/segad_bottle_train/xgb_model_bottle_seed_333.pkl'),
                        help='Path to XGBoost pickle model')
    parser.add_argument('--segm_map_good_dir', type=str,
                        default=to_linux_path(f'{base}/work_dirs/segad_bottle/segmentation_maps/good'),
                        help='Directory containing good segmentation maps')
    parser.add_argument('--segm_map_bad_dir', type=str,
                        default=to_linux_path(f'{base}/work_dirs/segad_bottle/segmentation_maps/bad'),
                        help='Directory containing bad segmentation maps')
    parser.add_argument('--model_cfg', type=str,
                        default=to_linux_path(f'{base}/projects/csy_efficientad/configs/efficientad_small.py'),
                        help='Path to model config')
    parser.add_argument('--deploy_cfg_onnx', type=str,
                        default=to_linux_path(f'{base}/mmdeploy/configs/mmanomaly/anomaly_detection_onnxruntime_dynamic.py'),
                        help='Path to ONNX deploy config')
    parser.add_argument('--deploy_cfg_trt', type=str,
                        default=to_linux_path(f'{base}/mmdeploy/configs/mmanomaly/anomaly_detection_tensorrt_dynamic.py'),
                        help='Path to TensorRT deploy config')
    parser.add_argument('--num_components', type=int, default=1,
                        help='Number of components in segmentation map')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for inference (cpu or cuda:0)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for classification')

    return parser.parse_args()


def main():
    args = get_args()

    print("=" * 60)
    print("SegAD Inference")
    print("=" * 60)
    print(f"Backend: {args.backend}")
    print(f"EfficientAD ONNX dir: {args.efficientad_dir}")
    print(f"EfficientAD TRT dir: {args.efficientad_trt_dir}")
    print(f"XGBoost pickle: {args.xgboost_pkl}")
    print(f"Segm good dir: {args.segm_map_good_dir}")
    print(f"Segm bad dir: {args.segm_map_bad_dir}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Initialize SegAD deploy
    segad = SegADDeploy(
        efficientad_dir=args.efficientad_dir,
        efficientad_trt_dir=args.efficientad_trt_dir,
        xgboost_pkl=args.xgboost_pkl,
        segm_map_good_dir=args.segm_map_good_dir,
        segm_map_bad_dir=args.segm_map_bad_dir,
        model_cfg=args.model_cfg,
        deploy_cfg_onnx=args.deploy_cfg_onnx,
        deploy_cfg_trt=args.deploy_cfg_trt,
        num_components=args.num_components,
        models_list=['efficient_ad'],
        backend=args.backend,
        device=args.device
    )

    # Collect image paths
    img_paths = []
    is_good_list = []

    if args.img:
        img_paths.append(args.img)
        is_good_list.append(True)  # Default to good if not specified

    if args.img_dir:
        # Scan directories for images
        # args.img_dir is a list when using action='append'
        img_dirs = args.img_dir if isinstance(args.img_dir, list) else [args.img_dir]
        for img_dir in img_dirs:
            # Determine if good based on directory path, not filename
            is_good_dir = 'good' in img_dir
            for fname in sorted(os.listdir(img_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_paths.append(os.path.join(img_dir, fname))
                    is_good_list.append(is_good_dir)

    if args.img_list:
        img_paths.extend(args.img_list)
        is_good_list.extend([True] * len(args.img_list))

    if not img_paths:
        print("Error: No images specified. Use --img, --img_dir, or --img_list")
        return

    # Run inference
    print(f"\nProcessing {len(img_paths)} image(s)...\n")

    if len(img_paths) == 1:
        # Single image inference
        result = segad.forward(img_paths[0], is_good=is_good_list[0])

        print("=" * 60)
        print("Results:")
        print(f"  Image: {result['img_name']}")
        print(f"  Anomaly Score: {result['anomaly_score']:.4f}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Probabilities: Normal={result['probabilities'][0]:.4f}, Defective={result['probabilities'][1]:.4f}")
        print("=" * 60)

        # Show top features
        features = result['features']
        feature_names = segad.feature_extractor.feature_names
        sorted_idx = np.argsort(features)[::-1]
        print("Top features:")
        for i in range(min(5, len(sorted_idx))):
            idx = sorted_idx[i]
            print(f"  {feature_names[idx]}: {features[idx]:.4f}")
    else:
        # Batch inference
        results = segad.forward_batch(img_paths, is_good_list)

        # Print summary
        print("\n" + "=" * 60)
        print("Batch Results Summary:")
        print("=" * 60)

        good_scores = []
        bad_scores = []

        for result in results:
            if 'error' in result:
                print(f"  {result['img_name']}: ERROR - {result['error']}")
                continue

            score = result['anomaly_score']
            pred = result['prediction']
            is_good = is_good_list[img_paths.index(result['img_path'])] if result['img_path'] in img_paths else True

            if is_good:
                good_scores.append(score)
            else:
                bad_scores.append(score)

            status = "✓" if (is_good and pred == "Normal") or (not is_good and pred == "Defective") else "✗"
            print(f"  [{status}] {result['img_name']}: score={score:.4f}, pred={pred}")

        print("-" * 60)
        if good_scores:
            print(f"Good images:   mean={np.mean(good_scores):.4f}, min={np.min(good_scores):.4f}, max={np.max(good_scores):.4f}")
        if bad_scores:
            print(f"Defect images: mean={np.mean(bad_scores):.4f}, min={np.min(bad_scores):.4f}, max={np.max(bad_scores):.4f}")
        print("=" * 60)


if __name__ == '__main__':
    main()
