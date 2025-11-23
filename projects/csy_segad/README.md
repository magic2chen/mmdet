# SegAD: Supervised Anomaly Detection for Complex Industrial Images

This is an MMDetection integration of the SegAD model from the CVPR 2024 paper: [Supervised Anomaly Detection for Complex Industrial Images](https://arxiv.org/abs/2405.04953)

## Description

SegAD combines segmentation maps and anomaly maps from base anomaly detection models (EfficientAD, RD4AD) to perform supervised anomaly detection using XGBoost. This implementation integrates SegAD into the MMDetection framework.

## Installation

### Requirements

The project requires the following dependencies (in addition to MMDetection):

```bash
numpy>=1.24.4
pandas>=2.0.3
scikit-learn>=1.5.0
scipy>=1.10.1
xgboost>=2.0.3
```

### Data Preparation

1. Download [segmentation maps for VisA](https://drive.google.com/file/d/1ZVMxtb6PY958qigxAQcLEifWsRdnLaI4/view?usp=sharing).
2. Download [anomaly maps for EfficientAD](https://drive.google.com/file/d/1mknzBIE6Heqfr5_BQIFOojzuPDQG2o_O/view?usp=sharing).
3. Download [anomaly maps for RD4AD](https://drive.google.com/file/d/1Pap5-8x74_AROFRxjcBvIu9XdqvzHMs8/view?usp=sharing).

The data structure should look as follows:

```
data
├── visa_segm
│   ├── candle
│   │   ├── good
│   │   └── bad
│   ├── capsules
│   └── ...
└── anomaly_maps
    ├── efficient_ad
    │   ├── candle
    │   │   ├── anomaly_maps
    │   │   │   ├── good
    │   │   │   └── bad
    │   │   ├── df_training.csv
    │   │   └── df_test.csv
    │   └── ...
    └── rd4ad
        └── ...
```

## Usage

### Training and Evaluation

SegAD is fully integrated into MMDetection and can be trained using the standard `tools/train.py` script:

```bash
# Train using standard MMDetection training script
python tools/train.py \
    projects/csy_segad/configs/segad_efficient_ad.py \
    --work-dir ./work_dirs/segad
```

You can also override configuration parameters via command line:

```bash
python tools/train.py \
    projects/csy_segad/configs/segad_efficient_ad.py \
    --work-dir ./work_dirs/segad \
    --cfg-options train_cfg.category=pcb1 train_cfg.bad_parts=10
```

**Note**: The custom training loop (`SegADTrainLoop`) handles XGBoost training internally, so you don't need to use a separate training script.

### Configuration

The configuration file `configs/segad_efficient_ad.py` contains:

- Model configuration (number of components, XGBoost parameters)
- Dataset paths (segmentation maps, anomaly maps)
- Category-specific settings

You can modify the configuration to:
- Change the base anomaly detection model (`efficient_ad`, `rd4ad`, or `all_ad`)
- Adjust XGBoost hyperparameters
- Set different data paths

### Available Models

- `efficient_ad`: Use EfficientAD anomaly maps
- `rd4ad`: Use RD4AD anomaly maps
- `all_ad`: Use both EfficientAD and RD4AD anomaly maps

### Available Categories

The VisA dataset includes the following categories:
- `candle`, `capsules`, `cashew`, `chewinggum`, `fryum`
- `macaroni1`, `macaroni2`
- `pcb1`, `pcb2`, `pcb3`, `pcb4`
- `pipe_fryum`

## Code Structure

```
csy_segad/
├── __init__.py
├── models/
│   ├── __init__.py
│   └── segad.py              # SegAD model class
├── datasets/
│   ├── __init__.py
│   └── segad_dataset.py      # Dataset class
├── metrics/
│   ├── __init__.py
│   └── segad_metric.py       # Evaluation metrics
├── runner/
│   ├── __init__.py
│   └── segad_train_loop.py  # Custom training loop for XGBoost
├── utils/
│   ├── __init__.py
│   └── feature_extractor.py  # Feature extraction utilities
├── configs/
│   └── segad_efficient_ad.py # Configuration file
└── README.md
```

## Results

Class-level AUROC (image-level) for SegAD with different sources of anomaly maps:

| Model            | Mean  | Candle | Capsules | Cashew | Chewinggum | Fryum | Macaroni1 | Macaroni2 | PCB1 | PCB2 | PCB3 | PCB4 | Pipe_fryum |
|------------------|:-----:|:------:|:--------:|:------:|:----------:|:-----:|:---------:|:---------:|:----:|:----:|:----:|:----:|:----------:|
| RD4AD + SegAD    | 95.3  | 98.5   | 80.2     | 98.9   | 99.4       | 96.1  | 97.4      | 90.7      | 96.4 | 96.3 | 94.1 | 99.9 | 95.8       |
| EfficientAD + SegAD | 98.3 | 98.7   | 89.7     | 98.6   | 99.9       | 98.6  | 99.5      | 98.1      | 99.5 | 99.7 | 98.4 | 99.3 | 99.2       |
| All AD + SegAD   | 98.4  | 99.0   | 90.7     | 99.0   | 99.9       | 98.5  | 99.4      | 98.1      | 99.2 | 99.7 | 98.3 | 99.8 | 99.1       |

## Key Features

1. **MMDetection Integration**: Fully integrated with MMDetection framework
2. **Modular Design**: Separate modules for models, datasets, metrics, and utilities
3. **Flexible Configuration**: Easy to configure via config files
4. **Multiple Base Models**: Support for EfficientAD, RD4AD, or both

## Differences from Original Implementation

- Integrated into MMDetection framework
- Uses MMDetection's dataset and metric registration system
- Modular code structure following MMDetection conventions
- Custom training loop (`SegADTrainLoop`) that integrates XGBoost training into MMEngine's training pipeline
- Can be trained using standard `tools/train.py` script (no need for separate training script)

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{baitieva2024supervised,
    title={Supervised Anomaly Detection for Complex Industrial Images}, 
    author={Aimira Baitieva and David Hurych and Victor Besnier and Olivier Bernard},
    booktitle={CVPR},
    year={2024},
    pages={17754-17762}
}
```

## Acknowledgement

We use [EfficientAD](https://github.com/nelson1425/EfficientAD) and [Anomalib](https://github.com/openvinotoolkit/anomalib/tree/main) for baseline anomaly detection models. We are thankful for their amazing work!

