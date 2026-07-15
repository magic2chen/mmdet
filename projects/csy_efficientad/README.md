# EfficientAD 异常检测模型

本项目将 EfficientAD 异常检测模型集成到 MMDetection 框架，支持训练、测试和部署。

## 目录

- [快速开始](#快速开始)
- [数据集准备](#数据集准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [模型部署（ONNX/TensorRT）](#模型部署onnxtensorrt)
- [常见问题](#常见问题)

---

## 快速开始

> **重要说明**：本项目使用修改后的 mmdeploy（位于 `mmdeploy/` 目录），包含 mmanomaly codebase 的完整实现。**不能使用标准 PyPI 版 mmdeploy**，必须使用本项目中的 mmdeploy。

### 1. 环境安装

```bash
# 创建 conda 环境
conda create -n mmlab python=3.10
conda activate mmlab

# 安装 PyTorch (CUDA 12.1)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 安装 MMDetection 和相关依赖
pip install mmdet==3.3.0 mmengine==0.10.0 mmcv==2.1.0 openmim
mim install mmdet mmengine mmcv
```

### 2. 下载预训练模型

下载 EfficientAD 预训练 teacher 模型：

```bash
# 放置在 ck4efficientad/models/ 目录下
ck4efficientad/models/
├── teacher_small.pth   # small 版本
└── teacher_medium.pth  # medium 版本
```

### 3. 快速测试（使用已导出的模型）

```bash
# ONNX 推理
python projects/csy_efficientad/mmdeploy_inference.py \
    --backend onnx \
    --img ck4efficientad/bottle/test/good/000.png

# TensorRT 推理（需要 GPU）
python projects/csy_efficientad/mmdeploy_inference.py \
    --backend tensorrt \
    --img ck4efficientad/bottle/test/good/000.png
```

输出示例：
```
Backend: tensorrt
Device: cuda:0
Anomaly score: 1.5483
Prediction: Normal
```

---

## 数据集准备

### MVTec AD 数据集结构

```
ck4efficientad/
├── models/                    # 预训练模型
│   └── teacher_small.pth
└── bottle/                   # 你的数据集（以 bottle 为例）
    └── test/
        ├── good/             # 正常样本
        │   ├── 000.png
        │   └── 001.png
        ├── broken_large/     # 大面积缺陷
        ├── broken_small/      # 小面积缺陷
        └── contamination/     # 污染缺陷
```

### 配置文件中的数据集路径

在 `projects/csy_efficientad/configs/efficientad_small.py` 中修改：

```python
dataset_root = 'ck4efficientad'      # 数据集根目录
subdataset = 'bottle'                # 子数据集名称
```

支持的子数据集（MVTec AD）：
`bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`, `leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, `zipper`

---

## 模型训练

### 基本训练

```bash
python tools/train.py projects/csy_efficientad/configs/efficientad_small.py
```

### 训练输出

训练过程中会在以下位置保存模型：

```
work_dirs/efficientad_small/
├── iter_10000.pth    # 每 10000 次迭代保存
├── iter_20000.pth
├── ...
├── iter_70000.pth
└── last.pth          # 最后一个 checkpoint

output/trainings/mvtec_ad/bottle/    # EfficientAD 格式
├── teacher_final.pth
├── student_final.pth
└── autoencoder_final.pth
```

### 修改训练配置

通过命令行参数修改配置：

```bash
# 训练不同的子数据集
python tools/train.py \
    projects/csy_efficientad/configs/efficientad_small.py \
    --cfg-options test_dataloader.Data.subdataset=cable
```

---

## 模型测试

### 基本测试

```bash
python tools/test.py \
    projects/csy_efficientad/configs/efficientad_small.py \
    work_dirs/efficientad_small/iter_70000.pth
```

### 保存异常图

```bash
python tools/test.py \
    projects/csy_efficientad/configs/efficientad_small.py \
    work_dirs/efficientad_small/iter_70000.pth \
    --cfg-options test_evaluator.save_dir=output_anomaly_maps/bottle
```

### 输出说明

| 分数范围 | 预测结果 |
|---------|---------|
| < 1.6 | Normal（正常） |
| >= 1.6 | Defective（缺陷） |

测试完成后会输出 AUC 指标，越接近 1越好。

---

## 模型部署（ONNX/TensorRT）

### 导出模型（仅需一次）

> **前提**：必须使用本项目中的 mmdeploy（`mmdeploy/` 目录），不能使用 pip 安装的标准版。

```bash
# 切换到部署环境
conda activate mmlab

# 安装修改后的 mmdeploy
cd mmdeploy && pip install -e . && cd ..

# 导出 ONNX
python mmdeploy/tools/torch2onnx.py \
    mmdeploy/configs/mmanomaly/anomaly_detection_onnxruntime_dynamic.py \
    projects/csy_efficientad/configs/efficientad_small.py \
    work_dirs/efficientad_small/iter_70000.pth \
    ck4efficientad/bottle/test/good/000.png \
    --work-dir ./deploy_efficientad \
    --device cuda:0

# 导出 TensorRT（同时生成 ONNX 和 engine）
python mmdeploy/tools/deploy.py \
    mmdeploy/configs/mmanomaly/anomaly_detection_tensorrt_dynamic.py \
    projects/csy_efficientad/configs/efficientad_small.py \
    work_dirs/efficientad_small/iter_70000.pth \
    ck4efficientad/bottle/test/good/000.png \
    --work-dir ./deploy_efficientad_trt \
    --device cuda:0
```

### 导出后的文件

```
deploy_efficientad/              # ONNX
└── end2end.onnx                 # (~83MB)

deploy_efficientad_trt/          # TensorRT
├── end2end.onnx
└── end2end.engine               # (~43MB)
```

### 使用 mmdeploy 推理

**Python 脚本方式**（推荐）：

```python
from mmdeploy.apis import inference_model

result = inference_model(
    model_cfg='projects/csy_efficientad/configs/efficientad_small.py',
    deploy_cfg='configs/mmanomaly/anomaly_detection_onnxruntime_dynamic.py',  # 或 tensorrt
    backend_files=['deploy_efficientad/end2end.onnx'],  # 或 .engine
    img='ck4efficientad/bottle/test/good/000.png',
    device='cpu'  # 或 'cuda:0'
)

anomaly_map = result[0].pred_anomaly_map
score = anomaly_map.max().item()
print(f'Anomaly score: {score:.4f}')
```

**命令行方式**：

```bash
# ONNX
python projects/csy_efficientad/mmdeploy_inference.py \
    --backend onnx \
    --img ck4efficientad/bottle/test/good/000.png

# TensorRT
python projects/csy_efficientad/mmdeploy_inference.py \
    --backend tensorrt \
    --img ck4efficientad/bottle/test/good/000.png
```

### 不同部署方式对比

| 方式 | 速度 | 硬件要求 | 适用场景 |
|------|------|---------|---------|
| PyTorch 原生 | 慢 | GPU | 训练/调试 |
| ONNX Runtime | 中等 | CPU/GPU | 部署（跨平台） |
| TensorRT | 最快 | NVIDIA GPU | 高性能部署 |

---

## 常见问题

### Q1: 训练需要多长时间？

A: 在 RTX 3090 上，训练 70000 次迭代约需 4-6 小时。

### Q2: 如何选择合适的 checkpoint？

A: 查看 `work_dirs/` 下的 `last.pth`，或选择验证集上 AUC 最高的迭代。

### Q3: Normal 样本分数偏高怎么办？

A: 这是模型训练不足的表现。可以：
1. 继续训练更多迭代
2. 调整阈值（默认 1.6）到更合适的值

### Q4: TensorRT 导出失败？

A: 确保：
1. 已安装 `tensorrt-cu12`
2. 使用正确的配置文件 `mmanomaly/anomaly_detection_tensorrt_dynamic.py`

### Q5: 运行时找不到 mmdeploy 模块？

A: 确保在正确的 conda 环境中运行：
```bash
conda activate mmlab
python your_script.py
```

### Q6: 为什么不能用 pip 安装的标准 mmdeploy？

A: 因为本项目对 mmdeploy 进行了修改：
- 新增 `mmdeploy/codebase/mmanomaly/` （异常检测 codebase）
- 新增 `mmdeploy/configs/mmanomaly/` （部署配置文件）
- 修改 `mmdeploy/mmdeploy/__init__.py` 和 `constants.py`

标准 mmdeploy 没有 `mmanomaly` 模块，无法用于 EfficientAD 部署。

### Q7: 如何正确使用修改后的 mmdeploy？

A: 必须将本项目的 `mmdeploy/` 目录作为 Python 包安装：

```bash
cd mmdeploy
pip install -e .
cd ..
```

---

## 目录结构

```
projects/csy_efficientad/
├── configs/
│   └── efficientad_small.py      # 训练配置文件
├── models/
│   └── efficientad.py            # 模型定义
├── mmdeploy_inference.py          # 推理脚本（推荐使用）
├── test_onnx.py                   # ONNX 推理（旧版）
└── test_trt.py                    # TensorRT 推理（旧版）

mmdeploy/
├── configs/mmanomaly/
│   ├── anomaly_detection_onnxruntime_dynamic.py
│   ├── anomaly_detection_tensorrt_dynamic.py
│   └── test_onnx_mmdeploy.py
│   └── test_trt_mmdeploy.py
└── mmdeploy/codebase/mmanomaly/   # mmdeploy 集成代码
```
