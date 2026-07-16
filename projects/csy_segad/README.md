# SegAD 异常检测模型

本项目将 SegAD（Supervisor Anomaly Detection）集成到 MMDetection 框架，支持训练和部署。

SegAD 是一种监督式异常检测方法，结合 EfficientAD 生成的异常图和分割图特征，使用 XGBoost 进行最终分类。

## 目录

- [训练部分](#训练部分)
  - [环境安装](#环境安装)
  - [数据准备](#数据准备)
  - [模型训练](#模型训练)
- [部署部分](#部署部分)
  - [部署架构](#部署架构)
  - [模型准备](#模型准备)
  - [推理示例](#推理示例)
- [常见问题](#常见问题)

---

# 训练部分

## 环境安装

```bash
# 创建 conda 环境
conda create -n mmlab python=3.10
conda activate mmlab

# 安装 PyTorch (CUDA 12.1)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 安装 MMDetection 和相关依赖
pip install mmdet==3.3.0 mmengine==0.10.0 mmcv==2.1.0 openmim
mim install mmdet mmengine mmcv

# 安装 SegAD 额外依赖
pip install xgboost scikit-learn scipy pandas
```

## 数据准备

SegAD 训练需要以下数据：

### 数据目录结构

```
work_dirs/segad_bottle/                    # SegAD 数据根目录
├── anomaly_maps/                          # EfficientAD 生成的异常图
│   ├── bottle/good/xxx.npy               # Good 样本的异常图
│   └── bottle/bad/xxx.npy                # Defective 样本的异常图
├── segmentation_maps/                     # 分割图（预计算）
│   ├── good/xxx.npy                      # Good 样本的分割图
│   └── bad/xxx.npy                       # Defective 样本的分割图
├── df_training.csv                        # 训练集 CSV
└── df_test.csv                            # 测试集 CSV
```

### CSV 文件格式

```csv
filepath,an_map_path,label,prediction_an_det
train/good/000.png,bottle/good/000.npy,0,0.0
test/broken_large/000.png,bottle/bad/000.npy,1,0.0
```

| 字段 | 说明 |
|------|------|
| filepath | 原始图像路径 |
| an_map_path | 异常图路径（相对于 anomaly_maps 目录） |
| label | 标签（0=正常, 1=缺陷） |
| prediction_an_det | EfficientAD 预测分数（可为空） |

### 生成数据脚本

使用 `prepare_segad_data.py` 脚本准备 bottle 数据集：

```bash
python projects/csy_segad/prepare_segad_data.py
```

脚本会：
1. 将 ground_truth mask 转换为分割图
2. 使用 EfficientAD 生成异常图
3. 创建 CSV 文件

### 自定义数据集

如需使用其他数据集，需要：

1. 准备分割图（`.npy` 格式，值为组件编号 0, 1, 2...）
2. 使用 EfficientAD 生成异常图
3. 创建 CSV 文件

## 模型训练

### 基本训练

```bash
python tools/train.py projects/csy_segad/configs/segad_bottle.py \
    --work-dir work_dirs/segad_bottle_train
```

### 修改配置

通过命令行参数修改配置：

```bash
# 修改数据路径
python tools/train.py \
    projects/csy_segad/configs/segad_bottle.py \
    --work-dir work_dirs/segad_bottle_train \
    --cfg-options train_dataloader.dataset.an_path=/path/to/anomaly_maps
```

### 训练输出

训练完成后会保存：

```
work_dirs/segad_bottle_train/
├── xgb_model_bottle_seed_333.pkl    # XGBoost 模型
├── xgb_model_bottle_seed_576.pkl
├── ...
├── results.csv                       # 汇总结果
└── results_detailed.csv              # 详细结果
```

### 支持的数据集

| 数据集 | 类别 | 分割图组件数 |
|--------|------|-------------|
| bottle | bottle | 1 |
| dianziyan | dianziyan | 2 |
| VisA | candle, capsules, pcb1, ... | 1-8 |

如需添加新类别，修改 `projects/csy_segad/runner/segad_train_loop.py` 中的 `CATEGORIES` 和 `NUM_COMPONENTS`。

---

# 部署部分

## 部署架构

SegAD 部署采用**两阶段架构**：

```
输入图像
    ↓
┌─────────────────────────────────────┐
│  阶段1：EfficientAD (mmdeploy)       │
│  - ONNX 或 TensorRT 后端            │
│  - 输出异常图                        │
└─────────────────────────────────────┘
    ↓ 异常图
┌─────────────────────────────────────┐
│  阶段2：特征提取 + XGBoost           │
│  - 分割图（预计算 .npy）            │
│  - 提取统计特征（q995, 偏度, 峰度）  │
│  - XGBoost 分类 + sigmoid 校准       │
└─────────────────────────────────────┘
    ↓
最终预测（Normal / Defective）
```

**与 EfficientAD 的区别**：

| 特性 | EfficientAD | SegAD |
|------|-------------|-------|
| 学习方式 | 无监督 | 监督学习 |
| 输入 | 原始图像 | 异常图 + 分割图 |
| 输出 | 异常分数 | 二分类（Normal/Defective） |
| 适用场景 | 简单产品 | 复杂多组件产品 |

## 模型准备

### 1. EfficientAD 模型

```bash
# 导出 ONNX
python mmdeploy/tools/torch2onnx.py \
    mmdeploy/configs/mmanomaly/anomaly_detection_onnxruntime_dynamic.py \
    projects/csy_efficientad/configs/efficientad_small.py \
    work_dirs/efficientad_small/iter_30000.pth \
    ck4efficientad/bottle/test/good/000.png \
    --work-dir ./deploy_efficientad \
    --device cpu

# 导出 TensorRT
python mmdeploy/tools/deploy.py \
    mmdeploy/configs/mmanomaly/anomaly_detection_tensorrt_dynamic.py \
    projects/csy_efficientad/configs/efficientad_small.py \
    work_dirs/efficientad_small/iter_30000.pth \
    ck4efficientad/bottle/test/good/000.png \
    --work-dir ./deploy_efficientad_trt \
    --device cuda:0
```

### 2. XGBoost 模型

训练完成后会生成 pickle 模型：`work_dirs/segad_bottle_train/xgb_model_bottle_seed_333.pkl`

### 3. 目录结构确认

```
deploy_efficientad/                    # EfficientAD ONNX 模型
└── end2end.onnx

deploy_efficientad_trt/                # EfficientAD TensorRT 模型
└── end2end.engine

work_dirs/segad_bottle_train/          # XGBoost 模型
└── xgb_model_bottle_seed_333.pkl

work_dirs/segad_bottle/                # SegAD 数据
├── anomaly_maps/bottle/good/
├── anomaly_maps/bottle/bad/
├── segmentation_maps/good/
└── segmentation_maps/bad/
```

## 推理示例

### 单图推理

```bash
# ONNX 后端（CPU）
python projects/csy_segad/segad_inference.py \
    --img ck4efficientad/bottle/test/good/000.png \
    --backend onnx

# TensorRT 后端（GPU）
python projects/csy_segad/segad_inference.py \
    --img ck4efficientad/bottle/test/good/000.png \
    --backend tensorrt \
    --device cuda:0
```

**输出示例**：
```
============================================================
Results:
  Image: 000
  Anomaly Score: 0.0632
  Prediction: Normal
  Probabilities: Normal=0.9368, Defective=0.0632
============================================================
```

### 批量推理

```bash
# 批量推理 - 同时指定 good 和 bad 目录
python projects/csy_segad/segad_inference.py \
    --img_dir ck4efficientad/bottle/test/good \
    --img_dir ck4efficientad/bottle/test/broken_large \
    --backend onnx
```

**输出示例**：
```
============================================================
Batch Results Summary:
============================================================
  [✓] 000: score=0.0632, pred=Normal
  [✓] 001: score=0.0516, pred=Normal
  ...
  [✓] 000: score=0.5579, pred=Defective
  [✓] 001: score=0.7612, pred=Defective
------------------------------------------------------------
Good images:   mean=0.0522, min=0.0516, max=0.0632
Defect images: mean=0.5828, min=0.5579, max=0.8522
============================================================
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--img` | 单张图像路径 | - |
| `--img_dir` | 图像目录（可多次指定） | - |
| `--backend` | 后端类型：`onnx` 或 `tensorrt` | `onnx` |
| `--efficientad_dir` | EfficientAD ONNX 模型目录 | `deploy_efficientad` |
| `--efficientad_trt_dir` | EfficientAD TensorRT 模型目录 | `deploy_efficientad_trt` |
| `--xgboost_pkl` | XGBoost pickle 模型路径 | - |
| `--segm_map_good_dir` | Good 分割图目录 | - |
| `--segm_map_bad_dir` | Bad 分割图目录 | - |
| `--device` | 推理设备 | `cpu` |

---

# 常见问题

### Q1: 分数如何解读？

A: SegAD 输出的是缺陷概率（0-1 之间）：
- `score < 0.5`：预测为 Normal（正常）
- `score >= 0.5`：预测为 Defective（缺陷）

### Q2: 为什么 Good 样本分数不是 0？

A: SegAD 是监督学习方法，Good 样本的分数取决于训练时使用的 XGBoost 模型。理想情况下 Good 样本分数应该接近 0，Defective 接近 1。

### Q3: 为什么需要分割图？

A: 分割图将图像分成不同区域（如 PCB 的不同组件），SegAD 对每个区域分别提取异常特征，然后汇总判断。这比直接用整张图的异常分数更准确。

### Q4: TensorRT 和 ONNX 结果一样吗？

A: 是的，两种后端使用相同的模型和数据，理论上结果完全一致（可能有极小的浮点误差）。

### Q5: 如何处理新的产品类别？

A: 需要：
1. 准备该类别的分割图数据
2. 使用 EfficientAD 生成异常图
3. 创建 CSV 文件
4. 在 `segad_train_loop.py` 中添加类别配置
5. 训练新的 XGBoost 模型
6. 更新推理脚本中的路径配置

### Q6: 为什么分割图组件数重要？

A: 分割图组件数决定了特征维度：
- 1 个组件 × 1 个模型 × 4 个统计特征 + 1 个全局分数 = 5 维特征
- 2 个组件 × 1 个模型 × 4 个统计特征 + 1 个全局分数 = 9 维特征

### Q7: SegAD 和 EfficientAD 如何选择？

A: 根据产品复杂度选择：
- **简单产品**（如 bottle）：直接使用 EfficientAD 即可
- **复杂产品**（如 PCB，多组件）：使用 SegAD 可以获得更好的准确率

---

## 目录结构

```
projects/csy_segad/
├── configs/
│   ├── segad_efficient_ad.py      # 原始配置文件（VisA 数据集）
│   └── segad_bottle.py            # Bottle 数据集配置
├── datasets/
│   └── segad_dataset.py           # 数据集类
├── models/
│   └── segad.py                   # SegAD 模型定义
├── runner/
│   └── segad_train_loop.py        # 自定义训练循环
├── utils/
│   └── feature_extractor.py       # 特征提取工具
├── prepare_segad_data.py          # 数据准备脚本
├── xgboost_to_onnx.py             # XGBoost 转 ONNX（暂未使用）
├── segad_inference.py             # 推理脚本
└── README.md                      # 本文档
```
