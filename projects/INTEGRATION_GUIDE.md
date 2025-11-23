# MMDetection 项目集成指南：从零开始

本指南将帮助你系统地将一个全新的项目集成到 MMDetection 框架中。我们将按照清晰的步骤，逐步完成整个集成过程。

## 目录

1. [前期准备](#前期准备)
2. [项目结构规划](#项目结构规划)
3. [步骤一：创建项目目录结构](#步骤一创建项目目录结构)
4. [步骤二：实现数据集模块](#步骤二实现数据集模块)
5. [MMDetection 默认架构详解](#mmdetection-默认架构详解)
6. [步骤三：实现模型模块](#步骤三实现模型模块)
7. [步骤四：实现评估指标模块](#步骤四实现评估指标模块)
8. [步骤五：实现训练/测试循环（如需要）](#步骤五实现训练测试循环如需要)
9. [步骤六：实现钩子（如需要）](#步骤六实现钩子如需要)
10. [步骤七：创建配置文件](#步骤七创建配置文件)
11. [步骤八：测试和调试](#步骤八测试和调试)
12. [常见问题排查](#常见问题排查)

---

## 前期准备

### 1. 理解你的项目需求

在开始之前，明确以下问题：

- **项目类型**：目标检测、实例分割、异常检测、分类等？
- **模型架构**：深度学习模型（PyTorch）还是传统机器学习模型（XGBoost、SVM等）？
- **训练方式**：标准训练流程还是需要自定义训练循环？
- **数据格式**：图像、视频、点云等？数据如何组织？
- **评估指标**：需要哪些评估指标（mAP、AUC、准确率等）？

### 2. 分析原始项目

如果已有原始实现，分析：

- 数据加载逻辑
- 模型架构和前向传播
- 损失函数计算
- 训练循环逻辑
- 评估指标计算

---

## 项目结构规划

### 标准项目结构

```
projects/your_project/
├── __init__.py              # 导入所有模块（触发注册）
├── configs/                 # 配置文件目录
│   └── your_config.py       # 主配置文件
├── models/                   # 模型定义
│   ├── __init__.py
│   └── your_model.py
├── datasets/                 # 数据集定义
│   ├── __init__.py
│   └── your_dataset.py
├── metrics/                  # 评估指标
│   ├── __init__.py
│   └── your_metric.py
├── runner/                   # 训练/测试循环（可选）
│   ├── __init__.py
│   └── your_loop.py
├── hooks/                    # 钩子（可选）
│   ├── __init__.py
│   └── your_hook.py
├── utils/                    # 工具函数（可选）
│   ├── __init__.py
│   └── your_utils.py
└── README.md                 # 项目说明文档
```

### 模块选择指南

| 模块 | 必需性 | 使用场景 |
|------|--------|----------|
| `models/` | ✅ 必需 | 定义模型架构 |
| `datasets/` | ✅ 必需 | 定义数据加载逻辑 |
| `metrics/` | ✅ 必需 | 定义评估指标 |
| `configs/` | ✅ 必需 | 配置文件 |
| `runner/` | ⚠️ 可选 | 需要自定义训练/测试流程时 |
| `hooks/` | ⚠️ 可选 | 需要在特定时机执行操作时 |
| `utils/` | ⚠️ 可选 | 需要可复用的工具函数时 |

---

## 步骤一：创建项目目录结构

### 1.1 创建项目目录

```bash
cd /path/to/mmdetection/projects
mkdir your_project
cd your_project
```

### 1.2 创建子目录

```bash
mkdir configs models datasets metrics
# 可选目录
mkdir runner hooks utils
```

### 1.3 创建 `__init__.py` 文件

这是**最关键**的文件，用于触发模块注册。

#### `your_project/__init__.py`

```python
# Copyright (c) OpenMMLab. All rights reserved.

# 导入所有需要注册的模块
from . import models      # 注册模型
from . import datasets    # 注册数据集
from . import metrics     # 注册评估指标

# 如果使用了自定义循环或钩子，也需要导入
# from . import runner
# from . import hooks
```

**关键点**：
- 导入模块时会自动执行模块内的 `@register_module()` 装饰器
- 确保所有需要注册的模块都被导入

---

## 步骤二：实现数据集模块

### 2.1 创建数据集文件

#### `datasets/__init__.py`

```python
from .your_dataset import YourDataset

__all__ = ['YourDataset']
```

#### `datasets/your_dataset.py`

```python
# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from mmdet.registry import DATASETS


@DATASETS.register_module()  # 注册数据集
class YourDataset(Dataset):
    """你的数据集类
    
    Args:
        data_root (str): 数据集根目录
        split (str): 数据集划分（'train', 'val', 'test'）
        # 添加你的其他参数
    """
    
    def __init__(self, 
                 data_root: str,
                 split: str = 'train',
                 # 添加你的初始化参数
                 **kwargs):
        super().__init__()
        self.data_root = data_root
        self.split = split
        
        # 准备数据列表
        self.data_list = self._prepare_data_list()
    
    def _prepare_data_list(self) -> List[Dict]:
        """准备数据列表
        
        Returns:
            List[Dict]: 数据列表，每个元素包含数据路径等信息
        """
        data_list = []
        
        # 根据你的数据组织方式，扫描文件
        # 示例：假设数据按类别组织
        split_dir = os.path.join(self.data_root, self.split)
        
        if os.path.exists(split_dir):
            for class_name in os.listdir(split_dir):
                class_dir = os.path.join(split_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                
                for img_name in os.listdir(class_dir):
                    if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                        continue
                    
                    img_path = os.path.join(class_dir, img_name)
                    label = 0 if class_name == 'normal' else 1  # 根据实际情况调整
                    
                    data_list.append({
                        'img_path': img_path,
                        'label': label,
                        'class_name': class_name,
                    })
        
        return data_list
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data_list)
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """加载图像
        
        Args:
            img_path (str): 图像路径
            
        Returns:
            np.ndarray: 图像数组
        """
        img = Image.open(img_path).convert('RGB')
        return np.array(img)
    
    def __getitem__(self, index: int) -> Dict:
        """获取数据样本
        
        Args:
            index (int): 样本索引
            
        Returns:
            Dict: 包含以下键的字典：
                - img: 图像张量或数组
                - label: 标签
                - 其他需要的字段
        """
        item = self.data_list[index]
        img_path = item['img_path']
        label = item['label']
        
        # 加载图像
        img = self._load_image(img_path)
        
        # 转换为张量（如果需要）
        # img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        
        return {
            'img': img,           # 图像数据
            'label': label,       # 标签
            'img_path': img_path, # 图像路径（用于调试）
            # 添加其他需要的字段
        }
```

### 2.2 数据集实现要点

1. **继承 `torch.utils.data.Dataset`**
2. **实现 `__len__()` 和 `__getitem__()`**
3. **`__getitem__()` 返回字典**，键名会被模型使用
4. **支持多路径查找**，提高灵活性
5. **处理数据加载异常**，提供清晰的错误信息

### 2.3 常见数据格式适配

#### 格式1：按类别组织
```
data_root/
├── train/
│   ├── normal/
│   └── abnormal/
├── test/
│   ├── normal/
│   └── abnormal/
```

#### 格式2：CSV文件
```python
def _prepare_data_list(self):
    import pandas as pd
    csv_path = os.path.join(self.data_root, f'{self.split}.csv')
    df = pd.read_csv(csv_path)
    
    data_list = []
    for _, row in df.iterrows():
        data_list.append({
            'img_path': row['image_path'],
            'label': row['label'],
        })
    return data_list
```

#### 格式3：自定义格式
根据你的实际数据格式调整 `_prepare_data_list()` 方法。

---

## MMDetection 默认架构详解

在实现自定义模型之前，深入理解 MMDetection 的架构设计理念和核心机制非常重要。这将帮助你更好地集成自己的项目，并充分利用框架的能力。

### MMDetection 架构设计理念

MMDetection 基于以下核心设计理念构建：

1. **模块化设计**：将复杂系统分解为独立、可复用的模块
2. **配置驱动**：通过配置文件灵活组合和调整组件
3. **注册机制**：统一的组件注册和管理系统
4. **接口标准化**：定义清晰的接口规范，确保组件间的兼容性
5. **层次化抽象**：从底层数据到高层任务的清晰层次结构

### MMDetection 完整架构体系

MMDetection 的架构可以分为以下几个层面：

```
MMDetection 完整架构
│
├── 核心机制层 (Core Mechanism Layer)
│   ├── Registry: 组件注册表（统一管理所有可注册组件）
│   ├── Builder: 组件构建器（根据配置动态构建组件）
│   ├── Config: 配置系统（配置文件解析和继承）
│   └── Runner: 运行器（统一管理训练/测试流程）
│
├── 数据层 (Data Layer)
│   ├── Dataset: 数据集抽象（定义数据加载逻辑）
│   ├── DataLoader: 数据加载器（批量数据加载和采样）
│   ├── Pipeline: 数据流水线（数据预处理和增强）
│   ├── Transform: 数据变换（图像变换、标注变换等）
│   └── Collate: 数据整理（批次数据组织）
│
├── 模型层 (Model Layer)
│   ├── BaseModel: 模型基类（定义模型接口）
│   ├── DataPreprocessor: 数据预处理器（模型输入预处理）
│   ├── Backbone: 骨干网络（特征提取）
│   ├── Neck: 特征融合网络（多尺度特征融合）
│   ├── Head: 任务头（任务特定预测）
│   ├── Loss: 损失函数（损失计算）
│   └── PostProcessor: 后处理器（预测结果后处理）
│
├── 训练层 (Training Layer)
│   ├── TrainLoop: 训练循环（训练流程控制）
│   ├── OptimWrapper: 优化器包装器（优化器管理）
│   ├── ParamScheduler: 参数调度器（学习率等参数调度）
│   ├── Hook: 钩子系统（在训练过程中插入自定义逻辑）
│   └── Logger: 日志系统（训练信息记录）
│
├── 评估层 (Evaluation Layer)
│   ├── TestLoop: 测试循环（测试流程控制）
│   ├── Evaluator: 评估器（评估流程管理）
│   ├── Metric: 评估指标（指标计算）
│   └── Visualizer: 可视化器（结果可视化）
│
└── 工具层 (Utility Layer)
    ├── Checkpoint: 检查点管理（模型保存和加载）
    ├── Distributed: 分布式训练（多GPU/多机训练）
    └── Analysis: 分析工具（模型分析、性能分析等）
```

### 核心机制详解

#### 1. 注册机制 (Registry)

**设计目的**：统一管理所有可注册的组件，实现配置到代码的映射

**工作原理**：
- 通过装饰器 `@MODELS.register_module()` 等注册组件
- 将类名与类对象建立映射关系
- 配置文件通过 `type` 字段引用注册的类名
- 框架根据 `type` 从注册表中查找并实例化对应类

**注册表类型**：
- `MODELS`：模型注册表
- `DATASETS`：数据集注册表
- `METRICS`：评估指标注册表
- `HOOKS`：钩子注册表
- `LOOPS`：训练/测试循环注册表
- `TRANSFORMS`：数据变换注册表
- `OPTIMIZERS`：优化器注册表
- `SCHEDULERS`：调度器注册表

**优势**：
- 解耦配置与实现
- 支持动态组件替换
- 便于扩展和维护

#### 2. 构建机制 (Builder)

**设计目的**：根据配置字典动态构建组件实例

**工作原理**：
- 从配置字典中提取 `type` 字段
- 在对应的注册表中查找类
- 使用配置中的其他参数实例化类
- 支持递归构建嵌套配置

**构建流程**：
```
配置字典 → 提取type → 查找注册表 → 实例化类 → 返回对象
```

#### 3. 配置系统 (Config)

**设计目的**：统一管理所有配置参数，支持配置继承和覆盖

**核心特性**：
- **配置继承**：通过 `_base_` 字段继承其他配置
- **深度合并**：字典配置进行深度合并而非完全替换
- **参数覆盖**：子配置可以覆盖父配置的参数
- **类型检查**：支持配置参数的类型验证

### 各层组件详解

#### 数据层组件

**1. Dataset（数据集）**
- **职责**：定义数据加载逻辑，将原始数据转换为模型可用格式
- **接口**：继承 `torch.utils.data.Dataset`，实现 `__getitem__()` 和 `__len__()`
- **设计原则**：数据加载与预处理分离，支持多种数据格式

**2. Pipeline（数据流水线）**
- **职责**：定义数据预处理和增强的序列
- **组成**：由多个 Transform 按顺序组成
- **设计原则**：可组合、可复用，支持条件执行

**3. Transform（数据变换）**
- **职责**：执行具体的数据变换操作（如缩放、翻转、归一化等）
- **类型**：图像变换、标注变换、格式转换等
- **设计原则**：单一职责，易于扩展

#### 模型层组件

**1. BaseModel（模型基类）**
- **职责**：定义模型的统一接口和行为
- **核心方法**：
  - `train_step()`：训练步骤，返回损失字典
  - `test_step()`：测试步骤，返回预测结果
  - `forward()`：前向传播（可选）
- **设计原则**：接口标准化，支持多种训练模式

**2. DataPreprocessor（数据预处理器）**
- **职责**：在数据进入模型前进行标准化预处理
- **常见操作**：归一化、格式转换、填充、批量组织
- **设计原则**：与模型解耦，可独立配置

**3. Backbone（骨干网络）**
- **职责**：从原始输入中提取特征表示
- **设计特点**：通常输出多尺度特征，支持特征提取的不同阶段
- **设计原则**：特征提取与任务解耦，可替换

**4. Neck（特征融合网络）**
- **职责**：融合和增强 Backbone 提取的特征
- **常见功能**：多尺度特征融合、特征金字塔构建
- **设计原则**：可选组件，根据任务需求决定是否使用

**5. Head（任务头）**
- **职责**：基于特征进行任务特定的预测
- **设计特点**：任务相关，包含预测逻辑和损失计算
- **设计原则**：任务特定，可包含多个子头（如分类头、回归头）

**6. Loss（损失函数）**
- **职责**：计算预测与真实标签之间的差异
- **设计特点**：可组合使用，支持损失权重
- **设计原则**：模块化，易于替换和组合

#### 训练层组件

**1. TrainLoop（训练循环）**
- **职责**：控制训练流程的执行
- **类型**：`EpochBasedTrainLoop`（按epoch）、`IterBasedTrainLoop`（按迭代）
- **设计原则**：流程标准化，支持自定义扩展

**2. OptimWrapper（优化器包装器）**
- **职责**：管理优化器和梯度更新逻辑
- **功能**：梯度裁剪、参数分组、混合精度训练等
- **设计原则**：封装优化器，提供额外功能

**3. ParamScheduler（参数调度器）**
- **职责**：动态调整训练参数（如学习率）
- **类型**：学习率调度器、动量调度器等
- **设计原则**：可组合，支持复杂调度策略

**4. Hook（钩子）**
- **职责**：在训练过程的特定时机执行自定义逻辑
- **钩子时机**：训练开始/结束、迭代前后、epoch前后等
- **设计原则**：非侵入式，灵活扩展

#### 评估层组件

**1. TestLoop（测试循环）**
- **职责**：控制测试/推理流程的执行
- **设计原则**：与训练循环分离，支持独立配置

**2. Evaluator（评估器）**
- **职责**：管理评估流程，协调多个评估指标
- **设计原则**：可组合多个指标，统一接口

**3. Metric（评估指标）**
- **职责**：计算具体的评估指标
- **接口**：`process()` 收集结果，`evaluate()` 计算指标
- **设计原则**：指标独立，易于扩展

### 数据流和组件交互

#### 训练时的数据流

```
原始数据
  ↓
Dataset.__getitem__()          # 数据加载
  ↓
Pipeline (Transform序列)        # 数据预处理和增强
  ↓
DataLoader (批量组织)           # 批量数据
  ↓
DataPreprocessor                # 模型输入预处理
  ↓
Model.train_step()              # 训练步骤
  ├─> Backbone                  # 特征提取
  ├─> Neck                      # 特征融合（可选）
  ├─> Head                      # 预测
  └─> Loss                      # 损失计算
  ↓
OptimWrapper                    # 梯度更新
  ↓
Hook (记录、保存等)             # 训练钩子
```

#### 测试时的数据流

```
原始数据
  ↓
Dataset.__getitem__()          # 数据加载
  ↓
Pipeline (Transform序列)        # 数据预处理
  ↓
DataLoader (批量组织)           # 批量数据
  ↓
DataPreprocessor                # 模型输入预处理
  ↓
Model.test_step()              # 测试步骤
  ├─> Backbone                  # 特征提取
  ├─> Neck                      # 特征融合（可选）
  └─> Head                      # 预测
  ↓
PostProcessor                   # 后处理（NMS等）
  ↓
Evaluator.process()             # 收集结果
  ↓
Metric.evaluate()               # 计算指标
```

### 组件间的依赖关系

```
Runner (运行器)
  ├─> 管理 TrainLoop/TestLoop
  ├─> 管理 OptimWrapper
  ├─> 管理 ParamScheduler
  └─> 管理 Hooks

TrainLoop/TestLoop
  ├─> 使用 DataLoader
  ├─> 调用 Model.train_step()/test_step()
  └─> 触发 Hooks

Model
  ├─> 包含 DataPreprocessor
  ├─> 包含 Backbone
  ├─> 可选包含 Neck
  ├─> 包含 Head
  └─> Head 包含 Loss

DataLoader
  ├─> 使用 Dataset
  └─> Dataset 使用 Pipeline
```

### 自定义时的修改策略

理解架构后，可以根据项目需求选择不同的修改策略。以下是系统化的修改指南：

#### 修改策略分类

**策略1：最小化修改（推荐）**
- **适用场景**：项目与标准架构高度相似
- **修改范围**：只修改必要的组件（如 Head、Loss）
- **优势**：工作量小，稳定性高，易于维护
- **示例**：改进检测头、使用新的损失函数

**策略2：组件替换**
- **适用场景**：需要替换特定组件但保持整体架构
- **修改范围**：替换 Backbone、Neck、Head 等组件
- **优势**：保持框架兼容性，复用训练流程
- **示例**：使用新的 Backbone、替换特征融合网络

**策略3：架构扩展**
- **适用场景**：需要添加新组件但保持兼容
- **修改范围**：添加新的组件类型，扩展模型结构
- **优势**：灵活扩展，不影响现有功能
- **示例**：添加辅助头、多任务学习

**策略4：完全自定义**
- **适用场景**：项目与标准架构差异很大
- **修改范围**：实现完整的模型、训练循环等
- **优势**：完全控制，不受框架限制
- **示例**：异常检测、传统ML模型集成

#### 各层组件的自定义指南

**数据层的自定义**

| 组件 | 何时需要自定义 | 自定义要点 |
|------|---------------|-----------|
| **Dataset** | 数据格式特殊、加载逻辑复杂 | 实现 `__getitem__()` 返回字典，键名与模型期望一致 |
| **Transform** | 需要特殊的数据变换 | 实现 `__call__()` 方法，输入输出格式统一 |
| **Pipeline** | 需要特殊的预处理流程 | 组合现有 Transform 或添加自定义 Transform |
| **DataPreprocessor** | 数据格式与标准不同 | 实现预处理逻辑，处理批量数据 |

**模型层的自定义**

| 组件 | 何时需要自定义 | 自定义要点 |
|------|---------------|-----------|
| **BaseModel** | 模型架构完全不同 | 继承 `BaseModule`，实现 `train_step()` 和 `test_step()` |
| **Backbone** | 使用新的特征提取网络 | 实现特征提取逻辑，输出多尺度特征（如需要） |
| **Neck** | 需要新的特征融合方式 | 实现特征融合逻辑，输入输出通道匹配 |
| **Head** | 任务特定或新的预测方式 | 实现预测逻辑和损失计算 |
| **Loss** | 使用新的损失函数 | 实现损失计算，返回标量或字典 |

**训练层的自定义**

| 组件 | 何时需要自定义 | 自定义要点 |
|------|---------------|-----------|
| **TrainLoop** | 训练流程特殊（如XGBoost） | 继承 `EpochBasedTrainLoop` 或 `IterBasedTrainLoop` |
| **Hook** | 需要在特定时机执行操作 | 继承 `Hook`，实现对应的钩子方法 |
| **OptimWrapper** | 需要特殊的优化策略 | 通常不需要自定义，使用现有配置即可 |
| **ParamScheduler** | 需要特殊的参数调度 | 继承调度器基类，实现调度逻辑 |

**评估层的自定义**

| 组件 | 何时需要自定义 | 自定义要点 |
|------|---------------|-----------|
| **Metric** | 需要新的评估指标 | 继承 `BaseMetric`，实现 `process()` 和 `evaluate()` |
| **TestLoop** | 测试流程特殊 | 继承 `TestLoop`，自定义测试逻辑 |
| **Evaluator** | 需要特殊的评估流程 | 通常不需要自定义，组合多个 Metric 即可 |

#### 自定义决策流程

```
开始自定义
  │
  ├─> 分析项目需求
  │   ├─> 数据格式是否标准？
  │   ├─> 模型架构是否标准？
  │   ├─> 训练流程是否标准？
  │   └─> 评估指标是否标准？
  │
  ├─> 确定修改策略
  │   ├─> 最小化修改 → 只修改必要组件
  │   ├─> 组件替换 → 替换特定组件
  │   ├─> 架构扩展 → 添加新组件
  │   └─> 完全自定义 → 实现完整流程
  │
  ├─> 识别需要修改的组件
  │   ├─> 数据层：Dataset、Transform、Pipeline
  │   ├─> 模型层：Model、Backbone、Neck、Head、Loss
  │   ├─> 训练层：TrainLoop、Hook
  │   └─> 评估层：Metric、TestLoop
  │
  └─> 实现和测试
      ├─> 注册组件
      ├─> 创建配置文件
      └─> 测试验证
```

#### 常见自定义场景

**场景1：标准任务的新方法**
- **修改**：Head、Loss（可能）
- **保持**：Backbone、Neck、训练流程
- **示例**：新的检测头设计

**场景2：新任务类型**
- **修改**：Model、Head、Metric
- **保持**：数据层（如果数据格式相似）、训练流程
- **示例**：异常检测、关键点检测

**场景3：特殊数据格式**
- **修改**：Dataset、Pipeline、DataPreprocessor（可能）
- **保持**：模型层、训练流程
- **示例**：多模态数据、点云数据

**场景4：非深度学习模型**
- **修改**：Model、TrainLoop
- **保持**：数据层（如果格式兼容）、评估层
- **示例**：XGBoost、SVM 等传统 ML 模型

**场景5：特殊训练流程**
- **修改**：TrainLoop、Hook（可能）
- **保持**：模型层、数据层
- **示例**：多阶段训练、课程学习

### 架构设计原则总结

**1. 单一职责原则**
- 每个组件只负责一个明确的功能
- 组件之间职责清晰，边界明确

**2. 开闭原则**
- 对扩展开放：可以轻松添加新组件
- 对修改封闭：修改不影响现有组件

**3. 依赖倒置原则**
- 高层模块不依赖低层模块，都依赖抽象
- 通过接口和注册机制实现解耦

**4. 接口隔离原则**
- 定义清晰的接口，组件只需实现必要的方法
- 避免强制实现不需要的方法

**5. 配置驱动原则**
- 通过配置文件组合组件，而非硬编码
- 配置与实现分离，提高灵活性

### 自定义最佳实践

**1. 遵循接口规范**
- 确保自定义组件符合框架定义的接口
- 方法签名、返回值格式要与标准一致

**2. 充分利用现有组件**
- 优先使用框架提供的组件
- 只在必要时才自定义新组件

**3. 保持模块化**
- 将复杂功能拆分为多个小模块
- 每个模块职责单一，易于测试和维护

**4. 文档和注释**
- 为自定义组件添加清晰的文档字符串
- 说明组件的用途、参数和使用方法

**5. 测试验证**
- 逐步测试每个自定义组件
- 确保组件能正确集成到框架中

**6. 参考示例项目**
- 学习 `csy_efficientad` 和 `csy_segad` 等项目的实现
- 参考 MMDetection 官方项目的设计模式

---

## 步骤三：实现模型模块

### 3.1 创建模型文件

#### `models/__init__.py`

```python
from .your_model import YourModel

__all__ = ['YourModel']
```

#### `models/your_model.py`

```python
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet.registry import MODELS


@MODELS.register_module()  # 注册模型
class YourModel(BaseModule):
    """你的模型类
    
    Args:
        num_classes (int): 类别数量
        # 添加你的其他参数
    """
    
    def __init__(self,
                 num_classes: int = 2,
                 # 添加你的初始化参数
                 init_cfg: Optional[Dict] = None):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        
        # 定义模型架构
        self.backbone = self._build_backbone()
        self.head = self._build_head()
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
    
    def _build_backbone(self) -> nn.Module:
        """构建骨干网络"""
        # 实现你的骨干网络
        # 示例：简单的CNN
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
    
    def _build_head(self) -> nn.Module:
        """构建分类头"""
        return nn.Linear(64, self.num_classes)
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """前向传播（用于推理）
        
        Args:
            img (torch.Tensor): 输入图像 [B, C, H, W]
            
        Returns:
            torch.Tensor: 模型输出
        """
        features = self.backbone(img)
        features = features.view(features.size(0), -1)
        output = self.head(features)
        return output
    
    def train_step(self, data: Dict, optim_wrapper) -> Dict:
        """训练步骤
        
        Args:
            data (Dict): 数据批次，包含 'img' 和 'label' 等键
            optim_wrapper: 优化器包装器
            
        Returns:
            Dict: 必须包含 'loss' 键
        """
        img = data['img']
        label = data['label']
        
        # 如果数据是numpy数组，转换为张量
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
            if img.ndim == 3:
                img = img.permute(2, 0, 1).unsqueeze(0)
        
        # 确保标签是张量
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        
        # 前向传播
        output = self.forward(img)
        
        # 计算损失
        loss = self.criterion(output, label)
        
        # 返回损失字典
        return dict(loss=loss)
    
    def test_step(self, data_batch: Dict, **kwargs) -> List[Dict]:
        """测试步骤
        
        Args:
            data_batch (Dict): 数据批次
            
        Returns:
            List[Dict]: 结果列表，每个元素对应一个样本
        """
        img = data_batch['img']
        label = data_batch.get('label', None)
        
        # 处理输入
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
            if img.ndim == 3:
                img = img.permute(2, 0, 1).unsqueeze(0)
        
        # 推理
        with torch.no_grad():
            output = self.forward(img)
            probs = torch.softmax(output, dim=1)
            score = probs[0, 1].item()  # 假设是二分类，取正类概率
        
        # 构建结果
        results = [dict(
            score=score,
            label=int(label) if label is not None else -1,
            img_path=data_batch.get('img_path', ''),
        )]
        
        return results
```

### 3.2 模型实现要点

1. **继承 `BaseModule`**（MMEngine 的基类）
2. **实现 `train_step()`**：用于训练，必须返回包含 `loss` 的字典
3. **实现 `test_step()`**：用于测试，返回结果字典列表
4. **可选实现 `forward()`**：用于推理
5. **处理数据类型转换**：numpy 数组 → 张量

### 3.3 特殊模型类型

#### 类型1：多子网络模型（如 EfficientAD）

```python
@MODELS.register_module()
class MultiNetworkModel(BaseModule):
    def __init__(self, ...):
        super().__init__()
        self.network1 = self._build_network1()
        self.network2 = self._build_network2()
    
    def train_step(self, data, optim_wrapper):
        # 使用多个网络
        output1 = self.network1(data['img1'])
        output2 = self.network2(data['img2'])
        loss = self._compute_loss(output1, output2, data)
        return dict(loss=loss)
```

#### 类型2：非深度学习模型（如 XGBoost）

```python
@MODELS.register_module()
class XGBoostModel(BaseModule):
    def __init__(self, ...):
        super().__init__()
        from xgboost import XGBClassifier
        self.xgb = XGBClassifier(...)
        # 添加虚拟参数以满足MMEngine要求
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=True)
    
    def fit(self, features, labels):
        """直接训练XGBoost（不通过train_step）"""
        self.xgb.fit(features, labels)
    
    def train_step(self, data, optim_wrapper):
        """占位方法（XGBoost不使用）"""
        return dict(loss=torch.tensor(0.0))
```

---

## 步骤四：实现评估指标模块

### 4.1 创建评估指标文件

#### `metrics/__init__.py`

```python
from .your_metric import YourMetric

__all__ = ['YourMetric']
```

#### `metrics/your_metric.py`

```python
# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS


@METRICS.register_module()  # 注册评估指标
class YourMetric(BaseMetric):
    """你的评估指标类
    
    Args:
        save_dir (str, optional): 结果保存目录
        # 添加你的其他参数
    """
    
    def __init__(self,
                 save_dir: str = '',
                 # 添加你的其他参数
                 **kwargs):
        super().__init__(**kwargs)
        self.save_dir = save_dir
        
        # 初始化缓冲区，用于收集批次结果
        self.scores: List[float] = []
        self.labels: List[int] = []
        self.img_paths: List[str] = []
    
    def process(self, data_batch: Dict, data_samples: Sequence[Dict]) -> None:
        """处理一批数据样本
        
        Args:
            data_batch (Dict): 数据批次（通常不使用）
            data_samples (Sequence[Dict]): 模型test_step返回的结果列表
        """
        for sample in data_samples:
            # 从test_step返回的结果中提取数据
            self.scores.append(sample['score'])
            self.labels.append(sample['label'])
            self.img_paths.append(sample.get('img_path', ''))
    
    def compute_metrics(self, results: List) -> Dict[str, float]:
        """计算指标（MMEngine接口）
        
        Args:
            results (List): 结果列表（通常不使用，我们使用缓冲区）
            
        Returns:
            Dict[str, float]: 指标字典
        """
        return self.evaluate()
    
    def evaluate(self, size: int = 0) -> Dict[str, float]:
        """计算最终指标
        
        Args:
            size (int): 数据集大小（可选）
            
        Returns:
            Dict[str, float]: 指标字典
        """
        if not self.scores:
            return dict(auc=0.0, accuracy=0.0)
        
        # 转换为numpy数组
        scores = np.asarray(self.scores, dtype=np.float32)
        labels = np.asarray(self.labels, dtype=np.int32)
        
        # 计算指标
        metrics_dict = {}
        
        # 1. ROC-AUC
        if len(np.unique(labels)) >= 2:
            metrics_dict['auc'] = float(roc_auc_score(labels, scores))
        else:
            metrics_dict['auc'] = 0.0
        
        # 2. 准确率（需要阈值）
        predictions = (scores > 0.5).astype(int)
        metrics_dict['accuracy'] = float(accuracy_score(labels, predictions))
        
        # 3. 保存结果（如果需要）
        if self.save_dir:
            self._save_results(scores, labels, self.img_paths)
        
        # 清空缓冲区，准备下次评估
        self.scores.clear()
        self.labels.clear()
        self.img_paths.clear()
        
        return metrics_dict
    
    def _save_results(self, scores: np.ndarray, labels: np.ndarray, 
                      img_paths: List[str]) -> None:
        """保存结果到文件
        
        Args:
            scores (np.ndarray): 预测分数
            labels (np.ndarray): 真实标签
            img_paths (List[str]): 图像路径列表
        """
        if not self.save_dir:
            return
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 保存为CSV
        import pandas as pd
        df = pd.DataFrame({
            'img_path': img_paths,
            'label': labels,
            'score': scores,
        })
        df.to_csv(os.path.join(self.save_dir, 'results.csv'), index=False)
```

### 4.2 评估指标实现要点

1. **继承 `BaseMetric`**
2. **实现 `process()`**：收集批次结果到缓冲区
3. **实现 `evaluate()`**：计算最终指标
4. **清空缓冲区**：每次评估后清空，准备下次使用
5. **处理边界情况**：如只有一个类别的情况

---

## 步骤五：实现训练/测试循环（如需要）

### 5.1 何时需要自定义循环

- 使用非标准训练流程（如 XGBoost）
- 需要特殊的数据处理逻辑
- 需要多阶段训练
- 需要特殊的评估流程

### 5.2 创建训练循环

#### `runner/__init__.py`

```python
from .your_train_loop import YourTrainLoop

__all__ = ['YourTrainLoop']
```

#### `runner/your_train_loop.py`

```python
# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.logging import print_log
from mmengine.runner.loops import EpochBasedTrainLoop

from mmdet.registry import LOOPS


@LOOPS.register_module()  # 注册训练循环
class YourTrainLoop(EpochBasedTrainLoop):
    """自定义训练循环
    
    Args:
        runner: Runner实例
        dataloader: 数据加载器
        max_epochs (int): 最大epoch数
        val_interval (int): 验证间隔
        # 添加你的其他参数
    """
    
    def __init__(self,
                 runner,
                 dataloader,
                 max_epochs: int = 1,
                 val_interval: int = 1,
                 # 添加你的其他参数
                 **kwargs):
        super().__init__(runner, dataloader, max_epochs, val_interval)
        # 保存你的自定义参数
    
    def run(self) -> None:
        """运行训练循环"""
        # 调用训练前钩子
        self.runner.call_hook('before_train')
        self.runner.call_hook('before_train_epoch')
        
        # 你的自定义训练逻辑
        print_log('Starting custom training...')
        
        # 示例：自定义训练流程
        for epoch in range(self.max_epochs):
            # 训练代码
            self._train_epoch()
            
            # 验证（如果需要）
            if self.val_interval > 0 and (epoch + 1) % self.val_interval == 0:
                self._validate()
        
        # 调用训练后钩子
        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')
    
    def _train_epoch(self):
        """训练一个epoch"""
        # 实现你的训练逻辑
        pass
    
    def _validate(self):
        """验证"""
        # 实现你的验证逻辑
        pass
```

### 5.3 创建测试循环

#### `runner/your_test_loop.py`

```python
# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.logging import print_log
from mmengine.runner.loops import TestLoop

from mmdet.registry import LOOPS


@LOOPS.register_module()
class YourTestLoop(TestLoop):
    """自定义测试循环"""
    
    def run(self) -> None:
        """运行测试循环"""
        # 先执行标准测试
        super().run()
        
        # 然后执行你的自定义逻辑
        print_log('Running custom post-processing...')
        # 你的自定义代码
```

---

## 步骤六：实现钩子（如需要）

### 6.1 何时需要钩子

- 定期保存模型
- 记录特殊指标
- 在特定时机执行操作
- 修改训练过程

### 6.2 创建钩子文件

#### `hooks/__init__.py`

```python
from .your_hook import YourHook

__all__ = ['YourHook']
```

#### `hooks/your_hook.py`

```python
# Copyright (c) OpenMMLab. All rights reserved.
import os
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()  # 注册钩子
class YourHook(Hook):
    """自定义钩子
    
    Args:
        interval (int): 执行间隔
        # 添加你的其他参数
    """
    
    def __init__(self, interval: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.interval = interval
    
    def before_train(self, runner: Runner) -> None:
        """训练开始前调用"""
        print_log('Training is about to start...')
    
    def after_train_iter(self, runner: Runner, batch_idx: int, 
                        data_batch=None, outputs=None) -> None:
        """每次训练迭代后调用"""
        if runner.iter % self.interval == 0:
            # 你的自定义逻辑
            print_log(f'Iteration {runner.iter} completed')
    
    def after_train_epoch(self, runner: Runner) -> None:
        """每个epoch结束后调用"""
        # 你的自定义逻辑
        pass
    
    def after_train(self, runner: Runner) -> None:
        """训练结束后调用"""
        # 你的自定义逻辑
        pass
```

---

## 步骤七：创建配置文件

### 7.1 基础配置文件模板

#### `configs/your_config.py`

```python
# 1. 导入项目模块（必须！）
custom_imports = dict(
    imports=['projects.your_project'],
    allow_failed_imports=False
)

# 2. 设置默认scope
default_scope = 'mmdet'

# 3. 模型配置
model = dict(
    type='YourModel',  # 使用注册的模型类名
    num_classes=2,
    # 添加你的模型参数
)

# 4. 数据集配置
data_root = '/path/to/your/data'

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='YourDataset',  # 使用注册的数据集类名
        data_root=data_root,
        split='train',
        # 添加你的数据集参数
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='YourDataset',
        data_root=data_root,
        split='val',
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='YourDataset',
        data_root=data_root,
        split='test',
    ),
)

# 5. 评估器配置
val_evaluator = dict(
    type='YourMetric',  # 使用注册的评估指标类名
    save_dir='',  # 留空则不保存
)

test_evaluator = dict(
    type='YourMetric',
    save_dir='./results',  # 保存结果
)

# 6. 训练配置
train_cfg = dict(
    type='EpochBasedTrainLoop',  # 或 'IterBasedTrainLoop'
    max_epochs=100,
    val_interval=10,
)

# 如果使用自定义训练循环
# train_cfg = dict(
#     type='YourTrainLoop',
#     max_epochs=100,
#     # 你的自定义参数
# )

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 如果使用自定义测试循环
# test_cfg = dict(type='YourTestLoop')

# 7. 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-4),
)

# 8. 学习率调度器（可选）
param_scheduler = [
    dict(
        type='MultiStepLR',
        by_epoch=True,
        milestones=[50, 80],
        gamma=0.1,
    )
]

# 9. 钩子配置
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,  # 每10个epoch保存一次
        max_keep_ckpts=3,  # 最多保留3个checkpoint
    ),
    logger=dict(type='LoggerHook', interval=10),
)

# 自定义钩子（可选）
# custom_hooks = [
#     dict(
#         type='YourHook',
#         interval=1000,
#     ),
# ]

# 10. 日志配置
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
```

### 7.2 配置文件要点

1. **`custom_imports` 必须在最前面**，用于触发模块注册
2. **`type` 字段必须与注册的类名完全一致**
3. **使用字典嵌套结构**，便于参数传递和覆盖
4. **路径使用绝对路径**，避免相对路径问题

### 7.2.1 嵌套字典结构的含义

配置文件中的嵌套字典结构（如 `model` 字典中包含多层嵌套）反映了**模型的层次化架构**和**组件的模块化设计**。

#### 嵌套字典的本质

```python
model = dict(
    type='YourModel',           # 顶层：模型类型
    backbone=dict(              # 第二层：子模块（backbone）
        type='ResNet',
        depth=50,
        norm_cfg=dict(           # 第三层：子模块的配置（norm_cfg）
            type='BN',
            requires_grad=False
        ),
        init_cfg=dict(          # 第三层：子模块的配置（init_cfg）
            type='Pretrained',
            checkpoint='...'
        )
    ),
    neck=dict(                  # 第二层：子模块（neck）
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
    ),
    head=dict(                  # 第二层：子模块（head）
        type='RPNHead',
        in_channels=256,
        loss_cls=dict(          # 第三层：子模块的配置（loss_cls）
            type='CrossEntropyLoss',
            loss_weight=1.0
        ),
        loss_bbox=dict(         # 第三层：子模块的配置（loss_bbox）
            type='L1Loss',
            loss_weight=1.0
        )
    )
)
```

#### 嵌套字典的含义

**1. 反映模型架构的层次结构**
- **顶层字典**（`model`）：定义整个模型的结构
- **第二层字典**（`backbone`, `neck`, `head`）：对应模型的各个子模块
- **第三层字典**（`norm_cfg`, `loss_cls` 等）：对应子模块的具体配置参数

**2. 实现模块化设计**
- 每个嵌套字典对应一个可独立构建的模块
- 通过 `type` 字段指定模块类型，框架自动构建对应实例
- 参数通过字典键值对传递，实现配置与代码的解耦

**3. 参数传递机制**
```python
# 配置中的嵌套字典
backbone=dict(
    type='ResNet',
    depth=50,
    norm_cfg=dict(type='BN', requires_grad=False)
)

# 框架会自动转换为：
# backbone = ResNet(depth=50, norm_cfg=BN(requires_grad=False))
```

**4. 实际应用示例**

以 CO-DETR 配置文件为例：

```python
model = dict(
    type='CoDETR',                    # 主模型
    backbone=dict(                    # Backbone 模块
        type='ResNet',
        depth=50,
        norm_cfg=dict(...),           # 归一化配置
        init_cfg=dict(...)            # 初始化配置
    ),
    query_head=dict(                  # Query Head 模块
        type='CoDINOHead',
        transformer=dict(             # Transformer 子模块
            type='CoDinoTransformer',
            encoder=dict(              # Encoder 子模块
                type='DetrTransformerEncoder',
                transformerlayers=dict(...)  # 更深的嵌套
            ),
            decoder=dict(...)          # Decoder 子模块
        ),
        loss_cls=dict(...),           # 损失函数配置
        loss_bbox=dict(...)
    )
)
```

这个嵌套结构清晰地表达了：
- `CoDETR` 模型包含 `ResNet` backbone 和 `CoDINOHead` query head
- `CoDINOHead` 包含 `CoDinoTransformer`
- `CoDinoTransformer` 包含 `Encoder` 和 `Decoder`
- 每个组件都有自己独立的配置参数

#### 嵌套字典的优势

1. **直观反映架构**：配置结构与模型架构一一对应
2. **易于理解和修改**：可以清楚地看到每个组件的配置
3. **支持部分覆盖**：在继承配置时，可以只修改特定层级的参数
4. **类型安全**：通过 `type` 字段确保构建正确的模块类型

#### 注意事项

1. **字典深度合并**：继承配置时，相同路径的字典会进行深度合并，不是完全替换
2. **`type` 字段必需**：每个需要构建实例的字典必须包含 `type` 字段
3. **参数命名一致**：字典的键名必须与模块的 `__init__` 参数名一致

### 7.3 嵌套配置文件（配置文件继承）

#### 7.3.1 嵌套配置文件的概念

MMDetection 支持通过 `_base_` 字段实现配置文件的继承和嵌套，这是配置管理的重要特性。

#### 7.3.2 嵌套产生的原因

**1. 代码复用和模块化**
- 避免重复配置：多个配置文件可能共享相同的模型架构、数据集设置等
- 提高可维护性：修改基础配置时，所有继承的配置自动更新
- 模块化设计：将不同方面的配置分离到不同文件

**2. 配置层次化管理**
- 基础配置：定义通用的、不常变化的配置
- 特定配置：在基础配置上添加或修改特定参数
- 实验配置：基于特定配置进行微调

**3. 实验管理**
- 快速创建变体：只需修改少量参数即可创建新实验
- 版本控制友好：基础配置和实验配置分离，便于 Git 管理
- 配置对比：可以清楚地看到不同实验之间的差异

#### 7.3.3 嵌套配置文件的使用方法

##### 方法1：使用 `_base_` 字段继承单个配置文件

```python
# configs/base_config.py（基础配置文件）
custom_imports = dict(
    imports=['projects.your_project'],
    allow_failed_imports=False
)

default_scope = 'mmdet'

# 模型配置
model = dict(
    type='YourModel',
    num_classes=2,
    backbone=dict(
        type='ResNet',
        depth=50,
    ),
)

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-4),
)

# 训练配置
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,
    val_interval=10,
)
```

```python
# configs/experiment_config.py（实验配置文件）
_base_ = './base_config.py'  # 继承基础配置

# 只覆盖需要修改的部分
model = dict(
    backbone=dict(
        depth=101,  # 修改深度为101
    ),
)

train_cfg = dict(
    max_epochs=200,  # 修改训练轮数
)
```

##### 方法2：使用列表继承多个配置文件

```python
# configs/models/resnet50.py（模型配置）
model = dict(
    type='YourModel',
    backbone=dict(
        type='ResNet',
        depth=50,
    ),
)

# configs/datasets/coco.py（数据集配置）
data_root = '/path/to/coco'
train_dataloader = dict(
    dataset=dict(
        type='YourDataset',
        data_root=data_root,
        split='train',
    ),
)

# configs/schedules/adam_1x.py（训练策略配置）
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-4),
)
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,
)

# configs/default_runtime.py（运行时配置）
default_scope = 'mmdet'
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10),
    logger=dict(type='LoggerHook', interval=10),
)

# configs/your_experiment.py（最终实验配置）
_base_ = [
    'models/resnet50.py',      # 继承模型配置
    'datasets/coco.py',        # 继承数据集配置
    'schedules/adam_1x.py',    # 继承训练策略
    'default_runtime.py',      # 继承运行时配置
]

# 只添加或修改特定参数
custom_imports = dict(
    imports=['projects.your_project'],
    allow_failed_imports=False
)

# 覆盖部分参数
model = dict(
    backbone=dict(depth=101),  # 修改为ResNet101
)
```

#### 7.3.4 配置文件继承的规则

**1. 继承顺序**
- 列表中的配置文件按顺序继承
- 后面的配置会覆盖前面配置的相同字段
- 字典会进行深度合并（deep merge）

**2. 覆盖规则**
```python
# 基础配置
model = dict(
    type='YourModel',
    backbone=dict(depth=50, num_stages=4),
    head=dict(num_classes=10),
)

# 继承配置
_base_ = ['./base_config.py']
model = dict(
    backbone=dict(depth=101),  # 只覆盖depth，num_stages保持不变
    # head配置保持不变
)

# 最终结果
# model = dict(
#     type='YourModel',
#     backbone=dict(depth=101, num_stages=4),  # depth被覆盖，num_stages保留
#     head=dict(num_classes=10),  # 保持不变
# )
```

**3. 删除字段**
```python
# 如果要删除基础配置中的字段，设置为None
_base_ = ['./base_config.py']
model = dict(
    head=None,  # 删除head配置
)
```

#### 7.3.5 实际应用示例

##### 示例1：不同数据集的实验

```python
# configs/base_model.py
model = dict(type='YourModel', num_classes=2)
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='Adam', lr=1e-4))

# configs/dataset_a.py
_base_ = ['./base_model.py']
data_root = '/path/to/dataset_a'
train_dataloader = dict(dataset=dict(type='YourDataset', data_root=data_root))

# configs/dataset_b.py
_base_ = ['./base_model.py']
data_root = '/path/to/dataset_b'
train_dataloader = dict(dataset=dict(type='YourDataset', data_root=data_root))
```

##### 示例2：不同学习率的实验

```python
# configs/base_experiment.py
_base_ = ['./base_model.py', './dataset_a.py']
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100)

# configs/experiment_lr1e3.py
_base_ = ['./base_experiment.py']
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-3),  # 修改学习率
)

# configs/experiment_lr1e5.py
_base_ = ['./base_experiment.py']
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-5),  # 修改学习率
)
```

##### 示例3：复杂嵌套（参考 MMDetection 官方配置）

```python
# configs/_base_/models/faster-rcnn_r50_fpn.py
model = dict(
    type='FasterRCNN',
    backbone=dict(...),
    neck=dict(...),
    rpn_head=dict(...),
    roi_head=dict(...),
)

# configs/_base_/datasets/coco_detection.py
data_root = 'data/coco/'
train_dataloader = dict(...)
val_dataloader = dict(...)
test_dataloader = dict(...)

# configs/_base_/schedules/schedule_1x.py
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12)
optim_wrapper = dict(...)
param_scheduler = [...]

# configs/_base_/default_runtime.py
default_scope = 'mmdet'
default_hooks = dict(...)

# configs/faster-rcnn/faster-rcnn_r50_fpn_1x_coco.py
_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]
```

#### 7.3.6 嵌套配置文件的优势

1. **减少代码重复**：共享配置只需定义一次
2. **易于维护**：修改基础配置，所有实验自动更新
3. **清晰的实验对比**：实验配置文件只包含差异部分
4. **模块化设计**：不同方面的配置分离管理
5. **版本控制友好**：基础配置和实验配置分离，便于 Git 管理

#### 7.3.7 注意事项

1. **路径问题**
   - 使用相对路径时，相对于当前配置文件的位置
   - 建议使用绝对路径或相对于项目根目录的路径

2. **循环依赖**
   - 避免配置文件之间的循环引用
   - 保持配置文件的层次结构清晰

3. **字段覆盖**
   - 字典会进行深度合并，不是完全替换
   - 要完全替换某个字典，需要显式设置所有字段

4. **调试技巧**
   - 使用 `Config.fromfile()` 加载配置后，可以查看合并后的完整配置
   - 使用 `cfg.pretty_text` 查看格式化的配置内容

```python
from mmengine.config import Config

# 加载配置（会自动处理_base_继承）
cfg = Config.fromfile('configs/your_experiment.py')

# 查看完整配置
print(cfg.pretty_text)

# 查看特定字段
print(cfg.model)
print(cfg.train_dataloader)
```

---

## 步骤八：测试和调试

### 8.1 验证模块注册

创建测试脚本 `test_registration.py`：

```python
from mmdet.registry import MODELS, DATASETS, METRICS

# 导入项目（触发注册）
from projects.your_project import *

# 检查注册
print("Registered models:", list(MODELS._module_dict.keys()))
print("Registered datasets:", list(DATASETS._module_dict.keys()))
print("Registered metrics:", list(METRICS._module_dict.keys()))

# 应该能看到你的类名
assert 'YourModel' in MODELS._module_dict
assert 'YourDataset' in DATASETS._module_dict
assert 'YourMetric' in METRICS._module_dict
print("✓ All modules registered successfully!")
```

### 8.2 测试数据集

```python
from mmengine.dataset import build_dataloader
from mmengine.config import Config

cfg = Config.fromfile('projects/your_project/configs/your_config.py')

# 构建数据集
dataloader = build_dataloader(cfg.train_dataloader)

# 测试数据加载
for i, batch in enumerate(dataloader):
    print(f"Batch {i}:", batch.keys())
    if i >= 2:  # 只测试前3个批次
        break
print("✓ Dataset loading works!")
```

### 8.3 测试模型

```python
from mmengine.config import Config
from mmdet.registry import MODELS

cfg = Config.fromfile('projects/your_project/configs/your_config.py')

# 构建模型
model = MODELS.build(cfg.model)
print("✓ Model built successfully!")

# 测试前向传播
import torch
dummy_input = torch.randn(1, 3, 224, 224)
output = model.forward(dummy_input)
print(f"✓ Forward pass works! Output shape: {output.shape}")
```

### 8.4 完整训练测试

```bash
# 使用少量数据进行快速测试
python tools/train.py \
    projects/your_project/configs/your_config.py \
    --work-dir ./work_dirs/test \
    --cfg-options train_cfg.max_epochs=1 \
                  train_dataloader.dataset.data_list=data_list[:10]
```

### 8.5 调试技巧

1. **添加日志**：使用 `print_log()` 输出调试信息
2. **检查数据格式**：确保数据格式符合模型期望
3. **逐步测试**：先测试单个模块，再测试完整流程
4. **使用断点**：在关键位置设置断点调试

---

## 常见问题排查

### 问题1：模块未注册

**错误信息**：
```
KeyError: 'YourModel' is not in the MODELS registry
```

**解决方案**：
1. 检查 `__init__.py` 是否导入了模块
2. 检查装饰器 `@MODELS.register_module()` 是否正确
3. 检查配置文件中的 `custom_imports` 是否正确
4. 确保类名与配置文件中的 `type` 一致

### 问题2：数据类型不匹配

**错误信息**：
```
RuntimeError: Expected tensor but got numpy.ndarray
```

**解决方案**：
1. 在 `train_step()` 和 `test_step()` 中处理数据类型转换
2. 确保张量在正确的设备上（CPU/GPU）

### 问题3：数据加载失败

**错误信息**：
```
FileNotFoundError: Image not found
```

**解决方案**：
1. 检查数据路径是否正确
2. 实现多路径查找逻辑
3. 添加文件存在性检查

### 问题4：损失为 NaN

**解决方案**：
1. 检查输入数据是否归一化
2. 检查学习率是否过大
3. 添加梯度裁剪
4. 检查损失函数实现

### 问题5：内存不足

**解决方案**：
1. 减小 `batch_size`
2. 减小图像尺寸
3. 使用梯度累积
4. 减少 `num_workers`

---

## 完整示例：快速开始模板

### 最小化项目模板

```bash
projects/your_project/
├── __init__.py
├── configs/
│   └── your_config.py
├── models/
│   ├── __init__.py
│   └── your_model.py
├── datasets/
│   ├── __init__.py
│   └── your_dataset.py
└── metrics/
    ├── __init__.py
    └── your_metric.py
```

### 快速检查清单

- [ ] 创建项目目录结构
- [ ] 实现数据集类（继承 `Dataset`，注册 `@DATASETS.register_module()`）
- [ ] 实现模型类（继承 `BaseModule`，注册 `@MODELS.register_module()`）
- [ ] 实现评估指标类（继承 `BaseMetric`，注册 `@METRICS.register_module()`）
- [ ] 在 `__init__.py` 中导入所有模块
- [ ] 创建配置文件（包含 `custom_imports`）
- [ ] 测试模块注册
- [ ] 测试数据加载
- [ ] 测试模型构建
- [ ] 运行完整训练流程

---

## 总结

集成新项目到 MMDetection 的关键步骤：

1. **理解项目需求**：明确模型类型、训练方式、数据格式
2. **规划项目结构**：确定需要哪些模块
3. **逐步实现模块**：按数据集 → 模型 → 评估指标 → 其他模块的顺序
4. **使用注册机制**：所有组件通过装饰器注册
5. **创建配置文件**：统一管理所有参数
6. **测试和调试**：逐步验证每个模块

遵循这个流程，你可以系统地将任何项目集成到 MMDetection 框架中！

