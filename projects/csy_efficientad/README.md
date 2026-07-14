# EfficientAD for MMDetection

本项目将 EfficientAD 集成到 MMDetection 框架中，支持使用标准的 `tools/train.py` 和 `tools/test.py` 脚本进行训练和推理，同时支持通过 mmdeploy 进行 ONNX 和 TensorRT 部署。

## 目录

- [环境要求](#环境要求)
- [训练](#训练)
- [推理测试](#推理测试)
- [ONNX 部署](#onnx-部署)
- [TensorRT 部署](#tensorrt-部署)
- [配置文件说明](#配置文件说明)

## 环境要求

### 基础环境

```bash
# 建议使用 conda 环境
conda create -n mmlab python=3.10
conda activate mmlab

# 安装 PyTorch (CUDA 12.1)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 安装 MMDetection 和相关依赖
pip install mmdet==3.3.0 mmengine==0.10.0 mmcv==2.1.0

# 安装 OpenMMLab 其他依赖
pip install openmim
mim install mmdet mmengine mmcv
```

### mmdeploy 环境 (用于 ONNX/TensorRT 部署)

```bash
# 创建独立环境
conda create -n trt_export python=3.10
conda activate trt_export

# 安装 TensorRT (根据你的 CUDA 版本选择)
# CUDA 12.x:
pip install tensorrt-cu12

# 安装 mmdeploy
cd mmdeploy
pip install -e .

# 验证安装
python -c "import mmdeploy; print(mmdeploy.__version__)"
```

### 预训练模型

需要下载 EfficientAD 的预训练 teacher 模型：

```bash
# 放置在 ck4efficientad/models/ 目录下
ck4efficientad/models/
├── teacher_small.pth   # small 版本 teacher
└── teacher_medium.pth  # medium 版本 teacher
```

## 训练

## Checkpoint 保存策略

训练过程中会按以下策略保存 checkpoint，**与原始 EfficientAD 项目的输出格式完全一致**：

### MMEngine 标准 Checkpoint（在 `work_dirs/efficientad_small/`）
- **每 10000 次迭代保存一次**：`iter_10000.pth`, `iter_20000.pth`, ..., `iter_70000.pth`
- **保留最后 8 个 checkpoint**：包括所有中间 checkpoint 和最后一个 checkpoint
- **保存最后一个 checkpoint**：`last.pth`（指向最新的 checkpoint）

### EfficientAD 格式模型（在 `output/trainings/mvtec_ad/bottle/`）
- **每 1000 次迭代保存临时模型**：
  - `teacher_tmp.pth`
  - `student_tmp.pth`
  - `autoencoder_tmp.pth`
- **训练结束后保存最终模型**：
  - `teacher_final.pth`
  - `student_final.pth`
  - `autoencoder_final.pth`

这与原始 `efficientad.py` 的输出格式完全一致，可以直接用于原始 EfficientAD 项目的推理脚本。

## 使用方法

### 1. 训练

使用标准的训练脚本：

```bash
python tools/train.py projects/csy_efficientad/configs/efficientad_small.py
```

### 2. 推理/测试

使用 `tools/test.py` 脚本进行推理和评估：

#### 基本用法

```bash
python tools/test.py <config_file> <checkpoint_file>
```

#### 示例

```bash
# 使用训练好的 checkpoint 进行测试
python tools/test.py \
    projects/csy_efficientad/configs/efficientad_small.py \
    work_dirs/efficientad_small/iter_70000.pth
```

#### 常用参数

- `--work-dir`: 指定工作目录（用于保存评估结果）
  ```bash
  python tools/test.py \
      projects/csy_efficientad/configs/efficientad_small.py \
      work_dirs/efficientad_small/iter_70000.pth \
      --work-dir work_dirs/efficientad_small_test
  ```

- `--out`: 将预测结果保存为 pickle 文件
  ```bash
  python tools/test.py \
      projects/csy_efficientad/configs/efficientad_small.py \
      work_dirs/efficientad_small/iter_70000.pth \
      --out results.pkl
  ```

- `--show`: 显示预测结果（需要图形界面）
  ```bash
  python tools/test.py \
      projects/csy_efficientad/configs/efficientad_small.py \
      work_dirs/efficientad_small/iter_70000.pth \
      --show
  ```

- `--show-dir`: 将可视化结果保存到指定目录
  ```bash
  python tools/test.py \
      projects/csy_efficientad/configs/efficientad_small.py \
      work_dirs/efficientad_small/iter_70000.pth \
      --show-dir visualization
  ```

- `--cfg-options`: 通过命令行覆盖配置参数
  ```bash
  # 修改测试数据集类别
  python tools/test.py \
      projects/csy_efficientad/configs/efficientad_small.py \
      work_dirs/efficientad_small/iter_70000.pth \
      --cfg-options test_dataloader.Data.subdataset=cable
  ```

#### 保存异常图和 CSV 文件

使用 `EfficientADTestLoop`（已在配置文件中默认启用），运行测试时会**自动生成** `df_test.csv` 和 `df_training.csv`，与 `test_dianziyan.py` 的行为一致。

```bash
python tools/test.py \
    projects/csy_efficientad/configs/efficientad_small.py \
    work_dirs/efficientad_small/iter_70000.pth \
    --cfg-options \
        test_evaluator.save_dir=output_anomaly_maps/dianziyan/anomaly_maps \
        test_evaluator.data_root=/home/ubuntu22/PycharmProjects/PythonProject/EfficientAD-main/Data/100K_dataset \
        test_evaluator.save_format=npy
```

**输出结构**：
```
output_anomaly_maps/dianziyan/
├── anomaly_maps/
│   ├── good/          # 正常样本的异常图（.npy格式）
│   └── bad/           # 异常样本的异常图（.npy格式）
├── df_test.csv        # 测试集CSV文件（自动生成）
└── df_training.csv    # 训练集CSV文件（自动生成）
```

**参数说明**：
- `save_dir`: 异常图保存目录（会创建 `good/` 和 `bad/` 子目录）
- `data_root`: 数据集根目录（用于构建 CSV 中的 `filepath`）
- `save_format`: 保存格式，`'npy'`（与 test_dianziyan.py 一致）或 `'tiff'`，默认为 `'npy'`

**工作原理**：

`EfficientADTestLoop` 会：
1. 首先运行测试集推理，生成 `df_test.csv`
2. 然后自动运行验证集推理，生成 `df_training.csv`
3. 两个 CSV 文件都保存在 `save_dir` 的父目录中（类别目录）

这完全匹配 `test_dianziyan.py` 的行为，无需手动运行两次推理。

## 配置说明

### 测试数据集配置

在配置文件中，`test_dataloader` 定义了测试数据集：

```python
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='EfficientADDataset',
        root=dataset_root,
        dataset_type='mvtec_ad',
        subdataset='bottle',  # 可以修改为其他类别
        split='test',
        image_size=image_size,
    ),
)
```

### 评估指标配置

`test_evaluator` 定义了评估指标：

```python
test_evaluator = dict(
    type='AnomalyMetric',
    save_dir='',  # 设置为非空字符串以保存异常图
)
```

## 输出结果

测试完成后，会输出以下指标：

- **AUC (ROC-AUC)**: 异常检测的 ROC 曲线下面积，用于评估模型性能

如果设置了 `save_dir`，异常图将保存到指定目录，文件结构如下：

```
save_dir/
  ├── good/          # 正常样本的异常图
  ├── defect_type1/  # 缺陷类型1的异常图
  ├── defect_type2/  # 缺陷类型2的异常图
  └── ...
```

## 输出目录结构

训练完成后，会产生以下输出：

```
output/
└── trainings/
    └── mvtec_ad/
        └── bottle/
            ├── teacher_tmp.pth      # 临时模型（每1000次迭代更新）
            ├── student_tmp.pth      # 临时模型（每1000次迭代更新）
            ├── autoencoder_tmp.pth  # 临时模型（每1000次迭代更新）
            ├── teacher_final.pth    # 最终模型（训练结束后）
            ├── student_final.pth    # 最终模型（训练结束后）
            └── autoencoder_final.pth # 最终模型（训练结束后）

work_dirs/efficientad_small/
├── iter_10000.pth
├── iter_20000.pth
├── ...
├── iter_70000.pth
└── last.pth
```

这些模型文件可以直接用于原始 EfficientAD 项目的推理脚本。

## 注意事项

1. **Checkpoint 路径**: 
   - MMEngine 格式的 checkpoint 位于 `work_dirs/<config_name>/` 目录下
   - EfficientAD 格式的模型位于 `output/trainings/<dataset>/<subdataset>/` 目录下

2. **数据集路径**: 确保配置文件中 `dataset_root` 指向正确的数据集目录。

3. **Teacher 模型**: EfficientAD 需要预训练的 teacher 模型，确保 `teacher_checkpoint` 路径正确。

4. **GPU 内存**: 如果遇到 GPU 内存不足，可以减小 `batch_size`。

5. **多 GPU 测试**: 可以使用 `--launcher pytorch` 进行多 GPU 测试：
   ```bash
   python tools/test.py \
       projects/csy_efficientad/configs/efficientad_small.py \
       work_dirs/efficientad_small/iter_70000.pth \
       --launcher pytorch
   ```

6. **输出目录配置**: 可以通过修改配置文件中的 `custom_hooks` 来更改输出目录：
   ```python
   custom_hooks = [
       dict(
           type='EfficientADSaveHook',
           output_dir='./your_output_dir',  # 修改这里
           dataset='mvtec_ad',
           subdataset='bottle',
           interval=1000,
       ),
   ]
   ```

## 完整示例

```bash
# 1. 测试 bottle 类别，保存异常图
python tools/test.py \
    projects/csy_efficientad/configs/efficientad_small.py \
    work_dirs/efficientad_small/iter_70000.pth \
    --work-dir work_dirs/efficientad_small_test \
    --cfg-options test_evaluator.save_dir=output_anomaly_maps/bottle

# 2. 测试 cable 类别
python tools/test.py \
    projects/csy_efficientad/configs/efficientad_small.py \
    work_dirs/efficientad_small/iter_70000.pth \
    --cfg-options test_dataloader.Data.subdataset=cable \
                   test_evaluator.save_dir=output_anomaly_maps/cable
```

---

## ONNX 部署

### 使用 mmdeploy 导出 ONNX

需要使用 `torch2onnx.py` 脚本和 `mmanomaly` 配置：

```bash
# 切换到 trt_export 环境
conda activate trt_export

# 导出 ONNX
python mmdeploy/tools/torch2onnx.py \
    mmdeploy/configs/mmanomaly/anomaly_detection_onnxruntime_dynamic.py \
    projects/csy_efficientad/configs/efficientad_small.py \
    work_dirs/efficientad_small/iter_30000.pth \
    ck4efficientad/bottle/test/good/000.png \
    --work-dir ./deploy_efficientad \
    --device cuda:0
```

### 验证 ONNX 模型

```bash
python -c "
import onnx

model = onnx.load('deploy_efficientad/end2end.onnx')
print('Inputs:')
for i, inp in enumerate(model.graph.input):
    print(f'  {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}')
print('Outputs:')
for i, out in enumerate(model.graph.output):
    print(f'  {out.name}')
"
```

输出示例：
```
Inputs:
  input: [0, 3, 256, 256]
Outputs:
  anomaly_map
```

### ONNX Runtime 推理测试

```bash
python -c "
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession('deploy_efficientad/end2end.onnx', providers=['CUDAExecutionProvider'])
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# 准备输入 (1, 3, 256, 256)
img = np.random.randn(1, 3, 256, 256).astype(np.float32)
output = sess.run([output_name], {input_name: img})[0]
print(f'Output shape: {output.shape}')
"
```

---

## TensorRT 部署

### 配置文件

已创建专门的 TensorRT 配置文件：

- `mmdeploy/configs/efficientad/efficientad_tensorrt_static.py`

### 使用 mmdeploy 导出 TensorRT

```bash
# 切换到 trt_export 环境
conda activate trt_export

# 导出 TensorRT (同时生成 ONNX 和 engine)
python mmdeploy/tools/deploy.py \
    mmdeploy/configs/efficientad/efficientad_tensorrt_static.py \
    projects/csy_efficientad/configs/efficientad_small.py \
    work_dirs/efficientad_small/iter_30000.pth \
    ck4efficientad/bottle/test/good/000.png \
    --work-dir ./deploy_efficientad_trt \
    --device cuda
```

### 输出文件

```
deploy_efficientad_trt/
├── end2end.onnx       # ONNX 模型 (83MB)
└── end2end.engine     # TensorRT Engine (43MB)
```

### TensorRT 推理测试

运行时需要设置 cuDNN 库路径：

```bash
# 设置库路径 (根据实际安装位置调整)
export LD_LIBRARY_PATH=/home/csy/miniconda3/envs/trt_export/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# 验证 engine
python -c "
import tensorrt as trt

with open('./deploy_efficientad_trt/end2end.engine', 'rb') as f:
    engine_data = f.read()

logger = trt.Logger(trt.Logger.ERROR)
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(engine_data)

print(f'Engine created successfully! Tensors: {engine.num_io_tensors}')
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = engine.get_tensor_shape(name)
    print(f'  {i}: {name} {shape}')
"
```

输出示例：
```
Engine created successfully! Tensors: 2
  0: input (-1, 3, 256, 256)
  1: anomaly_map (-1, 1, 56, 56)
```

### 完整 TensorRT 推理代码

```bash
python -c "
import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image

# 加载 engine
with open('./deploy_efficientad_trt/end2end.engine', 'rb') as f:
    engine_data = f.read()

logger = trt.Logger(trt.Logger.ERROR)
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(engine_data)
context = engine.create_execution_context()

# 获取 tensor 信息
input_name = engine.get_tensor_name(0)
output_name = engine.get_tensor_name(1)

# 分配 GPU 内存
d_input = cuda.mem_alloc(1 * 3 * 256 * 256 * 4)  # float32
d_output = cuda.mem_alloc(1 * 1 * 56 * 56 * 4)

# 预处理图像
def preprocess(img_path):
    img = Image.open(img_path).convert('RGB').resize((256, 256), Image.BILINEAR)
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)[np.newaxis]
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    return ((img_np - mean) / std).astype(np.float32)

# 运行推理
def run(img_path):
    img = preprocess(img_path)
    cuda.memcpy_htod(d_input, np.ascontiguousarray(img))
    context.set_input_shape(input_name, (1, 3, 256, 256))
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    success = context.execute_v2([int(d_input), int(d_output)])
    if not success:
        return np.zeros((56, 56), dtype=np.float32)
    h_out = np.empty((1, 1, 56, 56), dtype=np.float32)
    cuda.memcpy_dtoh(h_out, int(d_output))
    return h_out[0, 0]

# 测试
print('[Normal]')
r = run('./ck4efficientad/bottle/test/good/000.png')
print(f'  Score: {r.max():.4f}')

print('[Defective]')
r = run('./ck4efficientad/bottle/test/broken_large/000.png')
print(f'  Score: {r.max():.4f}')

print('Done!')
"
```

---

## 配置文件说明

### 训练配置

`projects/csy_efficientad/configs/efficientad_small.py` 主要参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `model_size` | 模型大小，可选 `small` 或 `medium` | `small` |
| `out_channels` | 输出通道数 | `384` |
| `teacher_checkpoint` | 预训练 teacher 模型路径 | - |
| `teacher_stats_momentum` | teacher 统计更新动量 | `0.01` |
| `quantile` | 硬样本分位数 | `0.999` |
| `lambda_penalty` | 惩罚项权重 | `1.0` |
| `lambda_ae` | 自编码器损失权重 | `1.0` |
| `lambda_stae` | STAE 损失权重 | `1.0` |

### 数据集配置

| 参数 | 说明 |
|------|------|
| `dataset_root` | MVTec AD 数据集根目录 |
| `dataset_type` | 数据集类型，`mvtec_ad` 或 `mvtec_loco` |
| `subdataset` | 子数据集名称，如 `bottle`, `cable` 等 |
| `split` | 数据划分，`train`, `val` 或 `test` |
| `image_size` | 输入图像大小 | `256` |

### 部署配置文件说明

#### ONNX 部署配置

使用 `mmdeploy/configs/mmanomaly/anomaly_detection_onnxruntime_dynamic.py`，关键配置：

```python
onnx_config = dict(
    input_names=['input'],
    output_names=['anomaly_map'],
    dynamic_axes={...},  # 支持动态 batch
)
codebase_config = dict(
    type='mmanomaly',   # 使用 mmanomaly，不是 mmdet
    task='AnomalyDetection',
)
backend_config = dict(type='onnxruntime')
```

#### TensorRT 部署配置

使用 `mmdeploy/configs/efficientad/efficientad_tensorrt_static.py`：

```python
onnx_config = dict(
    type='onnx',
    input_names=['input'],
    output_names=['anomaly_map'],
    save_file='end2end.onnx',
    dynamic_axes={
        'input': {0: 'batch'},      # 只支持 batch 动态
        'anomaly_map': {0: 'batch'},
    },
)

codebase_config = dict(
    type='mmanomaly',
    task='AnomalyDetection',
)

backend_config = dict(
    type='tensorrt',
    common_config=dict(
        max_workspace_size=1 << 30,  # 1GB workspace
        fp16_mode=True,               # 启用 FP16
    ),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 256, 256],
                    opt_shape=[1, 3, 256, 256],
                    max_shape=[4, 3, 256, 256])))
    ],
)
```

#### 配置关键字段说明

| 字段 | 说明 |
|------|------|
| `onnx_config.type` | 必须是 `'onnx'` 才能被 mmdeploy 识别 |
| `onnx_config.save_file` | ONNX 文件保存名 |
| `codebase_config.type` | 使用 `'mmanomaly'`（异常检测任务） |
| `backend_config.type` | 后端类型，`'tensorrt'` 或 `'onnxruntime'` |
| `backend_config.common_config.fp16_mode` | 是否启用 FP16 加速 |
| `backend_config.model_inputs` | TensorRT 优化 profile 配置 |

---

## 常见问题

### 1. TensorRT 转换失败：Dynamic shape axis 错误

确保使用正确的配置文件和导出顺序：
1. 先用 `torch2onnx.py` + `mmanomaly` 配置导出 ONNX
2. 再用 `deploy.py` + `efficientad_tensorrt_static.py` 转换 TensorRT

### 2. TensorRT 运行时找不到 libcudnn

运行时需要设置 LD_LIBRARY_PATH：
```bash
export LD_LIBRARY_PATH=/path/to/cudnn/lib:$LD_LIBRARY_PATH
```

### 3. ONNX 导出时模型输出维度错误

如果模型 forward 返回 tuple，mmdeploy 导出会有问题。已修改 `forward` 方法返回单个 `anomaly_map`（`map_st + map_ae`）。

### 4. mmdeploy 找不到 codebase

使用 `mmanomaly` 配置时，`codebase_config` 的 `type` 应为 `mmanomaly`，不是 `mmdet`。

