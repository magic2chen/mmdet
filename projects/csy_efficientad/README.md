# EfficientAD for MMDetection

本项目将 EfficientAD 集成到 MMDetection 框架中，支持使用标准的 `tools/train.py` 和 `tools/test.py` 脚本进行训练和推理。

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

