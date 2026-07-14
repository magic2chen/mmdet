# DefectFill 数据工具集

`projects.csy_defectfill` 项目的**数据准备与后处理**命令行工具。所有脚本都是
**自包含的**（只依赖 stdlib + `cv2` / `numpy` / `PIL` / `tqdm`），不依赖 mmdet
或 `DefectFill/` 包里的代码，可以独立运行。

原位置：`mmdet/DefectFill/tools/`（2026-07 重构时搬过来）。
`mmdet/DefectFill/` 现在只保留数据 (`DATA/`) 和 SD 权重 (`ck/`)。

## 脚本列表

| 脚本 | 用途 | 模式 |
|------|------|------|
| `filter_labelme_by_label` | 按关键字筛选 labelme 标注（含子目录的 json），命中后**按实际标签名分目录**输出 | 批处理 |
| `labelme_split_by_category` | labelme 标注按**所有出现过的标签类别**拆分到子目录（无关键字过滤） | 批处理 |
| `labelme_to_mask_and_crop` | labelme → mask 生成（输出 `*_mask.png`，**与 DefectFill 推理 1:1 配对**）或缺陷块裁剪 | `--mode mask` / `--mode crop` |
| `generate_glow` | 根据 labelme 多边形生成**中心亮边缘暗的高斯亮点**，叠加到原图上 | 批处理（cwd 找 `*.json`） |
| `paste_back` | 把 512x512 推理结果**粘贴回原始大图**（带边缘融合），单张/批量两种模式 | 单张 / 批处理 |

## 用法

所有脚本都可以通过 `python -m tools.defectfill_utils.<name>` 调用，也可以
`python tools/defectfill_utils/<name>.py` 直接跑。

### 1. `filter_labelme_by_label`

labelme 目录中按"标签字段"筛选（只保留 label 包含指定关键字的图片+json，
并按实际标签名分目录输出）。

```bash
python -m tools.defectfill_utils.filter_labelme_by_label \
    --input_dir /path/to/labelme \
    --keyword 压伤
```

可选：
- `--output_dir` 自定义输出根目录
- `--case_sensitive` 区分大小写

### 2. `labelme_split_by_category`

把 labelme 目录**按全部出现过的标签类别**拆分到子目录（无关键字过滤）。
同名 json 自动加 8 位 hash 后缀避免覆盖。

```bash
python -m tools.defectfill_utils.labelme_split_by_category \
    --input_dir /path/to/labelme \
    --output_dir /path/to/output
```

可选：
- `--skip_empty_label` 忽略无标签的 shape

### 3. `labelme_to_mask_and_crop`

**Mask 生成**（与 `DefectFill` 推理 1:1 配对：原图存到 `good/`，mask 存到
`masks/`，命名 `xxx.png` / `xxx_mask.png`）：

```bash
python -m tools.defectfill_utils.labelme_to_mask_and_crop \
    --mode mask \
    --input_dir /path/to/labelme \
    --output_dir /path/to/mask_out \
    --object_class mhuashang
```

**缺陷块裁剪**（按 mask 中心 + padding 切 256x256 patch）：

```bash
python -m tools.defectfill_utils.labelme_to_mask_and_crop \
    --mode crop \
    --input_dir /path/to/labelme \
    --output_dir /path/to/patches \
    --patch_size 256
```

### 4. `generate_glow`

按 labelme 多边形位置生成**高斯亮点**（中心亮边缘暗，随机扰动中心），
把亮点叠加到原图上输出 `*_glow.png`。**只处理 cwd 下的 `*.json`**。

```bash
cd /path/to/labelme
python -m tools.defectfill_utils.generate_glow
```

输出到 cwd 下 `liangdian/` 目录。

### 5. `paste_back`

把 512x512 推理结果粘贴回原图。带**边缘融合**（默认 margin=10 像素），
避免接缝。算法是 `DefectFill.inference.smart_crop_dynamic` 的反向操作，
但实现里**独立写了一份** `smart_crop_get_coords`（不依赖 `DefectFill/`）。

**批处理**模式（根据 mask 中心定位裁剪区，批量处理整个推理目录）：

```bash
python -m tools.defectfill_utils.paste_back \
    --inference_dir /path/to/inference_results \
    --data_dir /path/to/my_train \
    --object_class phone \
    --defect_type baidian \
    --output_dir /path/to/pasted \
    --create_comparison
```

**单张**模式：

```bash
python -m tools.defectfill_utils.paste_back \
    --original /path/to/orig.png \
    --mask /path/to/orig_mask.png \
    --generated /path/to/0000_generated.png \
    --output_dir /path/to/single_out
```

可选：
- `--blend_margin N` 边缘融合宽度
- `--dilate_mask --mask_kernel_size K` mask 膨胀（与训练时一致）
- `--create_comparison` 输出原图/粘贴/差异三联图

## 依赖

| 包 | 用途 | 安装 |
|----|------|------|
| `opencv-python` | 图像 I/O、polygon fill、resize | `pip install opencv-python` |
| `numpy` | 数组运算 | `pip install numpy` |
| `Pillow` | PIL Image 读取 | `pip install Pillow` |
| `tqdm` | 进度条 | `pip install tqdm` |

无 mmdet / mmcv / torch 依赖。

## 重构历史

- 2026-07：原 `mmdet/DefectFill/tools/` 整体迁移至此。`DefectFill/` 目录
  不再含 Python 代码，只留 `DATA/` 和 `ck/`。
