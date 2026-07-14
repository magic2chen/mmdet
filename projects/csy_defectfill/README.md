# CSY_DefectFill

基于 **Stable Diffusion 2 Inpainting + LoRA + Textual Inversion** 的工业缺陷生成 / 检测模型，
集成于 mmdetection 3.x。

训练完成后可以做两件事:
- **异常检测**:用模型对一张图做 "inpaint 补成正常外观",缺陷区域 LPIPS 高,正常区域 LPIPS 低 →
  得到图像级 anomaly score → ROC-AUC 评估(`tools/test.py`)
- **缺陷合成生成**:在 `good 图 + 缺陷 mask` 上生成最像真实缺陷的候选(`projects/csy_defectfill/tools/infer_generate.py`)

---

## 目录结构

```
projects/csy_defectfill/
├── __init__.py                    # 导出 DefectFillDetector / DefectFillCore / MVTecDefectDataset / DefectMetric
├── configs/
│   ├── defectfill_base.py         # 通用基础配置(默认 object_class='bottle')
│   ├── defectfill_phone.py        # phone 全类别训练配置(8 类 defect,1325 张)
│   ├── defectfill_phone_infer.py  # phone 推理配置(LoRA rank 32)
│   └── defectfill_phone_yashang.py# phone yashang 单类微调配置(细训自 v2)
├── models/
│   ├── defectfill_core.py         # DefectFillCore:SD 2 Inpainting + LoRA + Textual Inversion
│   ├── defectfill_model.py        # DefectFillDetector:MMDET 兼容的训练/推理封装
│   └── loss.py                    # DefectFillLoss
├── datasets/
│   └── mvtec_defect_dataset.py    # MVTecDefectDataset
├── metrics/
│   └── defect_metric.py           # DefectMetric(ROC-AUC)
└── tools/
    └── infer_generate.py          # 缺陷合成生成脚本(独立于 TestLoop)
```

外部数据 / 权重(在仓库根 `DefectFill/` 下,**不在** projects/ 内,也没有 Python 代码):
```
mmdet/DefectFill/
├── DATA/
│   ├── my_train/                  # 训练集(1325 张,8 类 defect)
│   └── my_val/                    # 验证集(64 正 + 16 负)
└── ck/                            # SD 2 Inpainting 本地权重(~25GB,必须有 scheduler/ 子目录)
```

> 🔑 **2026-07 重构说明**:原 `mmdet/DefectFill/model.py` 已合并到
> `projects/csy_defectfill/models/defectfill_core.py`(类重命名为 `DefectFillCore`),
> 不再走 `sys.path.insert` hack。`mmdet/DefectFill/` 下只剩数据和权重。

---

## 1. 训练环境

| 组件 | 版本 | 备注 |
|------|------|------|
| OS | WSL2 Ubuntu 22.04 / Linux 原生 | 已验证在 WSL 下训练,Windows 文件路径用 `/mnt/d/csy/mmdet` 访问 |
| Python | 3.10 | |
| PyTorch | 2.1.2 + cu121 | |
| CUDA Toolkit | 12.1 | |
| mmcv | 2.1.0 | **必须** 从 openmmlab 预编译源安装(默认 PyPI 没 cp310+cu121+torch2.1 组合) |
| mmengine | 0.10.7 | |
| mmdet | 3.3.0 | |
| numpy | <2 | 与 torch 2.1 ABI 兼容 |
| diffusers | 0.27 ≤ x < 0.31 | pin 范围 |
| transformers | 4.40 ≤ x < 4.50 | pin 范围(否则 pip 会拉到 5.x,与 torch 2.1 不兼容) |
| peft | 0.10 ≤ x < 0.13 | pin 范围 |
| huggingface_hub | 0.20 ≤ x < 0.27 | pin 范围 |
| tokenizers | 0.19 ≤ x < 0.21 | pin 范围 |
| accelerate | 0.34 ≤ x < 0.36 | pin 范围 |
| scikit-learn | latest | DefectMetric 用 |
| opencv-python | 4.x | dataset / infer_generate 用 |
| lpips | 含在 DefectFillCore 内 | |

### 一次性安装命令

```bash
# 0. torch / mmcv / mmengine / mmdet
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html mmcv==2.1.0
pip install mmengine==0.10.7
pip install mmdet==3.3.0

# 1. pin HF 生态(顺序敏感:先 tokenizers,再 transformers)
pip install "tokenizers>=0.19,<0.21" \
            "transformers>=4.40,<4.50" \
            "huggingface_hub>=0.20,<0.27" \
            "accelerate>=0.34,<0.36" \
            "peft>=0.10,<0.13" \
            "diffusers>=0.27,<0.31"

# 2. 其余依赖
pip install "numpy<2" scikit-learn opencv-python
```

---

### 1.1 运行前必须设置 `PYTHONPATH`（⚠ 首次训练必看）

`configs/defectfill_phone.py` 顶部声明了：

```python
custom_imports = dict(
    imports=['projects.csy_defectfill'],
    allow_failed_imports=False)
```

要求 `from projects.csy_defectfill import ...` 在 import 配置时就能成功。
但当你跑 `python tools/train.py ...` 时，Python **只把脚本所在目录
`tools/` 加进 `sys.path[0]`，不会自动把项目根目录加进去**，于是首次
训练会报：

```
ModuleNotFoundError: No module named 'projects'
ImportError: Failed to import custom modules from {'imports': ['projects.csy_defectfill'], ...}
You should set `PYTHONPATH` to make `sys.path` include the directory which contains your custom module
```

**解决:把仓库根目录 `/mnt/d/csy/mmdet` 加到 `PYTHONPATH`**。

```bash
# 方式 A：单次命令生效（推荐新手先试这个）
PYTHONPATH=/mnt/d/csy/mmdet python tools/train.py ...

# 方式 B：当前 shell 永久生效
export PYTHONPATH=/mnt/d/csy/mmdet:$PYTHONPATH

# 方式 C：写到 ~/.bashrc，每次新 shell 自动生效（常驻推荐）
echo 'export PYTHONPATH=/mnt/d/csy/mmdet:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

> 🔧 **为什么 MMEngine 不自动处理？** `projects/` 在 MMEngine 看来是
> custom import,它要求项目根目录在 *cfg 被 import 之前*就已经在
> `sys.path` 里。这是 `custom_imports` 机制的设计选择——所有走
> custom_imports 的项目都得手动配 PYTHONPATH,不是本项目特有的问题。
>
> ⚠️ `python -m tools.train` 也会让 cwd 进入 sys.path,但会绕过
> argparse 里 `--resume` 的 `nargs='?'` 解析,行为不一样,故**不推荐**
> 用 `-m` 方式绕开。

---

## 2. 训练预训练权重(SD 2 Inpainting)

模型基于 **Stable Diffusion 2 Inpainting** (`sd2-community/stable-diffusion-2-inpainting`),
**必须**本地化放到 `mmdet/DefectFill/ck/` 下(HF 在线下载受网络限制不可用)。

### 准备方式

1. 从 HuggingFace `sd2-community/stable-diffusion-2-inpainting` 下载整个 snapshot
2. 解压到 `mmdet/DefectFill/ck/`,目录里必须含以下子目录/文件:

```
DefectFill/ck/
├── scheduler/             # 必需,cfg 里 model.pretrained_model_path 指向此目录
│   └── scheduler_config.json
├── text_encoder/
├── tokenizer/
├── unet/
├── vae/
├── feature_extractor/
├── model_index.json
├── 512-inpainting-ema.safetensors   # 或 .ckpt
└── ...
```

总大小约 **25GB**。

### 在 cfg 中的引用

```python
model = dict(
    type='DefectFillDetector',
    pretrained_model_path='./DefectFill/ck',   # 相对路径,要在 mmdet 根目录下运行
    lora_rank=32,
    lora_alpha=64,
    placeholder_token='<defect>',
    ...
)
```

> ⚠️ 注意:cfg 里 `pretrained_model_path` 路径是**相对 mmdet 仓库根目录**的相对路径。
> 训练启动必须在 `/mnt/d/csy/mmdet` 目录下执行。

---

## 3. 训练数据

`MVTecDefectDataset` 沿用 MVTec AD 目录约定,但 train/val 语义做了扩展(支持 per-defect-type 过滤)。

### 数据集目录约定

```
DATA_ROOT/
└── phone/                              # object_class
    ├── train/                          # split='train'
    │   ├── defective/
    │   │   ├── baidian/*.png          # 8 类缺陷子目录
    │   │   ├── cashang/*.png
    │   │   └── ...                    # myashang1 / yashang / mhuashang 等
    │   └── defective_masks/
    │       ├── baidian/{原图名}_mask.png
    │       └── ...                     # 与 defective/ 同名,带 _mask.png 后缀
    └── val/                            # split='val'
        ├── defective/<type>/*.png      # 正样本
        ├── defective_masks/<type>/*.png
        └── good/*.png                  # 负样本,**必须有**(ROC-AUC 必备)
```

> 🔑 **val 集必须同时含 `defective/` 和 `good/`**;缺一会让 AUC 退化成 0.5。
> mask 命名规则:`<basename>_mask.png`;不一致时 dataset 会回退到同名目录模糊匹配。

### 当前 phone 数据实测清单

| 路径 | split | 样本数 | 用途 |
|------|-------|--------|------|
| `DefectFill/DATA/my_train/phone/train/` | train | 1325 | 8 类 defect 全量训练用 |
| `DefectFill/DATA/my_val/phone/val/` | val | 80 | 64 正(8 类 × 8 张) + 16 负 |

各类样本数明细(train):

| 类别 | 含义 | 样本数 |
|------|------|--------|
| baidian | 白点 | 17 |
| cashang | 擦伤 | 233 |
| mhuashang | 划伤 | 89 |
| mhuashang_bak | 划伤备份 | 94 |
| mliangdian | 亮点 | 25 |
| mtusuan | 凸点 | 30 |
| myashang1 | 压伤大类 | 808 |
| **yashang** | **压伤子类(本次单类训练目标)** | **29** |

val 中 yashang 子类 8 张正样本 + good 16 张 → 共 24 张可用于 yashang 专项 ROC-AUC。

---

## 4. 训练启动脚本

训练入口统一走 mmdetection 的 `tools/train.py`,传入 cfg 即可。三套典型工作流:

### 4.1 通过参数指定参与训练的缺陷类别(推荐)

`configs/defectfill_phone.py` 顶部有一个顶层变量 `defect_types`(默认空列表 = 全类别 1325 张),
通过 `--cfg-options defect_types="[...]"` 在命令行覆盖。`tools/train.py` 会在 cfg 加载完后
**自动**把顶层 `defect_types` 同步到 train / val / test 三个 dataset,所以**只需在顶层声明一次**。
ROC-AUC 与训练子集保持一致。

> ⚙️ **实现细节**:cfg 文件里 `dataset=dict(defect_types=defect_types, ...)` 在 Python 求值时
> 已经把空 list 引用拷进 dict 了,`--cfg-options` 改顶层 var 改不到 dataset。
> 所以 `tools/train.py::main()` 显式调了一次 `_propagate_defect_types(cfg)` 来同步。
> 如果你写了自己的 train 脚本,记得照搬这段(否则 dataset.defect_types 永远是默认空 list)。

```bash
cd /mnt/d/csy/mmdet

# 1. 全类别(默认行为)
python tools/train.py projects/csy_defectfill/configs/defectfill_phone.py

# 2. 只训 yashang(29 张)+ 短训练 + 细训自 v2
python tools/train.py projects/csy_defectfill/configs/defectfill_phone.py \
    --cfg-options defect_types="[yashang]" \
                 max_iters=4000 \
                 load_from=./work_dirs/defectfill_phone_v2/iter_40000.pth \
                 param_scheduler='[dict(by_epoch=False,gamma=0.1,milestones=[2500],type="MultiStepLR")]'

# 3. 训两类压伤(yashang + myashang1,共 837 张)
python tools/train.py projects/csy_defectfill/configs/defectfill_phone.py \
    --cfg-options defect_types="[yashang,myashang1]"

# 4. 排除某几类(用空 list 不行,需要"白名单"语义;若要排除用反向 mask 后传给 cfg)
#    本项目暂只支持白名单,排除需求请编辑 cfg。
```

| `defect_types` 取值 | train 样本数 | val 正样本数 | 说明 |
|---|---|---|---|
| `[]`(默认) | 1325 | 64(8 类 × 8) | 全类别基线 |
| `[yashang]` | 29 | 8 | 单类细训 |
| `[yashang, myashang1]` | 837 | 16 | 两类压伤合训 |
| `[cashang]` | 233 | 8 | 单类擦伤 |

> ⚠️ **`<defect>` placeholder 不变**:无论子集怎么选,训练的占位 token 永远是
> `'<defect>'`,prompt 永远是 `A phone with <defect>`(见 `models/defectfill_model.py`)。
> 子集只决定"用哪些样本教这个 token",不会改 token 本身的语义。

### 4.2 预设 cfg(便捷封装)

- `defectfill_phone.py` — 主配置,**所有参数可通过 `--cfg-options` 覆盖**
- `defectfill_phone_yashang.py` — yashang 细训预设(`_base_` 继承主 cfg,只改 `defect_types=['yashang']` +
  `max_iters=4000` + `load_from=v2/iter_40000.pth`)
- `defectfill_base.py` — 通用基础(默认 `object_class='bottle'`,需手动指定数据路径)
- `defectfill_phone_infer.py` — 推理配置(给 `tools/infer_generate.py` 用)

```bash
# yashang 预设(等价于 4.1 第 2 条的子集,但已经写好了 iters / load_from)
python tools/train.py projects/csy_defectfill/configs/defectfill_phone_yashang.py
```

### 4.3 其他常用覆盖

```bash
# 切到 bottle 数据集(需保证 DefectFill/DATA/my_train/bottle/... 存在)
python tools/train.py projects/csy_defectfill/configs/defectfill_phone.py \
    --cfg-options \
        model.pretrained_model_path=/abs/path/to/sd2-inpainting \
        train_dataloader.dataset.object_class=bottle \
        train_dataloader.batch_size=1 \
        train_cfg.max_iters=5000
```

### 4.4 训练监控

- 日志:`work_dirs/<work_dir>/<时间戳>/<时间戳>.log`,每 10 iter 一行
- TensorBoard:`work_dirs/<work_dir>/<时间戳>/vis_data/<时间戳>.json`
- 关键 loss 字段:
  - `loss_defect`:在 GT mask 区域监督 inpaint
  - `loss_object`:在随机 mask 区域监督 inpaint(保护 object 完整性)
  - `loss_attn`:`<defect>` token 注意力对齐 mask 的 L2
- 健康轨迹:`loss_attn` 应在 0.001~0.005,`loss_defect+loss_object` 在 1.5~2.5 区间(LPIPS 量级)。

### 4.5 中断后恢复训练（`--resume`）

`tools/train.py` 自带 `--resume` 参数:训练过程中 `CheckpointHook`
每 2000 iter 自动存一份 `.pth` 到 `work_dir`,并在 work_dir 根目录
维护一个 `last_checkpoint` 文件指向最新 ckpt。中断(断电 / OOM /
Ctrl+C / 机器重启)后可以一键续训。

```bash
cd /mnt/d/csy/mmdet

# 方式 1:自动续训 —— 从 work_dir/last_checkpoint 恢复
# 同时恢复:模型权重 + 优化器状态 + LR scheduler 状态 + 当前 iter 计数
PYTHONPATH=/mnt/d/csy/mmdet python tools/train.py \
    projects/csy_defectfill/configs/defectfill_phone.py \
    --cfg-options defect_types="[yashang,myashang1]" \
    --resume

# 方式 2:指定某个 ckpt 续训(回退到 iter_8000.pth 然后继续)
PYTHONPATH=/mnt/d/csy/mmdet python tools/train.py \
    projects/csy_defectfill/configs/defectfill_phone.py \
    --cfg-options defect_types="[yashang,myashang1]" \
    --resume=./work_dirs/defectfill_phone_v2/iter_8000.pth
```

启动后日志开头会打印 `Resume from ...` 并从断点 iter 继续。

| ⚠ 关键区别 | 含义 |
|---|---|
| `--resume` 或 `--resume=path/to/xxx.pth` | load 权重 + 优化器状态 + LR scheduler + iter 计数(**中断恢复的正确方式**) |
| `--cfg-options load_from=...` | 只 load 权重;其他全部重置 → 冷启动 + 加权重,训练行为不可预测 |

> 💡 `work_dir` 不动 —— `--resume` 会自动续写进同一个 work_dir 下的
> 新时间戳子目录,`last_checkpoint` 同步更新,不会污染之前的 ckpt。
>
> 💡 `default_hooks.checkpoint.max_keep_ckpts=3`,只保留最近 3 个
> ckpt + last。手动清理旧版本(v1/v2)的大文件前确认已无依赖。

### 4.6 调整训练迭代次数（`train_cfg.max_iters`）

本项目走的是 **IterBasedTrainLoop**(迭代式,不是 epoch loop),
总迭代数由 `train_cfg.max_iters` 控制。

**方式 A:命令行 `--cfg-options` 覆盖(临时改,最常用)**

```bash
# 例子:从 iter_12000 续跑到 60000,同时把 LR 衰减点也推到 50000
PYTHONPATH=/mnt/d/csy/mmdet python tools/train.py \
    projects/csy_defectfill/configs/defectfill_phone.py \
    --cfg-options defect_types="[yashang,myashang1]" \
                 train_cfg.max_iters=60000 \
                 param_scheduler.0.milestones="[50000]" \
    --resume=./work_dirs/defectfill_phone_v2/iter_12000.pth
```

> ⌨️ `--cfg-options key.subkey=value` 走 MMEngine `DictAction` 的
> 点号嵌套语法,等价于 Python `cfg["key"]["subkey"] = value`。

**方式 B:直接编辑 cfg 文件(持久化)**

编辑 `projects/csy_defectfill/configs/defectfill_phone.py`:

```python
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=60000,        # ← 改这里
    val_interval=5000,
)
```

**改 max_iters 后的联动项(容易踩坑)**

只改 `max_iters` 不调 LR 节奏的话,scheduler 会在还没收敛时提前衰减,
或者跑完 max_iters 了 LR 都没动过一次。要根据新目标同步以下参数:

| 配置项 | 默认值 | 作用 | max_iters 改了后建议 |
|---|---|---|---|
| `train_cfg.max_iters` | 40000 | 总迭代数 | 改成目标值 |
| `param_scheduler[0].milestones` | `[30000]` | LR 衰减点 | 推到接近 max_iters 的位置(如 60000 → `[50000]`) |
| `param_scheduler[0].gamma` | 0.1 | 每次衰减倍数 | 一般不动 |
| `train_cfg.val_interval` | 5000 | 验证间隔 | 不动 |
| `default_hooks.checkpoint.interval` | 2000 | 存盘间隔 | 不动 |

**缩短训练的特殊情况**

如果想缩到比当前 iter 还小的值(例如当前在 12000,改成 10000
跑完即停),直接 `--cfg-options train_cfg.max_iters=10000 --resume`
会**立刻结束**(runner 判定"已超过 max_iters")。两种处理:

- **回退到更早的 ckpt**:`--resume=./work_dirs/.../iter_8000.pth train_cfg.max_iters=10000`
- **强制扩 iter 再续训**:把 max_iters 改成 ≥ 当前 iter,跑完

> ⚠ **坑**:`--cfg-options max_iters=...`(**不带 `train_cfg.` 前缀**)
> 只会创建顶层 `max_iters` 键,**不会**改 `train_cfg.max_iters`——
> 必须带前缀。改 cfg 文件同理,必须找 `train_cfg = dict(...)` 里
> 那一行,不要去改顶层。

---

## 5. 推理:查看生成的缺陷图片

### 5.1 缺陷合成生成(看视觉效果)

用 `projects/csy_defectfill/tools/infer_generate.py`,给定 `(good 图, 缺陷 mask)` 对,
生成 N 个候选并选 LPIPS-argmax 那张作为"最像真实缺陷"的输出。

**数据 layout 假设**:
- 默认走 DefectFill2 的 `<root>/<class>/test/good` + `<root>/<class>/train/defective_masks/<type>` 配对
- 也可以 `--good-dir` + `--mask-dir` **显式指定**,绕过布局假设(本项目默认用这个,因为 good 图只在 val/good/ 下)

#### 命令模板(已实测)

```bash
cd /mnt/d/csy/mmdet

python projects/csy_defectfill/tools/infer_generate.py \
    projects/csy_defectfill/configs/defectfill_phone_infer.py \
    work_dirs/<your_ckpt>.pth \
    --good-dir DefectFill/DATA/my_val/phone/val/good \
    --mask-dir DefectFill/DATA/my_train/phone/train/defective_masks/<defect_type> \
    --object-class phone \
    --defect-type <defect_type> \
    --output-dir work_dirs/<your_infer_output> \
    --total-images 4 \
    --num-samples 4 \
    --steps 50 \
    --batch-size 2 \
    --guidance-scale 2.0 \
    --seed 0
```

#### 关键参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--good-dir` | (无) | good 图目录;若指定 `--mask-dir` 也必须给 |
| `--mask-dir` | (无) | mask 模板目录 |
| `--defect-type` | 必填 | 用于 prompt / 输出命名 |
| `--total-images` | 8 | 生成多少张(从 good 目录里循环) |
| `--num-samples` | 8 | 每张图采样几个候选,选 LPIPS 最大的 |
| `--steps` | 50 | DDIM 步数 |
| `--batch-size` | 4 | 候选 batch(显存够可调大) |
| `--guidance-scale` | 2.0 | CFG,**训练时是 7.5**——推理效果差时优先拉这个 |
| `--seed` | 0 | 随机种子(逐图加 1 偏移) |
| `--dilate-mask` | False | 是否膨胀 mask |
| `--mask-kernel-size` | 3 | 膨胀 kernel 半径 |

#### 产物结构

```
<output-dir>/
├── <defect-type>/
│   ├── 0000_original.png      # 输入 good 图(裁剪到 512)
│   ├── 0000_mask.png          # 套上去的 mask
│   ├── 0000_generated.png     # ★ 生成的缺陷图(看这个)
│   ├── 0001_*.png
│   └── ...
└── inference_log.json          # 每个样本的 best_lpips + 全部 LPIPS 分数
```

#### yashang 推理实战示例

```bash
# 用 v2 ckpt 跑 yashang 生成
python projects/csy_defectfill/tools/infer_generate.py \
    projects/csy_defectfill/configs/defectfill_phone_infer.py \
    work_dirs/defectfill_phone_v2/iter_40000.pth \
    --good-dir DefectFill/DATA/my_val/phone/val/good \
    --mask-dir DefectFill/DATA/my_train/phone/train/defective_masks/yashang \
    --object-class phone --defect-type yashang \
    --output-dir work_dirs/defectfill_phone_v2/infer_yashang \
    --total-images 4 --num-samples 4 --steps 50 \
    --batch-size 2 --guidance-scale 7.5 --seed 0
```

调参 tips:
- **生成的缺陷不明显**:`--guidance-scale` 拉到 5.0~7.5(训练对齐值)
- **生成的缺陷过头、像非 phone**:`--guidance-scale` 降到 1.5
- **LPIPS 候选都差不多、想看多样性**:`--num-samples 8` 或 16

### 5.2 异常检测评测(算 ROC-AUC)

用 `tools/test.py`,跑完整 val 集,自动计算 anomaly_score + AUC。

```bash
cd /mnt/d/csy/mmdet

python tools/test.py \
    projects/csy_defectfill/configs/defectfill_phone.py \
    work_dirs/<your_ckpt>.pth \
    --work-dir work_dirs/<test_output> \
    --show-dir work_dirs/<test_output>/vis
```

可选:
```bash
# 同时存每张图的 anomaly map(.npy 格式)
python tools/test.py \
    projects/csy_defectfill/configs/defectfill_phone_yashang.py \
    work_dirs/defectfill_phone_yashang/iter_4000.pth \
    --work-dir work_dirs/defectfill_phone_yashang/test \
    --show-dir work_dirs/defectfill_phone_yashang/test/vis \
    --cfg-options test_evaluator.save_dir=work_dirs/defectfill_phone_yashang/test/anomaly_maps \
        test_evaluator.save_format=npy
```

跑完关键看:
- 终端最后打印的 **`auc=...`**(8 正 + 16 负 = 24 张)
- `vis/` 下每张图的可视化(含原图、inpaint 结果、anomaly map 叠加)
- `anomaly_maps/` 下的 `.npy` / `.tiff`(用于自己后续可视化)

### 5.3 直接调用 Python API(脱离 mmengine)

```python
import torch
from projects.csy_defectfill import DefectFillDetector

model = DefectFillDetector(
    pretrained_model_path='./DefectFill/ck',
    placeholder_token='<defect>',
    lora_rank=32, lora_alpha=64,
)
state = torch.load('work_dirs/defectfill_phone_v2/iter_40000.pth',
                   map_location='cpu')['state_dict']
model.load_state_dict(state, strict=False)
model = model.cuda().eval()

with torch.no_grad():
    result = model.test_step({
        'img': img,                # [1, 3, 512, 512] on cuda, [-1, 1]
        'mask': mask,              # [1, 1, 512, 512] on cuda, [0, 1]
        'is_defect': torch.tensor([0]),
        'object_class': ['phone'],
    })
    anomaly_score = result[0].anomaly_score     # 0-D tensor,越高越像缺陷
```

---

## 故障排查

| 症状 | 原因 | 解决 |
|------|------|------|
| `size mismatch for lora_A.default.weight` | cfg 的 `lora_rank/alpha` 与 checkpoint 不一致 | 改 cfg 里的 `lora_rank` / `lora_alpha` 与训练时一致(本次 yashang 是 32 / 64) |
| `DefectFillCore is not importable` | `projects/csy_defectfill/models/defectfill_core.py` 缺失 | 确认文件存在;`python -c "from projects.csy_defectfill.models.defectfill_core import DefectFillCore"` 验证 |
| `mmcv/_ext.so undefined symbol: c10_cuda_check_implementation` | torch / mmcv ABI 不匹配 | 严格按 README 装 torch 2.1.2 + mmcv 2.1.0 from cu121/torch2.1 index |
| `ImportError: huggingface_hub` 循环 import | transformers 升到 5.x | 按 pin 矩阵重装 HF 生态 |
| `auc: 0.5000` | val 集只有 defect 或只有 good | 把 val root 改成同时含 `defective/` 和 `good/` 的目录 |
| `auc: 0.0000` | 训练不充分 / 数据排序问题 | 跑满 4000 iter(yashang)或 40000 iter(v2);smoke 阶段 0.0 正常 |
| `DefaultSampler` 在 0 样本上崩溃 | val 集路径错了,dataset 加载 0 个样本 | `python -c "from projects.csy_defectfill.datasets import MVTecDefectDataset; ds = MVTecDefectDataset(root='./DefectFill/DATA/my_val', object_class='phone', split='val', defect_type='yashang'); print(len(ds))"` 验证 |
| `DataLoader worker (pid xxx) is killed` | `num_workers>0` + WSL 共享 Windows 盘 IO 慢 | `num_workers=0` |
| 推理生成的图颜色怪、不像 defect | `--guidance-scale` 与训练时不匹配 | 训练时是 7.5;推理默认 cfg 是 2.0,改成 7.5 通常立竿见影 |
| `--cfg-options defect_types=[yashang]` 不生效 / 报 `KeyError` | 引号没加,shell 把方括号展开 | 必须加双引号:`--cfg-options defect_types="[yashang]"`,mmengine 会解析为 Python list |
| `ModuleNotFoundError: No module named 'projects'` 首次训练报错 | `python tools/train.py ...` 不会自动把项目根目录加进 `sys.path` | `export PYTHONPATH=/mnt/d/csy/mmdet:$PYTHONPATH` 后再跑(详见 §1.1) |
| 训练中断后重启却发现 iter 从 0 开始 / loss 突然跳变 | 用错了 `load_from=` 而不是 `--resume`,只 load 了权重 | 必须用 `--resume` 或 `--resume=path/to/xxx.pth`,会一并恢复优化器 + LR scheduler + 当前 iter 计数(详见 §4.5) |
| `--cfg-options train_cfg.max_iters=N` 后训练立刻结束 / 0 iter 就退 | 当前 iter 已经 ≥ N,runner 判定已完成 | 用更早 ckpt:`--resume=.../iter_8000.pth`,或把 N 调大(详见 §4.6) |
| `--cfg-options max_iters=N` 没生效,训练照旧跑到原 max_iters | 没带 `train_cfg.` 前缀,新键只创建在 cfg 顶层,没碰到训练循环 | 必须写成 `--cfg-options train_cfg.max_iters=N`(详见 §4.6) |

---

## 参考

- mmdetection 3.x:https://github.com/open-mmlab/mmdetection
- mmengine 0.10.7:https://github.com/open-mmlab/mmengine
- SD 2 Inpainting:https://huggingface.co/sd2-community/stable-diffusion-2-inpainting
- MVTec AD 数据集:https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
- DefectFill 原始 repo(部分借鉴):`DefectFill2/`(独立目录,不在 mmdet 里)