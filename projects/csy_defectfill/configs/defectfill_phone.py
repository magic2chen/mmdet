# Config for DefectFill on phone dataset
#
# ─── v2 (2026-07-06) ──────────────────────────────────────────────────────
# Changes vs the previous v1 (iter_20000.pth) that produced greenish-yellow
# blob artifacts on yashang inference:
#
#   lora_rank    8  → 32         # learn subtle press-dent features
#   lora_alpha  16  → 64         # 2:1 alpha/rank ratio preserved
#   lambda_attn 0.05 → 0.5       # force <defect> attention onto mask
#   lambda_obj  0.2 → 0.5        # protect non-mask background
#   max_iters   20000 → 40000     # rank 4× needs ~2× more steps
#   milestones  [15000] → [30000]  # defer LR decay
#   checkpoint  interval 1000 → 2000, max_keep 5 → 3
#   work_dir    → work_dirs/defectfill_phone_v2  (DO NOT clobber v1 ckpts)
#
# Validation/eval unchanged — diagnostic confirmed masks are correctly placed
# on real yashang locations (subtle edge/button dents), so no data changes.
# ──────────────────────────────────────────────────────────────────────────

custom_imports = dict(
    imports=['projects.csy_defectfill'],
    allow_failed_imports=False)

default_scope = 'mmdet'

# Model configuration
model = dict(
    type='DefectFillDetector',
    lora_rank=32,                 # ↑ was 8
    lora_alpha=64,                # ↑ was 16  (2:1 ratio preserved)
    placeholder_token='<defect>',
    pretrained_model_path='./DefectFill/ck',
    lambda_defect=1.0,            # unchanged — defect branch is the primary objective
    lambda_obj=0.5,               # ↑ was 0.2 — better background protection
    lambda_attn=0.5,              # ↑ was 0.05 — force attention to align with mask
    alpha=0.3,                    # unchanged — object loss bg weight
    text_encoder_lr=4e-5,
    unet_lr=2e-4,
    lr_warmup_steps=100,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42,
)

# Image size for training/inference
image_size = 512

# Dataset root - phone data
dataset_root = './DefectFill/DATA/my_train'
# Separate val root: positive (defective) + negative (good) for real AUC
val_dataset_root = './DefectFill/DATA/my_val'

# ─── Subset filter ─────────────────────────────────────────────────────────
# Which defect types participate in training/val/test. Empty list = all
# 8 types (1325 train samples). Override at the CLI:
#
#   --cfg-options defect_types="[yashang]"               # single type
#   --cfg-options defect_types="[yashang,myashang1]"     # multi-type
#   --cfg-options defect_types="[]"                      # all types (same as default)
#
# The same value flows into train / val / test dataloaders so ROC-AUC
# stays consistent with the trained subset.
# ──────────────────────────────────────────────────────────────────────────
defect_types = []

# Training dataloader
train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MVTecDefectDataset',
        root=dataset_root,
        object_class='phone',
        split='train',
        defect_types=defect_types,
        image_size=image_size,
        dilate_mask=False,
        mask_kernel_size=3,
    ),
)

# Validation dataloader (use val split with both defective + good for ROC-AUC)
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTecDefectDataset',
        root=val_dataset_root,
        object_class='phone',
        split='val',
        defect_types=defect_types,
        image_size=image_size,
        dilate_mask=False,
        mask_kernel_size=3,
    ),
)

# Test dataloader (use val split so DefectMetric gets both positive/negative samples)
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTecDefectDataset',
        root=val_dataset_root,
        object_class='phone',
        split='val',
        defect_types=defect_types,
        image_size=image_size,
        dilate_mask=False,
        mask_kernel_size=3,
    ),
)

# Evaluators
val_evaluator = dict(type='DefectMetric', save_dir='')
test_evaluator = dict(type='DefectMetric', save_dir='')

# Training loop configuration
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=40000,                # ↑ was 20000
    val_interval=5000,
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer wrapper
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-5),
    clip_grad=None,
)

# Learning rate scheduler
param_scheduler = [
    dict(
        type='MultiStepLR',
        by_epoch=False,
        milestones=[30000],          # ↑ was [15000]
        gamma=0.1,
    )
]

# Default hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=2000,               # ↑ was 1000
        max_keep_ckpts=3,            # ↓ was 5
        by_epoch=False,
        save_last=True,
    ),
    logger=dict(type='LoggerHook', interval=10),
)

# Log processor
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

# Environment config
env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(), dist_cfg=dict(backend='nccl'))

# Vis backends
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# Log level
log_level = 'INFO'

# Work dir — DO NOT overwrite the v1 ckpt; train into a fresh folder.
work_dir = './work_dirs/defectfill_phone_v2'

# Load from checkpoint (for resuming)
load_from = None

# Resume from checkpoint
resume = False
