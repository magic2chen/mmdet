# Base configuration for CSY_DefectFill
# Usage: python tools/train.py projects/csy_defectfill/configs/defectfill_base.py

custom_imports = dict(
    imports=['projects.csy_defectfill'],
    allow_failed_imports=False)

default_scope = 'mmdet'

# Model configuration
model = dict(
    type='DefectFillDetector',
    lora_rank=8,
    lora_alpha=16,
    placeholder_token='<defect>',
    pretrained_model_path=None,  # Set to local SD checkpoint path, e.g., './ck'
    lambda_defect=1.0,
    lambda_obj=0.2,
    lambda_attn=0.05,
    alpha=0.3,
    text_encoder_lr=4e-5,
    unet_lr=2e-4,
    lr_warmup_steps=100,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42,
)

# Image size for training/inference
image_size = 512

# Dataset root - should contain object_class subdirs with MVTec AD structure
dataset_root = './data/mvtec_ad'

# Training dataloader
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='MVTecDefectDataset',
        root=dataset_root,
        object_class='bottle',  # Override in per-class config
        split='train',
        image_size=image_size,
        dilate_mask=False,
        mask_kernel_size=3,
    ),
)

# Validation dataloader
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTecDefectDataset',
        root=dataset_root,
        object_class='bottle',
        split='val',
        image_size=image_size,
        dilate_mask=False,
        mask_kernel_size=3,
    ),
)

# Test dataloader
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MVTecDefectDataset',
        root=dataset_root,
        object_class='bottle',
        split='test',
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
    max_iters=20000,
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
        milestones=[15000],
        gamma=0.1,
    )
]

# Default hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        max_keep_ckpts=3,
        by_epoch=False,
        save_last=True,
    ),
    logger=dict(type='LoggerHook', interval=100),
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

# Load from checkpoint (for resuming)
load_from = None

# Resume from checkpoint
resume = False