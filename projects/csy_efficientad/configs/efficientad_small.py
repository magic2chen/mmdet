custom_imports = dict(
    imports=['projects.csy_efficientad'], allow_failed_imports=False)

default_scope = 'mmdet'

model = dict(
    type='EfficientADModel',
    model_size='medium',
    out_channels=384,
    teacher_checkpoint=(
        '/home/ubuntu22/PycharmProjects/PythonProject/mmdetection/checkpoints/efficientad/models/teacher_medium.pth'),
    teacher_stats_momentum=0.01,
    quantile=0.999,
    lambda_penalty=1.0,
    lambda_ae=1.0,
    lambda_stae=1.0,
)

image_size = 256
dataset_root = ('/home/ubuntu22/PycharmProjects/PythonProject/mmdetection/Data/dianziyan/100K_dataset')

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='EfficientADDataset',
        root=dataset_root,
        dataset_type='mvtec_ad',
        subdataset='dianziyan',
        split='train',
        image_size=image_size,
        imagenet_train_path='none',
        seed=42,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='EfficientADDataset',
        root=dataset_root,
        dataset_type='mvtec_ad',
        subdataset='dianziyan',
        split='val',
        image_size=image_size,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='EfficientADDataset',
        root=dataset_root,
        dataset_type='mvtec_ad',
        subdataset='dianziyan',
        split='test',
        image_size=image_size,
    ),
)

val_evaluator = dict(
    type='AnomalyMetric',
    save_dir='',
)

test_evaluator = dict(
    type='AnomalyMetric',
    save_dir='/home/ubuntu22/PycharmProjects/PythonProject/mmdetection/work_dirs/efficientad_small/output_anomaly_maps/dianziyan',
    data_root='/home/ubuntu22/PycharmProjects/PythonProject/mmdetection/Data/dianziyan/100K_dataset',
    save_format='npy'
)

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=30000,
    val_interval=6000,
)

val_cfg = dict(type='ValLoop')
# Use EfficientADTestLoop to automatically generate both df_test.csv and df_training.csv
test_cfg = dict(type='EfficientADTestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-4, weight_decay=1e-5),
    clip_grad=None,
)

param_scheduler = [
    dict(
        type='MultiStepLR',
        by_epoch=False,
        milestones=[66500],
        gamma=0.1,
    )
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10000,  # Save every 10000 iterations
        max_keep_ckpts=8,  # Keep last 8 checkpoints (iter_10000 to iter_70000)
        by_epoch=False,  # Save by iteration, not by epoch
        save_last=True,  # Save the last checkpoint
    ),
    logger=dict(type='LoggerHook', interval=100),
)

# Custom hook to save teacher, student, and autoencoder separately
# This matches the output format of the original EfficientAD project
# 这里可选
# custom_hooks = [
#     dict(
#         type='EfficientADSaveHook',
#         output_dir='./output',  # Base output directory
#         Data='mvtec_ad',  # Dataset name
#         subdataset='dianziyan',  # Sub-Data name
#         interval=1000,  # Save every 1000 iterations (as in original)
#     ),
# ]

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)
