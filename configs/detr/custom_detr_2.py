
_base_ = './detr_r50_8xb2-150e_coco.py'

# learning policy
max_epochs = 500
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=10)

# Match head output dims with custom Data classes
model = dict(
    bbox_head=dict(num_classes=8)
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=1000,  # 前1000个iteration预热
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=500,
        by_epoch=True,
        milestones=[50, 100, 150, 200, 250, 300, 350, 400, 450],
        # 每50个epoch调整一次
        gamma=0.9,  # 每次降低10%
    ),
]

# 简化训练pipeline，减少过强的数据增强
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        keep_ratio=True,
        scale=(800, 1333),  # 使用固定的合理尺寸
        type='Resize'
    ),
    dict(type='PackDetInputs'),
]

# 修改训练数据加载器
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=8,
    dataset=dict(
        ann_file='annotations/instances_train2017.json',
        backend_args=None,
        data_prefix=dict(img='train2017/'),
        data_root=(
            '/home/ubuntu22/PycharmProjects/PythonProject/mmdetection/'
            'Data/coco_dataset2_split/'
        ),
        filter_cfg=None,
        metainfo=dict(
            classes=(
                'global_zangwu',
                'global_maoxie',
                'cashang',
                'huashang',
                'yashang',
                'split_cashang',
                'split_huashang',
                'split_yashang',
            ),
            palette=[
                (220, 20, 60),
                (119, 11, 32),
                (0, 0, 142),
                (0, 0, 230),
                (106, 0, 228),
                (0, 60, 100),
                (0, 80, 100),
                (0, 0, 70),
            ]
        ),
        pipeline=train_pipeline,  # 使用简化的pipeline
        type='CocoDataset'
    ),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler')
)

# only keep latest 2 checkpoints
default_hooks = dict(checkpoint=dict(max_keep_ckpts=2))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
