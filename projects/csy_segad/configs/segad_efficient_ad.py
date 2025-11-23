custom_imports = dict(
    imports=['projects.csy_segad'], allow_failed_imports=False)

default_scope = 'mmdet'

# Model configuration
model = dict(
    type='SegADModel',
    num_components=2,  # Default, can be overridden per category
    models_list=['efficient_ad'],
    seed=333,
    scale_pos_weight=1.0,  # Will be computed based on data
    xgb_params=dict(
        n_estimators=10,
        max_depth=5,
        num_parallel_tree=200,
        learning_rate=0.3,
        objective='binary:logitraw',
        colsample_bynode=0.6,
        colsample_bytree=0.6,
        subsample=0.6,
        reg_alpha=1.0,
    ),
)

# Data paths
data_root = '/home/ubuntu22/PycharmProjects/PythonProject/mmdetection/Data/dianziyan/100K_dataset'
segm_path = '/Data/dianziyan/dianziyan_seg'
# an_path should be the root directory containing all models and categories
# CSV files are at: {an_path}/{model}/{category}/df_training.csv
an_path = '/home/ubuntu22/PycharmProjects/PythonProject/mmdetection/work_dirs/efficientad_small/output_anomaly_maps'
category = 'dianziyan'  # Can be changed for different categories
num_components = 2  # Category-specific, override in Data config

# Dataset configuration
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='SegADDataset',
        data_root=data_root,
        segm_path=segm_path,
        an_path=an_path,
        models_list=['efficient_ad'],
        category=category,
        split='train',
        csv_file='',  # Training loop loads CSV files directly, so this can be empty
        num_components=num_components,
    ),
)

# Test dataloader is not needed since SegAD handles data loading internally
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     Data=dict(
#         type='SegADDataset',
#         data_root=data_root,
#         segm_path=segm_path,
#         an_path=an_path,
#         models_list=['efficient_ad'],
#         category=category,
#         split='test',
#         csv_file=f'{an_path}/efficient_ad/{category}/df_test.csv',
#         num_components=num_components,
#     ),
# )

# Evaluator configuration
# Note: SegAD uses custom training loop that handles evaluation internally
# So we don't need val_evaluator and test_evaluator here
# val_evaluator = dict(
#     type='SegADMetric',
#     save_dir='',
# )

# test_evaluator = dict(
#     type='SegADMetric',
#     save_dir='',
# )

# Training configuration
train_cfg = dict(
    type='SegADTrainLoop',
    max_epochs=1,  # Not used for XGBoost
    val_interval=1,  # Not used for XGBoost
    category=category,
    models_list=['efficient_ad'],
    bad_parts=10,
    segm_path=segm_path,    an_path=an_path,
)

# Validation and test configs are not needed since SegAD handles evaluation internally
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')

# Optimizer (not used for XGBoost, but required by MMEngine)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-4),
)

# Hooks
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1),
    logger=dict(type='LoggerHook', interval=1),
)

log_processor = dict(type='LogProcessor', window_size=1, by_epoch=False)
