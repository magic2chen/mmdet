custom_imports = dict(
    imports=['projects.csy_segad'], allow_failed_imports=False)

default_scope = 'mmdet'

# Model configuration
model = dict(
    type='SegADModel',
    num_components=1,  # Bottle has 1 component (single product)
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

# Data paths - using prepared SegAD data
data_root = 'ck4efficientad/bottle'
segm_path = 'work_dirs/segad_bottle/segmentation_maps'
an_path = 'work_dirs/segad_bottle/anomaly_maps'
category = 'bottle'
num_components = 1

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
        csv_file='work_dirs/segad_bottle/anomaly_maps/df_training.csv',
        num_components=num_components,
    ),
)

# Evaluator configuration - SegAD uses custom training loop that handles evaluation internally
# So we don't use val_evaluator and test_evaluator
# val_evaluator = dict(
#     type='SegADMetric',
#     save_dir='work_dirs/segad_bottle/eval_results',
# )
# test_evaluator = dict(
#     type='SegADMetric',
#     save_dir='work_dirs/segad_bottle/eval_results',
# )

# Training configuration
train_cfg = dict(
    type='SegADTrainLoop',
    max_epochs=1,
    val_interval=1,
    category=category,
    models_list=['efficient_ad'],
    bad_parts=10,
    segm_path=segm_path,
    an_path=an_path,
)

# Validation and test are handled internally by SegADTrainLoop
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
