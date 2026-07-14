# yashang fine-tune preset — convenience config that wraps defectfill_phone.py
# with the yashang-specific recipe. Most settings (model hyper-params, data
# roots, dataloader structure) are inherited from the base cfg via `_base_`;
# only the subset filter + lower iter count + fine-tune-from-v2 init need
# to be different.
#
# ─── Usage ───────────────────────────────────────────────────────────────
# Direct invocation:
#   python tools/train.py projects/csy_defectfill/configs/defectfill_phone_yashang.py
#
# For flexibility (any subset, any iters), call the base cfg directly:
#   python tools/train.py projects/csy_defectfill/configs/defectfill_phone.py \
#       --cfg-options defect_types="[yashang]" \
#                    max_iters=4000 \
#                    load_from=./work_dirs/defectfill_phone_v2/iter_40000.pth \
#                    param_scheduler=[dict(by_epoch=False,gamma=0.1,milestones=[2500],type='MultiStepLR')]
# ──────────────────────────────────────────────────────────────────────────

_base_ = ['./defectfill_phone.py']

# Subset filter — only yashang. (Replaces the old `defect_type='yashang'`.)
defect_types = ['yashang']

# Fine-tune from v2 (already converged on the broader `<defect>` concept).
# resume=False so scheduler / optimizer / iter counter start fresh.
load_from = './work_dirs/defectfill_phone_v2/iter_40000.pth'
resume = False

# Smaller iter budget — 29 samples × batch 2 ≈ 15 iters/epoch.
# 4000 iters ≈ 267 epochs; LR drops 10× at iter 2500 to settle into yashang space.
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=4000,
    val_interval=1000,
)
param_scheduler = [
    dict(
        type='MultiStepLR',
        by_epoch=False,
        milestones=[2500],
        gamma=0.1,
    )
]

# Fresh work dir (don't clobber v2 ckpts); finer-grained checkpoints because
# the dataset is tiny and we want more snapshots along the trajectory.
work_dir = './work_dirs/defectfill_phone_yashang'
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=500,
        max_keep_ckpts=3,
        by_epoch=False,
        save_last=True,
    ),
    logger=dict(type='LoggerHook', interval=10),
)