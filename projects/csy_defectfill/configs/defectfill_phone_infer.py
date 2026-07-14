# Config for DefectFill SYNTHETIC inference on phone (mirrors DefectFill2/inference.py).
# Loads a trained MMDET checkpoint, then runs ``generate_defect`` for each
# (image, mask) pair in ``--data-root`` and writes:
#
#   <output_dir>/<defect_type>/NNNN_generated.png
#   <output_dir>/<defect_type>/NNNN_original.png
#   <output_dir>/<defect_type>/NNNN_mask.png
#   <output_dir>/inference_log.json
#
# Usage:
#   python projects/csy_defectfill/tools/infer_generate.py \
#       projects/csy_defectfill/configs/defectfill_phone_infer.py \
#       work_dirs/defectfill_phone/iter_20000.pth \
#       --data-root DefectFill/DATA/my_infer/phone \
#       --object-class phone --defect-type yashang \
#       --output-dir work_dirs/defectfill_phone/infer_yashang \
#       --total-images 16 --num-samples 8 --steps 50

custom_imports = dict(
    imports=['projects.csy_defectfill'],
    allow_failed_imports=False)

default_scope = 'mmdet'

# Model: hyper-params MUST match defectfill_phone.py (the training cfg),
# otherwise load_state_dict raises "size mismatch for lora_A / lora_B" at
# every LoRA layer (text encoder + UNet).
model = dict(
    type='DefectFillDetector',
    lora_rank=32,                                # was 8; checkpoint trained at rank 32
    lora_alpha=64,                               # was 16; checkpoint trained at alpha 64
    placeholder_token='<defect>',
    pretrained_model_path='./DefectFill/ck',     # SD 2 inpainting base, same as training
    lambda_defect=1.0,
    lambda_obj=0.5,
    lambda_attn=0.5,
    alpha=0.3,
    text_encoder_lr=4e-5,
    unet_lr=2e-4,
    lr_warmup_steps=100,
    num_inference_steps=50,
    guidance_scale=2.0,                          # DefectFill2 default for inference
    seed=42,
)

# Sanity dataloader for the val split (used to enumerate image/mask pairs).
# We override ``object_class`` and ``root`` at the CLI when needed.
image_size = 512

# No training/eval loops — inference is driven by the standalone script.
train_cfg = None
val_cfg = None
test_cfg = None
train_dataloader = None
val_dataloader = None
test_dataloader = None
val_evaluator = None
test_evaluator = None
optim_wrapper = None
param_scheduler = None

env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(), dist_cfg=dict(backend='nccl'))
launcher = 'none'
log_level = 'INFO'
load_from = None                # CLI overrides
resume = False
work_dir = './work_dirs/defectfill_phone/infer_default'

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)

# Default inference hyper-params (CLI can override these too)
infer_cfg = dict(
    num_samples=8,
    num_inference_steps=50,
    guidance_scale=2.0,
    batch_size=4,
    seed=0,
    total_images=8,
    dilate_mask=False,
    mask_kernel_size=3,
)
