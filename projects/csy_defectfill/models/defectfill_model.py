# Copyright (c) OpenMMLab. All rights reserved.
"""DefectFillDetector - MMDET-compatible wrapper for DefectFill."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample

# Core DefectFill (SD 2 inpainting + LoRA + Textual Inversion) lives next to
# this file at projects/csy_defectfill/models/defectfill_core.py. It used to
# live at `mmdet/DefectFill/model.py` and was injected via `sys.path.insert`;
# that hack is gone as of the 2026-07 refactor (see plan: graceful-sprouting-
# rocket). External SD weights / data still live under `DefectFill/{ck,DATA}/`
# and are addressed by cfg-level paths, not by Python imports.
from projects.csy_defectfill.models.defectfill_core import DefectFillCore


@MODELS.register_module()
class DefectFillDetector(BaseModule):
    """MMDET-compatible wrapper for DefectFillCore.

    Wraps the StableDiffusionInpaintPipeline with LoRA + Textual Inversion,
    exposing train_step/val_step/test_step that align with MMDET's runner.

    Args:
        lora_rank (int): LoRA rank for UNet and text encoder.
        lora_alpha (int): LoRA alpha scaling.
        placeholder_token (str): Learnable token for defect concept.
        pretrained_model_path (str): Path to pretrained SD inpainting model.
        lambda_defect (float): Weight for defect loss.
        lambda_obj (float): Weight for object integrity loss.
        lambda_attn (float): Weight for attention loss.
        alpha (float): Background weight for object loss.
        text_encoder_lr (float): Learning rate for text encoder LoRA.
        unet_lr (float): Learning rate for UNet LoRA.
        lr_warmup_steps (int): LR warmup steps.
        num_inference_steps (int): DDIM steps for inference.
        guidance_scale (float): CFG guidance scale.
        init_cfg (dict, optional): Initialization config.
    """

    def __init__(
        self,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        placeholder_token: str = '<defect>',
        pretrained_model_path: Optional[str] = None,
        lambda_defect: float = 1.0,
        lambda_obj: float = 0.2,
        lambda_attn: float = 0.05,
        alpha: float = 0.3,
        text_encoder_lr: float = 4e-5,
        unet_lr: float = 2e-4,
        lr_warmup_steps: int = 100,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = 42,
        init_cfg: Optional[Dict] = None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.placeholder_token = placeholder_token
        self.pretrained_model_path = pretrained_model_path
        self.lambda_defect = lambda_defect
        self.lambda_obj = lambda_obj
        self.lambda_attn = lambda_attn
        self.alpha = alpha
        self.text_encoder_lr = text_encoder_lr
        self.unet_lr = unet_lr
        self.lr_warmup_steps = lr_warmup_steps
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed

        # Lazy initialization - don't build pipeline in __init__
        self.model = None
        self.noise_scheduler = None
        self.pipeline = None
        self.text_encoder = None
        self.unet = None
        self._initialized = False

        # Dummy parameter to ensure model has trainable parameters
        # This prevents "optimizer got an empty parameter list" error
        # during optimizer construction, before lazy init builds the real model
        self._dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    def _build_pipeline(self):
        """Initialize the DefectFillCore (StableDiffusion + LoRA + Textual Inversion).

        Since the 2026-07 refactor, ``DefectFillCore`` is imported normally
        at the top of this file (sibling of this module). No ``sys.path``
        juggling is needed.
        """
        if DefectFillCore is None:
            raise RuntimeError(
                'DefectFillCore is not importable. Check that '
                'projects/csy_defectfill/models/defectfill_core.py exists.')

        self.model = DefectFillCore(
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            seed=self.seed,
            placeholder_token=self.placeholder_token,
            model_path=self.pretrained_model_path,
        )

        # Initialize noise scheduler (DDPM for training)
        hf_model_id = "sd2-community/stable-diffusion-2-inpainting"
        if self.pretrained_model_path and os.path.isdir(self.pretrained_model_path):
            from diffusers import DDPMScheduler
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                self.pretrained_model_path, subfolder="scheduler")
        else:
            # Fallback to HuggingFace with mirror
            from diffusers import DDPMScheduler
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                hf_model_id, subfolder="scheduler")

        # Set to train mode
        self.model.pipeline.unet.train()
        self.model.pipeline.text_encoder.train()

        # Store references for optimizer
        self.pipeline = self.model.pipeline
        self.text_encoder = self.model.pipeline.text_encoder
        self.unet = self.model.pipeline.unet

        # Remove dummy param now that real model is initialized
        if hasattr(self, '_dummy_param'):
            del self._dummy_param

        # Store base LRs for warmup
        self._base_lrs = [self.text_encoder_lr, self.unet_lr]
        self._initialized = True

    def load_state_dict(self, state_dict, strict=True, logger=None):
        """Trigger lazy pipeline init BEFORE loading checkpoint.

        MMEngine's ``Runner.resume_or_load_checkpoint`` runs ``load_state_dict``
        on the module BEFORE the first ``train_step``/``test_step``. Because
        ``DefectFillDetector`` lazy-builds ``DefectFillCore`` (SD base + LoRA
        + LPIPS + placeholder token) only inside those steps, the checkpoint
        keys would all be reported as "unexpected" and silently dropped.

        Override here: if any key in the incoming state_dict targets a layer
        that only exists after ``_build_pipeline()``, run the build first so
        those keys actually match parameters.
        """
        needs_build = (
            self.model is None
            and any(k.startswith(('model.pipeline.', 'model.lpips_model.',
                                  'model.placeholder_token'))
                    for k in state_dict.keys()))
        if needs_build:
            self._build_pipeline()
        return super().load_state_dict(state_dict, strict=strict)

    def train_step(self, data_batch: Dict, optim_wrapper) -> Dict[str, Tensor]:
        """MMDET train_step implementing dual-branch training.

        Args:
            data_batch (Dict): Contains:
                - img: defective images [B, 3, H, W] normalized to [-1, 1]
                - mask: defect masks [B, 1, H, W] in [0, 1]
                - background: I * (1-M) [B, 3, H, W]
                - adjusted_mask: for object loss [B, 1, H, W]
                - is_defect: bool tensor indicating if sample has GT mask
                - object_class: list of object class names

        Returns:
            Dict with 'loss' containing total loss
        """
        # Lazy initialization if model not built yet
        if self.model is None:
            self._build_pipeline()

        img = data_batch['img']
        mask = data_batch['mask']
        is_defect = data_batch['is_defect']

        # Ensure we only process defective samples
        # Convert lists to tensors if needed
        if isinstance(img, list):
            img = torch.stack(img)
        if isinstance(mask, list):
            mask = torch.stack(mask)
        if isinstance(is_defect, list):
            is_defect = torch.stack(is_defect)
        if isinstance(data_batch.get('background'), list):
            data_batch['background'] = torch.stack(data_batch['background'])
        if isinstance(data_batch.get('adjusted_mask'), list):
            data_batch['adjusted_mask'] = torch.stack(data_batch['adjusted_mask'])

        # Move tensors to the same device as the model
        device = next(self.model.pipeline.vae.parameters()).device
        img = img.to(device)
        mask = mask.to(device)
        is_defect = is_defect.to(device)
        data_batch['background'] = data_batch['background'].to(device)
        data_batch['adjusted_mask'] = data_batch['adjusted_mask'].to(device)

        # Ensure float32 and normalize if needed (input should be [-1, 1])
        if img.dtype == torch.uint8:
            img = img.float() / 127.5 - 1.0  # normalize to [-1, 1]
        if mask.dtype != torch.float32:
            mask = mask.float()
        if data_batch['background'].dtype == torch.uint8:
            data_batch['background'] = data_batch['background'].float() / 127.5 - 1.0
        if data_batch['adjusted_mask'].dtype != torch.float32:
            data_batch['adjusted_mask'] = data_batch['adjusted_mask'].float()

        defect_indices = torch.nonzero(is_defect).squeeze(-1).long()
        if defect_indices.numel() == 0:
            return {'loss': torch.tensor(0.0, device=img.device)}

        # Extract defect samples
        defect_images = img[defect_indices]
        defect_masks = mask[defect_indices]
        defect_backgrounds = data_batch['background'][defect_indices]
        adjusted_masks = data_batch['adjusted_mask'][defect_indices]
        # Handle object_class which may be a list of strings
        obj_class_list = data_batch['object_class']
        defect_idx_list = defect_indices.tolist() if torch.is_tensor(defect_indices) else list(defect_indices)
        if isinstance(obj_class_list, (list, tuple)):
            object_classes = [obj_class_list[i] for i in defect_idx_list]
        else:
            object_classes = [obj_class_list] * len(defect_indices)

        # ========== PHASE 1: Defect Branch ==========
        defect_prompts = [
            f"A photo of {self.model.placeholder_token}"
            for _ in range(len(defect_indices))
        ]
        text_embeddings = self.model.get_text_embeddings(
            defect_prompts, enable_grad=True)

        # Encode to latent space (VAE frozen)
        with torch.no_grad():
            latents = self.model.pipeline.vae.encode(defect_images).latent_dist.sample()
            latents = latents * self.model.pipeline.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device
        )
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Build masked_image_latents: b = E(I * (1-M))
        with torch.no_grad():
            masked_images = defect_images * (1 - defect_masks)
            masked_image_latents = self.model.pipeline.vae.encode(
                masked_images).latent_dist.sample()
            masked_image_latents = (
                masked_image_latents * self.model.pipeline.vae.config.scaling_factor
            )

        mask_latents = F.interpolate(
            defect_masks, size=(latents.shape[2], latents.shape[3]))

        # Forward pass (9-channel input)
        self.model.attention_maps = {}
        outputs = self.model(
            noisy_latents=noisy_latents,
            masked_image_latents=masked_image_latents,
            mask_latents=mask_latents,
            timesteps=timesteps,
            encoder_hidden_states=text_embeddings
        )

        noise_pred = outputs["noise_pred"]
        defect_loss = self.model.compute_defect_loss(noise_pred, noise, mask_latents)
        attention_loss = outputs.get(
            "attention_loss", torch.tensor(0.0, device=latents.device))

        # ========== PHASE 2: Object Branch ==========
        # Generate random masks for structural context learning
        batch_size, _, h, w = defect_images.shape
        random_masks = torch.zeros(
            batch_size, 1, h, w, device=defect_images.device)

        for i in range(batch_size):
            for _ in range(30):  # 30 random boxes
                min_size = int(min(h, w) * 0.03)
                max_size = int(min(h, w) * 0.25)
                rect_h = torch.randint(
                    min_size, max(1, max_size), (1,)).item()
                rect_w = torch.randint(
                    min_size, max(1, max_size), (1,)).item()
                y = torch.randint(0, max(1, h - rect_h), (1,)).item()
                x = torch.randint(0, max(1, w - rect_w), (1,)).item()
                random_masks[i, 0, y:y+rect_h, x:x+rect_w] = 1.0

        obj_prompts = [
            f"A {obj_class} with {self.model.placeholder_token}"
            for obj_class in object_classes
        ]
        obj_text_embeddings = self.model.get_text_embeddings(
            obj_prompts, enable_grad=True)

        obj_noise = torch.randn_like(latents)
        obj_timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device
        )
        obj_noisy_latents = self.noise_scheduler.add_noise(
            latents, obj_noise, obj_timesteps)

        with torch.no_grad():
            random_masked_images = defect_images * (1 - random_masks)
            random_masked_image_latents = self.model.pipeline.vae.encode(
                random_masked_images).latent_dist.sample()
            random_masked_image_latents = (
                random_masked_image_latents
                * self.model.pipeline.vae.config.scaling_factor
            )

        random_mask_latents = F.interpolate(
            random_masks, size=(latents.shape[2], latents.shape[3]))

        obj_outputs = self.model(
            noisy_latents=obj_noisy_latents,
            masked_image_latents=random_masked_image_latents,
            mask_latents=random_mask_latents,
            timesteps=obj_timesteps,
            encoder_hidden_states=obj_text_embeddings
        )

        object_loss = self.model.compute_object_loss(
            obj_outputs["noise_pred"], obj_noise, random_mask_latents,
            alpha=self.alpha)

        # ========== Total Loss ==========
        total_loss = (
            self.lambda_defect * defect_loss
            + self.lambda_obj * object_loss
            + self.lambda_attn * attention_loss
        )

        # NaN check
        if torch.isnan(total_loss):
            print(f"Warning: NaN loss detected in train_step")
            return {'loss': torch.tensor(0.0, device=img.device)}

        # Backward and optimize
        optim_wrapper.backward(total_loss)
        optim_wrapper.step()
        optim_wrapper.zero_grad()

        return {
            'loss': total_loss,
            'loss_defect': defect_loss.detach(),
            'loss_object': object_loss.detach(),
            'loss_attn': attention_loss.detach(),
        }

    def val_step(self, data_batch: Dict, **kwargs) -> List[DetDataSample]:
        """Validation step - returns anomaly scores."""
        return self.test_step(data_batch, **kwargs)

    def test_step(self, data_batch: Dict, **kwargs) -> List[DetDataSample]:
        """Generate inpainted images and compute anomaly scores.

        Args:
            data_batch (Dict): Contains:
                - img: input images [B, 3, H, W]
                - mask: optional masks [B, 1, H, W]
                - is_defect: bool tensor indicating if sample is a defect
                - object_class: list of object class names
                - img_path: optional list of image paths

        Returns:
            List of DetDataSample with pred_images and anomaly_scores
        """
        # Lazy initialization if model not built yet
        if self.model is None:
            self._build_pipeline()

        img = data_batch['img']
        mask = data_batch.get('mask')

        # Convert lists to tensors if needed
        if isinstance(img, list):
            img = torch.stack(img)
        if isinstance(mask, list):
            mask = torch.stack(mask)

        # Handle object_class robustly (may be None or non-list)
        object_classes = data_batch.get('object_class', None)
        if object_classes is None:
            object_classes = ['unknown'] * img.shape[0]

        # Handle is_defect: use to set label for the metric
        is_defect = data_batch.get('is_defect', None)
        if is_defect is None:
            is_defect = torch.zeros(img.shape[0], dtype=torch.bool)
        elif isinstance(is_defect, list):
            is_defect = torch.stack(is_defect)
        elif not torch.is_tensor(is_defect):
            is_defect = torch.as_tensor(is_defect)

        # Optional image paths (for downstream metric/saving)
        img_paths = data_batch.get('img_path', None)
        if isinstance(img_paths, str):
            img_paths = [img_paths] * img.shape[0]

        # ---- BUG FIX: move all tensors to the model's device ----
        # Default BaseModule.data_preprocessor is nn.Identity(), so the
        # data_batch coming from the val/test loop is on CPU. Without this
        # move, the CPU tensor would be fed into the GPU SD pipeline and
        # raise a device-mismatch RuntimeError.
        device = next(self.model.pipeline.vae.parameters()).device
        img = img.to(device)
        if mask is not None:
            mask = mask.to(device)
        if torch.is_tensor(is_defect):
            is_defect = is_defect.to(device)
        # --------------------------------------------------------

        # Ensure model is in eval mode
        self.model.pipeline.unet.eval()
        self.model.pipeline.text_encoder.eval()

        results = []
        for i in range(img.shape[0]):
            single_img = img[i:i+1]
            single_mask = mask[i:i+1] if mask is not None else None
            sample_is_defect = bool(
                is_defect[i].item()) if torch.is_tensor(is_defect) else False

            # If no mask provided, use random masks for inference
            if single_mask is None:
                h, w = single_img.shape[2], single_img.shape[3]
                single_mask = torch.zeros(
                    1, 1, h, w, device=single_img.device)
                for _ in range(8):  # 8 random masks for inference
                    min_size = int(min(h, w) * 0.03)
                    max_size = int(min(h, w) * 0.25)
                    rect_h = torch.randint(
                        min_size, max(1, max_size), (1,)).item()
                    rect_w = torch.randint(
                        min_size, max(1, max_size), (1,)).item()
                    y = torch.randint(0, max(1, h - rect_h), (1,)).item()
                    x = torch.randint(0, max(1, w - rect_w), (1,)).item()
                    single_mask[0, 0, y:y+rect_h, x:x+rect_w] = 1.0

            # Generate inpainted image
            obj_cls = (object_classes[i]
                       if isinstance(object_classes, (list, tuple))
                       else object_classes)
            prompt = f"A {obj_cls} with {self.model.placeholder_token}"
            generated = self.model.generate(
                image=single_img,
                mask=single_mask,
                prompt=prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
            )

            # Compute LPIPS-based anomaly score
            lpips_score = self._compute_lpips_score(single_img, generated, single_mask)

            # Build DetDataSample (move large tensors to CPU to save GPU mem)
            data_sample = DetDataSample()
            data_sample.set_field(generated.detach().cpu(), 'pred_images')
            data_sample.set_field(lpips_score.detach().cpu(), 'anomaly_score')
            data_sample.set_field(obj_cls, 'object_class')
            # Pass label through to the metric for proper AUC computation
            data_sample.set_field(int(sample_is_defect), 'label')
            if img_paths is not None:
                data_sample.set_field(
                    img_paths[i] if isinstance(img_paths, list) else img_paths,
                    'img_path')

            results.append(data_sample)

        return results

    def _compute_lpips_score(
        self,
        original: Tensor,
        generated: Tensor,
        mask: Tensor
    ) -> Tensor:
        """Compute LPIPS-based anomaly score between original and generated.

        Convention: higher score = more anomalous.

        DefectFill is trained so the inpaint prompt contains the learnable
        ``<defect>`` placeholder token, which teaches the model to
        reconstruct defect regions faithfully. Therefore at inference:

        * on defective samples with the real defect mask, the model
          successfully reconstructs the defect → raw LPIPS(recon, orig)
          in the masked region is LOW,
        * on good samples with a random mask, the model is asked to fill
          a normal region while still being told the prompt contains a
          ``<defect>`` → the reconstruction diverges more from the
          original → raw LPIPS is HIGHER.

        So raw masked-LPIPS is anti-correlated with "is this a defect".
        We negate it so the returned score behaves like a standard anomaly
        score (higher for defect / lower for good).
        """
        # Normalize to [-1, 1] if needed
        if original.min() >= 0:
            original = 2 * original - 1
        if generated.min() >= 0:
            generated = 2 * generated - 1

        # Resize mask to match image dimensions
        mask_resized = F.interpolate(
            mask, size=original.shape[2:], mode='nearest')

        # Use LPIPS model from DefectFillModel
        lpips_loss = self.model.lpips_model(
            generated.float(), original.float())

        # Weight by mask - higher score where mask is 1
        weighted_score = (lpips_loss * mask_resized).mean()

        # Flip polarity: trained-to-reconstruct-defect region is "anomalous"
        # in the operational sense, so negate the faithfulness score.
        return -weighted_score

    # ===================================================================
    # Synthetic-defect generation (mirrors ``fixed_inference_batch`` in
    # ``DefectFill2/inference.py``). This is the "make-fake-defect" path
    # that complements ``test_step``'s anomaly-detection path.
    # ===================================================================
    @torch.no_grad()
    def generate_defect(
        self,
        image: Tensor,
        mask: Tensor,
        object_class: Optional[str] = None,
        num_samples: int = 8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        batch_size: int = 4,
        seed: int = 0,
    ) -> Tuple[Tensor, int, Tensor]:
        """Generate N candidates and return the LPIPS-best one.

        Implements the same algorithm as
        ``DefectFill2/inference.py::fixed_inference_batch``:

        * split ``num_samples`` candidates into ``batch_size`` chunks,
          run ``model.generate`` for each chunk (deterministic seed per chunk),
        * stack all candidates and compute spatial LPIPS in the mask region
          in a single batched forward pass,
        * ``argmax`` LPIPS to pick the most "defect-like" candidate (largest
          perceptual distance in the masked region ⇒ model agrees least with
          the original there ⇒ most defect-looking).

        Args:
            image: ``[1, 3, H, W]`` float in [-1, 1], already cropped to 512.
            mask:  ``[1, 1, H, W]`` float in [0, 1].
            object_class: object class name to embed in the prompt. Defaults
                to ``'unknown'`` so behaviour is unchanged from DefectFill2's
                prompt ``A {obj_cls} with <defect>``.
            num_samples: total candidates to draw per input image.
            num_inference_steps: DDIM steps for each ``model.generate``.
            guidance_scale: CFG value.
            batch_size: number of candidates generated per batched forward.
            seed: deterministic starting seed for the candidate generation.

        Returns:
            (best_image, best_idx, all_lpips_scores)
                best_image    ``[1, 3, H, W]`` in [-1, 1]
                best_idx      index of selected candidate
                all_lpips_scores ``[num_samples]`` raw spatial LPIPS (higher
                                  ⇒ more defect-like), un-negated so the same
                                  signal as DefectFill2.
        """
        if self.model is None:
            self._build_pipeline()
        obj_cls = object_class if object_class is not None else 'unknown'
        prompt = f'A {obj_cls} with {self.model.placeholder_token}'

        h, w = image.shape[-2:]
        device = image.device
        dtype = image.dtype

        all_samples: List[Tensor] = []
        num_batches = (num_samples + batch_size - 1) // batch_size
        for b in range(num_batches):
            start = b * batch_size
            cur_bs = min(batch_size, num_samples - start)
            # ``model.generate`` ends with ``torch.randn(..., device=image.device)``,
            # so the generator must live on the same device as ``image`` —
            # otherwise ``Expected a 'cuda' device type for generator but found
            # 'cpu'``. Fall back to CPU when the input is on CPU.
            gen_device = image.device if image.device.type == 'cuda' else 'cpu'
            gen = torch.Generator(device=gen_device).manual_seed(seed + start)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + start)
                samples = self.model.generate(
                    image=image.repeat(cur_bs, 1, 1, 1),
                    mask=mask.repeat(cur_bs, 1, 1, 1),
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=gen,
                )                                                          # [cur_bs,3,H,W] in [0,1]
            all_samples.append(samples)

        candidates = torch.cat(all_samples, dim=0)                            # [N,3,H,W] in [0,1]

        # ``model.generate`` returns [0, 1]; LPIPS lives in [-1, 1]
        candidates_for_lpips = (candidates * 2.0) - 1.0
        ref_for_lpips = image.to(device=device, dtype=torch.float32)

        # Use the same spatial LPIPS helper as DefectFill2/utils.py.
        # Border-smoothing is enabled to avoid hard mask edges dominating
        # the score.
        lpips_scores = self._spatial_lpips_batch(
            self.model.lpips_model,
            ref_for_lpips, candidates_for_lpips, mask.to(torch.float32),
            smooth_boundary=True)
        best_idx = int(lpips_scores.argmax().item())
        best_image = (candidates[best_idx:best_idx + 1] * 2.0) - 1.0
        return best_image, best_idx, lpips_scores

    @staticmethod
    def _spatial_lpips_batch(
        lpips_model,
        reference: Tensor,
        samples: Tensor,
        mask: Tensor,
        smooth_boundary: bool = True,
    ) -> Tensor:
        """Vectorized spatial LPIPS — port of ``compute_spatial_lpips_batch``.

        Args:
            lpips_model: the LPIPS-VGG instance (already built).
            reference: ``[1, 3, H, W]`` float in [-1, 1].
            samples:   ``[N, 3, H, W]`` float in [-1, 1].
            mask:      ``[1, 1, H, W]`` float in [0, 1].

        Returns:
            ``[N]`` LPIPS score per sample, masked-region weighted.
        """
        n = samples.shape[0]
        ref_exp = reference.repeat(n, 1, 1, 1).to(dtype=next(lpips_model.parameters()).dtype)
        samp = samples.to(dtype=next(lpips_model.parameters()).dtype)
        mask_exp = mask.repeat(n, 1, 1, 1)

        lpips_maps = lpips_model(ref_exp, samp)                              # [N,1,H',W']
        mask_resized = F.interpolate(
            mask_exp.to(dtype=lpips_maps.dtype),
            size=lpips_maps.shape[-2:], mode='bilinear', align_corners=False)
        if smooth_boundary:
            mask_resized = F.avg_pool2d(
                F.pad(mask_resized, (2, 2, 2, 2), mode='replicate'),
                kernel_size=5, stride=1)
        weighted_sum = (lpips_maps * mask_resized).sum(dim=(2, 3))           # [N,1]
        mask_sum = mask_resized.sum(dim=(2, 3)) + 1e-8                       # [N,1]
        return (weighted_sum / mask_sum).squeeze(1)                          # [N]

    def forward(self, *args, **kwargs):
        """Not implemented - use train_step/val_step/test_step."""
        raise NotImplementedError(
            "DefectFillDetector does not support direct forwarding; "
            "use train_step/val_step/test_step instead."
        )