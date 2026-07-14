# Copyright (c) OpenMMLab. All rights reserved.
"""Loss functions for DefectFill."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS


@MODELS.register_module()
class DefectFillLoss(nn.Module):
    """Loss functions wrapping DefectFill's loss computations.

    This module provides the loss functions used in DefectFill training:
    - Defect Loss (L_def): MSE restricted to the defect mask region
    - Object Loss (L_obj): Weighted MSE for object integrity learning
    - Attention Loss (L_attn): L2 loss between attention maps and mask

    Args:
        lambda_defect (float): Weight for defect loss.
        lambda_obj (float): Weight for object integrity loss.
        lambda_attn (float): Weight for attention loss.
        alpha (float): Background weight for object loss (M' = M + alpha*(1-M)).
    """

    def __init__(
        self,
        lambda_defect: float = 1.0,
        lambda_obj: float = 0.2,
        lambda_attn: float = 0.05,
        alpha: float = 0.3,
    ):
        super().__init__()
        self.lambda_defect = lambda_defect
        self.lambda_obj = lambda_obj
        self.lambda_attn = lambda_attn
        self.alpha = alpha

    def compute_masked_mse(
        self,
        noise_pred: torch.Tensor,
        noise: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute MSE loss only within the masked area.

        Args:
            noise_pred: Predicted noise [B, C, H, W]
            noise: Target noise [B, C, H, W]
            mask: Binary mask [B, 1, H, W] in [0, 1]

        Returns:
            Scalar loss
        """
        weighted_loss = mask * ((noise_pred - noise) ** 2)
        return torch.sum(weighted_loss) / (torch.sum(mask) + 1e-8)

    def compute_defect_loss(
        self,
        noise_pred: torch.Tensor,
        noise: torch.Tensor,
        mask_latents: torch.Tensor
    ) -> torch.Tensor:
        """L_def loss: MSE restricted to the defect mask region.

        Args:
            noise_pred: Predicted noise
            noise: Target noise
            mask_latents: Defect mask in latent space

        Returns:
            Scalar defect loss
        """
        return self.compute_masked_mse(noise_pred, noise, mask_latents)

    def compute_object_loss(
        self,
        noise_pred: torch.Tensor,
        noise: torch.Tensor,
        mask_latents: torch.Tensor,
        alpha: Optional[float] = None
    ) -> torch.Tensor:
        """L_obj loss: Weighted mask for object context preservation.

        Uses weighted mask M' = M + alpha*(1-M) to compute MSE,
        which preserves object context while focusing on defect region.

        Args:
            noise_pred: Predicted noise
            noise: Target noise
            mask_latents: Mask in latent space
            alpha: Background weight (if None, uses self.alpha)

        Returns:
            Scalar object loss
        """
        if alpha is None:
            alpha = self.alpha
        weighted_mask = mask_latents + alpha * (1 - mask_latents)
        return self.compute_masked_mse(noise_pred, noise, weighted_mask)

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise: torch.Tensor,
        mask_latents: torch.Tensor,
        attention_loss: Optional[torch.Tensor] = None,
        is_defect: bool = True
    ) -> dict:
        """Compute total loss as weighted combination of loss terms.

        Args:
            noise_pred: Predicted noise
            noise: Target noise
            mask_latents: Mask in latent space
            attention_loss: Optional attention loss from attention maps
            is_defect: Whether this is a defect sample (if False, only object loss)

        Returns:
            Dict with 'loss' and individual loss components
        """
        if is_defect:
            defect_loss = self.compute_defect_loss(noise_pred, noise, mask_latents)
        else:
            defect_loss = torch.tensor(0.0, device=noise_pred.device)

        object_loss = self.compute_object_loss(noise_pred, noise, mask_latents)

        if attention_loss is None:
            attention_loss = torch.tensor(0.0, device=noise_pred.device)

        total_loss = (
            self.lambda_defect * defect_loss
            + self.lambda_obj * object_loss
            + self.lambda_attn * attention_loss
        )

        return {
            'loss': total_loss,
            'loss_defect': defect_loss.detach(),
            'loss_object': object_loss.detach(),
            'loss_attn': attention_loss.detach(),
        }