from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmengine.model import BaseModule
from mmdet.registry import MODELS


def _build_autoencoder(out_channels: int) -> nn.Sequential:
    """Construct the lightweight autoencoder used by EfficientAD."""
    layers: List[nn.Module] = [
        nn.Conv2d(3, 32, 4, 2, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, 4, 2, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, 4, 2, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 4, 2, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 4, 2, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 8),
    ]

    def _upsample(size: int) -> nn.Module:
        return nn.Upsample(size=size, mode='bilinear', align_corners=False)

    decoder: List[nn.Module] = [
        _upsample(3),
        nn.Conv2d(64, 64, 4, 1, 2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        _upsample(8),
        nn.Conv2d(64, 64, 4, 1, 2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        _upsample(15),
        nn.Conv2d(64, 64, 4, 1, 2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        _upsample(32),
        nn.Conv2d(64, 64, 4, 1, 2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        _upsample(63),
        nn.Conv2d(64, 64, 4, 1, 2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        _upsample(127),
        nn.Conv2d(64, 64, 4, 1, 2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        _upsample(56),
        nn.Conv2d(64, 64, 3, 1, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, out_channels, 3, 1, 1),
    ]
    layers.extend(decoder)
    return nn.Sequential(*layers)


def _build_pdn_small(
        out_channels: int,
        padding: bool = False) -> nn.Sequential:
    pad_mult = 1 if padding else 0
    layers: List[nn.Module] = [
        nn.Conv2d(3, 128, 4, padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(2, 2, padding=1 * pad_mult),
        nn.Conv2d(128, 256, 4, padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(2, 2, padding=1 * pad_mult),
        nn.Conv2d(256, 256, 3, padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, out_channels, 4),
    ]
    return nn.Sequential(*layers)


def _build_pdn_medium(
        out_channels: int,
        padding: bool = False) -> nn.Sequential:
    pad_mult = 1 if padding else 0
    layers: List[nn.Module] = [
        nn.Conv2d(3, 256, 4, padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(2, 2, padding=1 * pad_mult),
        nn.Conv2d(256, 512, 4, padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(2, 2, padding=1 * pad_mult),
        nn.Conv2d(512, 512, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, 3, padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, out_channels, 4),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 1),
    ]
    return nn.Sequential(*layers)


def build_pdn(model_size: str, out_channels: int,
              padding: bool = False) -> nn.Sequential:
    if model_size == 'small':
        return _build_pdn_small(out_channels, padding)
    if model_size == 'medium':
        return _build_pdn_medium(out_channels, padding)
    raise ValueError(f'Unsupported model_size: {model_size}')


@MODELS.register_module()
class EfficientADModel(BaseModule):
    """MMEngine compatible implementation of EfficientAD.

    This module wraps teacher, student and autoencoder networks and exposes
    ``train_step``/``val_step`` that align with MMEngine's runner.
    """

    def __init__(self,
                 model_size: str = 'small',
                 out_channels: int = 384,
                 teacher_checkpoint: Optional[str] = None,
                 teacher_stats_momentum: float = 0.01,
                 quantile: float = 0.999,
                 lambda_penalty: float = 1.0,
                 lambda_ae: float = 1.0,
                 lambda_stae: float = 1.0,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.model_size = model_size
        self.out_channels = out_channels
        self.teacher_stats_momentum = teacher_stats_momentum
        self.quantile = quantile
        self.lambda_penalty = lambda_penalty
        self.lambda_ae = lambda_ae
        self.lambda_stae = lambda_stae

        # teacher is frozen during training
        self.teacher = build_pdn(model_size, out_channels)
        if teacher_checkpoint is not None:
            state_dict = torch.load(teacher_checkpoint, map_location='cpu')
            if isinstance(state_dict, dict):
                self.teacher.load_state_dict(state_dict)
            else:
                self.teacher = state_dict
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.teacher.eval()

        # trainable student and autoencoder
        self.student = build_pdn(model_size, out_channels * 2)
        self.autoencoder = _build_autoencoder(out_channels)

        # buffers that store teacher statistics for feature normalisation
        self.register_buffer(
            'teacher_mean',
            torch.zeros(1, out_channels, 1, 1),
            persistent=True)
        self.register_buffer(
            'teacher_std',
            torch.ones(1, out_channels, 1, 1),
            persistent=True)
        self.stats_initialized = False

    @torch.no_grad()
    def _update_teacher_stats(self, teacher_output: Tensor) -> None:
        """Update running statistics of teacher features."""
        mean = teacher_output.mean(dim=[0, 2, 3], keepdim=True)
        var = ((teacher_output - mean) ** 2).mean(dim=[0, 2, 3], keepdim=True)
        std = torch.sqrt(var + 1e-12)

        if not self.stats_initialized:
            self.teacher_mean.copy_(mean)
            self.teacher_std.copy_(std)
            self.stats_initialized = True
        else:
            momentum = self.teacher_stats_momentum
            self.teacher_mean.lerp_(mean, 1 - momentum)
            self.teacher_std.lerp_(std, 1 - momentum)

    def _normalise_teacher(self, teacher_output: Tensor) -> Tensor:
        eps = 1e-6
        return (teacher_output - self.teacher_mean) / (self.teacher_std + eps)

    def _forward_maps(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        teacher_output = self.teacher(image)
        teacher_output = self._normalise_teacher(teacher_output)

        student_output = self.student(image)
        autoencoder_output = self.autoencoder(image)

        map_st = torch.mean(
            (teacher_output - student_output[:, :self.out_channels])**2,
            dim=1,
            keepdim=True)
        map_ae = torch.mean(
            (autoencoder_output - student_output[:, self.out_channels:])**2,
            dim=1,
            keepdim=True)
        return map_st, map_ae

    def train_step(self,
                   data_batch: Dict,
                   optim_wrapper) -> Dict[str, Tensor]:
        img_student: Tensor = data_batch['img_student']
        img_autoencoder: Tensor = data_batch['img_autoencoder']
        img_penalty: Optional[Tensor] = data_batch.get('img_penalty')
        use_penalty_val = data_batch.get('use_penalty', False)
        if isinstance(use_penalty_val, torch.Tensor):
            use_penalty = bool(use_penalty_val.item())
        else:
            use_penalty = bool(use_penalty_val)

        def _ensure_tensor(x: Tensor) -> Tensor:
            if isinstance(x, list):
                if isinstance(x[0], torch.Tensor):
                    x = torch.stack(x, dim=0)
                else:
                    x = torch.tensor(x)
            return x

        img_student = _ensure_tensor(img_student)
        img_autoencoder = _ensure_tensor(img_autoencoder)
        if img_penalty is not None:
            img_penalty = _ensure_tensor(img_penalty)

        if torch.cuda.is_available():
            img_student = img_student.cuda(non_blocking=True)
            img_autoencoder = img_autoencoder.cuda(non_blocking=True)
            if img_penalty is not None:
                img_penalty = img_penalty.cuda(non_blocking=True)

        with torch.no_grad():
            teacher_out_st = self.teacher(img_student)
            self._update_teacher_stats(teacher_out_st)
            teacher_out_st = self._normalise_teacher(teacher_out_st)

        student_out_st = self.student(img_student)[:, :self.out_channels]
        distance_st = (teacher_out_st - student_out_st)**2
        dist_flat = distance_st.flatten(1)
        q = torch.quantile(
            dist_flat,
            self.quantile,
            dim=1,
            keepdim=True,
            interpolation='nearest')
        mask = dist_flat >= q
        loss_hard = (dist_flat * mask).sum() / mask.sum().clamp(min=1)

        loss_penalty = torch.tensor(0., device=img_student.device)
        if use_penalty and img_penalty is not None:
            student_penalty = self.student(img_penalty)
            student_out_penalty = student_penalty[:, :self.out_channels]
            loss_penalty = torch.mean(student_out_penalty**2)

        with torch.no_grad():
            teacher_out_ae = self.teacher(img_autoencoder)
            teacher_out_ae = self._normalise_teacher(teacher_out_ae)

        student_out_full = self.student(img_autoencoder)
        ae_out = self.autoencoder(img_autoencoder)

        student_out_ae = student_out_full[:, self.out_channels:]

        distance_ae = (teacher_out_ae - ae_out)**2
        loss_ae = torch.mean(distance_ae)

        distance_stae = (ae_out - student_out_ae)**2
        loss_stae = torch.mean(distance_stae)

        total_loss = (
            loss_hard +
            self.lambda_penalty * loss_penalty +
            self.lambda_ae * loss_ae +
            self.lambda_stae * loss_stae)

        optim_wrapper.update_params(total_loss)

        return {
            'loss': total_loss,
            'loss_hard': loss_hard.detach(),
            'loss_penalty': loss_penalty.detach(),
            'loss_ae': loss_ae.detach(),
            'loss_stae': loss_stae.detach()
        }

    def val_step(self, data_batch: Dict,
                 **kwargs) -> List[Dict[str, Tensor]]:
        return self._infer_step(data_batch)

    def test_step(self, data_batch: Dict,
                  **kwargs) -> List[Dict[str, Tensor]]:
        return self._infer_step(data_batch)

    def _infer_step(self,
                    data_batch: Dict) -> List[Dict[str, Tensor]]:
        img: Tensor = data_batch['img']
        if isinstance(img, list):
            if isinstance(img[0], torch.Tensor):
                img = torch.stack(img, dim=0)
            else:
                img = torch.tensor(img)
        if torch.cuda.is_available():
            img = img.cuda(non_blocking=True)

        with torch.no_grad():
            map_st, map_ae = self._forward_maps(img)
        map_st = map_st.cpu()
        map_ae = map_ae.cpu()

        results: List[Dict[str, Tensor]] = []
        batch_size = img.size(0)
        paths: List[str] = data_batch['path']
        labels: Tensor = data_batch['label']
        orig_sizes: List[Tuple[int, int]] = data_batch['orig_size']

        for idx in range(batch_size):
            results.append(
                dict(
                    map_st=map_st[idx, 0],
                    map_ae=map_ae[idx, 0],
                    path=paths[idx],
                    label=int(labels[idx]),
                    orig_size=orig_sizes[idx],
                ))
        return results

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            'EfficientADModel does not support direct forwarding; '
            'use train_step/val_step/test_step instead.')
