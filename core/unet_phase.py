import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../")

from core.extractor import DepthAnythingFeature
from core.unet import UnderwaterUNet


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch), nn.GELU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class UNetPhaseNet(nn.Module):
    def __init__(self, process_res="auto", unet_variant='large', unet_dropout=0.3):
        super().__init__()
        self.extractor = DepthAnythingFeature(process_res)
        self.unet_branch = UnderwaterUNet(in_channels=4, variant=unet_variant, base_drop_rate=unet_dropout)
        self.unet_head = nn.Sequential(
            ResBlock(64, 96),
            ResBlock(96, 64),
            ResBlock(64, 32),
            nn.BatchNorm2d(32), nn.ELU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def _to_tensor(self, data, device):
        if not isinstance(data, torch.Tensor):
            return torch.tensor(data, device=device, dtype=torch.float32)
        return data.to(device=device, dtype=torch.float32)

    def _input_to_nchw(self, x, device):
        if isinstance(x, torch.Tensor):
            tensor = x.to(device=device, dtype=torch.float32)
        else:
            if isinstance(x, np.ndarray):
                array = x
            else:
                array = np.stack([np.asarray(img) for img in x], axis=0)
            tensor = torch.as_tensor(array, device=device, dtype=torch.float32)

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        if tensor.ndim != 4:
            raise ValueError(f"Expected input image batch with 4 dims, got {tuple(tensor.shape)}")

        if tensor.shape[-1] in (1, 3, 4):
            tensor = tensor[..., :3].permute(0, 3, 1, 2).contiguous()
        elif tensor.shape[1] in (1, 3, 4):
            tensor = tensor[:, :3, :, :].contiguous()
        else:
            raise ValueError(f"Expected RGB image batch, got {tuple(tensor.shape)}")

        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor

    def forward(self, x):
        with torch.no_grad():
            features = self.extractor(x)

        device = self.unet_branch.inc[0].weight.device
        img_nchw = self._input_to_nchw(x, device)
        depth_raw = self._to_tensor(features['depth'], device)

        depth_nchw = depth_raw.unsqueeze(1) if depth_raw.ndim == 3 else depth_raw
        depth_nchw = depth_nchw.contiguous()
        if depth_nchw.shape[-2:] != img_nchw.shape[-2:]:
            depth_nchw = F.interpolate(depth_nchw, size=img_nchw.shape[-2:], mode='bilinear', align_corners=True)

        device_type = 'cuda' if img_nchw.device.type == 'cuda' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=img_nchw.device.type == 'cuda'):
            unet_input = torch.cat([img_nchw, depth_nchw], dim=1)
            unet_feat = self.unet_branch(unet_input)
            residual = self.unet_head(unet_feat)
            residual = torch.tanh(residual) * 0.5

        out = torch.clamp(img_nchw + residual.float(), 0.0, 1.0)
        return out
