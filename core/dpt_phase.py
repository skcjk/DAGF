import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dpt import DPTBranch
from extractor import DepthAnythingFeature
from unet import UnderwaterUNet


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ELU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.skip(x)


class DPTPhaseNet(nn.Module):
    def __init__(
        self,
        process_res="auto",
        dpt_embed_dim=None,
        unet_branch_path=None,
        unet_variant="large",
        dpt_dropout=0.3,
        auto_load_unet=True,
    ):
        super().__init__()
        self.extractor = DepthAnythingFeature(process_res)
        self.unet_branch = UnderwaterUNet(
            in_channels=4,
            variant=unet_variant,
            base_drop_rate=0.0,
        )
        self.dpt_branch = DPTBranch(
            in_channels=1024,
            embed_dim=dpt_embed_dim,
            out_channels=256,
            dpt_dropout=dpt_dropout,
        )

        self.dpt_head = nn.Sequential(
            ResBlock(256, 128),
            ResBlock(128, 64),
            ResBlock(64, 32),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
        )

        self.loaded_unet_branch_path = None
        if auto_load_unet:
            self.loaded_unet_branch_path = self.load_unet_branch(unet_branch_path)

        self.freeze_extractor()
        self.freeze_unet_branch()

    def _default_unet_paths(self):
        return [
            "checkpoints/unet_large/unet_branch_best.pth",
            "checkpoints/unet_large/unet_branch.pth",
        ]

    def load_unet_branch(self, checkpoint_path=None):
        candidates = []
        if checkpoint_path:
            candidates.append(checkpoint_path)
        candidates.extend(self._default_unet_paths())

        for path in candidates:
            if path and os.path.isfile(path):
                state_dict = torch.load(path, map_location="cpu")
                self.unet_branch.load_state_dict(state_dict, strict=True)
                return path
        return None

    def freeze_extractor(self):
        self.extractor.eval()
        for param in self.extractor.parameters():
            param.requires_grad = False
        return self

    def freeze_unet_branch(self):
        self.unet_branch.eval()
        for param in self.unet_branch.parameters():
            param.requires_grad = False
        return self

    def _to_tensor(self, data, device):
        if isinstance(data, torch.Tensor):
            return data.to(device=device, dtype=torch.float32)
        return torch.tensor(data, device=device, dtype=torch.float32)

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

    def forward_features(self, x):
        with torch.no_grad():
            features = self.extractor(x)

        device = next(self.dpt_branch.parameters()).device
        img_nchw = self._input_to_nchw(x, device)
        depth_raw = self._to_tensor(features["depth"], device)
        l1 = self._to_tensor(features["layer_1_features"], device)
        l5 = self._to_tensor(features["layer_5_features"], device)
        l11 = self._to_tensor(features["layer_11_features"], device)
        l16 = self._to_tensor(features["layer_16_features"], device)

        depth_nchw = depth_raw.unsqueeze(1) if depth_raw.ndim == 3 else depth_raw
        depth_nchw = depth_nchw.contiguous()
        if depth_nchw.shape[-2:] != img_nchw.shape[-2:]:
            depth_nchw = F.interpolate(depth_nchw, size=img_nchw.shape[-2:], mode="bilinear", align_corners=True)

        target_size = img_nchw.shape[-2:]
        device_type = "cuda" if device.type == "cuda" else "cpu"
        autocast_enabled = device.type == "cuda"
        with torch.autocast(device_type=device_type, enabled=autocast_enabled):
            unet_input = torch.cat([img_nchw, depth_nchw], dim=1)
            with torch.no_grad():
                _, unet_features = self.unet_branch(unet_input, return_features=True)
            dpt_feat = self.dpt_branch(
                l1,
                l5,
                l11,
                l16,
                target_size=target_size,
                unet_features=unet_features,
            )

        return {
            "preprocessed_image": img_nchw,
            "depth": depth_nchw,
            "layer_1_features": l1,
            "layer_5_features": l5,
            "layer_11_features": l11,
            "layer_16_features": l16,
            "unet_features": unet_features,
            "dpt_feat": dpt_feat,
        }


    def forward(self, x):
        feats = self.forward_features(x)
        device = feats["dpt_feat"].device
        device_type = "cuda" if device.type == "cuda" else "cpu"
        with torch.autocast(device_type=device_type, enabled=device.type == "cuda"):
            residual = self.dpt_head(feats["dpt_feat"])
            residual = torch.tanh(residual) * 0.5
        out = torch.clamp(feats["preprocessed_image"] + residual.float(), 0.0, 1.0)
        return out
