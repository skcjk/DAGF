import math
import random
from collections import defaultdict

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from extractor import normalize_image


def _as_train_sizes(train_sizes):
    if isinstance(train_sizes, str):
        train_sizes = [int(x.strip()) for x in train_sizes.split(",") if x.strip()]
    sizes = sorted({int(size) for size in train_sizes})
    if not sizes:
        raise ValueError("train_sizes must contain at least one size")
    for size in sizes:
        if size <= 0:
            raise ValueError("train_sizes must be positive")
        if size % 14 != 0:
            raise ValueError(f"train size must be divisible by 14, got {size}")
    return tuple(sizes)


def _floor_to_multiple(value, multiple=14):
    return max(multiple, (int(value) // multiple) * multiple)


def _nearest_train_size(value, train_sizes):
    return min(train_sizes, key=lambda size: (abs(size - value), -size))


def _resize_interpolation(src_h, src_w, dst_h, dst_w):
    if dst_h > src_h or dst_w > src_w:
        return cv2.INTER_LANCZOS4
    return cv2.INTER_AREA


def _resize_image(img, target_h, target_w):
    src_h, src_w = img.shape[:2]
    interp = _resize_interpolation(src_h, src_w, target_h, target_w)
    return cv2.resize(img, (target_w, target_h), interpolation=interp)


def _lanczos4_axis(tensor, output_size, axis):
    input_size = tensor.shape[axis]
    if input_size == output_size:
        return tensor
    if input_size <= 0 or output_size <= 0:
        raise ValueError("Lanczos resize dimensions must be positive")

    if axis == -2:
        return _lanczos4_axis(tensor.transpose(-2, -1), output_size, -1).transpose(-2, -1)
    if axis != -1:
        raise ValueError("Lanczos resize only supports the last two tensor dimensions")

    work = tensor.float() if tensor.dtype in (torch.float16, torch.bfloat16) else tensor
    device = work.device
    dtype = work.dtype
    scale = input_size / float(output_size)
    positions = (torch.arange(output_size, device=device, dtype=dtype) + 0.5) * scale - 0.5
    left = torch.floor(positions).to(torch.long) - 3
    offsets = torch.arange(8, device=device, dtype=torch.long)
    indices = left[:, None] + offsets[None, :]

    distances = positions[:, None] - indices.to(dtype)
    weights = torch.sinc(distances) * torch.sinc(distances / 4.0)
    weights = torch.where(distances.abs() < 4.0, weights, torch.zeros_like(weights))
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(dtype).eps)

    if input_size == 1:
        indices = torch.zeros_like(indices)
    else:
        period = 2 * (input_size - 1)
        indices = torch.remainder(indices, period)
        indices = torch.where(indices >= input_size, period - indices, indices)

    samples = work.index_select(-1, indices.reshape(-1))
    samples = samples.reshape(*work.shape[:-1], output_size, 8)
    weight_shape = [1] * (samples.ndim - 2) + [output_size, 8]
    return (samples * weights.reshape(weight_shape)).sum(dim=-1)


def resize_tensor_lanczos4(tensor, size):

    if tensor.ndim != 4:
        raise ValueError(f"Expected NCHW tensor, got shape {tuple(tensor.shape)}")
    if isinstance(size, int):
        target_h = target_w = size
    else:
        target_h, target_w = (int(size[0]), int(size[1]))

    resized = _lanczos4_axis(tensor, target_w, -1)
    return _lanczos4_axis(resized, target_h, -2)


def _ensure_rgb(img):
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    if img.shape[2] > 3:
        return img[:, :, :3]
    return img


def _read_hw(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return img.shape[:2]


class UnderwaterDataset(Dataset):


    def __init__(
        self,
        file_pairs,
        is_train=False,
        split_prob=0.2,
        train_res_mode="fixed",
        train_sizes=(252,),
        native_max_side=1008,
    ):
        if train_res_mode not in {"fixed", "multiscale", "native"}:
            raise ValueError(f"Unsupported train_res_mode: {train_res_mode}")

        self.file_pairs = list(file_pairs)
        self.is_train = is_train
        self.train_res_mode = train_res_mode
        self.train_sizes = _as_train_sizes(train_sizes)
        self.native_max_side = int(native_max_side)
        self.samples = []
        self.bucket_to_indices = defaultdict(list)

        self.split_prob = 0.2 if split_prob is None else float(split_prob)
        if self.split_prob < 0.0 or self.split_prob > 1.0:
            raise ValueError(f"split_prob must be in [0, 1], got {self.split_prob}")

        for raw_path, gt_path in self.file_pairs:
            raw_hw = _read_hw(raw_path)
            if raw_hw is None:
                continue
            target_hw = self._select_target_hw(raw_hw)
            index = len(self.samples)
            self.samples.append(
                {
                    "raw_path": raw_path,
                    "gt_path": gt_path,
                    "raw_hw": raw_hw,
                    "target_hw": target_hw,
                }
            )
            self.bucket_to_indices[target_hw].append(index)

    def __len__(self):
        return len(self.samples)

    def _select_target_hw(self, raw_hw):
        h, w = raw_hw
        if self.train_res_mode == "fixed":
            size = self.train_sizes[0]
            return size, size

        if self.train_res_mode == "native":
            return self._native_target_hw(h, w)

        min_side = min(h, w)
        size = _nearest_train_size(min_side, self.train_sizes)
        return size, size

    def _native_target_hw(self, h, w):
        if self.native_max_side > 0 and max(h, w) > self.native_max_side:
            scale = self.native_max_side / float(max(h, w))
            h = max(14, int(math.floor(h * scale)))
            w = max(14, int(math.floor(w * scale)))
            return _floor_to_multiple(h), _floor_to_multiple(w)
        return _floor_to_multiple(h), _floor_to_multiple(w)

    def bucket_counts(self):
        return {bucket: len(indices) for bucket, indices in sorted(self.bucket_to_indices.items())}

    def _load_pair(self, sample):
        raw_img = _ensure_rgb(imageio.imread(sample["raw_path"]))
        gt_img = _ensure_rgb(imageio.imread(sample["gt_path"]))

        raw_h, raw_w = raw_img.shape[:2]
        gt_h, gt_w = gt_img.shape[:2]
        if (gt_h, gt_w) != (raw_h, raw_w):
            gt_img = _resize_image(gt_img, raw_h, raw_w)
        return raw_img, gt_img

    def _crop_pair(self, raw_img, gt_img, target_h, target_w):
        h, w = raw_img.shape[:2]
        if h < target_h or w < target_w:
            raw_img = _resize_image(raw_img, max(h, target_h), max(w, target_w))
            gt_img = _resize_image(gt_img, raw_img.shape[0], raw_img.shape[1])
            h, w = raw_img.shape[:2]

        if self.is_train:
            y = random.randint(0, h - target_h) if h > target_h else 0
            x = random.randint(0, w - target_w) if w > target_w else 0
        else:
            y = max((h - target_h) // 2, 0)
            x = max((w - target_w) // 2, 0)

        return (
            raw_img[y:y + target_h, x:x + target_w],
            gt_img[y:y + target_h, x:x + target_w],
        )

    def _resize_or_crop_native_pair(self, raw_img, gt_img, target_h, target_w):
        h, w = raw_img.shape[:2]
        if (h, w) == (target_h, target_w):
            return raw_img, gt_img

        if target_h <= h and target_w <= w and (target_h > h * 0.9 and target_w > w * 0.9):
            y = max((h - target_h) // 2, 0)
            x = max((w - target_w) // 2, 0)
            return (
                raw_img[y:y + target_h, x:x + target_w],
                gt_img[y:y + target_h, x:x + target_w],
            )

        return _resize_image(raw_img, target_h, target_w), _resize_image(gt_img, target_h, target_w)

    def _fixed_pair(self, raw_img, gt_img, target_h, target_w):
        h, w = raw_img.shape[:2]
        is_split_applied = False

        if (
            self.is_train
            and h >= 320
            and w >= 320
            and h >= target_h
            and w >= target_w
            and random.random() < self.split_prob
        ):
            y = random.randint(0, h - target_h)
            x = random.randint(0, w - target_w)
            raw_img = raw_img[y:y + target_h, x:x + target_w]
            gt_img = gt_img[y:y + target_h, x:x + target_w]
            is_split_applied = True

        if not is_split_applied:
            raw_img = _resize_image(raw_img, target_h, target_w)
            gt_img = _resize_image(gt_img, target_h, target_w)

        return raw_img, gt_img

    def __getitem__(self, idx):
        sample = self.samples[idx]
        target_h, target_w = sample["target_hw"]
        raw_img, gt_img = self._load_pair(sample)

        if self.train_res_mode == "native":
            raw_img, gt_img = self._resize_or_crop_native_pair(raw_img, gt_img, target_h, target_w)
        elif self.train_res_mode == "fixed":
            raw_img, gt_img = self._fixed_pair(raw_img, gt_img, target_h, target_w)
        else:
            raw_img, gt_img = self._crop_pair(raw_img, gt_img, target_h, target_w)

        if self.is_train:
            if random.random() > 0.5:
                raw_img = cv2.flip(raw_img, 1)
                gt_img = cv2.flip(gt_img, 1)

            if random.random() > 0.5:
                raw_img = cv2.flip(raw_img, 0)
                gt_img = cv2.flip(gt_img, 0)

            if target_h == target_w:
                rot_k = random.randint(0, 3)
                if rot_k > 0:
                    raw_img = np.rot90(raw_img, rot_k).copy()
                    gt_img = np.rot90(gt_img, rot_k).copy()

        raw_img = np.ascontiguousarray(raw_img)
        gt_img = np.ascontiguousarray(gt_img)
        gt_tensor = normalize_image(torch.from_numpy(gt_img).permute(2, 0, 1).float())
        return raw_img, gt_tensor


class ResolutionBucketBatchSampler(Sampler):


    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def __iter__(self):
        batches = []
        buckets = list(self.dataset.bucket_to_indices.items())
        for _, indices in buckets:
            bucket_indices = list(indices)
            if self.shuffle:
                random.shuffle(bucket_indices)
            for start in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[start:start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        total = 0
        for indices in self.dataset.bucket_to_indices.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += math.ceil(len(indices) / self.batch_size)
        return total


def custom_collate_fn(batch):

    raw_imgs = [item[0] for item in batch]
    gt_tensors = torch.stack([item[1] for item in batch])
    return raw_imgs, gt_tensors


