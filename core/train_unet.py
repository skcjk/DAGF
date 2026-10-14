import argparse
import datetime
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import (
    ResolutionBucketBatchSampler,
    UnderwaterDataset,
    custom_collate_fn,
    resize_tensor_lanczos4,
)
from Discriminator import Discriminator
from losses import UnderwaterEnhancementLoss
from unet_phase import UNetPhaseNet
from utils_logger import format_duration, get_gpu_memory, setup_logger


def _autocast_context(device):
    device_type = "cuda" if device == "cuda" else "cpu"
    return torch.autocast(device_type=device_type, enabled=device == "cuda")


def _parse_train_sizes(train_sizes):
    if isinstance(train_sizes, str):
        train_sizes = [int(x.strip()) for x in train_sizes.split(",") if x.strip()]
    sizes = tuple(sorted({int(size) for size in train_sizes}))
    if not sizes:
        raise ValueError("train_sizes must contain at least one value")
    return sizes


def _format_bucket_counts(dataset):
    counts = dataset.bucket_counts()
    if not counts:
        return "empty"
    return " ".join([f"{h}x{w}:{count}" for (h, w), count in counts.items()])


def _make_bucket_loader(dataset, batch_size, shuffle, num_workers=4):
    sampler = ResolutionBucketBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )


def _to_discriminator_input(tensor, size=252):
    if tensor.shape[-2:] == (size, size):
        return tensor
    if size > tensor.shape[-2] or size > tensor.shape[-1]:
        return resize_tensor_lanczos4(tensor, (size, size)).clamp(0.0, 1.0)
    return torch.nn.functional.interpolate(tensor, size=(size, size), mode="area")


def _freeze_batchnorm(module):
    count = 0
    for layer in module.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            layer.eval()
            count += 1
            for param in layer.parameters():
                param.requires_grad = False
    return count


def train(
    checkpoint_g=None,
    checkpoint_h=None,
    checkpoint_d=None,
    start_epoch=0,
    scratch=False,
    unet_variant="large",
    train_res_mode="multiscale",
    train_sizes=(252, 504, 756),
    batch_size=4,
    depth_process_res="auto",
    unet_dropout=0.3,
    native_max_side=1008,
    lambda_adv=0.0,
    freeze_bn=False,
    fixed_split_prob=0.2,
    train_discriminator=True,
    data_dir="data/underwater",
    epochs=100,
    lr=1e-4,
    num_workers=4,
    checkpoint_dir="checkpoints/unet_large",
):
    logger, log_path = setup_logger()
    logger.info(f"[Phase: DANet] Log file: {log_path}")

    train_raw = os.path.join(data_dir, "train", "raw")
    train_gt = os.path.join(data_dir, "train", "gt")
    test_raw = os.path.join(data_dir, "test", "raw")
    test_gt = os.path.join(data_dir, "test", "gt")

    for directory in (train_raw, train_gt, test_raw, test_gt):
        if not os.path.isdir(directory):
            raise FileNotFoundError(
                f"Dataset directory not found: {directory}. "
                "Prepare train/raw, train/gt, test/raw and test/gt under --data-dir."
            )
    if epochs <= start_epoch or lr <= 0 or num_workers < 0:
        raise ValueError("Require epochs > start_epoch, lr > 0 and num_workers >= 0")

    def _list_pairs(raw_dir, gt_dir):
        pairs = []
        for filename in os.listdir(raw_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                raw_path = os.path.join(raw_dir, filename)
                gt_path = os.path.join(gt_dir, filename)
                if os.path.isfile(gt_path):
                    pairs.append((raw_path, gt_path))
        return pairs

    train_pairs = _list_pairs(train_raw, train_gt)
    test_pairs = _list_pairs(test_raw, test_gt)
    if not train_pairs:
        raise ValueError(f"No matching training image pairs found under {data_dir}")
    logger.info(f"Dataset: {len(train_pairs)} train / {len(test_pairs)} test")

    train_sizes = _parse_train_sizes(train_sizes)
    lambda_adv = float(lambda_adv)
    freeze_bn = bool(freeze_bn or train_res_mode == "native")
    train_discriminator = bool(train_discriminator)
    if not train_discriminator:
        lambda_adv = 0.0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_dir = checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Batch Size: {batch_size} | Epochs: {epochs} | Fixed LR: {lr}")
    logger.info(
        f"Resolution mode: {train_res_mode} | train_sizes={train_sizes} | "
        f"depth_process_res={depth_process_res} | native_max_side={native_max_side}"
    )
    if train_res_mode == "fixed":
        logger.info(f"Fixed-mode random crop probability: {fixed_split_prob}")
    logger.info(f"DANet bottleneck dropout: {unet_dropout}")
    logger.info(f"Freeze generator BatchNorm: {freeze_bn}")
    logger.info(f"Train discriminator: {train_discriminator}")
    logger.info(f"Adversarial lambda: {lambda_adv}")
    if not train_discriminator:
        logger.info("Discriminator disabled: no D construction, checkpoint loading, forward, backward, or saving.")
    logger.info("Loss weights: L1=0.10 L2=0.1 Edge=0.20 Contrast=0.10 Dark=0.1 Bright=0.1 Lab=0.2 LCh=0.12 Percep=0.01 WB=0.1")
    logger.info(f"DANet variant: {unet_variant}")
    logger.info(f"Checkpoint directory: {ckpt_dir}/")

    train_dataset = UnderwaterDataset(
        train_pairs,
        is_train=True,
        train_res_mode=train_res_mode,
        train_sizes=train_sizes,
        native_max_side=native_max_side,
        split_prob=fixed_split_prob,
    )
    test_dataset = UnderwaterDataset(
        test_pairs,
        is_train=False,
        train_res_mode=train_res_mode,
        train_sizes=train_sizes,
        native_max_side=native_max_side,
        split_prob=0.0,
    )
    logger.info(f"Train buckets: {_format_bucket_counts(train_dataset)}")
    logger.info(f"Test buckets: {_format_bucket_counts(test_dataset)}")

    train_loader = _make_bucket_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = _make_bucket_loader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = UNetPhaseNet(
        process_res=depth_process_res,
        unet_variant=unet_variant,
        unet_dropout=unet_dropout,
    ).to(device)
    model.extractor.eval()
    for param in model.extractor.parameters():
        param.requires_grad = False

    discriminator = Discriminator(in_channels=3).to(device) if train_discriminator else None

    ckpt_g = os.path.join(ckpt_dir, "unet_branch_best.pth")
    ckpt_h = os.path.join(ckpt_dir, "unet_head_best.pth")
    ckpt_d = os.path.join(ckpt_dir, "discriminator_best.pth")

    if scratch:
        logger.info("--scratch enabled: ignoring all checkpoints and training from scratch.")
    elif checkpoint_g and os.path.isfile(checkpoint_g):
        model.unet_branch.load_state_dict(torch.load(checkpoint_g, map_location=device), strict=True)
        logger.info(f"Loaded DANet branch: {checkpoint_g}")
    elif os.path.isfile(ckpt_g):
        model.unet_branch.load_state_dict(torch.load(ckpt_g, map_location=device), strict=True)
        logger.info(f"Loaded DANet branch: {ckpt_g}")
    else:
        logger.info("DANet branch checkpoint not found; training from scratch.")

    if scratch:
        pass
    elif checkpoint_h and os.path.isfile(checkpoint_h):
        model.unet_head.load_state_dict(torch.load(checkpoint_h, map_location=device), strict=True)
        logger.info(f"Loaded DANet head: {checkpoint_h}")
    elif os.path.isfile(ckpt_h):
        model.unet_head.load_state_dict(torch.load(ckpt_h, map_location=device), strict=True)
        logger.info(f"Loaded DANet head: {ckpt_h}")
    else:
        logger.info("DANet head checkpoint not found; training from scratch.")

    if not train_discriminator:
        logger.info("Discriminator checkpoint loading skipped.")
    elif scratch:
        pass
    elif checkpoint_d and os.path.isfile(checkpoint_d):
        try:
            discriminator.load_state_dict(torch.load(checkpoint_d, map_location=device), strict=False)
            logger.info(f"Loaded discriminator: {checkpoint_d}")
        except Exception as exc:
            logger.warning(f"Failed to load discriminator checkpoint: {exc}; training D from scratch.")
    elif os.path.isfile(ckpt_d):
        try:
            discriminator.load_state_dict(torch.load(ckpt_d, map_location=device), strict=False)
            logger.info(f"Loaded discriminator: {ckpt_d}")
        except Exception as exc:
            logger.warning(f"Failed to load discriminator checkpoint: {exc}; training D from scratch.")
    else:
        logger.info("Discriminator checkpoint not found; training from scratch.")

    if freeze_bn:
        n_frozen_bn = _freeze_batchnorm(model)
        logger.info(f"Frozen generator BatchNorm layers: {n_frozen_bn}")

    trainable_params_g = [param for param in model.parameters() if param.requires_grad]
    optimizer_g = torch.optim.AdamW(trainable_params_g, lr=lr, weight_decay=1e-4)
    trainable_params_d = [param for param in discriminator.parameters() if param.requires_grad] if discriminator is not None else []
    optimizer_d = torch.optim.Adam(trainable_params_d, lr=lr, betas=(0.5, 0.999)) if train_discriminator and trainable_params_d else None

    criterion = UnderwaterEnhancementLoss().to(device)
    criterion_bce = nn.BCEWithLogitsLoss().to(device)
    scaler_g = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    scaler_d = torch.amp.GradScaler("cuda", enabled=device == "cuda") if optimizer_d is not None else None

    n_params_g = sum(param.numel() for param in trainable_params_g)
    n_params_d = sum(param.numel() for param in trainable_params_d)
    logger.info(f"Trainable parameters: G={n_params_g:,} D={n_params_d:,}")
    if lambda_adv == 0:
        logger.info("Generator adversarial loss disabled because lambda_adv=0.")
    if not train_discriminator:
        logger.info("Discriminator branch fully skipped.")
    logger.info("=" * 60)
    logger.info("Training started.")

    model.eval()
    if discriminator is not None:
        discriminator.eval()
    init_test_loss = 0.0
    init_sub = {key: 0.0 for key in [
        "L1", "L2", "Edge", "Contrast", "Dark", "Bright", "Lab", "LCh", "Percep", "WB",
        "DeadBlack", "DeadWhite", "PredMean", "PredStd", "PredMin", "PredMax",
    ]}
    n_test = len(test_loader)
    if n_test > 0:
        with torch.no_grad():
            for raw_imgs, gt_tensors in test_loader:
                gt_tensors = gt_tensors.to(device)
                preds = model(raw_imgs)
                loss, sub = criterion(preds, gt_tensors, return_detail=True)
                init_test_loss += loss.item()
                for key in init_sub:
                    init_sub[key] += sub.get(key, 0.0)
        init_test_loss /= n_test
        for key in init_sub:
            init_sub[key] /= n_test
        logger.info(f"Initial eval: Test Loss={init_test_loss:.4e} | PredMean={init_sub['PredMean']:.4e}")
        logger.info(
            f"  Sub-loss: L1={init_sub['L1']:.4e} L2={init_sub['L2']:.4e} "
            f"Edge={init_sub['Edge']:.4e} Contrast={init_sub['Contrast']:.4e}"
        )
        logger.info(
            f"            Dark={init_sub['Dark']:.4e} Bright={init_sub['Bright']:.4e} "
            f"Lab={init_sub['Lab']:.4e} LCh={init_sub['LCh']:.4e} "
            f"Percep={init_sub['Percep']:.4e} WB={init_sub['WB']:.4e}"
        )
    else:
        logger.info("Initial eval skipped: empty test loader.")


    model.train()
    if train_discriminator and discriminator is not None:
        discriminator.train()

    best_test_loss = float("inf")
    if n_test == 0:
        logger.warning("Best checkpoint selection disabled: test loader is empty.")
    train_start = time.time()

    if start_epoch > 0:
        logger.info(f"Resuming at epoch {start_epoch}; fixed LR remains {lr:.2e}.")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        model.train()
        if train_discriminator and discriminator is not None:
            discriminator.train()
        model.extractor.eval()
        if freeze_bn:
            _freeze_batchnorm(model)

        epoch_g_loss = 0.0
        epoch_g_content = 0.0
        epoch_d_loss = 0.0
        epoch_g_adv = 0.0
        epoch_d_real_out = 0.0
        epoch_d_fake_out = 0.0
        epoch_sub_losses = {key: 0.0 for key in [
            "L1", "L2", "Edge", "Contrast", "Dark", "Bright", "Lab", "LCh", "Percep", "WB",
            "DeadBlack", "DeadWhite", "PredMean", "PredStd", "PredMin", "PredMax",
        ]}

        for batch_idx, (raw_imgs, gt_tensors) in enumerate(train_loader):
            gt_tensors = gt_tensors.to(device)
            preds_clamped = model(raw_imgs)

            loss_d = torch.tensor(0.0, device=device)
            loss_g_adv = torch.tensor(0.0, device=device)
            d_real_out = torch.tensor(0.0, device=device)
            d_fake_out = torch.tensor(0.0, device=device)

            if train_discriminator and optimizer_d is not None and discriminator is not None:
                real = _to_discriminator_input(gt_tensors)
                fake = _to_discriminator_input(preds_clamped.detach())

                optimizer_d.zero_grad()
                with _autocast_context(device):
                    d_real_out = discriminator(real)
                    d_fake_out = discriminator(fake)
                    real_label = torch.full_like(d_real_out, 0.9)
                    fake_label = torch.full_like(d_fake_out, 0.1)
                    loss_d_real = criterion_bce(d_real_out, real_label)
                    loss_d_fake = criterion_bce(d_fake_out, fake_label)
                    loss_d = (loss_d_real + loss_d_fake) * 0.5
                scaler_d.scale(loss_d).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()

            optimizer_g.zero_grad()
            if lambda_adv > 0 and discriminator is not None:
                with _autocast_context(device):
                    g_fake_out = discriminator(_to_discriminator_input(preds_clamped))
                    g_label = torch.full_like(g_fake_out, 0.9)
                    loss_g_adv = criterion_bce(g_fake_out, g_label)

            loss_g_content, sub_detail = criterion(preds_clamped, gt_tensors, return_detail=True)
            loss_g = loss_g_content + lambda_adv * loss_g_adv

            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            epoch_g_loss += loss_g.item()
            epoch_g_content += loss_g_content.item()
            epoch_d_loss += loss_d.item()
            epoch_g_adv += loss_g_adv.item()
            epoch_d_real_out += d_real_out.mean().item()
            epoch_d_fake_out += d_fake_out.mean().item()
            for key in epoch_sub_losses:
                epoch_sub_losses[key] += sub_detail.get(key, 0.0)

            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - epoch_start
                steps_per_epoch = len(train_loader)
                eta_step = elapsed / (batch_idx + 1) * (steps_per_epoch - batch_idx - 1)
                logger.info(
                    f"E{epoch + 1:03d}/{epochs} "
                    f"B{batch_idx + 1:03d}/{steps_per_epoch} | "
                    f"G:{loss_g.item():.4e} (C:{loss_g_content.item():.4e} A:{loss_g_adv.item():.4e}) | "
                    f"D:{loss_d.item():.4e} (R:{d_real_out.mean().item():.4e} F:{d_fake_out.mean().item():.4e}) | "
                    f"ETA:{format_duration(eta_step)}"
                )

        n_steps = len(train_loader)
        if n_steps == 0:
            logger.warning("Empty train loader; stopping.")
            break

        avg_g_loss = epoch_g_loss / n_steps
        avg_g_content = epoch_g_content / n_steps
        avg_d_loss = epoch_d_loss / n_steps
        avg_g_adv = epoch_g_adv / n_steps
        avg_d_real = epoch_d_real_out / n_steps
        avg_d_fake = epoch_d_fake_out / n_steps
        avg_sub = {key: value / n_steps for key, value in epoch_sub_losses.items()}

        model.eval()
        if discriminator is not None:
            discriminator.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for raw_imgs, gt_tensors in test_loader:
                gt_tensors = gt_tensors.to(device)
                preds = model(raw_imgs)
                loss = criterion(preds, gt_tensors)
                epoch_test_loss += loss.item()
        avg_test_loss = epoch_test_loss / n_test if n_test > 0 else float("nan")

        epoch_duration = time.time() - epoch_start
        total_elapsed = time.time() - train_start
        eta_total = total_elapsed / (epoch + 1) * (epochs - epoch - 1)
        gpu_alloc, _ = get_gpu_memory()
        current_lr_g = optimizer_g.param_groups[0]["lr"]
        current_lr_d = optimizer_d.param_groups[0]["lr"] if optimizer_d is not None else 0.0

        logger.info("-" * 60)
        logger.info(
            f"[Epoch {epoch + 1:03d}/{epochs}] duration={format_duration(epoch_duration)} | "
            f"elapsed={format_duration(total_elapsed)} | remaining={format_duration(eta_total)}"
        )
        logger.info(
            f"  Train: G={avg_g_loss:.4e} | Content={avg_g_content:.4e} | "
            f"D={avg_d_loss:.4e} | G_Adv={avg_g_adv:.4e} | "
            f"D(real)={avg_d_real:.4e} | D(fake)={avg_d_fake:.4e}"
        )
        logger.info(f"  Test: {avg_test_loss:.4e}")
        logger.info(
            f"  Sub-loss: L1={avg_sub['L1']:.4e} L2={avg_sub['L2']:.4e} "
            f"Edge={avg_sub['Edge']:.4e} Contrast={avg_sub['Contrast']:.4e}"
        )
        logger.info(
            f"            Dark={avg_sub['Dark']:.4e} Bright={avg_sub['Bright']:.4e} "
            f"Lab={avg_sub['Lab']:.4e} LCh={avg_sub['LCh']:.4e} "
            f"Percep={avg_sub['Percep']:.4e} WB={avg_sub['WB']:.4e}"
        )
        logger.info(
            f"  Prediction stats: Mean={avg_sub['PredMean']:.4e} Std={avg_sub['PredStd']:.4e} "
            f"Min={avg_sub['PredMin']:.4e} Max={avg_sub['PredMax']:.4e}"
        )
        logger.info(
            f"  Saturation: DeadBlack={avg_sub['DeadBlack']:.4e} DeadWhite={avg_sub['DeadWhite']:.4e}"
        )
        logger.info(f"  Resources: GPU={gpu_alloc:.0f}MB | LR: G={current_lr_g:.2e} D={current_lr_d:.2e}")

        is_best = n_test > 0 and math.isfinite(avg_test_loss) and avg_test_loss < best_test_loss
        if is_best:
            best_test_loss = avg_test_loss
            logger.info(f"  New best model. Test Loss: {best_test_loss:.4e}")
        logger.info("-" * 60)

        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save(model.unet_branch.state_dict(), f"{ckpt_dir}/unet_branch_ep{epoch + 1}_{stamp}.pth")
            torch.save(model.unet_head.state_dict(), f"{ckpt_dir}/unet_head_ep{epoch + 1}_{stamp}.pth")
            if train_discriminator and discriminator is not None:
                torch.save(discriminator.state_dict(), f"{ckpt_dir}/discriminator_ep{epoch + 1}_{stamp}.pth")
            logger.info(f"Saved checkpoint files under {ckpt_dir}/ for epoch {epoch + 1}.")

        if is_best:
            torch.save(model.unet_branch.state_dict(), ckpt_g)
            torch.save(model.unet_head.state_dict(), ckpt_h)
            if train_discriminator and discriminator is not None:
                torch.save(discriminator.state_dict(), ckpt_d)
                logger.info(f"Best checkpoints: G={ckpt_g} Head={ckpt_h} D={ckpt_d}")
            else:
                logger.info(f"Best checkpoints: G={ckpt_g} Head={ckpt_h}")

    total_duration = time.time() - train_start
    logger.info("=" * 60)
    logger.info(f"Training complete. Total time: {format_duration(total_duration)}")
    if math.isfinite(best_test_loss):
        logger.info(f"Best Test Loss: {best_test_loss:.4e}")
    else:
        logger.info("Best Test Loss: unavailable (no finite epoch test loss)")
    logger.info(f"Log file: {log_path}")
    logger.info("=" * 60)

    eval_pairs(logger, model, device, native_max_side=native_max_side)


def eval_pairs(logger, model, device, native_max_side=1008):

    import cv2
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    def _resize_image(img, target_h, target_w):
        h, w = img.shape[:2]
        if (h, w) == (target_h, target_w):
            return img
        interp = cv2.INTER_LANCZOS4 if target_h > h or target_w > w else cv2.INTER_AREA
        return cv2.resize(img, (target_w, target_h), interpolation=interp)

    def _resize_max_side(img, max_side):
        if max_side <= 0 or max(img.shape[:2]) <= max_side:
            return img
        h, w = img.shape[:2]
        scale = max_side / float(max(h, w))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        return _resize_image(img, new_h, new_w)

    def _predict_rgb(model, raw_rgb):
        with torch.no_grad():
            pred = model([raw_rgb])
            pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (pred_np * 255).clip(0, 255).astype(np.uint8)

    eval_pairs_config = [
        ("SUIM_ENS", "data/SUIM_ENS/test/raw", "data/SUIM_ENS/test/ref"),
    ]

    model.eval()
    logger.info("=" * 60)
    logger.info("Post-training evaluation (PSNR/SSIM)")
    logger.info("=" * 60)

    for name, raw_dir, gt_dir in eval_pairs_config:
        if not os.path.isdir(raw_dir) or not os.path.isdir(gt_dir):
            logger.info(f"  {name:>12s}: skipped (path not found)")
            continue

        images = sorted([f for f in os.listdir(raw_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        if not images:
            continue

        psnr_252, ssim_252 = [], []
        psnr_native, ssim_native = [], []
        for filename in images:
            raw_path = os.path.join(raw_dir, filename)
            gt_path = os.path.join(gt_dir, filename)
            if not os.path.isfile(gt_path):
                continue

            raw_img = cv2.imread(raw_path)
            gt_img = cv2.imread(gt_path)
            if raw_img is None or gt_img is None:
                continue

            raw_252 = _resize_image(raw_img, 252, 252)
            gt_252 = _resize_image(gt_img, 252, 252)
            raw_252_rgb = cv2.cvtColor(raw_252, cv2.COLOR_BGR2RGB)
            gt_252_rgb = cv2.cvtColor(gt_252, cv2.COLOR_BGR2RGB)
            pred_252 = _predict_rgb(model, raw_252_rgb)
            psnr_252.append(peak_signal_noise_ratio(gt_252_rgb, pred_252, data_range=255))
            ssim_252.append(structural_similarity(gt_252_rgb, pred_252, channel_axis=2, data_range=255))

            if raw_img.shape[:2] == gt_img.shape[:2]:
                raw_native = _resize_max_side(raw_img, native_max_side)
                gt_native = _resize_max_side(gt_img, native_max_side)
                if raw_native.shape[:2] == gt_native.shape[:2]:
                    raw_native_rgb = cv2.cvtColor(raw_native, cv2.COLOR_BGR2RGB)
                    gt_native_rgb = cv2.cvtColor(gt_native, cv2.COLOR_BGR2RGB)
                    pred_native = _predict_rgb(model, raw_native_rgb)
                    if pred_native.shape[:2] == gt_native_rgb.shape[:2]:
                        psnr_native.append(peak_signal_noise_ratio(gt_native_rgb, pred_native, data_range=255))
                        ssim_native.append(
                            structural_similarity(gt_native_rgb, pred_native, channel_axis=2, data_range=255)
                        )

        avg_psnr_252 = np.mean(psnr_252) if psnr_252 else 0
        avg_ssim_252 = np.mean(ssim_252) if ssim_252 else 0
        avg_psnr_native = np.mean(psnr_native) if psnr_native else 0
        avg_ssim_native = np.mean(ssim_native) if ssim_native else 0
        logger.info(
            f"  {name:>12s} legacy252: PSNR={avg_psnr_252:.2f} dB | "
            f"SSIM={avg_ssim_252:.4f} | ({len(psnr_252)} images)"
        )
        logger.info(
            f"  {name:>12s} native:    PSNR={avg_psnr_native:.2f} dB | "
            f"SSIM={avg_ssim_native:.4f} | ({len(psnr_native)} images)"
        )

    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone DANet phase training")
    parser.add_argument("--checkpoint-g", type=str, default=None, help="DANet branch checkpoint path")
    parser.add_argument("--checkpoint-h", type=str, default=None, help="DANet head checkpoint path")
    parser.add_argument("--checkpoint-d", type=str, default=None, help="Discriminator checkpoint path")
    parser.add_argument("--scratch", action="store_true", help="Train from scratch and ignore all checkpoints")
    parser.add_argument("--start-epoch", type=int, default=0, help="Starting epoch index")
    parser.add_argument("--unet-variant", type=str, default="large", choices=["large"], help="DANet variant")
    parser.add_argument("--train-res-mode", type=str, default="multiscale", choices=["fixed", "multiscale", "native"], help="Resolution training mode")
    parser.add_argument("--train-sizes", type=str, default="252,378,504,756", help="Comma-separated square training sizes")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for bucketed training")
    parser.add_argument("--depth-process-res", type=str, default="auto", help="DepthAnything processing resolution, or auto to match each batch")
    parser.add_argument("--unet-dropout", type=float, default=0.3, help="DANet bottleneck dropout")
    parser.add_argument("--native-max-side", type=int, default=1008, help="Max long side for native resolution training")
    parser.add_argument("--lambda-adv", type=float, default=0.0, help="Adversarial loss weight for generator")
    parser.add_argument("--freeze-bn", action="store_true", help="Freeze generator BatchNorm; native mode enables this automatically")
    parser.add_argument("--fixed-split-prob", type=float, default=0.2, help="Fixed-mode probability of same-position random crop before resize fallback")
    discriminator_group = parser.add_mutually_exclusive_group()
    discriminator_group.add_argument("--train-discriminator", dest="train_discriminator", action="store_true", default=True, help="Enable discriminator construction, loading, forward, optimizer, and saving")
    discriminator_group.add_argument("--no-train-discriminator", dest="train_discriminator", action="store_false", help="Disable discriminator construction, loading, forward, optimizer, and saving")
    parser.add_argument("--data-dir", default="data/underwater", help="Paired train/test dataset root")
    parser.add_argument("--epochs", type=int, default=100, help="Total epoch count")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument("--checkpoint-dir", default="checkpoints/unet_large", help="Checkpoint directory")
    args = parser.parse_args()

    if args.scratch:
        train(
            scratch=True,
            start_epoch=0,
            unet_variant=args.unet_variant,
            train_res_mode=args.train_res_mode,
            train_sizes=args.train_sizes,
            batch_size=args.batch_size,
            depth_process_res=args.depth_process_res,
            unet_dropout=args.unet_dropout,
            native_max_side=args.native_max_side,
            lambda_adv=args.lambda_adv,
            freeze_bn=args.freeze_bn,
            fixed_split_prob=args.fixed_split_prob,
            train_discriminator=args.train_discriminator,
            data_dir=args.data_dir,
            epochs=args.epochs,
            lr=args.lr,
            num_workers=args.num_workers,
            checkpoint_dir=args.checkpoint_dir,
        )
    else:
        train(
            checkpoint_g=args.checkpoint_g,
            checkpoint_h=args.checkpoint_h,
            checkpoint_d=args.checkpoint_d,
            start_epoch=args.start_epoch,
            unet_variant=args.unet_variant,
            train_res_mode=args.train_res_mode,
            train_sizes=args.train_sizes,
            batch_size=args.batch_size,
            depth_process_res=args.depth_process_res,
            unet_dropout=args.unet_dropout,
            native_max_side=args.native_max_side,
            lambda_adv=args.lambda_adv,
            freeze_bn=args.freeze_bn,
            fixed_split_prob=args.fixed_split_prob,
            train_discriminator=args.train_discriminator,
            data_dir=args.data_dir,
            epochs=args.epochs,
            lr=args.lr,
            num_workers=args.num_workers,
            checkpoint_dir=args.checkpoint_dir,
        )
