import argparse
import datetime
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Discriminator import Discriminator
from data_loader import (
    ResolutionBucketBatchSampler,
    UnderwaterDataset,
    custom_collate_fn,
    resize_tensor_lanczos4,
)
from dpt_phase import DPTPhaseNet
from losses import UnderwaterEnhancementLoss
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


def _count_params(module):
    total = sum(param.numel() for param in module.parameters())
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return total, trainable


def _list_pairs(raw_dir, gt_dir):
    pairs = []
    for fname in os.listdir(raw_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            raw_path = os.path.join(raw_dir, fname)
            gt_path = os.path.join(gt_dir, fname)
            if os.path.isfile(gt_path):
                pairs.append((raw_path, gt_path))
    return pairs


def _load_matching_state(module, checkpoint_path, logger, label, device, strict=False):
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return False

    loaded = torch.load(checkpoint_path, map_location=device)
    if isinstance(loaded, dict) and "state_dict" in loaded:
        loaded = loaded["state_dict"]

    if strict:
        module.load_state_dict(loaded, strict=True)
        logger.info(f"Loaded {label}: {checkpoint_path}")
        return True

    current = module.state_dict()
    filtered = {}
    skipped = []
    for key, value in loaded.items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            filtered[key] = value
        else:
            skipped.append(key)

    result = module.load_state_dict(filtered, strict=False)
    logger.info(
        f"Loaded {label}: {checkpoint_path} | matched={len(filtered)} "
        f"missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)} skipped={len(skipped)}"
    )
    return True


def train(
    checkpoint_phase=None,
    checkpoint_g=None,
    checkpoint_h=None,
    checkpoint_d=None,
    checkpoint_unet=None,
    start_epoch=0,
    scratch=False,
    train_res_mode="multiscale",
    train_sizes=(252, 504, 756),
    batch_size=4,
    depth_process_res="auto",
    native_max_side=1008,
    lambda_adv=0.0,
    dpt_dropout=0.3,
    freeze_bn=False,
    fixed_split_prob=0.2,
    train_discriminator=True,
    data_dir="data/underwater",
    epochs=100,
    lr=1e-4,
    num_workers=4,
    checkpoint_dir="checkpoints/dpt",
):
    logger, log_path = setup_logger()
    logger.info(f"[Phase: DPT] Log file: {log_path}")
    logger.info(f"Training start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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

    train_pairs = _list_pairs(train_raw, train_gt)
    test_pairs = _list_pairs(test_raw, test_gt)
    if not train_pairs:
        raise ValueError(f"No matching training image pairs found under {data_dir}")

    train_sizes = _parse_train_sizes(train_sizes)
    lambda_adv = float(lambda_adv)
    freeze_bn = bool(freeze_bn or train_res_mode == "native")
    train_discriminator = bool(train_discriminator)
    if not train_discriminator:
        lambda_adv = 0.0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_dir = checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_phase = os.path.join(ckpt_dir, "dpt_phase_best.pth")
    ckpt_branch = os.path.join(ckpt_dir, "dpt_branch_best.pth")
    ckpt_head = os.path.join(ckpt_dir, "dpt_head_best.pth")
    ckpt_disc = os.path.join(ckpt_dir, "discriminator_best.pth")
    legacy_ckpt_phase = os.path.join(ckpt_dir, "dpt_phase.pth")
    legacy_ckpt_branch = os.path.join(ckpt_dir, "dpt_branch.pth")
    legacy_ckpt_head = os.path.join(ckpt_dir, "dpt_head.pth")
    legacy_ckpt_disc = os.path.join(ckpt_dir, "discriminator.pth")

    logger.info("=" * 60)
    logger.info("Training config")
    logger.info("=" * 60)
    logger.info(f"Dataset: {len(train_pairs)} train / {len(test_pairs)} test")
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
    logger.info(f"Freeze generator BatchNorm: {freeze_bn}")
    logger.info(f"DPT bottleneck dropout: {dpt_dropout}")
    logger.info(f"Train discriminator: {train_discriminator}")
    logger.info(f"Adversarial lambda: {lambda_adv}")
    if not train_discriminator:
        logger.info("Discriminator disabled: no D construction, checkpoint loading, forward, backward, or saving.")
    logger.info("Loss weights: L1=0.10 L2=0.1 Edge=0.20 Contrast=0.10 Dark=0.1 Bright=0.1 Lab=0.2 LCh=0.12 Percep=0.01 WB=0.1")
    logger.info(f"Checkpoint directory: {ckpt_dir}/")
    logger.info("Full DPT phase checkpoint saving: disabled")

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

    model = DPTPhaseNet(
        process_res=depth_process_res,
        unet_branch_path=checkpoint_unet,
        dpt_dropout=dpt_dropout,
    ).to(device)
    model.freeze_extractor()
    model.freeze_unet_branch()
    if model.loaded_unet_branch_path:
        logger.info(f"Loaded frozen DANet branch: {model.loaded_unet_branch_path}")
    else:
        logger.warning("Frozen DANet branch checkpoint not found; using randomly initialized frozen DANet branch.")

    discriminator = Discriminator(in_channels=3).to(device) if train_discriminator else None

    if scratch:
        logger.info("--scratch enabled: ignoring all DPT/discriminator checkpoints.")
    else:
        loaded_phase = _load_matching_state(model, checkpoint_phase, logger, "DPT phase checkpoint", device, strict=False)
        if not loaded_phase:
            loaded_branch = _load_matching_state(model.dpt_branch, checkpoint_g, logger, "DPT branch checkpoint", device, strict=False)
            if not loaded_branch:
                loaded_branch = _load_matching_state(model.dpt_branch, ckpt_branch, logger, "default DPT branch checkpoint", device, strict=False)
            if not loaded_branch:
                loaded_branch = _load_matching_state(model.dpt_branch, legacy_ckpt_branch, logger, "legacy DPT branch checkpoint", device, strict=False)
            loaded_head = _load_matching_state(model.dpt_head, checkpoint_h, logger, "DPT head checkpoint", device, strict=False)
            if not loaded_head:
                loaded_head = _load_matching_state(model.dpt_head, ckpt_head, logger, "default DPT head checkpoint", device, strict=False)
            if not loaded_head:
                loaded_head = _load_matching_state(model.dpt_head, legacy_ckpt_head, logger, "legacy DPT head checkpoint", device, strict=False)

            if not loaded_branch and not loaded_head:
                loaded_phase = _load_matching_state(model, ckpt_phase, logger, "legacy best DPT phase checkpoint", device, strict=False)
                if not loaded_phase:
                    _load_matching_state(model, legacy_ckpt_phase, logger, "legacy DPT phase checkpoint", device, strict=False)
            elif not loaded_branch or not loaded_head:
                logger.warning("Only part of the DPT split checkpoint was loaded; the missing module starts from scratch.")

        if train_discriminator:
            loaded_disc = _load_matching_state(discriminator, checkpoint_d, logger, "discriminator checkpoint", device, strict=False)
            if not loaded_disc:
                loaded_disc = _load_matching_state(discriminator, ckpt_disc, logger, "default discriminator checkpoint", device, strict=False)
            if not loaded_disc:
                _load_matching_state(discriminator, legacy_ckpt_disc, logger, "legacy discriminator checkpoint", device, strict=False)
        else:
            logger.info("Discriminator checkpoint loading skipped.")

    model.freeze_extractor()
    model.freeze_unet_branch()
    if freeze_bn:
        n_frozen_bn = _freeze_batchnorm(model)
        logger.info(f"Frozen generator BatchNorm layers: {n_frozen_bn}")

    unet_total, unet_trainable = _count_params(model.unet_branch)
    logger.info(f"Frozen DANet params: total={unet_total:,} trainable={unet_trainable:,}")

    trainable_params_g = [p for p in model.parameters() if p.requires_grad]
    optimizer_g = torch.optim.AdamW(trainable_params_g, lr=lr, weight_decay=1e-4)
    trainable_params_d = [p for p in discriminator.parameters() if p.requires_grad] if discriminator is not None else []
    optimizer_d = torch.optim.Adam(trainable_params_d, lr=lr, betas=(0.5, 0.999)) if train_discriminator and trainable_params_d else None

    criterion_g = UnderwaterEnhancementLoss().to(device)
    criterion_bce = nn.BCEWithLogitsLoss().to(device)
    scaler_g = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    scaler_d = torch.amp.GradScaler("cuda", enabled=device == "cuda") if optimizer_d is not None else None

    n_params_g = sum(p.numel() for p in trainable_params_g)
    n_params_d = sum(p.numel() for p in trainable_params_d)
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
                loss, sub = criterion_g(preds, gt_tensors, return_detail=True)
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
        model.freeze_extractor()
        model.freeze_unet_branch()
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
            preds = model(raw_imgs)

            loss_d = torch.tensor(0.0, device=device)
            d_real_out = torch.tensor(0.0, device=device)
            d_fake_out = torch.tensor(0.0, device=device)

            if train_discriminator and optimizer_d is not None and discriminator is not None:
                real = _to_discriminator_input(gt_tensors)
                fake = _to_discriminator_input(preds.detach())

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
            loss_g_adv = torch.tensor(0.0, device=device)
            if lambda_adv > 0 and discriminator is not None:
                with _autocast_context(device):
                    g_fake_out = discriminator(_to_discriminator_input(preds))
                    g_label = torch.full_like(g_fake_out, 0.9)
                    loss_g_adv = criterion_bce(g_fake_out, g_label)

            loss_g_content, sub_detail = criterion_g(preds, gt_tensors, return_detail=True)
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
                loss = criterion_g(preds, gt_tensors)
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
            torch.save(model.dpt_branch.state_dict(), f"{ckpt_dir}/dpt_branch_ep{epoch + 1}_{stamp}.pth")
            torch.save(model.dpt_head.state_dict(), f"{ckpt_dir}/dpt_head_ep{epoch + 1}_{stamp}.pth")
            if train_discriminator and discriminator is not None:
                torch.save(discriminator.state_dict(), f"{ckpt_dir}/discriminator_ep{epoch + 1}_{stamp}.pth")
            logger.info(f"Saved checkpoint files under {ckpt_dir}/ for epoch {epoch + 1}.")

        if is_best:
            torch.save(model.dpt_branch.state_dict(), ckpt_branch)
            torch.save(model.dpt_head.state_dict(), ckpt_head)
            if train_discriminator and discriminator is not None:
                torch.save(discriminator.state_dict(), ckpt_disc)
                logger.info(f"Best checkpoints: branch={ckpt_branch} head={ckpt_head} D={ckpt_disc}")
            else:
                logger.info(f"Best checkpoints: branch={ckpt_branch} head={ckpt_head}")

    total_duration = time.time() - train_start
    logger.info("=" * 60)
    logger.info(f"Training complete. Total time: {format_duration(total_duration)}")
    if math.isfinite(best_test_loss):
        logger.info(f"Best Test Loss: {best_test_loss:.4e}")
    else:
        logger.info("Best Test Loss: unavailable (no finite epoch test loss)")
    logger.info(f"Log file: {log_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage-2 DPT-DANet fusion training")
    parser.add_argument("--checkpoint-phase", type=str, default=None, help="Legacy full DPT phase checkpoint (load only)")
    parser.add_argument("--checkpoint-g", type=str, default=None, help="DPT branch checkpoint")
    parser.add_argument("--checkpoint-h", type=str, default=None, help="DPT head checkpoint")
    parser.add_argument("--checkpoint-d", type=str, default=None, help="Discriminator checkpoint")
    parser.add_argument("--checkpoint-unet", type=str, default=None, help="Frozen DANet branch checkpoint")
    parser.add_argument("--scratch", action="store_true", help="Ignore existing DPT/discriminator checkpoints")
    parser.add_argument("--start-epoch", type=int, default=0, help="Starting epoch")
    parser.add_argument("--train-res-mode", type=str, default="multiscale", choices=["fixed", "multiscale", "native"], help="Resolution training mode")
    parser.add_argument("--train-sizes", type=str, default="252,504,756", help="Comma-separated square training sizes")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for bucketed training")
    parser.add_argument("--depth-process-res", type=str, default="auto", help="DepthAnything processing resolution, or auto to match each batch")
    parser.add_argument("--native-max-side", type=int, default=1008, help="Max long side for native resolution training")
    parser.add_argument("--lambda-adv", type=float, default=0.0, help="Adversarial loss weight for generator")
    parser.add_argument("--dpt-dropout", type=float, default=0.3, help="DPT coarse-stage residual dropout")
    parser.add_argument("--freeze-bn", action="store_true", help="Freeze generator BatchNorm; native mode enables this automatically")
    parser.add_argument("--fixed-split-prob", type=float, default=0.2, help="Fixed-mode probability of same-position random crop before resize fallback")
    discriminator_group = parser.add_mutually_exclusive_group()
    discriminator_group.add_argument("--train-discriminator", dest="train_discriminator", action="store_true", default=True, help="Enable discriminator construction, loading, forward, optimizer, and saving")
    discriminator_group.add_argument("--no-train-discriminator", dest="train_discriminator", action="store_false", help="Disable discriminator construction, loading, forward, optimizer, and saving")
    parser.add_argument("--data-dir", default="data/underwater", help="Paired train/test dataset root")
    parser.add_argument("--epochs", type=int, default=100, help="Total epoch count")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument("--checkpoint-dir", default="checkpoints/dpt", help="Checkpoint directory")
    args = parser.parse_args()

    train(
        checkpoint_phase=args.checkpoint_phase,
        checkpoint_g=args.checkpoint_g,
        checkpoint_h=args.checkpoint_h,
        checkpoint_d=args.checkpoint_d,
        checkpoint_unet=args.checkpoint_unet,
        start_epoch=args.start_epoch,
        scratch=args.scratch,
        train_res_mode=args.train_res_mode,
        train_sizes=args.train_sizes,
        batch_size=args.batch_size,
        depth_process_res=args.depth_process_res,
        native_max_side=args.native_max_side,
        lambda_adv=args.lambda_adv,
        dpt_dropout=args.dpt_dropout,
        freeze_bn=args.freeze_bn,
        fixed_split_prob=args.fixed_split_prob,
        train_discriminator=args.train_discriminator,
        data_dir=args.data_dir,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
    )
