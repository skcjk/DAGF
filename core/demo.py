import argparse
import math
import os

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from dpt_phase import DPTPhaseNet
from unet_phase import UNetPhaseNet


def _resize_interpolation(src_h, src_w, dst_h, dst_w):
    if dst_h > src_h or dst_w > src_w:
        return cv2.INTER_LANCZOS4
    return cv2.INTER_AREA


def _aligned_hw(h, w, multiple=14, target_size=None):
    if multiple <= 1 and target_size is None:
        return h, w

    if target_size is None and h % multiple == 0 and w % multiple == 0:
        return h, w

    if target_size is not None:
        max_h, max_w = target_size
        scale = min(max_h / float(h), max_w / float(w))
        desired_h = max(1.0, h * scale)
        desired_w = max(1.0, w * scale)
    else:
        max_h, max_w = None, None
        desired_h = float(h)
        desired_w = float(w)

    if multiple <= 1:
        return max(1, int(round(desired_h))), max(1, int(round(desired_w)))

    ratio = w / float(h)
    h_center = desired_h / multiple
    w_center = desired_w / multiple
    h_min = max(1, int(math.floor(h_center)) - 4)
    h_max = max(h_min, int(math.ceil(h_center)) + 4)
    w_min = max(1, int(math.floor(w_center)) - 4)
    w_max = max(w_min, int(math.ceil(w_center)) + 4)

    best_hw = None
    best_score = float("inf")
    for h_units in range(h_min, h_max + 1):
        cand_h = h_units * multiple
        if max_h is not None and cand_h > max_h:
            continue
        for w_units in range(w_min, w_max + 1):
            cand_w = w_units * multiple
            if max_w is not None and cand_w > max_w:
                continue

            aspect_error = abs((cand_w / float(cand_h)) - ratio) / max(ratio, 1e-6)
            size_error = abs(cand_h - desired_h) / desired_h + abs(cand_w - desired_w) / desired_w
            upscale_penalty = 0.0
            if target_size is None and (cand_h > h or cand_w > w):
                upscale_penalty = 0.25
            score = aspect_error * 5.0 + size_error + upscale_penalty
            if score < best_score:
                best_score = score
                best_hw = (cand_h, cand_w)

    if best_hw is not None:
        return best_hw

    fallback_h = max(multiple, int(math.floor(desired_h / multiple)) * multiple)
    fallback_w = max(multiple, int(math.floor(desired_w / multiple)) * multiple)
    return fallback_h, fallback_w


def _resize_for_network(raw_img, target_size=None, multiple=14):
    h, w = raw_img.shape[:2]
    target_h, target_w = _aligned_hw(h, w, multiple=multiple, target_size=target_size)
    if (target_h, target_w) == (h, w):
        return raw_img

    interp = _resize_interpolation(h, w, target_h, target_w)
    print(f"Aligned inference size: {h}x{w} -> {target_h}x{target_w}")
    return cv2.resize(raw_img, (target_w, target_h), interpolation=interp)


def _predict_image(raw_img, model, device):
    original_h, original_w = raw_img.shape[:2]
    raw_img = _resize_for_network(raw_img, multiple=14)
    input_data = [np.array(raw_img)]

    with torch.no_grad():
        device_type = "cuda" if device.type == "cuda" else "cpu"
        with torch.autocast(device_type=device_type, enabled=device.type == "cuda"):
            preds = model(input_data)

    out_img_np = preds.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_img_np = np.clip(out_img_np, 0.0, 1.0)
    if out_img_np.shape[:2] != (original_h, original_w):
        out_img_np = cv2.resize(out_img_np, (original_w, original_h), interpolation=cv2.INTER_LANCZOS4)
    return out_img_np


def _tile_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, stride))
    last_start = length - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _smooth_axis_weight(length, overlap, touches_start, touches_end):
    weight = np.ones(length, dtype=np.float32)
    ramp_len = min(overlap, max(length - 1, 0))
    if ramp_len <= 0:
        return weight

    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, ramp_len, endpoint=False))
    if not touches_start:
        weight[:ramp_len] = np.minimum(weight[:ramp_len], ramp)
    if not touches_end:
        weight[-ramp_len:] = np.minimum(weight[-ramp_len:], ramp[::-1])
    return weight


def _tile_weight(tile_h, tile_w, overlap, y0, x0, img_h, img_w):
    y_weight = _smooth_axis_weight(tile_h, overlap, y0 == 0, y0 + tile_h == img_h)
    x_weight = _smooth_axis_weight(tile_w, overlap, x0 == 0, x0 + tile_w == img_w)
    return y_weight[:, None] * x_weight[None, :]


def _predict_tiled(raw_img, model, device, tile_size, tile_overlap):
    img_h, img_w = raw_img.shape[:2]
    stride = tile_size - tile_overlap
    y_starts = _tile_starts(img_h, tile_size, stride)
    x_starts = _tile_starts(img_w, tile_size, stride)
    print(
        f"Tiled inference: tile_size={tile_size}, overlap={tile_overlap}, "
        f"tiles={len(y_starts) * len(x_starts)}"
    )

    accum = np.zeros((img_h, img_w, 3), dtype=np.float32)
    weights = np.zeros((img_h, img_w, 1), dtype=np.float32)

    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + tile_size, img_h)
            x1 = min(x0 + tile_size, img_w)
            tile = raw_img[y0:y1, x0:x1]
            pred = _predict_image(tile, model, device)

            if pred.shape[:2] != tile.shape[:2]:
                pred = cv2.resize(pred, (tile.shape[1], tile.shape[0]), interpolation=cv2.INTER_LANCZOS4)

            weight = _tile_weight(tile.shape[0], tile.shape[1], tile_overlap, y0, x0, img_h, img_w)
            weight = weight[..., None]
            accum[y0:y1, x0:x1] += pred * weight
            weights[y0:y1, x0:x1] += weight

    return accum / np.maximum(weights, 1e-6)


def enhance_image(img_path, output_path, model, device, target_size=None, tile_size=0, tile_overlap=64):
    print(f"Processing: {img_path} ...")

    raw_img = imageio.imread(img_path)
    original_shape = raw_img.shape[:2]
    raw_img = _resize_for_network(raw_img, target_size=target_size, multiple=14)

    if tile_size and max(raw_img.shape[:2]) > tile_size:
        out_img_np = _predict_tiled(raw_img, model, device, tile_size, tile_overlap)
    else:
        out_img_np = _predict_image(raw_img, model, device)

    out_img_np = np.clip(out_img_np, 0.0, 1.0)
    out_img_np = (out_img_np * 255.0).astype(np.uint8)

    if out_img_np.shape[:2] != original_shape:
        out_img_np = cv2.resize(out_img_np, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LANCZOS4)

    base, ext = os.path.splitext(output_path)
    if ext.lower() in {".jpg", ".jpeg"}:
        output_path = base + ".png"
    imageio.imwrite(output_path, out_img_np)
    print(f"Saved to: {output_path}")


def _load_unet_weights(model, weights_path, device):
    if os.path.isfile(weights_path):
        print(f"Loading checkpoint: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint, strict=True)
        return

    if os.path.isdir(weights_path):
        branch_candidates = [
            os.path.join(weights_path, "unet_branch.pth"),
            os.path.join(weights_path, "unet_branch_best.pth"),
        ]
        head_candidates = [
            os.path.join(weights_path, "unet_head.pth"),
            os.path.join(weights_path, "unet_head_best.pth"),
        ]
        branch_path = next((path for path in branch_candidates if os.path.isfile(path)), None)
        head_path = next((path for path in head_candidates if os.path.isfile(path)), None)
        if branch_path is not None and head_path is not None:
            print(f"Loading UNet branch: {branch_path}")
            model.unet_branch.load_state_dict(torch.load(branch_path, map_location=device), strict=True)
            print(f"Loading UNet head: {head_path}")
            model.unet_head.load_state_dict(torch.load(head_path, map_location=device), strict=True)
            return
        raise FileNotFoundError(
            f"Cannot find unet_branch.pth and unet_head.pth under: {weights_path}"
        )

    raise FileNotFoundError(f"Weights not found: {weights_path}")


def _unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _load_dpt_weights(model, weights_path, device):
    if os.path.isfile(weights_path):
        print(f"Loading DPT phase checkpoint: {weights_path}")
        checkpoint = _unwrap_state_dict(torch.load(weights_path, map_location=device))
        model.load_state_dict(checkpoint, strict=True)
        return "phase"

    if os.path.isdir(weights_path):
        branch_candidates = [
            os.path.join(weights_path, "dpt_branch_best.pth"),
            os.path.join(weights_path, "dpt_branch.pth"),
        ]
        head_candidates = [
            os.path.join(weights_path, "dpt_head_best.pth"),
            os.path.join(weights_path, "dpt_head.pth"),
        ]
        branch_path = next((path for path in branch_candidates if os.path.isfile(path)), None)
        head_path = next((path for path in head_candidates if os.path.isfile(path)), None)
        if branch_path is not None and head_path is not None:
            if model.loaded_unet_branch_path is None:
                raise FileNotFoundError(
                    "DPT branch/head inference also requires a frozen UNet branch. "
                    "Pass --unet-weights checkpoints/unet_large/unet_branch_best.pth."
                )
            print(f"Loading frozen UNet branch: {model.loaded_unet_branch_path}")
            print(f"Loading DPT branch: {branch_path}")
            model.dpt_branch.load_state_dict(
                _unwrap_state_dict(torch.load(branch_path, map_location=device)),
                strict=True,
            )
            print(f"Loading DPT head: {head_path}")
            model.dpt_head.load_state_dict(
                _unwrap_state_dict(torch.load(head_path, map_location=device)),
                strict=True,
            )
            return "split"

        phase_candidates = [
            os.path.join(weights_path, "dpt_phase_best.pth"),
            os.path.join(weights_path, "dpt_phase.pth"),
        ]
        phase_path = next((path for path in phase_candidates if os.path.isfile(path)), None)
        if phase_path is not None:
            print(f"Loading DPT phase checkpoint: {phase_path}")
            checkpoint = _unwrap_state_dict(torch.load(phase_path, map_location=device))
            model.load_state_dict(checkpoint, strict=True)
            return "phase"

        raise FileNotFoundError(
            f"Cannot find DPT branch/head or phase checkpoints under: {weights_path}"
        )

    raise FileNotFoundError(f"Weights not found: {weights_path}")


def main():
    parser = argparse.ArgumentParser(description="UNet/DPT underwater enhancement demo")
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "dpt"], help="Enhancement model")
    parser.add_argument("--input", type=str, required=True, help="Input image path or folder")
    parser.add_argument("--output", type=str, default="results", help="Output path or folder")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Selected model checkpoint file or checkpoint directory",
    )
    parser.add_argument(
        "--unet-weights",
        type=str,
        default=None,
        help="Frozen UNet branch checkpoint used by DPT split checkpoints",
    )
    parser.add_argument("--resize", action="store_true", help="Fit input inside 1008x1008 before 14-aligned inference")
    parser.add_argument("--tile-size", type=int, default=0, help="Enable tiled inference with this tile size; 0 disables it")
    parser.add_argument("--tile-overlap", type=int, default=64, help="Overlap pixels used for tiled inference")
    parser.add_argument("--process-res", type=str, default="auto", help="DepthAnything processing resolution, or auto to match the inference image")
    parser.add_argument("--unet-dropout", type=float, default=0.3, help="UNet bottleneck dropout used at construction")
    parser.add_argument("--dpt-dropout", type=float, default=0.3, help="DPT coarse-stage dropout used at construction")
    args = parser.parse_args()

    if args.tile_size < 0:
        raise ValueError("--tile-size must be >= 0")
    if args.tile_overlap < 0:
        raise ValueError("--tile-overlap must be >= 0")
    if args.tile_size > 0 and args.tile_overlap >= args.tile_size:
        raise ValueError("--tile-overlap must be smaller than --tile-size")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.model == "unet":
        model = UNetPhaseNet(
            process_res=args.process_res,
            unet_variant="large",
            unet_dropout=args.unet_dropout,
        )
        _load_unet_weights(model, args.weights, "cpu")
    else:
        model = DPTPhaseNet(
            process_res=args.process_res,
            unet_branch_path=args.unet_weights,
            dpt_dropout=args.dpt_dropout,
        )
        _load_dpt_weights(model, args.weights, "cpu")
        model.freeze_extractor()
        model.freeze_unet_branch()

    model = model.to(device)
    model.eval()

    target_size = (1008, 1008) if args.resize else None

    if os.path.isfile(args.input):
        if not os.path.splitext(args.output)[-1]:
            os.makedirs(args.output, exist_ok=True)
            filename = os.path.basename(args.input)
            out_path = os.path.join(args.output, filename)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            out_path = args.output

        enhance_image(args.input, out_path, model, device, target_size, args.tile_size, args.tile_overlap)
        return

    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        image_names = [f for f in os.listdir(args.input) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for img_name in image_names:
            in_path = os.path.join(args.input, img_name)
            out_path = os.path.join(args.output, img_name)
            enhance_image(in_path, out_path, model, device, target_size, args.tile_size, args.tile_overlap)
        return

    raise FileNotFoundError("Invalid input path, please provide an image file or a folder.")


if __name__ == "__main__":
    main()
