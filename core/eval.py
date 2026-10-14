import argparse
import csv
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


MODEL_OUTPUT_DIRS = {"DAGF": "results"}
AUTO_DISCOVER_OTHER_MODELS = False
OTHER_MODELS_ROOT = "comparisons"
SELECTED_MODELS = None
DATASETS = {
    "EUVP": {"gt_dir": "data/EUVP/test/gt", "output_subdirs": ("EUVP", "Inp")},
    "SUIM_ENS": {"gt_dir": "data/SUIM_ENS/test/ref", "output_subdirs": ("SUIM_ENS",)},
    "HICRD_paired": {"gt_dir": "data/HICRD_paired/testB", "output_subdirs": ("HICRD_paired",)},
}

ENABLED_METRICS = ("PSNR", "SSIM", "LPIPS", "UCIQE", "UIQM", "NIQE")


SUMMARY_CSV = "evaluation_results/metrics_summary.csv"
DETAIL_CSV = "evaluation_results/metrics_per_image.csv"


RESIZE_PREDICTION_TO_GT = True
RECURSIVE_IMAGE_SEARCH = False
SHOW_PROGRESS = True
CPU_WORKERS = min(16, os.cpu_count() or 1)
OPENCV_THREADS_PER_WORKER = 1
IQA_DEVICE = "auto"  


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
METRIC_ORDER = ("PSNR", "SSIM", "LPIPS", "UCIQE", "UIQM", "NIQE")
CPU_METRICS = {"PSNR", "SSIM", "UCIQE", "UIQM"}
DEEP_METRICS = {"LPIPS", "NIQE"}

try:
    import torch
    import pyiqa

    HAS_PYIQA = True
    PYIQA_IMPORT_ERROR = ""
except Exception as exc:
    HAS_PYIQA = False
    PYIQA_IMPORT_ERROR = str(exc)


def platform_path(path_value):

    if path_value is None:
        return None
    raw = os.path.expandvars(os.path.expanduser(str(path_value)))
    if os.name != "nt":
        match = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
        if match:
            parts = [part for part in re.split(r"[\\/]+", match.group(2)) if part]
            return Path("/mnt") / match.group(1).lower() / Path(*parts)
    return Path(raw)


def list_images(directory):
    directory = platform_path(directory)
    if directory is None or not directory.is_dir():
        return []
    iterator = directory.rglob("*") if RECURSIVE_IMAGE_SEARCH else directory.iterdir()
    return sorted(
        (path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: str(path).casefold(),
    )


def discover_models():
    models = {name: platform_path(path) for name, path in MODEL_OUTPUT_DIRS.items()}
    if AUTO_DISCOVER_OTHER_MODELS:
        root = platform_path(OTHER_MODELS_ROOT)
        if root.is_dir():
            for repository in sorted(root.iterdir(), key=lambda path: path.name.casefold()):
                output_dir = repository / "output"
                if repository.is_dir() and output_dir.is_dir():
                    models.setdefault(repository.name, output_dir)
    return models


def find_prediction_dir(model_output_dir, dataset_config):
    for subdir in dataset_config["output_subdirs"]:
        candidate = model_output_dir / subdir
        if candidate.is_dir():
            return candidate
    return None


def build_gt_map(gt_dir):
    gt_map = {}
    for path in list_images(gt_dir):
        gt_map.setdefault(path.stem.casefold(), path)
    return gt_map


def resolve_gt_path(filename, gt_map):
    stem = Path(filename).stem
    candidates = [stem]
    prefixes = ("enhanced_", "result_", "output_")
    suffixes = ("_enhanced", "_result", "_output", "_out")
    for prefix in prefixes:
        if stem.casefold().startswith(prefix):
            candidates.append(stem[len(prefix) :])
    for suffix in suffixes:
        if stem.casefold().endswith(suffix):
            candidates.append(stem[: -len(suffix)])
    for candidate in candidates:
        match = gt_map.get(candidate.casefold())
        if match is not None:
            return match
    return None


def calculate_uciqe_matlab(image_bgr):

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    eps = np.finfo(float).eps
    img_lum = lab[:, :, 0].astype(np.float64) / 255.0 + eps
    img_a = lab[:, :, 1].astype(np.float64) / 255.0
    img_b = lab[:, :, 2].astype(np.float64) / 255.0

    img_chr = np.sqrt(img_a**2 + img_b**2)
    img_sat = img_chr / np.sqrt(img_chr**2 + img_lum**2)

    aver_sat = np.mean(img_sat)
    aver_chr = np.mean(img_chr)
    var_chr = np.sqrt(np.mean(np.abs(1 - (aver_chr / (img_chr + eps)) ** 2)))
    tol_low, tol_high = np.percentile(img_lum, [1, 99])
    con_lum = tol_high - tol_low

    coefficient_1, coefficient_2, coefficient_3 = 0.4680, 0.2745, 0.2576
    return float(coefficient_1 * var_chr + coefficient_2 * con_lum + coefficient_3 * aver_sat)


def _trimmed_channel_statistics(values, trim_ratio=0.1):
    values = np.sort(values.reshape(-1).astype(np.float64))
    count = values.size
    trim = min(int(np.ceil(trim_ratio * count)), max(0, (count - 1) // 2))
    trimmed = values[trim : count - trim] if trim else values
    mean = float(np.mean(trimmed))
    variance = float(np.mean((values - mean) ** 2))
    return mean, variance


def _eme(channel, block_size=10):
    height, width = channel.shape
    eps = np.finfo(np.float64).eps
    block_rows = int(np.ceil(height / block_size))
    block_columns = int(np.ceil(width / block_size))
    padded = np.pad(
        channel,
        ((0, block_rows * block_size - height), (0, block_columns * block_size - width)),
        mode="edge",
    )
    blocks = padded.reshape(block_rows, block_size, block_columns, block_size).transpose(0, 2, 1, 3)
    minimum = np.min(blocks, axis=(2, 3))
    maximum = np.max(blocks, axis=(2, 3))
    valid = (maximum > eps) & (minimum > eps)
    values = np.zeros_like(maximum, dtype=np.float64)
    values[valid] = np.log(maximum[valid] / minimum[valid])
    return float(2.0 * np.sum(values) / max(values.size, 1))


def _uiconm(rgb, block_size=10):
    height, width = rgb.shape[:2]
    eps = np.finfo(np.float64).eps
    intensity = np.max(rgb, axis=2)
    block_rows = int(np.ceil(height / block_size))
    block_columns = int(np.ceil(width / block_size))
    padded = np.pad(
        intensity,
        ((0, block_rows * block_size - height), (0, block_columns * block_size - width)),
        mode="edge",
    )
    blocks = padded.reshape(block_rows, block_size, block_columns, block_size).transpose(0, 2, 1, 3)
    minimum = np.min(blocks, axis=(2, 3))
    maximum = np.max(blocks, axis=(2, 3))
    ratio = (maximum - minimum) / (maximum + minimum + eps)
    valid = ratio > eps
    values = np.zeros_like(ratio, dtype=np.float64)
    values[valid] = ratio[valid] * np.log(ratio[valid])
    return float(-np.sum(values) / max(values.size, 1))


def calculate_uiqm(image_bgr):

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    red, green, blue = cv2.split(rgb)

    rg = red - green
    yb = 0.5 * (red + green) - blue
    rg_mean, rg_variance = _trimmed_channel_statistics(rg)
    yb_mean, yb_variance = _trimmed_channel_statistics(yb)
    uicm = -0.0268 * np.sqrt(rg_mean**2 + yb_mean**2) + 0.1586 * np.sqrt(
        rg_variance + yb_variance
    )

    edge_eme = []
    for channel in (red, green, blue):
        sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        edge_eme.append(_eme(channel * np.hypot(sobel_x, sobel_y)))
    uism = 0.299 * edge_eme[0] + 0.587 * edge_eme[1] + 0.114 * edge_eme[2]
    uiconm = _uiconm(rgb)
    return float(0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm)


def calculate_psnr(gt_rgb, pred_rgb):
    difference = gt_rgb.astype(np.float64) - pred_rgb.astype(np.float64)
    mse = float(np.mean(difference**2))
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def _calculate_ssim_channel(gt_channel, pred_channel):
    gt_channel = gt_channel.astype(np.float64)
    pred_channel = pred_channel.astype(np.float64)
    constant_1 = (0.01 * 255.0) ** 2
    constant_2 = (0.03 * 255.0) ** 2

    mean_gt = cv2.GaussianBlur(gt_channel, (11, 11), 1.5)
    mean_pred = cv2.GaussianBlur(pred_channel, (11, 11), 1.5)
    mean_gt_squared = mean_gt**2
    mean_pred_squared = mean_pred**2
    mean_product = mean_gt * mean_pred

    variance_gt = cv2.GaussianBlur(gt_channel**2, (11, 11), 1.5) - mean_gt_squared
    variance_pred = cv2.GaussianBlur(pred_channel**2, (11, 11), 1.5) - mean_pred_squared
    covariance = cv2.GaussianBlur(gt_channel * pred_channel, (11, 11), 1.5) - mean_product

    numerator = (2.0 * mean_product + constant_1) * (2.0 * covariance + constant_2)
    denominator = (mean_gt_squared + mean_pred_squared + constant_1) * (
        variance_gt + variance_pred + constant_2
    )
    return float(np.mean(numerator / denominator))


def calculate_ssim(gt_rgb, pred_rgb):
    return float(
        np.mean(
            [
                _calculate_ssim_channel(gt_rgb[:, :, channel], pred_rgb[:, :, channel])
                for channel in range(3)
            ]
        )
    )


def finite_mean(values):
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    valid = array[~np.isnan(array)]
    return float(np.mean(valid)) if valid.size else None


def csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return value


def create_deep_metrics(metrics, requested_device):
    requested = [name for name in ("LPIPS", "NIQE") if name in metrics]
    if not requested:
        return {}, None
    if not HAS_PYIQA:
        print(f"[警告] pyiqa 不可用，跳过 {'/'.join(requested)}：{PYIQA_IMPORT_ERROR}")
        return {}, None

    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[警告] 指定了 CUDA，但 PyTorch 未检测到 GPU；LPIPS/NIQE 将跳过。")
        return {}, None

    models = {}
    for metric_name in requested:
        try:
            models[metric_name] = pyiqa.create_metric(metric_name.lower(), device=device)
            print(f"{metric_name} device: {device}")
        except Exception as exc:
            print(f"[警告] {metric_name} 初始化失败，将跳过：{exc}")
    return models, device


def image_to_tensor(image_bgr, device):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return (
        torch.from_numpy(np.ascontiguousarray(image_rgb))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div(255.0)
        .to(device)
    )


def new_detail_row(model_name, dataset_name, pred_path, gt_path):
    return {
        "model": model_name,
        "dataset": dataset_name,
        "filename": Path(pred_path).name,
        "pred_path": str(pred_path),
        "gt_path": str(gt_path) if gt_path else "",
        "resized": False,
        "error": "",
        "_failed_read": 0,
        "_metric_errors": 0,
        **{metric.lower(): None for metric in METRIC_ORDER},
    }


def evaluate_cpu_image(task):

    model_name, dataset_name, pred_path, gt_path, metrics, resize_to_gt = task
    row = new_detail_row(model_name, dataset_name, pred_path, gt_path)
    try:
        pred_img = cv2.imread(str(pred_path), cv2.IMREAD_COLOR)
        if pred_img is None:
            row["_failed_read"] = 1
            row["error"] = "cannot read prediction"
            return row

        for metric_name, metric_function in (
            ("UCIQE", calculate_uciqe_matlab),
            ("UIQM", calculate_uiqm),
        ):
            if metric_name in metrics:
                try:
                    row[metric_name.lower()] = metric_function(pred_img)
                except Exception as exc:
                    row["_metric_errors"] += 1
                    row["error"] += f"{metric_name}: {exc}; "

        if not gt_path or not ({"PSNR", "SSIM"} & set(metrics)):
            return row

        gt_img = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
        if gt_img is None:
            row["_failed_read"] += 1
            row["error"] += "cannot read ground truth; "
            return row

        pred_for_reference = pred_img
        if pred_img.shape != gt_img.shape:
            if not resize_to_gt:
                row["error"] += f"shape mismatch pred={pred_img.shape[:2]} gt={gt_img.shape[:2]}; "
                return row
            pred_for_reference = cv2.resize(
                pred_img,
                (gt_img.shape[1], gt_img.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            row["resized"] = True

        pred_rgb = cv2.cvtColor(pred_for_reference, cv2.COLOR_BGR2RGB)
        gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        if "PSNR" in metrics:
            row["psnr"] = calculate_psnr(gt_rgb, pred_rgb)
        if "SSIM" in metrics:
            row["ssim"] = calculate_ssim(gt_rgb, pred_rgb)
    except Exception as exc:
        row["_metric_errors"] += 1
        row["error"] += f"CPU worker: {exc}; "
    return row


def add_deep_metrics(row, metrics, deep_models, device):
    requested = DEEP_METRICS & set(metrics) & set(deep_models)
    if not requested or row["_failed_read"]:
        return row

    pred_img = cv2.imread(row["pred_path"], cv2.IMREAD_COLOR)
    if pred_img is None:
        row["_failed_read"] += 1
        row["error"] += "cannot reread prediction for deep metrics; "
        return row

    pred_tensor = None
    if "NIQE" in requested:
        try:
            pred_tensor = image_to_tensor(pred_img, device)
            with torch.inference_mode():
                row["niqe"] = float(deep_models["NIQE"](pred_tensor).item())
        except Exception as exc:
            row["_metric_errors"] += 1
            row["error"] += f"NIQE: {exc}; "

    if "LPIPS" not in requested or not row["gt_path"]:
        return row
    try:
        gt_img = cv2.imread(row["gt_path"], cv2.IMREAD_COLOR)
        if gt_img is None:
            row["_failed_read"] += 1
            row["error"] += "cannot read ground truth for LPIPS; "
            return row
        pred_for_reference = pred_img
        if pred_img.shape != gt_img.shape:
            if not RESIZE_PREDICTION_TO_GT:
                row["error"] += f"LPIPS shape mismatch pred={pred_img.shape[:2]} gt={gt_img.shape[:2]}; "
                return row
            pred_for_reference = cv2.resize(
                pred_img,
                (gt_img.shape[1], gt_img.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            row["resized"] = True
        pred_tensor = image_to_tensor(pred_for_reference, device)
        gt_tensor = image_to_tensor(gt_img, device)
        with torch.inference_mode():
            row["lpips"] = float(deep_models["LPIPS"](pred_tensor, gt_tensor).item())
    except Exception as exc:
        row["_metric_errors"] += 1
        row["error"] += f"LPIPS: {exc}; "
    return row


def empty_summary(model_name, dataset_name, pred_dir, gt_dir, status):
    reference_count = len(build_gt_map(gt_dir))
    return {
        "model": model_name,
        "dataset": dataset_name,
        "status": status,
        "pred_dir": str(pred_dir) if pred_dir else "",
        "gt_dir": str(gt_dir) if gt_dir else "",
        "prediction_count": 0,
        "reference_count": reference_count,
        "matched_count": 0,
        "missing_reference_count": reference_count,
        "unmatched_prediction_count": 0,
        "resized_count": 0,
        "failed_read_count": 0,
        "metric_error_count": 0,
        **{metric.lower(): None for metric in METRIC_ORDER},
    }


def evaluate_directory(
    model_name,
    dataset_name,
    pred_dir,
    gt_dir,
    metrics,
    deep_models,
    deep_device,
    show_progress,
    cpu_workers,
):
    pred_paths = list_images(pred_dir)
    gt_map = build_gt_map(gt_dir)
    reference_count = len(gt_map)
    tasks = []
    for pred_path in pred_paths:
        gt_path = resolve_gt_path(pred_path.name, gt_map) if gt_map else None
        tasks.append(
            (
                model_name,
                dataset_name,
                str(pred_path),
                str(gt_path) if gt_path else "",
                tuple(metrics),
                RESIZE_PREDICTION_TO_GT,
            )
        )

    cv2.setNumThreads(OPENCV_THREADS_PER_WORKER)
    if cpu_workers > 1:
        with ThreadPoolExecutor(max_workers=cpu_workers, thread_name_prefix="eval-cpu") as executor:
            cpu_results = executor.map(evaluate_cpu_image, tasks)
            details = list(
                tqdm(
                    cpu_results,
                    total=len(tasks),
                    desc=f"CPU {model_name} / {dataset_name}",
                    unit="img",
                    disable=not show_progress,
                )
            )
    else:
        details = list(
            tqdm(
                map(evaluate_cpu_image, tasks),
                total=len(tasks),
                desc=f"CPU {model_name} / {dataset_name}",
                unit="img",
                disable=not show_progress,
            )
        )

    if DEEP_METRICS & set(metrics) & set(deep_models):
        for row in tqdm(
            details,
            desc=f"IQA {model_name} / {dataset_name}",
            unit="img",
            disable=not show_progress,
        ):
            add_deep_metrics(row, metrics, deep_models, deep_device)

    matched_references = {
        row["gt_path"].casefold() for row in details if row["gt_path"]
    }
    unmatched_predictions = sum(1 for row in details if gt_map and not row["gt_path"])
    resized_count = sum(bool(row["resized"]) for row in details)
    failed_read_count = sum(row["_failed_read"] for row in details)
    metric_error_count = sum(row["_metric_errors"] for row in details)
    scores = {
        metric: [
            row[metric.lower()]
            for row in details
            if row.get(metric.lower()) is not None
        ]
        for metric in METRIC_ORDER
    }

    matched_count = len(matched_references)
    if reference_count == 0:
        status = "no_gt"
    elif matched_count < reference_count or unmatched_predictions:
        status = "partial"
    else:
        status = "ok"

    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "status": status,
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir) if gt_dir else "",
        "prediction_count": len(pred_paths),
        "reference_count": reference_count,
        "matched_count": matched_count,
        "missing_reference_count": max(reference_count - matched_count, 0),
        "unmatched_prediction_count": unmatched_predictions,
        "resized_count": resized_count,
        "failed_read_count": failed_read_count,
        "metric_error_count": metric_error_count,
        **{metric.lower(): finite_mean(scores[metric]) for metric in METRIC_ORDER},
    }
    return summary, details


def write_csv(path, rows, fieldnames):
    path = platform_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})
    return path


def select_names(items, requested, label, parser):
    if isinstance(requested, str):
        requested = [requested]
    if not requested or (len(requested) == 1 and requested[0].casefold() == "all"):
        return items
    lookup = {name.casefold(): name for name in items}
    selected = []
    unknown = []
    for value in requested:
        actual = lookup.get(value.casefold())
        if actual is None:
            unknown.append(value)
        elif actual not in selected:
            selected.append(actual)
    if unknown:
        parser.error(f"未知{label}: {', '.join(unknown)}；可选: {', '.join(items)}")
    return selected


def print_configuration(models, cpu_workers, iqa_device):
    print("=" * 88)
    print("评估路径配置（可在 eval.py 文件开头修改）")
    print("=" * 88)
    print("模型输出目录：")
    for name, path in models.items():
        print(f"  {name}: {path}")
    print("数据集 GT 目录：")
    for name, config in DATASETS.items():
        aliases = ", ".join(config["output_subdirs"])
        print(f"  {name}: {platform_path(config['gt_dir'])}  [输出子目录: {aliases}]")
    print(f"汇总 CSV: {platform_path(SUMMARY_CSV)}")
    print(f"逐图 CSV: {platform_path(DETAIL_CSV)}")
    print(f"CPU 并行线程: {cpu_workers}")
    print(f"LPIPS/NIQE 设备: {iqa_device}")
    print("=" * 88)


def format_score(value):
    return "-" if value is None else f"{value:.4f}"


def print_summary(rows, metrics):
    table_width = max(92, 72 + 10 * len(metrics))
    print("\n" + "=" * table_width)
    print(
        f"{'Model':<31} {'Dataset':<14} {'Status':<9} {'Pred':>5} {'Match':>6} "
        + " ".join(f"{metric:>9}" for metric in metrics)
    )
    print("-" * table_width)
    for row in rows:
        print(
            f"{row['model'][:30]:<31} {row['dataset'][:13]:<14} {row['status']:<9} "
            f"{row['prediction_count']:>5} {row['matched_count']:>6} "
            + " ".join(f"{format_score(row.get(metric.lower())):>9}" for metric in metrics)
        )
    print("=" * table_width)


def parse_metrics(values, parser):
    metrics = tuple(value.upper() for value in (values or ENABLED_METRICS))
    invalid = [value for value in metrics if value not in METRIC_ORDER]
    if invalid:
        parser.error(f"未知指标: {', '.join(invalid)}；可选: {', '.join(METRIC_ORDER)}")
    return tuple(dict.fromkeys(metrics))


def main():
    parser = argparse.ArgumentParser(
        description="批量计算多个水下图像增强模型、多个数据集的质量指标"
    )
    model_selection = parser.add_mutually_exclusive_group()
    model_selection.add_argument("--model", help="只评估一个已配置/自动发现的模型")
    model_selection.add_argument("--models", nargs="+", help="模型名，可写多个；默认使用文件顶部 SELECTED_MODELS")
    parser.add_argument("--datasets", nargs="+", help="数据集名，可写多个；默认 all")
    parser.add_argument(
        "--metrics", nargs="+", help="PSNR SSIM LPIPS UCIQE UIQM NIQE 的任意组合"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=CPU_WORKERS,
        help="PSNR/SSIM/UCIQE/UIQM 的 CPU 并行线程数；1 表示关闭并行",
    )
    parser.add_argument(
        "--iqa-device",
        choices=("auto", "cuda", "cpu"),
        default=IQA_DEVICE,
        help="LPIPS/NIQE 的运行设备",
    )
    parser.add_argument("--pred-dir", "--pred_dir", dest="pred_dir", help="兼容旧接口：仅评估一个输出目录")
    parser.add_argument("--gt-dir", "--gt_dir", dest="gt_dir", help="兼容旧接口：单目录模式的 GT")
    parser.add_argument("--model-name", default="single", help="单目录模式显示的模型名")
    parser.add_argument("--dataset-name", default="single", help="单目录模式显示的数据集名")
    parser.add_argument("--summary-csv", default=SUMMARY_CSV, help="汇总 CSV 保存路径")
    parser.add_argument("--detail-csv", default=DETAIL_CSV, help="逐图 CSV 保存路径")
    parser.add_argument("--no-save", action="store_true", help="只打印结果，不保存 CSV")
    parser.add_argument("--no-progress", action="store_true", help="关闭进度条")
    parser.add_argument("--list", action="store_true", help="只显示当前路径配置")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers 必须至少为 1")

    models = discover_models()
    print_configuration(models, args.workers, args.iqa_device)
    if args.list:
        return

    metrics = parse_metrics(args.metrics, parser)
    deep_models, deep_device = create_deep_metrics(metrics, args.iqa_device)
    summaries = []
    all_details = []
    show_progress = SHOW_PROGRESS and not args.no_progress

    if args.pred_dir:
        pred_dir = platform_path(args.pred_dir)
        gt_dir = platform_path(args.gt_dir)
        if not pred_dir.is_dir():
            parser.error(f"输出目录不存在: {pred_dir}")
        summary, details = evaluate_directory(
            args.model_name,
            args.dataset_name,
            pred_dir,
            gt_dir,
            metrics,
            deep_models,
            deep_device,
            show_progress,
            args.workers,
        )
        summaries.append(summary)
        all_details.extend(details)
    else:
        requested_models = [args.model] if args.model else args.models
        if requested_models is None:
            requested_models = SELECTED_MODELS
        selected_models = select_names(list(models), requested_models, "模型", parser)
        selected_datasets = select_names(list(DATASETS), args.datasets, "数据集", parser)
        for model_name in selected_models:
            model_output_dir = models[model_name]
            for dataset_name in selected_datasets:
                dataset_config = DATASETS[dataset_name]
                gt_dir = platform_path(dataset_config["gt_dir"])
                pred_dir = find_prediction_dir(model_output_dir, dataset_config)
                if pred_dir is None:
                    print(f"[跳过] {model_name} / {dataset_name}: 找不到输出子目录")
                    summaries.append(
                        empty_summary(
                            model_name,
                            dataset_name,
                            model_output_dir,
                            gt_dir,
                            "missing_output",
                        )
                    )
                    continue
                if not list_images(pred_dir):
                    print(f"[跳过] {model_name} / {dataset_name}: 输出目录为空 ({pred_dir})")
                    summaries.append(
                        empty_summary(model_name, dataset_name, pred_dir, gt_dir, "empty_output")
                    )
                    continue
                print(f"\n评估 {model_name} / {dataset_name}: {pred_dir}")
                summary, details = evaluate_directory(
                    model_name,
                    dataset_name,
                    pred_dir,
                    gt_dir,
                    metrics,
                    deep_models,
                    deep_device,
                    show_progress,
                    args.workers,
                )
                summaries.append(summary)
                all_details.extend(details)

    print_summary(summaries, metrics)

    if not args.no_save:
        summary_fields = [
            "model",
            "dataset",
            "status",
            "pred_dir",
            "gt_dir",
            "prediction_count",
            "reference_count",
            "matched_count",
            "missing_reference_count",
            "unmatched_prediction_count",
            "resized_count",
            "failed_read_count",
            "metric_error_count",
            *[metric.lower() for metric in METRIC_ORDER],
        ]
        detail_fields = [
            "model",
            "dataset",
            "filename",
            "pred_path",
            "gt_path",
            "resized",
            "error",
            *[metric.lower() for metric in METRIC_ORDER],
        ]
        summary_path = write_csv(args.summary_csv, summaries, summary_fields)
        detail_path = write_csv(args.detail_csv, all_details, detail_fields)
        print(f"\n汇总结果已保存: {summary_path}")
        print(f"逐图结果已保存: {detail_path}")


if __name__ == "__main__":
    main()
