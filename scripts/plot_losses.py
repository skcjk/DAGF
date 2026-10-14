import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NUM = r"[\d.naninfEe\-\+]+"
SUMMARY_KEYS = [
    ("G", "G"),
    ("G_Content", "Cnt"),
    ("D", "D"),
    ("Test", "Test"),
    ("EUVP_PSNR", "EUVP"),
    ("G_Adv", "GAdv"),
    ("L1", "L1"),
    ("L2", "L2"),
    ("Edge", "Edge"),
    ("Contrast", "Ctr"),
    ("Dark", "Dark"),
    ("Bright", "Brt"),
    ("Lab", "Lab"),
    ("LCh", "LCh"),
    ("Percep", "Prc"),
    ("WB", "WB"),
    ("DeadBlack", "DBlk"),
    ("DeadWhite", "DWht"),
    ("PredMean", "PMean"),
    ("PredStd", "PStd"),
    ("PredMin", "PMin"),
    ("PredMax", "PMax"),
    ("D_real", "Dre"),
    ("D_fake", "Dfk"),
    ("LR_G", "LR_G"),
    ("LR_D", "LR_D"),
]


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def metric_pairs(line):
    return dict(re.findall(r"([A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?)\s*[:=]\s*(" + NUM + ")", line))


def maybe_legend(ax, *args, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(*args, **kwargs)


def _append_if_present(data, pairs, mapping):
    for src, dst in mapping:
        if src in pairs:
            data[dst].append(safe_float(pairs[src]))


def _append_epoch_metrics(data, pairs):
    _append_if_present(
        data,
        pairs,
        [
            ("L1", "L1"),
            ("L2", "L2"),
            ("Edge", "Edge"),
            ("Contrast", "Contrast"),
            ("Dark", "Dark"),
            ("Bright", "Bright"),
            ("Lab", "Lab"),
            ("LCh", "LCh"),
            ("Percep", "Percep"),
            ("WB", "WB"),
            ("DeadBlack", "DeadBlack"),
            ("DeadWhite", "DeadWhite"),
            ("Mean", "PredMean"),
            ("Std", "PredStd"),
            ("Min", "PredMin"),
            ("Max", "PredMax"),
        ],
    )


def parse_log(filepath):
    data = defaultdict(list)
    per_batch = defaultdict(list)
    euvp_psnr_by_epoch = []
    lines = Path(filepath).read_text(encoding="utf-8-sig", errors="replace").splitlines()

    for i, line in enumerate(lines):
        match = re.search(r"\[Epoch\s+(\d+)/(\d+)\]", line)
        if not match:
            continue

        epoch, total = int(match.group(1)), int(match.group(2))
        data["epoch"].append(epoch)
        data["_total"] = total
        epoch_euvp_psnr = float("nan")

        for nl in lines[i + 1:i + 10]:
            if re.search(r"\[Epoch\s+\d+/\d+\]", nl):
                break
            pairs = metric_pairs(nl)
            _append_epoch_metrics(data, pairs)
            if "Train" in nl and "G" in pairs:
                _append_if_present(
                    data,
                    pairs,
                    [
                        ("G", "G"),
                        ("Content", "G_Content"),
                        ("D", "D"),
                        ("G_Adv", "G_Adv"),
                        ("D(real)", "D_real"),
                        ("D(fake)", "D_fake"),
                    ],
                )
            elif re.search(r"\bTest\b", nl):
                test_match = re.search(r"Test\s*[:=]?\s*(" + NUM + ")", nl)
                if test_match:
                    data["Test"].append(safe_float(test_match.group(1)))
            if "EUVP" in nl and "PSNR" in nl:
                euvp_match = re.search(
                    r"EUVP\s+(?:demo\d+|native)\s+PSNR\s*:\s*(" + NUM + ")",
                    nl,
                )
                if euvp_match:
                    epoch_euvp_psnr = safe_float(euvp_match.group(1))
            elif "Resources" in nl or "LR:" in nl:
                lr_match = re.search(r"LR:\s*G[=:](" + NUM + r")\s+D[=:](" + NUM + ")", nl)
                if lr_match:
                    data["LR_G"].append(safe_float(lr_match.group(1)))
                    data["LR_D"].append(safe_float(lr_match.group(2)))
        euvp_psnr_by_epoch.append(epoch_euvp_psnr)

    if _valid_values(euvp_psnr_by_epoch):
        data["EUVP_PSNR"] = euvp_psnr_by_epoch

    for line in lines:
        if re.search(r"E\d+/\d+\s+B\d+/\d+", line) is None:
            continue

        match = re.search(
            r"E(\d+)/\d+\s+B(\d+)/\d+\s+\|\s+"
            r"G[:=](" + NUM + r")\s+\(C[:=](" + NUM + r")\s+A[:=](" + NUM + r")"
            r"(?:\s+(?:lambda_adv|lambda|lam|l_adv)[:=](" + NUM + r"))?\)\s+\|\s+"
            r"D[:=](" + NUM + r")\s+\(R[:=](" + NUM + r")\s+F[:=](" + NUM + r")\)",
            line,
        )
        if not match:
            continue

        per_batch["epoch"].append(int(match.group(1)))
        per_batch["batch"].append(int(match.group(2)))
        per_batch["G"].append(safe_float(match.group(3)))
        per_batch["C"].append(safe_float(match.group(4)))
        per_batch["G_Adv"].append(safe_float(match.group(5)))
        if match.group(6) is not None:
            per_batch["lambda_adv"].append(safe_float(match.group(6)))
        per_batch["D"].append(safe_float(match.group(7)))
        per_batch["D_real"].append(safe_float(match.group(8)))
        per_batch["D_fake"].append(safe_float(match.group(9)))

    if per_batch["epoch"]:
        steps_per_epoch = max(per_batch["batch"])
        per_batch["step"] = [
            (epoch - 1) * steps_per_epoch + batch
            for epoch, batch in zip(per_batch["epoch"], per_batch["batch"])
        ]

    return data, per_batch


def _valid_values(values):
    return [v for v in values if not np.isnan(v) and not np.isinf(v)]


def text_summary(data_dicts, names, output_path):
    out = ["=" * 120, "  Training Loss Summary", "=" * 120]

    for idx, (data, _) in enumerate(data_dicts):
        name = names[idx]
        epochs = data.get("epoch", [])
        out.append(f"\n-- {name} --")

        if not epochs:
            out.append("  (no epoch data)")
            continue

        active_keys = [(key, abbr) for key, abbr in SUMMARY_KEYS if data.get(key)]
        if not active_keys:
            out.append("  (no metric data)")
            continue

        if len(epochs) <= 30:
            step = 1
        elif len(epochs) <= 60:
            step = 2
        elif len(epochs) <= 100:
            step = 5
        else:
            step = max(1, len(epochs) // 50)

        header = f"  {'Ep':>5s}" + "".join(f" {abbr:>10s}" for _, abbr in active_keys)
        out.append(header)
        out.append("  " + "-" * (6 + 11 * len(active_keys)))

        for i, epoch in enumerate(epochs):
            if i % step != 0 and i != 0 and i != len(epochs) - 1:
                continue
            row = f"  {epoch:>5d}"
            for key, _ in active_keys:
                values = data.get(key, [])
                value = values[i] if i < len(values) else float("nan")
                row += f" {'-':>10s}" if np.isnan(value) or np.isinf(value) else f" {value:>10.4e}"
            out.append(row)

        out.append(f"\n  Summary (epoch {epochs[0]} -> {epochs[-1]})")
        for key, abbr in active_keys:
            values = _valid_values(data.get(key, []))
            if not values:
                continue
            first, last = values[0], values[-1]
            indexed_values = [
                (i, value)
                for i, value in enumerate(data.get(key, []))
                if not np.isnan(value) and not np.isinf(value)
            ]
            if key == "EUVP_PSNR":
                best_idx, best = max(indexed_values, key=lambda item: item[1])
            else:
                best_idx, best = min(indexed_values, key=lambda item: item[1])
            best_ep = epochs[best_idx] if best_idx < len(epochs) else "?"
            delta = last - first
            direction = "down" if delta < 0 else ("up" if delta > 0 else "flat")
            pct = abs(delta) / (abs(first) + 1e-8) * 100
            out.append(
                f"    {abbr:>6s}: {first:.4e} -> {last:.4e} {direction} | "
                f"delta={delta:+.4e} ({pct:.1f}%) | best={best:.4e} @ep{best_ep}"
            )

    out.append("\n" + "=" * 120)
    text = "\n".join(out)
    Path(output_path).write_text(text, encoding="utf-8")
    print(text)
    return text


def _xy(data, key):
    values = data.get(key, [])
    epochs = data.get("epoch", [])
    pairs = [(e, v) for e, v in zip(epochs, values) if not np.isnan(v) and not np.isinf(v)]
    if not pairs:
        return [], []
    xs, ys = zip(*pairs)
    return list(xs), list(ys)


def plot_all(data_dicts, names, output_dir, prefix=""):
    output_dir = Path(output_dir)
    cmap = plt.cm.tab10
    colors = [cmap(i % 10) for i in range(len(data_dicts))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    has_lr = False
    for idx, (data, pb_data) in enumerate(data_dicts):
        for key, marker, label in [("G", "o-", "G"), ("G_Content", "^-", "Content"), ("Test", "s--", "Test")]:
            x, y = _xy(data, key)
            if x:
                ax1.plot(x, y, marker, c=colors[idx], ms=3, lw=1.2, label=f"{names[idx]} {label}")

        for key, style, label in [("LR_G", "-", "LR_G"), ("LR_D", "--", "LR_D")]:
            x, y = _xy(data, key)
            if x:
                ax2.plot(x, y, style, c=colors[idx], lw=1.3, label=f"{names[idx]} {label}")
                has_lr = True

        if pb_data.get("lambda_adv") and pb_data.get("epoch"):
            ep_avg = {}
            for epoch, value in zip(pb_data["epoch"], pb_data["lambda_adv"]):
                ep_avg.setdefault(epoch, []).append(value)
            xs = sorted(ep_avg)
            ys = [np.mean(ep_avg[e]) for e in xs]
            ax2b = ax2.twinx()
            ax2b.set_ylabel("lambda_adv", fontsize=8, color="red")
            ax2b.plot(xs, ys, ":", c="red", lw=1.2, alpha=0.7, label=f"{names[idx]} lambda_adv")
            maybe_legend(ax2b, fontsize=6, loc="lower right")
            has_lr = True

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("G Train and Test")
    maybe_legend(ax1, fontsize=7)
    ax1.grid(alpha=0.3)
    if has_lr:
        ax2.set_yscale("log")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("LR / lambda_adv")
    ax2.set_title("LR Schedule and lambda_adv")
    maybe_legend(ax2, fontsize=6)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}loss_lr.png", dpi=150)
    plt.close(fig)

    has_euvp_psnr = any(_xy(data, "EUVP_PSNR")[0] for data, _ in data_dicts)
    if has_euvp_psnr:
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        for idx, (data, _) in enumerate(data_dicts):
            x, y = _xy(data, "EUVP_PSNR")
            if not x:
                continue
            ax.plot(x, y, "o-", c=colors[idx], ms=4, lw=1.4, label=names[idx])
            best_idx = int(np.argmax(y))
            best_x, best_y = x[best_idx], y[best_idx]
            ax.plot(best_x, best_y, marker="*", c=colors[idx], ms=13)
            ax.annotate(
                f"{best_y:.3f} dB @ep{best_x}",
                (best_x, best_y),
                xytext=(6, 8),
                textcoords="offset points",
                fontsize=8,
                color=colors[idx],
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title("EUVP PSNR")
        maybe_legend(ax, fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}euvp_psnr.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    subplots = [
        ("L1", "L1"),
        ("L2", "L2"),
        ("Edge", "Edge MultiScale"),
        ("Contrast", "Contrast"),
        ("Dark", "Dark"),
        ("Bright", "Bright"),
        ("Lab", "Lab"),
        ("LCh", "LCh"),
        ("Percep", "Perceptual"),
        ("WB", "White Balance"),
        (None, None),
        (None, None),
    ]
    for ax, (key, title) in zip(axes.flat, subplots):
        if key is None:
            ax.set_visible(False)
            continue
        for idx, (data, _) in enumerate(data_dicts):
            x, y = _xy(data, key)
            if x:
                ax.plot(x, y, "o-", c=colors[idx], ms=3, lw=1.2, label=names[idx])
        ax.set_title(title)
        maybe_legend(ax, fontsize=6)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}sub_losses.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    stat_plots = [
        ("PredMean", "Pred Mean"),
        ("PredStd", "Pred Std"),
        ("PredMin", "Pred Min"),
        ("PredMax", "Pred Max"),
        ("DeadBlack", "Dead Black"),
        ("DeadWhite", "Dead White"),
    ]
    for ax, (key, title) in zip(axes.flat, stat_plots):
        if key is None:
            ax.set_visible(False)
            continue
        for idx, (data, _) in enumerate(data_dicts):
            x, y = _xy(data, key)
            if x:
                ax.plot(x, y, "o-", c=colors[idx], ms=3, lw=1.2, label=names[idx])
        ax.set_title(title)
        maybe_legend(ax, fontsize=6)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}prediction_stats.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (key, title) in zip(axes, [("Lab", "Lab"), ("LCh", "LCh"), ("Percep", "Perceptual")]):
        for idx, (data, _) in enumerate(data_dicts):
            x, y = _xy(data, key)
            if x:
                ax.plot(x, y, "o-", c=colors[idx], ms=3, lw=1.2, label=names[idx])
        ax.set_title(title)
        maybe_legend(ax, fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}color_losses.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (key, title) in zip(axes, [("G_Adv", "G Adv"), ("D", "D Loss")]):
        for idx, (data, _) in enumerate(data_dicts):
            x, y = _xy(data, key)
            if x:
                ax.plot(x, y, "o-", c=colors[idx], ms=3, lw=1.2, label=names[idx])
        ax.set_title(title)
        maybe_legend(ax, fontsize=7)
        ax.grid(alpha=0.3)
    for idx, (data, _) in enumerate(data_dicts):
        rx, rv = _xy(data, "D_real")
        fx, fv = _xy(data, "D_fake")
        if rx:
            axes[2].plot(rx, rv, "o-", c=colors[idx], ms=3, lw=1.2, label=f"{names[idx]} real")
        if fx:
            axes[2].plot(fx, fv, "s--", c=colors[idx], ms=3, lw=1.2, alpha=0.7, label=f"{names[idx]} fake")
    axes[2].axhline(0, color="gray", ls=":", alpha=0.5)
    axes[2].set_title("D(real) vs D(fake)")
    maybe_legend(axes[2], fontsize=6)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}gan.png", dpi=150)
    plt.close(fig)

    if data_dicts and data_dicts[0][1].get("step"):
        pb = data_dicts[0][1]
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes[0, 0].plot(pb["step"], pb["G"], "-", c=colors[0], lw=0.5, alpha=0.7)
        axes[0, 0].set_title("G per Batch")
        axes[0, 0].grid(alpha=0.3)
        axes[0, 1].plot(pb["step"], pb["D"], "-", c=colors[0], lw=0.5, alpha=0.7)
        axes[0, 1].set_title("D per Batch")
        axes[0, 1].grid(alpha=0.3)
        window = max(1, len(pb["step"]) // 200)
        smooth_g = np.convolve(pb["G"], np.ones(window) / window, mode="valid")
        axes[1, 0].plot(pb["step"][window - 1:], smooth_g, "-", c=colors[0], lw=1.0)
        axes[1, 0].set_title(f"G (MA-{window})")
        axes[1, 0].grid(alpha=0.3)
        axes[1, 1].plot(pb["step"], pb["D_real"], "-", c="green", lw=0.5, alpha=0.6, label="D(real)")
        axes[1, 1].plot(pb["step"], pb["D_fake"], "-", c="red", lw=0.5, alpha=0.6, label="D(fake)")
        axes[1, 1].axhline(0, color="gray", ls=":", alpha=0.5)
        axes[1, 1].set_title("D(real) vs D(fake)")
        maybe_legend(axes[1, 1], fontsize=7)
        axes[1, 1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}batch.png", dpi=150)
        plt.close(fig)


def stitch_logs(data_dicts, names):
    merged_data = defaultdict(list)
    merged_pb = defaultdict(list)
    epoch_offset = 0
    step_offset = 0

    for data, pb in data_dicts:
        epochs = data.get("epoch", [])
        local_offset = epoch_offset
        for key, values in data.items():
            if key.startswith("_"):
                continue
            if key == "epoch":
                merged_data[key].extend([e + local_offset for e in values])
            else:
                merged_data[key].extend(values)

        if epochs:
            epoch_offset += max(epochs)

        if pb.get("epoch"):
            for key, values in pb.items():
                if key == "epoch":
                    merged_pb[key].extend([e + local_offset for e in values])
                elif key == "step":
                    merged_pb[key].extend([s + step_offset for s in values])
                else:
                    merged_pb[key].extend(values)
            if pb.get("step"):
                step_offset = max(merged_pb["step"])

    merged_name = names[0].rsplit("_", 1)[0] + f"_stitched_{len(names)}logs"
    return [(merged_data, merged_pb)], [merged_name]


def main():
    parser = argparse.ArgumentParser(description="Train loss visualization and text summary")
    parser.add_argument("logs", nargs="+", help="One or more training log files")
    parser.add_argument("--output", "-o", default="output/plots", help="Output directory")
    parser.add_argument("--prefix", "-p", default="", help="Filename prefix")
    parser.add_argument("--stitch", action="store_true", help="Stitch multiple logs into one continuous run")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    data_dicts, names = [], []
    for log_path in args.logs:
        path = Path(log_path)
        if not path.exists():
            print(f"[skip] {log_path}")
            continue
        print(f"[parse] {path.name} ...", end=" ", flush=True)
        data, per_batch = parse_log(path)
        epoch_count = len(data.get("epoch", []))
        batch_count = len(per_batch.get("batch", []))
        if epoch_count == 0:
            print("FAIL (no epoch data)")
            continue
        print(f"OK ({epoch_count} epochs, {batch_count} batches)")
        data_dicts.append((data, per_batch))
        names.append(path.stem[:35])

    if not data_dicts:
        print("No valid data. Exit.")
        return

    if args.stitch and len(data_dicts) > 1:
        print(f"\n[stitch] merging {len(data_dicts)} logs ...")
        data_dicts, names = stitch_logs(data_dicts, names)
        print(f"  merged into {len(data_dicts[0][0]['epoch'])} epochs")

    summary_path = out / f"{args.prefix}summary.txt"
    print(f"\n[summary] -> {summary_path}")
    text_summary(data_dicts, names, summary_path)

    print(f"[plots] -> {out.resolve()}/")
    plot_all(data_dicts, names, out, args.prefix)

    print("\nDone. Files:")
    for file_path in sorted(out.glob(f"{args.prefix}*")):
        print(f"  {file_path.name}")


if __name__ == "__main__":
    main()
