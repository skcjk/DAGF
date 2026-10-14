import os, random, shutil, argparse
from pathlib import Path

OUT_DIR = 'data/underwater'
TEST_RATIO = 0.03
SEED = 42


def main():
    parser = argparse.ArgumentParser(description='预分割水下图像数据集')
    parser.add_argument('--test-ratio', type=float, default=TEST_RATIO)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--out', type=str, default=OUT_DIR)
    parser.add_argument('--copy', action='store_true', help='复制文件（默认使用符号链接）')
    parser.add_argument('--pair', nargs=2, action='append', required=True, metavar=('RAW_DIR', 'GT_DIR'), help='Paired source directories; may be repeated')
    args = parser.parse_args()
    if not 0 < args.test_ratio < 1:
        parser.error('--test-ratio must be between 0 and 1')

    out = Path(args.out)
    copy_fn = shutil.copy2 if args.copy else os.symlink


    all_pairs = []
    for raw_dir, gt_dir in args.pair:
        raw_dir = str(Path(raw_dir).resolve())
        gt_dir = str(Path(gt_dir).resolve())
        if not os.path.isdir(raw_dir) or not os.path.isdir(gt_dir):
            parser.error(f'Source directory not found: {raw_dir} or {gt_dir}')
        imgs = sorted(f for f in os.listdir(raw_dir) if f.lower().endswith(('.png','.jpg','.jpeg')))
        for fname in imgs:
            rp = os.path.join(raw_dir, fname)
            gp = os.path.join(gt_dir, fname)
            if os.path.isfile(gp):
                all_pairs.append((rp, gp, fname))
    if len(all_pairs) < 2:
        parser.error('At least two matching image pairs are required')
    if out.exists() and any(out.iterdir()):
        parser.error('Output directory must be empty to avoid mixing dataset splits')
    print(f'总图像对: {len(all_pairs)}')


    random.seed(args.seed)
    random.shuffle(all_pairs)
    split = min(len(all_pairs) - 1, max(1, int(len(all_pairs) * (1 - args.test_ratio))))
    train_pairs = all_pairs[:split]
    test_pairs = all_pairs[split:]


    for split_name, pairs in [('train', train_pairs), ('test', test_pairs)]:
        raw_out = out / split_name / 'raw'
        gt_out = out / split_name / 'gt'
        raw_out.mkdir(parents=True, exist_ok=True)
        gt_out.mkdir(parents=True, exist_ok=True)
        print(f'\n[{split_name}] {len(pairs)} 对 -> {raw_out}')

        for rp, gp, fname in pairs:

            dst_name = fname
            raw_dst = raw_out / dst_name
            if raw_dst.exists():
                src_tag = os.path.basename(os.path.dirname(rp))[:8]
                stem, ext = os.path.splitext(fname)
                dst_name = f'{stem}_{src_tag}{ext}'
                raw_dst = raw_out / dst_name

                counter = 1
                while raw_dst.exists():
                    dst_name = f'{stem}_{src_tag}_{counter}{ext}'
                    raw_dst = raw_out / dst_name
                    counter += 1
            copy_fn(rp, raw_dst)
            copy_fn(gp, gt_out / dst_name)

    print(f'\n完成! 输出: {out}')
    print(f'  train: {len(train_pairs)} 对')
    print(f'  test:  {len(test_pairs)} 对')


if __name__ == '__main__':
    main()
