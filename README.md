# DAGF

Depth-Aware Gated Fusion of Foundation-Model Features for Underwater Image Enhancement.

## Installation

Environment: Ubuntu 22.04, Python 3.11, PyTorch 2.4.1, torchvision 0.19.1, CUDA 12.1.

```bash
conda create -n dagf python=3.11 -y
conda activate dagf
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirement.txt --no-build-isolation
```

Run all commands from the repository root. The first run downloads [Depth Anything 3](https://huggingface.co/depth-anything/da3mono-large) weights; training also requires VGG-19 ImageNet weights.

## Pretrained Weights

The pretrained weights will be released upon acceptance of the paper.

## Dataset

Place paired images with identical filenames in the following folders:

```text
data/underwater/
├── train/
│   ├── raw/
│   └── gt/
└── test/
    ├── raw/
    └── gt/
```

Supported formats: PNG and JPEG. The `test` split is used for validation and checkpoint selection.

To split a source dataset:

```bash
python scripts/split_dataset.py \
  --pair data/source/raw data/source/gt \
  --out data/underwater \
  --test-ratio 0.03 --copy
```

Repeat `--pair` for multiple sources. The output directory must be empty. Omit `--copy` to use symbolic links.

## Training

### Stage 1: DANet

DANet uses the `unet` prefix in script names, arguments, and checkpoint files.

```bash
python core/train_unet.py \
  --scratch \
  --data-dir data/underwater \
  --checkpoint-dir checkpoints/unet_large \
  --epochs 100 --batch-size 4 \
  --train-res-mode multiscale \
  --train-sizes 252,378,504,756
```

### Stage 2: DPT fusion

Stage 2 uses the frozen DANet branch trained in Stage 1.

```bash
python core/train_dpt.py \
  --scratch \
  --checkpoint-unet checkpoints/unet_large/unet_branch_best.pth \
  --data-dir data/underwater \
  --checkpoint-dir checkpoints/dpt \
  --epochs 100 --batch-size 4 \
  --train-res-mode multiscale \
  --train-sizes 252,504,756
```

Both stages save `*_best.pth` when an epoch achieves a **lower mean test loss**. Selection starts with the first finite epoch loss and resets on each training run. An empty test split or non-finite loss cannot produce a best checkpoint. Numbered checkpoints are also saved every 10 epochs and at the final epoch.

Training supports `fixed`, `multiscale`, and `native` resolution modes. Training sizes must be multiples of 14. The default learning rate is `1e-4`.

The discriminator is trained by default, with generator adversarial weight `--lambda-adv 0.0`. Set a positive weight to enable generator adversarial loss, or use `--no-train-discriminator` to disable the discriminator.

To continue from weights, omit `--scratch` and specify `--checkpoint-g`, `--checkpoint-h`, and optionally `--checkpoint-d` and `--start-epoch`. Optimizer states are not restored.

## Inference

### DANet

```bash
python core/demo.py \
  --model unet \
  --weights checkpoints/unet_large \
  --input data/demo \
  --output results/unet
```

### DPT fusion

```bash
python core/demo.py \
  --model dpt \
  --weights checkpoints/dpt \
  --unet-weights checkpoints/unet_large/unet_branch_best.pth \
  --input data/demo \
  --output results/dpt
```

Checkpoint directories must contain matching branch/head files:

| Model | Files |
| --- | --- |
| DANet | `unet_branch_best.pth`, `unet_head_best.pth` |
| DPT | `dpt_branch_best.pth`, `dpt_head_best.pth` |

Names without `_best` are also supported. To use numbered checkpoints, copy the selected branch/head pair to a separate directory and rename them accordingly. DPT inference also requires the corresponding DANet branch.

`--input` accepts an RGB image or a folder of PNG/JPEG images. Outputs retain the original dimensions; JPEG inputs are saved as PNG. Use unique filename stems to avoid collisions.

For large images, add `--tile-size 504 --tile-overlap 64` for tiled inference, or `--resize` to fit within `1008 × 1008` before enhancement.

## Evaluation

```bash
python core/eval.py \
  --pred-dir results/dpt \
  --gt-dir data/evaluation/gt \
  --metrics PSNR SSIM UCIQE UIQM \
  --model-name DAGF --dataset-name evaluation
```

Results are saved to `evaluation_results/`. Predictions are resized to reference dimensions for PSNR/SSIM when needed.

To include LPIPS and NIQE, install `pyiqa` and add them to `--metrics`:

```bash
python -m pip install pyiqa==0.1.15.post2
```

## Training logs

Logs are saved to `logs/`. To plot losses:

```bash
python scripts/plot_losses.py logs/dagf_train_RUN.log --output output/plots
```

## Third-party code

Depth Anything 3 is included in `depth_anything_3/`. See its [license](depth_anything_3/LICENSE).
