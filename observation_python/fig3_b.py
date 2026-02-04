"""
Figure 3b: High-Frequency Energy vs. Quantized Model Performance

This script creates two scatter plots with trendlines:
1. DMI (DeepInversion) - X: HF energy, Y: Quantized model accuracy
2. SMI (Sparse Model Inversion) - X: HF energy, Y: Quantized model accuracy

Each plot contains 50 scatter points (seeds 1-50) with a linear trendline.
"""

import os
import sys
import random
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import *
from quantization import *
from utils.data_utils import find_non_zero_patches

# -------------------------
# Environment / Reproducibility
# -------------------------
def set_seed(seed: int):
    sys.setrecursionlimit(100000)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


# -------------------------
# File utilities
# -------------------------
def list_image_files(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if not folder.exists():
        return []
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in exts]


# -------------------------
# Standard ImageNet-style preprocessing (resize shorter side -> center crop)
# -------------------------
def resize_shorter_side(pil_img: Image.Image, shorter_side: int) -> Image.Image:
    w, h = pil_img.size
    if min(w, h) == shorter_side:
        return pil_img

    if w < h:
        new_w = shorter_side
        new_h = int(round(h * (shorter_side / w)))
    else:
        new_h = shorter_side
        new_w = int(round(w * (shorter_side / h)))

    return pil_img.resize((new_w, new_h), Image.BICUBIC)


def center_crop(pil_img: Image.Image, size: int) -> Image.Image:
    w, h = pil_img.size
    if w < size or h < size:
        return pil_img.resize((size, size), Image.BICUBIC)

    left = (w - size) // 2
    top = (h - size) // 2
    return pil_img.crop((left, top, left + size, top + size))


def load_image_standard(path: str | Path, size: int = 224, resize_short: int = 256) -> np.ndarray:
    """
    Standard: resize shorter side to 256, then center crop to 224.
    Returns float32 in [0,1], shape (H,W,3).
    """
    img = Image.open(path).convert("RGB")
    img = resize_shorter_side(img, resize_short)
    img = center_crop(img, size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def load_all_images_from_folder(folder: Path, size: int = 224, resize_short: int = 256) -> List[np.ndarray]:
    """Load all images from a folder."""
    paths = list_image_files(folder)
    return [load_image_standard(p, size=size, resize_short=resize_short) for p in paths]


# -------------------------
# Frequency analysis
# -------------------------
def radial_profile_mean(spec2d: np.ndarray, center_xy: Tuple[float, float]) -> np.ndarray:
    """
    spec2d: (H,W) nonnegative
    center_xy: (cx, cy) in pixel coords
    Returns radial mean per integer radius bin r=0..rmax.
    """
    h, w = spec2d.shape
    y, x = np.indices((h, w))
    cx, cy = center_xy
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)

    max_r = int(r.max()) + 1
    radial_sum = np.bincount(r.ravel(), weights=spec2d.ravel(), minlength=max_r)
    radial_cnt = np.bincount(r.ravel(), minlength=max_r)
    return radial_sum / np.maximum(radial_cnt, 1)


def compute_radial_spectrum(
    image: np.ndarray,
    *,
    kind: str = "amplitude",
    fft_norm: str | None = "ortho",
    normalize: str = "none",
    eps: float = 1e-12,
) -> np.ndarray:
    """
    image: float32 [0,1], (H,W,3)
    kind="amplitude": radial mean of |F|
    kind="power":     radial mean of |F|^2
    normalize="sum1": prof /= prof.sum()
    normalize="none": absolute profile
    """
    image = image.astype(np.float32, copy=False)
    image = image - image.mean(axis=(0, 1), keepdims=True)

    h, w, c = image.shape
    spec_sum = np.zeros((h, w), dtype=np.float64)

    for ch in range(c):
        fft = np.fft.fftshift(np.fft.fft2(image[:, :, ch], norm=fft_norm))
        mag = np.abs(fft)

        if kind == "amplitude":
            spec = mag
        elif kind == "power":
            spec = mag ** 2
        else:
            raise ValueError(f"Unknown kind: {kind}")

        spec_sum += spec

    center = (w / 2.0, h / 2.0)
    prof = radial_profile_mean(spec_sum, center)

    if normalize == "sum1":
        prof = prof / (prof.sum() + eps)

    return prof


def compute_hf_band_ratio(
    image: np.ndarray,
    *,
    kind: str = "amplitude",
    fft_norm: str | None = "ortho",
    hf_start: int = 80,
    size: int = 224,
    eps: float = 1e-12,
) -> float:
    """Compute high-frequency band ratio for a single image."""
    prof = compute_radial_spectrum(
        image,
        kind=kind,
        fft_norm=fft_norm,
        normalize="none",
        eps=eps,
    )

    max_radius = size // 2
    prof = prof[: max_radius + 1]

    denom = prof.sum() + eps
    hf = prof[hf_start : max_radius + 1].sum()

    return float(hf / denom)


def compute_mean_hf_ratio_for_folder(
    folder: Path,
    *,
    kind: str = "amplitude",
    fft_norm: str | None = "ortho",
    hf_start: int = 80,
    size: int = 224,
) -> float:
    """Compute mean HF ratio for all images in a folder."""
    images = load_all_images_from_folder(folder, size=size, resize_short=256)
    if not images:
        return 0.0
    
    ratios = [
        compute_hf_band_ratio(img, kind=kind, fft_norm=fft_norm, hf_start=hf_start, size=size)
        for img in images
    ]
    return float(np.mean(ratios))


# -------------------------
# Quantization model setup
# -------------------------
class Config:
    def __init__(self, w_bit, a_bit):
        self.weight_bit = w_bit
        self.activation_bit = a_bit


def get_teacher(name):
    teacher_name = {
        'deit_tiny_16_imagenet': 'deit_tiny_patch16_224',
        'deit_base_16_imagenet': 'deit_base_patch16_224',
    }
    if name.split("_")[-1] == 'imagenet':
        teacher = build_model(teacher_name[name], Pretrained=True)
    else:
        raise NotImplementedError
    return teacher


def get_student(name):
    model_zoo = {
        'deit_tiny_16_imagenet': deit_tiny_patch16_224,
        'deit_base_16_imagenet': deit_base_patch16_224,
    }
    print('Model: %s' % model_zoo[name].__name__)
    return model_zoo[name]


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def validate(val_loader, model, criterion, device):
    """Validate the quantized model and return accuracy."""
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    for data, target in val_loader:
        target = target.to(device)
        data = data.to(device)

        with torch.no_grad():
            output = model(data)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

    return top1.avg, top5.avg


def evaluate_quantized_model(
    datapool_path: Path,
    model_name: str,
    dataset_path: str,
    device: torch.device,
    calib_batchsize: int = 200,
    w_bit: int = 4,
    a_bit: int = 8,
) -> Tuple[float, float]:
    """
    Evaluate the quantized model using calibration images from the given datapool.
    Returns (top1_accuracy, top5_accuracy).
    """
    cfg = Config(w_bit, a_bit)
    
    # Build student model
    model = get_student(model_name)(pretrained=True, cfg=cfg)
    model = model.to(device)
    model.eval()
    
    # Build dataloader
    _, val_loader, num_classes, train_transform, _, normalizer = build_dataset(
        model_name.split("_")[0], model_name.split("_")[-1], calib_batchsize,
        train_aug=False, keep_zero=True, train_inverse=True, dataset_path=dataset_path
    )
    
    patch_size = 16 if '16' in model_name else 32
    patch_num = 197 if patch_size == 16 else 50
    
    # Create DataPool and load images
    from synthesis._utils import ImagePool  # ImagePool is in _utils module
    
    data_pool = ImagePool(root=str(datapool_path))
    dst = data_pool.get_dataset(transform=train_transform)
    
    if len(dst) == 0:
        print(f"Warning: No images found in {datapool_path}")
        return 0.0, 0.0
    
    calib_loader = DataLoader(
        dst,
        batch_size=calib_batchsize,
        shuffle=False, num_workers=0, pin_memory=True,
    )
    
    # Calibration
    model.model_unfreeze()
    
    with torch.no_grad():
        for calibrate_data in calib_loader:
            calibrate_data = calibrate_data.to(device)
            
            current_abs_index = torch.arange(patch_num, device=calibrate_data.device).repeat(calibrate_data.shape[0], 1)
            next_relative_index = torch.cat(
                [
                    torch.zeros(calibrate_data.shape[0], 1, dtype=torch.long).to(calibrate_data.device),
                    find_non_zero_patches(images=calibrate_data, patch_size=patch_size)
                ], dim=1
            )
            
            _ = model(
                calibrate_data,
                current_abs_index=current_abs_index,
                next_relative_index=next_relative_index
            )
    
    model.model_quant()
    model.model_freeze()
    
    # Validation
    criterion = nn.CrossEntropyLoss().to(device)
    top1, top5 = validate(val_loader, model, criterion, device)
    
    return top1, top5


# -------------------------
# Get folder paths for DMI/SMI seeds
# -------------------------
def get_dmi_folder(base_dir: Path, seed: int) -> Path:
    """Get the DMI folder path for a given seed."""
    return base_dir / f"DMI-4000-{seed}-32-W4A8"


def get_smi_folder(base_dir: Path, seed: int) -> Path:
    """Get the SMI folder path for a given seed."""
    return base_dir / f"SMI-4000-50-100-200-300-0.3-0.3-0.3-0.3-{seed}-32-W4A8"


# -------------------------
# Main plotting function
# -------------------------
def plot_hf_vs_accuracy_scatter(
    hf_values: List[float],
    accuracy_values: List[float],
    title: str,
    output_path: str,
    xlabel: str = "High-Frequency Energy (HF Ratio)",
    ylabel: str = "Quantized Model Accuracy (%)",
    color: str = "blue",
    marker: str = "o",
):
    """
    Create a scatter plot with a linear trendline.
    """
    hf = np.array(hf_values)
    acc = np.array(accuracy_values)
    
    # Linear regression for trendline
    slope, intercept, r_value, p_value, std_err = stats.linregress(hf, acc)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    
    # Scatter points
    plt.scatter(hf, acc, c=color, marker=marker, s=60, alpha=0.7, edgecolors='black', linewidths=0.5)
    
    # Trendline
    x_line = np.linspace(hf.min(), hf.max(), 100)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, color='red', linestyle='--', linewidth=2, 
             label=f'Trendline (R²={r_value**2:.3f})')
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    print(f"  Correlation: R²={r_value**2:.4f}, slope={slope:.4f}, p-value={p_value:.4e}")


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Figure 3b: HF Energy vs Quantized Model Performance")
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
    parser.add_argument("--dataset", type=str, default="/home/mjatwk/data/imagenet", help="Path to ImageNet dataset")
    parser.add_argument("--output_dir", type=str, default="./observation/00_Figures", help="Output directory for figures")
    parser.add_argument("--hf_start", type=int, default=80, help="HF band start radius")
    parser.add_argument("--size", type=int, default=224, help="Image size")
    parser.add_argument("--seed_start", type=int, default=1, help="Start seed (inclusive)")
    parser.add_argument("--seed_end", type=int, default=50, help="End seed (inclusive)")
    parser.add_argument("--w_bit", type=int, default=4, help="Weight bit precision")
    parser.add_argument("--a_bit", type=int, default=8, help="Activation bit precision")
    parser.add_argument("--skip_eval", action="store_true", help="Skip model evaluation (use cached results)")
    parser.add_argument("--cache_file", type=str, default="./observation/00_Figures/fig3b_cache.json", help="Cache file path")
    args = parser.parse_args()
    
    # Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    set_seed(0)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot style
    plt.style.use("ggplot")
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.rcParams["axes.grid"] = True
    
    # Data directories
    base_dir = Path("./dataset/deit_base_16_imagenet")
    dmi_base = base_dir / "DMI_Seed"
    smi_base = base_dir / "SMI_Seed"
    
    model_name = "deit_base_16_imagenet"
    seeds = list(range(args.seed_start, args.seed_end + 1))
    
    # Cache handling
    cache = {}
    if os.path.exists(args.cache_file):
        with open(args.cache_file, 'r') as f:
            cache = json.load(f)
    
    # -------------------------
    # Collect data for DMI
    # -------------------------
    print("\n" + "="*60)
    print("Processing DMI...")
    print("="*60)
    
    dmi_hf_values = []
    dmi_acc_values = []
    
    for seed in tqdm(seeds, desc="DMI Seeds"):
        folder = get_dmi_folder(dmi_base, seed)
        cache_key = f"DMI_{seed}"
        
        if cache_key in cache and args.skip_eval:
            hf_ratio = cache[cache_key]["hf_ratio"]
            top1_acc = cache[cache_key]["top1_acc"]
        else:
            # Compute HF ratio
            hf_ratio = compute_mean_hf_ratio_for_folder(
                folder, kind="amplitude", fft_norm="ortho",
                hf_start=args.hf_start, size=args.size
            )
            
            # Evaluate quantized model
            if not args.skip_eval:
                top1_acc, _ = evaluate_quantized_model(
                    datapool_path=folder,
                    model_name=model_name,
                    dataset_path=args.dataset,
                    device=device,
                    w_bit=args.w_bit,
                    a_bit=args.a_bit,
                )
            else:
                top1_acc = cache.get(cache_key, {}).get("top1_acc", 0.0)
            
            # Cache results
            cache[cache_key] = {"hf_ratio": hf_ratio, "top1_acc": top1_acc}
        
        dmi_hf_values.append(hf_ratio)
        dmi_acc_values.append(top1_acc)
        print(f"  Seed {seed}: HF={hf_ratio:.6f}, Acc={top1_acc:.2f}%")
    
    # -------------------------
    # Collect data for SMI
    # -------------------------
    print("\n" + "="*60)
    print("Processing SMI...")
    print("="*60)
    
    smi_hf_values = []
    smi_acc_values = []
    
    for seed in tqdm(seeds, desc="SMI Seeds"):
        folder = get_smi_folder(smi_base, seed)
        cache_key = f"SMI_{seed}"
        
        if cache_key in cache and args.skip_eval:
            hf_ratio = cache[cache_key]["hf_ratio"]
            top1_acc = cache[cache_key]["top1_acc"]
        else:
            # Compute HF ratio
            hf_ratio = compute_mean_hf_ratio_for_folder(
                folder, kind="amplitude", fft_norm="ortho",
                hf_start=args.hf_start, size=args.size
            )
            
            # Evaluate quantized model
            if not args.skip_eval:
                top1_acc, _ = evaluate_quantized_model(
                    datapool_path=folder,
                    model_name=model_name,
                    dataset_path=args.dataset,
                    device=device,
                    w_bit=args.w_bit,
                    a_bit=args.a_bit,
                )
            else:
                top1_acc = cache.get(cache_key, {}).get("top1_acc", 0.0)
            
            # Cache results
            cache[cache_key] = {"hf_ratio": hf_ratio, "top1_acc": top1_acc}
        
        smi_hf_values.append(hf_ratio)
        smi_acc_values.append(top1_acc)
        print(f"  Seed {seed}: HF={hf_ratio:.6f}, Acc={top1_acc:.2f}%")
    
    # Save cache
    os.makedirs(os.path.dirname(args.cache_file), exist_ok=True)
    with open(args.cache_file, 'w') as f:
        json.dump(cache, f, indent=2)
    print(f"\nCache saved to: {args.cache_file}")
    
    # -------------------------
    # Create plots
    # -------------------------
    print("\n" + "="*60)
    print("Creating plots...")
    print("="*60)
    
    # Plot 1: DMI
    plot_hf_vs_accuracy_scatter(
        hf_values=dmi_hf_values,
        accuracy_values=dmi_acc_values,
        title="DMI (DeepInversion): HF Energy vs. Quantized Model Accuracy",
        output_path=os.path.join(args.output_dir, "fig3b_DMI.png"),
        color="blue",
        marker="o",
    )
    
    # Plot 2: SMI
    plot_hf_vs_accuracy_scatter(
        hf_values=smi_hf_values,
        accuracy_values=smi_acc_values,
        title="SMI (Sparse Model Inversion): HF Energy vs. Quantized Model Accuracy",
        output_path=os.path.join(args.output_dir, "fig3b_SMI.png"),
        color="green",
        marker="s",
    )
    
    print("\n" + "="*60)
    print("Done!")
    print(f"Outputs saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
