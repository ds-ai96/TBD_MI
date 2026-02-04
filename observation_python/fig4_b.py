"""
Figure 4b: Saliency Weight Distribution - Cumulative Patch Count by Saliency Weight Bins

This script creates one figure with 3 line plots:
1. ImageNet (real images) - 1,000 images (1 per class)
2. DeepInversion (DMI) - 1,000 synthetic images (1 per class)
3. Sparse Model Inversion (SMI) - 1,000 synthetic images (1 per class)

X-axis: Saliency weight bins (normalized 0-1 range)
Y-axis: Cumulative count of patches per bin (or average ratio per image)
"""

import os
import sys
import random
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import *


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
# Standard ImageNet-style preprocessing
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


def load_image_tensor(path: str | Path, size: int = 224, resize_short: int = 256) -> torch.Tensor:
    """
    Load image and return as torch tensor [1, 3, H, W] normalized to [0, 1].
    """
    img = Image.open(path).convert("RGB")
    img = resize_shorter_side(img, resize_short)
    img = center_crop(img, size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return tensor


# -------------------------
# Saliency computation
# -------------------------
def compute_saliency_map(
    image_tensor: torch.Tensor,
    teacher_model: torch.nn.Module,
    target_class: int,
    device: torch.device,
    patch_size: int = 16,
) -> np.ndarray:
    """
    Compute gradient-based saliency map for an image.
    Returns a 2D saliency map normalized to [0, 1].
    
    image_tensor: [1, 3, H, W]
    Returns: saliency map [H, W] normalized to [0, 1]
    """
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad_(True)
    
    # ViT model forward pass
    patch_num = 197  # 14*14 + 1 (CLS token)
    current_abs_index = torch.arange(patch_num, device=device).repeat(1, 1)
    next_relative_index = torch.ones(1, patch_num, dtype=torch.long, device=device)
    
    try:
        logits, _, _ = teacher_model(image_tensor, current_abs_index, next_relative_index)
    except:
        # Fallback for models without extra arguments
        logits = teacher_model(image_tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
    
    # Get score for target class
    if target_class == -1:
        # If no target class provided, use the predicted class
        target_class = logits.argmax(dim=1).item()
    
    score = logits[0, target_class]
    
    # Compute gradient
    grad = torch.autograd.grad(score, image_tensor, create_graph=False, retain_graph=False)[0]
    
    # Saliency map: absolute max across channels
    saliency = grad.abs().amax(dim=1, keepdim=True)  # [1, 1, H, W]
    saliency = saliency.squeeze().cpu().numpy()  # [H, W]
    
    # Normalize to [0, 1]
    sal_min = saliency.min()
    sal_max = saliency.max()
    if sal_max - sal_min > 1e-8:
        saliency = (saliency - sal_min) / (sal_max - sal_min)
    else:
        saliency = np.zeros_like(saliency)
    
    return saliency


def compute_patch_saliency_weights(
    saliency_map: np.ndarray,
    patch_size: int = 16,
) -> np.ndarray:
    """
    Compute average saliency weight for each patch.
    
    saliency_map: [H, W] normalized saliency values
    Returns: [num_patches_h, num_patches_w] array of patch saliency weights
    """
    H, W = saliency_map.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    
    patch_weights = np.zeros((num_patches_h, num_patches_w))
    
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch = saliency_map[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patch_weights[i, j] = patch.mean()
    
    return patch_weights


def compute_saliency_histogram(
    images_folder: Path,
    teacher_model: torch.nn.Module,
    device: torch.device,
    num_bins: int = 20,
    max_images: int = 1000,
    patch_size: int = 16,
    desc: str = "Processing",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute histogram of saliency weights across all images in a folder.
    
    Returns:
        bin_edges: array of bin edges (len = num_bins + 1)
        avg_ratios: average ratio of patches falling in each bin per image (len = num_bins)
    """
    image_paths = list_image_files(images_folder)[:max_images]
    
    if not image_paths:
        print(f"Warning: No images found in {images_folder}")
        return np.linspace(0, 1, num_bins + 1), np.zeros(num_bins)
    
    # Collect all patch saliency weights
    all_weights = []
    per_image_histograms = []
    
    bin_edges = np.linspace(0, 1, num_bins + 1)
    
    for img_path in tqdm(image_paths, desc=desc):
        try:
            image_tensor = load_image_tensor(img_path)
            saliency_map = compute_saliency_map(
                image_tensor, teacher_model, target_class=-1, device=device, patch_size=patch_size
            )
            patch_weights = compute_patch_saliency_weights(saliency_map, patch_size)
            
            # Flatten and normalize to [0, 1] per image
            weights_flat = patch_weights.flatten()
            w_min, w_max = weights_flat.min(), weights_flat.max()
            if w_max - w_min > 1e-8:
                weights_norm = (weights_flat - w_min) / (w_max - w_min)
            else:
                weights_norm = np.zeros_like(weights_flat)
            
            all_weights.extend(weights_norm.tolist())
            
            # Per-image histogram (proportion)
            counts, _ = np.histogram(weights_norm, bins=bin_edges)
            per_image_histograms.append(counts / len(weights_norm))
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Average ratio per bin across all images
    if per_image_histograms:
        avg_ratios = np.mean(per_image_histograms, axis=0)
    else:
        avg_ratios = np.zeros(num_bins)
    
    return bin_edges, avg_ratios


def compute_saliency_histogram_for_imagenet(
    imagenet_root: Path,
    teacher_model: torch.nn.Module,
    device: torch.device,
    num_bins: int = 20,
    num_classes: int = 1000,
    n_samples_per_class: int = 1,
    patch_size: int = 16,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute saliency histogram for ImageNet images (1 per class).
    """
    import torchvision
    
    rng = random.Random(seed)
    
    try:
        dataset = torchvision.datasets.ImageNet(root=str(imagenet_root), split='val', transform=None)
    except Exception as e:
        print(f"Error loading ImageNet: {e}")
        return np.linspace(0, 1, num_bins + 1), np.zeros(num_bins)
    
    # Group images by class
    class_to_paths: List[List[str]] = [[] for _ in range(num_classes)]
    for img_path, class_idx in dataset.samples:
        if 0 <= class_idx < num_classes:
            class_to_paths[class_idx].append(img_path)
    
    # Sample one image per class
    selected_paths = []
    for class_idx in range(num_classes):
        paths = class_to_paths[class_idx]
        if paths:
            selected_paths.append(rng.choice(paths))
    
    bin_edges = np.linspace(0, 1, num_bins + 1)
    per_image_histograms = []
    
    for img_path in tqdm(selected_paths, desc="ImageNet"):
        try:
            image_tensor = load_image_tensor(img_path)
            saliency_map = compute_saliency_map(
                image_tensor, teacher_model, target_class=-1, device=device, patch_size=patch_size
            )
            patch_weights = compute_patch_saliency_weights(saliency_map, patch_size)
            
            # Flatten and normalize to [0, 1] per image
            weights_flat = patch_weights.flatten()
            w_min, w_max = weights_flat.min(), weights_flat.max()
            if w_max - w_min > 1e-8:
                weights_norm = (weights_flat - w_min) / (w_max - w_min)
            else:
                weights_norm = np.zeros_like(weights_flat)
            
            # Per-image histogram (proportion)
            counts, _ = np.histogram(weights_norm, bins=bin_edges)
            per_image_histograms.append(counts / len(weights_norm))
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Average ratio per bin across all images
    if per_image_histograms:
        avg_ratios = np.mean(per_image_histograms, axis=0)
    else:
        avg_ratios = np.zeros(num_bins)
    
    return bin_edges, avg_ratios


# -------------------------
# Model loading
# -------------------------
def get_teacher(name: str, device: torch.device):
    """Load the teacher model."""
    teacher_name = {
        'deit_tiny_16_imagenet': 'deit_tiny_patch16_224',
        'deit_base_16_imagenet': 'deit_base_patch16_224',
    }
    if name.split("_")[-1] == 'imagenet':
        teacher = build_model(teacher_name[name], Pretrained=True)
    else:
        raise NotImplementedError
    
    teacher = teacher.to(device)
    teacher.eval()
    return teacher


# -------------------------
# Plotting
# -------------------------
def plot_saliency_distribution(
    bin_edges: np.ndarray,
    imagenet_ratios: np.ndarray,
    dmi_ratios: np.ndarray,
    smi_ratios: np.ndarray,
    output_path: str,
):
    """
    Create a line plot comparing saliency weight distributions.
    """
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    plt.figure(figsize=(10, 6))
    
    # Plot lines
    plt.plot(bin_centers, imagenet_ratios, 'o-', color='blue', linewidth=2, 
             markersize=6, label='ImageNet (Real)')
    plt.plot(bin_centers, dmi_ratios, 's-', color='red', linewidth=2, 
             markersize=6, label='DeepInversion (DMI)')
    plt.plot(bin_centers, smi_ratios, '^-', color='green', linewidth=2, 
             markersize=6, label='Sparse Model Inversion (SMI)')
    
    plt.xlabel('Saliency Weight (Normalized 0-1)', fontsize=12)
    plt.ylabel('Average Ratio of Patches per Image', fontsize=12)
    plt.title('Saliency Weight Distribution: ImageNet vs. Synthetic Images', fontsize=14)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, None)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Figure 4b: Saliency Weight Distribution")
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
    parser.add_argument("--imagenet_root", type=str, default="/home/mjatwk/data/imagenet", 
                        help="Path to ImageNet dataset")
    parser.add_argument("--dmi_folder", type=str, default="./observation/08_Confidence/DMI",
                        help="Path to DMI images folder")
    parser.add_argument("--smi_folder", type=str, default="./observation/08_Confidence/SMI",
                        help="Path to SMI images folder")
    parser.add_argument("--output_dir", type=str, default="./observation/00_Figures",
                        help="Output directory for figures")
    parser.add_argument("--output_name", type=str, default="fig4b_saliency_distribution.png",
                        help="Output filename")
    parser.add_argument("--model", type=str, default="deit_base_16_imagenet",
                        help="Model name for saliency computation")
    parser.add_argument("--num_bins", type=int, default=20, 
                        help="Number of bins for saliency weight histogram")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    
    # Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot style
    plt.style.use("ggplot")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.grid"] = True
    
    # Load teacher model
    print("Loading teacher model...")
    teacher = get_teacher(args.model, device)
    
    # -------------------------
    # Compute saliency distributions
    # -------------------------
    
    # 1) ImageNet
    print("\n" + "="*60)
    print("Processing ImageNet...")
    print("="*60)
    imagenet_bin_edges, imagenet_ratios = compute_saliency_histogram_for_imagenet(
        imagenet_root=Path(args.imagenet_root),
        teacher_model=teacher,
        device=device,
        num_bins=args.num_bins,
        num_classes=1000,
        n_samples_per_class=1,
        patch_size=args.patch_size,
        seed=args.seed,
    )
    print(f"ImageNet: {len(imagenet_ratios)} bins computed")
    
    # 2) DMI (DeepInversion)
    print("\n" + "="*60)
    print("Processing DMI (DeepInversion)...")
    print("="*60)
    dmi_bin_edges, dmi_ratios = compute_saliency_histogram(
        images_folder=Path(args.dmi_folder),
        teacher_model=teacher,
        device=device,
        num_bins=args.num_bins,
        max_images=1000,
        patch_size=args.patch_size,
        desc="DMI",
    )
    print(f"DMI: {len(dmi_ratios)} bins computed")
    
    # 3) SMI (Sparse Model Inversion)
    print("\n" + "="*60)
    print("Processing SMI (Sparse Model Inversion)...")
    print("="*60)
    smi_bin_edges, smi_ratios = compute_saliency_histogram(
        images_folder=Path(args.smi_folder),
        teacher_model=teacher,
        device=device,
        num_bins=args.num_bins,
        max_images=1000,
        patch_size=args.patch_size,
        desc="SMI",
    )
    print(f"SMI: {len(smi_ratios)} bins computed")
    
    # -------------------------
    # Create plot
    # -------------------------
    print("\n" + "="*60)
    print("Creating plot...")
    print("="*60)
    
    output_path = os.path.join(args.output_dir, args.output_name)
    plot_saliency_distribution(
        bin_edges=imagenet_bin_edges,
        imagenet_ratios=imagenet_ratios,
        dmi_ratios=dmi_ratios,
        smi_ratios=smi_ratios,
        output_path=output_path,
    )
    
    print("\n" + "="*60)
    print("Done!")
    print(f"Output saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
