import os
import sys
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import mannwhitneyu

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


def sample_one_per_class(root: Path, rng: random.Random) -> List[Path]:
    """
    If synthetic_root has class subdirs, pick 1 random image per class dir.
    """
    samples: List[Path] = []
    if not root.exists():
        return samples

    class_dirs = [p for p in root.iterdir() if p.is_dir()]
    for class_dir in sorted(class_dirs):
        images = list_image_files(class_dir)
        if images:
            samples.append(rng.choice(images))
    return samples


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
        # Safety: if resize produced something smaller (shouldn't), fallback to direct resize
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


# -------------------------
# Sampling
# -------------------------
def sample_imagenet_images(
    imagenet_root: str | Path,
    n_samples_per_class: int = 1,
    num_classes: int = 1000,
    size: int = 224,
    split: str = "val",
    seed: int = 0,
    resize_short: int = 256,
) -> List[np.ndarray]:
    rng = random.Random(seed)

    dataset = torchvision.datasets.ImageNet(root=str(imagenet_root), split=split, transform=None)
    class_to_paths: List[List[str]] = [[] for _ in range(num_classes)]
    for img_path, class_idx in dataset.samples:
        if 0 <= class_idx < num_classes:
            class_to_paths[class_idx].append(img_path)

    images: List[np.ndarray] = []
    for class_idx in range(num_classes):
        paths = class_to_paths[class_idx]
        if not paths:
            continue

        chosen = rng.sample(paths, n_samples_per_class) if len(paths) >= n_samples_per_class else paths
        for p in chosen:
            images.append(load_image_standard(p, size=size, resize_short=resize_short))

    return images


def sample_synthetic_images(
    synthetic_root: str | Path,
    size: int = 224,
    seed: int = 0,
    resize_short: int = 256,
    max_images_if_flat: Optional[int] = 1000,
) -> List[np.ndarray]:
    rng = random.Random(seed)
    synthetic_root = Path(synthetic_root)
    if not synthetic_root.exists():
        return []

    has_subdirs = any(p.is_dir() for p in synthetic_root.iterdir())
    if has_subdirs:
        paths = sample_one_per_class(synthetic_root, rng)
    else:
        paths = list_image_files(synthetic_root)
        if max_images_if_flat is not None:
            paths = paths[:max_images_if_flat]

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
    kind: str = "amplitude",        # "amplitude" | "power"
    fft_norm: str | None = "ortho", # None | "ortho"
    normalize: str = "none",        # "none" | "sum1"
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


# -------------------------
# HF band ratio
# -------------------------
def compute_hf_band_ratio(
    image: np.ndarray,
    *,
    kind: str = "amplitude",      # "amplitude" | "power"
    fft_norm: str | None = "ortho",
    hf_start: int = 80,
    size: int = 224,
    eps: float = 1e-12,
) -> float:
    prof = compute_radial_spectrum(
        image,
        kind=kind,
        fft_norm=fft_norm,
        normalize="none",   # 절대량
        eps=eps,
    )

    max_radius = size // 2
    prof = prof[: max_radius + 1]   # 0..112

    denom = prof.sum() + eps
    hf = prof[hf_start : max_radius + 1].sum()

    return float(hf / denom)

def collect_hf_band_ratios(
    images: list[np.ndarray],
    *,
    kind: str,
    fft_norm: str | None,
    hf_start: int,
    size: int,
) -> np.ndarray:
    ratios = [
        compute_hf_band_ratio(
            img,
            kind=kind,
            fft_norm=fft_norm,
            hf_start=hf_start,
            size=size,
        )
        for img in images
    ]
    return np.asarray(ratios, dtype=np.float64)


def cohens_d(a, b):
    a = np.asarray(a); b = np.asarray(b)
    na, nb = len(a), len(b)
    sa2, sb2 = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((na-1)*sa2 + (nb-1)*sb2) / (na+nb-2))
    return (a.mean() - b.mean()) / (sp + 1e-12)


def get_significance_stars(p_value):
    """Return significance stars based on p-value."""
    if p_value < 1e-3:
        return "***"
    elif p_value < 1e-2:
        return "**"
    elif p_value < 5e-2:
        return "*"
    else:
        return "n.s."


def plot_hf_band_ratio_boxplot_three(
    imagenet_images,
    synthetic_images,
    filtered_images,
    *,
    kind: str = "amplitude",
    fft_norm: str | None = "ortho",
    hf_start: int = 80,
    size: int = 224,
    output_path: str | None = None,
):
    """
    1x3 boxplot comparing ImageNet, Synthetic(DMI), and DMI_Filtered.
    Computes p-value between ImageNet and each of the other two groups.
    """
    # --- collect ratios
    hf_real = collect_hf_band_ratios(
        imagenet_images,
        kind=kind,
        fft_norm=fft_norm,
        hf_start=hf_start,
        size=size,
    )
    hf_syn = collect_hf_band_ratios(
        synthetic_images,
        kind=kind,
        fft_norm=fft_norm,
        hf_start=hf_start,
        size=size,
    )
    hf_filtered = collect_hf_band_ratios(
        filtered_images,
        kind=kind,
        fft_norm=fft_norm,
        hf_start=hf_start,
        size=size,
    )

    # --- statistics (one-sided: synthetic/filtered > real)
    u_stat_syn, p_value_syn = mannwhitneyu(
        hf_syn, hf_real, alternative="greater"
    )
    u_stat_filt, p_value_filt = mannwhitneyu(
        hf_filtered, hf_real, alternative="greater"
    )

    print(f"HF ratio (ImageNet):     mean={hf_real.mean():.4f}, std={hf_real.std():.4f}")
    print(f"HF ratio (Synthetic):    mean={hf_syn.mean():.4f}, std={hf_syn.std():.4f}")
    print(f"HF ratio (DMI_Filtered): mean={hf_filtered.mean():.4f}, std={hf_filtered.std():.4f}")
    print(f"Mann–Whitney U (Synthetic vs ImageNet): U={u_stat_syn:.1f}, p={p_value_syn:.3e}")
    print(f"Mann–Whitney U (DMI_Filtered vs ImageNet): U={u_stat_filt:.1f}, p={p_value_filt:.3e}")

    # --- significance stars
    sig_syn = get_significance_stars(p_value_syn)
    sig_filt = get_significance_stars(p_value_filt)

    # --- Cohen's d
    d_syn = cohens_d(hf_syn, hf_real)
    d_filt = cohens_d(hf_filtered, hf_real)

    # --- plot (1x3 figure)
    fig, ax = plt.subplots(figsize=(8, 4))

    labels = ["ImageNet", "Synthetic", "DMI_Filtered"]
    data = [hf_real, hf_syn, hf_filtered]

    try:
        bp = ax.boxplot(data, tick_labels=labels, showfliers=False)
    except TypeError:
        bp = ax.boxplot(data, labels=labels, showfliers=False)

    # --- p-value annotations
    y_max = max(hf_real.max(), hf_syn.max(), hf_filtered.max())
    
    # Annotation: ImageNet vs Synthetic (positions 1 and 2)
    y_bar1 = y_max * 1.10
    ax.plot([1, 2], [y_bar1, y_bar1], color="black", linewidth=1)
    ax.text(
        1.5, y_bar1 * 1.02,
        f"{sig_syn} (p={p_value_syn:.2e}, d={d_syn:.2f})",
        ha="center", va="bottom", fontsize=9
    )

    # Annotation: ImageNet vs DMI_Filtered (positions 1 and 3)
    y_bar2 = y_max * 1.25
    ax.plot([1, 1], [y_bar1 * 1.02, y_bar2], color="black", linewidth=1)
    ax.plot([1, 3], [y_bar2, y_bar2], color="black", linewidth=1)
    ax.text(
        2.0, y_bar2 * 1.02,
        f"{sig_filt} (p={p_value_filt:.2e}, d={d_filt:.2f})",
        ha="center", va="bottom", fontsize=9
    )

    ax.set_ylabel(f"HF band ratio (r ∈ [{hf_start}, {size//2}])")
    ax.set_title("High-frequency band ratio (amplitude)")
    ax.set_ylim(top=y_max * 1.40)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=300)
    plt.show()
    print("Output path: ", output_path)


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    seed = 0
    set_seed(seed)

    # plot env
    plt.style.use("ggplot")
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.rcParams["axes.grid"] = True
    print("Environment setup completed.")

    # 1) 이미지 리스트 로드 (중요)
    imagenet_images = sample_imagenet_images(
        imagenet_root="/home/mjatwk/data/imagenet",
        n_samples_per_class=1,
        num_classes=1000,
        size=224,
        split="val",
        seed=seed,
        resize_short=256,
    )
    synthetic_images = sample_synthetic_images(
        synthetic_root="./observation/08_Confidence/DMI",
        size=224,
        seed=seed,
        resize_short=256,
        max_images_if_flat=1000,
    )
    filtered_images = sample_synthetic_images(
        synthetic_root="./observation/08_Confidence/DMI_Filtered",
        size=224,
        seed=seed,
        resize_short=256,
        max_images_if_flat=1000,
    )

    print(f"# ImageNet images:     {len(imagenet_images)}")
    print(f"# Synthetic images:    {len(synthetic_images)}")
    print(f"# DMI_Filtered images: {len(filtered_images)}")

    print(imagenet_images[0].shape, imagenet_images[0].dtype, imagenet_images[0].min(), imagenet_images[0].max())

    # 2) boxplot 실행 (1x3)
    plot_hf_band_ratio_boxplot_three(
        imagenet_images=imagenet_images,
        synthetic_images=synthetic_images,
        filtered_images=filtered_images,
        kind="amplitude",
        fft_norm="ortho",
        hf_start=60,
        size=224,
        output_path="./observation/I1_challenge/hf_band_ratio_boxplot_pvalue_three.png",
    )

