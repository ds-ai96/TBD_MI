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


# -------------------------
# SNR (Signal-to-Noise Ratio)
# -------------------------
def compute_snr(
    image: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """
    Compute Signal-to-Noise Ratio for an image.
    SNR = mean / std of pixel intensities.
    """
    image = image.astype(np.float32, copy=False)
    # Convert to grayscale for consistent SNR computation
    if image.ndim == 3:
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        gray = image
    
    signal = gray.mean()
    noise = gray.std() + eps
    return float(signal / noise)


def compute_snr_frequency(
    image: np.ndarray,
    *,
    kind: str = "power",
    fft_norm: str | None = "ortho",
    lf_end: int = 20,
    size: int = 224,
    eps: float = 1e-12,
) -> float:
    """
    Compute frequency-based SNR.
    SNR = low-frequency energy / high-frequency energy
    Low frequency is considered as signal, high frequency as noise.
    """
    prof = compute_radial_spectrum(
        image,
        kind=kind,
        fft_norm=fft_norm,
        normalize="none",
        eps=eps,
    )

    max_radius = size // 2
    prof = prof[: max_radius + 1]

    # Low frequency (signal): [0, lf_end)
    lf = prof[:lf_end].sum()
    # High frequency (noise): [lf_end, max_radius]
    hf = prof[lf_end:].sum() + eps

    return float(lf / hf)


def collect_snr_values(
    images: list[np.ndarray],
    *,
    snr_type: str = "frequency",  # "intensity" | "frequency"
    kind: str = "power",
    fft_norm: str | None = "ortho",
    lf_end: int = 20,
    size: int = 224,
) -> np.ndarray:
    if snr_type == "intensity":
        values = [compute_snr(img) for img in images]
    else:  # frequency
        values = [
            compute_snr_frequency(
                img,
                kind=kind,
                fft_norm=fft_norm,
                lf_end=lf_end,
                size=size,
            )
            for img in images
        ]
    return np.asarray(values, dtype=np.float64)


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


def plot_boxplot_three(
    imagenet_images,
    synthetic_images,
    filtered_images=None,  # Optional: None이면 ImageNet vs Synthetic만 비교
    *,
    metric: str = "hf_ratio",  # "hf_ratio" | "snr_frequency" | "snr_intensity"
    kind: str = "amplitude",
    fft_norm: str | None = "ortho",
    hf_start: int = 80,
    lf_end: int = 20,  # for SNR frequency
    size: int = 224,
    output_path: str | None = None,
    box_start: float = 1.0,   # 첫 번째 boxplot의 x 위치
    box_gap: float = 1.0,     # boxplot 간 간격
):
    """
    Boxplot comparing ImageNet, Synthetic(DMI), and optionally DMI_Filtered.
    If filtered_images is None, compares only ImageNet vs Synthetic (1x2).
    Computes p-value between ImageNet and each of the other groups.
    
    metric options:
      - "hf_ratio": High-frequency band ratio (original)
      - "snr_frequency": Frequency-based SNR (LF/HF energy)
      - "snr_intensity": Intensity-based SNR (mean/std)
    """
    include_filtered = filtered_images is not None and len(filtered_images) > 0
    
    # --- collect metric values
    if metric == "hf_ratio":
        vals_real = collect_hf_band_ratios(
            imagenet_images, kind=kind, fft_norm=fft_norm, hf_start=hf_start, size=size
        )
        vals_syn = collect_hf_band_ratios(
            synthetic_images, kind=kind, fft_norm=fft_norm, hf_start=hf_start, size=size
        )
        if include_filtered:
            vals_filt = collect_hf_band_ratios(
                filtered_images, kind=kind, fft_norm=fft_norm, hf_start=hf_start, size=size
            )
        ylabel = f"High-Frequency Ratio"
        # title = f"High-frequency band ratio ({kind})"
        # For HF ratio: synthetic/filtered > real
        alternative = "greater"
    elif metric == "snr_frequency":
        vals_real = collect_snr_values(
            imagenet_images, snr_type="frequency", kind=kind, fft_norm=fft_norm, lf_end=lf_end, size=size
        )
        vals_syn = collect_snr_values(
            synthetic_images, snr_type="frequency", kind=kind, fft_norm=fft_norm, lf_end=lf_end, size=size
        )
        if include_filtered:
            vals_filt = collect_snr_values(
                filtered_images, snr_type="frequency", kind=kind, fft_norm=fft_norm, lf_end=lf_end, size=size
            )
        ylabel = f"SNR (LF/HF, lf_end={lf_end})"
        title = f"Signal-to-Noise Ratio (frequency-based, {kind})"
        # For SNR: real > synthetic/filtered (real images have higher SNR)
        alternative = "less"
    elif metric == "snr_intensity":
        vals_real = collect_snr_values(imagenet_images, snr_type="intensity")
        vals_syn = collect_snr_values(synthetic_images, snr_type="intensity")
        if include_filtered:
            vals_filt = collect_snr_values(filtered_images, snr_type="intensity")
        ylabel = "SNR (mean/std)"
        title = "Signal-to-Noise Ratio (intensity-based)"
        alternative = "less"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # --- statistics
    u_stat_syn, p_value_syn = mannwhitneyu(
        vals_syn, vals_real, alternative=alternative
    )
    if include_filtered:
        u_stat_filt, p_value_filt = mannwhitneyu(
            vals_filt, vals_real, alternative=alternative
        )

    print(f"{metric} (ImageNet):     mean={vals_real.mean():.4f}, std={vals_real.std():.4f}")
    print(f"{metric} (Synthetic):    mean={vals_syn.mean():.4f}, std={vals_syn.std():.4f}")
    if include_filtered:
        print(f"{metric} (DMI_Filtered): mean={vals_filt.mean():.4f}, std={vals_filt.std():.4f}")
    print(f"Mann–Whitney U (Synthetic vs ImageNet): U={u_stat_syn:.1f}, p={p_value_syn:.3e}")
    if include_filtered:
        print(f"Mann–Whitney U (DMI_Filtered vs ImageNet): U={u_stat_filt:.1f}, p={p_value_filt:.3e}")

    # --- significance stars
    sig_syn = get_significance_stars(p_value_syn)
    if include_filtered:
        sig_filt = get_significance_stars(p_value_filt)

    # --- Cohen's d
    d_syn = cohens_d(vals_syn, vals_real)
    if include_filtered:
        d_filt = cohens_d(vals_filt, vals_real)

    # --- plot
    if include_filtered:
        fig, ax = plt.subplots(figsize=(5, 4))
        labels = ["ImageNet", "Synthetic", "DMI_Filtered"]
        data = [vals_real, vals_syn, vals_filt]
    else:
        fig, ax = plt.subplots(figsize=(4, 4))
        labels = ["ImageNet", "DeepInversion"]
        data = [vals_real, vals_syn]

    positions = [box_start + i * box_gap for i in range(len(data))]
    try:
        bp = ax.boxplot(data, positions=positions, widths=0.6, tick_labels=labels, showfliers=False)
    except TypeError:
        bp = ax.boxplot(data, positions=positions, widths=0.6, labels=labels, showfliers=False)

    ax.set_xlim(positions[0] - 0.5, positions[-1] + 0.5)

    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    # ax.set_title(title)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=600)
        # Save statistics to .txt file
        txt_path = output_path.rsplit(".", 1)[0] + ".txt"
        with open(txt_path, "w") as f:
            f.write(f"{metric} (ImageNet):     mean={vals_real.mean():.4f}, std={vals_real.std():.4f}\n")
            f.write(f"{metric} (Synthetic):    mean={vals_syn.mean():.4f}, std={vals_syn.std():.4f}\n")
            if include_filtered:
                f.write(f"{metric} (DMI_Filtered): mean={vals_filt.mean():.4f}, std={vals_filt.std():.4f}\n")
            f.write(f"Mann–Whitney U (Synthetic vs ImageNet): U={u_stat_syn:.1f}, p={p_value_syn:.3e}\n")
            if include_filtered:
                f.write(f"Mann–Whitney U (DMI_Filtered vs ImageNet): U={u_stat_filt:.1f}, p={p_value_filt:.3e}\n")
            f.write(f"Cohen's d (Synthetic vs ImageNet): {d_syn:.4f}\n")
            if include_filtered:
                f.write(f"Cohen's d (DMI_Filtered vs ImageNet): {d_filt:.4f}\n")
        print("Stats path: ", txt_path)
    plt.show()
    print("Output path: ", output_path)


# Backward compatibility alias
def plot_hf_band_ratio_boxplot_three(*args, **kwargs):
    """Alias for backward compatibility. Use plot_boxplot_three instead."""
    kwargs.setdefault("metric", "hf_ratio")
    return plot_boxplot_three(*args, **kwargs)


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    seed = 0
    set_seed(seed)

    # plot env
    # plt.style.use("ggplot")
    plt.rcParams["figure.figsize"] = (3, 4)
    # plt.rcParams["axes.grid"] = True
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
    # metric = "hf_ratio"  # ["hf_ratio", "snr_frequency", "snr_intensity"]
    # kind = "amplitude" # ["amplitude", "power"]
    
    plot_boxplot_three(
        imagenet_images=imagenet_images,
        synthetic_images=synthetic_images,
        # filtered_images=filtered_images,
        metric="hf_ratio",
        kind="power",
        fft_norm="ortho",
        hf_start=80,
        lf_end=32,
        size=224,
        output_path=f"./observation_python/figures/fig4_a.jpg",
        box_start=1.0,   # 첫 번째 boxplot 시작 위치 (오른쪽으로 옮기려면 값 증가)
        box_gap=1.0,     # boxplot 사이 간격 (넓히려면 값 증가)
    )

    # Hyperparameter search
    # for metric in ["hf_ratio", "snr_frequency", "snr_intensity"]:
    #     for kind in ["amplitude", "power"]:
    #         if metric == "hf_ratio":
    #             hfs = [64, 72, 80, 88]
    #             lfs = [32]
    #         elif metric == "snr_frequency":
    #             hfs = [60]
    #             lfs = [16, 20, 24, 32]
    #         elif metric == "snr_intensity":
    #             hfs = [88]
    #             lfs = [32]
    #         for hf_start in hfs:
    #             for lf_end in lfs:           
    #                 plot_boxplot_three(
    #                     imagenet_images=imagenet_images,
    #                     synthetic_images=synthetic_images,
    #                     # filtered_images=filtered_images,
    #                     metric=metric,
    #                     kind=kind,
    #                     fft_norm="ortho",
    #                     hf_start=hf_start,
    #                     lf_end=lf_end,
    #                     size=224,
    #                     output_path=f"./observation/I1_challenge/{metric}_boxplot_{kind}_{hf_start}_{lf_end}_three.png",
    #                 )
