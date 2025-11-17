#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from torchvision import transforms
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Observation / Visualization script for synthetic images"
    )
    parser.add_argument(
        "--mode",
        type=int,
        choices=[1, 2, 3],
        # 1: Noise-Performance, 2: Iteration-Noise,
        # 3: Iteration-Performance, 4: Variance-Performance
        help="시각화 모드 설정",
    )
    parser.add_argument(
        "--img_folder",
        type=str,
        help="*.jpg 이미지가 폴더 경로",
    )
    parser.add_argument(
        "--local_path",
        type=str,
        help="시각화 결과가 저장될 폴더 경로",
    )
    return parser.parse_args()


def find_images(img_folder: Path) -> List[Path]:
    """
    Recursively find *.jpg images in img_folder.
    """
    if not img_folder.exists(): 
        raise FileNotFoundError(f"이미지 폴더가 존재하지 않습니다: {img_folder}")

    image_paths = sorted(
        [p for p in img_folder.rglob("*.jpg") if p.is_file()]
    )

    if len(image_paths) == 0:
        raise FileNotFoundError(f"*.jpg 이미지가 폴더에 없습니다: {img_folder}")

    return image_paths


def load_images(
    image_paths: List[Path],
) -> List[Tuple[Path, torch.Tensor]]:
    """
    [0, 1] 범위의 float 텐서로 이미지를 로드하고, [C,H,W] 형태로 변환합니다.
    """
    to_tensor = transforms.ToTensor()
    loaded = []

    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            tensor = to_tensor(img)  # [C,H,W], float32 in [0,1]
            loaded.append((p, tensor))
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}", file=sys.stderr)

    if len(loaded) == 0:
        raise RuntimeError("모든 이미지 로드에 실패했습니다; 시각화할 이미지가 없습니다.")

    return loaded


# ======================================================================
# 시각화 모드 (모드 1 / 2 / 3)
# ======================================================================

def visualize_mode1(
    images: List[Tuple[Path, torch.Tensor]],
    save_dir: Path,
):
    """
    Mode 1 visualization.
    You can use this for e.g. low-pass filtering visualization,
    FFT noise distribution, etc.
    """
    raise NotImplementedError(
        "Mode 1 visualization is not implemented yet. "
        "Fill this function with your visualization logic."
    )


def visualize_mode2(
    images: List[Tuple[Path, torch.Tensor]],
    save_dir: Path,
):
    """
    Mode 2 visualization.
    You can use this for e.g. saliency-centered metrics,
    edge maps, Grad-CAM overlays, etc.
    """
    raise NotImplementedError(
        "Mode 2 visualization is not implemented yet. "
        "Fill this function with your visualization logic."
    )


def visualize_mode3(
    images: List[Tuple[Path, torch.Tensor]],
    save_dir: Path,
):
    """
    Mode 3 visualization.
    You can use this for e.g. radial sparsification visualization,
    patch-wise energy, peripheral vs center comparison, etc.
    """
    raise NotImplementedError(
        "Mode 3 visualization is not implemented yet. "
        "Fill this function with your visualization logic."
    )


def main():
    args = parse_args()

    img_folder = Path(args.img_folder).expanduser().resolve()
    save_dir = Path(args.local_path).expanduser().resolve()

    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Mode       : {args.mode}")
    print(f"[INFO] Image dir  : {img_folder}")
    print(f"[INFO] Save dir   : {save_dir}")

    # 1) Find images
    image_paths = find_images(img_folder, max_images=args.max_images)
    print(f"[INFO] Found {len(image_paths)} images")

    # 2) Load images as tensors
    images = load_images(image_paths)
    print(f"[INFO] Successfully loaded {len(images)} images")

    # 3) Dispatch by mode
    if args.mode == "mode1":
        visualize_mode1(images, save_dir)
    elif args.mode == "mode2":
        visualize_mode2(images, save_dir)
    elif args.mode == "mode3":
        visualize_mode3(images, save_dir)
    else:
        # 이 줄은 theoretically unreachable (choices로 이미 걸러짐)
        raise ValueError(f"Unknown mode: {args.mode}")

    print("[INFO] Visualization finished.")


if __name__ == "__main__":
    main()