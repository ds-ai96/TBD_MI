"""
This script supports two analysis modes:
1. "per_class": sample exactly one image per ImageNet class from both the
   real dataset and a synthetic counterpart that mirrors the ImageFolder
   structure. The classifier-layer gradients for each set are averaged and
   compared.
2. "prebaked": use pre-generated synthetic images stored under
   `DMI_Seed/DMI-4000-{seed}-32-W4A8` and their labels recorded in
   `observation/06_Saliency_Analysis/targets/saliency_seed_{seed}.txt`.
   The script retrieves matching real ImageNet samples for those labels and
   compares gradients against the synthetic set.

The output highlights gradient statistics and the L2 distance between the
average classifier gradients of real and synthetic samples.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from main_quant import get_teacher
from utils import build_model
from synthesis import SMI
from synthesis._utils import Normalizer

@dataclass
class GradientSummary:
    weight: torch.Tensor
    bias: torch.Tensor | None

    def l2_distance(self, other: "GradientSummary") -> dict[str, float]:
        weight_diff = torch.norm(self.weight - other.weight, p=2).item()
        bias_diff = None
        if self.bias is not None and other.bias is not None:
            bias_diff = torch.norm(self.bias - other.bias, p=2).item()
        return {"weight_l2": weight_diff, "bias_l2": bias_diff}

    def norms(self) -> dict[str, float]:
        weight_norm = torch.norm(self.weight, p=2).item()
        bias_norm = torch.norm(self.bias, p=2).item() if self.bias is not None else None
        return {"weight": weight_norm, "bias": bias_norm}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classifier gradient comparison for real vs synthetic images")
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU to use"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["per_class", "prebaked"],
        required=True,
        help="Select between per-class sampling or pre-generated synthetic images",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deit_base_16_imagenet",
        help="Teacher model name (main_quant.get_teacher compatible); defaults to DeiT-Base/16",
    )
    parser.add_argument(
        "--imagenet-root",
        type=str,
        default="/home/mjatwk/data/imagenet/",
        help="Path to ImageNet validation folder (ImageFolder layout)",
    )
    parser.add_argument(
        "--synthetic-root",
        type=str,
        default="/dataset/deit_base_16_imagenet/DMI_Seed",
        help="Synthetic dataset root with ImageFolder-style class subfolders (per_class mode)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(1, 51)),
        help="Seed IDs to analyze in prebaked mode",
    )
    parser.add_argument(
        "--prebaked-root",
        type=str,
        default="./dataset/deit_base_16_imagenet/DMI_Seed",
        help="Root directory that holds DMI-4000-{seed}-32-W4A8 folders",
    )
    parser.add_argument(
        "--target-root",
        type=str,
        default=os.path.join("observation", "06_Saliency_Analysis", "targets"),
        help="Directory containing saliency_seed_{seed}.txt label files",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size for gradient computation")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (used when building tensors)")
    parser.add_argument(
        "--limit-classes",
        type=int,
        default=None,
        help="Optional cap on the number of classes to sample in per_class mode",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation",
    )
    return parser.parse_args()


def get_model(name):
    model_name = {'deit_tiny_16_imagenet': 'deit_tiny_patch16_224',
                    'deit_base_16_imagenet': 'deit_base_patch16_224',
                    }
    if name.split("_")[-1]=='imagenet':
        model = build_model(model_name[name], Pretrained=True)
    else:
        raise NotImplementedError
    return model

def generate_synthetic_dataset(
    root: str,
    model_name: str,
    student_model: nn.Module,
    device: torch.device,
    class_to_idx: dict[str, int],
    limit_classes: int | None = None
) -> None:
    print(f"Directory {root} does not exist. Generating synthetic images...")
    
    # Use a subdirectory for SMI's internal dump to avoid polluting the root or conflict with ImageFolder content
    dump_dir = os.path.join(root, "__raw_dump")
    os.makedirs(dump_dir, exist_ok=True)
    
    teacher = get_teacher(model_name).to(device)
    teacher.eval()
    
    # ImageNet stats
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalizer = Normalizer(mean, std)
    
    synthesizer = SMI(
        teacher=teacher,
        teacher_name=model_name,
        student=student_model,
        num_classes=1000,
        img_shape=(3, 224, 224),
        iterations=500,
        lr_g=0.25,
        synthesis_batch_size=32,
        sample_batch_size=32,
        save_dir=dump_dir,
        transform=None,
        normalizer=normalizer,
        device=device
    )
    
    classes_to_gen = sorted(list(class_to_idx.items()), key=lambda x: x[1])
    if limit_classes:
        classes_to_gen = classes_to_gen[:limit_classes]
    
    batch_size = 32
    
    import shutil
    
    total_batches = (len(classes_to_gen) + batch_size - 1) // batch_size
    
    for i in range(0, len(classes_to_gen), batch_size):
        batch_classes = classes_to_gen[i : i + batch_size]
        current_bs = len(batch_classes)
        
        targets = torch.tensor([idx for _, idx in batch_classes], device=device, dtype=torch.long)
        
        if current_bs < batch_size:
            pad_size = batch_size - current_bs
            # Pad with 0
            targets_padded = torch.cat([targets, torch.zeros(pad_size, device=device, dtype=torch.long)])
        else:
            targets_padded = targets
            
        print(f"Generating batch {i // batch_size + 1}/{total_batches}")
        
        results = synthesizer.synthesize(targets=targets_padded, num_patches=197, prune_it=[-1], prune_ratio=[0])
        synthetic_images = results['synthetic'] # Denormalized tensors [B, 3, H, W]
        
        for j, (class_name, class_idx) in enumerate(batch_classes):
            img_tensor = synthetic_images[j]
            img_np = (img_tensor.detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            
            class_dir = os.path.join(root, class_name)
            os.makedirs(class_dir, exist_ok=True)
            img_pil.save(os.path.join(class_dir, f"{class_name}_0.png"))
            
    # Cleanup raw dump
    try:
        shutil.rmtree(dump_dir)
    except Exception as e:
        print(f"Warning: Could not remove temporary dump directory {dump_dir}: {e}")

    print("Generation complete.")

def _load_one_image_per_class(dataset, label_mapper, expected_classes):
    seen = set()
    samples: List[Tuple[torch.Tensor, int]] = []
    for image, label in dataset:
        mapped_label = label_mapper(label)
        if mapped_label is None or mapped_label in seen:
            continue
        seen.add(mapped_label)
        samples.append((image, mapped_label))
        if len(samples) >= expected_classes:
            break
    if len(samples) < expected_classes:
        raise RuntimeError(f"Requested {expected_classes} classes but only found {len(samples)}")
    return samples

def load_one_image_per_class_imagenet(
    root: str, transform: T.Compose, max_classes: int | None = None
) -> Tuple[List[Tuple[torch.Tensor, int]], dict[str, int]]:
    dataset = datasets.ImageNet(root=root, split="val", transform=transform)
    expected = max_classes or len(dataset.classes)
    samples = _load_one_image_per_class(dataset, label_mapper=lambda lbl: lbl, expected_classes=expected)
    return samples, dataset.class_to_idx

def load_one_image_per_class_imagefolder(
    root: str, 
    transform: T.Compose, 
    class_to_idx: dict[str, int], 
    max_classes: int | None = None,
    model: nn.Module | None = None,
    model_name: str | None = None,
    device: torch.device | None = None
) -> List[Tuple[torch.Tensor, int]]:
    
    if not os.path.exists(root):
        if model is None or model_name is None or device is None:
             raise RuntimeError(f"Directory {root} missing and model/device info not provided for auto-generation")
        generate_synthetic_dataset(root, model_name, model, device, class_to_idx, max_classes)

    dataset = datasets.ImageFolder(root=root, transform=transform)
    expected = max_classes or len(class_to_idx)

    def label_mapper(label: int) -> int | None:
        class_name = dataset.classes[label]
        return class_to_idx.get(class_name)

    return _load_one_image_per_class(dataset, label_mapper=label_mapper, expected_classes=expected)

def build_loader_from_samples(samples: Sequence[Tuple[torch.Tensor, int]], batch_size: int, num_workers: int) -> DataLoader:
    images = torch.stack([img for img, _ in samples])
    labels = torch.tensor([label for _, label in samples], dtype=torch.long)
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def forward_teacher_logits(model: nn.Module, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    batch = images.size(0)
    if not hasattr(model, "pos_embed"):
        raise AttributeError("Teacher model is expected to expose pos_embed for patch indexing")
    patch_tokens = model.pos_embed.shape[1]
    current_abs_index = torch.arange(patch_tokens, device=device).unsqueeze(0).repeat(batch, 1)
    next_relative_index = current_abs_index.clone()
    outputs = model(images.to(device), current_abs_index, next_relative_index)
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def compute_classifier_gradient(
    model: nn.Module, images: torch.Tensor, labels: torch.Tensor, device: torch.device
) -> GradientSummary:
    criterion = nn.CrossEntropyLoss()
    model.zero_grad()
    outputs = forward_teacher_logits(model, images, device)
    loss = criterion(outputs, labels.to(device))
    loss.backward()
    classifier = model.head
    weight_grad = classifier.weight.grad.detach().clone()
    bias_grad = None
    if getattr(classifier, "bias", None) is not None:
        bias_grad = classifier.bias.grad.detach().clone()
    return GradientSummary(weight=weight_grad, bias=bias_grad)


def aggregate_gradients(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> GradientSummary:
    weight_sum = None
    bias_sum = None
    batches = 0
    for images, labels in loader:
        grad = compute_classifier_gradient(model, images, labels, device)
        if weight_sum is None:
            weight_sum = torch.zeros_like(grad.weight)
            bias_sum = torch.zeros_like(grad.bias) if grad.bias is not None else None
        weight_sum += grad.weight
        if grad.bias is not None and bias_sum is not None:
            bias_sum += grad.bias
        batches += 1
    if batches == 0:
        raise RuntimeError("No batches were processed; check dataset loading")
    weight_mean = weight_sum / batches
    bias_mean = bias_sum / batches if bias_sum is not None else None
    return GradientSummary(weight=weight_mean, bias=bias_mean)


def read_targets_from_file(path: Path) -> List[int]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    targets = payload.get("targets")
    if not isinstance(targets, list):
        raise ValueError(f"No 'targets' list found in {path}")
    return [int(t) for t in targets]


def load_synthetic_seed_images(root: Path, seed: int) -> List[Path]:
    image_dir = root / f"DMI-4000-{seed}-32-W4A8"
    if not image_dir.exists():
        raise FileNotFoundError(f"Synthetic image directory missing: {image_dir}")
    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")
    return image_paths


def load_images_from_paths(paths: Sequence[Path], transform: T.Compose) -> List[torch.Tensor]:
    tensors = []
    for path in paths:
        with Image.open(path) as img:
            tensors.append(transform(img.convert("RGB")))
    return tensors


def gather_real_images_for_labels(root: str, labels: Sequence[int], transform: T.Compose) -> List[Tuple[torch.Tensor, int]]:
    required = Counter(labels)
    dataset = datasets.ImageNet(root=root, split='val', transform=transform)
    collected: List[Tuple[torch.Tensor, int]] = []
    remaining = sum(required.values())
    for image, label in dataset:
        if required[label] <= 0:
            continue
        collected.append((image, label))
        required[label] -= 1
        remaining -= 1
        if remaining == 0:
            break
    if remaining != 0:
        missing = {lbl: cnt for lbl, cnt in required.items() if cnt > 0}
        raise RuntimeError(f"Insufficient real images for labels: {missing}")
    return collected


def run_per_class_mode(args, model, device, transform) -> None:
    real_samples, class_to_idx = load_one_image_per_class_imagenet(args.imagenet_root, transform, args.limit_classes)
    synthetic_samples = load_one_image_per_class_imagefolder(
        args.synthetic_root, transform, class_to_idx, args.limit_classes,
        model=model, model_name=args.model, device=device
    )

    real_loader = build_loader_from_samples(real_samples, args.batch_size, args.num_workers)
    synthetic_loader = build_loader_from_samples(synthetic_samples, args.batch_size, args.num_workers)

    real_grad = aggregate_gradients(model, loader=real_loader, device=device)
    synthetic_grad = aggregate_gradients(model, loader=synthetic_loader, device=device)

    report_results("per_class", real_grad, synthetic_grad)


def run_prebaked_mode(args, model, device, transform):
    synthetic_root = Path(args.prebaked_root)
    target_root = Path(args.target_root)

    synthetic_samples = []
    real_samples = []

    for seed in args.seeds:
        target_path = target_root / f"saliency_seed_{seed}.txt"
        labels = read_targets_from_file(target_path)
        image_paths = load_synthetic_seed_images(synthetic_root, seed)
        if len(image_paths) != len(labels):
            raise RuntimeError(
                f"Seed {seed}: label count ({len(labels)}) and image count ({len(image_paths)}) do not match"
            )
        synthetic_tensors = load_images_from_paths(image_paths, transform)
        synthetic_samples.extend(zip(synthetic_tensors, labels))

        real_batch = gather_real_images_for_labels(args.imagenet_root, labels, transform)
        real_samples.extend(real_batch)

    real_loader = build_loader_from_samples(real_samples, args.batch_size, args.num_workers)
    synthetic_loader = build_loader_from_samples(synthetic_samples, args.batch_size, args.num_workers)

    real_grad = aggregate_gradients(model, real_loader, device)
    synthetic_grad = aggregate_gradients(model, synthetic_loader, device)

    report_results("prebaked", real_grad, synthetic_grad)


def report_results(mode: str, real_grad: GradientSummary, synthetic_grad: GradientSummary) -> None:
    distance = real_grad.l2_distance(synthetic_grad)
    print(f"\n[{mode}] Gradient comparison")
    print("Real gradient norms:", real_grad.norms())
    print("Synthetic gradient norms:", synthetic_grad.norms())
    print("L2 distance (weight, bias):", distance)


def set_seed(seed):
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

def main():
    args = parse_args()

    set_seed(42)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model(args.model)
    model = model.to(device)
    model.eval()

    transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.mode == "per_class":
        run_per_class_mode(args, model, device, transform)
    elif args.mode == "prebaked":
        run_prebaked_mode(args, model, device, transform)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()