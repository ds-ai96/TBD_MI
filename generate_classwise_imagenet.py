import argparse
import os
from typing import Iterable, List

import torch
from tqdm import tqdm

from main_quant import Config, get_student, get_teacher, set_seed
from synthesis.smi_per_class import SMIClasswise
from utils import build_dataset


def _chunk(items: List[int], chunk_size: int) -> Iterable[List[int]]:
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one (or more) synthetic ImageNet images per class using main_quant components.",
        add_help=True,
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store class-wise images.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the ImageNet dataset.")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA device id to use.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--model",
        default="deit_base_16_imagenet",
        choices=["deit_base_16_imagenet", "deit_tiny_16_imagenet"],
        help="Teacher/student model to use.",
    )
    parser.add_argument("--iterations", type=int, default=4000, help="Inversion iterations per batch.")
    parser.add_argument("--synthetic_bs", type=int, default=32, help="Batch size used during synthesis.")
    parser.add_argument("--images_per_class", type=int, default=1, help="Number of images to generate for each class.")
    parser.add_argument("--w_bit", type=int, default=8, help="Weight bit width for the student model.")
    parser.add_argument("--a_bit", type=int, default=8, help="Activation bit width for the student model.")
    parser.add_argument("--prune_it", nargs="+", type=int, default=[-1], help="Iterations to prune attention patches.")
    parser.add_argument(
        "--prune_ratio",
        nargs="+",
        type=float,
        default=[0],
        help="Prune ratio for each iteration specified by --prune_it.",
    )
    parser.add_argument("--use_soft_label", action="store_true", help="Use soft labels during synthesis.")
    parser.add_argument(
        "--soft_label_alpha",
        type=float,
        default=0.9,
        help="Alpha value when generating soft labels (probability assigned to the target class).",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1000,
        help="Total number of target classes to synthesize (defaults to ImageNet-1k).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = Config(args.w_bit, args.a_bit)
    student = get_student(args.model)(pretrained=True, cfg=cfg).to(device)
    student.eval()
    teacher = get_teacher(args.model).to(device)
    teacher.eval()

    _, _, num_classes, train_transform, _, normalizer = build_dataset(
        args.model.split("_")[0],
        args.model.split("_")[-1],
        args.synthetic_bs,
        train_aug=False,
        keep_zero=True,
        train_inverse=True,
        dataset_path=args.dataset,
    )

    target_classes = list(range(min(args.num_classes, num_classes)))
    requested_targets: List[int] = []
    for class_id in target_classes:
        requested_targets.extend([class_id] * args.images_per_class)

    patch_size = 16 if "16" in args.model else 32
    patch_num = 197 if patch_size == 16 else 50

    synthesizer = SMIClasswise(
        teacher=teacher,
        teacher_name=args.model,
        student=student,
        num_classes=num_classes,
        img_shape=(3, 224, 224),
        iterations=args.iterations,
        patch_size=patch_size,
        lr_g=0.25,
        synthesis_batch_size=args.synthetic_bs,
        adv=0,
        bn=0,
        oh=1,
        tv1=0,
        tv2=0.0001,
        l2=0,
        save_dir=args.output_dir,
        transform=train_transform,
        normalizer=normalizer,
        device=device,
        bnsource="resnet50v1",
    )

    total_batches = (len(requested_targets) + args.synthetic_bs - 1) // args.synthetic_bs
    for batch_targets in tqdm(_chunk(requested_targets, args.synthetic_bs), total=total_batches, desc="Per-class synthesis"):
        target_tensor = torch.tensor(batch_targets, device=device)
        results = synthesizer.synthesize(
            targets=target_tensor,
            num_patches=patch_num,
            prune_it=args.prune_it,
            prune_ratio=args.prune_ratio,
            use_soft_label=args.use_soft_label,
            soft_label_alpha=args.soft_label_alpha,
        )


if __name__ == "__main__":
    main()