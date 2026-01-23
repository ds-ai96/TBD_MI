import os
import sys
import time
import wandb
import random
import shutil
import argparse
from tqdm import tqdm

import numpy as np
from torch.utils.data import DataLoader

from utils import *
from synthesis import SMI, SMI_VAR, TBD_MI, DMI_EDGE
from quantization import *
from utils.data_utils import find_non_zero_patches

def get_args_parser():
    parser = argparse.ArgumentParser(description="Model Inversion (Quantization)", add_help=False)
    # Environment
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU to use"
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="seed"
    )

    # Experiments
    parser.add_argument(
        "--mode",
        type=str,
        choices=["DMI", "SMI", "Gaussian", "TBD_MI"],
        help="mode to use for data generation"
    )
    parser.add_argument(
        "--model",
        default="deit_base_16_imagenet",
        choices=["deit_base_16_imagenet", "deit_tiny_16_imagenet"],
        help="model to use"
    )
    parser.add_argument(
        "--dataset",
        default="/path/to/dataset",
        help="path to dataset"
    )
    parser.add_argument(
        "--datapool",
        default="/path/to/datapool",
        help="path to datapool"
    )

    # Data Generation
    parser.add_argument(
        "--iterations",
        type=int,
        default=4000,
        help="total number of iterations for inversion"
    )
    parser.add_argument(
        "--synthetic_bs",
        type=int,
        default=32,
        help="batch size for synthetic data"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="number of runs to generate synthetic data (# of images = num_runs * synthetic_bs)"
    )    

    # Downstream Tasks
    parser.add_argument(
        "--calib_batchsize",
        type=int,
        default=200,
        help="batch size for calibration"
    )
    parser.add_argument(
        "--w_bit",
        type=int,
        default=8,
        help="bit-precision of weights"
    )
    parser.add_argument(
        "--a_bit",
        type=int,
        default=8,
        help="bit-precision of activation"
    )

    # SMI
    parser.add_argument(
        "--prune_it",
        nargs='+',
        type=int,
        default=[-1],
        help='the iteration indexes for inversion stopping; -1: to densely invert data; t1 t2 ... tn: to sparsely invert data and perform inversion stopping at t1, t2, ..., tn'
    )
    parser.add_argument(
        "--prune_ratio",
        nargs='+',
        type=float,
        default=[0],
        help='the proportion of patches to be pruned relative to the current remaining patches; 0: to densely invert data; r1 r2 ... rn: progressively stopping the inversion of a fraction (r1, r2, ..., rn)$$ of patches at iterations (t1, t2, ..., tn), respectively'
    )

    # TODO: Proposed MI
    ### Idea 1. Low-Pass Filtering
    parser.add_argument(
        "--lpf",
        action="store_true",
        help="use low-pass filtering"
    )
    parser.add_argument(
        "--lpf_type",
        type=str,
        default="cutoff",
        choices=["gaussian", "cutoff", "bilateral"],
        help="type of low-pass filtering"
    )
    parser.add_argument(
        "--lpf_every",
        type=int,
        default=100,
        help="interval for low-pass filtering"
    )
    parser.add_argument(
        "--cutoff_ratio",
        type=float,
        default=0.5,
        help="cutoff ratio for low-pass filtering"
    )
    ##### For Bilateral Filter
    parser.add_argument(
        "--bi_kernel",
        type=int,
        default=5,
        help="kernel size for bilateral filter"
    )
    parser.add_argument(
        "--bi_sigma_s",
        type=float,
        default=2.0,
        help="sigma_s for bilateral filter"
    )
    parser.add_argument(
        "--bi_sigma_r",
        type=float,
        default=1.0,
        help="sigma_r for bilateral filter"
    )

    ### Idea 2. Saliency Map Centering
    parser.add_argument(
        "--sc_center",
        action="store_true",
        help="use saliency map centering"
    )
    parser.add_argument(
        "--sc_every",
        type=int,
        default=100,
        help="interval for saliency map centering"
    )
    parser.add_argument(
        "--sc_center_lambda",
        type=float,
        default=1.0,
        help="lambda for saliency map centering"
    )

    # Observation
    ## 가우시안 노이즈에서 variance 변화에 따른 성능 차이
    parser.add_argument(
        "--variance",
        type=float,
        default=-1,
        help="variance of the Gaussian noise; -1: to use SMI"
    )

    ## saliency map & sparsification 위치에 대한 실험
    parser.add_argument(
        "--saliency_anchor",
        type=str,
        default="c",
        choices=["se", "sw", "ne", "nw", "e", "w", "s", "n", "seh", "swh", "neh", "nwh", "eh", "wh", "sh", "nh", "c"],
        help="anchor for saliency map centering"
    )

    ## LPF 이후에 reward에 대한 가정 검증 실험
    parser.add_argument(
        "--reward_after_lpf",
        action="store_true",
        help="use reward after LPF"
    )

    ### edge 정보 더할 때, smoothness 조절
    parser.add_argument(
        "--smoothness",
        type=float,
        default=0.0,
        help="smoothness for edge information"
    )

    ### edge 정보 align 할 때, scaling 조절
    parser.add_argument(
        "--scale_edge",
        type=float,
        default=0.0,
        help="scale for edge information"
    )

    ## Hard-label 대신 Soft-label을 사용하는 경우
    parser.add_argument(
        "--use_soft_label",
        action="store_true",
        help="use soft labels for targets"
    )

    parser.add_argument(
        "--soft_label_alpha",
        type=float,
        default=0.9,
        help="alpha for soft labels (target class probability)"
    )

    # Logging
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="use wandb to log results"
    )
    parser.add_argument(
        "--project_name",
        type=str,
        help="name of the project"
    )

    return parser.parse_args()

# def kldiv( logits, targets, T=1.0, reduction='batchmean'):
#     q = F.log_softmax(logits/T, dim=1)
#     p = F.softmax( targets/T, dim=1 )
#     return F.kl_div( q, p, reduction=reduction ) * (T*T)

# class KLDiv(nn.Module):
#     def __init__(self, T=1.0, reduction='batchmean'):
#         super().__init__()
#         self.T = T
#         self.reduction = reduction

#     def forward(self, logits, targets):
#         return kldiv(logits, targets, T=self.T, reduction=self.reduction)

class Config:
    def __init__(self, w_bit, a_bit):
        self.weight_bit = w_bit
        self.activation_bit = a_bit


def get_teacher(name):
    teacher_name = {'deit_tiny_16_imagenet': 'deit_tiny_patch16_224',
                    'deit_base_16_imagenet': 'deit_base_patch16_224',
                    }
    if name.split("_")[-1]=='imagenet':
        teacher=build_model(teacher_name[name], Pretrained=True)
    else:
        raise NotImplementedError
    return teacher

def get_student(name):
    model_zoo = {'deit_tiny_16_imagenet': deit_tiny_patch16_224,
                 'deit_base_16_imagenet': deit_base_patch16_224,
                 }
    print('Model: %s' % model_zoo[name].__name__)
    return model_zoo[name]

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
    args = get_args_parser()

    if args.wandb:
        if args.mode == "SMI":
            if args.variance == -1:
                run_name = f"{args.mode}-{args.iterations}-{'-'.join(map(str, args.prune_it))}-{'-'.join(map(str, args.prune_ratio))}-{args.seed}-{args.synthetic_bs*args.num_runs}-W{args.w_bit}A{args.a_bit}"
            else:
                run_name = f"{args.mode}-{args.iterations}-{args.variance}-{'-'.join(map(str, args.prune_it))}-{'-'.join(map(str, args.prune_ratio))}-{args.seed}-{args.synthetic_bs*args.num_runs}-W{args.w_bit}A{args.a_bit}"
        elif args.mode == "DMI":
            if args.variance == -1:
                run_name = f"{args.mode}-{args.iterations}-{args.seed}-{args.synthetic_bs*args.num_runs}-W{args.w_bit}A{args.a_bit}"
                if args.reward_after_lpf:
                    run_name += f"-LPF_EDGE-{args.smoothness}-{args.scale_edge}"
                if args.use_soft_label:
                    run_name += f"-SOFT_LABEL-{args.soft_label_alpha}"
            else:
                run_name = f"{args.mode}-{args.iterations}-{args.variance}-{args.seed}-{args.synthetic_bs*args.num_runs}-W{args.w_bit}A{args.a_bit}"
        elif args.mode == "TBD_MI":
            if args.variance == -1:
                run_name = (
                    f"{args.mode}-SC-{args.iterations}-"
                    f"{str(args.lpf_every)}-"
                    f"{str(args.cutoff_ratio)}-"
                    f"{str(args.sc_every)}-"
                    f"{str(args.sc_center_lambda)}-"
                    f"{str(args.saliency_anchor)}-"
                    f"{'-'.join(map(str, args.prune_it))}-"
                    f"{'-'.join(map(str, args.prune_ratio))}-"
                    f"{args.seed}-{args.synthetic_bs*args.num_runs}-"
                    f"W{args.w_bit}A{args.a_bit}"
                )
                if args.reward_after_lpf:
                    run_name += f"-LPF_EDGE-{args.smoothness}-{args.scale_edge}"
                if args.use_soft_label:
                    run_name += f"-Soft-{args.soft_label_alpha}"
            else:
                run_name = (
                    f"{args.mode}-{args.iterations}-{args.variance}-"
                    f"{str(args.lpf_every)}-"
                    f"{str(args.cutoff_ratio)}-"
                    f"{str(args.sc_every)}-"
                    f"{str(args.sc_center_lambda)}-"
                    f"{str(args.saliency_anchor)}-"
                    f"{'-'.join(map(str, args.prune_it))}-"
                    f"{'-'.join(map(str, args.prune_ratio))}-"
                    f"{args.seed}-{args.synthetic_bs*args.num_runs}-"
                    f"W{args.w_bit}A{args.a_bit}"
                )
                if args.reward_after_lpf:
                    run_name += f"-LPF_EDGE-{args.smoothness}-{args.scale_edge}"
                if args.use_soft_label:
                    run_name += f"-Soft-{args.soft_label_alpha}"
        else:
            raise NotImplementedError

        wandb.init(
            project=args.project_name,
            name=run_name,
            config=vars(args)
        )
    
    print(f"Arguments: {args}")

    # Set Environments
    set_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load bit-config
    cfg = Config(args.w_bit, args.a_bit)

    # Build model
    model = get_student(args.model)(pretrained=True, cfg=cfg)
    model = model.to(device)
    model.eval()

    teacher=get_teacher(args.model)
    teacher=teacher.to(device)
    teacher.eval()

    # Build dataloader
    _, val_loader, num_classes, train_transform, _, normalizer = build_dataset(
        args.model.split("_")[0], args.model.split("_")[-1], args.calib_batchsize,
        train_aug=False, keep_zero=True, train_inverse=True, dataset_path=args.dataset
    )

    #########################################################
    # Model Inversion (Quantization)
    #########################################################


    # Define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(device)

    # Get calibration set
    # Case 0: inversion
    if args.mode in ["DMI", "SMI"]:
        iterations = args.iterations #Total number of iterations for inversion
        lr_g = 0.25 #Learning rate for inversion
        
        # Coefficient for inversion
        adv = 0 # Coefficient of adversarial regularization, we do not use it in our work
        bn = 0.0 # Coefficient of batch normalization regularization, dose not apply to ViTs due to the absence of batch normalization, we borrow a CNN to only facilitate visualization (refer to GradViT: Gradient Inversion of Vision Transformers)
        oh = 1 # Coefficient of classification loss
        tv1 = 0 # Coefficient of total variance regularization with l1 norm, we do not use it in our work
        tv2 = 0.0001 # Coefficient of total variance regularization with l2 norm
        l2 = 0 # Coefficient of l2 norm regularization, we do not use it in our work
        
        # Parameters for SMI
        prune_it = args.prune_it
        prune_ratio = args.prune_ratio

        patch_size = 16 if '16' in args.model else 32
        patch_num = 197 if patch_size==16 else 50

        if args.mode == "DMI":
            if args.variance == -1:
                run_name = f"{args.mode}-{args.iterations}-{args.seed}-{args.synthetic_bs*args.num_runs}-W{args.w_bit}A{args.a_bit}"
            else:
                run_name = f"{args.mode}-{args.iterations}-{args.variance}-{args.seed}-{args.synthetic_bs*args.num_runs}-W{args.w_bit}A{args.a_bit}"
            img_tag = run_name
        elif args.mode == "SMI":
            # img_tag = f"{args.mode}-{iterations}-{str(prune_it)}-{str(prune_ratio)}-{str(args.seed)}-{str(args.synthetic_bs*args.num_runs)}-W{args.w_bit}A{args.a_bit}"
            img_tag = run_name
        else:
            raise NotImplementedError
        
        datapool_path=os.path.join(args.datapool,'%s/%s'%(args.model,img_tag)) # The path to store inverted data
        if os.path.exists(datapool_path):
            shutil.rmtree(datapool_path)
            print(f"Removed existing data at {datapool_path}")
        
        # Generate synthetic data

        if args.reward_after_lpf:
            synthesizer = DMI_EDGE(
                teacher=teacher, teacher_name=args.model, student=model, num_classes=num_classes,
                img_shape=(3, 224, 224), iterations=iterations, patch_size=patch_size, lr_g=lr_g,
                synthesis_batch_size=args.synthetic_bs, sample_batch_size=args.calib_batchsize,
                adv=adv, bn=bn, oh=oh, tv1=tv1,tv2=tv2, l2=l2,
                save_dir=datapool_path, transform=train_transform,
                normalizer=normalizer, device=device, bnsource='resnet50v1', init_dataset=None
            )
        elif args.variance == -1:
            synthesizer = SMI(
                teacher=teacher, teacher_name=args.model, student=model, num_classes=num_classes,
                img_shape=(3, 224, 224), iterations=iterations, patch_size=patch_size, lr_g=lr_g,
                synthesis_batch_size=args.synthetic_bs, sample_batch_size=args.calib_batchsize,
                adv=adv, bn=bn, oh=oh, tv1=tv1,tv2=tv2, l2=l2,
                save_dir=datapool_path, transform=train_transform,
                normalizer=normalizer, device=device, bnsource='resnet50v1', init_dataset=None
            )
        else:
            synthesizer = SMI_VAR(
                teacher=teacher, teacher_name=args.model, student=model, num_classes=num_classes,
                img_shape=(3, 224, 224), iterations=iterations, patch_size=patch_size, lr_g=lr_g,
                synthesis_batch_size=args.synthetic_bs, sample_batch_size=args.calib_batchsize,
                adv=adv, bn=bn, oh=oh, tv1=tv1,tv2=tv2, l2=l2,
                save_dir=datapool_path, transform=train_transform,
                normalizer=normalizer, device=device, bnsource='resnet50v1', init_dataset=None,
                variance=args.variance
            )

        print(f"Generating data to {datapool_path}...")
        total_imgs = 0
        
        val_iter = iter(val_loader)
        
        for run_idx in tqdm(range(args.num_runs), desc="Synthesizing"):
            start = time.time()
            if args.reward_after_lpf:
                # Sample real images
                try:
                    real_images, real_targets = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    real_images, real_targets = next(val_iter)
                
                # Ensure batch size matches synthetic_bs
                if real_images.size(0) < args.synthetic_bs:
                    # Should unlikely fail with small synthetic_bs, but handled simple by repeating or slicing
                    pass
                real_images = real_images[:args.synthetic_bs].to(device)
                real_targets = real_targets[:args.synthetic_bs].to(device)
                
                results = synthesizer.synthesize(
                    targets=real_targets,
                    real_images=real_images,
                    num_patches=patch_num,
                    prune_it=prune_it,
                    prune_ratio=prune_ratio,
                    lpf=args.lpf, 
                    lpf_every=args.lpf_every, 
                    cutoff_ratio=args.cutoff_ratio,
                    smoothness=args.smoothness,
                    scale_edge=args.scale_edge
                )
            else:
                # Question07: Soft label 사용 관련
                results = synthesizer.synthesize(
                    num_patches=patch_num,
                    prune_it=prune_it,
                    prune_ratio=prune_ratio,
                    use_soft_label=args.use_soft_label,
                    soft_label_alpha=args.soft_label_alpha
                )
            if args.wandb and 'targets' in results:
                wandb.log({"targets": results['targets']}, step=args.seed)

            elapsed = time.time() - start
            print(f"[Run {run_idx+1}/{args.num_runs}] Time: {elapsed:.2f}s")

        dst = synthesizer.data_pool.get_dataset(transform=train_transform)
        print(f"[Calibration] Total synthetic images in pool: {len(dst)}")

        calib_loader = DataLoader(
            dst,
            batch_size=args.calib_batchsize,
            shuffle=False, num_workers=0, pin_memory=True,
        )

        print(f"Calibrating with generated data (full pool pass)...")
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

        # Validate the quantized model
        print("Validating...")
        val_loss, val_prec1, val_prec5 = validate(
            args, val_loader, model, criterion, device
        )

        if args.wandb:
            wandb.log({
                "val_loss": float(val_loss),
                "val_prec1": float(val_prec1),
                "val_prec5": float(val_prec5)
            })

    # TODO: Mode 1: TBD_MI
    elif args.mode == "TBD_MI":
        iterations = args.iterations
        lr_g = 0.25

        # Hyperparameters
        adv = 0
        bn = 0.0
        oh = 1
        tv1 = 0
        tv2 = 0.0001
        l2 = 0

        # Parameters for Sparsification
        prune_it = args.prune_it
        prune_ratio = args.prune_ratio

        patch_size = 16 if '16' in args.model else 32
        patch_num = 197 if patch_size==16 else 50

        img_tag = run_name
        datapool_path=os.path.join(args.datapool,'%s/%s'%(args.model,img_tag)) # The path to store inverted data
        if os.path.exists(datapool_path):
            shutil.rmtree(datapool_path)
            print(f"Removed existing data at {datapool_path}")

        synthesizer = TBD_MI(
            teacher=teacher, teacher_name=args.model, student=model, num_classes=num_classes,
                img_shape=(3, 224, 224), iterations=iterations, patch_size=patch_size, lr_g=lr_g,
                synthesis_batch_size=args.synthetic_bs, sample_batch_size=args.calib_batchsize,
                adv=adv, bn=bn, oh=oh, tv1=tv1,tv2=tv2, l2=l2,
                save_dir=datapool_path, transform=train_transform,
                normalizer=normalizer, device=device, bnsource='resnet50v1', init_dataset=None
        )

        print(f"Generating data to {datapool_path}...")
        total_imgs = 0
        for run_idx in tqdm(range(args.num_runs), desc="Synthesizing"):
            start = time.time()
            results = synthesizer.synthesize(
                num_patches=patch_num, prune_it=prune_it, prune_ratio=prune_ratio,
                lpf=args.lpf, lpf_every=args.lpf_every, cutoff_ratio=args.cutoff_ratio,
                scale_edge=args.scale_edge,
                sc_center=args.sc_center, sc_every=args.sc_every, sc_center_lambda=args.sc_center_lambda,
                saliency_anchor=args.saliency_anchor,
                use_soft_label=args.use_soft_label, soft_label_alpha=args.soft_label_alpha
            )
            if args.wandb and 'targets' in results:
                wandb.log({"targets": results['targets']}, step=args.seed)

            elapsed = time.time() - start
            print(f"[Run {run_idx+1}/{args.num_runs}] Time: {elapsed:.2f}s")

        dst = synthesizer.data_pool.get_dataset(transform=train_transform)
        print(f"[Calibration] Total synthetic images in pool: {len(dst)}")

        calib_loader = DataLoader(
            dst,
            batch_size=args.calib_batchsize,
            shuffle=False, num_workers=0, pin_memory=True,
        )

        print(f"Calibrating with generated data (full pool pass)...")
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

        # Validate the quantized model
        print("Validating...")
        val_loss, val_prec1, val_prec5 = validate(
            args, val_loader, model, criterion, device
        )

        if args.wandb:
            wandb.log({
                "val_loss": float(val_loss),
                "val_prec1": float(val_prec1),
                "val_prec5": float(val_prec5)
            })

        

    # Case 1: Gaussian noise
    elif args.mode == "Gaussian":
        calibrate_data = torch.randn((args.calib_batchsize, 3, 224, 224)).to(device)
        print("Calibrating with Gaussian noise...")
        with torch.no_grad():
            output = model(calibrate_data)
        # Freeze model
        model.model_quant()
        model.model_freeze()

        # Validate the quantized model
        print("Validating...")
        val_loss, val_prec1, val_prec5 = validate(
            args, val_loader, model, criterion, device
        )

    # Not implemented
    else:
        raise NotImplementedError

def validate(args, val_loader, model, criterion, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    val_start_time = end = time.time()
    for i, (data, target) in enumerate(val_loader):
        target = target.to(device)
        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(data)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    val_end_time = time.time()
    print(" * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {time:.3f}".format(
        top1=top1, top5=top5, time=val_end_time - val_start_time))

    return losses.avg, top1.avg, top5.avg


class AverageMeter(object):
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


if __name__ == "__main__":
    main()
