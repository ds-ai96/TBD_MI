import os
import sys
import copy
import time
import wandb
import random
import shutil
import argparse
from tqdm import tqdm

import numpy as np
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.ao.quantization as ao_quant
from torch.utils.data import DataLoader

from utils import *
from utils.data_utils import build_dataset
from synthesis import SMI, TBD_MI_CNN
from quantization_utils.quant_modules import QuantAct, QuantLinear, QuantConv2d

from pytorchcv.models.resnet import ResUnit
from pytorchcv.models.mobilenet import DwsConvBlock
from pytorchcv.models.mobilenetv2 import LinearBottleneck

MODEL_CONFIGS = {
    "resnet20_cifar10": {
        "dataset": "cifar10",
        "num_classes": 10,
        "ptcv_name": "resnet20_cifar10",
        "pretrained": True,
    },
    "resnet20_cifar100": {
        "dataset": "cifar100",
        "num_classes": 100,
        "ptcv_name": "resnet20_cifar100",
        "pretrained": True,
    },
    "resnet18_imagenet": {
        "dataset": "imagenet",
        "num_classes": 1000,
        "ptcv_name": "resnet18",
        "pretrained": True,
    },
    "resnet50_imagenet": {
        "dataset": "imagenet",
        "num_classes": 1000,
        "ptcv_name": "resnet50",
        "pretrained": True,
    },
    "mobilenetv2_imagenet": {
        "dataset": "imagenet",
        "num_classes": 1000,
        "ptcv_name": "mobilenetv2_w1",
        "pretrained": True,
    },
}

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
        choices=["DMI", "TBD_MI", "Gaussian"],
        help="mode to use for data generation"
    )
    parser.add_argument(
        "--model",
        default="resnet20_cifar10",
        choices=[
            "resnet20_cifar10",
            "resnet20_cifar100",
            "resnet18_imagenet",
            "resnet50_imagenet",
            "mobilenetv2_imagenet"
        ],
        help="model and dataset combination"
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
        "--quant_mode",
        type=str,
        choices=["qat", "ptq"],
        help="quantization mode"
    )
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

    # TBD-MI
    ### Idea 1. Low-Pass Filtering
    parser.add_argument(
        "--lpf",
        action="store_true",
        help="use low-pass filtering"
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
    
    ### Idea 3. Soft Label
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

    # Observation
    ## saliency map & sparsification 위치에 대한 실험
    parser.add_argument(
        "--saliency_anchor",
        type=str,
        default="c",
        choices=["se", "sw", "ne", "nw", "e", "w", "s", "n", "seh", "swh", "neh", "nwh", "eh", "wh", "sh", "nh", "c"],
        help="anchor for saliency map centering"
    )

    # SynQ QAT Hyperparameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=4.0,
        help="temperature for KD loss"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="alpha for KD loss (scaling factor for KL divergence)"
    )
    parser.add_argument(
        "--lambda_ce",
        type=float,
        default=0.1,
        help="weight for cross-entropy loss in QAT"
    )
    parser.add_argument(
        "--lambda_fa",
        type=float,
        default=1.0,
        help="weight for feature alignment loss in QAT"
    )
    parser.add_argument(
        "--qat_epochs",
        type=int,
        default=100,
        help="number of QAT epochs"
    )
    parser.add_argument(
        "--qat_lr",
        type=float,
        default=1e-4,
        help="learning rate for QAT"
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

def build_cnn_model(model_key, num_classes):
    config = MODEL_CONFIGS[model_key]

    model_name = config["ptcv_name"]
    pretrained = bool(config.get("pretrained", True))

    model = ptcv_get_model(model_name, pretrained=pretrained)
    # model = _adjust_classifier_if_needed(model, num_classes)
    return model

def quantize_model(model, w_bit, a_bit):
    if isinstance(model, nn.Conv2d):
        quant_mod = QuantConv2d(weight_bit=w_bit)
        quant_mod.set_param(model)
        return quant_mod
    if isinstance(model, nn.Linear):
        quant_mod = QuantLinear(weight_bit=w_bit)
        quant_mod.set_param(model)
        return quant_mod
    if isinstance(model, (nn.ReLU, nn.ReLU6)):
        return nn.Sequential(*[model, QuantAct(activation_bit=a_bit)])
    if isinstance(model, nn.Sequential):
        mods = []
        for _, m in model.named_children():
            mods.append(quantize_model(m, w_bit, a_bit))
        return nn.Sequential(*mods)

    q_model = copy.deepcopy(model)
    for attr in dir(model):
        mod = getattr(model, attr)
        if isinstance(mod, nn.Module) and 'norn' not in attr:
            setattr(q_model, attr, quantize_model(mod, w_bit, a_bit))

    return q_model

# TODO: 이거 왜 필요하지??
@torch.no_grad()
def calibrate_act_range(q_model, calib_loader, device, num_batches):
    q_model.eval()
    cnt = 0
    for data, _ in calib_loader:
        data = data.to(device, non_blocking=True)
        _ = q_model(data)
        cnt += 1
        if cnt >= num_batches:
            break

def freeze_model(model):
    if isinstance(model, QuantAct):
        model.fix()
    elif isinstance(model, nn.Sequential):
        for _, m in model.named_children():
            freeze_model(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                freeze_model(mod)

def unfreeze_model(model):
    if isinstance(model, QuantAct):
        model.unfix()
    elif isinstance(model, nn.Sequential):
        for _, m in model.named_children():
            unfreeze_model(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                unfreeze_model(mod)

def channel_attention(x):
    return F.normalize(x.pow(2).mean([2, 3]).view(x.size(0), -1))


def loss_fn_kd(output, labels, teacher_outputs, temperature, alpha, kl_loss_fn, ce_loss_fn):
    a = F.log_softmax(output / temperature, dim=1) + 1e-7
    b = F.softmax(teacher_outputs / temperature, dim=1)
    c = alpha * temperature * temperature
    loss_kl = kl_loss_fn(a, b) * c
    loss_ce = ce_loss_fn(output, labels)
    return loss_kl, loss_ce

def compute_loss_fa(activation_student, activation_teacher, lam):
    fa = torch.zeros(1, device=activation_student[0].device)
    for l, lth_activation in enumerate(activation_student):
        fa += (lth_activation - activation_teacher[l]).pow(2).mean()
    return lam * fa


class ActivationHooks:
    """Activation hooks for capturing intermediate activations."""
    def __init__(self):
        self.activations = []
        self.handles = []
    
    def hook(self, module, _input, output):
        self.activations.append(channel_attention(output.clone()))
    
    def clear(self):
        self.activations.clear()
    
    def register_hooks(self, model):
        """Register hooks on ResUnit, DwsConvBlock, LinearBottleneck modules."""
        for m in model.modules():
            if isinstance(m, ResUnit):
                h = m.body.register_forward_hook(self.hook)
                self.handles.append(h)
            elif isinstance(m, DwsConvBlock):
                h = m.pw_conv.bn.register_forward_hook(self.hook)
                self.handles.append(h)
            elif isinstance(m, LinearBottleneck):
                h = m.conv3.register_forward_hook(self.hook)
                self.handles.append(h)
    
    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def qat_training(
    q_model, teacher, train_loader, val_loader, criterion, device, epochs,
    lr, weight_decay, temperature, alpha, lambda_ce, lambda_fa
):
    teacher.eval()
    
    # Loss functions
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean').to(device)
    ce_loss_fn = nn.CrossEntropyLoss().to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(q_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Register activation hooks for feature alignment (separate instances for teacher and student)
    teacher_hooks = ActivationHooks()
    student_hooks = ActivationHooks()
    teacher_hooks.register_hooks(teacher)
    student_hooks.register_hooks(q_model)

    for epoch in range(epochs):
        if epoch < 4:
            unfreeze_model(q_model)
        else:
            freeze_model(q_model)

        q_model.train()
        for x, labels in train_loader:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            teacher_hooks.clear()
            student_hooks.clear()
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = q_model(x)  # This will populate student activations
            
            # Compute losses
            loss_kl, loss_ce = loss_fn_kd(
                s_logits, labels, t_logits, temperature, alpha, kl_loss_fn, ce_loss_fn
            )
            loss_fa = compute_loss_fa(
                student_hooks.activations, teacher_hooks.activations, lambda_fa
            )
            
            loss = loss_kl + lambda_ce * loss_ce + loss_fa
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"[QAT] Epoch {epoch+1}/{epochs} | loss_kl: {loss_kl.item():.4f} | "
              f"loss_ce: {loss_ce.item():.4f} | loss_fa: {loss_fa.item():.4f}")
        
        # Validation every N epochs
        if (epoch + 1) % 10 == 0:
            q_model.eval()
            val_loss, val_prec1, val_prec5 = _validate_qat(val_loader, q_model, criterion, device)
            print(f"[QAT Validation] Epoch {epoch+1}/{epochs} | "
                  f"val_loss: {val_loss:.4f} | Prec@1: {val_prec1:.2f}% | Prec@5: {val_prec5:.2f}%")
    
    # Cleanup hooks
    teacher_hooks.remove_hooks()
    student_hooks.remove_hooks()


@torch.no_grad()
def _validate_qat(val_loader, model, criterion, device):
    """Internal validation function for QAT training."""
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    model.eval()
    for data, target in val_loader:
        data = data.to(device)
        target = target.to(device)
        
        output = model(data)
        loss = criterion(output, target)
        
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))
    
    return losses.avg, top1.avg, top5.avg

def main():
    args = get_args_parser()

    if args.wandb:
        if args.mode == "DMI":
            run_name = f"{args.mode}-{args.model}-{args.iterations}-{args.seed}-W{args.w_bit}A{args.a_bit}"
        elif args.mode == "TBD_MI":
            run_name = f"{args.mode}-{args.model}-{args.iterations}-{'-'.join(map(str, args.prune_it))}-{'-'.join(map(str, args.prune_ratio))}-{args.seed}-W{args.w_bit}A{args.a_bit}"
            if args.lpf:
                run_name += f"-LPF{args.lpf_every}/{args.cutoff_ratio}"
            if args.sc_center:
                run_name += f"-SC{args.sc_every}/{args.sc_center_lambda}"
            if args.use_soft_label:
                run_name += f"-SL{args.soft_label_alpha}"
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
    
    model_config = MODEL_CONFIGS[args.model]
    # TODO: 이거 필요없어?
    # Load bit-config
    # cfg = Config(args.w_bit, args.a_bit)

    # Build FP models (teacher/student)
    model = build_cnn_model(args.model, model_config["num_classes"]).to(device)
    teacher = build_cnn_model(args.model, model_config["num_classes"]).to(device).eval()

    if model_config["dataset"] == "imagenet":
        img_shape = (3, 224, 224)
    elif model_config["dataset"] in ["cifar10", "cifar100"]:
        img_shape = (3, 32, 32)
    else:
        raise NotImplementedError


    # Build dataloaders for validation ( + transforms/normalizer for synthetisis pipeline)
    _, val_loader, num_classes, train_transform, _, normalizer = build_dataset(
        model_type="cnn",
        dataset_type=model_config["dataset"],
        calib_batchsize=args.calib_batchsize,
        train_aug=False,
        keep_zero=True,
        train_inverse=True,
        dataset_path=args.dataset
    )

    #########################################################
    # Model Inversion (Quantization)
    #########################################################
    # Define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(device)

    # Get calibration set
    # Case 0: inversion
    if args.mode == "DMI":
        raise NotImplementedError

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

        img_tag = run_name
        datapool_path=os.path.join(args.datapool,'%s/%s'%(args.model,img_tag)) # The path to store inverted data
        if os.path.exists(datapool_path):
            shutil.rmtree(datapool_path)
            print(f"Removed existing data at {datapool_path}")

        synthesizer = TBD_MI_CNN(
            teacher=teacher, teacher_name=args.model, student=model, num_classes=num_classes,
                img_shape=img_shape, iterations=iterations, lr_g=lr_g,
                synthesis_batch_size=args.synthetic_bs, sample_batch_size=args.calib_batchsize,
                adv=adv, bn=bn, oh=oh, tv1=tv1,tv2=tv2, l2=l2,
                save_dir=datapool_path, transform=train_transform,
                normalizer=normalizer, device=device, bnsource='resnet50v1', init_dataset=None
        )

        print(f"Generating data to {datapool_path}...")
        for run_idx in tqdm(range(args.num_runs), desc="Synthesizing"):
            start = time.time()
            results = synthesizer.synthesize(
                prune_it=prune_it, prune_ratio=prune_ratio,
                lpf=args.lpf, lpf_every=args.lpf_every, cutoff_ratio=args.cutoff_ratio,
                sc_center=args.sc_center, sc_every=args.sc_every, sc_center_lambda=args.sc_center_lambda,
                saliency_anchor=args.saliency_anchor,
                use_soft_label=args.use_soft_label, soft_label_alpha=args.soft_label_alpha
            )
            if args.wandb and 'targets' in results:
                wandb.log({"targets": results['targets']}, step=args.seed)

            elapsed = time.time() - start
            print(f"[Run {run_idx+1}/{args.num_runs}] Time: {elapsed:.2f}s")

        dst = synthesizer.data_pool.get_dataset(transform=train_transform)

        # Quantization
        if args.quant_mode == "qat":
            qat_loader = DataLoader(
                dst,
                batch_size=args.calib_batchsize,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                drop_last=True
            )

            q_model = quantize_model(model, w_bit=args.w_bit, a_bit=args.a_bit).to(device)

            def freeze_bn(m):
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                    m.eval()
            q_model.apply(freeze_bn)

            unfreeze_model(q_model)
            q_model.eval()
            calibrate_act_range(q_model, qat_loader, device, num_batches=20)
            freeze_model(q_model)

            mins, maxs = [], []
            for m in q_model.modules():
                if isinstance(m, QuantAct):
                    mins.append(float(m.x_min))
                    maxs.append(float(m.x_max))
            print("After calib:", (min(mins), max(maxs)))

            qat_training(
                q_model=q_model, teacher=teacher,
                train_loader=qat_loader, val_loader=val_loader,
                criterion=criterion, device=device,
                epochs=args.qat_epochs,
                lr=args.qat_lr, weight_decay=0.0,
                temperature=args.temperature, alpha=args.alpha,
                lambda_ce=args.lambda_ce, lambda_fa=args.lambda_fa
            )


        elif args.quant_mode == "ptq":
            pass
        
        else:
            raise NotImplementedError

        print(f"[Calibration] Total synthetic images in pool: {len(dst)}")

        calib_loader = DataLoader(
            dst,
            batch_size=args.calib_batchsize,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        # print(f"Calibrating with generated data (full pool pass)...")
        # quantized_model = quantize_model(model, calib_loader)

        # Validate the quantized model
        print("Validating...")
        val_loss, val_prec1, val_prec5 = validate(
            args, val_loader, q_model, criterion, device
        )

        if args.wandb:
            wandb.log({
                "val_loss": float(val_loss),
                "val_prec1": float(val_prec1),
                "val_prec5": float(val_prec5)
            })

    elif args.mode == "Gaussian":
        raise NotImplementedError

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
