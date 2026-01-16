import argparse
import shutil
import time
import os
import sys
import random

import numpy as np
import wandb
import torch.nn.functional as F

from synthesis import SMI, TBD_MI
from utils import *
from utils.data_utils import find_non_zero_patches, NORMALIZE_DICT

def get_args_parser():
    parser = argparse.ArgumentParser(description="Model Inversion (Knowledge Transfer)", add_help=False)
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
        default="deit_tiny_16_cifar10",
        choices=["deit_tiny_16_cifar10", "deit_base_16_cifar10", "deit_tiny_16_cifar100", "deit_base_16_cifar100"],
        help="model to use for data generation"
        )
    parser.add_argument(
        '--dataset',
        default="/path/to/dataset",
        help='path to dataset'
        )
    parser.add_argument(
        '--model_path',
        default="/path/to/model",
        help='path to model'
        )
    parser.add_argument(
        '--datapool',
        default="/path/to/datapool",
        help='path to datapool'
        )

    # Knowledge Transfer
    parser.add_argument(
        "--kr-batchsize",
        default=32,
        type=int,
        help="batchsize of knowledge transfer"
        )
    parser.add_argument(
        "--val-batchsize", 
        default=200,
        type=int, 
        help="batchsize of validation set"
        )
    parser.add_argument(
        "--num-workers", 
        default=4, 
        type=int,
        help="number of data loading workers (default: 16)"
        )
    parser.add_argument(
        "--epoches", 
        default=1000, 
        type=int, 
        help="number of epoches for knowledge transfer"
        )

    # Data Generation
    parser.add_argument(
        '--prune_it', 
        nargs='+', 
        type=int,
        default=[-1],
        help='the iteration indexes for inversion stopping; -1: to densely invert data; t1 t2 ... tn: to sparsely invert data and perform inversion stopping at t1, t2, ..., tn'
        )
    parser.add_argument(
        '--prune_ratio', 
        nargs='+', 
        type=float,
        default=[0],
        help='the proportion of patches to be pruned relative to the current remaining patches; 0: to densely invert data; r1 r2 ... rn: progressively stopping the inversion of a fraction (r1, r2, ..., rn)$$ of patches at iterations (t1, t2, ..., tn), respectively'
        )

    # TODO: TBD-MI
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

    ### Idea 3. Soft label
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

def kldiv( logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax( targets/T, dim=1 )
    return F.kl_div( q, p, reduction=reduction ) * (T*T)

class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)


def get_teacher(name, path):
    teacher_name = {
                    'deit_tiny_16_cifar10': 'deit_tiny_patch16_224',
                    'deit_base_16_cifar10': 'deit_base_patch16_224',
                    'deit_tiny_16_cifar100': 'deit_tiny_patch16_224',
                    'deit_base_16_cifar100': 'deit_base_patch16_224',
                    }
    dataset_name=name.split('_')[-1]
    if dataset_name=='cifar10':
        num_classes=10
    elif dataset_name=='cifar100':
        num_classes=100
    else:
        raise NotImplementedError
    teacher = build_model(teacher_name[name], Pretrained=False)
    teacher.reset_classifier(num_classes)
    teacher.load_state_dict(torch.load(os.path.join(path,'{}.pth'.format(name))))
    return teacher

def get_student(name,fine_tuning_method):
    model_zoo = {'deit_tiny_16_cifar10': 'deit_tiny_patch16_224',
                 'deit_base_16_cifar10': 'deit_base_patch16_224',
                 'deit_tiny_16_cifar100': 'deit_tiny_patch16_224',
                 'deit_base_16_cifar100': 'deit_base_patch16_224',
                 }
    student = build_model(model_zoo[name], Pretrained=True)
    dataset_name = name.split('_')[-1]
    if dataset_name == 'cifar10':
        num_classes = 10
    elif dataset_name == 'cifar100':
        num_classes = 100
    else:
        raise NotImplementedError
    if fine_tuning_method=='linear':
        for param in student.parameters():
            param.requires_grad = False
        student.reset_classifier(num_classes)
    else:
        raise NotImplementedError
    return student

def seed(seed=0):
    sys.setrecursionlimit(100000)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def main():
    args = get_args_parser()
    
    # Set WandB
    if args.wandb:
        if args.mode == "SMI":
            run_name = f"{args.mode}-{args.model}-{'-'.join(map(str, args.prune_it))}-{'-'.join(map(str, args.prune_ratio))}-{args.seed}" 
        elif args.mode == "DMI":
            run_name = f"{args.mode}-{args.model}-{args.seed}" 
        elif args.mode == "TBD_MI":
            run_name = f"{args.mode}-{args.model}-{'-'.join(map(str, args.prune_it))}-{'-'.join(map(str, args.prune_ratio))}-{args.seed}" 
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
            config=vars(args),
        )

    print(f"Arguments: {args}")

    # Set Environments
    seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build model
    model = get_student(args.model, fine_tuning_method='linear')
    model = model.to(device)
    model.train()

    teacher = get_teacher(args.model, args.model_path)
    teacher = teacher.to(device)
    teacher.eval()

    # Build dataloader
    _, val_loader, num_classes, train_transform, _, normalizer = build_dataset(
        args.model.split("_")[0], args.model.split("_")[-1], args.kr_batchsize, 
        train_aug=True, keep_zero=True, train_inverse=True, dataset_path=args.dataset
    )

    #########################################################
    #################### Model Inversion ####################
    #########################################################

    # Inversion setup
    iterations = 4000  # total number of iterations for inversion
    lr_g = 0.25  # learning rate for inversion
    
    # Coefficient for inversion
    adv = 0 # Coefficient of adversarial regularization, we do not use it in our work
    bn = 0.0 # Coefficient of batch normalization regularization, dose not apply to ViTs due to the absence of batch normalization, we borrow a CNN to only facilitate visualization (refer to GradViT: Gradient Inversion of Vision Transformers)
    oh = 1 # Coefficient of classification loss
    tv1 = 0 # Coefficient of total variance regularization with l1 norm, we do not use it in our work
    tv2 = 0.0001 # Coefficient of total variance regularization with l2 norm
    l2 = 0 # Coefficient of l2 norm regularization, we do not use it in our work
    
    # Parameters for DMI/SMI
    prune_it = args.prune_it
    prune_ratio = args.prune_ratio

    patch_size=16 if '16' in args.model else 32
    patch_num=197 if patch_size==16 else 50

    # img_tag = "{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-smi".format(args.model,iterations, lr_g, adv, bn, oh, tv1, tv2, l2,str(prune_it), str(prune_ratio))
    img_tag = run_name

    datapool_path=os.path.join(args.datapool,'%s/%s_aug'%(args.model,img_tag))
    if os.path.exists(datapool_path):
        shutil.rmtree(datapool_path)
        print(f"Removed existing data at {datapool_path}")

    if args.mode in ["DMI", "SMI"]:
        synthesizer = SMI(
            teacher=teacher, teacher_name=args.model, student=model, num_classes=num_classes,
            img_shape=(3, 224, 224), iterations=iterations, patch_size=patch_size, lr_g=lr_g,
            synthesis_batch_size=100, sample_batch_size=args.kr_batchsize,
            adv=adv, bn=bn, oh=oh, tv1=tv1, tv2=tv2, l2=l2,
            save_dir=datapool_path, transform=train_transform,
            normalizer=normalizer, device=device, bnsource='resnet50v1', init_dataset=None
        )
    elif args.mode in ["TBD_MI"]:
        synthesizer = TBD_MI(
            teacher=teacher, teacher_name=args.model, student=model, num_classes=num_classes,
            img_shape=(3, 224, 224), iterations=iterations, patch_size=patch_size, lr_g=lr_g,
            synthesis_batch_size=args.synthetic_bs, sample_batch_size=args.calib_batchsize,
            adv=adv, bn=bn, oh=oh, tv1=tv1,tv2=tv2, l2=l2,
            save_dir=datapool_path, transform=train_transform,
            normalizer=normalizer, device=device, bnsource='resnet50v1', init_dataset=None
        )
    else:
        raise NotImplementedError

    #########################################################
    ################### Knowledge Transfer ##################
    #########################################################

    # Define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(device)

    T = 20
    criterion_kd = KLDiv(T=T)
    optimizer_kd = torch.optim.SGD(model.parameters(), 0.1, weight_decay=1e-4, momentum=0.9)
    best_top1=0

    for epoch in range(args.epoches):
        print("Generating data...")
        _ = synthesizer.synthesize(
            num_patches=patch_num, prune_it=prune_it, prune_ratio=prune_ratio
        )
        calibrate_data = synthesizer.sample()
        calibrate_data = calibrate_data.to(device)
        
        print("Fine-tuning and Knowledge transfer with generated data...")
        next_relative_index = torch.cat([
            torch.zeros(calibrate_data.shape[0], 1, dtype=torch.long).to(calibrate_data.device),
            find_non_zero_patches(images=calibrate_data,patch_size=patch_size)
        ], dim=1)

        with torch.no_grad():
            output_t = teacher(
                calibrate_data,
                current_abs_index=torch.arange(patch_num).repeat(calibrate_data.shape[0], 1).to(calibrate_data.device),
                next_relative_index=next_relative_index
            )

        output_s = model(
            calibrate_data,
            current_abs_index=torch.arange(patch_num).repeat(calibrate_data.shape[0], 1).to(calibrate_data.device),
            next_relative_index=next_relative_index
        )
        
        train_prec1, _ = accuracy(output_s[0].data, torch.max(output_t[0].data, 1)[1], topk=(1, 5))
        loss_cal = criterion_kd(output_s[0], output_t[0].detach())

        optimizer_kd.zero_grad()
        loss_cal.backward()
        optimizer_kd.step()
        
        # Validate the student model
        print(f"********** [Epoch {epoch} Validating... **********")
        val_loss, val_prec1, val_prec5 = validate(
            args, val_loader, model, criterion, device,patch_num
        )
        
        if args.wandb:
            wandb.log({
                "cal_loss": float(loss_cal),
                "train_prec1": float(train_prec1),
                "val_loss": float(val_loss),
                "val_prec1": float(val_prec1),
            })

        if val_prec1>best_top1:
            best_top1=val_prec1
        print(f"best_top1: {best_top1}")
        print(f"********** {img_tag} **********")

def validate(args, val_loader, model, criterion, device,patch_num):
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
            output = model(data,torch.LongTensor(list(range(patch_num))).repeat(data.shape[0], 1).to(device),torch.LongTensor(list(range(patch_num))).repeat(data.shape[0], 1).to(device))
        loss = criterion(output[0], target)

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(output[0].data, target, topk=(1, 5))
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
