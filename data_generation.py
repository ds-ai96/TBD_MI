import os
import sys
import time
import wandb
import random
import shutil
import argparse
from tqdm import tqdm

import numpy as np


from quantization import *
from synthesis import SMI
from utils import *

def get_args_parser():
    parser = argparse.ArgumentParser(description="Data Generation", add_help=False)
    
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
        choices=["DMI", "SMI", "Gaussian", "Proposed"],
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
    parser.add_argument(
        "--run_name",
        type=str,
        help="name of the run"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="interval to log results"
    )

    return parser.parse_args()

def get_model(name):
    model_name = {
        'deit_tiny_16_imagenet': 'deit_tiny_patch16_224',
        'deit_base_16_imagenet': 'deit_base_patch16_224',
    }

    if name not in model_name:
        raise ValueError(f"Invalid model name: {name}")

    if name.split("_")[-1]=='imagenet':
        model = build_model(model_name[name], Pretrained=True)
    else:
        raise NotImplementedError

    return model

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
    # Parse arguments
    args = get_args_parser()

    # Logging with wandb
    if args.wandb:
        # Initialize wandb
        wandb.init(
            project=args.project_name,
            name=args.run_name,
        )
    
    print(f"Arguments: {args}")
    
    # Set Environments
    set_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Single GPU

    # Build model
    model=get_model(args.model).to(device)
    model.eval()

    # Build dataloader
    _, _, num_classes, train_transform, _, normalizer = build_dataset(
        args.model.split("_")[0], args.model.split("_")[-1], args.calib_batchsize,
        train_aug=False, keep_zero=True, train_inverse=True, dataset_path=args.dataset
    )

    #########################################################
    # Generate synthetic data
    #########################################################

    # Mode 0: DMI & SMI
    if args.mode in ["DMI", "SMI"]:
        iterations=args.iterations # Total number of iterations for inversion
        lr_g=0.25 # Learning rate for inversion
        
        # Coefficient for Inversion
        adv=0 # Coefficient of adversarial regularization, we do not use it in our work
        bn=0.0 # Coefficient of batch normalization regularization, dose not apply to ViTs due to the absence of batch normalization, we borrow a CNN to only facilitate visualization (refer to GradViT: Gradient Inversion of Vision Transformers)
        oh=1 # Coefficient of classification loss
        tv1=0 # Coefficient of total variance regularization with l1 norm, we do not use it in our work
        tv2=0.0001 # Coefficient of total variance regularization with l2 norm
        l2=0 # Coefficient of l2 norm regularization, we do not use it in our work
        
        # Parameters for SMI
        prune_it = args.prune_it
        prune_ratio = args.prune_ratio

        patch_size = 16 if '16' in args.model else 32
        patch_num = 197 if patch_size==16 else 50

        if args.mode == "DMI":
            img_tag = f"{args.mode}-{iterations}-{str(args.seed)}-{str(args.synthetic_bs*args.num_runs)}"
        elif args.mode == "SMI":
            img_tag = f"{args.mode}-{iterations}-{str(prune_it)}-{str(prune_ratio)}-{str(args.seed)}-{str(args.synthetic_bs*args.num_runs)}"
        else:
            raise NotImplementedError
        datapool_path=os.path.join(args.datapool, f"{args.model}/{img_tag}") # The path to store inverted data
        if os.path.exists(datapool_path):
            shutil.rmtree(datapool_path)
            print(f"Removed existing data at {datapool_path}")

        synthesizer = SMI(
            teacher=model, teacher_name=args.model, student=model, num_classes=num_classes,
            img_shape=(3, 224, 224), iterations=iterations, patch_size=patch_size, lr_g=lr_g,
            synthesis_batch_size=args.synthetic_bs, sample_batch_size=args.calib_batchsize,
            adv=adv, bn=bn, oh=oh, tv1=tv1,tv2=tv2, l2=l2,
            save_dir=datapool_path, transform=train_transform,
            normalizer=normalizer, device=device, bnsource='resnet50v1', init_dataset=None
        )

        print(f"Generating data to {datapool_path}...")


        # Inversion Process
        calibrate_data = []
        targets = []

        for run_idx in tqdm(range(args.num_runs), desc="Synthesizing"):
            start = time.time()
            results = synthesizer.synthesize(
                num_patches=patch_num,
                prune_it=prune_it,
                prune_ratio=prune_ratio
            )

            elapsed = time.time() - start
            print(f"[Run {run_idx+1}/{args.num_runs}] Time: {elapsed:.2f}s")

        
            minibatch = synthesizer.sample()
            calibrate_data.append(minibatch)
            if "target" in results:
                targets.append(results["target"])
        
        calibrate_data = torch.cat(calibrate_data, dim=0)
        targets = torch.cat(targets, dim=0) if targets else torch.empty(0, dtype=torch.long)

    # TODO: Mode 1: Proposed MI
    elif args.mode == "Proposed":
        raise NotImplementedError

    # Mode 2: Gaussian noise
    elif args.mode == "Gaussian":
        calibrate_data = torch.randn((args.calib_batchsize, 3, 224, 224)).to(device)

    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
