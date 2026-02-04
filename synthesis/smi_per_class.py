import random
import time
from math import sqrt
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from ._utils import ImagePool, UnlabeledImageDataset, DataIter  # type: ignore
from .base import BaseSynthesis
from .hooks import DeepInversionHook

def Gaussian_low_pass_filter(images, cutoff_ratio=0.8):
    B, C, H, W =images.shape

    fft_images = torch.fft.fftshift(
        torch.fft.fft2(images, dim=(-2, -1)),
        dim=(-2, -1)
    )

    Y, X = torch.meshgrid(
        torch.arange(H, device=images.device),
        torch.arange(W, device=images.device),
        indexing='ij'
    )

    freq_grid = torch.stack([Y - H // 2, X - W // 2], dim=-1).float()
    distance = torch.norm(freq_grid, dim=-1)

    max_dist = distance.max()
    sigma_f = cutoff_ratio * max_dist

    gaussian_mask = torch.exp(-(distance ** 2) / (2 * sigma_f ** 2))
    gaussian_mask = gaussian_mask[None, None, :, :] # broadcast

    filtered_fft = fft_images * gaussian_mask
    filtered = torch.fft.ifft2(
        torch.fft.ifftshift(filtered_fft, dim=(-2, -1)),
        dim=(-2, -1)
    ).real

    return filtered

def _clip_images(image_tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    mean_arr = np.array(mean)
    std_arr = np.array(std)
    for c in range(3):
        m, s = mean_arr[c], std_arr[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def _get_image_prior_losses(inputs_jit: torch.Tensor):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
        diff3.abs() / 255.0
    ).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2


def _jitter_and_flip(inputs_jit: torch.Tensor, lim: float = 1.0 / 8.0, do_flip: bool = True):
    lim_0, lim_1 = int(inputs_jit.shape[-2] * lim), int(inputs_jit.shape[-1] * lim)
    off1 = random.randint(-lim_0, lim_0)
    off2 = random.randint(-lim_1, lim_1)
    inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))
    flip = random.random() > 0.5
    if flip and do_flip:
        inputs_jit = torch.flip(inputs_jit, dims=(3,))
    return inputs_jit, off1, off2, flip and do_flip


def _jitter_and_flip_index(
    pre_index_matrix: torch.Tensor, off1: int, off2: int, flip: bool, patch_size: int = 16, num_patches_per_dim: int = 14
) -> torch.Tensor:
    off1_int, off1_frac = int(off1 // patch_size), off1 % patch_size / patch_size
    off2_int, off2_frac = int(off2 // patch_size), off2 % patch_size / patch_size
    patch_indices = torch.arange(1, num_patches_per_dim * num_patches_per_dim + 1).reshape(
        num_patches_per_dim, num_patches_per_dim
    ).to(pre_index_matrix.device)
    patch_indices = torch.roll(patch_indices, shifts=(off1_int, off2_int), dims=(0, 1))
    if abs(off1_frac) >= 0.5:
        direction = 1 if off1_frac > 0 else -1
        patch_indices = torch.roll(patch_indices, shifts=(direction, 0), dims=(0, 1))
    if abs(off2_frac) >= 0.5:
        direction = 1 if off2_frac > 0 else -1
        patch_indices = torch.roll(patch_indices, shifts=(0, direction), dims=(0, 1))
    if flip:
        patch_indices = torch.flip(patch_indices, dims=[1])
    flat_patch_indices = patch_indices.flatten()
    non_zero_mask = pre_index_matrix != 0
    indices = (flat_patch_indices == pre_index_matrix[non_zero_mask].unsqueeze(-1)).nonzero(as_tuple=True)
    rows = indices[1] // num_patches_per_dim
    cols = indices[1] % num_patches_per_dim
    new_indices = rows * num_patches_per_dim + cols + 1
    new_index_matrix = torch.zeros_like(pre_index_matrix)
    new_index_matrix[non_zero_mask] = new_indices
    return new_index_matrix


def _get_top_k_relative_indices_including_first(pre_attention: torch.Tensor, k: int) -> torch.Tensor:
    batch_size, n = pre_attention.shape
    k = min(k, n)
    remaining_attention = pre_attention
    top_values, top_indices = torch.topk(remaining_attention, k, dim=1)
    _ = top_values  # silence lint
    top_indices_adjusted = top_indices + 1
    first_index = torch.zeros((batch_size, 1), dtype=torch.long, device=pre_attention.device)
    return torch.cat((first_index, top_indices_adjusted), dim=1)


class _Timer:
    def __init__(self):
        self.o = time.time()

    def measure(self, p: int = 1) -> str:
        x = (time.time() - self.o) / p
        if x >= 3600:
            return "{:.1f}h".format(x / 3600)
        if x >= 60:
            return "{}m".format(round(x / 60))
        return "{:.2f}s".format(x)


class SMIClasswise(BaseSynthesis):
    """A self-contained synthesis class for generating per-class ImageNet images without touching the original SMI code."""

    def __init__(
        self,
        teacher,
        teacher_name: str,
        student,
        num_classes: int,
        img_shape=(3, 224, 224),
        patch_size: int = 16,
        iterations: int = 2000,
        lr_g: float = 0.25,
        synthesis_batch_size: int = 128,
        adv: float = 0.0,
        bn: float = 0.0,
        oh: float = 1.0,
        tv1: float = 0.0,
        tv2: float = 1e-5,
        l2: float = 0.0,
        save_dir: Optional[str] = None,
        transform=None,
        normalizer=None,
        device: str = "cpu",
        bnsource: str = "resnet50v2",
    ):
        super().__init__(teacher, student)
        assert len(img_shape) == 3, "image size should be a 3-dimension tuple"

        self.img_size = img_shape
        self.patch_size = patch_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.normalizer = normalizer
        self.transform = transform
        self.synthesis_batch_size = synthesis_batch_size
        self.data_pool = ImagePool(root=save_dir or "./smi_classwise_pool")

        self.bn = bn
        if self.bn != 0:
            if bnsource == "resnet50v2":
                self.prior = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).cuda(
                    device
                )
            elif bnsource == "resnet50v1":
                self.prior = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).cuda(
                    device
                )
            else:
                raise NotImplementedError
            self.prior.eval()
            self.prior.cuda()
        self.adv = adv
        self.oh = oh
        self.tv1 = tv1
        self.tv2 = tv2
        self.l2 = l2
        self.num_classes = num_classes
        self.device = device

        if self.bn != 0:
            self.bn_hooks = []
            for m in self.prior.modules():
                if isinstance(m, nn.BatchNorm2d):
                    self.bn_hooks.append(DeepInversionHook(m))
            assert len(self.bn_hooks) > 0, "input model should contains at least one BN layer for DeepInversion"
        self.teacher_name = teacher_name
        self.timer = _Timer()

    def synthesize(
        self, targets=None,
        num_patches=197, prune_it=[-1], prune_ratio=[0],
        lpf=False, lpf_type="cufoff", lpf_every=10, cutoff_ratio=0.8,
        use_soft_label=False, soft_label_alpha=0.9,
        ):

        # Idea 1. Low-pass Filter
        self.lpf = lpf
        self.lpf_type = lpf_type
        self.lpf_every = lpf_every
        self.cutoff_ratio = cutoff_ratio

        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6
        if targets is not None and not torch.is_tensor(targets):
            targets = torch.tensor(targets)
        batch_size = targets.shape[0] if targets is not None else self.synthesis_batch_size
        inputs = torch.randn(size=[batch_size, *self.img_size], device=self.device).requires_grad_()
        if targets is None:
            targets = torch.randint(low=0, high=self.num_classes, size=(batch_size,))
            targets = targets.sort()[0]
        targets = targets.to(self.device)

        if use_soft_label and targets.dim() == 1:
            alpha = float(soft_label_alpha)
            if not 0.0 <= alpha <= 1.0:
                raise ValueError("soft_label_alpha must be between 0 and 1.")
            off_value = (1.0 - alpha) / (self.num_classes - 1)
            targets_soft = torch.full(
                (targets.size(0), self.num_classes),
                off_value,
                device=self.device,
                dtype=inputs.dtype,
            )
            targets_soft.scatter_(1, targets.view(-1, 1), alpha)
            targets = targets_soft

        optimizer = torch.optim.Adam([inputs], self.lr_g, betas=[0.5, 0.99])

        best_inputs = inputs.data

        current_abs_index = torch.LongTensor(list(range(num_patches))).repeat(best_inputs.shape[0], 1).to(self.device)
        next_relative_index = torch.LongTensor(list(range(num_patches))).repeat(best_inputs.shape[0], 1).to(self.device)
        inputs_aug = inputs
        for it in tqdm(range(self.iterations), desc=f"Synth {self.teacher_name}", leave=False):
            if it + 1 in prune_it:
                inputs_aug = inputs
                current_abs_index_aug = current_abs_index

                if self.lpf and ((it + 1) % self.lpf_every == 0):
                    if self.lpf_type == "gaussian":
                        inputs_aug = Gaussian_low_pass_filter(inputs_aug, cutoff_ratio=self.cutoff_ratio)
                    elif self.lpf_type == "cutoff":
                        inputs_aug = low_pass_filter(inputs_aug, cutoff_ratio=self.cutoff_ratio)
                    elif self.lpf_type == "bilateral":
                        inputs_aug = Bilateral_loss_pass_filter(inputs_aug, kernel_size=self.bi_kernel, sigma_s=self.bi_sigma_s, sigma_r=self.bi_sigma_r)
                    else:
                        raise ValueError("Invalid lpf_type")

                t_out, attention_weights, _ = self.teacher(inputs_aug, current_abs_index_aug, next_relative_index)
            elif it in prune_it:
                attention_weights = torch.mean(attention_weights[-1], dim=1)[:, 0, :][:, 1:]
                prune_ratio_value = list(prune_ratio)[list(prune_it).index(it)]
                top_k = int(attention_weights.shape[1] * (1.0 - prune_ratio_value))
                next_relative_index = _get_top_k_relative_indices_including_first(pre_attention=attention_weights, k=top_k).to(
                    self.device
                )
                inputs_aug = inputs
                current_abs_index_aug = current_abs_index

                if self.lpf and ((it + 1) % self.lpf_every == 0):
                    if self.lpf_type == "gaussian":
                        inputs_aug = Gaussian_low_pass_filter(inputs_aug, cutoff_ratio=self.cutoff_ratio)
                    elif self.lpf_type == "cutoff":
                        inputs_aug = low_pass_filter(inputs_aug, cutoff_ratio=self.cutoff_ratio)
                    elif self.lpf_type == "bilateral":
                        inputs_aug = Bilateral_loss_pass_filter(inputs_aug, kernel_size=self.bi_kernel, sigma_s=self.bi_sigma_s, sigma_r=self.bi_sigma_r)
                    else:
                        raise ValueError("Invalid lpf_type")

                t_out, attention_weights, current_abs_index = self.teacher(inputs_aug, current_abs_index_aug, next_relative_index)
            else:
                inputs_aug, off1, off2, flip = _jitter_and_flip(inputs)
                if current_abs_index.shape[1] == num_patches:
                    current_abs_index_aug = current_abs_index
                else:
                    current_abs_index_aug = _jitter_and_flip_index(
                        current_abs_index, off1, off2, flip, self.patch_size, int(224 // self.patch_size)
                    )
                
                if self.lpf and ((it + 1) % self.lpf_every == 0):
                    if self.lpf_type == "gaussian":
                        inputs_aug = Gaussian_low_pass_filter(inputs_aug, cutoff_ratio=self.cutoff_ratio)
                    elif self.lpf_type == "cutoff":
                        inputs_aug = low_pass_filter(inputs_aug, cutoff_ratio=self.cutoff_ratio)
                    elif self.lpf_type == "bilateral":
                        inputs_aug = Bilateral_loss_pass_filter(inputs_aug, kernel_size=self.bi_kernel, sigma_s=self.bi_sigma_s, sigma_r=self.bi_sigma_r)
                    else:
                        raise ValueError("Invalid lpf_type")
                
                t_out, attention_weights, _ = self.teacher(inputs_aug, current_abs_index_aug, next_relative_index)
            if self.bn != 0:
                _ = self.prior(inputs_aug)
                rescale = [10.0] + [1.0 for _ in range(len(self.bn_hooks) - 1)]
                loss_bn = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(self.bn_hooks)])
            else:
                loss_bn = 0

            if use_soft_label or targets.dim() == 2:
                loss_oh = F.kl_div(
                    F.log_softmax(t_out, dim=1),
                    targets,
                    reduction="batchmean",
                )
            else:
                loss_oh = F.cross_entropy(t_out, targets)
            if self.adv > 0:
                s_out = self.student(inputs_aug)
                loss_adv = -(
                    0.5
                    * F.kl_div(
                        torch.log_softmax(s_out / 3, dim=1),
                        torch.softmax(t_out / 3, dim=1),
                        reduction="batchmean",
                    )
                    + 0.5
                    * F.kl_div(
                        torch.log_softmax(t_out / 3, dim=1),
                        torch.softmax(s_out / 3, dim=1),
                        reduction="batchmean",
                    )
                )
            else:
                loss_adv = loss_oh.new_zeros(1)
            loss_tv1, loss_tv2 = _get_image_prior_losses(inputs)
            loss_l2 = torch.norm(inputs, 2)
            loss = (
                self.bn * loss_bn
                + self.oh * loss_oh
                + self.adv * loss_adv
                + self.tv1 * loss_tv1
                + self.tv2 * loss_tv2
                + self.l2 * loss_l2
            )

            if best_cost > loss.item():
                best_cost = loss.item()
                best_inputs = inputs.data

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            inputs.data = _clip_images(inputs.data, self.normalizer.mean, self.normalizer.std)

        self.student.train()
        if self.normalizer:
            best_inputs = self.normalizer(best_inputs, True)

        with torch.no_grad():
            t_out, attention_weights, current_abs_index = self.teacher(
                best_inputs.detach(),
                torch.LongTensor(list(range(num_patches))).repeat(best_inputs.shape[0], 1).to(self.device),
                torch.LongTensor(list(range(num_patches))).repeat(best_inputs.shape[0], 1).to(self.device),
            )

        attention_weights = torch.mean(attention_weights[-1], dim=1)[:, 0, :][:, 1:]

        def cumulative_mul(lst: Iterable[float]):
            current_mul = 1
            for num in lst:
                current_mul = current_mul * (1.0 - num)
            return current_mul

        top_k = int(num_patches * (cumulative_mul(prune_ratio)))

        next_relative_index = _get_top_k_relative_indices_including_first(pre_attention=attention_weights, k=top_k).to(
            self.device
        )

        mask = torch.zeros(next_relative_index.shape[0], int(sqrt(num_patches)), int(sqrt(num_patches)))
        for b in range(next_relative_index.shape[0]):
            mask[b, (next_relative_index[b][1:] - 1) // int(sqrt(num_patches)), (next_relative_index[b][1:] - 1) % int(sqrt(num_patches))] = 1
        expanded_mask = mask.repeat_interleave(self.patch_size, dim=1).repeat_interleave(self.patch_size, dim=2)
        expanded_mask = expanded_mask.to(self.device)
        masked_best_inputs = best_inputs * expanded_mask.unsqueeze(1)
        targets_np = targets.cpu().detach().numpy()
        if len(list(prune_ratio)) == 1 and list(prune_ratio)[0] == 0:
            self.data_pool.add(best_inputs)
        else:
            self.data_pool.add(masked_best_inputs)

        dst = self.data_pool.get_dataset(transform=self.transform)
        if self.transform is not None:
            loader = torch.utils.data.DataLoader(
                dst, batch_size=self.synthesis_batch_size, shuffle=False, num_workers=0, pin_memory=True
            )
            self.data_iter = DataIter(loader)

        return {"synthetic": best_inputs, "masked_synthetic": masked_best_inputs, "targets": targets_np}

    def sample(self, n):
        raise NotImplementedError("SMIClasswise is intended for direct synthesis, not sampling.")