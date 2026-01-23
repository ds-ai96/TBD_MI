import time
import random
import math
from math import sqrt

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from .base import BaseSynthesis
from .hooks import DeepInversionHook
from ._utils import UnlabeledImageDataset, DataIter, ImagePool

# Idea 1. Low-pass filter
def Bilateral_loss_pass_filter(images, kernel_size=5, sigma_s=2.0, sigma_r=1.0):
    """
     Differentiable bilateral filter (local window version)

     Args:
        images: Tensor [B, C, H, W]
        kernel_size: spatial window size (odd)
        sigma_s: spatial sigma
        sigma_r: range_sigma
    """

    B, C, H, W = images.shape
    pad = kernel_size // 2

    coords = torch.arange(kernel_size, device=images.device) - pad
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    spatial_kernel = torch.exp(-(xx*2 + yy*2) / (2 * sigma_s**2))
    spatial_kernel = spatial_kernel.view(1, 1, kernel_size, kernel_size)

    patches = F.unfold(
        images,
        kernel_size=kernel_size,
        padding=pad
    ) # [B, C*K*K, H*W]

    patches = patches.view(
        B, C, kernel_size * kernel_size, H * W
    ) # [B, C, K*K, H*W]

    center = images.view(B, C, 1, H * W) # [B, C, 1, H*W]

    range_weight = torch.exp(
        -((patches - center) ** 2) / (2 * sigma_r ** 2)
    )

    spatial_weight = spatial_kernel.view(1, 1, -1, 1)
    weights = spatial_weight * range_weight

    weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)

    filtered = (weights * patches).sum(dim=2) # [B, C, H*W]
    filtered = filtered.view(B, C, H, W)

    return filtered

def Gaussian_low_pass_filter(images, cutoff_ratio=0.8):
    B, C, H, W =images.shape

    fft_images = torch.fft_fftshift(
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

def low_pass_filter(images, cutoff_ratio=0.8):
    B, C, H, W = images.shape
    fft_images = torch.fft.fftshift(torch.fft.fft2(images, dim=(-2, -1)), dim=(-2, -1))

    Y, X = torch.meshgrid(
        torch.arange(H, device=images.device),
        torch.arange(W, device=images.device),
        indexing='ij'
    )
    freq_grid = torch.stack([Y - H // 2, X - W // 2], dim=-1).float()
    distance = torch.norm(freq_grid, dim=-1)
    max_dist = torch.max(distance)
    cutoff_radius = max_dist * cutoff_ratio

    low_pass_mask = (distance <= cutoff_radius).float()
    low_pass_mask_expanded = low_pass_mask.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)
    
    filtered_fft_images = fft_images * low_pass_mask_expanded
    filtered_images = torch.fft.ifft2(torch.fft.ifftshift(filtered_fft_images, dim=(-2, -1)), dim=(-2, -1)).real

    return filtered_images

class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{:.2f}s'.format(x)

def get_top_k_relative_indices_including_first(pre_attention, K):
    batch_size, N = pre_attention.shape
    K = min(K, N)
    remaining_attention = pre_attention
    top_values, top_indices = torch.topk(remaining_attention, K, dim=1)
    top_indices_adjusted = top_indices + 1
    first_index = torch.zeros((batch_size, 1), dtype=torch.long, device=pre_attention.device)
    top_k_indices = torch.cat((first_index, top_indices_adjusted), dim=1)
    return top_k_indices

def clip_images(image_tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor

def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1,loss_var_l2

def jsdiv( logits, targets, T=1.0, reduction='batchmean' ):
    P = F.softmax(logits / T, dim=1)
    Q = F.softmax(targets / T, dim=1)
    M = 0.5 * (P + Q)
    P = torch.clamp(P, 0.01, 0.99)
    Q = torch.clamp(Q, 0.01, 0.99)
    M = torch.clamp(M, 0.01, 0.99)
    return 0.5 * F.kl_div(torch.log(P), M, reduction=reduction) + 0.5 * F.kl_div(torch.log(Q), M, reduction=reduction)

def jitter_and_flip(inputs_jit, lim=1./8., do_flip=True):
    lim_0, lim_1 = int(inputs_jit.shape[-2] * lim), int(inputs_jit.shape[-1] * lim)
    # apply random jitter offsets
    off1 = random.randint(-lim_0, lim_0)
    off2 = random.randint(-lim_1, lim_1)
    inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))
    # Flipping
    flip = random.random() > 0.5
    if flip and do_flip:
        inputs_jit = torch.flip(inputs_jit, dims=(3,))
    return inputs_jit,off1,off2,flip and do_flip

def jitter_and_flip_index(pre_index_matrix, off1, off2, flip, patch_size=16, num_patches_per_dim=14):
    off1_int, off1_frac = int(off1 // patch_size), off1 % patch_size / patch_size
    off2_int, off2_frac = int(off2 // patch_size), off2 % patch_size / patch_size
    patch_indices = torch.arange(1, num_patches_per_dim * num_patches_per_dim + 1).reshape(num_patches_per_dim, num_patches_per_dim).to(pre_index_matrix.device)
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

class TBD_MI(BaseSynthesis):
    def __init__(self, teacher,teacher_name, student, num_classes, img_shape=(3, 224, 224),patch_size=16,
                 iterations=2000, lr_g=0.25,
                 synthesis_batch_size=128, sample_batch_size=128, 
                 adv=0.0, bn=0, oh=1,tv1=0.0, tv2=1e-5, l2=0.0,
                 save_dir='', transform=None,
                 normalizer=None, device='cpu',
                 bnsource='resnet50v2',init_dataset=None):
        super(TBD_MI, self).__init__(teacher, student)
        assert len(img_shape)==3, "image size should be a 3-dimension tuple"

        self.save_dir = save_dir
        self.img_size = img_shape
        self.patch_size=patch_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.normalizer = normalizer
        
        # Data pool
        self.data_pool = ImagePool(root=self.save_dir)
        self.data_iter = None
        self.transform = transform
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.init_dataset=init_dataset

        # Scaling factors
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.tv1 = tv1
        self.tv2 = tv2
        self.l2 = l2
        self.num_classes = num_classes

        if self.bn != 0:
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            if bnsource == 'resnet50v2':
                self.prior = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).cuda(
                    device)
                print(count_parameters(self.prior),'resnet50v2')
            elif bnsource == 'resnet50v1':
                self.prior = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).cuda(
                    device)
                print(count_parameters(self.prior),'resnet50v1')
            else:
                raise NotImplementedError
            self.prior.eval()
            self.prior.cuda()

        # training configs
        self.device = device

        # Idea 3. Sparsification
        self.num_patches_per_dim = int(self.img_size[1] // self.patch_size)
        Hp, Wp = self.num_patches_per_dim, self.num_patches_per_dim

        yy, xx = torch.meshgrid(
            torch.arange(Hp, device=self.device),
            torch.arange(Wp, device=self.device),
        )
        cy, cx = (Hp - 1) / 2.0, (Wp - 1) / 2.0
        r2 = (yy.float() - cy)**2 + (xx.float() - cx)**2

        radial_order_flat = torch.argsort(r2.view(-1))
        self.patch_radial_order = radial_order_flat

        # setup hooks for BN regularization
        if self.bn!=0:
            self.bn_hooks = []
            for m in self.prior.modules():
                if isinstance(m, nn.BatchNorm2d):
                    self.bn_hooks.append( DeepInversionHook(m) )
            assert len(self.bn_hooks)>0, 'input model should contains at least one BN layer for DeepInversion'

    # Idea 2. Saliency Map Centering
    def _center_rad2(self, B):
        """
        각 픽셀의 거리^2을 반환함.
        중심에 가까우면 0, 멀수록 1에 가까움.
        """
        H, W = self.img_size[1], self.img_size[2]
        
        # Use calculated anchor coordinates
        cy = self.anchor_cy
        cx = self.anchor_cx

        y = self._coord_y - cy
        x = self._coord_x - cx
        rad2 = (y*y + x*x) / self._norm_denom

        return rad2.expand(B, 1, H, W)

    def _saliency_p(self, x, targets, current_abs_index, next_relative_index):
        """
        teacher 기준으로 saliency map을 확률분포 p(i,j)로 반환함.
        x: [B, 3, H, W], requires_grad=False 텐서임
        """
        # 이 함수 내에서 requires_grad=True 텐서를 만들어서 사용함.
        x_req = x.clone().requires_grad_(True)

        logits, _, _ = self.teacher(x_req, current_abs_index, next_relative_index)
        if targets.dtype == torch.long and targets.dim() == 1:
            score = logits.gather(1, targets.view(-1, 1)).sum()
        else:
            hard_targets = targets.argmax(dim=1).long()
            score = logits.gather(1, hard_targets.view(-1, 1)).sum()

        grad = torch.autograd.grad(score, x_req, create_graph=True, retain_graph=True)[0]
        sal = grad.abs().amax(dim=1, keepdim=True) # [B, 1, H, W]

        sal_sum = sal.sum(dim=(2, 3), keepdim=True) + 1e-8
        p = sal / sal_sum # [B, 1, H, W]
        return p

    def _sobel_filter(img):
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=img.device).view(1,1,3,3)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=img.device).view(1,1,3,3)
        edges_x = F.conv2d(img, kx, padding=1)
        edges_y = F.conv2d(img, ky, padding=1)
        return torch.sqrt(edges_x**2 + edges_y**2 + 1e-8)

    def synthesize(self, targets=None,
                   num_patches=197, prune_it=[-1], prune_ratio=[0],
                   lpf=False, lpf_type="cutoff", lpf_every=10, cutoff_ratio=0.8,
                   bi_kernel=5, bi_sigma_s=2.0, bi_sigma_r=1.0,
                   sc_center=False, sc_every=50, sc_center_lambda=0.1,
                   saliency_anchor='c', scale_edge=0.0,
                   use_soft_label=False, soft_label_alpha=0.6):

        # Idea 1. Low-pass Filter
        self.lpf = lpf
        self.lpf_type = lpf_type
        self.lpf_every = lpf_every
        self.cutoff_ratio = cutoff_ratio

        ### Bilateral Filter
        self.bi_kernel = bi_kernel
        self.bi_sigma_s = bi_sigma_s
        self.bi_sigma_r = bi_sigma_r

        # Idea 2. Saliency Map Centering
        self.sc_center = sc_center
        self.sc_every = sc_every
        self.sc_center_lambda = sc_center_lambda
        self.saliency_anchor = saliency_anchor

        H, W = self.img_size[1], self.img_size[2]
        self._coord_y = torch.linspace(0, H-1, H, device=self.device).view(1,1,H,1)
        self._coord_x = torch.linspace(0, W-1, W, device=self.device).view(1,1,1,W)
        self._norm_denom = float(H*H + W*W)

        # Calculate Anchor Coordinates
        cy_c, cx_c = (H - 1) / 2.0, (W - 1) / 2.0
        
        anchors = {
            "nw": (0, 0),
            "ne": (0, W-1),
            "sw": (H-1, 0),
            "se": (H-1, W-1),
            "n": (0, cx_c),
            "s": (H-1, cx_c),
            "w": (cy_c, 0),
            "e": (cy_c, W-1),
            "c": (cy_c, cx_c)
        }
        
        # Calculate half-way anchors
        anchors["nwh"] = ((anchors["n"][0] + anchors["w"][0])/2, (anchors["n"][1] + anchors["w"][1])/2) # midpoint of c and nw logic check: (0+cy)/2, (cx+0)/2 ?? Wait. 
        # User defined: seh: midpoint of center and se
        # Center: (cy_c, cx_c)
        # SE: (H-1, W-1)
        # SEH: (cy_c + H-1)/2, (cx_c + W-1)/2
        
        def midpoint(p1, p2):
            return ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2)

        c_point = anchors["c"]
        anchors["seh"] = midpoint(c_point, anchors["se"])
        anchors["swh"] = midpoint(c_point, anchors["sw"])
        anchors["neh"] = midpoint(c_point, anchors["ne"])
        anchors["nwh"] = midpoint(c_point, anchors["nw"])
        anchors["eh"] = midpoint(c_point, anchors["e"])
        anchors["wh"] = midpoint(c_point, anchors["w"])
        anchors["sh"] = midpoint(c_point, anchors["s"])
        anchors["nh"] = midpoint(c_point, anchors["n"])

        if self.saliency_anchor in anchors:
            self.anchor_cy, self.anchor_cx = anchors[self.saliency_anchor]
        else:
            raise ValueError(f"Unknown saliency anchor: {self.saliency_anchor}")

        # Idea 3. Soft label
        self.use_soft_label = use_soft_label
        self.soft_label_alpha = soft_label_alpha

        self.student.eval()
        self.teacher.eval()

        best_cost = 1e6
        inputs = torch.randn(
            size=[self.synthesis_batch_size, *self.img_size],
            device=self.device
        ).requires_grad_()
        if targets is None:
            targets = torch.randint(
                low=0, high=self.num_classes,
                size=(self.synthesis_batch_size,)
            )
            targets = targets.sort()[0] # sort for better visualization
        targets = targets.to(self.device)

        # Idea 3. Soft Label
        if self.use_soft_label:
            alpha = float(self.soft_label_alpha)
            if not 0.0 <= alpha <= 1.0:
                raise ValueError("soft_label_alpha should be between 0.0 and 1.0")
            off_value = (1.0 - alpha) / (self.num_classes - 1)
            targets_soft = torch.full(
                (targets.size(0), self.num_classes),
                off_value,
                device=self.device,
                dtype=inputs.dtype
            )
            targets_soft.scatter_(1, targets.view(-1, 1), alpha)
            targets = targets_soft

        optimizer = torch.optim.Adam([inputs], self.lr_g, betas=[0.5, 0.99])
        best_inputs = inputs.data

        current_abs_index = torch.LongTensor(list(range(num_patches))).repeat(best_inputs.shape[0], 1).to(self.device)
        next_relative_index = torch.LongTensor(list(range(num_patches))).repeat(best_inputs.shape[0], 1).to(self.device)
        inputs_aug = inputs
        for it in tqdm(range(self.iterations)):
            if it+1 in prune_it:
                inputs_aug = inputs
                current_abs_index_aug = current_abs_index

                # Idea 1. Low-pass Filter
                if self.lpf and ((it + 1) % self.lpf_every == 0):
                    if self.lpf_type == "gaussian":
                        inputs_aug = Gaussian_low_pass_filter(inputs_aug, cutoff_ratio=self.cutoff_ratio)
                    elif self.lpf_type == "cutoff":
                        inputs_aug = low_pass_filter(inputs_aug, cutoff_ratio=self.cutoff_ratio)
                    elif self.lpf_type == "bilateral":
                        inputs_aug = Bilateral_loss_pass_filter(inputs_aug, kernel_size=self.bi_kernel, sigma_s=self.bi_sigma_s, sigma_r=self.bi_sigma_r)
                    else:
                        raise ValueError("Invalid lpf_type")

                t_out, attention_weights, _ = self.teacher(inputs_aug, current_abs_index_aug,next_relative_index)

            elif it in prune_it:
                prune_ratio_value = prune_ratio[prune_it.index(it)]

                if self.sc_center:
                    num_spatial = self.num_patches_per_dim * self.num_patches_per_dim
                    top_K = int(num_spatial * (1.0 - prune_ratio_value))
                    top_K = max(1, min(num_spatial, top_K))

                    print(f"top_K (Radial Order): {top_K} ### Iteration: {it}")

                    B = inputs.shape[0]

                    keep_flat_idx = self.patch_radial_order[:top_K]
                    keep_patch_ids = (keep_flat_idx + 1).long()

                    cls_col = torch.zeros(B, 1, dtype=torch.long, device=self.device)
                    keep_col = keep_patch_ids.view(1, -1).expand(B, -1)
                    next_relative_index = torch.cat([cls_col, keep_col], dim=1)
                else:
                    attention_weights = torch.mean(attention_weights[-1], dim=1)[:, 0, :][:, 1:]  # (B,heads,N,N)->(B,p-1)
                    prune_ratio_value = prune_ratio[prune_it.index(it)]
                    top_K=int(attention_weights.shape[1] * (1.0 - prune_ratio_value))
                    print('top_K:',top_K,'###',it)
                    next_relative_index=get_top_k_relative_indices_including_first(pre_attention=attention_weights, K=top_K).to(self.device)

                inputs_aug = (inputs)
                current_abs_index_aug = current_abs_index

                # Idea 1. Low-pass Filter
                if self.lpf and ((it + 1) % self.lpf_every == 0):
                    if self.lpf_type == "gaussian":
                        inputs_aug = Gaussian_low_pass_filter(inputs_aug, cutoff_ratio=self.cutoff_ratio)
                    elif self.lpf_type == "cutoff":
                        inputs_aug = low_pass_filter(inputs_aug, cutoff_ratio=self.cutoff_ratio)
                    elif self.lpf_type == "bilateral":
                        inputs_aug = Bilateral_loss_pass_filter(inputs_aug, kernel_size=self.bi_kernel, sigma_s=self.bi_sigma_s, sigma_r=self.bi_sigma_r)
                    else:
                        raise ValueError("Invalid lpf_type")

                t_out, attention_weights, current_abs_index = self.teacher(inputs_aug, current_abs_index_aug,next_relative_index)

            else:
                inputs_aug,off1,off2,flip = jitter_and_flip(inputs)
                if current_abs_index.shape[1]==num_patches:
                    current_abs_index_aug = current_abs_index
                else:
                    current_abs_index_aug =jitter_and_flip_index(current_abs_index,off1,off2,flip,self.patch_size,int(224//self.patch_size))

                # Idea 1. Low-pass Filter
                if self.lpf and ((it + 1) % self.lpf_every == 0):
                    if self.lpf_type == "gaussian":
                        inputs_aug = Gaussian_low_pass_filter(inputs_aug, cutoff_ratio=self.cutoff_ratio)
                    elif self.lpf_type == "cutoff":
                        inputs_aug = low_pass_filter(inputs_aug, cutoff_ratio=self.cutoff_ratio)
                    elif self.lpf_type == "bilateral":
                        inputs_aug = Bilateral_loss_pass_filter(inputs_aug, kernel_size=self.bi_kernel, sigma_s=self.bi_sigma_s, sigma_r=self.bi_sigma_r)
                    else:
                        raise ValueError("Invalid lpf_type")

                t_out,attention_weights,_ = self.teacher(inputs_aug,current_abs_index_aug,next_relative_index)

            # Loss
            if self.bn!=0:
                _ = self.prior(inputs_aug)
                rescale = [10.0] + [1. for _ in range(len(self.bn_hooks) - 1)]
                loss_bn = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(self.bn_hooks)])
            else:
                loss_bn=0

            if self.use_soft_label:
                loss_oh = F.kl_div(
                    F.log_softmax(t_out, dim=1),
                    targets,
                    reduction='batchmean'
                )
            else:
                loss_oh = F.cross_entropy( t_out, targets )
            if self.adv>0:
                s_out = self.student(inputs_aug)
                loss_adv = -jsdiv(s_out, t_out, T=3)
            else:
                loss_adv = loss_oh.new_zeros(1)
            loss_tv1,loss_tv2 = get_image_prior_losses(inputs)
            loss_l2 = torch.norm(inputs, 2)
            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv + self.tv1 * loss_tv1 + self.tv2*loss_tv2 + self.l2 * loss_l2
            
            loss_edge = 0
            if scale_edge > 0.0: # TODO: 추후에 수정하기 너무 코드 대충짬 & 해당 부분 gradient 체크하기
                synth_edges = self._sobel_filter(inputs_aug.mean(dim=1, keepdim=True)) # [B, 1, H, W] grayscale edge
                loss_edge = scale_edge * F.mse_loss(synth_edges, pre_synth_edges)
                loss = loss + loss_edge

            # Idea 2. Saliency Map Centering
            # if self.sc_center and ((it + 1) % self.sc_every == 0):
            if self.sc_center and ((it + 1) % self.sc_every == 0) and (it+1 < prune_it[-1]):
                B = inputs_aug.shape[0]
                p_sal = self._saliency_p(
                    inputs_aug, targets,
                    current_abs_index_aug,
                    next_relative_index
                )
                rad2 = self._center_rad2(B)

                L_center = (rad2 * p_sal).sum(dim=(1, 2, 3)).mean()
                loss = loss + self.sc_center_lambda * L_center

            if best_cost > loss.item():
                best_cost = loss.item()
                best_inputs = inputs.data

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            inputs.data = clip_images(inputs.data, self.normalizer.mean, self.normalizer.std)


        self.student.train()
        if self.normalizer:
            best_inputs = self.normalizer(best_inputs, True)
        if len(prune_ratio)==1 and prune_ratio[0]==0: #add non-masked image
            self.data_pool.add( best_inputs )

        with torch.no_grad():
            t_out,attention_weights,current_abs_index = self.teacher(best_inputs.detach(),torch.LongTensor(list(range(num_patches))).repeat(best_inputs.shape[0], 1).to(self.device),torch.LongTensor(list(range(num_patches))).repeat(best_inputs.shape[0], 1).to(self.device))

        # Idea 3. Sparsification
        num_patches_per_dim = int(sqrt(num_patches))
        num_spatial = num_patches - 1

        def cumulative_mul(lst):
            current_mul = 1
            for num in lst:
                current_mul = current_mul*(1.-num)
            return current_mul

        top_K = int(num_spatial * cumulative_mul(prune_ratio))
        top_K = max(1, min(num_spatial, top_K))

        B = best_inputs.shape[0]
        keep_flat_idx = self.patch_radial_order[:top_K]
        keep_patch_ids = (keep_flat_idx + 1).long()

        cls_col = torch.zeros(B, 1, dtype=torch.long, device=self.device)
        keep_col = keep_patch_ids.view(1, -1).expand(B, -1)
        next_relative_index = torch.cat([cls_col, keep_col], dim=1)

        mask = torch.zeros(B, num_patches_per_dim, num_patches_per_dim, device=self.device)
        for b in range(B):
            for pid in next_relative_index[b, 1:]:
                idx = int(pid.item()) - 1
                r = idx // num_patches_per_dim
                c = idx % num_patches_per_dim
                mask[b, r, c] = 1.0
        
        expanded_mask = mask.repeat_interleave(self.patch_size, dim=1).repeat_interleave(self.patch_size, dim=2)
        masked_best_inputs = best_inputs * expanded_mask.unsqueeze(1)
        #####

        if not(len(prune_ratio)==1 and prune_ratio[0]==0): #add masked image
            self.data_pool.add( masked_best_inputs )

        dst = self.data_pool.get_dataset(transform=self.transform)
        if self.init_dataset is not None:
            init_dst = UnlabeledImageDataset(self.init_dataset, transform=self.transform)
            dst = torch.utils.data.ConcatDataset([dst, init_dst])
        train_sampler = None
        loader = torch.utils.data.DataLoader(
            dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler)
        self.data_iter = DataIter(loader)
        return {'synthetic': best_inputs,'masked_synthetic':masked_best_inputs}
        
    def sample(self):
        return self.data_iter.next()
