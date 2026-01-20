import time
import random

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

class TBD_MI_CNN(BaseSynthesis):
    def __init__(
        self, teacher, teacher_name, student, num_classes,
        img_shape=(3, 224, 224), iterations=2000, lr_g=0.25,
        synthesis_batch_size=128, sample_batch_size=128, adv=0.0, bn=0,
        oh=1,tv1=0.0, tv2=1e-5, l2=0.0, save_dir='', transform=None,
        normalizer=None, device='cpu', bnsource='resnet50v2',init_dataset=None
    ):
        super(TBD_MI_CNN, self).__init__(teacher, student)
        assert len(img_shape)==3, "image size should be a 3-dimension tuple"

        self.save_dir = save_dir
        self.img_size = img_shape
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

        # setup hooks for BN regularization
        if self.bn!=0:
            self.bn_hooks = []
            for m in self.prior.modules():
                if isinstance(m, nn.BatchNorm2d):
                    self.bn_hooks.append( DeepInversionHook(m) )
            assert len(self.bn_hooks)>0, 'input model should contains at least one BN layer for DeepInversion'

    def _center_rad2(self, B):
        """
        각 픽셀의 거리^2을 반환함.
        anchor에 가까우면 0, 멀수록 1에 가까움.
        """
        H, W = self.img_size[1], self.img_size[2]
        
        # Use calculated anchor coordinates
        cy = self.anchor_cy
        cx = self.anchor_cx

        y = self._coord_y - cy
        x = self._coord_x - cx
        rad2 = (y*y + x*x) / self._norm_denom

        return rad2.expand(B, 1, H, W)

    def _saliency_p(self, x, targets):
        """
        teacher 기준으로 saliency map을 확률분포 p(i,j)로 반환함.
        x: [B, 3, H, W], requires_grad=False 텐서임
        """
        # 이 함수 내에서 requires_grad=True 텐서를 만들어서 사용함.
        x_req = x.clone().requires_grad_(True)

        logits = self.teacher(x_req)
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

    def _mask_from_ratio(self, dist_map, dist_sorted, prune_ratio_value):
        num_pixels = dist_sorted.numel()
        keep_pixels = int(num_pixels * (1.0 - prune_ratio_value))
        keep_pixels = max(1, min(num_pixels, keep_pixels))
        threshold = dist_sorted[keep_pixels - 1]
        return  (dist_map <= threshold).float()

    def synthesize(self, targets=None,
                   prune_it=[-1], prune_ratio=[0],
                   lpf=False, lpf_every=10, cutoff_ratio=0.8,
                   sc_center=False, sc_every=50, sc_center_lambda=0.1,
                   saliency_anchor='c',
                   use_soft_label=False, soft_label_alpha=0.6
        ):

        # Idea 1. Low-pass Filter
        self.lpf = lpf
        self.lpf_every = lpf_every
        self.cutoff_ratio = cutoff_ratio

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

        dist_map = torch.sqrt(
            (self._coord_y - self.anchor_cy) ** 2 + (self._coord_x - self.anchor_cx) ** 2
        )
        dist_sorted, _ = torch.sort(dist_map.view(-1))

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

        inputs_aug = inputs

        for it in tqdm(range(self.iterations)):
            if it+1 in prune_it:
                inputs_aug = inputs

                # Idea 1. Low-pass Filter
                if self.lpf and ((it + 1) % self.lpf_every == 0):
                    inputs_aug = low_pass_filter(inputs_aug, cutoff_ratio=self.cutoff_ratio)

                t_out = self.teacher(inputs_aug)

            elif it in prune_it:
                prune_ratio_value = prune_ratio[prune_it.index(it)]
                mask = self._mask_from_ratio(dist_map, dist_sorted, prune_ratio_value)
                inputs_aug = inputs_aug * mask

                # Idea 1. Low-pass Filter
                if self.lpf and ((it + 1) % self.lpf_every == 0):
                    inputs_aug = low_pass_filter(inputs_aug, cutoff_ratio=self.cutoff_ratio)

                t_out = self.teacher(inputs_aug)

            else:
                inputs_aug, off1, off2, flip = jitter_and_flip(inputs)

                # Idea 1. Low-pass Filter
                if self.lpf and ((it + 1) % self.lpf_every == 0):
                    inputs_aug = low_pass_filter(inputs_aug, cutoff_ratio=self.cutoff_ratio)

                t_out = self.teacher(inputs_aug)

            # Loss

            ### BN loss
            if self.bn!=0:
                _ = self.prior(inputs_aug)
                rescale = [10.0] + [1. for _ in range(len(self.bn_hooks) - 1)]
                loss_bn = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(self.bn_hooks)])
            else:
                loss_bn=0

            # Soft loss
            if self.use_soft_label:
                loss_oh = F.kl_div(
                    F.log_softmax(t_out, dim=1),
                    targets,
                    reduction='batchmean'
                )
            else:
                loss_oh = F.cross_entropy( t_out, targets )
            
            # Adversarial loss
            if self.adv > 0:
                s_out = self.student(inputs_aug)
                loss_adv = -jsdiv(s_out, t_out, T=3)
            else:
                loss_adv = loss_oh.new_zeros(1)

            ### Total variation loss
            loss_tv1,loss_tv2 = get_image_prior_losses(inputs)
            loss_l2 = torch.norm(inputs, 2)

            total_loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv + self.tv1 * loss_tv1 + self.tv2*loss_tv2 + self.l2 * loss_l2

            # Idea 2. Saliency Map Centering
            # if self.sc_center and ((it + 1) % self.sc_every == 0):
            if self.sc_center and ((it + 1) % self.sc_every == 0) and (it+1 < prune_it[-1]):
                B = inputs_aug.shape[0]
                p_sal = self._saliency_p(inputs_aug, targets)
                rad2 = self._center_rad2(B)

                L_center = (rad2 * p_sal).sum(dim=(1, 2, 3)).mean()
                total_loss = total_loss + self.sc_center_lambda * L_center

            if best_cost > total_loss.item():
                best_cost = total_loss.item()
                best_inputs = inputs.data

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            inputs.data = clip_images(inputs.data, self.normalizer.mean, self.normalizer.std)


        self.student.train()
        if self.normalizer:
            best_inputs = self.normalizer(best_inputs, True)
        if len(prune_ratio)==1 and prune_ratio[0]==0: #add non-masked image
            self.data_pool.add( best_inputs, targets=targets )

        def cumulative_mul(lst):
            current_mul = 1
            for num in lst:
                current_mul = current_mul*(1.-num)
            return current_mul

        keep_ratio = cumulative_mul(prune_ratio)
        overall_prune_ratio = 1. - keep_ratio
        final_mask = self._mask_from_ratio(dist_map, dist_sorted, overall_prune_ratio)
        masked_best_inputs = best_inputs * final_mask

        if not(len(prune_ratio)==1 and prune_ratio[0]==0): #add masked image
            self.data_pool.add( masked_best_inputs, targets=targets )

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
