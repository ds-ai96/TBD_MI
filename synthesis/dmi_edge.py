import torch
import torch.nn.functional as F
import random
from math import sqrt
import numpy as np
from tqdm import tqdm
import cv2

from .smi import SMI, jitter_and_flip, jitter_and_flip_index, clip_images, get_image_prior_losses, jsdiv, get_top_k_relative_indices_including_first
from .tbd_mi import low_pass_filter

class DMI_EDGE(SMI):
    def __init__(self, *args, **kwargs):
        super(DMI_EDGE, self).__init__(*args, **kwargs)
    
    def get_edge_map(self, images):
        """
        Extract Canny edges from a batch of images.
        images: [B, 3, H, W] tensor in range [min, max] (should be denormalized or standard range)
        Returns: [B, 1, H, W] edge map
        """
        mean = torch.tensor(self.normalizer.mean, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(self.normalizer.std, device=images.device).view(1, 3, 1, 1)
        
        # Denormalize
        imgs_denorm = images * std + mean
        imgs_denorm = torch.clamp(imgs_denorm, 0, 1) * 255.0
        imgs_np = imgs_denorm.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
        
        edge_maps = []
        for i in range(imgs_np.shape[0]):
            img = imgs_np[i]
            # Convert to gray
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # Canny
            edges = cv2.Canny(gray, 100, 200)
            edge_maps.append(edges)
            
        edge_maps = np.stack(edge_maps, axis=0) # [B, H, W]
        edge_maps = torch.from_numpy(edge_maps).float().to(images.device).unsqueeze(1) / 255.0 # [B, 1, H, W], 0 or 1
        
        return edge_maps

    def synthesize(self, targets=None, real_images=None, 
                   num_patches=197, prune_it=[-1], prune_ratio=[0],
                   lpf=False, lpf_start=100, lpf_every=10, cutoff_ratio=0.8,
                   smoothness=1.0, scale_edge=1.0):
        
        if real_images is not None:
             real_edge_maps = self.get_edge_map(real_images)
        else:
             real_edge_maps = None

        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6
        inputs = torch.randn( size=[self.synthesis_batch_size, *self.img_size], device=self.device ).requires_grad_()
        
        if targets is None:
            targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
            targets = targets.sort()[0] 
        targets = targets.to(self.device)

        optimizer = torch.optim.Adam([inputs], self.lr_g, betas=[0.5, 0.99])

        best_inputs = inputs.data

        current_abs_index = torch.LongTensor(list(range(num_patches))).repeat(best_inputs.shape[0], 1).to(self.device)
        next_relative_index = torch.LongTensor(list(range(num_patches))).repeat(best_inputs.shape[0], 1).to(self.device)
        inputs_aug = inputs
        
        current_lpf_applied = False

        for it in tqdm(range(self.iterations)):
            current_lpf_applied = False

            if lpf and (it >= lpf_start) and ((it + 1) % lpf_every == 0):
                pass

            if it+1 in prune_it:
                inputs_aug = inputs
                current_abs_index_aug = current_abs_index
                
                if lpf and (it >= lpf_start) and ((it + 1) % lpf_every == 0):
                     inputs_aug = low_pass_filter(inputs_aug, cutoff_ratio=cutoff_ratio)
                     current_lpf_applied = True

                t_out, attention_weights, _ = self.teacher(inputs_aug, current_abs_index_aug,next_relative_index)
            elif it in prune_it:
                attention_weights = torch.mean(attention_weights[-1], dim=1)[:, 0, :][:, 1:]
                prune_ratio_value = prune_ratio[prune_it.index(it)]
                top_K=int(attention_weights.shape[1] * (1.0 - prune_ratio_value))
                next_relative_index=get_top_k_relative_indices_including_first(pre_attention=attention_weights, K=top_K).to(self.device)
                inputs_aug = (inputs)
                current_abs_index_aug = current_abs_index
                
                if lpf and (it >= lpf_start) and ((it + 1) % lpf_every == 0):
                     inputs_aug = low_pass_filter(inputs_aug, cutoff_ratio=cutoff_ratio)
                     current_lpf_applied = True

                t_out, attention_weights, current_abs_index = self.teacher(inputs_aug, current_abs_index_aug,next_relative_index)
            else:
                inputs_aug,off1,off2,flip = jitter_and_flip(inputs)
                if current_abs_index.shape[1]==num_patches:
                    current_abs_index_aug = current_abs_index
                else:
                    current_abs_index_aug =jitter_and_flip_index(current_abs_index,off1,off2,flip,self.patch_size,int(224//self.patch_size))
                
                if lpf and (it >= lpf_start) and ((it + 1) % lpf_every == 0):
                     inputs_aug = low_pass_filter(inputs_aug, cutoff_ratio=cutoff_ratio)
                     current_lpf_applied = True

                t_out,attention_weights,_ = self.teacher(inputs_aug,current_abs_index_aug,next_relative_index)

            # Method 1: Direct addition
            if current_lpf_applied and real_edge_maps is not None and smoothness > 0.0:
                inputs_aug = inputs_aug + smoothness * real_edge_maps
                
            loss_edge = 0
            # Method 2: Loss optimization
            if real_edge_maps is not None and scale_edge > 0.0:
                def sobel(img):
                    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=img.device).view(1,1,3,3)
                    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=img.device).view(1,1,3,3)
                    edges_x = F.conv2d(img, kx, padding=1)
                    edges_y = F.conv2d(img, ky, padding=1)
                    return torch.sqrt(edges_x**2 + edges_y**2 + 1e-8)
                
                synth_edges = sobel(inputs_aug.mean(dim=1, keepdim=True)) # [B, 1, H, W] grayscale edge
                loss_edge = scale_edge * F.mse_loss(synth_edges, real_edge_maps)

            if self.bn!=0:
                _ = self.prior(inputs_aug)
                rescale = [10.0] + [1. for _ in range(len(self.bn_hooks) - 1)]
                loss_bn = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(self.bn_hooks)])
            else:
                loss_bn=0

            loss_oh = F.cross_entropy( t_out, targets )
            if self.adv>0:
                s_out = self.student(inputs_aug)
                loss_adv = -jsdiv(s_out, t_out, T=3)
            else:
                loss_adv = loss_oh.new_zeros(1)
            loss_tv1,loss_tv2 = get_image_prior_losses(inputs)
            loss_l2 = torch.norm(inputs, 2)
            
            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv + self.tv1 * loss_tv1 + self.tv2*loss_tv2 + self.l2 * loss_l2
            
            if real_edge_maps is not None:
                loss = loss + loss_edge
            
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
            
        # Logging targets
        targets_np = targets.cpu().detach().numpy()
        return {'synthetic': best_inputs,'masked_synthetic':best_inputs, 'targets': targets_np}

