"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import jittor as jt
from jittor import init
from jittor import nn
import numpy as np
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
import cv2
import random

def process(tensor):
    h, w = tensor.shape[-2:]
    # right-left flip
    tensor = tensor[:,:,:,::-1]
    max_crop_scale_ratio = 1
    min_crop_scale_ratio = 7/8
    crop_scale_ratio = random.random() * (max_crop_scale_ratio - min_crop_scale_ratio) + min_crop_scale_ratio
    crop_w = int(crop_scale_ratio * w)
    crop_h = int(crop_scale_ratio * h)
    x = random.randint(0, np.maximum(0, w - crop_w))
    y = random.randint(0, np.maximum(0, h - crop_h))
    # random_crop
    new_tensor = tensor[:,:,y:y+crop_h,x:x+crop_w]
    new_tensor = nn.interpolate(new_tensor, (h, w))
    return new_tensor



class AdaLayer(nn.Module):
    def __init__(self, norm_nc, pool_in_last=True, opt=None) :
        super().__init__()

        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)

        nhidden = 128
        ks = 3 if pool_in_last else 1
        pw = ks // 2
        if opt.encoder_control_strategy == 1 or opt.encoder_control_strategy == 2 or opt.encoder_control_strategy == 3:
            self.mlp_shared = nn.Sequential(
                nn.Conv2d(norm_nc, nhidden, kernel_size=ks, padding=pw),
                nn.ReLU()
            )

            self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
            self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

            init.constant_(self.mlp_shared[0].weight, 0.0)
            init.constant_(self.mlp_shared[0].bias, 0.0)
            init.constant_(self.mlp_gamma.weight, 0.0)
            init.constant_(self.mlp_gamma.bias, 0.0)
            init.constant_(self.mlp_beta.weight, 0.0)
            init.constant_(self.mlp_beta.bias, 0.0)

        self.opt = opt
        self.pool_in_last = pool_in_last

    def execute(self, x, condition):
        normalized = x
        if self.opt.encoder_control_strategy == 1 or self.opt.encoder_control_strategy == 2 or self.opt.encoder_control_strategy == 3:
            if self.pool_in_last:
                actv = self.mlp_shared(condition)
                gamma = self.pool(self.mlp_gamma(actv))
                beta = self.pool(self.mlp_beta(actv))
            else:
                actv = self.pool(self.mlp_shared(condition))
                gamma = self.mlp_gamma(actv)
                beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out
    
class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(
            nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(
            nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(
            nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(
            nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if opt.crop_size >= 256:
            self.layer6 = norm_layer(
                nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        if opt.use_mask_encoder:
            # re-load layer
            self.layer1 = norm_layer(nn.Conv2d(opt.semantic_nc, ndf, kw, stride=2, padding=pw))
            if opt.num_upsampling_layers == "normal":
                self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 16, kw, stride=2, padding=pw))
            if opt.num_upsampling_layers == "more":
                self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 16, kw, stride=2, padding=pw))
            if opt.num_upsampling_layers == "most":
                self.layer7 = norm_layer(nn.Conv2d(ndf * 16, ndf * 16, kw, stride=2, padding=pw))

        if opt.use_vae:
            self.so = s0 = 4
            self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
            self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)

        if opt.encoder_control_strategy != 0:
            # 1: pool_in_last   2: not pool_in_last
            pool_in_last = True if opt.encoder_control_strategy == 1 else False

            f_in = 3

            self.branch_layer1 = norm_layer(nn.Conv2d(f_in, ndf, kw, stride=2, padding=pw))
            self.ada_layer1 = AdaLayer(ndf, pool_in_last, opt)

            self.branch_layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
            self.ada_layer2 = AdaLayer(ndf * 2, pool_in_last, opt)

            self.branch_layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
            self.ada_layer3 = AdaLayer(ndf * 4, pool_in_last, opt)

            self.branch_layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
            self.ada_layer4 = AdaLayer(ndf * 8, pool_in_last, opt)

            self.branch_layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
            self.ada_layer5 = AdaLayer(ndf * 8, pool_in_last, opt)

        self.actvn = nn.LeakyReLU(0.2)
        self.opt = opt
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def execute(self, x, c=None):
        if self.opt.use_mask_encoder:
            if self.opt.encoder_control_strategy == 0:
                x = self.layer1(x)
                x = self.layer2(self.actvn(x))
                x = self.layer3(self.actvn(x))
                x = self.layer4(self.actvn(x))
                x = self.layer5(self.actvn(x))
                if self.opt.num_upsampling_layers == "more":
                    x = self.layer6(self.actvn(x))
                if self.opt.num_upsampling_layers == "most":
                    x = self.layer7(self.actvn(x))
                x = self.actvn(x)
                return x
            else:

                if self.opt.encoder_augument_strategy == 1 and self.opt.isTrain:
                    c = process(c)

                x = self.layer1(x)
                c = self.branch_layer1(c)
                x = self.ada_layer1(x, c)

                x = self.layer2(self.actvn(x))
                c = self.branch_layer2(self.actvn(c))
                x = self.ada_layer2(x, c)
            
                x = self.layer3(self.actvn(x))
                c = self.branch_layer3(self.actvn(c))
                x = self.ada_layer3(x, c)

                x = self.layer4(self.actvn(x))
                c = self.branch_layer4(self.actvn(c))
                x = self.ada_layer4(x, c)

                x = self.layer5(self.actvn(x))
                c = self.branch_layer5(self.actvn(c))
                x = self.ada_layer5(x, c)

                if self.opt.num_upsampling_layers == "more":
                    x = self.layer6(self.actvn(x))
                if self.opt.num_upsampling_layers == "most":
                    x = self.layer7(self.actvn(x))
                x = self.actvn(x)
                return x
            
        else:
            if x.size(2) != 256 or x.size(3) != 256:
                x = nn.interpolate(x, size=(256, 256), mode='bilinear')

            x = self.layer1(x)
            x = self.layer2(self.actvn(x))
            x = self.layer3(self.actvn(x))
            x = self.layer4(self.actvn(x))
            x = self.layer5(self.actvn(x))
            if self.opt.crop_size >= 256:
                x = self.layer6(self.actvn(x))
            x = self.actvn(x)

            x = x.view(x.size(0), -1)
            mu = self.fc_mu(x)
            logvar = self.fc_var(x)

            return mu, logvar
