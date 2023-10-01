"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re
import jittor as jt
from jittor import init
from jittor import nn
from models.networks.spectral_norm import spectral_norm
import time

# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            setattr(layer, 'bias', None)
            # layer.load_parameters({'bias': None})

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(
                get_out_channel(layer), affine=False)
        else:
            raise ValueError(
                'normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, opt = None):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)
        if opt.decoder_control_strategy == 2 or opt.decoder_control_strategy == 3:
            self.ins_norm = nn.InstanceNorm2d(norm_nc, affine=False)
            
        if opt.gn_norm_strategy == 1:
            self.param_free_norm = nn.GroupNorm(32, norm_nc, affine=False)
        elif opt.gn_norm_strategy == 2:
            self.param_free_norm = nn.GroupNorm(16, norm_nc, affine=False)
        elif opt.gn_norm_strategy == 3:
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)

        # add seg_noise
        if opt.use_seg_noise:
            k = opt.use_seg_noise_kernel
            self.seg_noise_var = nn.Conv2d(label_nc, norm_nc, k, padding=(k-1)//2)
            init.constant_(self.seg_noise_var.weight, 0.0)
            init.constant_(self.seg_noise_var.bias, 0.0)
        if opt.norm_before_noise == 1:
            k = opt.use_seg_noise_kernel
            self.second_seg_noise_var = nn.Conv2d(label_nc, norm_nc, k, padding=(k-1)//2)
            init.constant_(self.second_seg_noise_var.weight, 0.0)
            init.constant_(self.second_seg_noise_var.bias, 0.0)
        
        if opt.add_noise:
            self.noise_var = nn.Parameter(jt.zeros(norm_nc), requires_grad=True)
        
        if opt.norm_before_conv == 1:
            self.conv3x3 = nn.Conv2d(norm_nc, norm_nc, 3, 1, 1)
            self.act = nn.LeakyReLU(0.2)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        if opt.use_VQ == 1:
            self.mlp_shared = nn.Sequential(
                nn.Conv2d(128, nhidden, kernel_size=ks, padding=pw),
                nn.ReLU()
            )
        if opt.use_VQ == 2:
            self.mlp_shared = nn.Sequential(
                nn.Conv2d(32, nhidden, kernel_size=ks, padding=pw),
                nn.ReLU()
            )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.opt = opt

        if opt.decoder_control_strategy == 1 or opt.decoder_control_strategy == 2:
            self.control_mlp_shared = nn.Sequential(
                nn.Conv2d(512, nhidden, kernel_size=3, padding=1),
                nn.ReLU()
            )
            self.control_mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
            self.control_mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

            init.constant_(self.control_mlp_shared[0].weight, 0.0)
            init.constant_(self.control_mlp_shared[0].bias, 0.0)
            init.constant_(self.control_mlp_gamma.weight, 0.0)
            init.constant_(self.control_mlp_gamma.bias, 0.0)
            init.constant_(self.control_mlp_beta.weight, 0.0)
            init.constant_(self.control_mlp_beta.bias, 0.0)
        elif opt.decoder_control_strategy == 3:
            if opt.use_vit == 1:
                self.control_mlp_gamma = nn.Conv2d(1024, norm_nc, kernel_size=1, padding=0)
                self.control_mlp_beta = nn.Conv2d(1024, norm_nc, kernel_size=1, padding=0)
            else:
                self.control_mlp_gamma = nn.Conv2d(512, norm_nc, kernel_size=1, padding=0)
                self.control_mlp_beta = nn.Conv2d(512, norm_nc, kernel_size=1, padding=0)
            init.constant_(self.control_mlp_gamma.weight, 0.0)
            init.constant_(self.control_mlp_gamma.bias, 0.0)
            init.constant_(self.control_mlp_beta.weight, 0.0)
            init.constant_(self.control_mlp_beta.bias, 0.0)
        elif opt.decoder_control_strategy == 5:
            self.control_mlp_gamma = nn.Conv2d(512, norm_nc, kernel_size=1, padding=0)
            self.control_mlp_beta = nn.Conv2d(512, norm_nc, kernel_size=1, padding=0)
            self.blending_gamma = jt.zeros(1)
            self.blending_beta = jt.zeros(1)

        if opt.use_VQ_res == 1:
            self.res_mlp_shared = nn.Sequential(
                nn.Conv2d(128, nhidden, kernel_size=ks, padding=pw),
                nn.ReLU()
            )
            self.res_mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
            self.res_mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
            init.constant_(self.res_mlp_shared[0].weight, 0.0)
            init.constant_(self.res_mlp_shared[0].bias, 0.0)
            init.constant_(self.res_mlp_gamma.weight, 0.0)
            init.constant_(self.res_mlp_gamma.bias, 0.0)
            init.constant_(self.res_mlp_beta.weight, 0.0)
            init.constant_(self.res_mlp_beta.bias, 0.0)
        
            
    def execute(self, x, segmap, style_latent):

        # Part 1. generate parameter-free normalized activations
        if self.opt.use_seg_noise:
            seg = nn.interpolate(segmap, size=x.size()[2:], mode='nearest')
            noise = self.seg_noise_var(seg)
            added_noise = jt.randn(noise.shape[0], 1, noise.shape[2], noise.shape[3]) * noise
            normalized = x + added_noise
        elif self.add_noise:
            added_noise = (jt.randn(x.shape[0], x.shape[3], x.shape[2], 1) * self.noise_var).transpose(1, 3)
            normalized = x + added_noise
        else: 
            normalized = x
        # norm IN affine=False
        
        if self.opt.decoder_control_strategy == 2:
            normalized = self.ins_norm(normalized)
            style_actv = self.control_mlp_shared(style_latent)
            style_gamma = self.pool(self.control_mlp_gamma(style_actv))
            style_beta = self.pool(self.control_mlp_beta(style_actv))
            normalized = normalized * (1 + style_gamma) + style_beta
        
        if self.opt.decoder_control_strategy == 3 and self.opt.use_VQ == 0 and self.opt.use_VQ_res == 0:
            normalized = self.ins_norm(normalized)
            style_gamma = self.control_mlp_gamma(style_latent)
            style_beta = self.control_mlp_beta(style_latent)
            if self.opt.without_adain == 0:
                normalized = normalized * (1 + style_gamma) + style_beta

        if self.opt.norm_before_conv == 1:
            normalized = self.act(self.conv3x3(normalized))

        if self.opt.norm_before_noise == 1:
            seg = nn.interpolate(segmap, size=x.size()[2:], mode='nearest')
            noise = self.second_seg_noise_var(seg)
            added_noise = jt.randn(noise.shape[0], 1, noise.shape[2], noise.shape[3]) * noise
            normalized = normalized + added_noise

        # Part 2. produce scaling and bias conditioned on semantic map
        normalized = self.param_free_norm(normalized)
        segmap = nn.interpolate(segmap, size=x.size()[2:], mode='nearest')
        if self.opt.use_VQ == 1:
            bs, nc, h, w = segmap.shape
            segmap = segmap * jt.arange(0, nc).reshape(1, nc, 1, 1)
            segmap = jt.sum(segmap, dim=1, keepdims=True)
            segmap = segmap.view(bs, -1)
            selected_embeddings = style_latent.view(bs, -1, 128)[jt.arange(bs).unsqueeze(1), segmap]
            selected_embeddings = selected_embeddings.view(bs, h,  w, -1).permute(0,3,1,2)
            segmap = selected_embeddings

        if self.opt.use_VQ == 2:
            bs, nc, h, w = segmap.shape
            segmap = segmap * jt.arange(0, nc).reshape(1, nc, 1, 1)
            segmap = jt.sum(segmap, dim=1, keepdims=True)
            segmap = segmap.view(bs, -1)
            selected_embeddings = style_latent.view(bs, -1, 32)[jt.arange(bs).unsqueeze(1), segmap]
            selected_embeddings = selected_embeddings.view(bs, h,  w, -1).permute(0,3,1,2)
            segmap = selected_embeddings

        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        if self.opt.use_VQ_res == 1:
            bs, nc, h, w = segmap.shape
            segmap = segmap * jt.arange(0, nc).reshape(1, nc, 1, 1)
            segmap = jt.sum(segmap, dim=1, keepdims=True)
            segmap = segmap.view(bs, -1)
            selected_embeddings = style_latent.view(bs, -1, 128)[jt.arange(bs).unsqueeze(1), segmap]
            selected_embeddings = selected_embeddings.view(bs, h,  w, -1).permute(0,3,1,2)
            segmap = selected_embeddings
            res_actv = self.res_mlp_shared(segmap)
            res_gamma = self.res_mlp_gamma(res_actv)
            res_beta = self.res_mlp_beta(res_actv)
            gamma += res_gamma
            beta += res_beta

        if self.opt.decoder_control_strategy == 5:
            style_gamma = self.control_mlp_gamma(style_latent)
            style_beta = self.control_mlp_beta(style_latent)
            gamma_alpha = jt.sigmoid(self.blending_gamma)
            beta_alpha = jt.sigmoid(self.blending_beta)
            gamma = gamma_alpha * style_gamma + (1 - gamma_alpha) * gamma
            beta = beta_alpha * style_beta + (1 - beta_alpha) * beta

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        if self.opt.decoder_control_strategy == 1:
            style_actv = self.control_mlp_shared(style_latent)
            style_gamma = self.pool(self.control_mlp_gamma(style_actv))
            style_beta = self.pool(self.control_mlp_beta(style_actv))
            out = out * (1 + style_gamma) + style_beta

        return out
