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
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
from models.networks.architecture import Zencoder as Zencoder
from models.networks.vit import ViT
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

def flip_process(tensor):
    return tensor[:,:,:,::-1]

class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if opt.decoder_control_strategy == 3 or opt.decoder_control_strategy == 5:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.actvn = nn.LeakyReLU(0.2)
        self.opt = opt


    def execute(self, x):
        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        x = self.actvn(x)
        if self.opt.decoder_control_strategy == 3 or self.opt.decoder_control_strategy == 5:
            x = self.pool(x)
        return x

class NewConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.ConvTranspose2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.actvn = nn.LeakyReLU(0.2)
        self.opt = opt


    def execute(self, x):
        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        x = self.actvn(x)
        x = self.pool(x)
        return x
    

class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='more',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)
        self.layer_level = self.get_layer_lever(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        elif opt.use_mask_encoder:
            # In case of AutoEncoder, input 1024
            self.fc = nn.Identity()
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        # spade block (channel up)
        self.up_channel_blocks = nn.ModuleList()
        for level in range(self.layer_level):
            # level = 0 --> self.up_0   1024nc --> 512nc
            fin = pow(2, 4-level) * nf
            fout = fin // 2
            self.up_channel_blocks.add_module(f"up_{level}", SPADEResnetBlock(fin, fout, opt))
        
        final_nc = pow(2, 10 - self.layer_level)

        # toRGB layer for pg
        if opt.isTrain:
            if opt.num_D > 1 and opt.pg_niter > 0:
                self.toRGB = nn.ModuleList()
                mid_toRGB_num = opt.num_D - 1
                for i in range(mid_toRGB_num):
                    fin = final_nc * pow(2, mid_toRGB_num - i)
                    self.toRGB.add_module(f"toRGB_{i}", nn.Conv2d(fin, 3, kernel_size=3, padding=1))
        
        # self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        # self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        # self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        # self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        # final_nc = nf

        # if opt.num_upsampling_layers == 'most':
        #     self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
        #     final_nc = nf // 2

        self.out_conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

        if opt.decoder_control_strategy == 1:
            self.style_mapping_network = ConvEncoder(opt)
        elif opt.decoder_control_strategy == 2:
            self.style_mapping_network = ConvEncoder(opt)
        elif opt.decoder_control_strategy == 3:
            self.style_mapping_network = ConvEncoder(opt)
        elif opt.decoder_control_strategy == 5:
            self.style_mapping_network = ConvEncoder(opt)

        if opt.use_vit == 1:
            self.style_mapping_network = ViT(
                                            image_size_h = 192,
                                            image_size_w = 256, 
                                            patch_size = 16,
                                            dim = 1024,
                                            depth = 2,
                                            heads = 16,
                                            mlp_dim = 2048,
                                            dropout = 0.1,
                                            emb_dropout = 0.1,
                                            nc = 1)
        if opt.use_VQ == 1:
            self.style_mapping_network = ViT(
                                            image_size_h = 192,
                                            image_size_w = 256, 
                                            patch_size = 16,
                                            dim = 128,
                                            depth = 2,
                                            heads = 2,
                                            mlp_dim = 2048,
                                            dropout = 0.1,
                                            emb_dropout = 0.1,
                                            nc = 29)
            self.vit_norm = nn.LayerNorm(128)

        if opt.use_VQ == 2:
            self.style_mapping_network = ViT(
                                            image_size_h = 192,
                                            image_size_w = 256, 
                                            patch_size = 16,
                                            dim = 32,
                                            depth = 2,
                                            heads = 1,
                                            dim_head = 32,
                                            mlp_dim = 2048,
                                            dropout = 0.1,
                                            emb_dropout = 0.1,
                                            nc = 29)
            self.vit_norm = nn.LayerNorm(32)

        if opt.use_VQ_res == 1:
            self.style_mapping_network = ViT(
                                            image_size_h = 192,
                                            image_size_w = 256, 
                                            patch_size = 16,
                                            dim = 128,
                                            depth = 2,
                                            heads = 2,
                                            mlp_dim = 2048,
                                            dropout = 0.1,
                                            emb_dropout = 0.1,
                                            nc = 29)
            self.vit_norm = nn.LayerNorm(128)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def get_layer_lever(self, opt):
        if opt.num_upsampling_layers == 'normal':
            layer_level= 4
        elif opt.num_upsampling_layers == 'more':
            layer_level = 5
        elif opt.num_upsampling_layers == 'most':
            layer_level = 6
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                                opt.num_upsampling_layers)
        return layer_level

    def pg_merge(self, low_res, high_res, alpha):
        up_res =  nn.interpolate(low_res, high_res.shape[-2:])
        return up_res * (1 - alpha) + high_res * alpha
    
    def calculate_current(self, epoch):
        epoch_for_each_level = self.opt.pg_niter // (self.opt.num_D - 1)
        current_level = epoch // epoch_for_each_level
        alpha = (epoch % epoch_for_each_level) / (epoch_for_each_level / 2) - 1
        alpha = 0 if alpha < 0 else alpha
        return current_level, alpha
    
    def execute(self, input, epoch, z=None, style_img = None):
        
        ################ initial ################
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = jt.randn(input.size(0), self.opt.z_dim)
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        elif self.opt.use_mask_encoder:
            # we get z from encoder directly
            assert z.shape[-1] == self.sw, "the encoder output is do not match the generator"
            x = self.fc(z)
        else:
            # we downsample segmap and run convolution
            x = nn.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        ################ del intermediate layer ################
        if epoch > self.opt.pg_niter:
            if hasattr(self, "toRGB"):
                print("del toRGB")
                del self.toRGB

        ################ execute ################
        if self.opt.decoder_augument_strategy == 1:
            style_img = process(style_img)

        if self.opt.decoder_control_strategy == 1:
            style_latent = self.style_mapping_network(style_img)
        elif self.opt.decoder_control_strategy == 2:
            style_latent = self.style_mapping_network(style_img)
        elif self.opt.decoder_control_strategy == 3 and self.opt.use_vit == 0 and self.opt.use_VQ == 0 and self.opt.use_VQ_res == 0:
            style_latent = self.style_mapping_network(style_img)
        else:
            style_latent = None
        

        if self.opt.decoder_control_strategy == 5:
            style_latent = self.style_mapping_network(style_img)
        if self.opt.use_vit == 1:
            h, w = style_img.shape[-2:]
            style_img = nn.interpolate(style_img, (h//2, w//2))
            style_latent = self.style_mapping_network(style_img) # b, 1, d
            style_latent = style_latent.squeeze(1).unsqueeze(2).unsqueeze(3)
        
        if self.opt.use_VQ == 1 or self.opt.use_VQ == 2:
            h, w = style_img.shape[-2:]
            style_img = nn.interpolate(style_img, (h//2, w//2)) # b, 29, d
            style_latent = self.style_mapping_network(style_img)
            style_latent = self.vit_norm(style_latent)

        if self.opt.use_VQ_res == 1:
            h, w = style_img.shape[-2:]
            style_img = nn.interpolate(style_img, (h//2, w//2)) # b, 29, d
            style_latent = self.style_mapping_network(style_img)
            style_latent = self.vit_norm(style_latent)
        
        x = self.head_0(x, seg, style_latent)

        x = self.up(x)
        x = self.G_middle_0(x, seg, style_latent)

        x = self.G_middle_1(x, seg, style_latent)

        for level in range(self.layer_level):
            x = self.up(x)
            x = self.up_channel_blocks[level](x, seg, style_latent)

            if self.opt.isTrain and epoch < self.opt.pg_niter:
                if self.opt.pg_strategy == 1:
                    # the intermedia layer which need to output RGB
                    assert self.layer_level > self.opt.num_D, "the layer_level of generator must bigger than the num_D of discriminator"
                    to_RGB_level, alpha = self.calculate_current(epoch)
                    intermedia_output_level = to_RGB_level + (self.layer_level - self.opt.num_D)

                    # the intermedia output (RGB)
                    if level == intermedia_output_level and (level + 1) != self.layer_level:
                        low_res = self.toRGB[to_RGB_level](nn.leaky_relu(x, 2e-1))
                        low_res = jt.tanh(low_res)
                        
                        if alpha > 0:
                            # in processing stage
                            x = self.up(x)
                            x = self.up_channel_blocks[level + 1](x, seg, style_latent)
                            if to_RGB_level == self.opt.num_D - 2:
                                high_res = self.out_conv_img(nn.leaky_relu(x, 2e-1))
                            else:
                                high_res = self.toRGB[to_RGB_level + 1](nn.leaky_relu(x, 2e-1))
                            high_res = jt.tanh(high_res)
                            x = self.pg_merge(low_res, high_res, alpha)
                        else:
                            # in steady stage
                            x = low_res
                            
                        break
        if self.opt.isTrain and epoch < self.opt.pg_niter:
            return x
        else:
            x = self.out_conv_img(nn.leaky_relu(x, 2e-1))
            x = jt.tanh(x)
            return x


class Pix2PixHDGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int,
                            default=4, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9,
                            help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = opt.label_nc + \
            (1 if opt.contain_dontcare_label else 0) + \
            (0 if opt.no_instance else 1)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(False)

        model = []

        # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)),
                      activation]
            mult = mult // 2

        # final output conv
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def execute(self, input, z=None):
        return self.model(input)
