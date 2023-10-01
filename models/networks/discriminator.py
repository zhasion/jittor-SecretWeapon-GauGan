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
import util.util as util


class MultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=4,
                            help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()

        # define properties of each discriminator of the multiscale discriminator
        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator',
                                            'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.discriminator = nn.ModuleList()
        if opt.pg_strategy == 1:
            for i in range(opt.num_D):
                subnetD = self.create_single_discriminator(opt, i)
                self.discriminator.add_module(f"multiscale_discriminator_{i}", subnetD)
        else:
            self.sequential = nn.Sequential()
            for i in range(opt.num_D):
                subnetD = self.create_single_discriminator(opt)
                self.sequential.add_module(f'multiscale_discriminator_{i}', subnetD)
        
        self.have_load = False

    def create_single_discriminator(self, opt, from_RGB_level):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt, from_RGB_level)
        else:
            raise ValueError(
                'unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        # import ipdb
        # ipdb.set_trace()
        return nn.avg_pool2d(input, kernel_size=3,
                             stride=2, padding=1,
                             count_include_pad=False)
    
    def calculate_current(self, epoch):
        epoch_for_each_level = self.opt.pg_niter // (self.opt.num_D - 1)
        current_level = epoch // epoch_for_each_level
        alpha = (epoch % epoch_for_each_level) / (epoch_for_each_level / 2) - 1
        alpha = 0 if alpha < 0 else alpha
        return current_level, alpha
    
    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def execute(self, input, epoch):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss

        if self.opt.pg_strategy == 1:
            assert self.opt.pg_niter >= 0 and self.opt.num_D > 1
            # fine turning
            if epoch >= self.opt.pg_niter:
                if self.opt.reverse_map_D_ft:
                    ordered_D = range(self.opt.num_D) # 0 1 2 3
                else:
                    ordered_D = range(self.opt.num_D - 1, -1, -1) # 3 2 1 0

                for i in ordered_D:
                    out = self.discriminator[i](input)
                    if not get_intermediate_features:
                        out = [out]
                    result.append(out)
                    if self.opt.one_pg_D:
                        break
                    input = self.downsample(input)
            # process growing
            else:
                from_RGB_level, alpha = self.calculate_current(epoch)
                if alpha > 0:
                    _, alpha_last = self.calculate_current(epoch - 1)
                    if alpha_last == 0 and self.have_load == False:
                        self.discriminator[from_RGB_level + 1].load_state_dict(
                            self.discriminator[from_RGB_level].state_dict()
                        )
                        self.have_load = True
                    
                    out = self.discriminator[from_RGB_level + 1](input, alpha)
                    if not get_intermediate_features:
                        out = [out]
                    result.append(out)
                    input = self.downsample(input)
                    if self.opt.one_pg_D:
                        return result
                else:
                    self.have_load = False

                if self.opt.reverse_map_D_pg:
                    ordered_D = range(from_RGB_level + 1) # 0 1 2
                else:
                    ordered_D = range(from_RGB_level, -1, -1) # 2 1 0

                for i in ordered_D:
                    out = self.discriminator[i](input)
                    if not get_intermediate_features:
                        out = [out]
                    result.append(out)
                    if self.opt.one_pg_D:
                        break
                    input = self.downsample(input)  
        else:
            for name, D in self.sequential.items():
                out = D(input)
                if not get_intermediate_features:
                    out = [out]
                result.append(out)
                input = self.downsample(input)

        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, opt, from_RGB_level):
        super().__init__()
        self.opt = opt
        self.from_RGB_level = from_RGB_level

        kw = 4 
        padw = int(np.ceil((kw - 1.0) / 2))
        input_nc = self.compute_D_input_nc(opt)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)

        ################### create layer #####################

        # from_RGB layer (last 2 layer)
        nf = opt.ndf // pow(2, from_RGB_level) # 0 1 2
        self.from_RGB = nn.ModuleList()

        if opt.align_type == 1:
            nf = 2 * nf

        for i in range(from_RGB_level, -1, -1)[:2]:
            self.from_RGB.add_module(f"from_RGB_{i}", nn.Sequential(
                nn.Conv2d(input_nc, nf, kernel_size=kw, stride=1, padding=padw),
                nn.LeakyReLU(0.2))
                )
            nf = min(nf * 2, 512)

        # extra layer and model layer
        all_feature_layer_num = from_RGB_level + opt.n_layers_D
        nf = opt.ndf // pow(2, from_RGB_level) # 0 1 2
        
        if opt.align_type == 1:
            all_feature_layer_num = all_feature_layer_num - 1
            nf = 2 * nf

        self.sequence = nn.ModuleList()
        for i in range(all_feature_layer_num):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if i == all_feature_layer_num - 1 else 2
            if i < from_RGB_level:
                layer_name = f"extra_{from_RGB_level - i}"
            else:
                layer_name = f"model_{i - from_RGB_level}"
            
            self.sequence.add_module(layer_name, nn.Sequential(
                norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw)),
                nn.LeakyReLU(0.2)
            ))

        self.sequence.add_module(f"model_{opt.n_layers_D}", 
                              nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw))
        
        self.sequence_len = len(self.sequence)
        
    def compute_D_input_nc(self, opt):
        input_nc = opt.label_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        return input_nc

    def pg_merge(self, low_res, high_res, alpha):
        return low_res * (1 - alpha) + high_res * alpha
    
    def execute(self, input, alpha = 0):
        start = 0
        if alpha == 0:
            # steady stage
            intermediate_output = self.from_RGB[0](input)
            results = [intermediate_output]
        else:
            # growing stage
            low_res = nn.interpolate(input, scale_factor=0.5)
            low_res = self.from_RGB[1](low_res)
            high_res = self.from_RGB[0](input)
            high_res = self.sequence[0](high_res)
            intermediate_output = self.pg_merge(low_res, high_res, alpha)
            results = [intermediate_output]
            start = 1

        for i in range(start, self.sequence_len):
            intermediate_output = self.sequence[i](results[-1])
            results.append(intermediate_output)
        
        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]
