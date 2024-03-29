"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import jittor as jt
from jittor import init
from jittor import nn
import jittor.transform as transform
import models.networks as networks
import util.util as util
from util.util import DiffAugment #TODO: import
import numpy as np
import warnings
import time
warnings.filterwarnings("ignore")


class Pix2PixModel(nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = jt.float32

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = nn.L1Loss()
            
            assert opt.no_vgg_loss or opt.inception_type == 0, "only can use vgg_loss or inception_loss"
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.inception_type == 1:# TODO:import Inceptionloss
                self.criterionInception = networks.InceptionLoss(self.opt.gpu_ids)
            
            assert not (opt.use_vae and opt.use_mask_encoder), "only can use vae or mask_encoder"
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def execute(self, epoch, data, mode):
        input_semantics, real_image = self.preprocess_input(data)
        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image, epoch)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image, epoch)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            with jt.no_grad():
                fake_image, _ = self.generate_fake(input_semantics, real_image)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.frozen_type == 1:
            for i in range(len(G_params)):
                G_params[i].requires_grad = False
            print("G frozen")
        if opt.use_vae or opt.use_mask_encoder:
            G_params += list(self.netE.parameters())
            if opt.frozen_type == 1:
                for para in self.netE.named_parameters():
                    G_params.append(para[1])
                    if "ada" in para[0] or "bra" in para[0]:
                        G_params[-1].requires_grad = True
                    else:
                        G_params[-1].requires_grad = False
            
        if opt.isTrain:
            D_params = list(self.netD.parameters())
        
        if opt.frozen_type == 1:
            for i in range(len(D_params)):
                D_params[i].requires_grad = False
            print("D frozen")
            
        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = jt.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = jt.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)
        if self.opt.use_mask_encoder:
            util.save_network(self.netE, 'E', epoch, self.opt)
    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if (opt.use_vae or opt.use_mask_encoder) else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)
            if opt.use_mask_encoder:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)

        return netG, netD, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # change data types
        data['label'] = data['label'].long()

        # create one-hot label map
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label else self.opt.label_nc
        input_label = jt.zeros((bs, nc, h, w), dtype=self.FloatTensor)
        # print(jt.float32(1.0).dtype, label_map.dtype, input_label.dtype)
        input_semantics = input_label.scatter_(1, label_map, jt.float32(1.0))

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = jt.concat(
                (input_semantics, instance_edge_map), dim=1)
        
        return input_semantics, data['image']

    def compute_generator_loss(self, input_semantics, real_image, epoch):
        G_losses = {}
        fake_image, KLD_loss = self.generate_fake(
            input_semantics, real_image, epoch, compute_kld_loss=self.opt.use_vae)


        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image, epoch)

        G_losses['GAN'] = self.criterionGAN(
            pred_fake, True, for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(0.)
            # print(self.opt.lambda_feat, num_D, len(pred_fake), len(pred_fake[0]))
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    # unweighted_loss = jt.abs(jt.float32(pred_fake[i][j]-pred_real[i][j].detach())).mean().float_auto()
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
                    # print("#", pred_fake[i][j].shape, GAN_Feat_loss)
            G_losses['GAN_Feat'] = GAN_Feat_loss

        
        if not self.opt.no_vgg_loss:
            real_image = nn.interpolate(real_image, fake_image.shape[-2:])
            G_losses['VGG'] = self.criterionVGG(
                fake_image, real_image) * self.opt.lambda_vgg
        if self.opt.inception_type == 1:
            real_image = nn.interpolate(real_image, fake_image.shape[-2:])
            G_losses['Inception'] = self.criterionInception(
                fake_image, real_image) * self.opt.lambda_inception
            
        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image, epoch):
        D_losses = {}

        with jt.no_grad():
            fake_image, _ = self.generate_fake(input_semantics, real_image, epoch)
            # fake_image = fake_image.detach()
            # fake_image.requires_grad_()
            with jt.enable_grad():
                fake_image = fake_image.detach()
        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image, epoch)
        D_losses['D_Fake'] = self.criterionGAN(
            pred_fake, False, for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(
            pred_real, True, for_discriminator=True)

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def encode_m(self, input_semantics, condition=None):
        z = self.netE(input_semantics, condition)
        return z
    
    def generate_fake(self, input_semantics, real_image, epoch = 0, compute_kld_loss=False):
        z = None
        KLD_loss = None
        # if self.opt.use_vae and self.opt.isTrain:
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld
        if self.opt.use_mask_encoder:
            if self.opt.encoder_control_strategy != 0:
                z = self.encode_m(input_semantics, real_image)
            else:
                z = self.encode_m(input_semantics)
        fake_image = self.netG(input_semantics, epoch, z=z, style_img = real_image)
        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, KLD_loss

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image, epoch):
        if len(self.opt.diff_aug) > 0:
            real_image, fake_image, input_semantics = DiffAugment(real_image, fake_image, input_semantics, policy=self.opt.diff_aug)
        
        image_shape = fake_image.shape[-2:]
        input_semantics = nn.interpolate(input_semantics, image_shape)
        real_image = nn.interpolate(real_image, image_shape)
        
        fake_concat = jt.concat([input_semantics, fake_image], dim=1)
        real_concat = jt.concat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = jt.concat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real, epoch)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    # def get_edges(self, t):
    #     edge = self.ByteTensor(t.size()).zero_()
    #     edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    #     edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    #     edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    #     edge[:, :, :-1, :] = edge[:, :, :-1,:] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    #     return edge.float()

    def get_edges(self, t):
        edge = np.zeros(t.shape, dtype=np.byte)
        t = t.numpy()
        edge[:,1:] = edge[:,1:] | (t[:,1:] !=  t[:,:-1])
        edge[:,:-1] = edge[:,:-1] | (t[:,1:] !=  t[:,:-1])
        edge[1:,:] = edge[1:,:] | (t[1:,:] !=  t[:-1,:])
        edge[:-1,:] = edge[:-1,:] | (t[1:,:] !=  t[:-1,:])
        return jt.array(edge).float()

    def reparameterize(self, mu, logvar):
        std = jt.exp(0.5 * logvar)
        eps = jt.randn_like(std)
        return eps.multiply(std).add(mu)

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
