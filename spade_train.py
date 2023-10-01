"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import subprocess
import jittor as jt
from jittor import init
from jittor import nn
import jittor.transform as transform
import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from pix2pix_trainer import Pix2PixTrainer
import ipdb
import tqdm
import os
import warnings
warnings.filterwarnings("ignore")

jt.flags.use_cuda = 1

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

# the number visual for each stage
visual_num = 1

# use EMA
if opt.use_EMA == 1:
    from ema import EMA
    ema_model_G = EMA(trainer.pix2pix_model_on_one_gpu.netG, update_after_step=0)
    ema_model_D = EMA(trainer.pix2pix_model_on_one_gpu.netD, update_after_step=0)
    ema_model_E = EMA(trainer.pix2pix_model_on_one_gpu.netE, update_after_step=0)

# if opt.use_VQ == 1:
#     os.environ['JT_CHECK_NAN'] = "1"
#     os.environ['trace_py_var'] = "3"

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(epoch, data_i)

        # train discriminator
        trainer.run_discriminator_one_step(epoch, data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(
                losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visuals = OrderedDict([('input_label', data_i['label'][:visual_num]),
                                   ('synthesized_image',
                                    trainer.get_latest_generated()[:visual_num]),
                                   ('real_image', data_i['image'][:visual_num])])
            visualizer.display_current_results(
                visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()
        
        jt.sync_all(True)

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
 