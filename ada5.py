import os
import time 

def get_command(config:dict):
    command = "python spade_train.py "
    for k, v in config.items():
        if type(v) == type(True):
            if v == True:
                command += f"--{k} "
        else:
            command += f"--{k} {v} "
    return command

name = "ada5"

pg_stage_config = {
    "encoder_control_strategy" : 0,
    "decoder_control_strategy" : 3,
    "gn_norm_strategy" : 1,
    "norm_before_noise" :1,
    "norm_before_conv" :1,

    "name" : name,
    "no_instance" : True,
    "batchSize" : 3,
    "continue_train" : True,
    "niter" : 180,
    "pg_niter" : 180,
    "pg_strategy" : 1,
    # "which_epoch" : 90,
    "align_type" : 0,
    # "inception_type" : 1

    "crop_size" : 512,
    "load_size" : 572,
    "preprocess_mode" : "scale_width_and_crop",
    "label_dir" : "/opt/data/private/zhasion_gan/dataset/train/labels",
    "image_dir" : "/opt/data/private/zhasion_gan/dataset/train/imgs"
}


ft_stage_config = {
    "encoder_control_strategy" : 0,
    "decoder_control_strategy" : 3,
    "gn_norm_strategy" : 1,
    "norm_before_noise" :1,
    "norm_before_conv" :1,

    "name" : name,
    "no_instance" : True,
    "batchSize" : 3,
    "continue_train" : True,
    "niter" : 300,
    "pg_niter" : 180,
    "pg_strategy" : 1,
    # "which_epoch" : 240,
    "save_epoch_freq" : 1,
    "diff_aug" : "color,crop,translation",
    "align_type" : 0,
    "inception_type" : 1,
    "no_vgg_loss" : True,
    "encoder_control_strategy" : 0,

    "crop_size" : 512,
    "load_size" : 572,
    "preprocess_mode" : "scale_width_and_crop",
    "label_dir" : "/opt/data/private/zhasion_gan/dataset/train/labels",
    "image_dir" : "/opt/data/private/zhasion_gan/dataset/train/imgs"
}


while True:
    try:
        # os.system(get_command(pg_stage_config))
        os.system(get_command(ft_stage_config))
    except:
        pass
    time.sleep(60)
    pg_stage_config['continue_train'] = True
    if os.path.exists(f"checkpoints/{name}/latest_net_G.pkl"):
        which_epoch = "latest"
    else:
        which_epoch = "origin"
    ft_stage_config['which_epoch'] = which_epoch