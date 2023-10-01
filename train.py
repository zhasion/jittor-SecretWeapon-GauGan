import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_path',default="/opt/data/private/zhasion_gan/dataset/train_resized", type=str)

args = parser.parse_args()


def get_command(config:dict):
    command = "python spade_train.py "
    for k, v in config.items():
        if type(v) == type(True):
            if v == True:
                command += f"--{k} "
        else:
            command += f"--{k} {v} "
    return command

pg_stage_config = {
    "encoder_control_strategy" : 0,
    "decoder_control_strategy" : 3,
    "gn_norm_strategy" : 1,
    "norm_before_noise" :1,
    "norm_before_conv" :1,

    "name" : "ada5",
    "no_instance" : True,
    "batchSize" : 10,
    # "continue_train" : True,
    "niter" : 180,
    "pg_niter" : 180,
    "pg_strategy" : 1,
    # "which_epoch" : 90,
    "align_type" : 0,
    # "inception_type" : 1

    "crop_size" : 512,
    "load_size" : 572,
    "preprocess_mode" : "scale_width_and_crop",
    "label_dir" : os.path.join(args.input_path, "labels"),
    "image_dir" : os.path.join(args.input_path, "imgs")
}


ft_stage_config = {
    "encoder_control_strategy" : 0,
    "decoder_control_strategy" : 3,
    "gn_norm_strategy" : 1,
    "norm_before_noise" :1,
    "norm_before_conv" :1,

    "name" : "ada5",
    "no_instance" : True,
    "batchSize" : 5,
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
    "label_dir" : os.path.join(args.input_path, "labels"),
    "image_dir" : os.path.join(args.input_path, "imgs")
}

os.system(get_command(pg_stage_config))
os.system(get_command(ft_stage_config))



pg_stage_config = {
    "name" : "en1",
    "no_instance" : True,
    "batchSize" : 10,
    # "continue_train" : True,
    "niter" : 180,
    "pg_niter" : 180,
    "pg_strategy" : 1,
    # "which_epoch" : 90,
    "align_type" : 0,
    # "inception_type" : 1

    "crop_size" : 512,
    "load_size" : 572,
    "preprocess_mode" : "scale_width_and_crop",
    "label_dir" : os.path.join(args.input_path, "labels"),
    "image_dir" : os.path.join(args.input_path, "imgs")
}

ft_stage_config = {
    "name" : "en1",
    "no_instance" : True,
    "batchSize" : 5,
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
    "label_dir" : os.path.join(args.input_path, "labels"),
    "image_dir" : os.path.join(args.input_path, "imgs")
}

en_stage_config = {
    "encoder_control_strategy" : 1,
    "name" : "en1",
    "no_instance" : True,
    "batchSize" : 5,
    "continue_train" : True,
    "niter" : 451,
    "pg_niter" : 180,
    "pg_strategy" : 1,
    # "which_epoch" : "origin",
    # "which_origin_epoch" : "299",
    "save_epoch_freq" : 1,
    "diff_aug" : "color,crop,translation",
    "label_dir" : os.path.join(args.input_path, "labels"),
    "image_dir" : os.path.join(args.input_path, "imgs")
}

os.system(get_command(pg_stage_config))
os.system(get_command(ft_stage_config))
os.system(get_command(en_stage_config))