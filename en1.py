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

name = "en1"

if os.path.exists(f"checkpoints/{name}/latest_net_G.pkl"):
    which_epoch = "latest"
else:
    which_epoch = "origin"

ft_stage_config = {
    "encoder_control_strategy" : 1,
    "name" : name,
    "no_instance" : True,
    "batchSize" : 4,
    "continue_train" : True,
    "niter" : 151,
    "pg_niter" : 0,
    "pg_strategy" : 1,
    "which_epoch" : which_epoch,
    "which_origin_epoch" : "299",
    "save_epoch_freq" : 1,
    "diff_aug" : "color,crop,translation"
}

while True:
    try:
        os.system(get_command(ft_stage_config))
    except:
        pass
    time.sleep(60)
    if os.path.exists(f"checkpoints/{name}/latest_net_G.pkl"):
        which_epoch = "latest"
    else:
        which_epoch = "origin"
    ft_stage_config['which_epoch'] = which_epoch