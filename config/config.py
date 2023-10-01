import argparse

# the name of the checkpoint/result dir 


def set_config(parser:argparse.ArgumentParser):
    parser.set_defaults(name = "test")
    parser.set_defaults(no_instance = True)
    parser.set_defaults(use_mask_encoder = True)
    parser.set_defaults(use_vae = False)
    parser.set_defaults(add_noise = False)
    parser.set_defaults(use_seg_noise = True)
    parser.set_defaults(nThreads = 8)
    return parser
    

def set_train_config(parser:argparse.ArgumentParser):
    parser = set_config(parser)

    # for dispaly
    parser.set_defaults(display_freq = 5000)
    parser.set_defaults(print_freq = 500)
    parser.set_defaults(save_latest_freq = 5000)
    parser.set_defaults(save_epoch_freq = 10)
    
    parser.set_defaults(batchSize = 2)
    parser.set_defaults(label_dir = "")
    parser.set_defaults(image_dir = "")
    
    parser.set_defaults(remove_hard_imgs = True) 
    parser.set_defaults(remove_img_txt_path = "hard_image.txt")

    parser.set_defaults(continue_train = False)

    parser.set_defaults(niter = 180)
    parser.set_defaults(pg_niter = 180)
    parser.set_defaults(pg_strategy = 1)
    parser.set_defaults(one_pg_D = False)

    parser.set_defaults(no_inception_loss = True)
    parser.set_defaults(no_vgg_loss = False)
    
    parser.set_defaults(reverse_map_D_pg = False)
    parser.set_defaults(reverse_map_D_ft = True)
    parser.set_defaults(num_D = 4)

    # parser.set_defaults(diff_aug = "color,crop,translation")
    
    return parser
    

def set_test_config(parser:argparse.ArgumentParser):
    parser = set_config(parser)

    parser.set_defaults(batchSize = 1)
    parser.set_defaults(label_dir = "")
    parser.set_defaults(image_dir = "")
    
    parser.set_defaults(which_epoch = "latest")

    parser.set_defaults(seed = -1)
    parser.set_defaults(use_pure = True)

    parser.set_defaults(no_pairing_check = True)
    parser.set_defaults(no_instance = False)

    parser.set_defaults(pg_niter = 180)

    return parser