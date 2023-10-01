import os
import glob
import argparse
import json
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--input_path',default="/opt/data/private/zhasion_gan/dataset/val_B/val_B_labels_resized", type=str)
parser.add_argument('--img_path',default="/opt/data/private/zhasion_gan/dataset/train_resized", type=str)
parser.add_argument('--output_path', default="./results/", type=str)
args = parser.parse_args()

def get_valid_epochs(path):
    assert os.path.exists(path)
    pkl_list = glob.glob(os.path.join(path, "*.pkl"))
    result = []
    for each in pkl_list:
        id = each.split(os.path.sep)[-1].split("_")[0]
        if id.isdigit():
            id = int(id)
        else:
            continue
        if id == "latest":
            continue
        if os.path.exists(os.path.join(path, f"{id}_net_G.pkl")) and os.path.exists(os.path.join(path, f"{id}_net_E.pkl")):
            if id not in result:
                result.append(id)
    result.sort(key=lambda x: x)
    return result

def check_false(path):
    assert os.path.exists(path)
    pic_list = glob.glob(os.path.join(path, "*.jpg"))
    return len(pic_list) != 1000

label_dir = os.path.join(args.input_path)
image_dir = os.path.join(args.img_path, "imgs")

results_path = os.path.join("temp", "fusion", f"test_360", "images", "synthesized_image")
if not os.path.exists(results_path):
    os.system(f"python spade_test.py --label_dir {label_dir} --image_dir {image_dir} --results_dir ./temp/ --name fusion --seed 144 --which_epoch 360 --no_instance --encoder_control_strategy=1")
    while check_false(results_path):
        os.system(f"python spade_test.py --label_dir {label_dir} --image_dir {image_dir} --results_dir ./temp/ --name fusion --seed 144 --which_epoch 360 --no_instance --encoder_control_strategy=1")
best_fid_path = results_path

results_path = os.path.join("temp", "merge", f"test_avg_237_272", "images", "synthesized_image")
if not os.path.exists(results_path):
    os.system(f"python spade_test.py --label_dir {label_dir} --image_dir {image_dir} --results_dir ./temp/ --name merge --seed 49546 --which_epoch avg_237_272 --no_instance --gn_norm_strategy=1 --decoder_control_strategy=3 --norm_before_noise=1 --norm_before_conv=1")
    while check_false(results_path):
        os.system(f"python spade_test.py --label_dir {label_dir} --image_dir {image_dir} --results_dir ./temp/ --name merge --seed 49546 --which_epoch avg_237_272 --no_instance --gn_norm_strategy=1 --decoder_control_strategy=3 --norm_before_noise=1 --norm_before_conv=1")
best_style_path = results_path   

with open("label_to_img.json", "r") as f:
    label_to_img = json.load(f)

train_img_path = os.path.join(args.img_path, "imgs")
train_label_path = os.path.join(args.img_path, "labels")

if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)

for label_id, style_pic_id in label_to_img.items():
    print(label_id)
    best_style_pic_path = os.path.join(best_style_path, label_id.replace(".png", ".jpg"))
    best_fid_pic_path = os.path.join(best_fid_path, label_id.replace(".png", ".jpg"))
    label_pic_path = os.path.join(args.input_path, label_id)
    style_label_pic_path = os.path.join(train_label_path, style_pic_id.replace(".jpg", ".png"))
 
    best_style_pic = np.array(Image.open(best_style_pic_path))
    best_fid_pic = np.array(Image.open(best_fid_pic_path))
    label_pic = np.array(Image.open(label_pic_path))
    style_label_pic = np.array(Image.open(style_label_pic_path))

    result = best_style_pic.copy()
    label_class_judge, _ = np.histogram(label_pic,bins=29,range=(0,29))
    style_class_judge, _ = np.histogram(style_label_pic,bins=29,range=(0,29))
    for i in range(len(label_class_judge)):
        if style_class_judge[i] == 0 and label_class_judge[i] != 0:
            mask = (label_pic == i)
            result[mask] = best_fid_pic[mask]
    Image.fromarray(result).save(os.path.join(args.output_path, label_id.replace('.png', '.jpg')))

results_path = "./results/"
from util.style import cal_style_score
from util.fid import calculate_fid_given_paths
fid_score = calculate_fid_given_paths(train_path="/opt/data/private/zhasion_gan/dataset/train_resized/imgs", test_path = results_path)
print(results_path)
style_score = cal_style_score(results_path)
print(fid_score, style_score)