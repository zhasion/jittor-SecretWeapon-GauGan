from tqdm import tqdm
import cv2
import numpy as np
import json
import os

with open("/opt/data/private/zhasion_gan/dataset/val_B/label_to_img.json", "r") as f:
    label_to_img = json.load(f)

def cal_hsv_score(original_pic_path, style_pic_path):
    original_pic = cv2.imread(original_pic_path) # BGR
    original_pic = cv2.cvtColor(original_pic, cv2.COLOR_BGR2HSV)

    style_pic = cv2.imread(style_pic_path) # BGR
    style_pic = cv2.cvtColor(style_pic, cv2.COLOR_BGR2HSV)
    metric_val = []
    for i in range(1):
        if i == 0:
            original_hist = cv2.calcHist([original_pic],[i],None,[180],[0,180])
            style_hist = cv2.calcHist([style_pic],[i],None,[180],[0,180])
        else:
            original_hist = cv2.calcHist([original_pic],[i],None,[120],[0,256])
            style_hist = cv2.calcHist([style_pic],[i],None,[120],[0,256])
        metric_val.append(cv2.compareHist(original_hist, style_hist, cv2.HISTCMP_CORREL))
    metric_val = np.mean(metric_val)
    return metric_val

def cal_style_score(img_path):
    correl = []
    for k, v in tqdm(label_to_img.items()):
        # if "opt" not in img_path:
        #     img_path = os.path.join("/opt/data/private/zhasion_gan", img_path)
        original_pic_path = os.path.join(img_path, k.replace("png", "jpg")) 
        style_pic_path = os.path.join("/opt/data/private/zhasion_gan/dataset/train_resized/imgs", v)
        score = cal_hsv_score(original_pic_path, style_pic_path)
        correl.append(score)
    return np.mean(correl)


if __name__=='__main__':
    path_list = {
        # "submit_1": "jittor-Torile-PG_SPADE/results_last_best",
        # "submit_2": "result",
        # "submit_3": "gaugan/results/baseline/test_200/images/synthesized_image",
        # "submit_4": "gaugan/results/baseline/test_220/images/synthesized_image",
        # "submit_5": "gaugan/results/baseline/test_260/images/synthesized_image",
        # "submit_6": "gaugan/results/baseline_instance/test_258/images/synthesized_image",
        # "submit_7": "gaugan/results/baseline/test_295/images/synthesized_image",
        # "submit_8": "gaugan/results/baseline_instance/test_295/images/synthesized_image",
        # "submit_9": "gaugan/results/origin/test_299/images/synthesized_image",
        # "submit_10": "gaugan/results/last/test_200/images/synthesized_image",
        # "submit_11": "gaugan/results/en2/test_30/images/synthesized_image",
        # "submit_12": "gaugan/results/en3/test_30/images/synthesized_image",
        "submit_13": "gaugan/results/en1/test_56/images/synthesized_image",
        "submit_14": "gaugan/results/last/test_240/images/synthesized_image",
    }
    for submit_name, img_path in path_list.items():
        correl = []
        print(submit_name, cal_style_score(img_path))

