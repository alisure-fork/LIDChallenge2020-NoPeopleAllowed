import os
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image
import multiprocessing
from alisuretool.Tools import Tools


cam_path="/media/ubuntu/4T/ALISURE/USS/WSS_Model_SS_0602_EVAL/3_DeepLabV3PlusResNet152_201_10_18_1_352_8_balance/ss_7_train_500/train_final_resize"
voc12_root="/media/ubuntu/4T/ALISURE/Data/L2ID/data"


def read_image(image_path, is_rgb, max_size=500):
    im = Image.open(image_path)
    im = im.convert("RGB") if is_rgb else im

    if max_size > 0:
        value1 = max_size if im.size[0] > im.size[1] else im.size[0] * max_size // im.size[1]
        value2 = im.size[1] * max_size // im.size[0] if im.size[0] > im.size[1] else max_size
        im = im.resize((value1, value2), Image.NEAREST)
        pass
    return im


def get_ss_info_after_filter(pkl_root=None):
    _pkl_root = "/media/ubuntu/4T/ALISURE/USS/ConTa/pseudo_mask/result/2/sem_seg/train_ss.pkl"
    pkl_root = pkl_root if pkl_root is not None else _pkl_root
    Tools.print("Read pkl from {}".format(pkl_root))
    image_info_list = Tools.read_from_pkl(pkl_root)
    return image_info_list


data_info = get_ss_info_after_filter(
    pkl_root="/media/ubuntu/4T/ALISURE/USS/WSS_Model_SS_0602_EVAL/3_DeepLabV3PlusResNet152_"
             "201_10_18_1_352_8_balance/ss_7_train_500/train_final/train_ss_self_training.pkl")
train_image_path = [one_data[1] for one_data in data_info]
train_image_label = [one_data[0] for one_data in data_info]

train_image_path = [one_image_path for one_image_path in train_image_path
                    if os.path.exists(one_image_path.replace(os.path.join(
        voc12_root, "ILSVRC2017_DET/ILSVRC/Data/DET"), cam_path).replace(".JPEG", ".png"))]

train_label_path = [one_image_path.replace(os.path.join(
    voc12_root, "ILSVRC2017_DET/ILSVRC/Data/DET"), cam_path).replace(".JPEG", ".png")
                    for one_image_path in train_image_path]

for one in tqdm(train_label_path):
    mask = read_image(one, is_rgb=False, max_size=500)
    if np.asarray(mask).max() > 200:
        Tools.print(one)
        pass
    pass
