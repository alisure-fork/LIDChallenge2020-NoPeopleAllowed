import os
import collections
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from alisuretool.Tools import Tools


def read_image_info(image_info_root):
    image_info_path = os.path.join(image_info_root, "deal", "image_info_list_change_person2.pkl")
    # image_info_list = Tools.read_from_pkl(image_info_path)[::200]
    image_info_list = Tools.read_from_pkl(image_info_path)
    return image_info_list


def get_train_image_info(image_info_list, cam_path):
    result_image_info_list = []
    for index, image_one in tqdm(enumerate(image_info_list), total=len(image_info_list)):
        cam_name = os.path.join(cam_path, image_one[1].split("Data/DET/")[1].replace(".JPEG", ".png"))
        im = np.asarray(Image.open(cam_name))
        size = im.shape[0] * im.shape[1]
        im.resize((size,))
        counter = collections.Counter(im.tolist())
        keys = sorted(list(counter.keys()))
        now_list = [counter[key] / size for key in keys]

        if not (keys[0] != 0 or now_list[0] > 0.96 or now_list[0] < 0.16):
            result_image_info_list.append(image_one)
            pass
        pass
    return result_image_info_list


def deal(cam_path, voc12_root, result_file="train_ss.pkl"):
    image_info_list = read_image_info(voc12_root)
    Tools.print("{}".format(len(image_info_list)))

    result_image_info_list = get_train_image_info(image_info_list, cam_path=cam_path)
    Tools.write_to_pkl(os.path.join(cam_path, result_file), result_image_info_list)
    Tools.print("{} {}".format(len(result_image_info_list), os.path.join(cam_path, result_file)))
    pass


# 第一次
# deal(cam_path="/media/ubuntu/4T/ALISURE/USS/ConTa/pseudo_mask/result/2/sem_seg",
#      voc12_root="/media/ubuntu/4T/ALISURE/Data/L2ID/data", result_file="train_ss.pkl")

# 第二次
deal(cam_path="/media/ubuntu/4T/ALISURE/USS/WSS_Model_SS_0602_EVAL/3_DeepLabV3PlusResNet152_201_10_18_1_352_8_balance/ss_7_train_500/train_final",
     voc12_root="/media/ubuntu/4T/ALISURE/Data/L2ID/data", result_file="train_ss_self_training.pkl")

