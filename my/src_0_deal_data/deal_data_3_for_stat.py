import os
import numpy as np
from glob import glob
from tqdm import tqdm
import scipy.io as scio
import xml.etree.ElementTree as ET
from alisuretool.Tools import Tools


if __name__ == '__main__':
    data_root = "/media/ubuntu/4T/ALISURE/Data/L2ID/data"
    image_info_path = os.path.join(data_root, "deal", "image_info_list.pkl")
    image_info_list = Tools.read_from_pkl(image_info_path)

    cat_list = {}
    cat_single_list = {}
    for image_info in image_info_list:
        for one in image_info["object"]:
            cat_name = one[-1]["cat_name"]

            if cat_name not in cat_list:
                cat_list[cat_name] = []
            if cat_name not in cat_single_list:
                cat_single_list[cat_name] = []

            if len(image_info["object"]) == 1:
                cat_single_list[cat_name].append(image_info)
            else:
                # Tools.print(image_info["image_path"])
                pass
            cat_list[cat_name].append(image_info)
            pass
        pass

    cat_len_list = {}
    for key in cat_list:
        cat_len_list[key] = len(cat_list[key])
        pass
    cat_len_list_sort = sorted(cat_len_list.items(), key=lambda cat_len_list: cat_len_list[1], reverse=True)
    Tools.print(cat_len_list_sort)
    Tools.print(np.sum(list(cat_len_list.values())))

    cat_single_len_list = {}
    for key in cat_single_list:
        cat_single_len_list[key] = len(cat_single_list[key])
        pass
    cat_single_len_list_sort = sorted(cat_single_len_list.items(),
                                      key=lambda cat_single_len_list: cat_single_len_list[1], reverse=True)
    Tools.print(cat_single_len_list_sort)
    Tools.print(np.sum(list(cat_single_len_list.values())))

    cat_sub_len_list = {}
    for key in cat_single_list:
        cat_sub_len_list[key] = len(cat_list[key]) - len(cat_single_list[key])
        pass
    cat_sub_len_list_sort = sorted(cat_sub_len_list.items(),
                                   key=lambda cat_sub_len_list: cat_sub_len_list[1], reverse=True)
    Tools.print(cat_sub_len_list_sort)
    Tools.print(np.sum(list(cat_sub_len_list.values())))
    pass
