import os
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from alisuretool.Tools import Tools
from deal_data_0_global_info import get_data_root_path
from src_0_deal_data.deal_data_1_for_label_info import DealData


if __name__ == '__main__':
    data_root = get_data_root_path()
    cat_ref_path = os.path.join(data_root, "LID_track1_annotations", "cat_ref.txt")

    name_to_label_id, label_info_dict = DealData.get_class_name(os.path.join(data_root, "meta_det.mat"))
    with open(cat_ref_path, "r") as f:
        all_cat = f.readlines()
        cat_ref_dict = {int(one.strip().split(" ")[0]): one.strip().split(" ")[1] for one in all_cat}
        pass

    # label_info_dict 和 cat_ref_dict 一致
    pass
