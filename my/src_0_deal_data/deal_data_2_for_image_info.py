import os
import numpy as np
from glob import glob
from tqdm import tqdm
from alisuretool.Tools import Tools
from deal_data_0_global_info import get_data_root_path


if __name__ == '__main__':
    data_root = get_data_root_path()
    image_root_path = os.path.join(data_root, "ILSVRC2017_DET/ILSVRC/Data/DET/train")

    label_info_path = os.path.join(data_root, "deal", "label_info_list.pkl")
    label_info_list = Tools.read_from_pkl(label_info_path)
    image_info_path = os.path.join(data_root, "deal", "image_info_list2.pkl")

    image_info_list = []
    for i, label_info in tqdm(enumerate(label_info_list), total=len(label_info_list)):
        if "2013" in label_info["source"]:
            source = os.path.join("{}_train".format(label_info["source"].replace("_", "")), label_info["folder"])
        else:
            source = label_info["folder"]
        image_path = os.path.join(image_root_path, source, "{}.JPEG".format(label_info["filename"]))

        if not os.path.exists(image_path):
            Tools.print(image_path)

        label_info["image_path"] = image_path
        image_info_list.append(label_info)
        pass

    image_info_list = [one for one in image_info_list if len(one["object"]) > 0]
    Tools.write_to_pkl(_path=image_info_path, _data=image_info_list)
    pass
