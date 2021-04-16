import os
import numpy as np
from glob import glob
from tqdm import tqdm
from alisuretool.Tools import Tools
from deal_data_0_global_info import get_data_root_path, get_project_path


if __name__ == '__main__':
    data_root = get_data_root_path()
    image_info_path = os.path.join(data_root, "deal", "image_info_list2.pkl")
    person_pkl = os.path.join(data_root, "deal", "person2.pkl")
    result_image_info_path = os.path.join(data_root, "deal", "image_info_list_change_person2.pkl")

    image_info_list = Tools.read_from_pkl(image_info_path)
    person_info_list = Tools.read_from_pkl(person_pkl)

    result_image_info_list = []
    for i, (image_info, person_info) in tqdm(enumerate(zip(image_info_list, person_info_list)), total=len(image_info_list)):
        if not os.path.exists(image_info["image_path"]) or image_info["image_path"] != person_info[1]:
            Tools.print(image_info["image_path"])
            pass
        image_label = list(set([one[2] for one in image_info["object"]]+ ([124] if person_info[0] == 1 else [])))
        image_path = image_info["image_path"]
        result_image_info_list.append([image_label, image_path])
        pass

    Tools.write_to_pkl(_path=result_image_info_path, _data=result_image_info_list)
    pass
