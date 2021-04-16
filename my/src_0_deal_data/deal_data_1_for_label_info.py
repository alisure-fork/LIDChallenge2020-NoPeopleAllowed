import os
import numpy as np
from glob import glob
from tqdm import tqdm
import scipy.io as scio
import xml.etree.ElementTree as ET
from alisuretool.Tools import Tools
from deal_data_0_global_info import get_data_root_path


class DealData(object):

    @staticmethod
    def get_all_xml(anno_dir):
        xml_files = sorted(glob(os.path.join(anno_dir, 'ILSVRC2013_train/*/*.xml')))
        xml_files_2014 = sorted(glob(os.path.join(anno_dir, '*/*.xml')))
        xml_files.extend(xml_files_2014)
        return xml_files

    @staticmethod
    def get_class_name(mat_file):
        data = scio.loadmat(mat_file)

        label_info_dict = {}
        name_to_label_id = {}
        for item in data['synsets'][0]:
            label_id = item[0][0][0]
            name = item[1][0]
            cat_name = item[2][0]
            name_to_label_id[name] = label_id
            label_info_dict[label_id] = {"name": name, "cat_name": cat_name}
            pass
        return name_to_label_id, label_info_dict

    @staticmethod
    def parse_xml(xml_file):
        root = ET.parse(xml_file).getroot()

        now_info = {"folder": root.find("folder").text, "filename": root.find("filename").text,
                    "source": root.find("source")[0].text,
                    "size": (int(root.find("size").find("width").text),
                             int(root.find("size").find("height").text)),
                    "object": []}

        objects = root.findall("object")
        for one in objects:
            object_name = one.find("name").text
            _bnd_box = one.find("bndbox")
            object_box = (int(_bnd_box[0].text), int(_bnd_box[1].text),
                          int(_bnd_box[2].text), int(_bnd_box[3].text))
            label_id = name_to_label_id[object_name]
            label_info = label_info_dict[label_id]

            now_info["object"].append([object_name, object_box, label_id, label_info])
            pass
        return now_info

    pass


if __name__ == '__main__':
    data_root = get_data_root_path()
    result_path = Tools.new_dir(os.path.join(data_root, "deal", "label_info_list.pkl"))

    name_to_label_id, label_info_dict = DealData.get_class_name(os.path.join(data_root, "meta_det.mat"))
    xml_files = DealData.get_all_xml(os.path.join(data_root, "ILSVRC2017_DET/ILSVRC/Annotations/DET/train"))

    label_info_list = []
    for i, xml_file in tqdm(enumerate(xml_files), total=len(xml_files)):
        now_info = DealData.parse_xml(xml_file=xml_file)
        label_info_list.append(now_info)
        pass

    Tools.write_to_pkl(_path=result_path, _data=label_info_list)
    pass
