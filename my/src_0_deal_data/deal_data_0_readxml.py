import os
import numpy as np
from glob import glob
import xml.etree.ElementTree as ET
import scipy.io as scio

data = scio.loadmat(os.path.join("/media/ubuntu/4T/ALISURE/Data/L2ID/data", 'meta_det.mat'))
# anno_dir = './ILSVRC_DET/Annotations/DET/train'
anno_dir = "/media/ubuntu/4T/ALISURE/Data/L2ID/data/ILSVRC2017_DET/ILSVRC/Annotations/DET/train"
xml_files = sorted(glob(os.path.join(anno_dir, 'ILSVRC2013_train/*/*.xml')))
xml_files_2014 = sorted(glob(os.path.join(anno_dir, '*/*.xml')))
xml_files.extend(xml_files_2014)
print(len(xml_files))  # 349319

name_to_detlabelid = {}
for item in data['synsets'][0]:
    det_label_id = item[0][0][0]
    name = item[1][0]
    cat_name = item[2][0]
    # print(det_label_id)
    # print(cat_name)
    # print(name)
    name_to_detlabelid[name] = det_label_id

image_labels = np.zeros([len(xml_files), 200])
for i, xml_file in enumerate(xml_files):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = root.findall('object')
    for object in objects:
        object_name = object[0].text
        label_id = name_to_detlabelid[object_name]
        image_labels[i, label_id-1] = 1

print()