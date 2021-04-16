import os
import platform


def get_data_root_path():
    if "Linux" in platform.platform():
        data_root = '/mnt/4T/Data/data/L2ID/data'
        if not os.path.isdir(data_root):
            data_root = "/media/ubuntu/4T/ALISURE/Data/L2ID/data"
    else:
        data_root = "F:\\data\\L2ID\\data"
    return data_root


def get_project_path():
    if "Linux" in platform.platform():
        data_root = '/mnt/4T/ALISURE/USS/WSS'
        if not os.path.isdir(data_root):
            data_root = "/media/ubuntu/4T/ALISURE/USS/WSS"
    else:
        data_root = "F:\\USS\\WSS"
    return data_root


