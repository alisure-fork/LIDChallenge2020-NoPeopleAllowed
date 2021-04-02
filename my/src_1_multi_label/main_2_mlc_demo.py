import os
import sys
import glob
import torch
import random
import platform
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from util_network import CAMNet
from alisuretool.Tools import Tools
from util_data import DatasetUtil, MyTransform, DataUtil


class CAMRunner(object):

    def __init__(self, image_size=256, num_classes=200):
        self.image_size = image_size
        self.net = CAMNet(num_classes=num_classes).cuda()

        self.name_to_label_id, self.label_info_dict = DataUtil.get_class_name()
        self.transform_train, self.transform_test = MyTransform.transform_train_cam(image_size=self.image_size)
        pass

    # 1.训练MIC
    def demo_mlc(self, image_filename_list, model_file_name=None):
        Tools.print("Load model form {}".format(model_file_name))
        self.load_model(model_file_name)

        self.net.eval()
        with torch.no_grad():
            for image_filename in image_filename_list:
                image = Image.open(image_filename).convert("RGB")
                inputs = torch.unsqueeze(self.transform_test(image), dim=0).float().cuda()

                logits = self.net(inputs)
                net_out = torch.sigmoid(logits).detach().cpu().numpy()

                arg_sort = np.argsort(net_out)[0][-10:]
                for index, arg_one in enumerate(arg_sort):
                    label_info = self.label_info_dict[arg_one+1]
                    Tools.print("{:3d} {:3d} {:.4f} {} {}".format(
                        index, arg_one, net_out[0][arg_one], label_info["name"], label_info["cat_name"]))
                    pass
                Tools.print(image_filename)
                pass
            pass
        pass

    def load_model(self, model_file_name):
        Tools.print("Load model form {}".format(model_file_name))
        checkpoint = torch.load(model_file_name)
        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1:
            checkpoint = {key.replace("module.", ""): checkpoint[key] for key in checkpoint}
        self.net.load_state_dict(checkpoint, strict=True)
        Tools.print("Restore from {}".format(model_file_name))
        pass

    pass


"""
1/20 val mae:0.0775 f1:0.9067 ../../../WSS_Model/demo_CAMNet_200_60_128_5_224/mlc_final_60.pth
"""


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    cam_runner = CAMRunner(image_size=224, num_classes=200)

    data_root = "/media/ubuntu/4T/ALISURE/Data/L2ID/data/ILSVRC2017_DET/ILSVRC/Data/DET"
    image_filename_list = ["train/ILSVRC2014_train_0006/ILSVRC2014_train_00060002.JPEG",
                           "train/ILSVRC2014_train_0006/ILSVRC2014_train_00060027.JPEG",
                           "train/ILSVRC2014_train_0006/ILSVRC2014_train_00060024.JPEG",
                           "train/ILSVRC2014_train_0006/ILSVRC2014_train_00060127.JPEG"]
    all_image_file = [os.path.join(data_root, image_filename) for image_filename in image_filename_list]

    cam_runner.demo_mlc(image_filename_list=all_image_file,
                        model_file_name="../../../WSS_Model/demo_CAMNet_200_60_128_5_224/mlc_final_60.pth")
    pass
