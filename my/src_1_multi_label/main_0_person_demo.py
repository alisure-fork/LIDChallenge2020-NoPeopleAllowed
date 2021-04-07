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
from alisuretool.Tools import Tools
sys.path.append("../../")
from util_network import CAMNet, ClassNet
from util_data import DatasetUtil, MyTransform, DataUtil


class PersonRunner(object):

    def __init__(self, image_size=256, num_classes=2):
        self.net = ClassNet(num_classes=num_classes).cuda()
        self.transform_train, self.transform_test = MyTransform.transform_train_cam(image_size=image_size)
        # self.transform_test = MyTransform.transform_vis_cam(image_size=image_size)
        pass

    # 1.训练MIC
    def demo_person(self, image_filename_list, model_file_name=None):
        Tools.print("Load model form {}".format(model_file_name))
        self.load_model(model_file_name)

        self.net.eval()
        with torch.no_grad():
            for image_filename in image_filename_list:
                image = Image.open(image_filename).convert("RGB")
                inputs = torch.unsqueeze(self.transform_test(image), dim=0).float().cuda()

                logits = self.net(inputs).detach().cpu()
                logits = torch.softmax(logits, dim=1)
                net_out = torch.argmax(logits, dim=1).numpy()

                Tools.print("{} {} {}".format(net_out, logits.numpy()[0], image_filename))
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
val acc:0.9403 ../../../WSS_Model_Person/1_ClassNet_2_15_192_2_224/person_final_15.pth
"""


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    cam_runner = PersonRunner(image_size=224, num_classes=2)
    data_root = "/media/ubuntu/4T/ALISURE/Data/L2ID/data/ILSVRC2017_DET/ILSVRC/Data/DET"

    # image_filename_list = ["train/ILSVRC2014_train_0006/ILSVRC2014_train_00060002.JPEG",
    #                        "train/ILSVRC2014_train_0006/ILSVRC2014_train_00060024.JPEG",
    #                        "train/ILSVRC2014_train_0006/ILSVRC2014_train_00060027.JPEG",
    #                        "train/ILSVRC2014_train_0006/ILSVRC2014_train_00060127.JPEG"]
    # all_image_file = [os.path.join(data_root, image_filename) for image_filename in image_filename_list]

    # all_image_file = glob.glob(os.path.join(data_root, "train/ILSVRC2013_train/n00007846", "*.JPEG"))
    # all_image_file = glob.glob(os.path.join(data_root, "train/ILSVRC2013_train/n00141669", "*.JPEG"))
    # all_image_file = glob.glob(os.path.join(data_root, "train/ILSVRC2013_train/n00477639", "*.JPEG"))
    # all_image_file = glob.glob(os.path.join(data_root, "train/ILSVRC2013_train/n01443537", "*.JPEG"))
    # all_image_file = glob.glob(os.path.join(data_root, "train/ILSVRC2013_train/n01495701", "*.JPEG"))
    # all_image_file = glob.glob(os.path.join(data_root, "train/ILSVRC2013_train/n01496331", "*.JPEG"))
    all_image_file = glob.glob(os.path.join(data_root, "train/ILSVRC2013_train/n01503061", "*.JPEG"))

    cam_runner.demo_person(image_filename_list=all_image_file,
                           model_file_name="../../../WSS_Model_Person/1_ClassNet_2_15_192_2_224/person_final_15.pth")
    pass
