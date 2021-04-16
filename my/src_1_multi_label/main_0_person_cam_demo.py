import os
import sys
import cv2
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
from torch.utils.data import DataLoader, Dataset
sys.path.append("../../")
from util_data import DatasetUtil, MyTransform, DataUtil
from util_network import CAMNet, ClassNet


class PersonCAMRunner(object):

    def __init__(self, image_size=224, num_classes=200):
        self.image_size = image_size
        self.net = ClassNet(num_classes=num_classes).cuda()

        self.transform_train, self.transform_test = MyTransform.transform_train_cam(image_size=self.image_size)
        self.transform_un_normalize = MyTransform.transform_un_normalize()
        pass

    def demo_mlc_cam(self, image_filename_list, save_path=None, model_file_name=None):
        Tools.print("Load model form {}".format(model_file_name))
        self.load_model(model_file_name)

        self.net.eval()
        with torch.no_grad():
            for image_filename in image_filename_list:
                image = Image.open(image_filename).convert("RGB")
                image_inputs = self.transform_test(image)
                inputs = torch.unsqueeze(image_inputs, dim=0).float().cuda()

                logits, out_features = self.net.forward(inputs, is_vis=True)
                logits = logits.detach().cpu().numpy()

                arg_sort = np.argsort(logits)[0]

                image = self.transform_un_normalize(image_inputs)
                cam_list, cam_list1, cam_list2, cam_list3 = self.generate_cam(
                    weights=self.net.head_linear.weight, features=out_features,
                    indexes=arg_sort, image_size=inputs.shape[-2:])
                cam_list4 = (np.asarray(cam_list1) - 0.5) * 2
                cam_list4[cam_list4 < 0] = 0

                result_path = Tools.new_dir(os.path.join(save_path, os.path.basename(image_filename).replace(".JPEG", "")))
                image.save(os.path.join(result_path, "0.JPEG"))
                # Image.fromarray(np.asarray(cam_list[0][0] * 255, dtype=np.uint8)).save(os.path.join(result_path, "00.bmp"))
                # Image.fromarray(np.asarray(cam_list[0][1] * 255, dtype=np.uint8)).save(os.path.join(result_path, "01.bmp"))
                Image.fromarray(np.asarray(cam_list1[0][0] * 255, dtype=np.uint8)).save(os.path.join(result_path, "10.bmp"))
                Image.fromarray(np.asarray(cam_list1[0][1] * 255, dtype=np.uint8)).save(os.path.join(result_path, "11.bmp"))
                Image.fromarray(np.asarray(cam_list2[0][0] * 255, dtype=np.uint8)).save(os.path.join(result_path, "20.bmp"))
                Image.fromarray(np.asarray(cam_list2[0][1] * 255, dtype=np.uint8)).save(os.path.join(result_path, "21.bmp"))
                Image.fromarray(np.asarray(cam_list3[0][0] * 255, dtype=np.uint8)).save(os.path.join(result_path, "30.bmp"))
                Image.fromarray(np.asarray(cam_list3[0][1] * 255, dtype=np.uint8)).save(os.path.join(result_path, "31.bmp"))
                Image.fromarray(np.asarray(cam_list4[0][0] * 255, dtype=np.uint8)).save(os.path.join(result_path, "40.bmp"))
                Image.fromarray(np.asarray(cam_list4[0][1] * 255, dtype=np.uint8)).save(os.path.join(result_path, "41.bmp"))

                for index, arg_one in enumerate(arg_sort):
                    Tools.print("{:3d} {:3d} {:.4f}".format(index, arg_one, logits[0][arg_one]))
                    pass
                Tools.print(image_filename)
                pass
            pass
        pass

    @classmethod
    def generate_cam(cls, weights, features, indexes, image_size):
        bz, nc, h, w = features.shape
        cam_list, cam_list1, cam_list2, cam_list3 = [], [], [], []
        for i in range(bz):
            cam_i, cam_i1, cam_i2, cam_i3 = [], [], [], []
            for index in indexes:
                # CAM
                # weight = weights[index].view(nc, 1, 1).expand_as(features[i])
                # cam = torch.unsqueeze(torch.sum(torch.mul(weight, features[i]), dim=0, keepdim=True), 0)
                cam = torch.tensordot(weights[index], features[i], dims=((0,), (0,)))

                # resize
                cam = F.interpolate(torch.unsqueeze(torch.unsqueeze(cam, dim=0), dim=0),
                                    size=image_size, mode="bicubic")

                cam0 = torch.squeeze(torch.squeeze(cam, dim=0), dim=0).detach().cpu().numpy()

                # Sigmoid
                cam1 = torch.sigmoid(torch.squeeze(torch.squeeze(cam, dim=0), dim=0)).detach().cpu().numpy()

                # Norm
                cam2 = cls._feature_norm(cam)
                cam2 = torch.squeeze(torch.squeeze(cam2, dim=0), dim=0).detach().cpu().numpy()

                # Norm2
                cam3 = cls._feature_norm2(cam)
                cam3 = torch.squeeze(torch.squeeze(cam3, dim=0), dim=0).detach().cpu().numpy()

                cam_i.append(cam0)
                cam_i1.append(cam1)
                cam_i2.append(cam2)
                cam_i3.append(cam3)
                pass
            cam_list.append(cam_i)
            cam_list1.append(cam_i1)
            cam_list2.append(cam_i2)
            cam_list3.append(cam_i3)
            pass
        return cam_list, cam_list1, cam_list2, cam_list3

    @staticmethod
    def _feature_norm(feature_map):
        feature_shape = feature_map.size()
        batch_min, _ = torch.min(feature_map.view((feature_shape[0], -1)), dim=-1, keepdim=True)
        batch_max, _ = torch.max(feature_map.view((feature_shape[0], -1)), dim=-1, keepdim=True)
        norm = torch.div(feature_map.view((feature_shape[0], -1)) - batch_min, batch_max - batch_min)
        return norm.view(feature_shape)

    @staticmethod
    def _feature_norm2(feature_map):
        feature_shape = feature_map.size()
        batch_max, _ = torch.max(feature_map.view((feature_shape[0], -1)), dim=-1, keepdim=True)
        norm = torch.div(feature_map.view((feature_shape[0], -1)), batch_max)
        norm[norm < 0] = 0
        return norm.view(feature_shape)

    @staticmethod
    def grub_cut_mask(image, cam, label):
        x1, x2, x3 = np.percentile(cam, [15, 70, 99.5])
        new_mask = np.zeros(cam.shape, dtype=np.uint8)

        new_mask[cam > x3] = cv2.GC_FGD
        new_mask[cam <= x3] = cv2.GC_PR_FGD
        new_mask[cam <= x2] = cv2.GC_PR_BGD
        new_mask[cam <= x1] = cv2.GC_BGD

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        mask, _, _ = cv2.grabCut(image, new_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        return np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, label.item()).astype("uint8")

    def load_model(self, model_file_name):
        Tools.print("Load model form {}".format(model_file_name))
        checkpoint = torch.load(model_file_name)
        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1:
            checkpoint = {key.replace("module.", ""): checkpoint[key] for key in checkpoint}
        self.net.load_state_dict(checkpoint, strict=True)
        Tools.print("Restore from {}".format(model_file_name))
        pass

    @staticmethod
    def get_data_root_path():
        if "Linux" in platform.platform():
            data_root = '/mnt/4T/Data/data/L2ID/data'
            if not os.path.isdir(data_root):
                data_root = "/media/ubuntu/4T/ALISURE/Data/L2ID/data"
        else:
            data_root = "F:\\data\\L2ID\\data"
        return data_root

    pass


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    cam_runner = PersonCAMRunner(image_size=224, num_classes=2)

    data_root = os.path.join(cam_runner.get_data_root_path(), "ILSVRC2017_DET/ILSVRC/Data/DET")
    image_filename_list = [
        "train/ILSVRC2013_train/n00477639/n00477639_11150.JPEG",
        "train/ILSVRC2013_train/n00477639/n00477639_10591.JPEG",
        "train/ILSVRC2013_train/n00477639/n00477639_13940.JPEG",
        "train/ILSVRC2013_train/n00477639/n00477639_16814.JPEG",

        "train/ILSVRC2014_train_0006/ILSVRC2014_train_00060002.JPEG",
        "train/ILSVRC2014_train_0006/ILSVRC2014_train_00060027.JPEG",
        "train/ILSVRC2014_train_0006/ILSVRC2014_train_00060024.JPEG",
        "train/ILSVRC2014_train_0006/ILSVRC2014_train_00060127.JPEG",

        "train/ILSVRC2013_train/n01443537/n01443537_1219.JPEG",
        "train/ILSVRC2013_train/n01443537/n01443537_1306.JPEG",
        "train/ILSVRC2013_train/n01443537/n01443537_14365.JPEG",
        "train/ILSVRC2013_train/n01498041/n01498041_190.JPEG",
        "train/ILSVRC2013_train/n01498041/n01498041_21.JPEG",
    ]
    all_image_file = [os.path.join(data_root, image_filename) for image_filename in image_filename_list]

    cam_runner.demo_mlc_cam(image_filename_list=all_image_file,
                            save_path="../../../WSS_CAM/1_ClassNet_2_50_144_5_224/person_45",
                            model_file_name="../../../WSS_Model_Person/1_ClassNet_2_50_144_5_224/person_45.pth")
    pass
