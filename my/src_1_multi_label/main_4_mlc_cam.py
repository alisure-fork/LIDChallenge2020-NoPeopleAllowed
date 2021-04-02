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
from util_network import CAMNet
from alisuretool.Tools import Tools
from torch.utils.data import DataLoader, Dataset
from util_data import DatasetUtil, MyTransform, DataUtil


class CAMRunner(object):

    def __init__(self, config):
        self.config = config
        self.net = CAMNet(num_classes=self.config.mlc_num_classes).cuda()

        self.dataset_mlc_val = DatasetUtil.get_dataset_by_type(
            DatasetUtil.dataset_type_mlc_val, image_size=self.config.mlc_size,
            data_root=self.config.data_root_path, return_image_info=True)
        self.data_loader_mlc_val = DataLoader(self.dataset_mlc_val, self.config.mlc_batch_size,
                                                shuffle=False, num_workers=16)
        pass

    @staticmethod
    def label_select_strategy(logits, image_labels, thr=0.5):
        # 1. 每个图片取top_k
        # logits_sort = np.argsort(logits)[:, ::-1][:,:self.config.top_k]

        # 2. label + logits>th
        ok_value = np.where((logits + image_labels) > thr)
        logits_sort = [[] for _ in range(len(logits))]
        for one in zip(*ok_value):
            logits_sort[one[0]].append(one[1])
            pass
        return logits_sort

    def eval_mlc_cam(self):
        Tools.print("Load model form {}".format(self.config.model_file_name))
        self.load_model(self.config.model_file_name)

        self.net.eval()
        with torch.no_grad():
            for i, (inputs, labels, image_paths) in tqdm(enumerate(self.data_loader_mlc_val),
                                                         total=len(self.data_loader_mlc_val)):
                inputs = inputs.float().cuda()
                logits, out_features = self.net.forward_map(inputs)
                logits = logits.detach().cpu().numpy()

                # 标签选择策略
                label_for_cam = self.label_select_strategy(
                    logits=logits, image_labels=labels.numpy(), thr=self.config.top_k_thr)
                # 生成CAM
                np_cam = self.generate_cam(weights=self.net.head_linear.weight, features=out_features,
                                           indexes=label_for_cam, image_size=inputs.shape[-2:],
                                           num_classes=self.config.mlc_num_classes, min_thr=self.config.bg_thr)

                # 可视化
                for cam, image_path in zip(np_cam, image_paths):
                    # 预测结果
                    cam_label = np.asarray(np.argmax(cam, axis=0), dtype=np.uint8)

                    # 对结果进行彩色可视化
                    im_color = DataUtil.gray_to_color(cam_label)
                    im_color = im_color.resize(size=Image.open(image_path).size, resample=Image.NEAREST)
                    im_color.save(Tools.new_dir(os.path.join(
                        self.config.mlc_cam_dir, image_path.split("Data/DET/")[1].replace("JPEG", "png"))))
                    pass
                pass
            pass
        pass

    @staticmethod
    def generate_cam(weights, features, indexes, image_size, num_classes, min_thr=0.5):
        np_cam = np.zeros(shape=(len(features), num_classes + 1, image_size[0], image_size[1]))
        np_cam[:, 0] = min_thr
        for i, (feature, index_list) in enumerate(zip(features, indexes)):
            for index in index_list:
                cam = torch.tensordot(weights[index], feature, dims=((0,), (0,)))
                cam = F.interpolate(torch.unsqueeze(torch.unsqueeze(cam, dim=0), dim=0),
                                    size=image_size, mode="bicubic")
                np_cam[i, index + 1] = torch.sigmoid(cam).detach().cpu().numpy()
                pass
            pass
        return np_cam

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

    pass


class Config(object):

    def __init__(self):
        self.gpu_id = "3"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        self.mlc_num_classes = 200
        self.mlc_batch_size = 2 * len(self.gpu_id.split(","))
        self.bg_thr = 0.5
        # self.top_k = 3
        self.top_k_thr = 0.5

        self.data_root_path = self.get_data_root_path()

        self.Net = CAMNet

        # self.mlc_size = 224
        # self.model_file_name = "../../../WSS_Model/demo_CAMNet_200_60_128_5_224/mlc_final_60.pth"

        self.mlc_size = 256
        self.model_file_name = "../../../WSS_Model/1_CAMNet_200_60_128_5_256/mlc_40.pth"

        run_name = "1"
        self.model_name = "{}_{}_{}_{}_{}_{}_{}".format(
            run_name, "CAMNet", self.mlc_num_classes, self.mlc_batch_size, self.mlc_size, self.bg_thr, self.top_k_thr)
        Tools.print(self.model_name)

        self.mlc_cam_dir = "../../../WSS_CAM/{}".format(self.model_name)
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
    config = Config()
    cam_runner = CAMRunner(config)
    cam_runner.eval_mlc_cam()
    pass
