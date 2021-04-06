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
from util_crf import CRFTool
import torch.nn.functional as F
from alisuretool.Tools import Tools
from torch.utils.data import DataLoader, Dataset
from util_data import DatasetUtil, MyTransform, DataUtil
sys.path.append("../../")
from util_network import CAMNet


class CAMRunner(object):

    def __init__(self, config):
        self.config = config
        self.net = CAMNet(num_classes=self.config.mlc_num_classes).cuda()

        self.dataset_vis_cam = DatasetUtil.get_dataset_by_type(
            DatasetUtil.dataset_type_vis_cam, image_size=self.config.mlc_size,
            data_root=self.config.data_root_path, return_image_info=True, sampling=self.config.sampling)
        self.data_loader_vis_cam = DataLoader(self.dataset_vis_cam, self.config.mlc_batch_size,
                                                shuffle=False, num_workers=16)
        self.transform_un_normalize = MyTransform.transform_un_normalize()
        pass

    def eval_mlc_cam(self):
        Tools.print("Load model form {}".format(self.config.model_file_name))
        self.load_model(self.config.model_file_name)

        self.net.eval()
        with torch.no_grad():
            for _, (inputs, labels, image_paths) in tqdm(enumerate(self.data_loader_vis_cam),
                                                         total=len(self.data_loader_vis_cam)):
                inputs_cuda = inputs.float().cuda()
                logits, out_features = self.net.forward(inputs_cuda, is_vis=True)
                logits = logits.detach().cpu().numpy()

                # 标签选择策略
                label_for_cam = self.label_select_strategy(
                    logits=logits, image_labels=labels.numpy(), thr=self.config.top_k_thr)
                # 生成 CAM
                np_cam = self.generate_cam(weights=self.net.head_linear.weight, features=out_features,
                                           indexes=label_for_cam, image_size=inputs_cuda.shape[-2:],
                                           num_classes=self.config.mlc_num_classes, bg_thr=self.config.bg_thr)
                # CRF 或 GrubCut 细化 CAM
                np_cam_crf = np.zeros_like(np_cam)
                np_cam_cut = np.zeros_like(np_cam)
                for bz_i, (cam, label_for_cam_one, input_one) in enumerate(zip(np_cam, label_for_cam, inputs)):
                    image_input = np.asarray(self.transform_un_normalize(input_one))
                    for label_index, label_one in enumerate(label_for_cam_one):
                        # CRF
                        result_one = CRFTool.crf(image_input, np.expand_dims(cam[label_one + 1], axis=0), t=5)
                        np_cam_crf[bz_i, label_one + 1] = result_one
                        # GrubCut
                        result_one = self.grub_cut_mask(image_input, cam[label_one + 1])
                        np_cam_cut[bz_i, label_one + 1] = result_one
                        pass
                    pass

                #######################################################################################################
                # 可视化
                for bz_i, (cam, image_path) in enumerate(zip(np_cam, image_paths)):
                    # 预测结果
                    cam_label = np.asarray(np.argmax(cam, axis=0), dtype=np.uint8)

                    now_name = image_path.split("Data/DET/")[1]
                    result_filename = Tools.new_dir(os.path.join(self.config.mlc_cam_dir, now_name))

                    # 保存原图
                    Image.open(image_path).save(result_filename)
                    # 训练数据逆增强
                    image_input = self.transform_un_normalize(inputs[bz_i])
                    image_input.save(result_filename.replace(".JPEG", "_i.JPEG"))
                    # 对结果进行彩色可视化
                    im_color = DataUtil.gray_to_color(cam_label)
                    im_color = im_color.resize(size=Image.open(image_path).size, resample=Image.NEAREST)
                    im_color.save(result_filename.replace("JPEG", "png"))
                    # 对CAM进行可视化
                    for label_index, label_one in enumerate(label_for_cam[bz_i]):
                        now_cam_im = Image.fromarray(np.asarray(cam[label_one + 1] * 255, dtype=np.uint8))
                        now_cam_im.save(result_filename.replace(".JPEG", "_{}.bmp".format(label_one + 1)))

                        now_cam_crf_im = Image.fromarray(np.asarray(np_cam_crf[bz_i][label_one + 1] * 255, dtype=np.uint8))
                        now_cam_crf_im.save(result_filename.replace(".JPEG", "_{}_crf.bmp".format(label_one + 1)))

                        now_cam_cut_im = Image.fromarray(np.asarray(np_cam_cut[bz_i][label_one + 1] * 255, dtype=np.uint8))
                        now_cam_cut_im.save(result_filename.replace(".JPEG", "_{}_cut.bmp".format(label_one + 1)))
                        pass
                    pass
                #######################################################################################################

                pass
            pass
        pass

    @classmethod
    def generate_cam(cls, weights, features, indexes, image_size, num_classes, bg_thr=0.5):
        np_cam = np.zeros(shape=(len(features), num_classes + 1, image_size[0], image_size[1]))
        np_cam[:, 0] = bg_thr
        for i, (feature, index_list) in enumerate(zip(features, indexes)):
            for index in index_list:
                cam = torch.tensordot(weights[index], feature, dims=((0,), (0,)))
                cam = F.interpolate(torch.unsqueeze(torch.unsqueeze(cam, dim=0), dim=0),
                                    size=image_size, mode="bicubic")
                # cam = torch.sigmoid(cam)
                cam = cls._feature_norm(cam)
                np_cam[i, index + 1] = cam.detach().cpu().numpy()
                pass
            pass
        return np_cam

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

    @staticmethod
    def _feature_norm(feature_map):
        feature_shape = feature_map.size()
        batch_min, _ = torch.min(feature_map.view((feature_shape[0], -1)), dim=-1, keepdim=True)
        batch_max, _ = torch.max(feature_map.view((feature_shape[0], -1)), dim=-1, keepdim=True)
        norm = torch.div(feature_map.view((feature_shape[0], -1)) - batch_min, batch_max - batch_min)
        return norm.view(feature_shape)

    @staticmethod
    def grub_cut_mask(image, cam):
        x1, x2, x3 = np.percentile(cam, [15, 70, 99.5])
        new_mask = np.zeros(cam.shape, dtype=np.uint8)

        new_mask[cam > x3] = cv2.GC_FGD
        new_mask[cam <= x3] = cv2.GC_PR_FGD
        new_mask[cam <= x2] = cv2.GC_PR_BGD
        new_mask[cam <= x1] = cv2.GC_BGD

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        mask, _, _ = cv2.grabCut(image, new_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        return np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype("uint8")

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
        self.mlc_batch_size = 32 * len(self.gpu_id.split(","))
        self.bg_thr = 0.7
        # self.top_k = 3
        self.top_k_thr = 0.3

        self.data_root_path = self.get_data_root_path()

        self.Net = CAMNet

        # self.mlc_size = 224
        # self.model_file_name = "../../../WSS_Model/demo_CAMNet_200_60_128_5_224/mlc_final_60.pth"
        # self.mlc_size = 256
        # self.model_file_name = "../../../WSS_Model/1_CAMNet_200_60_128_5_256/mlc_20.pth"
        # self.sampling = True
        self.mlc_size = 256
        self.model_file_name = "../../../WSS_Model/1_CAMNet_200_15_96_2_224/mlc_final_15.pth"
        self.sampling = True

        run_name = "2"
        self.model_name = "{}_{}_{}_{}_{}_{}_{}{}".format(
            run_name, "CAMNet", self.mlc_num_classes, self.mlc_batch_size, self.mlc_size, self.bg_thr,
            self.top_k_thr, "_sampling" if self.sampling else "")
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
