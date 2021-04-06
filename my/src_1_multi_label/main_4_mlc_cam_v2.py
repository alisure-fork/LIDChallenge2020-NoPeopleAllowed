import os
import sys
import cv2
import glob
import torch
import random
import platform
import warnings
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import multiprocessing
from util_crf import CRFTool
import torch.nn.functional as F
from alisuretool.Tools import Tools
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from util_data import DatasetUtil, MyTransform, DataUtil
sys.path.append("../../")
from util_network import CAMNet


class CAMRunner(object):

    def __init__(self, config):
        self.config = config
        pass

    def eval_mlc_cam_1(self):
        net = CAMNet(num_classes=self.config.mlc_num_classes).cuda()

        dataset_vis_cam = DatasetUtil.get_dataset_by_type(
            DatasetUtil.dataset_type_vis_cam, image_size=self.config.mlc_size,
            data_root=self.config.data_root_path, return_image_info=True, sampling=False)
        data_loader_vis_cam = DataLoader(dataset_vis_cam, self.config.mlc_batch_size, shuffle=False, num_workers=16)

        Tools.print("Load model form {}".format(self.config.model_file_name))
        self.load_model(net=net, model_file_name=self.config.model_file_name)

        net.eval()
        with torch.no_grad():
            for _, (inputs, labels, image_paths) in tqdm(enumerate(data_loader_vis_cam), total=len(data_loader_vis_cam)):
                inputs_cuda = inputs.float().cuda()
                logits, out_features = net.forward(inputs_cuda, is_vis=True)
                logits = logits.detach().cpu().numpy()

                # 标签选择策略
                label_for_cam = self.label_select_strategy(
                    logits=logits, image_labels=labels.numpy(), thr=self.config.top_k_thr)
                # 生成 CAM
                cam_list = self.generate_cam(net.head_linear.weight, features=out_features, indexes=label_for_cam)

                for input_index, input_one in enumerate(inputs):
                    image_path_one = image_paths[input_index]
                    now_name = image_path_one.split("Data/DET/")[1]
                    result_filename = Tools.new_dir(os.path.join(self.config.mlc_cam_pkl_dir, now_name))

                    label_one = labels[input_index].numpy()
                    label_for_cam_one = label_for_cam[input_index]
                    cam_one = cam_list[input_index]
                    Tools.write_to_pkl(_path=result_filename.replace(".JPEG", ".pkl"),
                                       _data={"label": label_one, "image_path": image_path_one,
                                              "label_for_cam": label_for_cam_one, "cam": cam_one})
                    pass

                pass
            pass

        pass

    def eval_mlc_cam_2(self):
        all_pkl = glob.glob(os.path.join(self.config.mlc_cam_pkl_dir, "**/*.pkl"), recursive=True)
        if self.config.sampling:
            all_pkl = all_pkl[::50]
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
        for index, one_pkl in tqdm(enumerate(all_pkl), total=len(all_pkl)):
            pool.apply_async(self._eval_mlc_cam_2_inner, args=(index, one_pkl))
            #######################################################################################################
            pass
        pool.close()
        pool.join()
        pass

    def _eval_mlc_cam_2_inner(self, index, pkl_path):
        if index % 1000 == 0:
            Tools.print("now is {}".format(index))
            pass

        pkl_data = Tools.read_from_pkl(pkl_path)
        label_one = pkl_data["label"]
        image_path_one = pkl_data["image_path"]
        label_for_cam_one = pkl_data["label_for_cam"]
        cam_one = pkl_data["cam"]

        im = Image.open(image_path_one)
        image_size = im.size

        now_name = image_path_one.split("Data/DET/")[1]
        result_filename = Tools.new_dir(os.path.join(self.config.mlc_cam_dir, now_name))
        # 保存原图
        # im.save(result_filename)

        # 预测结果, 对结果进行彩色可视化
        np_single_cam = 0
        np_cam = np.zeros(shape=(self.config.mlc_num_classes + 1, image_size[1], image_size[0]))
        for label in label_for_cam_one:
            image_input = np.asarray(im.resize((self.config.mlc_size, self.config.mlc_size)))
            tensor_cam = torch.tensor(cam_one[label])
            norm_cam = torch.squeeze(self._feature_norm(torch.unsqueeze(tensor_cam, dim=0)))

            now_cam_im = Image.fromarray(np.asarray(norm_cam * 255, dtype=np.uint8)).resize(size=image_size)
            # now_cam_im.save(result_filename.replace(".JPEG", "_{}.bmp".format(label + 1)))
            np_single_cam += np.asarray(now_cam_im, dtype=np.float)

            cam_crf_one = self.torch_resize(norm_cam, size=(self.config.mlc_size, self.config.mlc_size))
            cam_crf_one = CRFTool.crf(image_input, np.expand_dims(cam_crf_one, axis=0), t=5)
            now_cam_crf_im = Image.fromarray(np.asarray(cam_crf_one * 255, dtype=np.uint8)).resize(size=image_size)
            # now_cam_crf_im.save(result_filename.replace(".JPEG", "_crf_{}.bmp".format(label + 1)))

            np_cam[label + 1] = np.asarray(now_cam_im) / 2 + np.asarray(now_cam_crf_im) / 2
            pass

        np_cam[0] = self.config.fg_thr * 255  # 0.7
        cam_label = np.asarray(np.argmax(np_cam, axis=0), dtype=np.uint8)
        cam_label[cam_label == 0] = 255
        if len(label_for_cam_one) > 0:
            cam_label[(np_single_cam / len(label_for_cam_one)) < self.config.bg_thr * 255] = 0  # 0.1
            pass

        im_color = DataUtil.gray_to_color(cam_label).resize(size=image_size, resample=Image.NEAREST)
        im_color.save(result_filename.replace("JPEG", "png"))
        pass

    def _eval_mlc_cam_2_inner_old(self, index, pkl_path):
        if index % 1000 == 0:
            Tools.print("now is {}".format(index))
            pass

        pkl_data = Tools.read_from_pkl(pkl_path)
        label_one = pkl_data["label"]
        image_path_one = pkl_data["image_path"]
        label_for_cam_one = pkl_data["label_for_cam"]
        cam_one = pkl_data["cam"]

        im = Image.open(image_path_one)
        image_size = im.size

        now_name = image_path_one.split("Data/DET/")[1]
        result_filename = Tools.new_dir(os.path.join(self.config.mlc_cam_dir, now_name))
        # 保存原图
        im.save(result_filename)

        # 预测结果, 对结果进行彩色可视化
        np_cam = np.zeros(shape=(self.config.mlc_num_classes + 1, image_size[1], image_size[0]))
        np_cam[0] = self.config.fg_thr * 255
        for label in label_for_cam_one:
            image_input = np.asarray(im.resize((self.config.mlc_size, self.config.mlc_size)))
            tensor_cam = torch.tensor(cam_one[label])
            # cam = torch.sigmoid(tensor_cam)
            norm_cam = torch.squeeze(self._feature_norm(torch.unsqueeze(tensor_cam, dim=0)))

            now_cam_im = Image.fromarray(np.asarray(norm_cam * 255, dtype=np.uint8)).resize(size=image_size)
            now_cam_im.save(result_filename.replace(".JPEG", "_{}.bmp".format(label + 1)))
            np_cam[label + 1] = np.asarray(now_cam_im)

            cam_crf_one = self.torch_resize(norm_cam, size=(self.config.mlc_size, self.config.mlc_size))
            cam_crf_one = CRFTool.crf(image_input, np.expand_dims(cam_crf_one, axis=0), t=5)
            now_cam_crf_im = Image.fromarray(np.asarray(cam_crf_one * 255, dtype=np.uint8)).resize(size=image_size)
            now_cam_crf_im.save(result_filename.replace(".JPEG", "_crf_{}.bmp".format(label + 1)))

            # cam_cut_one = self.torch_resize(tensor_cam, size=(self.config.mlc_size, self.config.mlc_size))
            # cam_cut_one = self.grub_cut_mask(image_input, cam_cut_one)
            # now_cam_cut_im = Image.fromarray(np.asarray(cam_cut_one * 255, dtype=np.uint8)).resize(size=image_size)
            # now_cam_cut_im.save(result_filename.replace(".JPEG", "_{}_cut.bmp".format(label + 1)))
            pass

        cam_label = np.asarray(np.argmax(np_cam, axis=0), dtype=np.uint8)
        im_color = DataUtil.gray_to_color(cam_label).resize(size=image_size, resample=Image.NEAREST)
        im_color.save(result_filename.replace("JPEG", "png"))
        pass

    @classmethod
    def generate_cam(cls, weights, features, indexes):
        cam_list = []
        for i, (feature, index_list) in enumerate(zip(features, indexes)):
            cam_one_list = {}
            for index in index_list:
                cam = torch.tensordot(weights[index], feature, dims=((0,), (0,)))
                # cam = F.interpolate(torch.unsqueeze(torch.unsqueeze(cam, dim=0), dim=0),
                #                     size=image_size, mode="bilinear")
                cam = cam.detach().cpu().numpy()
                # cam = torch.sigmoid(cam).detach().cpu().numpy()
                # cam = cls._feature_norm(cam).detach().cpu().numpy()
                cam_one_list[index] = cam
                pass
            cam_list.append(cam_one_list)
            pass
        return cam_list

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

    @staticmethod
    def _feature_norm(feature_map):
        feature_shape = feature_map.size()
        batch_min, _ = torch.min(feature_map.view((feature_shape[0], -1)), dim=-1, keepdim=True)
        batch_max, _ = torch.max(feature_map.view((feature_shape[0], -1)), dim=-1, keepdim=True)
        norm = torch.div(feature_map.view((feature_shape[0], -1)) - batch_min, batch_max - batch_min)
        return norm.view(feature_shape)

    @staticmethod
    def torch_resize(input_one, size):
        output_one = torch.squeeze(F.interpolate(torch.unsqueeze(torch.unsqueeze(
            torch.tensor(input_one), dim=0), dim=0), size=size, mode="bilinear", align_corners=False))
        return np.asarray(output_one)

    @staticmethod
    def load_model(net, model_file_name):
        Tools.print("Load model form {}".format(model_file_name))
        checkpoint = torch.load(model_file_name)
        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1:
            checkpoint = {key.replace("module.", ""): checkpoint[key] for key in checkpoint}
        net.load_state_dict(checkpoint, strict=True)
        Tools.print("Restore from {}".format(model_file_name))
        pass

    pass


class Config(object):

    def __init__(self):
        self.gpu_id = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        warnings.filterwarnings('ignore')

        self.mlc_num_classes = 200
        self.mlc_batch_size = 32 * len(self.gpu_id.split(","))
        self.bg_thr = 0.1
        self.fg_thr = 0.7
        self.top_k_thr = 0.5

        self.data_root_path = self.get_data_root_path()

        self.Net = CAMNet

        # self.mlc_size = 224
        # self.model_file_name = "../../../WSS_Model/demo_CAMNet_200_60_128_5_224/mlc_final_60.pth"
        # self.mlc_size = 256
        # self.model_file_name = "../../../WSS_Model/1_CAMNet_200_60_128_5_256/mlc_20.pth"
        # self.sampling = True
        self.mlc_size = 256
        self.model_file_name = "../../../WSS_Model/1_CAMNet_200_15_96_2_224/mlc_final_15.pth"
        self.sampling = False

        run_name = "2"
        self.model_name = "{}_{}_{}_{}_{}_{}".format(
            run_name, "CAMNet", self.mlc_num_classes, self.mlc_batch_size, self.mlc_size, self.top_k_thr)
        Tools.print(self.model_name)

        self.mlc_cam_dir = "../../../WSS_CAM/cam/{}".format(self.model_name)
        self.mlc_cam_pkl_dir = "../../../WSS_CAM/pkl/{}".format(self.model_name)
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
    # cam_runner.eval_mlc_cam_1()
    cam_runner.eval_mlc_cam_2()
    pass
