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
from util_network import CAMNet, ClassNet
warnings.filterwarnings('ignore')


class CAMRunner(object):

    def __init__(self, config):
        self.config = config
        pass

    def eval_mlc_cam_1(self):
        net = self.config.Net(num_classes=self.config.mlc_num_classes)
        # net = nn.DataParallel(net).cuda()
        # cudnn.benchmark = True
        net = net.cuda()

        _, _, dataset_cam = DatasetUtil.get_dataset_by_type(
            DatasetUtil.dataset_type_mlc, image_size=self.config.mlc_size, scales=self.config.scales,
            data_root=self.config.data_root_path, return_image_info=True, sampling=self.config.sampling)
        data_loader_cam = DataLoader(dataset_cam, self.config.mlc_batch_size, shuffle=False, num_workers=16)
        Tools.print("image num: {}".format(len(dataset_cam)))

        Tools.print("Load model form {}".format(self.config.model_file_name))
        self.load_model(net=net, model_file_name=self.config.model_file_name)

        net.eval()
        with torch.no_grad():
            for _, (inputs, labels, image_paths) in tqdm(enumerate(data_loader_cam), total=len(data_loader_cam)):
                all_logits = 0
                all_features = []
                for input_one in inputs:
                    input_one_cuda = input_one.float().cuda()
                    logits, out_features = net.forward(input_one_cuda, is_vis=True)
                    all_logits += torch.sigmoid(logits).detach().cpu().numpy()
                    all_features.append(out_features)
                    pass
                logits = all_logits / len(inputs)

                # 标签选择策略
                label_for_cam = self.label_select_strategy(
                    logits=logits, image_labels=labels.numpy(), thr=self.config.top_k_thr)
                # 生成 CAM
                cam_list = self.generate_cam(all_features=all_features, indexes=label_for_cam)

                for input_index, image_path_one in enumerate(image_paths):
                    now_name = image_path_one.split("Data/DET/")[1]
                    result_filename = Tools.new_dir(os.path.join(self.config.mlc_cam_pkl_dir, now_name))

                    cam_one = cam_list[input_index]
                    label_one = labels[input_index].numpy()
                    label_for_cam_one = label_for_cam[input_index]

                    Tools.write_to_pkl(_path=result_filename.replace(".JPEG", ".pkl"),
                                       _data={"label": label_one, "image_path": image_path_one,
                                              "label_for_cam": label_for_cam_one, "cam": cam_one})
                    pass

                pass
            pass

        pass

    def eval_mlc_cam_2(self):
        all_pkl = glob.glob(os.path.join(self.config.mlc_cam_pkl_dir, "**/*.pkl"), recursive=True)
        Tools.print("pkl num: {}".format(len(all_pkl)))
        if self.config.sampling:
            all_pkl = all_pkl[::50]
            Tools.print("sampling: {}".format(len(all_pkl)))
            pass

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

        try:
            if not os.path.exists(pkl_path):
                return

            pkl_data = Tools.read_from_pkl(pkl_path)
            label_one = pkl_data["label"]
            image_path_one = pkl_data["image_path"]
            label_for_cam_one = pkl_data["label_for_cam"]
            cam_one = pkl_data["cam"]

            now_name = image_path_one.split("Data/DET/")[1]
            result_filename = Tools.new_dir(os.path.join(self.config.mlc_cam_dir, now_name))

            if os.path.exists(result_filename.replace("JPEG", "png")):
                return

            im = Image.open(image_path_one)
            im.save(result_filename)
            image_size = im.size

            # 预测结果, 对结果进行彩色可视化
            np_single_cam = 0
            np_cam = np.zeros(shape=(self.config.mlc_num_classes + 1, image_size[1], image_size[0]))
            np_cam[0] = self.config.fg_thr * 255  # 0.25
            for label in label_for_cam_one:
                cam_resize = [self.torch_resize(cam, (self.config.mlc_size, self.config.mlc_size)) for cam in cam_one[label]]
                norm_cam = np.sum(cam_resize, axis=0) / len(cam_resize)
                norm_cam = norm_cam / np.max(norm_cam)

                now_cam_im = Image.fromarray(np.asarray(norm_cam * 255, dtype=np.uint8)).resize(size=image_size)
                now_cam_im.save(result_filename.replace(".JPEG", "_{}.bmp".format(label + 1)))
                np_cam[label + 1] = np.asarray(now_cam_im)

                np_single_cam += np.asarray(now_cam_im, dtype=np.float)
                pass

            # cam_crf_one = CRFTool.crf_inference(np.asarray(im), np_cam / 255, t=5, n_label=len(np_cam))
            # cam_crf_one = np.asarray(np.argmax(cam_crf_one, axis=0), dtype=np.uint8)
            # now_cam_crf_im = DataUtil.gray_to_color(cam_crf_one)
            # now_cam_crf_im.save(result_filename.replace(".JPEG", "_crf.png"))

            cam_label = np.asarray(np.argmax(np_cam, axis=0), dtype=np.uint8)
            cam_label[cam_label == 0] = 255
            if len(label_for_cam_one) > 0:
                cam_label[(np_single_cam / len(label_for_cam_one)) < self.config.bg_thr * 255] = 0  # 0.05
                pass
            im_color = DataUtil.gray_to_color(cam_label).resize(size=image_size, resample=Image.NEAREST)
            im_color.save(result_filename.replace("JPEG", "png"))
        except Exception():
            Tools.print("{} {}".format(index, pkl_path))
            pass
        pass

    @classmethod
    def generate_cam(cls, all_features, indexes):
        cam_list = []
        for index in range(len(all_features[0])):
            cam_one_dict = {}
            for label in indexes[index]:
                if label not in cam_one_dict:
                    cam_one_dict[label] = []
                for features in all_features:
                    cam = features[index][label]
                    cam /= torch.max(cam) + 1e-5
                    cam_one_dict[label].append(cam.detach().cpu().numpy())
                    pass
                pass
            cam_list.append(cam_one_dict)
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
        # checkpoint = {key.replace("module.", ""): checkpoint[key] for key in checkpoint}

        net.load_state_dict(checkpoint, strict=True)
        Tools.print("Restore from {}".format(model_file_name))
        pass

    pass


class Config(object):

    def __init__(self):
        self.gpu_id = "0"
        # self.gpu_id = "1, 2, 3"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        self.mlc_batch_size = 32 * len(self.gpu_id.split(","))
        self.top_k_thr = 0.5

        self.bg_thr = 0.05
        self.fg_thr = 0.33

        self.data_root_path = self.get_data_root_path()

        self.Net = ClassNet
        self.sampling = False

        self.mlc_size = 256
        self.scales = (1.0, 0.5, 1.5, 2.0)

        self.mlc_num_classes = 200
        self.model_file_name = "../../../WSS_Model/1_CAMNet_200_50_96_5_224/mlc_final_50.pth"

        # self.mlc_num_classes = 199
        # self.model_file_name = "../../../WSS_Model_No_Person/1_CAMNet_199_80_96_5_224/mlc_75.pth"

        run_name = "1"
        self.model_name = "{}_{}_{}_{}_{}".format(
            run_name, self.mlc_num_classes, self.mlc_batch_size, self.mlc_size, self.top_k_thr)
        Tools.print(self.model_name)

        self.mlc_cam_dir = "../../../WSS_CAM/cam_{}/3_{}".format(len(self.scales), self.model_name)
        self.mlc_cam_pkl_dir = "../../../WSS_CAM/pkl_{}/{}".format(len(self.scales), self.model_name)
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
