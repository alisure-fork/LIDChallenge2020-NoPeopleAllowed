import os
import sys
import glob
import torch
import random
import joblib
import platform
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import multiprocessing
import torch.optim as optim
import torch.nn.functional as F
from alisuretool.Tools import Tools
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
sys.path.append("../../")
from my_util_crf import DenseCRF
from my_util_data2 import DatasetUtil, DataUtil
from deep_labv3plus_pytorch.metrics import StreamSegMetrics


class VOCRunner(object):

    def __init__(self, config):
        self.config = config
        pass

    def inference_crf(self, dataset, logits_path):
        logit_file_path = Tools.new_dir("{}_logit".format(logits_path))
        crf_file_path = Tools.new_dir("{}_crf".format(logits_path))
        crf_final_file_path = Tools.new_dir("{}_crf_final".format(logits_path))

        postprocessor = DenseCRF()
        n_jobs = multiprocessing.cpu_count()

        def process(i):
            image_info, label_info = dataset.__getitem__(i)
            label = Image.fromarray(np.zeros_like(np.asarray(Image.open(image_info)))).convert("L") \
                if label_info == 1 else Image.open(label_info)

            basename = os.path.basename(image_info)
            im = Image.open(image_info)
            logit = np.load(os.path.join(logit_file_path, basename.replace(".jpg", ".npy")))

            ori_size = (im.size[1], im.size[0])
            crf_size = (logit.shape[1], logit.shape[2])

            logit_tensor = torch.FloatTensor(logit)[None, ...]
            logit_tensor = self._up_to_target(logit_tensor, target_size=crf_size)
            prob_one = F.softmax(logit_tensor, dim=1)[0].numpy()

            prob_crf = postprocessor(np.array(im.resize((crf_size[1], crf_size[0]))), prob_one)
            prob_crf_resize = self._up_to_target(torch.FloatTensor(prob_crf)[None, ...], target_size=ori_size)
            result = np.argmax(prob_crf_resize[0].numpy(), axis=0)

            # save
            im.save(os.path.join(crf_file_path, basename))
            DataUtil.gray_to_color(np.asarray(label, dtype=np.uint8)).save(
                os.path.join(crf_file_path, basename.replace(".jpg", "_l.png")))
            DataUtil.gray_to_color(np.asarray(result, dtype=np.uint8)).save(
                os.path.join(crf_file_path, basename.replace(".jpg", ".png")))
            Image.fromarray(np.asarray(result, dtype=np.uint8)).save(
                os.path.join(crf_final_file_path, basename.replace(".jpg", ".png")))

            return result, np.array(label)

        results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
            [joblib.delayed(process)(i) for i in range(len(dataset))])

        metrics = StreamSegMetrics(self.config.ss_num_classes)
        for preds, targets in results:
            metrics.update(targets, preds)
        Tools.print("{}".format(metrics.to_str(metrics.get_results())))
        Tools.print()
        pass

    @staticmethod
    def _up_to_target(source, target_size, mode="bilinear"):
        if source.size()[2] != target_size[0] or source.size()[3] != target_size[1]:
            align_corners = True if mode == "nearest" else False
            _source = torch.nn.functional.interpolate(source, size=target_size, mode=mode, align_corners=align_corners)
            pass
        else:
            _source = source
        return _source

    pass


def train(config):
    voc_runner = VOCRunner(config=config)

    dataset_ss_inference_train, dataset_ss_inference_val, dataset_ss_inference_test = DatasetUtil.get_dataset_by_type(
        DatasetUtil.dataset_type_ss_voc_crf, config.ss_size, data_root=config.data_root_path)

    voc_runner.inference_crf(dataset=dataset_ss_inference_train,
                             logits_path=Tools.new_dir(os.path.join(config.logits_path, "train")))
    # voc_runner.inference_crf(dataset=dataset_ss_inference_val,
    #                          logits_path=Tools.new_dir(os.path.join(config.logits_path, "val")))
    # voc_runner.inference_crf(dataset=dataset_ss_inference_test,
    #                          logits_path=Tools.new_dir(os.path.join(config.logits_path, "test")))
    pass


class Config(object):

    def __init__(self):
        self.ss_num_classes = 21
        self.ss_size = 352
        self.data_root_path = self.get_data_root_path()
        self.logits_path = "../../../WSS_Model_VOC_EVAL/5_DeepLabV3PlusResNet101_21_100_48_5_352/ss_100_scales_5"
        pass

    @staticmethod
    def get_data_root_path():
        if "Linux" in platform.platform():
            data_root = '/mnt/4T/Data/data/SS/voc'
            if not os.path.isdir(data_root):
                data_root = "/media/ubuntu/4T/ALISURE/Data/SS/voc"
        else:
            data_root = "F:\\data\\SS\\voc"
        return data_root

    pass


"""
4 GPU
../../../WSS_Model_VOC/5_DeepLabV3PlusResNet101_21_100_48_5_352/ss_final_100.pth
val
Overall Acc: 0.906540
Mean Acc: 0.773918
FreqW Acc: 0.835605
Mean IoU: 0.660385
crf val
Overall Acc: 0.913418
Mean Acc: 0.778007
FreqW Acc: 0.846002
Mean IoU: 0.677293

Train
Overall Acc: 0.879515
Mean Acc: 0.767492
FreqW Acc: 0.790510
Mean IoU: 0.650795
crf train
Overall Acc: 0.884620
Mean Acc: 0.766945
FreqW Acc: 0.797796
Mean IoU: 0.659136
"""


if __name__ == '__main__':
    config = Config()
    train(config=config)
    pass
