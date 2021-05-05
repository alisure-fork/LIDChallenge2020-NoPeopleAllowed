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
from my_util_network import DeepLabV3Plus, deeplabv3_resnet50, deeplabv3plus_resnet101


class VOCRunner(object):

    def __init__(self, config):
        self.config = config

        # Model
        self.net = DeepLabV3Plus(num_classes=self.config.ss_num_classes,
                                   output_stride=16, arch=deeplabv3plus_resnet101)
        self.net = nn.DataParallel(self.net).cuda()
        cudnn.benchmark = True
        pass

    def inference_crf(self, dataset, logits_path):
        logit_file_path = Tools.new_dir("{}_logit".format(logits_path))
        crf_file_path = Tools.new_dir("{}_crf".format(logits_path))
        crf_final_file_path = Tools.new_dir("{}_crf_final".format(logits_path))

        postprocessor = DenseCRF()
        n_jobs = multiprocessing.cpu_count()

        def process(i):
            image_list, mask, image_info, label_info = dataset.__getitem__(i)
            label = Image.fromarray(np.zeros_like(np.asarray(Image.open(image_info)))).convert("L") \
                if label_info == 1 else Image.open(label_info)

            basename = os.path.basename(image_info)
            im = Image.open(image_info)
            filename = os.path.join(logit_file_path, basename.replace(".jpg", ".npy"))
            logit = np.load(filename)
            logit = torch.FloatTensor(logit)[None, ...]

            prob_result = 0
            for image in image_list:
                logit_one = F.interpolate(logit, size=(image.shape[1], image.shape[2]), mode="bilinear", align_corners=False)
                prob_one = F.softmax(logit_one, dim=1)[0].numpy()
                im_data = np.array(im.resize((image.shape[2], image.shape[1])))
                prob_crf = postprocessor(im_data, prob_one)
                prob_crf_resize = F.interpolate(torch.FloatTensor(prob_crf)[None, ...],
                                                size=(im.size[1], im.size[0]), mode="bilinear", align_corners=False)
                prob_result += prob_crf_resize[0].numpy()
                pass

            result = np.argmax(prob_result, axis=0)

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

    def load_model(self, model_file_name):
        Tools.print("Load model form {}".format(model_file_name))
        checkpoint = torch.load(model_file_name)

        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1:
            # checkpoint = {key.replace("module.", ""): checkpoint[key] for key in checkpoint}
            pass

        self.net.load_state_dict(checkpoint, strict=True)
        Tools.print("Restore from {}".format(model_file_name))
        pass

    pass


def train(config):
    voc_runner = VOCRunner(config=config)

    dataset_ss_inference_train, dataset_ss_inference_val, dataset_ss_inference_test = DatasetUtil.get_dataset_by_type(
        DatasetUtil.dataset_type_ss_voc_val_scale, config.ss_size,
        scales=config.scales, data_root=config.data_root_path)

    # voc_runner.inference_crf(dataset=dataset_ss_inference_train,
    #                          logits_path=Tools.new_dir(os.path.join(config.logits_path, "train")))
    voc_runner.inference_crf(dataset=dataset_ss_inference_val,
                             logits_path=Tools.new_dir(os.path.join(config.logits_path, "val")))
    # voc_runner.inference_crf(dataset=dataset_ss_inference_test,
    #                          logits_path=Tools.new_dir(os.path.join(config.logits_path, "test")))
    pass


class Config(object):

    def __init__(self):
        self.ss_num_classes = 21

        # 图像大小
        # self.ss_size = 513
        self.ss_size = 352

        # 伪标签
        self.data_root_path = self.get_data_root_path()

        # 推理
        # self.scales = (1.0, 0.75, 0.5, 1.25, 1.5)
        self.scales = (1.0,)
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
Train
Overall Acc: 0.884174
Mean Acc: 0.770555
FreqW Acc: 0.797313
Mean IoU: 0.660239

1 scale crf val
Overall Acc: 0.915958
Mean Acc: 0.778497
FreqW Acc: 0.850119
Mean IoU: 0.682560
1 scale crf train
Overall Acc: 0.887899
Mean Acc: 0.768792
FreqW Acc: 0.802688
Mean IoU: 0.665256
"""


if __name__ == '__main__':
    config = Config()
    train(config=config)
    pass
