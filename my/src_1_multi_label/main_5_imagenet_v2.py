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
import torch.optim as optim
import torch.nn.functional as F
from alisuretool.Tools import Tools
import torch.backends.cudnn as cudnn
from util_blance_gpu import BalancedDataParallel
from torch.utils.data import DataLoader, Dataset
from util_data import DataUtil, DatasetUtil, MyTransform
sys.path.append("../../")
from util_network import DeepLabV3Plus
from deep_labv3plus_pytorch.metrics import StreamSegMetrics


class SSRunner(object):

    def __init__(self, config):
        self.config = config

        # Model
        self.net = self.config.Net(num_classes=self.config.ss_num_classes, output_stride=self.config.output_stride)

        self.net = BalancedDataParallel(0, self.net, dim=0).cuda()
        cudnn.benchmark = True

        # Optimize
        self.optimizer = optim.SGD(params=[
            {'params': self.net.module.model.backbone.parameters(), 'lr': self.config.ss_lr},
            {'params': self.net.module.model.classifier.parameters(), 'lr': self.config.ss_lr * 10},
        ], lr=self.config.ss_lr, momentum=0.9, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.config.ss_milestones, gamma=0.1)

        # Loss
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255, reduction='mean').cuda()
        pass

    def train_one_ss(self, data_loader, epoch=0, model_file_name=None):
        Tools.print()
        Tools.print('Epoch:{:2d}, lr={:.6f} lr2={:.6f}'.format(
            epoch, self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[1]['lr']),
            txt_path=self.config.ss_save_result_txt)

        if model_file_name is not None:
            Tools.print("Load model form {}".format(model_file_name), txt_path=self.config.ss_save_result_txt)
            self.load_model(model_file_name)
            pass

        ###########################################################################
        # 1 训练模型
        all_loss = 0.0
        self.net.train()
        for i, (inputs, labels, _) in tqdm(enumerate(data_loader), total=len(data_loader)):
            inputs, labels = inputs.float().cuda(), labels.long().cuda()
            self.optimizer.zero_grad()

            result = self.net(inputs)
            loss = self.ce_loss(result, labels)

            loss.backward()
            self.optimizer.step()

            all_loss += loss.item()
            pass
        Tools.print("[E:{:3d}/{:3d}] ss loss:{:.4f}".format(
            epoch, self.config.ss_epoch_num, all_loss / len(data_loader)),
            txt_path=self.config.ss_save_result_txt)
        ###########################################################################

        ###########################################################################
        # 2 保存模型
        Tools.print()
        save_file_name = Tools.new_dir(os.path.join(
            self.config.ss_model_dir, "ss_{}.pth".format(epoch)))
        torch.save(self.net.state_dict(), save_file_name)
        Tools.print("Save Model to {}".format(save_file_name), txt_path=self.config.ss_save_result_txt)
        Tools.print()
        ###########################################################################
        pass

    def train_eval_ss(self, data_loader, epoch=0, model_file_name=None):
        if model_file_name is not None:
            Tools.print("Load model form {}".format(model_file_name), txt_path=self.config.ss_save_result_txt)
            self.load_model(model_file_name)
            pass

        self.net.eval()
        with torch.no_grad():
            for i, (inputs, labels, image_paths) in tqdm(enumerate(data_loader), total=len(data_loader)):
                inputs = inputs.float().cuda()
                outputs = self.net(inputs)
                preds = outputs.detach().max(dim=1)[1].cpu().numpy()

                for j, (pred_one, label_one, image_path) in enumerate(zip(preds, labels, image_paths)):
                    im = Image.open(image_path)
                    now_name = image_path.split("Data/DET/")[1]
                    result_filename = Tools.new_dir(os.path.join(self.config.ss_train_eval_save_dir, now_name))
                    im.save(result_filename)

                    DataUtil.gray_to_color(np.asarray(pred_one, dtype=np.uint8)).resize(im.size).save(
                        result_filename.replace(".JPEG", "_new.png"))
                    DataUtil.gray_to_color(np.asarray(label_one, dtype=np.uint8)).resize(im.size).save(
                        result_filename.replace(".JPEG", "_old.png"))
                    pred_one[pred_one != label_one & label_one != 255] = 255
                    DataUtil.gray_to_color(np.asarray(pred_one, dtype=np.uint8)).resize(im.size).save(
                        result_filename.replace(".JPEG", ".png"))
                    pass
                pass
            pass
        pass

    def eval_ss(self, data_loader, epoch=0, model_file_name=None, save_path=None):
        if model_file_name is not None:
            Tools.print("Load model form {}".format(model_file_name), txt_path=self.config.ss_save_result_txt)
            self.load_model(model_file_name)
            pass

        un_norm = MyTransform.transform_un_normalize()
        self.net.eval()
        metrics = StreamSegMetrics(self.config.ss_num_classes)
        with torch.no_grad():
            for i, (inputs, labels, _) in tqdm(enumerate(data_loader), total=len(data_loader)):
                inputs = inputs.float().cuda()
                labels = labels.long().cuda()
                outputs = self.net(inputs)
                preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                targets = labels.cpu().numpy()

                metrics.update(targets, preds)

                if save_path:
                    for j, (input_one, label_one, pred_one) in enumerate(zip(inputs, targets, preds)):
                        un_norm(input_one.cpu()).save(os.path.join(save_path, "{}_{}.JPEG".format(i, j)))
                        DataUtil.gray_to_color(np.asarray(label_one, dtype=np.uint8)).save(
                            os.path.join(save_path, "{}_{}_l.png".format(i, j)))
                        DataUtil.gray_to_color(np.asarray(pred_one, dtype=np.uint8)).save(
                            os.path.join(save_path, "{}_{}_p.png".format(i, j)))
                        pass
                    pass
                pass
            pass

        score = metrics.get_results()
        Tools.print("{} {}".format(epoch, metrics.to_str(score)), txt_path=self.config.ss_save_result_txt)
        return score

    def load_model(self, model_file_name):
        Tools.print("Load model form {}".format(model_file_name), txt_path=self.config.ss_save_result_txt)
        checkpoint = torch.load(model_file_name)

        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1:
            # checkpoint = {key.replace("module.", ""): checkpoint[key] for key in checkpoint}
            pass

        self.net.load_state_dict(checkpoint, strict=True)
        Tools.print("Restore from {}".format(model_file_name), txt_path=self.config.ss_save_result_txt)
        pass

    pass


def train(config):
    ss_runner = SSRunner(config=config)

    model_file_name = None
    if model_file_name is not None:
        Tools.print("Load model form {}".format(model_file_name), txt_path=config.ss_save_result_txt)
        ss_runner.load_model(model_file_name)
        pass

    data_loader_ss_val = None
    for epoch in range(0, config.ss_epoch_num):

        train_label_path = config.ss_train_eval_save_dir if epoch > 0 and config.has_train_eval_ss else config.label_path
        dataset_ss_train, dataset_ss_train_eval, dataset_ss_val, _ = DatasetUtil.get_dataset_by_type(
            DatasetUtil.dataset_type_ss, config.ss_size, data_root=config.data_root_path,
            train_label_path=train_label_path, return_image_info=True, sampling=config.sampling)
        data_loader_ss_train = DataLoader(dataset_ss_train, config.ss_batch_size, True, num_workers=16)
        data_loader_ss_train_eval = DataLoader(dataset_ss_train_eval, config.ss_batch_size, False, num_workers=16)
        data_loader_ss_val = DataLoader(dataset_ss_val, config.ss_batch_size, False, num_workers=16)

        if config.has_eval_ss:
            Tools.print("Eval SS: {}".format(epoch))
            ss_runner.eval_ss(data_loader=data_loader_ss_val, epoch=epoch)
            Tools.print()
            pass

        # 训练MIC
        if config.has_train_ss:
            Tools.print("Train SS: {}".format(epoch))
            ss_runner.train_one_ss(data_loader=data_loader_ss_train, epoch=epoch)
            Tools.print()
            pass

        if config.has_train_eval_ss:
            Tools.print("Train Eval SS: {}".format(epoch))
            ss_runner.train_eval_ss(data_loader=data_loader_ss_train_eval, epoch=epoch)
            Tools.print()
            pass

        ss_runner.scheduler.step()
        pass

    # Final Save
    Tools.print()
    save_file_name = Tools.new_dir(os.path.join(config.ss_model_dir, "ss_final_{}.pth".format(config.ss_epoch_num)))
    torch.save(ss_runner.net.state_dict(), save_file_name)
    Tools.print("Save Model to {}".format(save_file_name), txt_path=config.ss_save_result_txt)
    Tools.print()

    ss_runner.eval_ss(data_loader=data_loader_ss_val, epoch=config.ss_epoch_num)
    pass


class Config(object):

    def __init__(self):
        # self.gpu_id = "0, 1, 2, 3"
        # self.gpu_id = "0"
        # self.gpu_id = "0, 1"
        self.gpu_id = "1, 2, 3"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        # 流程控制
        self.has_train_ss = True  # 是否训练SS
        self.has_eval_ss = True  # 是否评估SS
        self.has_train_eval_ss = False  # 是否SS

        self.sampling = False

        self.ss_num_classes = 201
        self.ss_epoch_num = 10
        self.ss_milestones = [5, 7]
        self.ss_batch_size = 12 * (len(self.gpu_id.split(",")) - 1)
        self.ss_lr = 0.001

        # 图像大小
        self.ss_size = 352
        self.output_stride = 16

        # 网络
        self.Net = DeepLabV3Plus

        self.data_root_path = self.get_data_root_path()
        # self.label_path = "/mnt/4T/ALISURE/USS/WSS_CAM/cam/1_CAMNet_200_32_256_0.5"
        self.label_path = "/mnt/4T/ALISURE/USS/WSS_CAM/cam_4/1_200_32_256_0.5"
        # self.label_path = "/mnt/4T/ALISURE/USS/WSS_CAM/cam_4/2_1_200_32_256_0.5"

        run_name = "4"
        self.model_name = "{}_{}_{}_{}_{}_{}".format(run_name, "DeepLabV3PlusResNet101", self.ss_num_classes,
                                                     self.ss_epoch_num, self.ss_batch_size, self.ss_size)
        Tools.print(self.model_name)

        self.ss_model_dir = "../../../WSS_Model_SS/{}".format(self.model_name)
        self.ss_train_eval_save_dir = "../../../WSS_Train_EVAL/{}".format(self.model_name)
        self.ss_save_dir = "../../../WSS_Model_SS_EVAL/{}".format(self.model_name)
        self.ss_save_result_txt = Tools.new_dir("{}/result.txt".format(self.ss_model_dir))
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


"""
../../../WSS_Model_SS/1_DeepLabV3PlusResNet101_201_10_24_1_352/ss_0.pth
Overall Acc: 0.756473
Mean Acc: 0.192014
FreqW Acc: 0.596403
Mean IoU: 0.138673

../../../WSS_Model_SS/2_DeepLabV3PlusResNet101_201_10_24_1_352/ss_0.pth
Overall Acc: 0.731829
Mean Acc: 0.483936
FreqW Acc: 0.604147
Mean IoU: 0.259592

../../../WSS_Model_SS/3_DeepLabV3PlusResNet101_201_10_24_1_352/ss_0.pth
Overall Acc: 0.761370
Mean Acc: 0.315599
FreqW Acc: 0.605419
Mean IoU: 0.220519

../../../WSS_Model_SS/4_DeepLabV3PlusResNet101_201_10_24_352/ss_0.pth
Overall Acc: 0.643250
Mean Acc: 0.362455
FreqW Acc: 0.525894
Mean IoU: 0.148717
"""


if __name__ == '__main__':
    config = Config()
    train(config=config)
    pass
