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
from sklearn import metrics
import torch.nn.functional as F
from alisuretool.Tools import Tools
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
sys.path.append("../../")
from my_util_network import MyNet
from my_util_data import DatasetUtil, DataUtil
from deep_labv3plus_pytorch.metrics import StreamSegMetrics, AverageMeter


class MyRunner(object):

    def __init__(self, config):
        self.config = config

        # Model
        self.net = self.config.Net(num_classes=self.config.num_classes)
        self.net = nn.DataParallel(self.net).cuda()
        cudnn.benchmark = True

        # 不同层设置不同的学习率
        # backbone = list(map(id, self.net.module.backbone.parameters()))
        # base_params = filter(lambda p: id(p) not in backbone, self.net.module.parameters())
        # self.optimizer = optim.Adam([
        #     {'params': base_params},
        #     {'params': self.net.module.backbone.parameters(), 'lr': self.config.lr * 10}],
        #     lr=self.config.lr, betas=(0.9, 0.999), weight_decay=0)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.config.lr, betas=(0.9, 0.999), weight_decay=0)

        # Loss
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().cuda()
        self.bce_loss = nn.BCELoss().cuda()

        # Data
        self.dataset_train = DatasetUtil.get_dataset_by_type(
            DatasetUtil.dataset_type_voc_train_dual, input_size=self.config.input_size, crop_size=self.config.crop_size,
            data_root=self.config.data_root_path, sampling=self.config.sampling)
        self.dataset_val = DatasetUtil.get_dataset_by_type(
            DatasetUtil.dataset_type_voc_val, input_size=self.config.input_size, crop_size=self.config.crop_size,
            data_root=self.config.data_root_path, sampling=self.config.sampling)
        self.data_loader_train = DataLoader(self.dataset_train, self.config.batch_size,
                                            shuffle=True, num_workers=16, drop_last=True)
        self.data_loader_val = DataLoader(self.dataset_val, 1, shuffle=False, num_workers=16)
        pass

    # 1.训练MIC
    def train(self, start_epoch=0, model_file_name=None):

        if model_file_name is not None:
            Tools.print("Load model form {}".format(model_file_name), txt_path=self.config.save_result_txt)
            self.load_model(model_file_name)
            pass

        # self.eval(epoch=0)

        for epoch in range(start_epoch, self.config.epoch_num):
            Tools.print()
            self._adjust_learning_rate(self.optimizer, epoch, lr=self.config.lr, change_epoch=self.config.change_epoch)
            Tools.print('Epoch:{:03d}, lr={:.6f}'.format(
                epoch, self.optimizer.param_groups[0]['lr']), txt_path=self.config.save_result_txt)

            ###########################################################################
            # 1 训练模型
            avg_meter = AverageMeter()
            self.net.train()
            for i, (pair_labels, inputs, masks, labels) in tqdm(
                    enumerate(self.data_loader_train), total=len(self.data_loader_train)):
                pair_labels = pair_labels.long().cuda()
                x1, x2 = inputs[0].float().cuda(), inputs[1].float().cuda()
                # mask1, mask2 = masks[0].cuda(), masks[1].cuda()
                label1, label2 = labels[0].cuda(), labels[1].cuda()
                self.optimizer.zero_grad()

                result = self.net(x1, x2, pair_labels, has_class=self.config.has_class,
                                  has_cam=self.config.has_cam, has_ss=self.config.has_ss)

                class_logits = result["class_logits"]
                loss_class = self.bce_with_logits_loss(class_logits["x1"], label1) + \
                             self.bce_with_logits_loss(class_logits["x2"], label2)
                loss = loss_class

                if self.config.has_ss:
                    where_cam_mask_large_1 = torch.squeeze(result["our"]["cam_mask_large_1"], dim=1) > 0.5
                    value_cam_mask_large_1 = result["our"]["d5_mask_2_to_1"][where_cam_mask_large_1]
                    where_cam_mask_large_2 = torch.squeeze(result["our"]["cam_mask_large_2"], dim=1) > 0.5
                    value_cam_mask_large_2 = result["our"]["d5_mask_1_to_2"][where_cam_mask_large_2]

                    where_cam_mask_min_large_1 = torch.squeeze(result["our"]["cam_mask_min_large_1"], dim=1) > 0.5
                    value_cam_mask_min_large_1 = result["our"]["d5_mask_2_to_1"][where_cam_mask_min_large_1]
                    where_cam_mask_min_large_2 = torch.squeeze(result["our"]["cam_mask_min_large_2"], dim=1) > 0.5
                    value_cam_mask_min_large_2 = result["our"]["d5_mask_1_to_2"][where_cam_mask_min_large_2]

                    loss_ss = self.bce_loss(value_cam_mask_large_1, torch.ones_like(value_cam_mask_large_1)) + \
                              self.bce_loss(value_cam_mask_large_2, torch.ones_like(value_cam_mask_large_2)) + \
                              self.bce_loss(value_cam_mask_min_large_1, torch.zeros_like(value_cam_mask_min_large_1)) + \
                              self.bce_loss(value_cam_mask_min_large_2, torch.zeros_like(value_cam_mask_min_large_2))
                    avg_meter.update("loss_ss", loss_ss.item())
                    loss = loss + loss_ss
                    pass

                loss.backward()
                self.optimizer.step()

                avg_meter.update("loss", loss.item())
                avg_meter.update("loss_class", loss_class.item())
                pass
            ###########################################################################

            Tools.print("[E:{:3d}/{:3d}] loss:{:.4f} class:{:.4f} ss:{:.4f}".format(
                epoch, self.config.epoch_num, avg_meter.get_results("loss"), avg_meter.get_results("loss_class"),
                avg_meter.get_results("loss_ss") if self.config.has_ss else 0.0),
                txt_path=self.config.save_result_txt)

            ###########################################################################
            # 2 保存模型
            if epoch % self.config.save_epoch_freq == 0:
                Tools.print()
                save_file_name = Tools.new_dir(os.path.join(self.config.model_dir, "{}.pth".format(epoch)))
                torch.save(self.net.state_dict(), save_file_name)
                Tools.print("Save Model to {}".format(save_file_name), txt_path=self.config.save_result_txt)
                Tools.print()
                pass
            ###########################################################################

            ###########################################################################
            # 3 评估模型
            if epoch % self.config.eval_epoch_freq == 0:
                self.eval(epoch=epoch)
                pass
            ###########################################################################

            pass

        # Final Save
        Tools.print()
        save_file_name = Tools.new_dir(os.path.join(self.config.model_dir, "final_{}.pth".format(self.config.epoch_num)))
        torch.save(self.net.state_dict(), save_file_name)
        Tools.print("Save Model to {}".format(save_file_name), txt_path=self.config.save_result_txt)
        Tools.print()

        self.eval(epoch=self.config.epoch_num)
        pass

    def eval(self, epoch=0, model_file_name=None):
        if model_file_name is not None:
            Tools.print("Load model form {}".format(model_file_name), txt_path=self.config.save_result_txt)
            self.load_model(model_file_name)
            pass

        avg_meter = AverageMeter()
        ss_meter = StreamSegMetrics(self.config.num_classes + 1)
        self.net.eval()
        with torch.no_grad():
            for i, (inputs, masks, labels) in tqdm(enumerate(self.data_loader_val), total=len(self.data_loader_val)):
                inputs = inputs.float().cuda()
                masks, labels = masks.numpy(), labels.numpy()

                result = self.net.module.forward_inference(inputs, has_class=self.config.has_class,
                                                           has_cam=self.config.has_cam, has_ss=self.config.has_ss)

                # SS
                if self.config.has_ss:
                    ss_out = result["ss"]["out_up"].detach().max(dim=1)[1].cpu().numpy()
                    ss_meter.update(masks, ss_out)
                    pass

                # Class
                class_out = torch.sigmoid(result["class_logits"]).detach().cpu().numpy()
                one, zero = labels == 1, labels != 1
                avg_meter.update("mae", (np.abs(class_out[one] - labels[one]).mean() +
                                         np.abs(class_out[zero] - labels[zero]).mean()) / 2)
                avg_meter.update("f1", metrics.f1_score(y_true=labels, y_pred=class_out > 0.5, average='micro'))
                avg_meter.update("acc", self._acc(net_out=class_out, labels=labels))
                pass
            pass

        Tools.print("[E:{:3d}] val mae:{:.4f} f1:{:.4f} acc:{:.4f}".format(
            epoch, avg_meter.get_results("mae"), avg_meter.get_results("f1"),
            avg_meter.get_results("acc")), txt_path=self.config.save_result_txt)
        if self.config.has_ss:
            Tools.print("[E:{:3d}] ss {}".format(epoch, ss_meter.to_str(ss_meter.get_results())))
            pass
        pass

    def load_model(self, model_file_name):
        Tools.print("Load model form {}".format(model_file_name), txt_path=self.config.save_result_txt)
        checkpoint = torch.load(model_file_name)
        self.net.load_state_dict(checkpoint, strict=True)
        Tools.print("Restore from {}".format(model_file_name), txt_path=self.config.save_result_txt)
        pass

    @staticmethod
    def _adjust_learning_rate(optimizer, epoch, lr, change_epoch=30):
        if epoch > change_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * 0.1
                pass
        pass

    @staticmethod
    def _acc(net_out, labels):
        acc_num, total_num = 0, 0
        for out_one, label_one in zip(net_out, labels):
            label_set = list(np.where(label_one)[0])
            out_set = list(np.where(out_one > 0.5)[0])
            all_num = len(label_set) + len(out_set)
            ok_num = (all_num - len(set(out_set + label_set)))
            acc_num += 2 * ok_num
            total_num += all_num
        return acc_num / total_num

    pass


def train(config):
    my_runner = MyRunner(config=config)

    # 训练MIC
    if config.has_train:
        my_runner.train(start_epoch=0, model_file_name=None)
        # my_runner.eval(epoch=0, model_file_name=os.path.join(config.model_dir, "5.pth"))
        pass

    pass


class Config(object):

    def __init__(self):
        self.gpu_id = "1, 2, 3"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        # 流程控制
        self.has_train = True  # 是否训练
        self.sampling = False

        self.has_class = True
        self.has_cam = True
        self.has_ss = True

        self.num_classes = 20
        self.epoch_num = 50
        self.change_epoch = 30
        self.batch_size = 4 * len(self.gpu_id.split(","))
        # self.batch_size = 16 * len(self.gpu_id.split(","))
        self.lr = 0.0001
        self.save_epoch_freq = 5
        self.eval_epoch_freq = 5

        # 图像大小
        # self.input_size = 352
        # self.crop_size = 320
        self.input_size = 256
        self.crop_size = 224

        # 网络
        self.Net = MyNet

        self.data_root_path = self.get_data_root_path()

        run_name = "1"
        self.model_name = "{}_{}_{}_{}_{}_{}_{}_{}".format(
            run_name, "MNet", self.num_classes, self.epoch_num,
            self.batch_size, self.save_epoch_freq, self.input_size, self.crop_size)
        Tools.print(self.model_name)

        self.model_dir = "../../../WSS_Model_My_SS/{}".format(self.model_name)
        self.save_result_txt = Tools.new_dir("{}/result.txt".format(self.model_dir))
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
../../../WSS_Model_My_SS/1_MNet_20_50_48_5_256_224/5.pth
mae:0.0969 f1:0.8342 acc:0.8342
"""


if __name__ == '__main__':
    config = Config()
    train(config=config)
    pass
