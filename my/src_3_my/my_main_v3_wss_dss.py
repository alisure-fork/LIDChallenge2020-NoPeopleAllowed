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
from my_util_network import DualNet
from my_util_data3 import DatasetUtil, DataUtil, MyTransform
from deep_labv3plus_pytorch.metrics import StreamSegMetrics, AverageMeter


class MyRunner(object):

    def __init__(self, config):
        self.config = config

        # Model
        self.net = self.config.Net(num_classes=self.config.num_classes)
        self.net = nn.DataParallel(self.net).cuda()
        cudnn.benchmark = True

        # 不同层设置不同的学习率
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.config.lr, betas=(0.9, 0.999), weight_decay=0)

        # Loss
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().cuda()
        self.bce_loss = nn.BCELoss().cuda()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255, reduction='mean').cuda()

        # Data
        self.dataset_train = DatasetUtil.get_dataset_by_type(
            DatasetUtil.dataset_type_voc_train_dual, self.config.input_size, data_root=self.config.data_root_path,
            sampling=self.config.sampling, train_label_path=self.config.train_label_path)
        self.dataset_val = DatasetUtil.get_dataset_by_type(
            DatasetUtil.dataset_type_voc_val, self.config.input_size, data_root=self.config.data_root_path,
            sampling=self.config.sampling, return_image_info=True, train_label_path=self.config.train_label_path)
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
                mask1, mask2 = masks[0].long().cuda(), masks[1].long().cuda()
                label1, label2 = labels[0].cuda(), labels[1].cuda()
                self.optimizer.zero_grad()

                result = self.net(x1, x2, pair_labels, has_class=self.config.has_class,
                                  has_cam=self.config.has_cam, has_ss=self.config.has_ss)

                # 分类损失
                class_logits = result["class_logits"]
                loss_class = 5 * (self.bce_with_logits_loss(class_logits["x1"], label1) +
                                  self.bce_with_logits_loss(class_logits["x2"], label2))
                loss = loss_class
                avg_meter.update("loss_class", loss_class.item())

                ss_where_cam_mask_large_1 = mask1 == (pair_labels + 1).view(-1, 1, 1).expand_as(mask1)
                ss_where_cam_mask_large_2 = mask2 == (pair_labels + 1).view(-1, 1, 1).expand_as(mask2)
                ss_where_cam_mask_min_large_1 = mask1 != (pair_labels + 1).view(-1, 1, 1).expand_as(mask1)
                ss_where_cam_mask_min_large_2 = mask2 != (pair_labels + 1).view(-1, 1, 1).expand_as(mask2)

                if self.config.has_ss:
                    ####################################################################################################
                    # CAM最大值掩码, 最小值掩码
                    ss_value_cam_mask_large_1 = result["our"]["d5_mask_2_to_1"][ss_where_cam_mask_large_1]  # 1
                    ss_value_cam_mask_large_2 = result["our"]["d5_mask_1_to_2"][ss_where_cam_mask_large_2]  # 1
                    ss_value_cam_mask_large_12 = result["our"]["d5_mask_2_to_1"][ss_where_cam_mask_min_large_1]  # 0
                    ss_value_cam_mask_large_22 = result["our"]["d5_mask_1_to_2"][ss_where_cam_mask_min_large_2]  # 0

                    # 特征相似度损失
                    loss_ss = 0
                    #########################################
                    if len(ss_value_cam_mask_large_1) > 0:
                        loss_ss = self.bce_loss(ss_value_cam_mask_large_1, torch.ones_like(ss_value_cam_mask_large_1))
                    if len(ss_value_cam_mask_large_2) > 0:
                        loss_ss += self.bce_loss(ss_value_cam_mask_large_2, torch.ones_like(ss_value_cam_mask_large_2))
                    if len(ss_value_cam_mask_large_12) > 0:
                        loss_ss += self.bce_loss(ss_value_cam_mask_large_12, torch.zeros_like(ss_value_cam_mask_large_12))
                    if len(ss_value_cam_mask_large_22) > 0:
                        loss_ss += self.bce_loss(ss_value_cam_mask_large_22, torch.zeros_like(ss_value_cam_mask_large_22))
                    #########################################
                    if loss_ss > 0:
                        loss = loss + loss_ss
                        avg_meter.update("loss_ss", loss_ss.item())
                        pass
                    ####################################################################################################

                    ####################################################################################################
                    # 预测损失
                    loss_ce = self.ce_loss(result["ss"]["out_up_1"], mask1) + \
                              self.ce_loss(result["ss"]["out_up_2"], mask2)
                    loss = loss + loss_ce
                    avg_meter.update("loss_ce", loss_ce.item())
                    ####################################################################################################
                    pass

                loss.backward()
                self.optimizer.step()
                avg_meter.update("loss", loss.item())
                pass
            ###########################################################################

            Tools.print("[E:{:3d}/{:3d}] loss:{:.4f} class:{:.4f} ss:{:.4f} ce:{:.4f}".format(
                epoch, self.config.epoch_num, avg_meter.get_results("loss"), avg_meter.get_results("loss_class"),
                avg_meter.get_results("loss_ss") if self.config.has_ss else 0.0,
                avg_meter.get_results("loss_ce") if self.config.has_ss else 0.0),
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

    def train_debug(self, model_file_name=None):

        if model_file_name is not None:
            Tools.print("Load model form {}".format(model_file_name), txt_path=self.config.save_result_txt)
            self.load_model(model_file_name)
            pass

        ###########################################################################
        # 1 Debug模型
        self.net.eval()
        for i, (pair_labels, inputs, masks, labels) in tqdm(
                enumerate(self.data_loader_train), total=len(self.data_loader_train)):
            pair_labels = pair_labels.long().cuda()
            x1, x2 = inputs[0].float().cuda(), inputs[1].float().cuda()
            mask1, mask2 = masks[0].long().cuda(), masks[1].long().cuda()

            result = self.net(x1, x2, pair_labels, has_class=self.config.has_class,
                              has_cam=self.config.has_cam, has_ss=self.config.has_ss)

            if self.config.has_ss:
                self.vis(x1, x2, mask1, mask2, pair_labels, result=result, i=0)
                Tools.print()
                pass
            pass
        ###########################################################################
        pass

    @staticmethod
    def vis(x1, x2, mask1, mask2, pair_labels, result, i=0):
        MyTransform.transform_un_normalize()(x1[i].detach().cpu()).save("1.jpg")
        MyTransform.transform_un_normalize()(x2[i].detach().cpu()).save("2.jpg")

        DataUtil.gray_to_color(np.asarray(mask1[i].detach().cpu().numpy(), dtype=np.uint8)).save("1.png")
        DataUtil.gray_to_color(np.asarray(mask2[i].detach().cpu().numpy(), dtype=np.uint8)).save("2.png")

        cam_mask_large_1 = result["our"]["cam_large_1"][i][0].detach().cpu().numpy()
        Image.fromarray(np.asarray(cam_mask_large_1 * 255, dtype=np.uint8)).save("1_mask.png")
        cam_mask_large_2 = result["our"]["cam_large_2"][i][0].detach().cpu().numpy()
        Image.fromarray(np.asarray(cam_mask_large_2 * 255, dtype=np.uint8)).save("2_mask.png")

        d5_mask_2_to_1 = result["our"]["d5_mask_2_to_1"][i].detach().cpu().numpy()
        Image.fromarray(np.asarray(d5_mask_2_to_1 * 255, dtype=np.uint8)).save("1_to.png")
        d5_mask_1_to_2 = result["our"]["d5_mask_1_to_2"][i].detach().cpu().numpy()
        Image.fromarray(np.asarray(d5_mask_1_to_2 * 255, dtype=np.uint8)).save("2_to.png")

        out_up_1 = result["ss"]["out_up_1"][i].detach().max(dim=0)[1].cpu().numpy()
        DataUtil.gray_to_color(np.asarray(out_up_1, dtype=np.uint8)).save("1_out.png")
        out_up_2 = result["ss"]["out_up_2"][i].detach().max(dim=0)[1].cpu().numpy()
        DataUtil.gray_to_color(np.asarray(out_up_2, dtype=np.uint8)).save("2_out.png")

        ss_where_cam_mask_large_1 = mask1 == (pair_labels + 1).view(-1, 1, 1).expand_as(mask1)
        ss_where_cam_mask_large_2 = mask2 == (pair_labels + 1).view(-1, 1, 1).expand_as(mask2)
        ss_where_cam_mask_min_large_1 = mask1 != (pair_labels + 1).view(-1, 1, 1).expand_as(mask1)
        ss_where_cam_mask_min_large_2 = mask2 != (pair_labels + 1).view(-1, 1, 1).expand_as(mask2)

        ss_cam_mask_large_1 = ss_where_cam_mask_large_1[i].detach().cpu().numpy()
        Image.fromarray(np.asarray(ss_cam_mask_large_1 * 255, dtype=np.uint8)).save("1_ss_mask.png")
        ss_cam_mask_large_2 = ss_where_cam_mask_large_2[i].detach().cpu().numpy()
        Image.fromarray(np.asarray(ss_cam_mask_large_2 * 255, dtype=np.uint8)).save("2_ss_mask.png")
        ss_cam_mask_min_large_1 = ss_where_cam_mask_min_large_1[i].detach().cpu().numpy()
        Image.fromarray(np.asarray(ss_cam_mask_min_large_1 * 255, dtype=np.uint8)).save("1_ss_mask_min.png")
        ss_cam_mask_min_large_2 = ss_where_cam_mask_min_large_2[i].detach().cpu().numpy()
        Image.fromarray(np.asarray(ss_cam_mask_min_large_2 * 255, dtype=np.uint8)).save("2_ss_mask_min.png")
        pass

    def eval(self, epoch=0, model_file_name=None, result_path=None):
        if model_file_name is not None:
            Tools.print("Load model form {}".format(model_file_name), txt_path=self.config.save_result_txt)
            self.load_model(model_file_name)
            pass

        avg_meter = AverageMeter()
        ss_meter = StreamSegMetrics(self.config.num_classes + 1)
        self.net.eval()
        with torch.no_grad():
            for i, (inputs, masks, labels, image_info_list, label_info_list) in tqdm(
                    enumerate(self.data_loader_val), total=len(self.data_loader_val)):
                assert len(image_info_list) == 1
                size = Image.open(label_info_list[0]).size

                inputs, labels = inputs.float().cuda(), labels.numpy()
                result = self.net.module.forward_inference(inputs, has_class=self.config.has_class,
                                                           has_cam=self.config.has_cam, has_ss=self.config.has_ss)

                # Class
                class_out = torch.sigmoid(result["class_logits"]).detach().cpu().numpy()
                one, zero = labels == 1, labels != 1
                avg_meter.update("mae", (np.abs(class_out[one] - labels[one]).mean() +
                                         np.abs(class_out[zero] - labels[zero]).mean()) / 2)
                avg_meter.update("f1", metrics.f1_score(y_true=labels, y_pred=class_out > 0.5, average='micro'))
                avg_meter.update("acc", self._acc(net_out=class_out, labels=labels))

                # SS
                if self.config.has_ss:
                    outputs = F.interpolate(result["ss"]["out_up"], size=(size[1], size[0]), mode="bilinear",
                                            align_corners=False).detach().max(dim=1)[1].cpu().numpy()
                    targets = np.expand_dims(np.asarray(Image.open(label_info_list[0])), axis=0)
                    ss_meter.update(targets, outputs)

                    if result_path is not None:
                        for image_info_one, ss_out_one, mask_one in zip(image_info_list, outputs, targets):
                            result_file = Tools.new_dir(os.path.join(result_path, os.path.basename(image_info_one)))
                            Image.open(image_info_one).save(result_file)
                            DataUtil.gray_to_color(np.asarray(
                                ss_out_one, dtype=np.uint8)).save(result_file.replace(".jpg", "_p.png"))
                            DataUtil.gray_to_color(np.asarray(
                                mask_one, dtype=np.uint8)).save(result_file.replace(".jpg", "_l.png"))
                            pass
                        pass

                    pass
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

    if config.only_train_debug:
        my_runner.train_debug(model_file_name=config.model_resume_pth)
        return

    if config.only_eval:
        my_runner.eval(epoch=0, model_file_name=config.model_pth, result_path=config.model_eval_dir)
        return

    if config.only_train:
        my_runner.train(start_epoch=0, model_file_name=None)
        return

    pass


class Config(object):

    def __init__(self):
        self.gpu_id = "1, 2, 3"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        # 流程控制
        self.only_train = True  # 是否训练
        self.only_eval = False
        self.only_train_debug = False
        self.is_supervised = True

        # Train
        self.sampling = False

        # Eval
        self.model_pth = None
        self.model_eval_dir = None
        self.model_pth = "../../../WSS_Model_My/DSS/1_DualNet_20_10_24_1_224/final_10.pth"
        self.model_eval_dir = "../../../WSS_Model_My/DEval/1_DualNet_20_10_24_1_224"

        # Debug
        self.model_resume_pth = "../../../WSS_Model_My/DSS/3_DualNet_20_20_24_1_224/17.pth"

        self.has_class = True
        self.has_cam = True
        self.has_ss = True

        # 伪标签
        if self.is_supervised:
            self.train_label_path = None
        else:
            self.train_label_path = "/mnt/4T/ALISURE/USS/ConTa/pseudo_mask_voc/result/2/sem_seg/train_aug"

        self.num_classes = 20
        self.lr = 0.0001
        self.epoch_num = 20
        self.change_epoch = 12
        self.save_epoch_freq = 1
        self.eval_epoch_freq = 1
        self.batch_size = 4 * len(self.gpu_id.split(","))
        # self.batch_size = 8 * len(self.gpu_id.split(","))

        # 图像大小
        self.input_size = 352
        # self.input_size = 224

        # 网络
        self.Net = DualNet
        self.data_root_path = self.get_data_root_path()

        run_name = "3"
        self.model_name = "{}_{}_{}_{}_{}_{}_{}".format(
            run_name, "DualNet", self.num_classes, self.epoch_num,
            self.batch_size, self.save_epoch_freq, self.input_size)
        Tools.print(self.model_name)

        self.model_dir = "../../../WSS_Model_My/DSS/{}".format(self.model_name)
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
../../../WSS_Model_My/DSS/1_DualNet_20_10_24_1_224/final_10.pth
mae:0.0881 f1:0.8518 acc:0.8518
Overall Acc: 0.933770
Mean Acc: 0.835691
FreqW Acc: 0.882858
Mean IoU: 0.716424

../../../WSS_Model_My/DSS/2_DualNet_20_10_24_1_224/final_10.pth
mae:0.0976 f1:0.8307 acc:0.8307
Overall Acc: 0.930030
Mean Acc: 0.818722
FreqW Acc: 0.876396
Mean IoU: 0.701166

../../../WSS_Model_My/DSS/3_DualNet_20_20_24_1_224/17.pth
mae:0.1034 f1:0.8296 acc:0.8296
Overall Acc: 0.930570
Mean Acc: 0.823591
FreqW Acc: 0.876711
Mean IoU: 0.704446
"""


if __name__ == '__main__':
    config = Config()
    train(config=config)
    pass