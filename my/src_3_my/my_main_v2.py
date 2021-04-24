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
from my_util_data import DatasetUtil, DataUtil, MyTransform
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
            DatasetUtil.dataset_type_voc_train_dual, input_size=self.config.input_size,
            data_root=self.config.data_root_path, sampling=self.config.sampling)
        self.dataset_val = DatasetUtil.get_dataset_by_type(
            DatasetUtil.dataset_type_voc_val, input_size=self.config.input_size,
            data_root=self.config.data_root_path, sampling=self.config.sampling, return_image_info=True)
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
                # mask1, mask2 = masks[0].cuda(), masks[1].cuda()
                label1, label2 = labels[0].cuda(), labels[1].cuda()
                self.optimizer.zero_grad()

                result = self.net(x1, x2, pair_labels, label1, label2, has_class=self.config.has_class,
                                  has_cam=self.config.has_cam, has_ss=self.config.has_ss)

                # 分类损失
                class_logits = result["class_logits"]
                loss_class = 10 * (self.bce_with_logits_loss(class_logits["x1"], label1) +
                                   self.bce_with_logits_loss(class_logits["x2"], label2))
                loss = loss_class
                avg_meter.update("loss_class", loss_class.item())

                if self.config.has_ss:
                    ####################################################################################################
                    # CAM最大值掩码, 最小值掩码
                    ss_where_cam_mask_large_1 = torch.squeeze(result["our"]["cam_mask_large_1"], dim=1) > 0.5
                    ss_where_cam_mask_large_2 = torch.squeeze(result["our"]["cam_mask_large_2"], dim=1) > 0.5
                    ss_where_cam_mask_min_large_1 = torch.squeeze(result["our"]["cam_mask_min_large_1"], dim=1) > 0.5
                    ss_where_cam_mask_min_large_2 = torch.squeeze(result["our"]["cam_mask_min_large_2"], dim=1) > 0.5
                    ss_value_cam_mask_large_1 = result["our"]["d5_mask_2_to_1"][ss_where_cam_mask_large_1]  # 1
                    ss_value_cam_mask_large_2 = result["our"]["d5_mask_1_to_2"][ss_where_cam_mask_large_2]  # 1
                    ss_value_cam_mask_min_large_12 = result["our"]["d5_mask_neg_2_to_1"][ss_where_cam_mask_large_1]  # 0
                    ss_value_cam_mask_min_large_22 = result["our"]["d5_mask_neg_1_to_2"][ss_where_cam_mask_large_2]  # 0
                    ss_value_cam_mask_large_12 = result["our"]["d5_mask_2_to_1"][ss_where_cam_mask_min_large_1]  # 0
                    ss_value_cam_mask_large_22 = result["our"]["d5_mask_1_to_2"][ss_where_cam_mask_min_large_2]  # 0

                    # 特征相似度损失
                    loss_ss = 0
                    #########################################
                    if len(ss_value_cam_mask_large_1) > 0:
                        loss_ss = self.bce_loss(ss_value_cam_mask_large_1, torch.ones_like(ss_value_cam_mask_large_1))
                    if len(ss_value_cam_mask_large_2) > 0:
                        loss_ss += self.bce_loss(ss_value_cam_mask_large_2, torch.ones_like(ss_value_cam_mask_large_2))
                    if len(ss_value_cam_mask_min_large_12) > 0:
                        loss_ss += self.bce_loss(ss_value_cam_mask_min_large_12, torch.zeros_like(ss_value_cam_mask_min_large_12))
                    if len(ss_value_cam_mask_min_large_22) > 0:
                        loss_ss += self.bce_loss(ss_value_cam_mask_min_large_22, torch.zeros_like(ss_value_cam_mask_min_large_22))
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
                    # 输出的正标签
                    ce_where_cam_mask_large_1 = ss_where_cam_mask_large_1
                    ce_mask_large_1 = torch.ones_like(ce_where_cam_mask_large_1).long() * 255
                    now_pair_labels_1 = (pair_labels + 1).view(-1, 1, 1).expand_as(ce_mask_large_1)
                    ce_mask_large_1[ce_where_cam_mask_large_1] = now_pair_labels_1[ce_where_cam_mask_large_1]
                    ce_where_cam_mask_large_2 = ss_where_cam_mask_large_2
                    ce_mask_large_2 = torch.ones_like(ce_where_cam_mask_large_2).long() * 255
                    now_pair_labels_2 = (pair_labels + 1).view(-1, 1, 1).expand_as(ce_mask_large_2)
                    ce_mask_large_2[ce_where_cam_mask_large_2] = now_pair_labels_2[ce_where_cam_mask_large_2]

                    # 输出的负标签
                    ce_where_cam_mask_min_large_1 = ss_where_cam_mask_min_large_1
                    ce_mask_min_large_1 = torch.ones_like(ce_where_cam_mask_min_large_1).long() * 255
                    ce_mask_min_large_1[ce_where_cam_mask_min_large_1] = 0
                    ce_where_cam_mask_min_large_2 = ss_where_cam_mask_min_large_2
                    ce_mask_min_large_2 = torch.ones_like(ce_where_cam_mask_min_large_2).long() * 255
                    ce_mask_min_large_2[ce_where_cam_mask_min_large_2] = 0

                    # 预测损失
                    loss_ce = self.ce_loss(result["ss"]["out_1"], ce_mask_large_1) + \
                              self.ce_loss(result["ss"]["out_2"], ce_mask_large_2) + \
                              self.ce_loss(result["ss"]["out_1"], ce_mask_min_large_1) + \
                              self.ce_loss(result["ss"]["out_2"], ce_mask_min_large_2)
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
            label1, label2 = labels[0].cuda(), labels[1].cuda()

            result = self.net(x1, x2, pair_labels, label1, label2, has_class=self.config.has_class,
                              has_cam=self.config.has_cam, has_ss=self.config.has_ss)

            if self.config.has_ss:
                # CAM最大值掩码
                ss_where_cam_mask_large_1 = torch.squeeze(result["our"]["cam_mask_large_1"], dim=1) > 0.5
                ss_where_cam_mask_large_2 = torch.squeeze(result["our"]["cam_mask_large_2"], dim=1) > 0.5

                # CAM最小值掩码
                ss_where_cam_mask_min_large_1 = torch.squeeze(result["our"]["cam_mask_min_large_1"], dim=1) > 0.5
                ss_where_cam_mask_min_large_2 = torch.squeeze(result["our"]["cam_mask_min_large_2"], dim=1) > 0.5

                # 输出的正标签
                ce_where_cam_mask_large_1 = ss_where_cam_mask_large_1
                ce_where_cam_mask_large_2 = ss_where_cam_mask_large_2

                # 输出的负标签
                ce_where_cam_mask_min_large_1 = ss_where_cam_mask_min_large_1
                ce_where_cam_mask_min_large_2 = ss_where_cam_mask_min_large_2
                Tools.print()
                pass
            pass
        ###########################################################################
        pass

    @staticmethod
    def vis():
        i = 0
        MyTransform.transform_un_normalize()(x1[i].detach().cpu()).save("1.jpg")
        MyTransform.transform_un_normalize()(x2[i].detach().cpu()).save("2.jpg")

        cam_norm_large_1 = result["cam"]["cam_norm_large_1"][i][0].detach().cpu().numpy()
        Image.fromarray(np.asarray(cam_norm_large_1 * 255, dtype=np.uint8)).save("1.png")
        cam_norm_large_2 = result["cam"]["cam_norm_large_2"][i][0].detach().cpu().numpy()
        Image.fromarray(np.asarray(cam_norm_large_2 * 255, dtype=np.uint8)).save("2.png")

        neg_cam_norm_large_1 = result["cam"]["neg_cam_norm_large_1"][i][0].detach().cpu().numpy()
        Image.fromarray(np.asarray(neg_cam_norm_large_1 * 255, dtype=np.uint8)).save("1_neg.png")
        neg_cam_norm_large_2 = result["cam"]["neg_cam_norm_large_2"][i][0].detach().cpu().numpy()
        Image.fromarray(np.asarray(neg_cam_norm_large_2 * 255, dtype=np.uint8)).save("2_neg.png")

        cam_mask_large_1 = result["our"]["cam_mask_large_1"][i][0].detach().cpu().numpy()
        Image.fromarray(np.asarray(cam_mask_large_1 * 255, dtype=np.uint8)).save("1_mask.png")
        cam_mask_large_2 = result["our"]["cam_mask_large_2"][i][0].detach().cpu().numpy()
        Image.fromarray(np.asarray(cam_mask_large_2 * 255, dtype=np.uint8)).save("2_mask.png")

        cam_mask_min_large_1 = result["our"]["cam_mask_min_large_1"][i][0].detach().cpu().numpy()
        Image.fromarray(np.asarray(cam_mask_min_large_1 * 255, dtype=np.uint8)).save("1_mask_neg.png")
        cam_mask_min_large_2 = result["our"]["cam_mask_min_large_2"][i][0].detach().cpu().numpy()
        Image.fromarray(np.asarray(cam_mask_min_large_2 * 255, dtype=np.uint8)).save("2_mask_neg.png")

        d5_mask_2_to_1 = result["our"]["d5_mask_2_to_1"][i].detach().cpu().numpy()
        Image.fromarray(np.asarray(d5_mask_2_to_1 * 255, dtype=np.uint8)).save("1_to.png")
        d5_mask_1_to_2 = result["our"]["d5_mask_1_to_2"][i].detach().cpu().numpy()
        Image.fromarray(np.asarray(d5_mask_1_to_2 * 255, dtype=np.uint8)).save("2_to.png")

        d5_mask_neg_2_to_1 = result["our"]["d5_mask_neg_2_to_1"][i].detach().cpu().numpy()
        Image.fromarray(np.asarray(d5_mask_neg_2_to_1 * 255, dtype=np.uint8)).save("1_to_neg.png")
        d5_mask_neg_1_to_2 = result["our"]["d5_mask_neg_1_to_2"][i].detach().cpu().numpy()
        Image.fromarray(np.asarray(d5_mask_neg_1_to_2 * 255, dtype=np.uint8)).save("2_to_neg.png")

        out_up_1 = result["ss"]["out_up_1"][i].detach().max(dim=0)[1].cpu().numpy()
        DataUtil.gray_to_color(np.asarray(out_up_1, dtype=np.uint8)).save("1_out.png")
        out_up_2 = result["ss"]["out_up_2"][i].detach().max(dim=0)[1].cpu().numpy()
        DataUtil.gray_to_color(np.asarray(out_up_2, dtype=np.uint8)).save("2_out.png")

        ss_1 = np.asarray(ss_where_cam_mask_large_1[i].detach().cpu().numpy(), dtype=np.int)
        Image.fromarray(np.asarray(ss_1 * 255, dtype=np.uint8)).save("1_ss_mask.png")
        ss_2 = np.asarray(ss_where_cam_mask_large_2[i].detach().cpu().numpy(), dtype=np.int)
        Image.fromarray(np.asarray(ss_2 * 255, dtype=np.uint8)).save("2_ss_mask.png")
        ss_min_1 = np.asarray(ss_where_cam_mask_min_large_1[i].detach().cpu().numpy(), dtype=np.int)
        Image.fromarray(np.asarray(ss_min_1 * 255, dtype=np.uint8)).save("1_ss_mask_neg.png")
        ss_min_2 = np.asarray(ss_where_cam_mask_min_large_2[i].detach().cpu().numpy(), dtype=np.int)
        Image.fromarray(np.asarray(ss_min_2 * 255, dtype=np.uint8)).save("2_ss_mask_neg.png")

        ce_1 = np.asarray(ce_where_cam_mask_large_1[i].detach().cpu().numpy(), dtype=np.int)
        Image.fromarray(np.asarray(ce_1 * 255, dtype=np.uint8)).save("1_ce_mask.png")
        ce_2 = np.asarray(ce_where_cam_mask_large_2[i].detach().cpu().numpy(), dtype=np.int)
        Image.fromarray(np.asarray(ce_2 * 255, dtype=np.uint8)).save("2_ce_mask.png")
        ce_min_1 = np.asarray(ce_where_cam_mask_min_large_1[i].detach().cpu().numpy(), dtype=np.int)
        Image.fromarray(np.asarray(ce_min_1 * 255, dtype=np.uint8)).save("1_ce_mask_neg.png")
        ce_min_2 = np.asarray(ce_where_cam_mask_min_large_2[i].detach().cpu().numpy(), dtype=np.int)
        Image.fromarray(np.asarray(ce_min_2 * 255, dtype=np.uint8)).save("2_ce_mask_neg.png")
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
            for i, (inputs, masks, labels, image_info_list) in tqdm(
                    enumerate(self.data_loader_val), total=len(self.data_loader_val)):
                inputs = inputs.float().cuda()
                masks, labels = masks.numpy(), labels.numpy()

                result = self.net.module.forward_inference(inputs, has_class=self.config.has_class,
                                                           has_cam=self.config.has_cam, has_ss=self.config.has_ss)

                # SS
                if self.config.has_ss:
                    ss_out = result["ss"]["out_up"].detach().max(dim=1)[1].cpu().numpy()
                    ss_meter.update(masks, ss_out)

                    if result_path is not None:
                        for image_info_one, ss_out_one, mask_one in zip(image_info_list, ss_out, masks):
                            result_file = Tools.new_dir(os.path.join(result_path, os.path.basename(image_info_one)))
                            Image.open(image_info_one).save(result_file)
                            DataUtil.gray_to_color(np.asarray(
                                ss_out_one, dtype=np.uint8)).save(result_file.replace(".jpg", "_p.png"))
                            DataUtil.gray_to_color(np.asarray(
                                mask_one, dtype=np.uint8)).save(result_file.replace(".jpg", "_l.png"))
                            pass
                        pass

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
        self.sampling = False

        # Eval
        self.only_eval = False
        self.model_pth = None
        self.model_eval_dir = None
        self.model_pth = "../../../WSS_Model_My/SS/5_MNet_20_15_24_1_256_224/2.pth"
        self.model_eval_dir = "../../../WSS_Model_My/Eval/5_MNet_20_15_24_1_256_224"

        # Debug
        self.only_train_debug = False
        self.model_resume_pth = "../../../WSS_Model_My/SS/5_MNet_20_10_24_1_224/final_10.pth"

        self.has_class = True
        self.has_cam = True
        self.has_ss = True

        self.num_classes = 20
        self.lr = 0.0001
        self.epoch_num = 10
        self.change_epoch = 6
        self.save_epoch_freq = 1
        self.eval_epoch_freq = 1
        self.batch_size = 8 * len(self.gpu_id.split(","))

        # 图像大小
        # self.input_size = 352
        self.input_size = 224

        # 网络
        self.Net = MyNet
        self.data_root_path = self.get_data_root_path()

        run_name = "5"
        self.model_name = "{}_{}_{}_{}_{}_{}_{}".format(
            run_name, "MNet", self.num_classes, self.epoch_num,
            self.batch_size, self.save_epoch_freq, self.input_size)
        Tools.print(self.model_name)

        self.model_dir = "../../../WSS_Model_My/SS/{}".format(self.model_name)
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
../../../WSS_Model_My/SS/1_MNet_20_50_48_5_256_224/5.pth
mae:0.0969 f1:0.8342 acc:0.8342

../../../WSS_Model_My/SS/1_MNet_20_50_24_1_256_224/12.pth
mae:0.1131 f1:0.8063 acc:0.8063
Overall Acc: 0.598694
Mean Acc: 0.556079
FreqW Acc: 0.478520
Mean IoU: 0.266939

../../../WSS_Model_My/SS/2_MNet_20_15_24_1_256_224/8.pth
mae:0.1093 f1:0.8187 acc:0.8187
Overall Acc: 0.694466
Mean Acc: 0.586936
FreqW Acc: 0.571327
Mean IoU: 0.321086

../../../WSS_Model_My/SS/3_MNet_20_15_24_1_256_224/7.pth  # no loss ss
mae:0.1032 f1:0.8184 acc:0.8184
Overall Acc: 0.698969
Mean Acc: 0.588587
FreqW Acc: 0.575668
Mean IoU: 0.324335

../../../WSS_Model_My/SS/5_MNet_20_15_24_1_224/1.pth
mae:0.1064 f1:0.8190 acc:0.8190
Overall Acc: 0.748733
Mean Acc: 0.534675
FreqW Acc: 0.624344
Mean IoU: 0.333368


"""


if __name__ == '__main__':
    config = Config()
    train(config=config)
    pass
