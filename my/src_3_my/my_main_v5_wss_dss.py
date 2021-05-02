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
from my_util_blance_gpu import BalancedDataParallel
sys.path.append("../../")
from my_util_network import DualNetDeepLabV3Plus
from my_util_data3 import DatasetUtil, DataUtil, MyTransform
from deep_labv3plus_pytorch.metrics import StreamSegMetrics, AverageMeter


class MyRunner(object):

    def __init__(self, config):
        self.config = config

        # Data
        self.dataset_train = DatasetUtil.get_dataset_by_type(
            DatasetUtil.dataset_type_voc_train_dual, self.config.input_size, data_root=self.config.data_root_path,
            sampling=self.config.sampling, train_label_path=self.config.train_label_path)
        self.dataset_val = DatasetUtil.get_dataset_by_type(
            DatasetUtil.dataset_type_voc_val, self.config.input_size, data_root=self.config.data_root_path,
            sampling=self.config.sampling, return_image_info=True)
        self.data_loader_train = DataLoader(self.dataset_train, self.config.batch_size,
                                            shuffle=True, num_workers=16, drop_last=True)
        self.data_loader_val = DataLoader(self.dataset_val, 1, shuffle=False, num_workers=16)

        # Model
        self.net = self.config.Net(num_classes=self.config.num_classes, output_stride=self.config.output_stride)
        if self.config.cuda_balance:
            self.net = BalancedDataParallel(0, self.net, dim=0).cuda()
        else:
            self.net = nn.DataParallel(self.net).cuda()
        cudnn.benchmark = True

        # Optimize
        self.optimizer = optim.SGD(params=[
            {'params': self.net.module.model.backbone.parameters(), 'lr': self.config.lr},
            {'params': self.net.module.model.classifier.parameters(), 'lr': self.config.lr * 10},
            {'params': self.net.module.model.cam_classifier.parameters(), 'lr': self.config.lr * 10},
        ], lr=self.config.lr, momentum=0.9, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config.milestones, gamma=0.1)

        # Loss
        self.mse_loss = nn.MSELoss().cuda()
        self.bce_loss = nn.BCEWithLogitsLoss().cuda()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255, reduction='mean').cuda()
        pass

    # 1.训练MIC
    def train(self, start_epoch=0, model_file_name=None):

        if model_file_name is not None:
            Tools.print("Load model form {}".format(model_file_name), txt_path=self.config.save_result_txt)
            self.load_model(model_file_name)
            pass

        for epoch in range(start_epoch, self.config.epoch_num):
            Tools.print()
            Tools.print('Epoch:{:2d}, lr={:.6f} lr2={:.6f}'.format(
                epoch, self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[1]['lr']),
                txt_path=self.config.save_result_txt)

            ###########################################################################
            # 1 训练模型
            self.net.train()
            avg_meter = AverageMeter()
            for i, (pair_labels, inputs, masks, labels) in tqdm(
                    enumerate(self.data_loader_train), total=len(self.data_loader_train)):
                pair_labels = pair_labels.long().cuda()
                x1, x2 = inputs[0].float().cuda(), inputs[1].float().cuda()
                mask1, mask2 = masks[0].long().cuda(), masks[1].long().cuda()
                label1, label2 = labels[0].cuda(), labels[1].cuda()
                self.optimizer.zero_grad()

                result = self.net(x1, x2, pair_labels, has_class=self.config.has_class,
                                  has_cam=self.config.has_cam, has_ss=self.config.has_ss)

                loss = 0

                ####################################################################################################
                # 分类损失
                if self.config.has_class:
                    class_logits = result["class_logits"]
                    loss_class = 5 * (self.bce_loss(class_logits["x1"], label1) +
                                      self.bce_loss(class_logits["x2"], label2))
                    loss = loss + loss_class
                    avg_meter.update("loss_class", loss_class.item())
                    pass
                ####################################################################################################

                ####################################################################################################
                # 激活图损失, 特征相似度损失
                if self.config.has_cam:
                    where_1 = torch.squeeze((result["cam"]["cam_norm_aff_1"].detach() < 1e-6) |
                                            (result["cam"]["cam_norm_aff_1"].detach() > 0.5), dim=1)
                    where_2 = torch.squeeze((result["cam"]["cam_norm_aff_2"].detach() < 1e-6) |
                                            (result["cam"]["cam_norm_aff_2"].detach() > 0.5), dim=1)
                    mask_where_1 = torch.squeeze(result["cam"]["cam_norm_aff_1"].detach() > 0.1, dim=1)
                    mask_where_2 = torch.squeeze(result["cam"]["cam_norm_aff_2"].detach() > 0.1, dim=1)

                    cam_mask_large_1 = torch.zeros_like(result["our"]["d5_mask_2_to_1"])
                    cam_mask_large_1[mask_where_1] = 1
                    cam_mask_large_2 = torch.zeros_like(result["our"]["d5_mask_1_to_2"])
                    cam_mask_large_2[mask_where_2] = 1

                    # 激活图损失
                    loss_cam = self.mse_loss(torch.squeeze(result["cam"]["cam_norm_1"], dim=1)[where_1], cam_mask_large_1[where_1]) + \
                               self.mse_loss(torch.squeeze(result["cam"]["cam_norm_2"], dim=1)[where_2], cam_mask_large_2[where_2])
                    loss = loss + loss_cam
                    avg_meter.update("loss_cam", loss_cam.item())
                    ##################################################
                    # 特征相似度损失
                    loss_ss = self.mse_loss(result["our"]["d5_mask_2_to_1"][where_1], cam_mask_large_1[where_1]) + \
                              self.mse_loss(result["our"]["d5_mask_1_to_2"][where_2], cam_mask_large_2[where_2])
                    loss = loss + loss_ss
                    avg_meter.update("loss_ss", loss_ss.item())
                    pass
                ####################################################################################################

                ####################################################################################################
                # 预测损失
                if self.config.has_ss:
                    final_mask1 = torch.zeros_like(mask1) + 255
                    black1 = torch.squeeze((result["cam"]["cam_norm_aff_1"].detach() < 1e-6), dim=1)
                    white1 = torch.squeeze((result["cam"]["cam_norm_aff_1"].detach() > 0.5), dim=1)
                    final_mask1[black1] = 0
                    final_mask1[white1] = 1
                    final_mask1 = final_mask1 * (pair_labels + 1).view(-1, 1, 1).expand_as(final_mask1)
                    final_mask1[final_mask1 >= 255] = 255

                    final_mask2 = torch.zeros_like(mask2) + 255
                    black2 = torch.squeeze((result["cam"]["cam_norm_aff_2"].detach() < 1e-6), dim=1)
                    white2 = torch.squeeze((result["cam"]["cam_norm_aff_2"].detach() > 0.5), dim=1)
                    final_mask2[black2] = 0
                    final_mask2[white2] = 1
                    final_mask2 = final_mask2 * (pair_labels + 1).view(-1, 1, 1).expand_as(final_mask2)
                    final_mask2[final_mask2 >= 255] = 255

                    loss_ce = self.ce_loss(result["ss"]["out_up_1"], final_mask1) + \
                              self.ce_loss(result["ss"]["out_up_2"], final_mask2)
                    loss = loss + loss_ce
                    avg_meter.update("loss_ce", loss_ce.item())
                    pass
                ####################################################################################################

                loss.backward()
                self.optimizer.step()
                avg_meter.update("loss", loss.item())
                pass
            self.scheduler.step()
            ###########################################################################

            Tools.print("[E:{:3d}/{:3d}] loss:{:.4f} class:{:.4f} ss:{:.4f} ce:{:.4f} cam:{:.4f}".format(
                epoch, self.config.epoch_num,
                avg_meter.get_results("loss"),
                avg_meter.get_results("loss_class") if self.config.has_class else 0.0,
                avg_meter.get_results("loss_ss") if self.config.has_cam else 0.0,
                avg_meter.get_results("loss_ce") if self.config.has_ss else 0.0,
                avg_meter.get_results("loss_cam") if self.config.has_cam else 0.0),
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
        out_up_1 = result["ss"]["out_up_1"][i].detach().max(dim=0)[1].cpu().numpy()
        DataUtil.gray_to_color(np.asarray(out_up_1, dtype=np.uint8)).save("1_out.png")
        out_up_2 = result["ss"]["out_up_2"][i].detach().max(dim=0)[1].cpu().numpy()
        DataUtil.gray_to_color(np.asarray(out_up_2, dtype=np.uint8)).save("2_out.png")

        cam_norm_1 = result["cam"]["cam_norm_1"][i][0].detach().cpu().numpy()
        Image.fromarray(np.asarray(cam_norm_1 * 255, dtype=np.uint8)).save("1_cam.png")
        cam_norm_2 = result["cam"]["cam_norm_2"][i][0].detach().cpu().numpy()
        Image.fromarray(np.asarray(cam_norm_2 * 255, dtype=np.uint8)).save("2_cam.png")
        cam_norm_aff_1 = result["cam"]["cam_norm_aff_1"][i][0].detach().cpu().numpy()
        Image.fromarray(np.asarray(cam_norm_aff_1 * 255, dtype=np.uint8)).save("1_cam_aff.png")
        cam_norm_aff_2 = result["cam"]["cam_norm_aff_2"][i][0].detach().cpu().numpy()
        Image.fromarray(np.asarray(cam_norm_aff_2 * 255, dtype=np.uint8)).save("2_cam_aff.png")

        d5_mask_2_to_1 = result["our"]["d5_mask_2_to_1"][i].detach().cpu().numpy()
        Image.fromarray(np.asarray(d5_mask_2_to_1 * 255, dtype=np.uint8)).save("1_to.png")
        d5_mask_1_to_2 = result["our"]["d5_mask_1_to_2"][i].detach().cpu().numpy()
        Image.fromarray(np.asarray(d5_mask_1_to_2 * 255, dtype=np.uint8)).save("2_to.png")


        cam_mask_large_1 = torch.zeros_like(result["our"]["cam_large_1"]) + 255
        cam_mask_large_1[result["cam"]["cam_norm_aff_1"] < 1e-6] = 0
        cam_mask_large_1[result["cam"]["cam_norm_aff_1"] > 0.5] = 1
        cam_mask_large_1 = cam_mask_large_1[i][0].detach().cpu().numpy()
        DataUtil.gray_to_color(np.asarray(cam_mask_large_1, dtype=np.uint8)).save("1_mask_hard.png")
        cam_mask_large_2 = torch.zeros_like(result["our"]["cam_large_2"]) + 255
        cam_mask_large_2[result["cam"]["cam_norm_aff_2"] < 1e-6] = 0
        cam_mask_large_2[result["cam"]["cam_norm_aff_2"] > 0.5] = 1
        cam_mask_large_2 = cam_mask_large_2[i][0].detach().cpu().numpy()
        DataUtil.gray_to_color(np.asarray(cam_mask_large_2, dtype=np.uint8)).save("2_mask_hard.png")

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

        final_mask1 = torch.zeros_like(mask1) + 255
        black1 = torch.squeeze((result["cam"]["cam_norm_aff_1"].detach() < 1e-6), dim=1)
        white1 = torch.squeeze((result["cam"]["cam_norm_aff_1"].detach() > 0.5), dim=1)
        final_mask1[black1] = 0
        final_mask1[white1] = 1
        final_mask1 = final_mask1 * (pair_labels + 1).view(-1, 1, 1).expand_as(final_mask1)
        final_mask1[final_mask1 >= 255] = 255

        final_mask2 = torch.zeros_like(mask2) + 255
        black2 = torch.squeeze((result["cam"]["cam_norm_aff_2"].detach() < 1e-6), dim=1)
        white2 = torch.squeeze((result["cam"]["cam_norm_aff_2"].detach() > 0.5), dim=1)
        final_mask2[black2] = 0
        final_mask2[white2] = 1
        final_mask2 = final_mask2 * (pair_labels + 1).view(-1, 1, 1).expand_as(final_mask2)
        final_mask2[final_mask2 >= 255] = 255

        DataUtil.gray_to_color(np.asarray(final_mask1[i].detach().cpu().numpy(), dtype=np.uint8)).save("1_mask_2.png")
        DataUtil.gray_to_color(np.asarray(final_mask2[i].detach().cpu().numpy(), dtype=np.uint8)).save("2_mask_2.png")
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
                if self.config.has_class:
                    class_out = torch.sigmoid(result["class_logits"]).detach().cpu().numpy()
                    one, zero = labels == 1, labels != 1
                    avg_meter.update("mae", (np.abs(class_out[one] - labels[one]).mean() +
                                             np.abs(class_out[zero] - labels[zero]).mean()) / 2)
                    avg_meter.update("f1", metrics.f1_score(y_true=labels, y_pred=class_out > 0.5, average='micro'))
                    avg_meter.update("acc", self._acc(net_out=class_out, labels=labels))
                    pass

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

        if self.config.has_class:
            Tools.print("[E:{:3d}] val mae:{:.4f} f1:{:.4f} acc:{:.4f}".format(
                epoch, avg_meter.get_results("mae"), avg_meter.get_results("f1"),
                avg_meter.get_results("acc")), txt_path=self.config.save_result_txt)
            pass
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
        # self.gpu_id = "1, 2, 3"
        self.gpu_id = "0, 1, 2, 3"
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
        self.model_pth = "../../../WSS_Model_My/DSS/8_DualNetDeepLabV3Plus_20_80_16_2_513_16/final_80.pth"
        self.model_eval_dir = "../../../WSS_Model_My/DEval/8_DualNetDeepLabV3Plus_20_80_16_2_513_16"

        # Debug
        self.model_resume_pth = "../../../WSS_Model_My/DSS/12_DualNetDeepLabV3Plus_20_80_32_2_352_16/10.pth"

        self.has_class = True
        self.has_cam = True
        self.has_ss = True

        self.output_stride = 16
        # self.input_size = 513
        # self.batch_size_one = 4
        self.input_size = 352
        self.batch_size_one = 6

        self.cuda_balance = False
        # self.cuda_balance = True
        if self.cuda_balance:
            self.batch_size = self.batch_size_one * (len(self.gpu_id.split(",")) - 1)
        else:
            self.batch_size = self.batch_size_one * len(self.gpu_id.split(","))

        self.num_classes = 20
        self.lr = 0.001
        self.epoch_num = 80
        self.milestones = [40, 60]
        self.save_epoch_freq = 2
        self.eval_epoch_freq = 2

        # 伪标签
        if self.is_supervised:
            self.train_label_path = None
        else:
            self.train_label_path = "/media/ubuntu/4T/ALISURE/USS/ConTa/pseudo_mask_voc/result/2/sem_seg/train_aug"
            # self.train_label_path = "/mnt/4T/ALISURE/USS/ConTa/pseudo_mask_voc/result/2/sem_seg/train_aug"
            pass

        # 网络
        self.Net, self.met_name = DualNetDeepLabV3Plus, "DualNetDeepLabV3Plus"
        self.data_root_path = self.get_data_root_path()

        run_name = "12"
        self.model_name = "{}_{}_{}_{}_{}_{}_{}_{}".format(
            run_name, self.met_name, self.num_classes, self.epoch_num,
            self.batch_size, self.save_epoch_freq, self.input_size, self.output_stride)
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
../../../WSS_Model_My/DSS/12_DualNetDeepLabV3Plus_20_80_32_2_352_16/10.pth
2021-05-02 18:36:08 [E: 10] val mae:0.0697 f1:0.8862 acc:0.8862
2021-05-02 18:36:08 [E: 10] ss 
Overall Acc: 0.764122
Mean Acc: 0.557173
FreqW Acc: 0.636729
Mean IoU: 0.367440
"""


if __name__ == '__main__':
    config = Config()
    train(config=config)
    pass
