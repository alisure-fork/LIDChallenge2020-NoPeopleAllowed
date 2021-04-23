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
from torch.nn.parallel.data_parallel import DataParallel
sys.path.append("../../")
from util_network import DeepLabV3Plus
from deep_labv3plus_pytorch.metrics import StreamSegMetrics


class SSRunner(object):

    def __init__(self, config):
        self.config = config

        # Data
        self.dataset_ss_train, _, self.dataset_ss_val = DatasetUtil.get_dataset_by_type(
            DatasetUtil.dataset_type_ss, self.config.ss_size, is_balance=self.config.is_balance_data,
            data_root=self.config.data_root_path, train_label_path=self.config.label_path)
        self.data_loader_ss_train = DataLoader(self.dataset_ss_train, self.config.ss_batch_size,
                                               True, num_workers=16, drop_last=True)
        self.data_loader_ss_val = DataLoader(self.dataset_ss_val, self.config.ss_batch_size,
                                             False, num_workers=16, drop_last=True)

        # Model
        self.net = self.config.Net(num_classes=self.config.ss_num_classes, output_stride=self.config.output_stride)

        if self.config.only_train_ss:
            self.net = BalancedDataParallel(0, self.net, dim=0).cuda()
        else:
            self.net = DataParallel(self.net).cuda()
            pass
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

    def train_ss(self, start_epoch=0, model_file_name=None):
        if model_file_name is not None:
            Tools.print("Load model form {}".format(model_file_name), txt_path=self.config.ss_save_result_txt)
            self.load_model(model_file_name)
            pass

        # self.eval_ss(epoch=0)

        for epoch in range(start_epoch, self.config.ss_epoch_num):
            Tools.print()
            Tools.print('Epoch:{:2d}, lr={:.6f} lr2={:.6f}'.format(
                epoch, self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[1]['lr']),
                txt_path=self.config.ss_save_result_txt)

            ###########################################################################
            # 1 训练模型
            all_loss = 0.0
            self.net.train()
            if self.config.is_balance_data:
                self.dataset_ss_train.reset()
                pass
            for i, (inputs, labels) in tqdm(enumerate(self.data_loader_ss_train),
                                            total=len(self.data_loader_ss_train)):
                inputs, labels = inputs.float().cuda(), labels.long().cuda()
                self.optimizer.zero_grad()

                result = self.net(inputs)
                loss = self.ce_loss(result, labels)

                loss.backward()
                self.optimizer.step()

                all_loss += loss.item()

                if (i + 1) % (len(self.data_loader_ss_train) // 10) == 0:
                    self.eval_ss(epoch=epoch)
                    pass
                pass
            self.scheduler.step()
            ###########################################################################

            Tools.print("[E:{:3d}/{:3d}] ss loss:{:.4f}".format(
                epoch, self.config.ss_epoch_num, all_loss / len(self.data_loader_ss_train)),
                txt_path=self.config.ss_save_result_txt)

            ###########################################################################
            # 2 保存模型
            if epoch % self.config.ss_save_epoch_freq == 0:
                Tools.print()
                save_file_name = Tools.new_dir(os.path.join(
                    self.config.ss_model_dir, "ss_{}.pth".format(epoch)))
                torch.save(self.net.state_dict(), save_file_name)
                Tools.print("Save Model to {}".format(save_file_name), txt_path=self.config.ss_save_result_txt)
                Tools.print()
                pass
            ###########################################################################

            ###########################################################################
            # 3 评估模型
            if epoch % self.config.ss_eval_epoch_freq == 0:
                self.eval_ss(epoch=epoch)
                pass
            ###########################################################################

            pass

        # Final Save
        Tools.print()
        save_file_name = Tools.new_dir(os.path.join(
            self.config.ss_model_dir, "ss_final_{}.pth".format(self.config.ss_epoch_num)))
        torch.save(self.net.state_dict(), save_file_name)
        Tools.print("Save Model to {}".format(save_file_name), txt_path=self.config.ss_save_result_txt)
        Tools.print()

        self.eval_ss(epoch=self.config.ss_epoch_num)
        pass

    def eval_ss(self, epoch=0, model_file_name=None):
        if model_file_name is not None:
            Tools.print("Load model form {}".format(model_file_name), txt_path=self.config.ss_save_result_txt)
            self.load_model(model_file_name)
            pass

        self.net.eval()
        metrics = StreamSegMetrics(self.config.ss_num_classes)
        with torch.no_grad():
            for i, (inputs, labels) in tqdm(enumerate(self.data_loader_ss_val), total=len(self.data_loader_ss_val)):
                inputs = inputs.float().cuda()
                labels = labels.long().cuda()
                outputs = self.net(inputs)
                preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                targets = labels.cpu().numpy()

                metrics.update(targets, preds)
                pass
            pass

        score = metrics.get_results()
        Tools.print("{} {}".format(epoch, metrics.to_str(score)), txt_path=self.config.ss_save_result_txt)
        return score

    def inference_ss(self, model_file_name=None, data_loader=None, save_path=None):
        if model_file_name is not None:
            Tools.print("Load model form {}".format(model_file_name), txt_path=self.config.ss_save_result_txt)
            self.load_model(model_file_name)
            pass

        final_save_path = Tools.new_dir("{}_final".format(save_path))

        self.net.eval()
        metrics = StreamSegMetrics(self.config.ss_num_classes)
        with torch.no_grad():
            for i, (inputs, labels, image_info_list) in tqdm(enumerate(data_loader), total=len(data_loader)):
                assert len(image_info_list) == 1

                # 标签
                max_size = 1000
                size = Image.open(image_info_list[0]).size
                basename = os.path.basename(image_info_list[0])
                final_name = os.path.join(final_save_path, basename.replace(".JPEG", ".png"))
                if os.path.exists(final_name):
                    continue

                if size[0] < max_size and size[1] < max_size:
                    targets = F.interpolate(torch.unsqueeze(labels[0].float().cuda(), dim=0),
                                            size=(size[1], size[0]), mode="nearest").detach().cpu()
                else:
                    targets = F.interpolate(torch.unsqueeze(labels[0].float(), dim=0),
                                            size=(size[1], size[0]), mode="nearest")
                targets = targets[0].long().numpy()

                # 预测
                outputs = 0
                for input_index, input_one in enumerate(inputs):
                    output_one = self.net(input_one.float().cuda())
                    if size[0] < max_size and size[1] < max_size:
                        outputs += F.interpolate(output_one, size=(size[1], size[0]),
                                                 mode="bilinear", align_corners=False).detach().cpu()
                    else:
                        outputs += F.interpolate(output_one.detach().cpu(), size=(size[1], size[0]),
                                                 mode="bilinear", align_corners=False)
                        pass
                    pass
                outputs = outputs / len(inputs)
                preds = outputs.max(dim=1)[1].numpy()

                # 计算
                metrics.update(targets, preds)

                if save_path:
                    Image.open(image_info_list[0]).save(os.path.join(save_path, basename))
                    DataUtil.gray_to_color(np.asarray(targets[0], dtype=np.uint8)).save(
                        os.path.join(save_path, basename.replace(".JPEG", "_l.png")))
                    DataUtil.gray_to_color(np.asarray(preds[0], dtype=np.uint8)).save(
                        os.path.join(save_path, basename.replace(".JPEG", ".png")))
                    Image.fromarray(np.asarray(preds[0], dtype=np.uint8)).save(final_name)
                    pass
                pass
            pass

        score = metrics.get_results()
        Tools.print("{}".format(metrics.to_str(score)), txt_path=self.config.ss_save_result_txt)
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

    if config.only_inference_ss:
        dataset_ss_inference_val, dataset_ss_inference_test = DatasetUtil.get_dataset_by_type(
            DatasetUtil.dataset_type_ss_scale, config.ss_size, scales=config.scales,
            data_root=config.data_root_path, train_label_path=config.label_path)
        data_loader_ss_inference_val = DataLoader(dataset_ss_inference_val, 1, False, num_workers=16)
        data_loader_ss_inference_test = DataLoader(dataset_ss_inference_test, 1, False, num_workers=16)

        ss_runner.inference_ss(model_file_name=config.model_file_name, data_loader=data_loader_ss_inference_val,
                               save_path=Tools.new_dir(os.path.join(config.eval_save_path, "val")))
        ss_runner.inference_ss(model_file_name=config.model_file_name, data_loader=data_loader_ss_inference_test,
                               save_path=Tools.new_dir(os.path.join(config.eval_save_path, "test")))
        return

    if config.only_eval_ss:
        ss_runner.eval_ss(epoch=0, model_file_name=config.model_file_name)
        return

    # 训练MIC
    if config.only_train_ss:
        ss_runner.train_ss(start_epoch=0, model_file_name=None)
        return

    pass


class Config(object):

    def __init__(self):
        # self.gpu_id_1, self.gpu_id_4 = "2", "0, 1, 2, 3"
        self.gpu_id_1, self.gpu_id_4 = "0", "0, 1, 2, 3"

        # 流程控制
        self.only_train_ss = False  # 是否训练SS
        self.is_balance_data = True
        self.only_eval_ss = False  # 是否评估SS
        self.only_inference_ss = True  # 是否推理SS

        # 原始数据训练
        # self.scales = (1.0, 0.5, 1.5)
        # self.model_file_name = "../../../WSS_Model_SS/4_DeepLabV3PlusResNet101_201_10_18_1_352/ss_final_10.pth"
        # self.eval_save_path = "../../../WSS_Model_SS_EVAL/4_DeepLabV3PlusResNet101_201_10_18_1_352/ss_final_10_scales"

        # 平衡数据训练
        self.scales = (1.0, 0.5, 1.5)
        self.model_file_name = "../../../WSS_Model_SS/6_DeepLabV3PlusResNet101_201_10_18_1_352_balance/ss_8.pth"
        self.eval_save_path = "../../../WSS_Model_SS_EVAL/6_DeepLabV3PlusResNet101_201_10_18_1_352_balance/ss_8_scales"

        # 其他方法生成的伪标签
        # self.label_path = "/mnt/4T/ALISURE/USS/WSS_CAM/cam/1_CAMNet_200_32_256_0.5"
        # self.label_path = "/mnt/4T/ALISURE/USS/WSS_CAM/cam_4/1_200_32_256_0.5"
        # self.label_path = "/mnt/4T/ALISURE/USS/WSS_CAM/cam_4/2_1_200_32_256_0.5"
        self.label_path = "/media/ubuntu/4T/ALISURE/USS/ConTa/pseudo_mask/result/2/sem_seg"

        self.ss_num_classes = 201
        self.ss_epoch_num = 10
        self.ss_milestones = [5, 8]
        self.ss_batch_size = 6 * (len(self.gpu_id_4.split(",")) - 1)
        self.ss_lr = 0.001
        self.ss_save_epoch_freq = 1
        self.ss_eval_epoch_freq = 1

        # 图像大小
        self.ss_size = 352
        self.output_stride = 16

        # 网络
        self.Net = DeepLabV3Plus

        self.data_root_path = self.get_data_root_path()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id_4) if self.only_train_ss else str(self.gpu_id_1)

        run_name = "6"
        self.model_name = "{}_{}_{}_{}_{}_{}_{}{}".format(
            run_name, "DeepLabV3PlusResNet101", self.ss_num_classes, self.ss_epoch_num, self.ss_batch_size,
            self.ss_save_epoch_freq, self.ss_size, "_balance" if self.is_balance_data else "")
        Tools.print(self.model_name)

        self.ss_model_dir = "../../../WSS_Model_SS/{}".format(self.model_name)
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
"""


"""
../../../WSS_Model_SS/4_DeepLabV3PlusResNet101_201_10_18_1_352/ss_final_10.pth
Overall Acc: 0.818186
Mean Acc: 0.662312
FreqW Acc: 0.715434
Mean IoU: 0.470527

Overall Acc: 0.843774
Mean Acc: 0.621268
FreqW Acc: 0.750521
Mean IoU: 0.461591

../../../WSS_Model_SS/5_DeepLabV3PlusResNet101_201_10_12_1_352_balance/ss_8.pth
Overall Acc: 0.819173
Mean Acc: 0.671673
FreqW Acc: 0.717279
Mean IoU: 0.460044

Overall Acc: 0.846122
Mean Acc: 0.633350
FreqW Acc: 0.753838
Mean IoU: 0.458760

../../../WSS_Model_SS/6_DeepLabV3PlusResNet101_201_10_18_1_352_balance/ss_8.pth  # all data balance
Overall Acc: 0.820996
Mean Acc: 0.685476
FreqW Acc: 0.720114
Mean IoU: 0.473003

Overall Acc: 0.848850
Mean Acc: 0.643933
FreqW Acc: 0.758975
Mean IoU: 0.465997
"""


if __name__ == '__main__':
    config = Config()
    train(config=config)
    pass
