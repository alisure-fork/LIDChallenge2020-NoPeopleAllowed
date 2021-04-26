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
from torch.utils.data import DataLoader, Dataset
sys.path.append("../../")
from my_util_data2 import DatasetUtil
from deep_labv3plus_pytorch.metrics import StreamSegMetrics
from my_util_network import DeepLabV3Plus, deeplabv3_resnet50, deeplabv3plus_resnet101


class VOCRunner(object):

    def __init__(self, config):
        self.config = config

        # Data
        self.dataset_voc_train = DatasetUtil.get_dataset_by_type(
            DatasetUtil.dataset_type_ss_voc_train, self.config.ss_size, data_root=self.config.data_root_path,
            train_label_path=self.config.train_label_path, sampling=self.config.sampling)
        self.data_loader_ss_train = DataLoader(
            self.dataset_voc_train, self.config.ss_batch_size, shuffle=True, num_workers=16)

        self.dataset_voc_val = DatasetUtil.get_dataset_by_type(
            DatasetUtil.dataset_type_ss_voc_val, self.config.ss_size, data_root=self.config.data_root_path,
            sampling=self.config.sampling)
        self.data_loader_ss_val = DataLoader(self.dataset_voc_val, 1, shuffle=False, num_workers=8)

        # Model
        self.net = self.config.Net(num_classes=self.config.ss_num_classes,
                                   output_stride=self.config.output_stride, arch=self.config.arch)
        self.net = nn.DataParallel(self.net).cuda()
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
            for i, (inputs, labels) in tqdm(enumerate(self.data_loader_ss_train),
                                            total=len(self.data_loader_ss_train)):
                inputs, labels = inputs.float().cuda(), labels.long().cuda()
                self.optimizer.zero_grad()

                result = self.net(inputs)
                loss = self.ce_loss(result, labels)

                loss.backward()
                self.optimizer.step()

                all_loss += loss.item()
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

    voc_runner = VOCRunner(config=config)

    if config.only_inference_ss:
        dataset_ss_inference_val, dataset_ss_inference_test = DatasetUtil.get_dataset_by_type(
            DatasetUtil.dataset_type_ss_voc_val_scale, config.ss_size,
            scales=config.scales, data_root=config.data_root_path)
        data_loader_ss_inference_val = DataLoader(dataset_ss_inference_val, 1, False, num_workers=8)
        data_loader_ss_inference_test = DataLoader(dataset_ss_inference_test, 1, False, num_workers=8)

        ss_runner.inference_ss(model_file_name=config.model_file_name, data_loader=data_loader_ss_inference_val,
                               save_path=Tools.new_dir(os.path.join(config.eval_save_path, "val")))
        ss_runner.inference_ss(model_file_name=config.model_file_name, data_loader=data_loader_ss_inference_test,
                               save_path=Tools.new_dir(os.path.join(config.eval_save_path, "test")))
        return

    if config.only_eval_ss:
        voc_runner.eval_ss(epoch=0, model_file_name=os.path.join(config.ss_model_dir, "ss_final_20.pth"))
        return

    if config.only_train_ss:
        voc_runner.train_ss(start_epoch=0, model_file_name=None)
        return

    pass


class Config(object):

    def __init__(self):
        self.gpu_id_1, self.gpu_id_4 = "1", "1, 2, 3"

        # 流程控制
        self.only_train_ss = True  # 是否训练
        self.only_inference_ss = False
        self.only_eval_ss = False
        self.sampling = False

        self.ss_num_classes = 21
        self.ss_epoch_num = 100
        self.ss_milestones = [40, 70]
        self.ss_batch_size = 6 * len(self.gpu_id_4.split(","))
        self.ss_lr = 0.001
        self.ss_save_epoch_freq = 5
        self.ss_eval_epoch_freq = 5

        # 图像大小
        self.ss_size = 513  # 352
        self.output_stride = 16

        # 伪标签
        self.train_label_path = "/mnt/4T/ALISURE/USS/ConTa/pseudo_mask_voc/result/2/sem_seg/train_aug"
        # self.train_label_path = None

        # 推理
        self.scales = (1.0, 0.75, 0.5, 1.25, 1.5, 1.75, 2.0)
        self.model_file_name = "../../../WSS_Model_VOC/1_DeepLabV3PlusResNet50_21_20_18_1_513/ss_final_20.pth"
        self.eval_save_path = "../../../WSS_Model_SS_EVAL/1_DeepLabV3PlusResNet50_21_20_18_1_513/ss_20_scales_7"

        # 网络
        self.Net = DeepLabV3Plus
        # self.arch, self.arch_name = deeplabv3_resnet50, "DeepLabV3PlusResNet50"
        self.arch, self.arch_name = deeplabv3plus_resnet101, "DeepLabV3PlusResNet101"

        self.data_root_path = self.get_data_root_path()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id_4) if self.only_train_ss else str(self.gpu_id_1)

        run_name = "5"
        self.model_name = "{}_{}_{}_{}_{}_{}_{}".format(
            run_name, self.arch_name, self.ss_num_classes, self.ss_epoch_num,
            self.ss_batch_size, self.ss_save_epoch_freq, self.ss_size)
        Tools.print(self.model_name)

        self.ss_model_dir = "../../../WSS_Model_VOC/{}".format(self.model_name)
        self.ss_save_result_txt = Tools.new_dir("{}/result.txt".format(self.ss_model_dir))
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
../../../WSS_Model_VOC/1_DeepLabV3PlusResNet50_21_20_18_1_513/ss_final_20.pth
Overall Acc: 0.893457
Mean Acc: 0.811502
FreqW Acc: 0.820158
Mean IoU: 0.629518

../../../WSS_Model_VOC/3_DeepLabV3PlusResNet50_21_100_18_5_513/ss_final_100.pth
Overall Acc: 0.942903
Mean Acc: 0.867867
FreqW Acc: 0.898047
Mean IoU: 0.754826

../../../WSS_Model_VOC/4_DeepLabV3PlusResNet50_21_100_18_5_513/ss_60.pth
Overall Acc: 0.902363
Mean Acc: 0.782532
FreqW Acc: 0.830100
Mean IoU: 0.651673
"""


if __name__ == '__main__':
    config = Config()
    train(config=config)
    pass
