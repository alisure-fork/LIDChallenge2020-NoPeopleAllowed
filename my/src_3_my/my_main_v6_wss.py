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
from my_util_data2 import DatasetUtil, DataUtil
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
            for i, (inputs, labels, image_info_list, label_info_list) in tqdm(
                    enumerate(self.data_loader_ss_val), total=len(self.data_loader_ss_val)):
                assert len(image_info_list) == 1
                size = Image.open(label_info_list[0]).size

                inputs = inputs.float().cuda()
                outputs = self.net(inputs)
                outputs = F.interpolate(outputs, size=(size[1], size[0]), mode="bilinear", align_corners=False)

                preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                targets = np.expand_dims(np.asarray(Image.open(label_info_list[0])), axis=0)

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
            for i, (inputs, labels, image_info_list, label_info_list) in tqdm(
                    enumerate(data_loader), total=len(data_loader)):
                assert len(image_info_list) == 1

                # 标签
                basename = os.path.basename(image_info_list[0])
                final_name = os.path.join(final_save_path, basename.replace(".jpg", ".png"))
                size = Image.open(image_info_list[0]).size
                if os.path.exists(final_name):
                    continue

                target_im = Image.fromarray(np.zeros_like(np.asarray(Image.open(image_info_list[0])))).convert("L") \
                    if label_info_list[0] == 1 else Image.open(label_info_list[0])
                targets = np.expand_dims(np.asarray(target_im), axis=0)

                # 预测
                outputs = 0
                for input_index, input_one in enumerate(inputs):
                    output_one = self.net(input_one.float().cuda())
                    outputs += F.interpolate(output_one, size=(size[1], size[0]),
                                             mode="bilinear", align_corners=False).detach().cpu()
                    pass
                outputs = outputs / len(inputs)
                preds = outputs.max(dim=1)[1].numpy()

                # 计算
                metrics.update(targets, preds)

                if save_path:
                    Image.open(image_info_list[0]).save(os.path.join(save_path, basename))
                    DataUtil.gray_to_color(np.asarray(targets[0], dtype=np.uint8)).save(
                        os.path.join(save_path, basename.replace(".jpg", "_l.png")))
                    DataUtil.gray_to_color(np.asarray(preds[0], dtype=np.uint8)).save(
                        os.path.join(save_path, basename.replace(".jpg", ".png")))
                    Image.fromarray(np.asarray(preds[0], dtype=np.uint8)).save(final_name)
                    pass
                pass
            pass

        score = metrics.get_results()
        Tools.print("{}".format(metrics.to_str(score)), txt_path=self.config.ss_save_result_txt)
        return score

    def inference_ss_logits(self, model_file_name=None, data_loader=None, save_path=None):
        if model_file_name is not None:
            Tools.print("Load model form {}".format(model_file_name))
            self.load_model(model_file_name)
            pass

        logit_save_path = Tools.new_dir("{}_logit".format(save_path))

        self.net.eval()
        metrics = StreamSegMetrics(self.config.ss_num_classes)
        with torch.no_grad():
            for i, (inputs, labels, image_info_list, label_info_list) in tqdm(
                    enumerate(data_loader), total=len(data_loader)):
                assert len(image_info_list) == 1

                # 标签
                basename = os.path.basename(image_info_list[0])
                size = Image.open(image_info_list[0]).size
                logit_file_path = os.path.join(logit_save_path, basename.replace(".jpg", ".npy"))
                if os.path.exists(logit_file_path):
                    continue

                target_im = Image.fromarray(np.zeros_like(np.asarray(Image.open(image_info_list[0])))).convert("L") \
                    if label_info_list[0] == 1 else Image.open(label_info_list[0])
                target_im = target_im.resize((size[0] // 4, size[1] // 4))
                targets = np.expand_dims(np.asarray(target_im), axis=0)

                # 预测
                outputs = 0
                for input_index, input_one in enumerate(inputs):
                    output_one = self.net(input_one.float().cuda())
                    outputs += F.interpolate(output_one, size=(size[1] // 4, size[0] // 4),
                                             mode="bilinear", align_corners=False).detach().cpu()
                    pass
                outputs = outputs / len(inputs)
                preds = outputs.max(dim=1)[1].numpy()

                # 计算
                metrics.update(targets, preds)

                if save_path:
                    np.save(logit_file_path, outputs[0].numpy())
                    pass
                pass
            pass

        score = metrics.get_results()
        Tools.print("{}".format(metrics.to_str(score)))
        return score

    def inference_ss_logits_single_scale(self, model_file_name=None, data_loader=None, save_path=None):
        if model_file_name is not None:
            Tools.print("Load model form {}".format(model_file_name))
            self.load_model(model_file_name)
            pass

        logit_save_path = Tools.new_dir("{}_logit".format(save_path))

        self.net.eval()
        metrics = StreamSegMetrics(self.config.ss_num_classes)
        with torch.no_grad():
            for i, (inputs, labels, image_info_list, label_info_list) in tqdm(
                    enumerate(data_loader), total=len(data_loader)):
                assert len(image_info_list) == 1

                basename = os.path.basename(image_info_list[0])
                logit_file_path = os.path.join(logit_save_path, basename.replace(".jpg", ".npy"))
                im = Image.open(image_info_list[0])

                ori_size = (im.size[1], im.size[0])
                # logit_size = (ori_size[0] // 4, ori_size[1] // 4)
                logit_size = ori_size

                # 标签
                target_im = Image.fromarray(np.zeros_like(np.asarray(im))).convert("L") \
                    if label_info_list[0] == 1 else Image.open(label_info_list[0])
                targets = np.expand_dims(np.array(target_im.resize((logit_size[1], logit_size[0]))), axis=0)

                # 预测
                output_one = self.net(inputs[0].float().cuda())
                outputs = self._up_to_target(output_one, target_size=logit_size).detach().cpu()
                preds = outputs.max(dim=1)[1].numpy()

                # 计算
                metrics.update(targets, preds)

                if save_path:
                    np.save(logit_file_path, outputs[0].numpy())
                    pass

                pass
            pass

        score = metrics.get_results()
        Tools.print("{}".format(metrics.to_str(score)))
        return score

    @staticmethod
    def _up_to_target(source, target_size, mode="bilinear"):
        if source.size()[2] != target_size[0] or source.size()[3] != target_size[1]:
            align_corners = True if mode == "nearest" else False
            _source = torch.nn.functional.interpolate(source, size=target_size, mode=mode, align_corners=align_corners)
            pass
        else:
            _source = source
        return _source

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
        dataset_ss_inference_train, dataset_ss_inference_val, dataset_ss_inference_test = DatasetUtil.get_dataset_by_type(
            DatasetUtil.dataset_type_ss_voc_val_scale, config.ss_size,
            scales=None if config.single_scale else config.scales, data_root=config.data_root_path)
        data_loader_ss_inference_train = DataLoader(dataset_ss_inference_train, 1, False, num_workers=8)
        data_loader_ss_inference_val = DataLoader(dataset_ss_inference_val, 1, False, num_workers=8)
        data_loader_ss_inference_test = DataLoader(dataset_ss_inference_test, 1, False, num_workers=8)

        if config.save_logits:
            inference_ss = voc_runner.inference_ss_logits
            if config.single_scale:
                inference_ss = voc_runner.inference_ss_logits_single_scale
        else:
            inference_ss = voc_runner.inference_ss
            pass

        inference_ss(model_file_name=config.model_file_name, data_loader=data_loader_ss_inference_train,
                     save_path=Tools.new_dir(os.path.join(config.eval_save_path, "train")))
        # inference_ss(model_file_name=config.model_file_name, data_loader=data_loader_ss_inference_val,
        #              save_path=Tools.new_dir(os.path.join(config.eval_save_path, "val")))
        # inference_ss(model_file_name=config.model_file_name, data_loader=data_loader_ss_inference_test,
        #              save_path=Tools.new_dir(os.path.join(config.eval_save_path, "test")))
        return

    if config.only_eval_ss:
        voc_runner.eval_ss(
            epoch=0, model_file_name="../../../WSS_Model_VOC/5_DeepLabV3PlusResNet101_21_100_18_5_513/ss_90.pth")
        return

    if config.only_train_ss:
        voc_runner.train_ss(start_epoch=0, model_file_name=None)
        return

    pass


class Config(object):

    def __init__(self):
        # 流程控制
        self.is_supervised = False
        self.only_train_ss = True  # 是否训练
        self.sampling = False
        self.only_eval_ss = False

        self.only_inference_ss = False
        self.single_scale = False
        self.save_logits = False

        # self.gpu_id_1, self.gpu_id_4 = "1", "1, 2, 3"
        self.gpu_id_1, self.gpu_id_4 = "1", "0, 1, 2, 3"

        self.ss_num_classes = 21
        self.ss_epoch_num = 100
        self.ss_milestones = [40, 70]
        self.ss_lr = 0.001
        self.ss_save_epoch_freq = 5
        self.ss_eval_epoch_freq = 5

        # 图像大小
        # self.ss_size = 513  # 352
        # self.ss_batch_size = 6 * len(self.gpu_id_4.split(","))
        self.ss_size = 352
        self.ss_batch_size = 12 * len(self.gpu_id_4.split(","))
        self.output_stride = 16

        # 伪标签
        self.train_label_path = self.get_train_label(self.is_supervised)

        # 推理
        self.scales = (1.0, 0.75, 0.5, 1.25, 1.5)
        # self.model_file_name = "../../../WSS_Model_VOC/5_DeepLabV3PlusResNet101_21_100_18_5_513/ss_90.pth"
        # self.eval_save_path = "../../../WSS_Model_VOC_EVAL/5_DeepLabV3PlusResNet101_21_100_18_5_513/ss_90_scales_5"
        # self.model_file_name = "../../../WSS_Model_VOC/6_DeepLabV3PlusResNet101_21_100_36_5_352/ss_final_100.pth"
        # self.eval_save_path = "../../../WSS_Model_VOC_EVAL/6_DeepLabV3PlusResNet101_21_100_36_5_352/ss_100_scales_5"

        # 1
        # self.model_file_name = "../../../WSS_Model_VOC/5_DeepLabV3PlusResNet101_21_100_48_5_352/ss_final_100.pth"
        # self.eval_save_path = "../../../WSS_Model_VOC_EVAL/5_DeepLabV3PlusResNet101_21_100_48_5_352/ss_100_scales_5"
        # 2
        self.scales = (1.0, )
        # self.model_file_name = "../../../WSS_Model_VOC/7_DeepLabV3PlusResNet101_21_100_48_5_352/ss_35.pth"
        # self.eval_save_path = "../../../WSS_Model_VOC_EVAL/7_DeepLabV3PlusResNet101_21_100_48_5_352/ss_35_scales_1"
        self.model_file_name = "../../../WSS_Model_VOC/7_DeepLabV3PlusResNet101_21_100_48_5_352/ss_final_100.pth"
        self.eval_save_path = "../../../WSS_Model_VOC_EVAL/7_DeepLabV3PlusResNet101_21_100_48_5_352/ss_100_scales_1"

        # 网络
        self.Net = DeepLabV3Plus
        # self.arch, self.arch_name = deeplabv3_resnet50, "DeepLabV3PlusResNet50"
        self.arch, self.arch_name = deeplabv3plus_resnet101, "DeepLabV3PlusResNet101"

        self.data_root_path = self.get_data_root_path()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id_4) if self.only_train_ss else str(self.gpu_id_1)

        run_name = "8"
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

    @staticmethod
    def get_train_label(is_supervised):
        if is_supervised:
            return None

        if "Linux" in platform.platform():
            # train_label_path = "/mnt/4T/ALISURE/USS/ConTa/pseudo_mask_voc/result/2/sem_seg/train_aug"
            train_label_path = "/mnt/4T/ALISURE/USS/WSS_Model_VOC_EVAL/6_DeepLabV3PlusResNet101_21_100_36_5_352/ss_100_scales_5/train_final"
            if not os.path.isdir(train_label_path):
                # 1
                # train_label_path = "/media/ubuntu/4T/ALISURE/USS/ConTa/pseudo_mask_voc/result/2/sem_seg/train_aug"
                # 2
                # train_label_path = "/media/ubuntu/4T/ALISURE/USS/WSS_Model_VOC_EVAL/5_DeepLabV3PlusResNet101_21_100_48_5_352/ss_100_scales_5/train_crf_final"
                # 3
                train_label_path = "/media/ubuntu/4T/ALISURE/USS/WSS_Model_VOC_EVAL/7_DeepLabV3PlusResNet101_21_100_48_5_352/ss_35_scales_1/train_crf_final"
                pass
            return train_label_path
        pass

    pass


"""
Supervised
../../../WSS_Model_VOC/3_DeepLabV3PlusResNet50_21_100_18_5_513/ss_final_100.pth
Overall Acc: 0.942903
Mean Acc: 0.867867
FreqW Acc: 0.898047
Mean IoU: 0.754826

../../../WSS_Model_VOC/6_DeepLabV3PlusResNet101_21_100_24_5_513/ss_final_100.pth
Overall Acc: 0.948508
Mean Acc: 0.879520
FreqW Acc: 0.907612
Mean IoU: 0.776816
"""


"""
3 GPU
../../../WSS_Model_VOC/5_DeepLabV3PlusResNet101_21_100_18_5_513/ss_90.pth
Overall Acc: 0.905740
Mean Acc: 0.792019
FreqW Acc: 0.835543
Mean IoU: 0.660259
../../../WSS_Model_VOC_EVAL/5_DeepLabV3PlusResNet101_21_100_18_5_513/ss_90_scales_5
Val
Overall Acc: 0.908860
Mean Acc: 0.798223
FreqW Acc: 0.840463
Mean IoU: 0.668947
Train
Overall Acc: 0.879955
Mean Acc: 0.789405
FreqW Acc: 0.792041
Mean IoU: 0.656029
"""


"""
3 GPU Self Training
../../../WSS_Model_VOC/6_DeepLabV3PlusResNet101_21_100_36_5_352/ss_final_100.pth
Overall Acc: 0.907379
Mean Acc: 0.804082
FreqW Acc: 0.838899
Mean IoU: 0.666433
../../../WSS_Model_VOC_EVAL/6_DeepLabV3PlusResNet101_21_100_36_5_352/ss_100_scales_5
val
Overall Acc: 0.911125
Mean Acc: 0.809898
FreqW Acc: 0.844886
Mean IoU: 0.675588
train
Overall Acc: 0.880241
Mean Acc: 0.796864
FreqW Acc: 0.793459
Mean IoU: 0.654750
"""


"""
4 GPU

1
../../../WSS_Model_VOC/5_DeepLabV3PlusResNet101_21_100_48_5_352/ss_final_100.pth
2021-05-04 17:14:58 100 
Overall Acc: 0.906540
Mean Acc: 0.773918
FreqW Acc: 0.835605
Mean IoU: 0.660385
../../../WSS_Model_VOC_EVAL/5_DeepLabV3PlusResNet101_21_100_48_5_352/ss_100_scales_5
val
Overall Acc: 0.911452
Mean Acc: 0.777744
FreqW Acc: 0.843194
Mean IoU: 0.672578
Train
Overall Acc: 0.884174
Mean Acc: 0.770555
FreqW Acc: 0.797313
Mean IoU: 0.660239

2
../../../WSS_Model_VOC/7_DeepLabV3PlusResNet101_21_100_48_5_352/ss_35.pth
val
Overall Acc: 0.912211
Mean Acc: 0.796920
FreqW Acc: 0.846161
Mean IoU: 0.677405
val scale 5
Overall Acc: 0.917007
Mean Acc: 0.801310
FreqW Acc: 0.853644
Mean IoU: 0.691245
train
Overall Acc: 0.880951
Mean Acc: 0.782357
FreqW Acc: 0.794273
Mean IoU: 0.652178
../../../WSS_Model_VOC/7_DeepLabV3PlusResNet101_21_100_48_5_352/ss_final_100.pth
val
Overall Acc: 0.911502
Mean Acc: 0.777925
FreqW Acc: 0.843317
Mean IoU: 0.670318
val scale 5
Overall Acc: 0.915412
Mean Acc: 0.781614
FreqW Acc: 0.849520
Mean IoU: 0.681968
train
Overall Acc: 0.881573
Mean Acc: 0.771317
FreqW Acc: 0.793992
Mean IoU: 0.651336
"""


if __name__ == '__main__':
    config = Config()
    train(config=config)
    pass
