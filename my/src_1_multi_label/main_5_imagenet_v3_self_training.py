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
from torchstat import stat
import torch.optim as optim
import torch.nn.functional as F
from alisuretool.Tools import Tools
import torch.backends.cudnn as cudnn
from util_blance_gpu import BalancedDataParallel
from torch.utils.data import DataLoader, Dataset
from util_data import DataUtil, DatasetUtil, MyTransform
from torch.nn.parallel.data_parallel import DataParallel
sys.path.append("../../")
from deep_labv3plus_pytorch.metrics import StreamSegMetrics
from util_network import DeepLabV3Plus, deeplabv3plus_resnet50, deeplabv3plus_resnet101, deeplabv3plus_resnet152


class SSRunner(object):

    def __init__(self, config):
        self.config = config

        # Data
        self.dataset_ss_train, _, self.dataset_ss_val = DatasetUtil.get_dataset_by_type(
            DatasetUtil.dataset_type_ss, self.config.ss_size, is_balance=self.config.is_balance_data,
            data_root=self.config.data_root_path, train_label_path=self.config.label_path, max_size=self.config.max_size)
        self.data_loader_ss_train = DataLoader(self.dataset_ss_train, self.config.ss_batch_size,
                                               True, num_workers=16, drop_last=True)
        self.data_loader_ss_val = DataLoader(self.dataset_ss_val, self.config.ss_batch_size,
                                             False, num_workers=16, drop_last=True)

        # Model
        self.net = self.config.Net(num_classes=self.config.ss_num_classes,
                                   output_stride=self.config.output_stride, arch=self.config.arch)

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
        best_iou = 0.0

        for epoch in range(self.config.ss_epoch_num):
            Tools.print()
            Tools.print('Epoch:{:2d}, lr={:.6f} lr2={:.6f}'.format(
                epoch, self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[1]['lr']),
                txt_path=self.config.ss_save_result_txt)

            if epoch < start_epoch:
                self.scheduler.step()
                continue

            ###########################################################################
            # 1 训练模型
            all_loss = 0.0
            self.net.train()
            if self.config.is_balance_data:
                self.dataset_ss_train.reset()
                pass
            for i, (inputs, _labels) in tqdm(enumerate(self.data_loader_ss_train),
                                            total=len(self.data_loader_ss_train)):
                inputs, labels = inputs.float().cuda(), _labels.long().cuda()
                self.optimizer.zero_grad()

                result = self.net(inputs)
                loss = self.ce_loss(result, labels)

                loss.backward()
                self.optimizer.step()

                all_loss += loss.item()

                if (i + 1) % (len(self.data_loader_ss_train) // 10) == 0:
                    score = self.eval_ss(epoch=epoch)
                    mean_iou = score["Mean IoU"]
                    if mean_iou > best_iou:
                        best_iou = mean_iou
                        save_file_name = Tools.new_dir(os.path.join(
                            self.config.ss_model_dir, "ss_{}_{}_{}.pth".format(epoch, i, best_iou)))
                        torch.save(self.net.state_dict(), save_file_name)
                        Tools.print("Save Model to {}".format(save_file_name), txt_path=self.config.ss_save_result_txt)
                        Tools.print()
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
                score = self.eval_ss(epoch=epoch)
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

    def inference_ss_train(self, model_file_name=None, data_loader=None, save_path=None):
        if model_file_name is not None:
            Tools.print("Load model form {}".format(model_file_name), txt_path=self.config.ss_save_result_txt)
            self.load_model(model_file_name)
            pass

        final_save_path = Tools.new_dir("{}_final".format(save_path))

        self.net.eval()
        with torch.no_grad():
            for i, (inputs, labels, image_info_list) in tqdm(enumerate(data_loader), total=len(data_loader)):
                assert len(image_info_list) == 1

                # 标签
                basename = image_info_list[0].split("Data/DET/")[1]
                final_name = os.path.join(final_save_path, basename.replace(".JPEG", ".png"))
                final_name = Tools.new_dir(final_name)
                if os.path.exists(final_name):
                    continue

                # 预测
                outputs = self.net(inputs[0].float().cuda()).cpu()
                preds = outputs.max(dim=1)[1].numpy()

                if save_path:
                    Image.open(image_info_list[0]).convert("RGB").save(
                        Tools.new_dir(os.path.join(save_path, basename)))
                    DataUtil.gray_to_color(np.asarray(preds[0], dtype=np.uint8)).save(
                        os.path.join(save_path, basename.replace(".JPEG", ".png")))
                    Image.fromarray(np.asarray(preds[0], dtype=np.uint8)).save(final_name)
                    pass
                pass
            pass

        pass

    def load_model(self, model_file_name):
        Tools.print("Load model form {}".format(model_file_name), txt_path=self.config.ss_save_result_txt)
        checkpoint = torch.load(model_file_name)

        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1:
            # checkpoint = {key.replace("module.", ""): checkpoint[key] for key in checkpoint}
            pass

        self.net.load_state_dict(checkpoint, strict=True)
        Tools.print("Restore from {}".format(model_file_name), txt_path=self.config.ss_save_result_txt)
        pass

    def stat(self):
        stat(self.net, (3, self.config.ss_size, self.config.ss_size))
        pass

    pass


def train(config):

    ss_runner = SSRunner(config=config)

    # 统计参数量和计算量
    # ss_runner.stat()

    if config.only_inference_ss:
        if config.inference_ss_train:
            dataset_ss_inference_train = DatasetUtil.get_dataset_by_type(
                DatasetUtil.dataset_type_ss_scale_train, config.ss_size, data_root=config.data_root_path,
                train_label_path=config.label_path, max_size=config.max_size_inference)
            data_loader_ss_inference_train = DataLoader(dataset_ss_inference_train, 1, False, num_workers=16)
            ss_runner.inference_ss_train(model_file_name=config.model_file_name,
                                         data_loader=data_loader_ss_inference_train,
                                         save_path=Tools.new_dir(os.path.join(config.eval_save_path, "train")))
        else:
            dataset_ss_inference_val, dataset_ss_inference_test = DatasetUtil.get_dataset_by_type(
                DatasetUtil.dataset_type_ss_scale, config.ss_size, scales=config.scales,
                data_root=config.data_root_path, train_label_path=config.label_path, max_size=config.max_size_inference)
            if config.inference_ss_val:
                data_loader_ss_inference_val = DataLoader(dataset_ss_inference_val, 1, False, num_workers=16)
                ss_runner.inference_ss(model_file_name=config.model_file_name, data_loader=data_loader_ss_inference_val,
                                       save_path=Tools.new_dir(os.path.join(config.eval_save_path, "val")))
            else:
                data_loader_ss_inference_test = DataLoader(dataset_ss_inference_test, 1, False, num_workers=16)
                ss_runner.inference_ss(model_file_name=config.model_file_name, data_loader=data_loader_ss_inference_test,
                                       save_path=Tools.new_dir(os.path.join(config.eval_save_path, "test")))
                pass
            pass
        return

    if config.only_eval_ss:
        ss_runner.eval_ss(epoch=0, model_file_name=config.model_file_name)
        return

    if config.only_train_ss:
        # model_file = "../../../WSS_Model_SS_0602/1_DeepLabV3PlusResNet152_201_10_24_1_352_balance/ss_5.pth"
        # ss_runner.train_ss(start_epoch=6, model_file_name=model_file)

        ss_runner.train_ss(start_epoch=0, model_file_name=None)

        # model_file = "../../../WSS_Model_SS_0602/3_DeepLabV3PlusResNet152_201_10_18_1_352_8_balance/ss_7_8887_0.49612685291103387.pth"
        # ss_runner.train_ss(start_epoch=6, model_file_name=model_file)
        return

    pass


class Config(object):

    def __init__(self):
        # self.gpu_id_1, self.gpu_id_4 = "0", "0, 1, 2, 3"
        # self.gpu_id_1, self.gpu_id_4 = "1", "0, 1, 2, 3"
        self.gpu_id_1, self.gpu_id_4 = "2", "0, 1, 2, 3"
        # self.gpu_id_1, self.gpu_id_4 = "3", "0, 1, 2, 3"

        # 流程控制
        self.only_train_ss = True  # 是否训练SS
        self.is_balance_data = True  # 是否平衡数据
        self.only_eval_ss = False  # 是否评估SS
        self.only_inference_ss = False  # 是否推理SS
        self.inference_ss_train = False  # 是否推理训练集
        self.inference_ss_val = True  # 是否推理验证集

        # 测试相关
        self.scales, self.model_file_name, self.eval_save_path = self.inference_param()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id_4) if self.only_train_ss else str(self.gpu_id_1)

        # 其他方法生成的伪标签
        self.label_path = "/media/ubuntu/4T/ALISURE/USS/ConTa/pseudo_mask/result/2/sem_seg"
        # self.label_path = "/media/ubuntu/4T/ALISURE/USS/WSS_Model_SS_0602_EVAL/" \
        #                   "3_DeepLabV3PlusResNet152_201_10_18_1_352_8_balance/ss_7_train_500/train_final_resize"

        # 参数
        self.ss_num_classes = 201
        self.ss_epoch_num = 10
        self.ss_milestones = [5, 8]
        self.ss_batch_size = 6 * (len(self.gpu_id_4.split(",")) - 1)
        self.ss_lr = 0.001
        self.ss_save_epoch_freq = 1
        self.ss_eval_epoch_freq = 1
        self.ss_size = 352
        self.output_stride = 8
        self.max_size = 500
        self.max_size_inference = 500
        # self.max_size_inference = 0
        self.data_root_path = self.get_data_root_path()

        # 网络
        self.Net = DeepLabV3Plus
        # self.arch, self.arch_name = deeplabv3plus_resnet50, "DeepLabV3PlusResNet50"
        # self.arch, self.arch_name = deeplabv3plus_resnet101, "DeepLabV3PlusResNet101"
        self.arch, self.arch_name = deeplabv3plus_resnet152, "DeepLabV3PlusResNet152"

        run_name = "4_self_training_temp2"
        self.model_name = "{}_{}_{}_{}_{}_{}_{}_{}{}".format(
            run_name, self.arch_name, self.ss_num_classes, self.ss_epoch_num, self.ss_batch_size,
            self.ss_save_epoch_freq, self.ss_size, self.output_stride, "_balance" if self.is_balance_data else "")
        self.ss_model_dir = "../../../WSS_Model_SS_0602/{}".format(self.model_name)
        self.ss_save_result_txt = Tools.new_dir("{}/result.txt".format(self.ss_model_dir))
        Tools.print(self.model_name)

        if not self.only_train_ss:
            self.ss_batch_size = self.ss_batch_size // (len(self.gpu_id_4.split(",")) - 1)
            pass
        pass

    @staticmethod
    def inference_param():
        # 1 原始数据训练
        # scales = (1.0, 0.5, 1.5)
        # model_file_name = "../../../WSS_Model_SS/4_DeepLabV3PlusResNet50_201_10_18_1_352/ss_final_10.pth"
        # eval_save_path = "../../../WSS_Model_SS_EVAL/4_DeepLabV3PlusResNet50_201_10_18_1_352/ss_final_10_scales"

        # 2 平衡数据训练
        # scales = (1.0, 0.5, 1.5)
        # model_file_name = "../../../WSS_Model_SS/6_DeepLabV3PlusResNet50_201_10_18_1_352_balance/ss_8.pth"
        # eval_save_path = "../../../WSS_Model_SS_EVAL/6_DeepLabV3PlusResNet50_201_10_18_1_352_balance/ss_8_scales"

        # 3 平衡数据训练
        # scales = (1.0, 0.5, 1.5, 2.0)
        # model_file_name = "../../../WSS_Model_SS/6_DeepLabV3PlusResNet50_201_10_18_1_352_balance/ss_8.pth"
        # eval_save_path = "../../../WSS_Model_SS_EVAL/6_DeepLabV3PlusResNet50_201_10_18_1_352_balance/ss_8_scales_4"

        # 4 平衡数据训练
        # scales = (1.0, 0.75, 0.5, 1.25, 1.5, 1.75, 2.0)
        # model_file_name = "../../../WSS_Model_SS/6_DeepLabV3PlusResNet50_201_10_18_1_352_balance/ss_8.pth"
        # eval_save_path = "../../../WSS_Model_SS_EVAL/6_DeepLabV3PlusResNet50_201_10_18_1_352_balance/ss_8_scales_7"

        # 4 平衡数据训练
        # scales = (1.0, 0.75, 0.5, 1.25, 1.5, 1.75, 2.0)
        # model_file_name = "../../../WSS_Model_SS/7_DeepLabV3PlusResNet101_201_10_18_1_352_balance/ss_7.pth"
        # eval_save_path = "../../../WSS_Model_SS_EVAL/7_DeepLabV3PlusResNet101_201_10_18_1_352_balance/ss_7_scales_7"
        # model_file_name = "../../../WSS_Model_SS/7_DeepLabV3PlusResNet101_201_10_18_1_352_balance/ss_final_10.pth"
        # eval_save_path = "../../../WSS_Model_SS_EVAL/7_DeepLabV3PlusResNet101_201_10_18_1_352_balance/ss_10_scales_7"

        # 5 Resnet152
        # scales = (1.0, 0.75, 0.5, 1.25, 1.5, 1.75, 2.0)
        # model_file_name = "../../../WSS_Model_SS_0602/2_DeepLabV3PlusResNet152_201_10_18_1_352_balance/ss_7_5554_0.4877185673519882.pth"
        # eval_save_path = "../../../WSS_Model_SS_0602_EVAL/2_DeepLabV3PlusResNet152_201_10_18_1_352_balance/ss_7_scales_{}".format(len(scales))

        # 6 other people: Resnet152
        # scales = (1.0,)
        # model_file_name = "../../../WSS_Model_SS_0602/2_DeepLabV3PlusResNet152_201_10_18_1_352_balance/ss_0_2221_0.30658016552565515.pth"
        # eval_save_path = "../../../WSS_Model_SS_0602_EVAL/2_DeepLabV3PlusResNet152_201_10_18_1_352_balance/ss_0_scales_{}".format(len(scales))
        # model_file_name = "../../../WSS_Model_SS_0602/2_DeepLabV3PlusResNet152_201_10_18_1_352_balance/ss_0_1110_0.16993653364544287.pth"
        # eval_save_path = "../../../WSS_Model_SS_0602_EVAL/2_DeepLabV3PlusResNet152_201_10_18_1_352_balance/ss_00_scales_{}".format(len(scales))

        # 7 Resnet152 output_stride = 8
        # scales = (1.0, 0.75, 0.5, 1.25, 1.5, 1.75, 2.0)
        # model_file_name = "../../../WSS_Model_SS_0602/3_DeepLabV3PlusResNet152_201_10_18_1_352_8_balance/ss_7_8887_0.49612685291103387.pth"
        # eval_save_path = "../../../WSS_Model_SS_0602_EVAL/3_DeepLabV3PlusResNet152_201_10_18_1_352_8_balance/ss_7_scales_{}_500".format(len(scales))

        # 8 other people: Resnet152 output_stride = 8
        # scales = (1.0,)
        # model_file_name = "../../../WSS_Model_SS_0602/3_DeepLabV3PlusResNet152_201_10_18_1_352_8_balance/ss_0_2221_0.31225540973527816.pth"
        # eval_save_path = "../../../WSS_Model_SS_0602_EVAL/3_DeepLabV3PlusResNet152_201_10_18_1_352_8_balance/ss_0_2221_scales_{}_500".format(len(scales))

        # 9 Resnet 50 or 152 output_stride = 8 train
        # scales = (1.0, )
        # model_file_name = "../../../WSS_Model_SS_0602/3_DeepLabV3PlusResNet152_201_10_18_1_352_8_balance/ss_7_8887_0.49612685291103387.pth"
        # eval_save_path = "../../../WSS_Model_SS_0602_EVAL/3_DeepLabV3PlusResNet152_201_10_18_1_352_8_balance/ss_7_train_500"

        # 10 self-training
        scales = (1.0, 0.75, 0.5, 1.25, 1.5, 1.75, 2.0)
        model_file_name = "../../../WSS_Model_SS_0602/3_self_training_resume_DeepLabV3PlusResNet152_201_10_18_1_352_8_balance/ss_6_9998_0.5015402698782628.pth"
        eval_save_path = "../../../WSS_Model_SS_0602_EVAL/3_self_training_resume_DeepLabV3PlusResNet152_201_10_18_1_352_8_balance/ss_6_9998_scales_{}_500".format(len(scales))

        return scales, model_file_name, eval_save_path

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
output_stride = 8
../../../WSS_Model_SS_0602/3_DeepLabV3PlusResNet152_201_10_18_1_352_8_balance/ss_7_8887_0.49612685291103387.pth
Overall Acc: 0.835645
Mean Acc: 0.696309
FreqW Acc: 0.737874
Mean IoU: 0.496127

ss_7_scales_1_0 max_size_inference = 0
Overall Acc: 0.855468
Mean Acc: 0.699628
FreqW Acc: 0.767776
Mean IoU: 0.491615
ss_7_scales_1_500 max_size_inference = 500
Overall Acc: 0.855467
Mean Acc: 0.700577
FreqW Acc: 0.767827
Mean IoU: 0.492327
ss_7_scales_7_0 max_size_inference = 0
2021-06-07 13:38:23 
Overall Acc: 0.865002
Mean Acc: 0.680683
FreqW Acc: 0.777233
Mean IoU: 0.506402
ss_7_scales_7_500 max_size_inference = 500
2021-06-07 13:41:24 
Overall Acc: 0.863998
Mean Acc: 0.682085
FreqW Acc: 0.776430
Mean IoU: 0.506957
ss_7_scales_4_500 max_size_inference = 500
2021-06-08 16:39:10 
Overall Acc: 0.860604
Mean Acc: 0.698268
FreqW Acc: 0.773550
Mean IoU: 0.503041
ss_7_scales_6_500 max_size_inference = 500
2021-06-08 18:51:16 
Overall Acc: 0.865059
Mean Acc: 0.697353
FreqW Acc: 0.777911
Mean IoU: 0.508361


../../../WSS_Model_SS_0602/3_self_training_resume_DeepLabV3PlusResNet152_201_10_18_1_352_8_balance/ss_6_9998_0.5015402698782628.pth
ss_7_scales_6_500 max_size_inference = 500
2021-06-09 22:24:36 
Overall Acc: 0.867581
Mean Acc: 0.690415
FreqW Acc: 0.781894
Mean IoU: 0.510991

"""


if __name__ == '__main__':
    config = Config()
    train(config=config)
    pass
