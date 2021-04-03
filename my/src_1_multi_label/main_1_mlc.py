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
from util_network import CAMNet
from util_data import DatasetUtil
from alisuretool.Tools import Tools
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset


class CAMRunner(object):

    def __init__(self, config):
        self.config = config

        self.dataset_mlc_train = DatasetUtil.get_dataset_by_type(
            DatasetUtil.dataset_type_mlc_train, image_size=self.config.mlc_size, data_root=self.config.data_root_path)
        self.dataset_mlc_val = DatasetUtil.get_dataset_by_type(
            DatasetUtil.dataset_type_mlc_val, image_size=self.config.mlc_size, data_root=self.config.data_root_path)
        self.data_loader_mlc_train = DataLoader(self.dataset_mlc_train, self.config.mlc_batch_size,
                                                shuffle=True, num_workers=16)
        self.data_loader_mlc_val = DataLoader(self.dataset_mlc_val, self.config.mlc_batch_size,
                                                shuffle=False, num_workers=16)

        # Model
        self.net = self.config.Net(num_classes=self.config.mlc_num_classes)
        self.net = nn.DataParallel(self.net).cuda()
        cudnn.benchmark = True

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.config.mlc_lr, betas=(0.9, 0.999), weight_decay=0)

        # Loss
        self.bce_loss = nn.BCEWithLogitsLoss().cuda()
        pass

    # 1.训练MIC
    def train_mlc(self, start_epoch=0, model_file_name=None):

        if model_file_name is not None:
            Tools.print("Load model form {}".format(model_file_name), txt_path=self.config.mlc_save_result_txt)
            self.load_model(model_file_name)
            pass

        self.eval_mlc(epoch=0)

        for epoch in range(start_epoch, self.config.mlc_epoch_num):
            Tools.print()
            self._adjust_learning_rate(self.optimizer, epoch, lr=self.config.mlc_lr,
                                       change_epoch=self.config.mlc_change_epoch)
            Tools.print('Epoch:{:03d}, lr={:.5f}'.format(epoch, self.optimizer.param_groups[0]['lr']),
                        txt_path=self.config.mlc_save_result_txt)

            ###########################################################################
            # 1 训练模型
            all_loss = 0.0
            self.net.train()
            for i, (inputs, labels) in tqdm(enumerate(self.data_loader_mlc_train),
                                            total=len(self.data_loader_mlc_train)):
                inputs, labels = inputs.type(torch.FloatTensor).cuda(), labels.cuda()
                self.optimizer.zero_grad()

                result = self.net(inputs)
                loss = self.bce_loss(result, labels)
                ######################################################################################################

                loss.backward()
                self.optimizer.step()

                all_loss += loss.item()
                pass

            Tools.print("[E:{:3d}/{:3d}] mlc loss:{:.3f}".format(
                epoch, self.config.mlc_epoch_num, all_loss/len(self.data_loader_mlc_train)),
                txt_path=self.config.mlc_save_result_txt)

            ###########################################################################
            # 2 保存模型
            if epoch % self.config.mlc_save_epoch_freq == 0:
                Tools.print()
                save_file_name = Tools.new_dir(os.path.join(
                    self.config.mlc_model_dir, "mlc_{}.pth".format(epoch)))
                torch.save(self.net.state_dict(), save_file_name)
                Tools.print("Save Model to {}".format(save_file_name), txt_path=self.config.mlc_save_result_txt)
                Tools.print()
                pass
            ###########################################################################

            ###########################################################################
            # 3 评估模型
            if epoch % self.config.mlc_eval_epoch_freq == 0:
                self.eval_mlc(epoch=epoch)
                pass
            ###########################################################################

            pass

        # Final Save
        Tools.print()
        save_file_name = Tools.new_dir(os.path.join(
            self.config.mlc_model_dir, "mlc_final_{}.pth".format(self.config.mlc_epoch_num)))
        torch.save(self.net.state_dict(), save_file_name)
        Tools.print("Save Model to {}".format(save_file_name), txt_path=self.config.mlc_save_result_txt)
        Tools.print()

        self.eval_mlc(epoch=self.config.mlc_epoch_num)
        pass

    def eval_mlc(self, epoch=0, model_file_name=None):
        if model_file_name is not None:
            Tools.print("Load model form {}".format(model_file_name), txt_path=self.config.mlc_save_result_txt)
            self.load_model(model_file_name)
            pass


        """
        test_pre_ts = metrics.precision_score(y_true=labels, y_pred=net_out>0.5, average='micro')
        test_rec_ts = metrics.recall_score(y_true=labels, y_pred=net_out>0.5, average='micro')
        test_F1_ts = metrics.f1_score(y_true=labels, y_pred=net_out>0.5, average='micro')
        test_auc = metrics.roc_auc_score(y_true=labels, y_score=net_out, average='micro')
        test_prc = metrics.average_precision_score(y_true=labels, y_score=net_out, average="micro")
        test_hamming = metrics.hamming_loss(labels, net_out>0.5)
        
        def mcc(net_out, labels):
            thresholds = np.arange(0.1, 0.9, 0.1)
            accuracies = []
            for i in range(net_out.shape[0]):
                acc = np.array([metrics.matthews_corrcoef(labels[i], net_out[i] > threshold) for threshold in thresholds])
                accuracies.append(acc)
                pass
            return np.mean(accuracies, axis=0)
        """


        acc, mae, f1 = 0.0, 0.0, 0.0
        self.net.eval()
        with torch.no_grad():
            for i, (inputs, labels) in tqdm(enumerate(self.data_loader_mlc_val), total=len(self.data_loader_mlc_val)):
                inputs = inputs.type(torch.FloatTensor).cuda()
                logits = self.net(inputs)
                labels = labels.numpy()
                net_out = torch.sigmoid(logits).detach().cpu().numpy()

                one, zero = labels == 1, labels != 1
                now_mae = (np.abs(net_out[one] - labels[one]).mean() + np.abs(net_out[zero] - labels[zero]).mean()) / 2
                now_f1 = metrics.f1_score(y_true=labels, y_pred=net_out > 0.5, average='micro')
                now_acc = self._acc(net_out=net_out, labels=labels)

                f1 += now_f1
                mae += now_mae
                acc += now_acc
                pass
            pass

        mae = mae / len(self.data_loader_mlc_val)
        f1 = f1 / len(self.data_loader_mlc_val)
        acc = acc / len(self.data_loader_mlc_val)
        Tools.print("[E:{:3d}] val mae:{:.4f} f1:{:.4f} acc:{:.4f}".format(epoch, mae, f1, acc),
                    txt_path=self.config.mlc_save_result_txt)
        return [mae, f1, acc]

    def load_model(self, model_file_name):
        Tools.print("Load model form {}".format(model_file_name), txt_path=self.config.mlc_save_result_txt)
        checkpoint = torch.load(model_file_name)
        self.net.load_state_dict(checkpoint, strict=True)
        Tools.print("Restore from {}".format(model_file_name), txt_path=self.config.mlc_save_result_txt)
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
    cam_runner = CAMRunner(config=config)

    # 训练MIC
    if config.has_train_mlc:
        cam_runner.train_mlc(start_epoch=0, model_file_name=None)
        # cam_runner.eval_mlc(epoch=0, model_file_name=os.path.join(config.mlc_model_dir, "mlc_5.pth"))
        pass

    pass


class Config(object):

    def __init__(self):
        self.gpu_id = "0, 1, 2, 3"
        # self.gpu_id = "0"
        # self.gpu_id = "0, 1"
        # self.gpu_id = "2, 3"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        # 流程控制
        self.has_train_mlc = True  # 是否训练MLC

        self.mlc_num_classes = 200
        self.mlc_epoch_num = 20
        self.mlc_change_epoch = 10
        self.mlc_batch_size = 32 * len(self.gpu_id.split(","))
        self.mlc_lr = 0.00001
        self.mlc_save_epoch_freq = 2
        self.mlc_eval_epoch_freq = 2

        # 图像大小
        self.mlc_size = 224

        # 网络
        self.Net = CAMNet

        self.data_root_path = self.get_data_root_path()

        run_name = "1"
        self.model_name = "{}_{}_{}_{}_{}_{}_{}".format(
            run_name, "CAMNet", self.mlc_num_classes, self.mlc_epoch_num,
            self.mlc_batch_size, self.mlc_save_epoch_freq, self.mlc_size)
        Tools.print(self.model_name)

        self.mlc_model_dir = "../../../WSS_Model/{}".format(self.model_name)
        self.mlc_save_result_txt = Tools.new_dir("{}/result.txt".format(self.mlc_model_dir))
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
1/20 val mae:0.0775 f1:0.9067 ../../../WSS_Model/demo_CAMNet_200_60_128_5_224/mlc_final_60.pth
1/1  val mae:0.1017 f1:0.8701 ../../../WSS_Model/1_CAMNet_200_60_128_5_256/mlc_40.pth
"""


if __name__ == '__main__':
    config = Config()
    train(config=config)
    pass
