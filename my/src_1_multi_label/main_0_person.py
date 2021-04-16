import os
import sys
import glob
import torch
import random
import shutil
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
sys.path.append("../../")
from util_data import DatasetUtil
from util_network import CAMNet, ClassNet
from torch.utils.data import DataLoader, Dataset


class PersonRunner(object):

    def __init__(self, config):
        self.config = config

        self.dataset_person_train, self.dataset_person_val, self.dataset_person_val_all = DatasetUtil.get_dataset_by_type(
            DatasetUtil.dataset_type_person, image_size=self.config.person_size,
            data_root=self.config.data_root_path, sampling=self.config.sampling)
        self.data_loader_person_train = DataLoader(self.dataset_person_train, self.config.person_batch_size,
                                                   shuffle=False, num_workers=16)
        self.data_loader_person_val = DataLoader(self.dataset_person_val, self.config.person_batch_size,
                                                shuffle=False, num_workers=16)
        self.data_loader_person_val_all = DataLoader(self.dataset_person_val_all, self.config.person_batch_size,
                                                     shuffle=False, num_workers=16)

        # Model
        self.net = self.config.Net(num_classes=self.config.person_num_classes)
        self.net = nn.DataParallel(self.net).cuda()
        cudnn.benchmark = True

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.config.person_lr,
                                    betas=(0.9, 0.999), weight_decay=0)

        # Loss
        self.ce_loss = nn.CrossEntropyLoss().cuda()
        pass

    # 1.训练Person
    def train_person(self, start_epoch=0, model_file_name=None):

        if model_file_name is not None:
            Tools.print("Load model form {}".format(model_file_name), txt_path=self.config.person_save_result_txt)
            self.load_model(model_file_name)
            pass

        Tools.print()
        self.eval_person(epoch=0)
        # images_list, count = self.change_person()
        # self.dataset_person_train.images_list = images_list
        # self.dataset_person_val.images_list = images_list
        # self.eval_person(epoch=0)
        # Tools.print("Change label num={}".format(count), txt_path=self.config.person_save_result_txt)

        for epoch in range(start_epoch, self.config.person_epoch_num):
            Tools.print()
            self._adjust_learning_rate(self.optimizer, epoch, lr=self.config.person_lr,
                                       change_epoch=self.config.person_change_epoch)
            Tools.print('Epoch:{:03d}, lr={:.6f}'.format(epoch, self.optimizer.param_groups[0]['lr']),
                        txt_path=self.config.person_save_result_txt)

            ###########################################################################
            # 1 训练模型
            all_loss = 0.0
            self.net.train()
            self.dataset_person_train.reset()
            for i, (inputs, labels) in tqdm(enumerate(self.data_loader_person_train),
                                            total=len(self.data_loader_person_train)):
                inputs, labels = inputs.type(torch.FloatTensor).cuda(), labels.cuda()
                self.optimizer.zero_grad()

                result = self.net(inputs)
                loss = self.ce_loss(result, labels)

                loss.backward()
                self.optimizer.step()

                all_loss += loss.item()
                pass
            ###########################################################################

            Tools.print("[E:{:3d}/{:3d}] person loss:{:.4f}".format(
                epoch, self.config.person_epoch_num, all_loss/len(self.data_loader_person_train)),
                txt_path=self.config.person_save_result_txt)

            ###########################################################################
            # 2 保存模型
            if epoch % self.config.person_save_epoch_freq == 0:
                Tools.print()
                save_file_name = Tools.new_dir(os.path.join(
                    self.config.person_model_dir, "person_{}.pth".format(epoch)))
                torch.save(self.net.state_dict(), save_file_name)
                Tools.print("Save Model to {}".format(save_file_name), txt_path=self.config.person_save_result_txt)
                pass
            ###########################################################################

            ###########################################################################
            # 3 改变标签
            if epoch % self.config.person_change_epoch_freq == 0:
                Tools.print()
                self.eval_person(epoch=epoch)
                Tools.print("Change label...", txt_path=self.config.person_save_result_txt)
                images_list, count = self.change_person()
                self.dataset_person_train.images_list = images_list
                self.dataset_person_val.images_list = images_list
                Tools.print("Change label num={}".format(count), txt_path=self.config.person_save_result_txt)
            ###########################################################################

            ###########################################################################
            # 4 评估模型
            if epoch % self.config.person_eval_epoch_freq == 0:
                Tools.print()
                self.eval_person(epoch=epoch)
                pass
            ###########################################################################

            pass

        # Final Save
        Tools.print()
        save_file_name = Tools.new_dir(os.path.join(
            self.config.person_model_dir, "person_final_{}.pth".format(self.config.person_epoch_num)))
        torch.save(self.net.state_dict(), save_file_name)
        Tools.print("Save Model to {}".format(save_file_name), txt_path=self.config.person_save_result_txt)
        Tools.print()

        self.eval_person(epoch=self.config.person_epoch_num)
        pass

    def change_person(self):
        self.net.eval()
        with torch.no_grad():
            count = 0
            images_list = [[o for o in one] for one in self.data_loader_person_val_all.dataset.images_list]
            for i, (inputs, labels, indexes) in tqdm(enumerate(self.data_loader_person_val_all),
                                                     total=len(self.data_loader_person_val_all)):
                logits = self.net(inputs.float().cuda()).detach().cpu()
                logits = torch.softmax(logits, dim=1)
                net_out = torch.argmax(logits, dim=1).numpy()
                labels = labels.numpy()

                for out_one, logit_one, label_one, index_one in zip(net_out, logits, labels, indexes):
                    if label_one != 1 and out_one == 1 and logit_one[1] > 0.90:
                        images_list[index_one][0] = 1
                        count += 1
                        pass
                    pass
                pass
            pass
        return images_list, count

    def save_person_result(self, result_path):
        self.net.eval()
        with torch.no_grad():
            count = 0
            images_list = [[o for o in one] for one in self.data_loader_person_val_all.dataset.images_list]
            for i, (inputs, labels, indexes) in tqdm(enumerate(self.data_loader_person_val_all),
                                                     total=len(self.data_loader_person_val_all)):
                logits = self.net(inputs.float().cuda()).detach().cpu()
                logits = torch.softmax(logits, dim=1)
                net_out = torch.argmax(logits, dim=1).numpy()
                labels = labels.numpy()

                for out_one, logit_one, label_one, index_one in zip(net_out, logits, labels, indexes):
                    if label_one != 1 and out_one == 1 and logit_one[1] > 0.90:
                        images_list[index_one][0] = 1
                        count += 1
                        pass
                    pass
                pass
            pass
        Tools.write_to_pkl(result_path, _data=images_list)
        Tools.print("change num = {}".format(count))
        pass

    def eval_person(self, epoch=0, model_file_name=None):
        if model_file_name is not None:
            Tools.print("Load model form {}".format(model_file_name), txt_path=self.config.person_save_result_txt)
            self.load_model(model_file_name)
            pass

        acc = 0.0
        self.net.eval()
        with torch.no_grad():
            self.dataset_person_val.reset()
            for i, (inputs, labels) in tqdm(enumerate(self.data_loader_person_val), total=len(self.data_loader_person_val)):
                inputs = inputs.type(torch.FloatTensor).cuda()
                logits = self.net(inputs)
                labels = labels.numpy()
                net_out = torch.argmax(logits, dim=1).detach().cpu().numpy()

                now_acc = np.sum(net_out == labels) / len(labels)
                acc += now_acc
                pass
            pass

        acc = acc / len(self.data_loader_person_val)
        Tools.print("[E:{:3d}] val acc:{:.4f}".format(epoch, acc), txt_path=self.config.person_save_result_txt)
        return acc

    def load_model(self, model_file_name):
        Tools.print("Load model form {}".format(model_file_name), txt_path=self.config.person_save_result_txt)
        checkpoint = torch.load(model_file_name)
        self.net.load_state_dict(checkpoint, strict=True)
        Tools.print("Restore from {}".format(model_file_name), txt_path=self.config.person_save_result_txt)
        pass

    @staticmethod
    def _adjust_learning_rate(optimizer, epoch, lr, change_epoch=30):
        if epoch > change_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * 0.1
                pass
        pass

    pass


def train(config):
    person_runner = PersonRunner(config=config)

    # 训练MIC
    if config.has_train_person:
        person_runner.train_person(start_epoch=0, model_file_name=None)
        # person_runner.eval_person(epoch=0, model_file_name=os.path.join(config.person_model_dir, "person_5.pth"))
        pass

    if config.has_change_person:
        # person_runner.load_model(model_file_name=os.path.join(
        #     config.person_model_dir, "person_final_{}.pth".format(config.person_epoch_num)))
        person_runner.load_model(model_file_name="../../../WSS_Model_Person/1_ClassNet_2_50_144_5_224/person_45.pth")
        person_runner.eval_person(epoch=0)

        result_path = os.path.join(config.person_model_dir, "person2.pkl")
        person_runner.save_person_result(result_path=result_path)
        shutil.copy(result_path, os.path.join(config.data_root_path, "deal"))
        pass

    pass


class Config(object):

    def __init__(self):
        # self.gpu_id = "0, 1, 2, 3"
        self.gpu_id = "1, 2, 3"
        # self.gpu_id = "1, 2"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        # 流程控制
        self.has_train_person = False  # 是否训练person
        self.has_change_person = True

        self.person_num_classes = 2
        self.person_epoch_num = 50
        self.person_change_epoch = 30
        self.person_batch_size = 48 * len(self.gpu_id.split(","))
        self.person_lr = 0.0001
        self.person_save_epoch_freq = 5
        self.person_change_epoch_freq = 5
        self.person_eval_epoch_freq = 5

        # 图像大小
        self.person_size = 224
        self.sampling = False

        # 网络
        self.Net = ClassNet

        self.data_root_path = self.get_data_root_path()

        run_name = "1"
        self.model_name = "{}_{}_{}_{}_{}_{}_{}".format(
            run_name, "ClassNet", self.person_num_classes, self.person_epoch_num,
            self.person_batch_size, self.person_save_epoch_freq, self.person_size)
        Tools.print(self.model_name)

        self.person_model_dir = "../../../WSS_Model_Person/{}".format(self.model_name)
        self.person_save_result_txt = Tools.new_dir("{}/result.txt".format(self.person_model_dir))
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
val acc:0.9403 ../../../WSS_Model_Person/1_ClassNet_2_15_192_2_224/person_final_15.pth
val acc:0.9858 ../../../WSS_Model_Person/1_ClassNet_2_50_192_5_224/person_final_50.pth
val acc:0.9887 ../../../WSS_Model_Person/1_ClassNet_2_50_144_5_224/person_45.pth 25687
"""


if __name__ == '__main__':
    config = Config()
    train(config=config)
    pass
