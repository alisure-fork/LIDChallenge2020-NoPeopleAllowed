import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
import scipy.io as scio
from skimage.io import imread
from torchvision import transforms
from alisuretool.Tools import Tools
from torch.utils.data import Dataset


class UnNormalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        pass

    def __call__(self, tensor):
        new_tensor = tensor.clone()
        for t, m, s in zip(new_tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        np_data = np.asarray(new_tensor.permute(1, 2, 0).numpy() * 255, dtype=np.uint8)
        return Image.fromarray(np_data)

    pass


class DataUtil(object):

    @staticmethod
    def get_data_info(data_root="/media/ubuntu/4T/ALISURE/Data/L2ID/data"):
        image_info_path = os.path.join(data_root, "deal", "image_info_list.pkl")
        image_info_list = Tools.read_from_pkl(image_info_path)
        return image_info_list

    @staticmethod
    def get_class_name(mat_file="/media/ubuntu/4T/ALISURE/Data/L2ID/data/meta_det.mat"):
        data = scio.loadmat(mat_file)

        label_info_dict = {}
        name_to_label_id = {}
        for item in data['synsets'][0]:
            label_id = item[0][0][0]
            name = item[1][0]
            cat_name = item[2][0]
            name_to_label_id[name] = label_id
            label_info_dict[label_id] = {"name": name, "cat_name": cat_name}
            pass
        return name_to_label_id, label_info_dict

    pass


class MyTransform(object):

    @staticmethod
    def normalize():
        return transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    @classmethod
    def transform_un_normalize(cls):
        return transforms.Compose([UnNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    @classmethod
    def transform_train_cam(cls, image_size=352):
        transform_train = transforms.Compose([transforms.RandomResizedCrop(size=image_size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(), cls.normalize()])
        transform_test = transforms.Compose([transforms.Resize(size=256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(), cls.normalize()])
        # transform_test = transforms.Compose([transforms.Resize(size=image_size),
        #                                      transforms.ToTensor(), cls.normalize()])
        return transform_train, transform_test

    pass


class ImageNetMLC(Dataset):
    """ImageNet dataset for multi-label classification"""

    def __init__(self, images_list, transform, num_classes):
        self.images_list = images_list
        self.transform = transform
        self.num_classes = num_classes
        pass

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_label, image_path = self.images_list[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label_encoded = torch.zeros(self.num_classes, dtype=torch.float32)
        label_encoded[np.array(image_label, dtype=np.int) - 1] = 1
        return image, label_encoded

    pass


class DatasetUtil(object):

    dataset_type_mlc_train = "mlc_train"
    dataset_type_mlc_val = "mlc_val"

    @classmethod
    def get_dataset_by_type(cls, dataset_type, image_size, data_root="/media/ubuntu/4T/ALISURE/Data/L2ID/data"):
        # data_info = DataUtil.get_data_info(data_root=data_root)[::20]
        data_info = DataUtil.get_data_info(data_root=data_root)
        label_image_path = [[[one_object[2] for one_object in one_data["object"]],
                             one_data["image_path"]] for one_data in data_info]

        if dataset_type == cls.dataset_type_mlc_train:
            return cls.get_imagenet_mlc_train(label_image_path, image_size=image_size)
        elif dataset_type == cls.dataset_type_mlc_val:
            return cls.get_imagenet_mlc_val(label_image_path, image_size=image_size)
        else:
            raise Exception("....")
        pass

    @staticmethod
    def get_imagenet_mlc_train(label_image_path, image_size):
        transform_train, transform_test = MyTransform.transform_train_cam(image_size=image_size)
        imagenet_mlc = ImageNetMLC(images_list=label_image_path, num_classes=200, transform=transform_train)
        return imagenet_mlc

    @staticmethod
    def get_imagenet_mlc_val(label_image_path, image_size):
        transform_train, transform_test = MyTransform.transform_train_cam(image_size=image_size)
        imagenet_mlc = ImageNetMLC(images_list=label_image_path, num_classes=200, transform=transform_test)
        return imagenet_mlc

    pass


if __name__ == '__main__':
    dataset_mlc_train = DatasetUtil.get_dataset_by_type(dataset_type=DatasetUtil.dataset_type_mlc_train, image_size=352)
    dataset_mlc_val = DatasetUtil.get_dataset_by_type(dataset_type=DatasetUtil.dataset_type_mlc_val, image_size=352)
    dataset_mlc_train.__getitem__(0)
    pass
