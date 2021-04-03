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

    @staticmethod
    def get_palette():
        # palette = np.load('../src_0_deal_data/palette.npy').tolist()

        palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64,
                   0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0,
                   64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128, 64,
                   64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0, 64, 64, 128, 192, 64, 128, 64, 192, 128, 192, 192, 128,
                   0, 0, 64, 128, 0, 64, 0, 128, 64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192,
                   64, 0, 64, 192, 0, 64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0, 192, 64, 128, 192, 192, 128,
                   192, 0, 64, 64, 128, 64, 64, 0, 192, 64, 128, 192, 64, 0, 64, 192, 128, 64, 192, 0, 192, 192, 128,
                   192, 192, 64, 64, 64, 192, 64, 64, 64, 192, 64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64, 192,
                   192, 192, 192, 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0, 128, 160, 0, 128, 32, 128,
                   128, 160, 128, 128, 96, 0, 0, 224, 0, 0, 96, 128, 0, 224, 128, 0, 96, 0, 128, 224, 0, 128, 96, 128,
                   128, 224, 128, 128, 32, 64, 0, 160, 64, 0, 32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32,
                   192, 128, 160, 192, 128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0, 96, 64, 128, 224, 64, 128,
                   96, 192, 128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32, 128, 64, 160, 128, 64, 32, 0, 192, 160, 0,
                   192, 32, 128, 192, 160, 128, 192, 96, 0, 64, 224, 0, 64, 96, 128, 64, 224, 128, 64, 96, 0, 192, 224,
                   0, 192, 96, 128, 192, 224, 128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64, 160, 192, 64, 32, 64, 192,
                   160, 64, 192, 32, 192, 192, 160, 192, 192, 96, 64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64, 96,
                   64, 192, 224, 64, 192, 96, 192, 192, 224, 192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0, 0,
                   32, 128, 128, 32, 128, 0, 160, 128, 128, 160, 128, 64, 32, 0, 192, 32, 0, 64, 160, 0, 192, 160, 0,
                   64, 32, 128, 192, 32, 128, 64, 160, 128, 192, 160, 128, 0, 96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0,
                   0, 96, 128, 128, 96, 128, 0, 224, 128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0, 192, 224, 0,
                   64, 96, 128, 192, 96, 128, 64, 224, 128, 192, 224, 128, 0, 32, 64, 128, 32, 64, 0, 160, 64, 128, 160,
                   64, 0, 32, 192, 128, 32, 192, 0, 160, 192, 128, 160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192,
                   160, 64, 64, 32, 192, 192, 32, 192, 64, 160, 192, 192, 160, 192, 0, 96, 64, 128, 96, 64, 0, 224, 64,
                   128, 224, 64, 0, 96, 192, 128, 96, 192, 0, 224, 192, 128, 224, 192, 64, 96, 64, 192, 96, 64, 64, 224,
                   64, 192, 224, 64, 64, 96, 192, 192, 96, 192, 64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32,
                   160, 0, 160, 160, 0, 32, 32, 128, 160, 32, 128, 32, 160, 128, 160, 160, 128, 96, 32, 0, 224, 32, 0,
                   96, 160, 0, 224, 160, 0, 96, 32, 128, 224, 32, 128, 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96,
                   0, 32, 224, 0, 160, 224, 0, 32, 96, 128, 160, 96, 128, 32, 224, 128, 160, 224, 128, 96, 96, 0, 224,
                   96, 0, 96, 224, 0, 224, 224, 0, 96, 96, 128, 224, 96, 128, 96, 224, 128, 224, 224, 128, 32, 32, 64,
                   160, 32, 64, 32, 160, 64, 160, 160, 64, 32, 32, 192, 160, 32, 192, 32, 160, 192, 160, 160, 192, 96,
                   32, 64, 224, 32, 64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224, 32, 192, 96, 160, 192, 224, 160,
                   192, 32, 96, 64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96, 192, 160, 96, 192, 32, 224, 192,
                   160, 224, 192, 96, 96, 64, 224, 96, 64, 96, 224, 64, 224, 224, 64, 96, 96, 192, 224, 96, 192, 96,
                   224, 192, 224, 224, 192]

        return palette

    @classmethod
    def gray_to_color(cls, np_label):
        im_color = Image.fromarray(np_label, "P")
        im_color.putpalette(cls.get_palette())
        return im_color

    pass


class MyTransform(object):

    @staticmethod
    def normalize():
        return transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    @classmethod
    def transform_un_normalize(cls):
        return transforms.Compose([UnNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    @classmethod
    def transform_train_cam(cls, image_size=224):
        transform_train = transforms.Compose([transforms.RandomResizedCrop(size=image_size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(), cls.normalize()])
        transform_test = transforms.Compose([transforms.Resize(size=256),
                                             transforms.CenterCrop(image_size),
                                             transforms.ToTensor(), cls.normalize()])
        return transform_train, transform_test

    @classmethod
    def transform_vis_cam(cls, image_size=256):
        transform_test = transforms.Compose([transforms.Resize(size=(image_size, image_size)),
                                             transforms.ToTensor(), cls.normalize()])
        return transform_test

    pass


class ImageNetMLC(Dataset):
    """ImageNet dataset for multi-label classification"""

    def __init__(self, images_list, transform, num_classes, return_image_info=False):
        self.images_list = images_list
        self.transform = transform
        self.num_classes = num_classes
        self.return_image_info = return_image_info
        pass

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_label, image_path = self.images_list[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label_encoded = torch.zeros(self.num_classes, dtype=torch.float32)
        label_encoded[np.array(image_label, dtype=np.int) - 1] = 1

        if self.return_image_info:
            return image, label_encoded, image_path
        return image, label_encoded

    pass


class DatasetUtil(object):

    dataset_type_mlc_train = "mlc_train"
    dataset_type_mlc_val = "mlc_val"
    dataset_type_vis_cam = "vis_cam"

    @classmethod
    def get_dataset_by_type(cls, dataset_type, image_size,
                            data_root="/media/ubuntu/4T/ALISURE/Data/L2ID/data", return_image_info=False):
        data_info = DataUtil.get_data_info(data_root=data_root)[::20]

        # data_info = DataUtil.get_data_info(data_root=data_root)
        label_image_path = [[[one_object[2] for one_object in one_data["object"]],
                             one_data["image_path"]] for one_data in data_info]

        if dataset_type == cls.dataset_type_mlc_train:
            return cls.get_imagenet_mlc_train(label_image_path, image_size=image_size,
                                              return_image_info=return_image_info)
        elif dataset_type == cls.dataset_type_mlc_val:
            return cls.get_imagenet_mlc_val(label_image_path, image_size=image_size,
                                            return_image_info=return_image_info)
        elif dataset_type == cls.dataset_type_vis_cam:
            return cls.get_imagenet_vis_cam(label_image_path, image_size=image_size,
                                            return_image_info=return_image_info)
        else:
            raise Exception("....")
        pass

    @staticmethod
    def get_imagenet_mlc_train(label_image_path, image_size, return_image_info):
        transform_train, transform_test = MyTransform.transform_train_cam(image_size=image_size)
        imagenet_mlc = ImageNetMLC(images_list=label_image_path, num_classes=200,
                                   transform=transform_train, return_image_info=return_image_info)
        return imagenet_mlc

    @staticmethod
    def get_imagenet_mlc_val(label_image_path, image_size, return_image_info):
        transform_train, transform_test = MyTransform.transform_train_cam(image_size=image_size)
        imagenet_mlc = ImageNetMLC(images_list=label_image_path, num_classes=200,
                                   transform=transform_test, return_image_info=return_image_info)
        return imagenet_mlc

    @staticmethod
    def get_imagenet_vis_cam(label_image_path, image_size, return_image_info):
        transform_test = MyTransform.transform_vis_cam(image_size=image_size)
        imagenet_cam = ImageNetMLC(images_list=label_image_path, num_classes=200,
                                   transform=transform_test, return_image_info=return_image_info)
        return imagenet_cam

    pass


if __name__ == '__main__':
    dataset_mlc_train = DatasetUtil.get_dataset_by_type(dataset_type=DatasetUtil.dataset_type_mlc_train, image_size=256)
    dataset_mlc_val = DatasetUtil.get_dataset_by_type(dataset_type=DatasetUtil.dataset_type_mlc_val, image_size=256)
    dataset_mlc_train.__getitem__(0)
    pass
