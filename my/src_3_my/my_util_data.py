import os
import cv2
import json
import torch
import random
import numbers
import torchvision
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image
import scipy.io as scio
from skimage.io import imread
from alisuretool.Tools import Tools
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision import transforms as vision_transforms


#######################################################################################################################


class Lambda(object):

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd
        pass

    def __call__(self, img):
        return self.lambd(img)

    pass


class ExtCompose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, lbl):
        for t in self.transforms:
            img, lbl = t(img, lbl)
        return img, lbl

    pass


class ExtCenterCrop(object):

    def __init__(self, size):
        self.size = (int(size), int(size)) if isinstance(size, numbers.Number) else size
        pass

    def __call__(self, img, lbl):
        return F.center_crop(img, self.size), F.center_crop(lbl, self.size)

    pass


class ExtRandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        if random.random() < self.p:
            return F.hflip(img), F.hflip(lbl)
        return img, lbl

    pass


class ExtToTensor(object):

    def __init__(self, normalize=True, target_type='uint8'):
        self.normalize = normalize
        self.target_type = target_type
        pass

    def __call__(self, pic, lbl):
        if self.normalize:
            return F.to_tensor(pic), torch.from_numpy(np.array(lbl, dtype=self.target_type))
        else:
            return torch.from_numpy(np.array(pic, dtype=np.float32).transpose(2, 0, 1)), torch.from_numpy(
                np.array(lbl, dtype=self.target_type))
        pass

    pass


class ExtNormalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        pass

    def __call__(self, tensor, lbl):
        return F.normalize(tensor, self.mean, self.std), lbl

    pass


class ExtRandomCrop(object):

    def __init__(self, size, padding=0, pad_if_needed=False):
        self.size = (int(size), int(size)) if isinstance(size, numbers.Number) else size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        pass

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, lbl):
        assert img.size == lbl.size, 'size of img and lbl should be the same. %s, %s' % (img.size, lbl.size)
        if self.padding > 0:
            img = F.pad(img, self.padding)
            lbl = F.pad(lbl, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2))
            lbl = F.pad(lbl, padding=int((1 + self.size[1] - lbl.size[0]) / 2))

        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2))
            lbl = F.pad(lbl, padding=int((1 + self.size[0] - lbl.size[1]) / 2))

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(lbl, i, j, h, w)

    pass


class ExtResize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or len(size) == 2
        self.size = size
        self.interpolation = interpolation
        pass

    def __call__(self, img, lbl):
        return F.resize(img, self.size, self.interpolation), F.resize(lbl, self.size, Image.NEAREST)

    pass


class ExtColorJitter(object):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
        pass

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        transforms = []
        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))
        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))
        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))
        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))
        random.shuffle(transforms)
        transform = vision_transforms.Compose(transforms)
        return transform

    def __call__(self, img, lbl):
        transform = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        return transform(img), lbl

    pass


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


#######################################################################################################################


class DataUtil(object):

    @staticmethod
    def get_voc_info(data_root=None, split="train_aug"):
        data_root = data_root if data_root is not None else "/media/ubuntu/4T/ALISURE/Data/SS/voc"
        split_info_path = os.path.join(data_root, "VOCdevkit/VOC2012/ImageSets/Segmentation")
        image_path = os.path.join(data_root, "VOCdevkit/VOC2012/JPEGImages")
        mask_path = os.path.join(data_root, "VOCdevkit/VOC2012/SegmentationClass")

        image_info_list = []

        if split == "train":
            with open(os.path.join(split_info_path, "train.txt"), "r") as f:
                file_names = [x.strip() for x in f.readlines()]
                images = [os.path.join(image_path, x + ".jpg") for x in file_names]
                masks = [os.path.join(mask_path, x + ".png") for x in file_names]
                image_info_list =[{"image_path": image, "label_path": mask} for image, mask in zip(images, masks)]
                pass
            pass

        if split == "train_aug":
            mask_aug_path = os.path.join(data_root, "VOCdevkit/VOC2012/SegmentationClassAug")
            with open(os.path.join(split_info_path, "train_aug.txt"), "r") as f:
                file_names = [x.strip() for x in f.readlines()]
                images = [os.path.join(image_path, x + ".jpg") for x in file_names]
                masks = [os.path.join(mask_aug_path, x + ".png") for x in file_names]
                image_info_list =[{"image_path": image, "label_path": mask} for image, mask in zip(images, masks)]
                pass
            pass

        if split == "train_val":
            with open(os.path.join(split_info_path, "trainval.txt"), "r") as f:
                file_names = [x.strip() for x in f.readlines()]
                images = [os.path.join(image_path, x + ".jpg") for x in file_names]
                masks = [os.path.join(mask_path, x + ".png") for x in file_names]
                image_info_list =[{"image_path": image, "label_path": mask} for image, mask in zip(images, masks)]
                pass
            pass

        if split == "val":
            with open(os.path.join(split_info_path, "val.txt"), "r") as f:
                file_names = [x.strip() for x in f.readlines()]
                images = [os.path.join(image_path, x + ".jpg") for x in file_names]
                masks = [os.path.join(mask_path, x + ".png") for x in file_names]
                image_info_list =[{"image_path": image, "label_path": mask} for image, mask in zip(images, masks)]
                pass
            pass

        return image_info_list

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

    @staticmethod
    def one_hot(image_label, num_classes):
        label_encoded = torch.zeros(num_classes, dtype=torch.float32)
        label_encoded[np.array(image_label, dtype=np.int)] = 1
        return label_encoded

    pass


class MyTransform(object):

    @staticmethod
    def normalize():
        return vision_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    @classmethod
    def transform_un_normalize(cls):
        return vision_transforms.Compose([UnNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    @classmethod
    def transform_train_class(cls, input_size=224):
        transform_train = ExtCompose([ExtResize(size=(input_size, input_size)),
                                      ExtRandomHorizontalFlip(),
                                      ExtToTensor(),
                                      ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_test = ExtCompose([ExtResize(size=(input_size, input_size)),
                                     ExtToTensor(),
                                     ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return transform_train, transform_test

    pass


class VOCSegmentation(Dataset):

    def __init__(self, images_list, transform, return_image_info=False, num_classes=20):
        self.transform = transform
        self.images_list = images_list
        self.return_image_info = return_image_info
        self.num_classes = num_classes
        pass

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        label_path, image_path = self.images_list[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(label_path)

        label = list(np.unique(np.asarray(mask)))
        if 0 in label:
            label.remove(0)
        if 255 in label:
            label.remove(255)
        label = DataUtil.one_hot(image_label=[one - 1 for one in label], num_classes=self.num_classes)

        image, mask = self.transform(image, mask)

        if self.return_image_info:
            return image, mask, label, image_path

        return image, mask, label

    pass


class VOCSegmentationDual(Dataset):

    def __init__(self, images_list, transform, return_image_info=False, num_classes=20):
        self.transform = transform
        self.return_image_info = return_image_info
        self.num_classes = num_classes
        self.images_list, self.label_dict = self.deal_data(images_list=images_list)
        pass

    def __len__(self):
        return len(self.images_list)

    @staticmethod
    def deal_data(images_list):
        all_label = []
        for png, jpg in tqdm(images_list, total=len(images_list)):
            mask = Image.open(png)
            label = list(np.unique(np.asarray(mask)))
            if 0 in label:
                label.remove(0)
            if 255 in label:
                label.remove(255)
            all_label.append([one - 1 for one in label])
            pass

        label_dict = {}
        for index, one_label in enumerate(all_label):
            for one in one_label:
                if one not in label_dict:
                    label_dict[one] = []
                label_dict[one].append(index)
            pass

        result_images_list = [[one[0], one[1], two] for one, two in zip(images_list, all_label)]
        return result_images_list, label_dict

    def __getitem__(self, idx):
        label_path, image_path, label_class = self.images_list[idx]
        pair_label_class = np.random.choice(label_class)
        idx_2 = np.random.choice(self.label_dict[pair_label_class])
        label_path_2, image_path_2, label_class_2 = self.images_list[idx_2]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(label_path)
        image, mask = self.transform(image, mask)
        label = DataUtil.one_hot(image_label=label_class, num_classes=self.num_classes)

        image_2 = Image.open(image_path_2).convert("RGB")
        mask_2 = Image.open(label_path_2)
        image_2, mask_2 = self.transform(image_2, mask_2)
        label_2 = DataUtil.one_hot(image_label=label_class_2, num_classes=self.num_classes)

        if self.return_image_info:
            return pair_label_class, [image, image_2], [mask, mask_2], [label, label_2], [image_path, image_path_2]

        return pair_label_class, [image, image_2], [mask, mask_2], [label, label_2]

    pass


class DatasetUtil(object):

    dataset_type_voc_train = "ss_voc_train"
    dataset_type_voc_val = "ss_voc_val"
    dataset_type_voc_train_dual = "ss_voc_train_dual"

    @classmethod
    def get_dataset_by_type(cls, dataset_type, input_size, data_root=None, return_image_info=False, sampling=False):
        ################################################################################################################
        if dataset_type == cls.dataset_type_voc_train:
            data_info = DataUtil.get_voc_info(data_root=data_root, split="train_aug")
            data_info = data_info[::20] if sampling else data_info

            label_image_path = [[one_data["label_path"], one_data["image_path"]] for one_data in data_info]

            transform_train, _ = MyTransform.transform_train_class(input_size=input_size)
            return VOCSegmentation(label_image_path, transform=transform_train, return_image_info=return_image_info)
        elif dataset_type == cls.dataset_type_voc_train_dual:
            data_info = DataUtil.get_voc_info(data_root=data_root, split="train_aug")
            data_info = data_info[::20] if sampling else data_info

            label_image_path = [[one_data["label_path"], one_data["image_path"]] for one_data in data_info]

            transform_train, _ = MyTransform.transform_train_class(input_size=input_size)
            return VOCSegmentationDual(label_image_path, transform=transform_train, return_image_info=return_image_info)
        elif dataset_type == cls.dataset_type_voc_val:
            data_info = DataUtil.get_voc_info(data_root=data_root, split="val")
            data_info = data_info[::20] if sampling else data_info

            label_image_path = [[one_data["label_path"], one_data["image_path"]] for one_data in data_info]

            _, transform_test = MyTransform.transform_train_class(input_size=input_size)
            return VOCSegmentation(label_image_path, transform=transform_test, return_image_info=return_image_info)
        ################################################################################################################
        else:
            raise Exception("....")
        pass

    pass


if __name__ == '__main__':
    # dataset_train = DatasetUtil.get_dataset_by_type(dataset_type=DatasetUtil.dataset_type_voc_train, input_size=500)
    # dataset_val = DatasetUtil.get_dataset_by_type(dataset_type=DatasetUtil.dataset_type_voc_val, input_size=500)
    # dataset_train.__getitem__(0)

    data_root = '/mnt/4T/Data/data/SS/voc'
    dataset_train = DatasetUtil.get_dataset_by_type(dataset_type=DatasetUtil.dataset_type_voc_train_dual,
                                                    data_root=data_root, input_size=500)
    dataset_train.__getitem__(0)
    pass
