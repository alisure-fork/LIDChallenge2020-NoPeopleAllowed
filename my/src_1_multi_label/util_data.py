import os
import cv2
import json
import torch
import random
import numbers
import torchvision
import numpy as np
from glob import glob
from PIL import Image
import scipy.io as scio
from skimage.io import imread
from torchvision import transforms
from alisuretool.Tools import Tools
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


#######################################################################################################################


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


class ExtRandomScale(object):
    def __init__(self, scale_range, interpolation=Image.BILINEAR):
        self.scale_range = scale_range
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        assert img.size == lbl.size
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        target_size = (int(img.size[1] * scale), int(img.size[0] * scale))
        return F.resize(img, target_size, self.interpolation), F.resize(lbl, target_size, Image.NEAREST)

    pass


class ExtScale(object):

    def __init__(self, scale, interpolation=Image.BILINEAR):
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        assert img.size == lbl.size
        target_size = (int(img.size[1] * self.scale), int(img.size[0] * self.scale))  # (H, W)
        return F.resize(img, target_size, self.interpolation), F.resize(lbl, target_size, Image.NEAREST)

    pass


class ExtRandomRotation(object):

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        pass

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, img, lbl):
        angle = self.get_params(self.degrees)
        return F.rotate(img, angle, self.resample, self.expand, self.center), \
               F.rotate(lbl, angle, self.resample, self.expand, self.center)

    pass


class ExtRandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        if random.random() < self.p:
            return F.hflip(img), F.hflip(lbl)
        return img, lbl

    pass


class ExtRandomVerticalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        if random.random() < self.p:
            return F.vflip(img), F.vflip(lbl)
        return img, lbl

    pass


class ExtPad(object):

    def __init__(self, diviser=32):
        self.diviser = diviser
        pass

    def __call__(self, img, lbl):
        h, w = img.size
        ph = (h // 32 + 1) * 32 - h if h % 32 != 0 else 0
        pw = (w // 32 + 1) * 32 - w if w % 32 != 0 else 0
        im = F.pad(img, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        lbl = F.pad(lbl, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        return im, lbl


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
        transform = Compose(transforms)
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
    def get_data_info(data_root=None):
        data_root = data_root if data_root is not None else "/media/ubuntu/4T/ALISURE/Data/L2ID/data"
        image_info_path = os.path.join(data_root, "deal", "image_info_list.pkl")
        image_info_list = Tools.read_from_pkl(image_info_path)
        return image_info_list

    @staticmethod
    def get_voc_info(data_root=None, split="train_aug"):
        data_root = data_root if data_root is not None else "/media/ubuntu/4T/ALISURE/Data/SS/voc"
        split_info_path = os.path.join(data_root, "VOCdevkit/VOC2012/ImageSets/Segmentation")
        train_txt = os.path.join(split_info_path, "train.txt")
        train_aug_txt = os.path.join(split_info_path, "train_aug.txt")
        train_val_txt = os.path.join(split_info_path, "trainval.txt")
        val_txt = os.path.join(split_info_path, "val.txt")

        image_path = os.path.join(data_root, "VOCdevkit/VOC2012/JPEGImages")
        mask_path = os.path.join(data_root, "VOCdevkit/VOC2012/SegmentationClass")
        mask_aug_path = os.path.join(data_root, "VOCdevkit/VOC2012/SegmentationClassAug")

        image_info_list = {}
        for txt, now_split in zip([train_txt, train_aug_txt, train_val_txt, val_txt],
                              ["train", "train_aug", "train_val", "val"]):
            now_mask_path = mask_aug_path if now_split == "train_aug" else mask_path
            with open(txt, "r") as f:
                file_names = [x.strip() for x in f.readlines()]
                images = [os.path.join(image_path, x + ".jpg") for x in file_names]
                masks = [os.path.join(now_mask_path, x + ".png") for x in file_names]
                image_info_list[now_split] = [{"image_path": image, "label_path": mask}
                                              for image, mask in zip(images, masks)]
            pass

        return image_info_list[split]

    @classmethod
    def get_ss_info(cls, data_root=None, split="train", train_label_path=None):
        if split == "train":
            train_label_path = train_label_path if train_label_path is not None else \
                "/media/ubuntu/4T/ALISURE/USS/WSS_CAM/cam/2_CAMNet_200_32_256_0.5"

            data_info = cls.get_data_info(data_root=data_root)
            train_image_path = [one_data["image_path"] for one_data in data_info]
            train_label_path = [one_image_path.replace(os.path.join(
                data_root, "ILSVRC2017_DET/ILSVRC/Data/DET"),
                train_label_path) for one_image_path in train_image_path]
            return [{"image_path": image, "label_path": mask} for image, mask in
                                     zip(train_image_path, train_label_path)]

        if split == "test":
            test_data_dir = os.path.join(data_root, "LID_track1_imageset/LID_track1/test")
            test_image_path = sorted(glob(os.path.join(test_data_dir, "*.JPEG")))
            return [{"image_path": image} for image in test_image_path]

        if split == "val":
            val_data_dir = os.path.join(data_root, "LID_track1_imageset/LID_track1/val")
            val_image_path = glob(os.path.join(val_data_dir, "*.JPEG"))
            val_label_dir = os.path.join(data_root, "LID_track1_annotations/track1_val_annotations")
            val_label_path = sorted(glob(os.path.join(val_label_dir, "*.png")))
            return [{"image_path": image, "label_path": mask} for image, mask in
                    zip(val_image_path, val_label_path)]
        pass

    @staticmethod
    def get_class_name(mat_file=None):
        mat_file = mat_file if mat_file is not None else "/media/ubuntu/4T/ALISURE/Data/L2ID/data/meta_det.mat"
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

    @classmethod
    def transform_train_voc_ss(cls, image_size=256):
        transform_train = ExtCompose([
            ExtRandomScale((0.5, 2.0)), ExtRandomCrop(size=(image_size, image_size), pad_if_needed=True),
            ExtRandomHorizontalFlip(), ExtToTensor(),
            ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_test = ExtCompose([ExtResize(size=image_size), ExtToTensor(),
                                     ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return transform_train, transform_test

    @classmethod
    def transform_train_voc_ss_center(cls, image_size=256):
        transform_train = ExtCompose([
            ExtRandomScale((0.5, 2.0)), ExtRandomCrop(size=(image_size, image_size), pad_if_needed=True),
            ExtRandomHorizontalFlip(), ExtToTensor(),
            ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_test = ExtCompose([
            ExtResize(size=image_size), ExtCenterCrop(size=image_size), ExtToTensor(),
            ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return transform_train, transform_test

    @classmethod
    def transform_train_ss(cls, image_size=513):
        transform_train = ExtCompose([
            ExtRandomScale((0.5, 2.0)), ExtRandomCrop(size=(image_size, image_size), pad_if_needed=True),
            ExtRandomHorizontalFlip(), ExtToTensor(),
            ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_test = ExtCompose([
            ExtResize(size=image_size), ExtCenterCrop(size=image_size), ExtToTensor(),
            ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return transform_train, transform_test

    @classmethod
    def transform_test_ss(cls, image_size=513):
        transform_test = ExtCompose([ExtResize(size=image_size), ExtToTensor(),
                                     ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
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


class ImageNetSegmentation(Dataset):

    def __init__(self, images_list, transform, return_image_info=False):
        self.transform = transform
        self.images_list = images_list
        self.return_image_info = return_image_info
        pass

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        label_path, image_path = self.images_list[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(label_path)
        image, mask = self.transform(image, mask)

        if self.return_image_info:
            return image, mask, image_path

        return image, mask


    pass


class VOCSegmentation(Dataset):

    def __init__(self, images_list, transform, return_image_info=False):
        self.transform = transform
        self.images_list = images_list
        self.return_image_info = return_image_info
        pass

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        label_path, image_path = self.images_list[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(label_path)
        image, mask = self.transform(image, mask)

        if self.return_image_info:
            return image, mask, image_path

        return image, mask

    pass


class DatasetUtil(object):

    dataset_type_mlc_train = "mlc_train"
    dataset_type_mlc_val = "mlc_val"
    dataset_type_vis_cam = "vis_cam"
    dataset_type_ss_train = "ss_train"
    dataset_type_ss_val = "ss_val"
    dataset_type_ss_test = "ss_test"
    dataset_type_ss_voc_train = "ss_voc_train"
    dataset_type_ss_voc_val = "ss_voc_val"
    dataset_type_ss_voc_val_center = "ss_voc_val_center"

    @classmethod
    def get_dataset_by_type(cls, dataset_type, image_size, data_root=None,
                            return_image_info=False, sampling=False, train_label_path=None):
        if dataset_type == cls.dataset_type_mlc_train:
            data_info = DataUtil.get_data_info(data_root=data_root)
            data_info = data_info[::20] if sampling else data_info
            label_image_path = [[[one_object[2] for one_object in one_data["object"]],
                                 one_data["image_path"]] for one_data in data_info]
            return cls._get_imagenet_mlc_train(label_image_path, image_size=image_size,
                                               return_image_info=return_image_info)
        elif dataset_type == cls.dataset_type_mlc_val:
            data_info = DataUtil.get_data_info(data_root=data_root)
            data_info = data_info[::20] if sampling else data_info
            label_image_path = [[[one_object[2] for one_object in one_data["object"]],
                                 one_data["image_path"]] for one_data in data_info]
            return cls._get_imagenet_mlc_val(label_image_path, image_size, return_image_info=return_image_info)
        elif dataset_type == cls.dataset_type_vis_cam:
            data_info = DataUtil.get_data_info(data_root=data_root)
            data_info = data_info[::20] if sampling else data_info
            label_image_path = [[[one_object[2] for one_object in one_data["object"]],
                                 one_data["image_path"]] for one_data in data_info]
            return cls._get_imagenet_vis_cam(label_image_path, image_size, return_image_info=return_image_info)
        ################################################################################################################
        elif dataset_type == cls.dataset_type_ss_voc_train:
            data_info = DataUtil.get_voc_info(data_root=data_root, split="train_aug")
            data_info = data_info[::20] if sampling else data_info
            label_image_path = [[one_data["label_path"], one_data["image_path"]] for one_data in data_info]
            return cls._get_voc_ss_train(label_image_path, image_size, return_image_info=return_image_info)
        elif dataset_type == cls.dataset_type_ss_voc_val:
            data_info = DataUtil.get_voc_info(data_root=data_root, split="val")
            data_info = data_info[::20] if sampling else data_info
            label_image_path = [[one_data["label_path"], one_data["image_path"]] for one_data in data_info]
            return cls._get_voc_ss_val(label_image_path, image_size, return_image_info=return_image_info)
        elif dataset_type == cls.dataset_type_ss_voc_val_center:
            data_info = DataUtil.get_voc_info(data_root=data_root, split="val")
            data_info = data_info[::20] if sampling else data_info
            label_image_path = [[one_data["label_path"], one_data["image_path"]] for one_data in data_info]
            return cls._get_voc_ss_val_center(label_image_path, image_size, return_image_info=return_image_info)
        ################################################################################################################
        elif dataset_type == cls.dataset_type_ss_train:
            data_info = DataUtil.get_ss_info(data_root=data_root, split="train", train_label_path=train_label_path)
            data_info = data_info[::20] if sampling else data_info
            label_image_path = [[one_data["label_path"], one_data["image_path"]] for one_data in data_info]
            return cls._get_ss_train(label_image_path, image_size, return_image_info=return_image_info)
        elif dataset_type == cls.dataset_type_ss_val:
            data_info = DataUtil.get_ss_info(data_root=data_root, split="val")
            data_info = data_info[::20] if sampling else data_info
            label_image_path = [[one_data["label_path"], one_data["image_path"]] for one_data in data_info]
            return cls._get_ss_val(label_image_path, image_size, return_image_info=return_image_info)
        elif dataset_type == cls.dataset_type_ss_test:
            data_info = DataUtil.get_ss_info(data_root=data_root, split="test")
            data_info = data_info[::20] if sampling else data_info
            label_image_path = [[one_data["label_path"], one_data["image_path"]] for one_data in data_info]
            return cls._get_ss_test(label_image_path, image_size, return_image_info=return_image_info)
        else:
            raise Exception("....")
        pass

    @staticmethod
    def _get_ss_train(label_image_path, image_size, return_image_info):
        transform_train, transform_test = MyTransform.transform_train_ss(image_size=image_size)
        seg = ImageNetSegmentation(label_image_path, transform=transform_train, return_image_info=return_image_info)
        return seg

    @staticmethod
    def _get_ss_val(label_image_path, image_size, return_image_info):
        transform_train, transform_test = MyTransform.transform_train_ss(image_size=image_size)
        seg = ImageNetSegmentation(label_image_path, transform=transform_test, return_image_info=return_image_info)
        return seg

    @staticmethod
    def _get_ss_test(label_image_path, image_size, return_image_info):
        transform_test = MyTransform.transform_test_ss(image_size=image_size)
        seg = ImageNetSegmentation(label_image_path, transform=transform_test, return_image_info=return_image_info)
        return seg

    @staticmethod
    def _get_voc_ss_train(label_image_path, image_size, return_image_info):
        transform_train, transform_test = MyTransform.transform_train_voc_ss(image_size=image_size)
        voc = VOCSegmentation(label_image_path, transform=transform_train, return_image_info=return_image_info)
        return voc

    @staticmethod
    def _get_voc_ss_val(label_image_path, image_size, return_image_info):
        transform_train, transform_test = MyTransform.transform_train_voc_ss(image_size=image_size)
        voc = VOCSegmentation(label_image_path, transform=transform_test, return_image_info=return_image_info)
        return voc

    @staticmethod
    def _get_voc_ss_val_center(label_image_path, image_size, return_image_info):
        transform_train, transform_test = MyTransform.transform_train_voc_ss_center(image_size=image_size)
        voc = VOCSegmentation(label_image_path, transform=transform_test, return_image_info=return_image_info)
        return voc

    @staticmethod
    def _get_imagenet_mlc_train(label_image_path, image_size, return_image_info):
        transform_train, transform_test = MyTransform.transform_train_cam(image_size=image_size)
        imagenet_mlc = ImageNetMLC(images_list=label_image_path, num_classes=200,
                                   transform=transform_train, return_image_info=return_image_info)
        return imagenet_mlc

    @staticmethod
    def _get_imagenet_mlc_val(label_image_path, image_size, return_image_info):
        transform_train, transform_test = MyTransform.transform_train_cam(image_size=image_size)
        imagenet_mlc = ImageNetMLC(images_list=label_image_path, num_classes=200,
                                   transform=transform_test, return_image_info=return_image_info)
        return imagenet_mlc

    @staticmethod
    def _get_imagenet_vis_cam(label_image_path, image_size, return_image_info):
        transform_test = MyTransform.transform_vis_cam(image_size=image_size)
        imagenet_cam = ImageNetMLC(images_list=label_image_path, num_classes=200,
                                   transform=transform_test, return_image_info=return_image_info)
        return imagenet_cam

    pass


if __name__ == '__main__':
    # dataset_mlc_train = DatasetUtil.get_dataset_by_type(dataset_type=DatasetUtil.dataset_type_mlc_train, image_size=256)
    # dataset_mlc_val = DatasetUtil.get_dataset_by_type(dataset_type=DatasetUtil.dataset_type_mlc_val, image_size=256)
    # dataset_mlc_train.__getitem__(0)
    dataset_voc_train = DatasetUtil.get_dataset_by_type(dataset_type=DatasetUtil.dataset_type_ss_voc_train, image_size=512)
    dataset_voc_val = DatasetUtil.get_dataset_by_type(dataset_type=DatasetUtil.dataset_type_ss_voc_val, image_size=512)
    dataset_voc_train.__getitem__(0)
    pass
