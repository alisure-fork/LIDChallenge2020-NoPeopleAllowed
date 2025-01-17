import os
import torch
import random
import numpy as np
from glob import glob
from PIL import Image
import scipy.io as scio
from torchvision import transforms
from alisuretool.Tools import Tools
from torch.utils.data import Dataset
from util_transform import ExtRandomCrop, ExtResize, ExtToTensor, UnNormalize, ExtNormalize
from util_transform import ExtCompose, ExtCenterCrop, ExtRandomScale, ExtRandomHorizontalFlip


class DataUtil(object):

    @staticmethod
    def get_data_info(data_root=None):
        data_root = data_root if data_root is not None else "/media/ubuntu/4T/ALISURE/Data/L2ID/data"
        image_info_path = os.path.join(data_root, "deal", "image_info_list2.pkl")
        image_info_list = Tools.read_from_pkl(image_info_path)
        return image_info_list

    @staticmethod
    def get_ss_info_after_filter(pkl_root=None):
        _pkl_root = "/media/ubuntu/4T/ALISURE/USS/ConTa/pseudo_mask/result/2/sem_seg/train_ss.pkl"
        pkl_root = pkl_root if pkl_root is not None else _pkl_root
        Tools.print("Read pkl from {}".format(pkl_root))
        image_info_list = Tools.read_from_pkl(pkl_root)
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
    def get_ss_info(cls, data_root=None, split="train", train_label_dir=None):

        if split == "train":
            assert train_label_dir is not None
            if "sem_seg" in train_label_dir:
                data_info = cls.get_ss_info_after_filter()

                train_image_path = [one_data[1] for one_data in data_info]
                train_image_label = [one_data[0] for one_data in data_info]
            elif "ss_7_train_500" in train_label_dir:
                data_info = cls.get_ss_info_after_filter(
                    pkl_root="/media/ubuntu/4T/ALISURE/USS/WSS_Model_SS_0602_EVAL/3_DeepLabV3PlusResNet152_"
                             "201_10_18_1_352_8_balance/ss_7_train_500/train_final/train_ss_self_training.pkl")

                train_image_path = [one_data[1] for one_data in data_info]
                train_image_label = [one_data[0] for one_data in data_info]
            else:
                data_info = cls.get_data_info(data_root=data_root)

                train_image_path = [one_data["image_path"] for one_data in data_info]
                train_image_label = [[one[2] for one in one_data["object"]] for one_data in data_info]
                pass

            ########################################################
            train_image_path = [one_image_path for one_image_path in train_image_path if os.path.exists(
                one_image_path.replace(os.path.join(data_root, "ILSVRC2017_DET/ILSVRC/Data/DET"),
                                       train_label_dir).replace(".JPEG", ".png"))]

            train_label_path = [
                one_image_path.replace(os.path.join(data_root, "ILSVRC2017_DET/ILSVRC/Data/DET"),
                                       train_label_dir).replace(".JPEG", ".png")
                for one_image_path in train_image_path]
            ########################################################

            Tools.print("{}".format(len(train_image_path)))

            return [{"image_path": image, "label_path": mask, "label": label} for image, mask, label in
                                     zip(train_image_path, train_label_path, train_image_label)]

        if split == "test":
            test_data_dir = os.path.join(data_root, "LID_track1_imageset/LID_track1/test")
            test_image_path = sorted(glob(os.path.join(test_data_dir, "*.JPEG")))
            return [{"image_path": image} for image in test_image_path]

        if split == "val":
            val_data_dir = os.path.join(data_root, "LID_track1_imageset/LID_track1/val")
            val_image_path = sorted(glob(os.path.join(val_data_dir, "*.JPEG")))
            val_label_dir = os.path.join(data_root, "LID_track1_annotations/track1_val_annotations")
            val_label_path = sorted(glob(os.path.join(val_label_dir, "*.png")))
            return [{"image_path": image, "label_path": mask} for image, mask in
                    zip(val_image_path, val_label_path)]

        if split == "inference_train":
            data_info = cls.get_data_info(data_root=data_root)

            train_image_path = [one_data["image_path"] for one_data in data_info]
            Tools.print("{}".format(len(train_image_path)))
            return [{"image_path": image} for image in train_image_path]

        pass

    @staticmethod
    def get_class_name(data_root=None, mat_file=None):
        data_root = data_root if data_root is not None else "/media/ubuntu/4T/ALISURE/Data/L2ID/data"
        mat_file = mat_file if mat_file is not None else "meta_det.mat"
        data = scio.loadmat(os.path.join(data_root, mat_file))

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

    @staticmethod
    def read_image(image_path, is_rgb, max_size=500):
        im = Image.open(image_path)
        im = im.convert("RGB") if is_rgb else im

        if max_size > 0:
            value1 = max_size if im.size[0] > im.size[1] else im.size[0] * max_size // im.size[1]
            value2 = im.size[1] * max_size // im.size[0] if im.size[0] > im.size[1] else max_size
            im = im.resize((value1, value2), resample=Image.NEAREST)
            pass
        return im

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
        # transform_test = ExtCompose([ExtResize(size=image_size), ExtToTensor(),
        #                              ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return transform_train, transform_test

    @classmethod
    def transform_train_ss_2(cls, image_size=513):
        transform_test = ExtCompose([
            ExtResize(size=(image_size, image_size)), ExtToTensor(),
            ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return transform_test

    @classmethod
    def transform_test_ss(cls, image_size=513):
        transform_test = ExtCompose([ExtResize(size=image_size), ExtToTensor(),
                                     ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return transform_test

    pass


class ImageNetPerson(Dataset):

    def __init__(self, images_list, transform, return_image_info=False):
        self.images_list = images_list
        self.transform = transform
        self.train_images_list = None
        self.return_image_info = return_image_info
        self.reset()
        pass

    def __len__(self):
        return len(self.train_images_list)

    def reset(self):
        person = [one for one in self.images_list if one[0] == 1]
        zero = [one for one in self.images_list if one[0] == 0]
        zero_index = list(range(len(zero)))
        np.random.shuffle(zero_index)
        zero = [zero[index] for index in zero_index[:len(person)]]
        self.train_images_list = zero + person
        np.random.shuffle(self.train_images_list)
        pass

    def __getitem__(self, idx):
        image_label, image_path = self.train_images_list[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        if self.return_image_info:
            return image, image_label, image_path
        return image, image_label

    pass


class ImageNetPersonAll(Dataset):

    def __init__(self, images_list, transform, return_image_info=False):
        self.images_list = images_list
        self.transform = transform
        self.return_image_info = return_image_info
        pass

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_label, image_path = self.images_list[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, image_label, idx

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


class ImageNetMLCScales(Dataset):
    """ImageNet dataset for multi-label classification"""

    def __init__(self, images_list, transform_list, num_classes, return_image_info=False):
        self.images_list = images_list
        self.transform_list = transform_list
        self.num_classes = num_classes
        self.return_image_info = return_image_info
        pass

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_label, image_path = self.images_list[idx]

        image = Image.open(image_path).convert("RGB")
        image_list = [transform(image) for transform in self.transform_list]

        label_encoded = torch.zeros(self.num_classes, dtype=torch.float32)
        label_encoded[np.array(image_label, dtype=np.int) - 1] = 1

        if self.return_image_info:
            return image_list, label_encoded, image_path
        return image_list, label_encoded

    pass


class ImageNetMLCBalance(Dataset):
    """ImageNet dataset for multi-label classification"""

    def __init__(self, images_list, transform, num_classes, return_image_info=False, sample_num=None):
        self.images_list = images_list
        self.transform = transform
        self.num_classes = num_classes
        self.return_image_info = return_image_info
        self.train_images_list = None
        self.sample_num = sample_num if sample_num is not None else 500

        self.all_image_dict = {}
        for one in self.images_list:
            for one_one in one[0]:
                if one_one not in self.all_image_dict:
                    self.all_image_dict[one_one] = []
                self.all_image_dict[one_one].append(one)
                pass
            pass

        self.reset()
        pass

    def __len__(self):
        return len(self.train_images_list)

    def reset(self):
        image_list = [random.choices(self.all_image_dict[key], k=self.sample_num) for key in self.all_image_dict]
        self.train_images_list = []
        for select_image in image_list:
            self.train_images_list += select_image
        np.random.shuffle(self.train_images_list)
        pass

    def __getitem__(self, idx):
        image_label, image_path = self.train_images_list[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label_encoded = torch.zeros(self.num_classes, dtype=torch.float32)
        label_encoded[np.array(image_label, dtype=np.int) - 1] = 1

        if self.return_image_info:
            return image, label_encoded, image_path
        return image, label_encoded

    pass


class ImageNetSegmentation(Dataset):

    def __init__(self, images_list, transform, return_image_info=False, max_size=500):
        self.transform = transform
        self.images_list = images_list
        self.return_image_info = return_image_info
        self.max_size = max_size
        pass

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        label_path, image_path, _ = self.images_list[idx]

        image = DataUtil.read_image(image_path, is_rgb=True, max_size=self.max_size)

        if label_path is not None:
            mask = DataUtil.read_image(label_path, is_rgb=False, max_size=self.max_size)

            image, mask = self.transform(image, mask)
            if self.return_image_info:
                return image, mask, image_path
            return image, mask
        else:
            mask = Image.fromarray(np.zeros_like(np.asarray(image))).convert("L")
            image, mask = self.transform(image, mask)
            if self.return_image_info:
                return image, mask, image_path
            return image, mask
            pass

        pass

    pass


class ImageNetSegmentationScales(Dataset):

    def __init__(self, images_list, transform_list, return_image_info=False, max_size=0):
        self.images_list = images_list
        self.transform_list = transform_list
        self.return_image_info = return_image_info
        self.max_size = max_size
        pass

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        label_path, image_path, _ = self.images_list[idx]

        image = DataUtil.read_image(image_path, is_rgb=True, max_size=self.max_size)

        if label_path is not None:
            mask = DataUtil.read_image(label_path, is_rgb=False, max_size=self.max_size)
        else:
            mask = Image.fromarray(np.zeros_like(np.asarray(image))).convert("L")
            pass

        image_mask_list = [transform(image, mask) for transform in self.transform_list]
        image_list = [one[0] for one in image_mask_list]
        mask_list = [one[1] for one in image_mask_list]

        if self.return_image_info:
            return image_list, mask_list, image_path
        return image_list, mask_list

    pass


class ImageNetSegmentationBalance(Dataset):

    def __init__(self, images_list, transform, return_image_info=False, sample_num=None, max_size=500):
        self.transform = transform
        self.images_list = images_list
        self.return_image_info = return_image_info
        self.train_images_list = None
        self.sample_num = sample_num if sample_num is not None else 1000
        self.max_size = max_size

        self.all_image_dict = {}
        for one in self.images_list:
            now_one = set(one[2])
            # if len(now_one) <= 2:
            if len(now_one) <= len(now_one):
                for one_one in now_one:
                    if one_one not in self.all_image_dict:
                        self.all_image_dict[one_one] = []
                    self.all_image_dict[one_one].append(one)
                    pass
                pass
            pass

        self.reset()
        pass

    def __len__(self):
        return len(self.train_images_list)

    def reset(self):
        image_list = [random.choices(self.all_image_dict[key], k=self.sample_num) for key in self.all_image_dict]
        self.train_images_list = []
        for select_image in image_list:
            self.train_images_list += select_image
        np.random.shuffle(self.train_images_list)
        Tools.print("Reset Sample: {}-{}".format(self.sample_num, len(self.train_images_list)))
        pass

    def __getitem__(self, idx):
        label_path, image_path, _ = self.train_images_list[idx]

        image = DataUtil.read_image(image_path, is_rgb=True, max_size=self.max_size)
        mask = DataUtil.read_image(label_path, is_rgb=False, max_size=self.max_size)

        image, mask = self.transform(image, mask)
        if self.return_image_info:
            return image, mask, image_path
        return image, mask

    pass


class ImageNetSegmentationBalance2(Dataset):

    def __init__(self, images_list, transform, return_image_info=False, sample_num=None, max_size=500):
        self.transform = transform
        self.images_list = images_list
        self.return_image_info = return_image_info
        self.train_images_list = None
        self.sample_num = sample_num if sample_num is not None else 10000
        self.max_size = max_size

        self.all_image_dict = {}
        for one in self.images_list:
            now_one = set(one[2])
            # if len(now_one) <= 2:
            if len(now_one) <= len(now_one):
                for one_one in now_one:
                    if one_one not in self.all_image_dict:
                        self.all_image_dict[one_one] = []
                    self.all_image_dict[one_one].append(one)
                    pass
                pass
            pass

        self.reset()
        pass

    def __len__(self):
        return len(self.train_images_list)

    def reset(self):
        image_list = [random.choices(self.all_image_dict[key], k=self.sample_num) for key in self.all_image_dict]
        self.train_images_list = []
        for select_image in image_list:
            self.train_images_list += select_image
        np.random.shuffle(self.train_images_list)
        Tools.print("Reset Sample: {}-{}".format(self.sample_num, len(self.train_images_list)))
        pass

    def __getitem__(self, idx):
        label_path, image_path, _ = self.train_images_list[idx]

        image = DataUtil.read_image(image_path, is_rgb=True, max_size=self.max_size)
        mask = DataUtil.read_image(label_path, is_rgb=False, max_size=self.max_size)

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

    dataset_type_person = "person"
    dataset_type_mlc_no_person = "mlc_no_person"
    dataset_type_mlc = "mlc"
    dataset_type_ss = "ss"
    dataset_type_ss_scale = "ss_scale"
    dataset_type_ss_scale_train = "ss_scale_train"

    dataset_type_ss_voc_train = "ss_voc_train"
    dataset_type_ss_voc_val = "ss_voc_val"
    dataset_type_ss_voc_val_center = "ss_voc_val_center"

    @classmethod
    def get_dataset_by_type(cls, dataset_type, image_size, data_root=None, scales=None, is_balance=False,
                            return_image_info=False, sampling=False, train_label_path=None, max_size=500):
        ################################################################################################################
        if dataset_type == cls.dataset_type_person:
            data_info = DataUtil.get_data_info(data_root=data_root)
            data_info = data_info[::20] if sampling else data_info

            label_image_path = []
            for one_data in data_info:
                is_person = 1 if 124 in [one[2] for one in one_data["object"]] else 0
                label_image_path.append([is_person, one_data["image_path"]])
                pass
            data_0 = cls._get_imagenet_person_train(label_image_path, image_size=image_size,
                                                    return_image_info=return_image_info)
            data_1 = cls._get_imagenet_person_val(label_image_path, image_size=image_size,
                                                  return_image_info=return_image_info)
            data_2 = cls._get_imagenet_person_all_val(label_image_path, image_size=image_size,
                                                      return_image_info=return_image_info)
            return data_0, data_1, data_2
        ################################################################################################################
        elif dataset_type == cls.dataset_type_mlc_no_person:
            data_info = DataUtil.get_data_info(data_root=data_root)
            data_info = data_info[::20] if sampling else data_info
            sample_num = 20 if sampling else None

            label_image_path = []
            for one_data in data_info:
                now_label = []
                for one in one_data["object"]:
                    if one[2] < 124:
                        now_label.append(one[2])
                    elif one[2] > 124:
                        now_label.append(one[2] - 1)
                    pass
                now_label = list(set(now_label))
                if len(now_label) > 0:
                    label_image_path.append([now_label, one_data["image_path"]])
                    pass
                pass

            data_0 = cls._get_imagenet_mlc_no_person_train(label_image_path, image_size=image_size,
                                                           return_image_info=return_image_info, sample_num=sample_num)
            data_1 = cls._get_imagenet_mlc_no_person_val(label_image_path, image_size=image_size,
                                                         return_image_info=return_image_info)
            return data_0, data_1
        ################################################################################################################
        elif dataset_type == cls.dataset_type_mlc:
            data_info = DataUtil.get_data_info(data_root=data_root)
            data_info = data_info[::20] if sampling else data_info
            sample_num = 20 if sampling else None

            label_image_path = [[[one_object[2] for one_object in one_data["object"]],
                                 one_data["image_path"]] for one_data in data_info]

            train =  cls._get_imagenet_mlc_train(
                label_image_path, image_size=image_size, return_image_info=return_image_info, sample_num=sample_num)
            val = cls._get_imagenet_mlc_val(label_image_path, image_size, return_image_info=return_image_info)
            cam = cls._get_imagenet_vis_cam(label_image_path, image_size,
                                            scales=scales, return_image_info=return_image_info)
            return train, val, cam
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
        elif dataset_type == cls.dataset_type_ss:
            data_info = DataUtil.get_ss_info(data_root=data_root, split="train", train_label_dir=train_label_path)
            data_info = data_info[::20] if sampling else data_info

            label_image_path = [[one_data["label_path"], one_data["image_path"], one_data["label"]] for one_data in data_info]
            transform_train, transform_test = MyTransform.transform_train_ss(image_size=image_size)
            ############################################################################################################
            train_class = ImageNetSegmentationBalance if is_balance else ImageNetSegmentation
            # train_class = ImageNetSegmentationBalance2 if is_balance else ImageNetSegmentation
            ############################################################################################################
            train = train_class(label_image_path, transform_train,
                                return_image_info=return_image_info, max_size=max_size)

            label_image_path = [[one_data["label_path"], one_data["image_path"], None] for one_data in data_info]
            transform_test = MyTransform.transform_train_ss_2(image_size=image_size)
            train_eval = ImageNetSegmentation(label_image_path, transform=transform_test,
                                              return_image_info=return_image_info, max_size=max_size)

            data_info = DataUtil.get_ss_info(data_root=data_root, split="val")
            data_info = data_info[::20] if sampling else data_info

            label_image_path = [[one_data["label_path"], one_data["image_path"], None] for one_data in data_info]
            transform_train, transform_test = MyTransform.transform_train_ss(image_size=image_size)
            val = ImageNetSegmentation(label_image_path, transform=transform_test,
                                       return_image_info=return_image_info, max_size=max_size)

            return train, train_eval, val
        ################################################################################################################
        elif dataset_type == cls.dataset_type_ss_scale:
            data_info = DataUtil.get_ss_info(data_root=data_root, split="val")
            data_info = data_info[::20] if sampling else data_info
            label_image_path = [[one_data["label_path"], one_data["image_path"], None] for one_data in data_info]

            #############################################
            transform_test_list = []
            if scales is None:
                transform_test = MyTransform.transform_test_ss(image_size=image_size)
                transform_test_list.append(transform_test)
            else:
                for scale in scales:
                    transform_test = MyTransform.transform_test_ss(image_size=int(scale * image_size))
                    transform_test_list.append(transform_test)
                    pass
                pass
            #############################################

            inference_scale_val = ImageNetSegmentationScales(
                label_image_path, transform_list=transform_test_list, return_image_info=True, max_size=max_size)

            data_info = DataUtil.get_ss_info(data_root=data_root, split="test")
            data_info = data_info[::20] if sampling else data_info
            label_image_path = [[None, one_data["image_path"], None] for one_data in data_info]

            inference_scale_test = ImageNetSegmentationScales(
                label_image_path, transform_list=transform_test_list, return_image_info=True, max_size=max_size)
            return inference_scale_val, inference_scale_test
        ################################################################################################################
        elif dataset_type == cls.dataset_type_ss_scale_train:
            data_info = DataUtil.get_ss_info(data_root=data_root, split="inference_train")
            # data_info = data_info[:len(data_info) // 6]
            # data_info = data_info[len(data_info) // 6 * 5:][::-1]
            # data_info = data_info[len(data_info) // 6 * 2:len(data_info) // 6 * 3]
            # data_info = data_info[len(data_info) // 6:len(data_info) // 6 * 2]
            # data_info = data_info[len(data_info) // 6 * 3:len(data_info) // 6 * 4][::-1]
            # data_info = data_info[len(data_info) // 6 * 4:len(data_info) // 6 * 5][::-1]
            data_info = data_info[::20] if sampling else data_info
            label_image_path = [[None, one_data["image_path"], None] for one_data in data_info]

            #############################################
            transform_test_list = []
            if scales is None:
                transform_test = MyTransform.transform_test_ss(image_size=image_size)
                transform_test_list.append(transform_test)
            else:
                for scale in scales:
                    transform_test = MyTransform.transform_test_ss(image_size=int(scale * image_size))
                    transform_test_list.append(transform_test)
                    pass
                pass
            #############################################

            inference_scale = ImageNetSegmentationScales(
                label_image_path, transform_list=transform_test_list, return_image_info=True, max_size=max_size)
            return inference_scale
        ################################################################################################################
        else:
            raise Exception("....")
        pass

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
    def _get_imagenet_mlc_train(label_image_path, image_size, return_image_info, sample_num=None):
        transform_train, transform_test = MyTransform.transform_train_cam(image_size=image_size)
        imagenet_mlc = ImageNetMLCBalance(images_list=label_image_path, num_classes=200, sample_num=sample_num,
                                          transform=transform_train, return_image_info=return_image_info)
        return imagenet_mlc

    @staticmethod
    def _get_imagenet_mlc_val(label_image_path, image_size, return_image_info):
        transform_train, transform_test = MyTransform.transform_train_cam(image_size=image_size)
        imagenet_mlc = ImageNetMLC(images_list=label_image_path, num_classes=200,
                                   transform=transform_test, return_image_info=return_image_info)
        return imagenet_mlc

    @staticmethod
    def _get_imagenet_vis_cam(label_image_path, image_size, return_image_info, scales=None):
        transform_test_list = []
        if scales is None:
            transform_test = MyTransform.transform_vis_cam(image_size=image_size)
            transform_test_list.append(transform_test)
        else:
            for scale in scales:
                transform_test = MyTransform.transform_vis_cam(image_size=int(scale * image_size))
                transform_test_list.append(transform_test)
                pass
            pass

        imagenet_cam = ImageNetMLCScales(images_list=label_image_path, num_classes=200,
                                         transform_list=transform_test_list, return_image_info=return_image_info)
        return imagenet_cam

    @staticmethod
    def _get_imagenet_person_train(label_image_path, image_size, return_image_info):
        transform_train, transform_test = MyTransform.transform_train_cam(image_size=image_size)
        person = ImageNetPerson(images_list=label_image_path,
                                transform=transform_train, return_image_info=return_image_info)
        return person

    @staticmethod
    def _get_imagenet_person_val(label_image_path, image_size, return_image_info):
        transform_train, transform_test = MyTransform.transform_train_cam(image_size=image_size)
        person = ImageNetPerson(images_list=label_image_path,
                                transform=transform_test, return_image_info=return_image_info)
        return person

    @staticmethod
    def _get_imagenet_person_all_val(label_image_path, image_size, return_image_info):
        transform_train, transform_test = MyTransform.transform_train_cam(image_size=image_size)
        person = ImageNetPersonAll(images_list=label_image_path,
                                   transform=transform_test, return_image_info=return_image_info)
        return person

    @staticmethod
    def _get_imagenet_mlc_no_person_train(label_image_path, image_size, return_image_info, sample_num=None):
        transform_train, transform_test = MyTransform.transform_train_cam(image_size=image_size)
        mlc_no_person = ImageNetMLCBalance(images_list=label_image_path, num_classes=199, sample_num=sample_num,
                                           transform=transform_train, return_image_info=return_image_info)
        return mlc_no_person

    @staticmethod
    def _get_imagenet_mlc_no_person_val(label_image_path, image_size, return_image_info):
        transform_train, transform_test = MyTransform.transform_train_cam(image_size=image_size)
        mlc_no_person = ImageNetMLC(images_list=label_image_path, num_classes=199,
                                    transform=transform_test, return_image_info=return_image_info)
        return mlc_no_person

    pass


if __name__ == '__main__':
    # dataset_mlc_train = DatasetUtil.get_dataset_by_type(dataset_type=DatasetUtil.dataset_type_mlc_train, image_size=256)
    # dataset_mlc_val = DatasetUtil.get_dataset_by_type(dataset_type=DatasetUtil.dataset_type_mlc_val, image_size=256)
    # dataset_mlc_train.__getitem__(0)
    dataset_voc_train = DatasetUtil.get_dataset_by_type(dataset_type=DatasetUtil.dataset_type_ss_voc_train, image_size=512)
    dataset_voc_val = DatasetUtil.get_dataset_by_type(dataset_type=DatasetUtil.dataset_type_ss_voc_val, image_size=512)
    dataset_voc_train.__getitem__(0)
    pass
