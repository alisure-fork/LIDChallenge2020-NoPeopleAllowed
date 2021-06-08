import torch
import random
import numbers
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F


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

