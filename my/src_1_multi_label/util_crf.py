import os
import torch
import numpy as np
import torch.nn as nn
from skimage import morphology
import torch.nn.functional as F
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, unary_from_labels


class CRFTool(object):

    @staticmethod
    def crf(image, annotation, t=5, n_label=2, a=0.1, b=0.9):  # [3, w, h], [1, w, h]
        image = np.ascontiguousarray(image)
        annotation = np.concatenate([annotation, 1 - annotation], axis=0)
        h, w = image.shape[:2]

        d = dcrf.DenseCRF2D(w, h, 2)
        unary = unary_from_softmax(annotation)
        unary = np.ascontiguousarray(unary)
        d.setUnaryEnergy(unary)
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=np.copy(image), compat=10)
        q = d.inference(t)

        result = np.array(q).reshape((2, h, w))
        return result[0]

    @staticmethod
    def crf_dss(image, annotation, t=1, n_label=2, a=0.1, b=0.9):  # [3, w, h], [1, w, h]
        image = np.ascontiguousarray(image)
        h, w = image.shape[:2]

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        anno_norm = annotation
        EPSILON = 1e-8
        tau = 1.05
        n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * sigmoid(1 - anno_norm))
        p_energy = -np.log(anno_norm + EPSILON) / (tau * sigmoid(anno_norm))
        U = np.zeros((2, h * w), dtype='float32')
        U[0, :] = n_energy.flatten()
        U[1, :] = p_energy.flatten()

        d = dcrf.DenseCRF2D(w, h, 2)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=np.copy(image), compat=5)
        q = d.inference(t)

        result = np.array(q)[1, :].reshape((h, w))
        return result

    @staticmethod
    def crf_label(image, annotation, t=5, n_label=2, a=0.3, b=0.5):
        image = np.ascontiguousarray(image)
        h, w = image.shape[:2]
        annotation = np.squeeze(np.array(annotation))

        a, b = (a * 255, b * 255) if np.max(annotation) > 1 else (a, b)
        label_extend = np.zeros_like(annotation, dtype=np.int)
        label_extend[annotation >= b] = 2
        label_extend[annotation <= a] = 1
        _, label = np.unique(label_extend, return_inverse=True)

        d = dcrf.DenseCRF2D(w, h, n_label)
        u = unary_from_labels(label, n_label, gt_prob=0.7, zero_unsure=True)
        u = np.ascontiguousarray(u)
        d.setUnaryEnergy(u)
        d.addPairwiseGaussian(sxy=(3, 3), compat=3)
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=np.copy(image), compat=10)
        q = d.inference(t)
        map_result = np.argmax(q, axis=0)
        result = map_result.reshape((h, w))
        return result

    @classmethod
    def crf_torch(cls, img, annotation, t=5, is_dss=False):
        img_data = np.asarray(img, dtype=np.uint8)
        annotation_data = np.asarray(annotation)
        result = []
        for img_data_one, annotation_data_one in zip(img_data, annotation_data):
            img_data_one = np.transpose(img_data_one, axes=(1, 2, 0))
            if is_dss:
                result_one = cls.crf_dss(img_data_one, annotation_data_one)
            else:
                result_one = cls.crf(img_data_one, annotation_data_one, t=t)
                # result_one = cls.crf_label(img_data_one, annotation_data_one, t=t)
                pass
            result.append(np.expand_dims(result_one, axis=0))
            pass
        return torch.tensor(np.asarray(result))

    pass
