import math
import torch
import random
import numpy as np
import torch.nn as nn
from pamr import PAMR
from torchvision import models
import torch.nn.functional as F
from alisuretool.Tools import Tools
from torchvision.models import resnet
from torch.utils.data import DataLoader, Dataset
from deep_labv3plus_pytorch.network._deeplab import ASPP
from torchvision.models._utils import IntermediateLayerGetter
from deep_labv3plus_pytorch.network.modeling import deeplabv3plus_resnet50, deeplabv3plus_resnet101, deeplabv3plus_mobilenet
from deep_labv3plus_pytorch.network.modeling import deeplabv3_resnet50, deeplabv3plus_resnet101, deeplabv3_mobilenet


class ConvBlock(nn.Module):

    def __init__(self, cin, cout, stride=1, has_relu=True, has_bn=True):
        super().__init__()
        self.has_relu = has_relu
        self.has_bn = has_bn

        self.conv = nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)
        if self.has_bn:
            self.bn = nn.BatchNorm2d(cout)
        if self.has_relu:
            self.relu = nn.ReLU(inplace=True)
        pass

    def forward(self, x):
        out = self.conv(x)
        if self.has_bn:
            out = self.bn(out)
        if self.has_relu:
            out = self.relu(out)
        return out

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class Normalize(nn.Module):

    def __init__(self, power=2):
        super().__init__()
        self.power = power
        pass

    def forward(self, x, dim=-1):
        norm = x.pow(self.power).sum(dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class ClassNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = models.vgg16(pretrained=True)
        self.head_linear = nn.Linear(512, self.num_classes)
        pass

    def forward(self, x):
        features = self.backbone.features(x)
        features = F.adaptive_avg_pool2d(features, output_size=(1, 1)).view((features.size()[0], -1))
        logits = self.head_linear(features)
        return logits

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class DeepLabV3Plus(nn.Module):

    def __init__(self, num_classes, output_stride=8, arch=deeplabv3plus_resnet50):
        super().__init__()
        self.model = arch(num_classes=num_classes, output_stride=output_stride, pretrained_backbone=True)
        pass

    def forward(self, x):
        out = self.model(x)
        return out

    def get_params_groups(self):
        return list(self.model.backbone.parameters()), list(self.model.classifier.parameters())

    pass


class MyNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # -------------Encoder--------------
        backbone = resnet.__dict__["resnet50"](pretrained=True, replace_stride_with_dilation=[False, False, False])
        return_layers = {'relu': 'e0', 'layer1': 'e1', 'layer2': 'e2', 'layer3': 'e3', 'layer4': 'e4'}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        self.l2norm = Normalize(2)

        # -------------Convert-------------
        self.convert_5 = ConvBlock(2048, 512)
        self.convert_4 = ConvBlock(1024, 512)
        self.convert_3 = ConvBlock(512, 256)
        self.convert_2 = ConvBlock(256, 256)
        self.convert_1 = ConvBlock(64, 128)

        # -------------MIC-------------
        convert_dim = 2048
        self.class_c1 = ConvBlock(convert_dim, convert_dim, has_relu=True)  # 28 32 40
        self.class_c2 = ConvBlock(convert_dim, convert_dim, has_relu=True)
        self.class_l1 = nn.Linear(convert_dim, self.num_classes)

        # -------------Decoder-------------
        self.decoder_1_b1 = resnet.BasicBlock(512, 512)  # 40
        self.decoder_1_b2 = resnet.BasicBlock(512, 512)  # 40
        self.decoder_1_c = ConvBlock(512, 512, has_relu=True)  # 40

        self.decoder_2_b1 = resnet.BasicBlock(512, 512)  # 40
        self.decoder_2_b2 = resnet.BasicBlock(512, 512)  # 40
        self.decoder_2_c = ConvBlock(512, 256, has_relu=True)  # 40

        self.decoder_3_b1 = resnet.BasicBlock(256, 256)  # 80
        self.decoder_3_b2 = resnet.BasicBlock(256, 256)  # 80
        self.decoder_3_c = ConvBlock(256, 256, has_relu=True)  # 80

        self.decoder_4_b1 = resnet.BasicBlock(256, 256)  # 160
        self.decoder_4_b2 = resnet.BasicBlock(256, 256)  # 160
        self.decoder_4_c = ConvBlock(256, 128, has_relu=True)  # 160

        self.decoder_5_b1 = resnet.BasicBlock(128, 128)  # 160
        self.decoder_5_b2 = resnet.BasicBlock(128, 128)  # 160
        self.decoder_5_out = nn.Conv2d(128, self.num_classes + 1, 3, padding=1, bias=False)  # 160
        pass

    def forward(self, x1, x2, pair_labels, label1, label2, has_class=False, has_cam=False, has_ss=False):
        result = {}

        # -------------Encoder-------------
        feature_1 = self.backbone(x1)  # (64, 160), (256, 80), (512, 40), (1024, 20), (2048, 10)
        feature_2 = self.backbone(x2)  # (64, 160), (256, 80), (512, 40), (1024, 20), (2048, 10)

        # -------------Class-------------
        class_feature_1, class_logits_1 = self._class_feature(feature=feature_1)
        class_feature_2, class_logits_2 = self._class_feature(feature=feature_2)

        if has_class:
            result["class_logits"] = {"x1": class_logits_1, "x2": class_logits_2}
            pass

        cam_1, neg_cam_1 = self._cluster_activation_map(pair_labels, class_feature_1,
                                                        self.class_l1.weight, labels=label1)  # 簇激活图
        cam_2, neg_cam_2 = self._cluster_activation_map(pair_labels, class_feature_2,
                                                        self.class_l1.weight, labels=label2)  # 簇激活图
        cam_norm_1 = self._feature_norm(cam_1)
        cam_norm_2 = self._feature_norm(cam_2)
        neg_cam_norm_1 = self._feature_norm(neg_cam_1)
        neg_cam_norm_2 = self._feature_norm(neg_cam_2)

        if has_cam:
            cam_norm_large_1 = self._up_to_target(cam_norm_1, x1)
            cam_norm_large_2 = self._up_to_target(cam_norm_2, x2)
            neg_cam_norm_large_1 = self._up_to_target(neg_cam_norm_1, x1)
            neg_cam_norm_large_2 = self._up_to_target(neg_cam_norm_2, x2)

            result["cam"] = {"cam_1": cam_1, "cam_2": cam_2,
                             "cam_norm_1": cam_norm_1, "cam_norm_2": cam_norm_2,
                             "cam_norm_large_1": cam_norm_large_1, "cam_norm_large_2": cam_norm_large_2,
                             "neg_cam_1": neg_cam_1, "neg_cam_2": neg_cam_2,
                             "neg_cam_norm_1": neg_cam_norm_1, "neg_cam_norm_2": neg_cam_norm_2,
                             "neg_cam_norm_large_1": neg_cam_norm_large_1, "neg_cam_norm_large_2": neg_cam_norm_large_2}
            pass

        # -------------Decoder-------------
        if has_ss:
            # -------------Convert-------------
            e0_1, e1_1, e2_1, e3_1, e4_1 = self._convert_feature(feature=feature_1)
            e0_2, e1_2, e2_2, e3_2, e4_2 = self._convert_feature(feature=feature_2)

            # -------------Decoder-------------
            d5_1 = self._encoder(e0_1, e1_1, e2_1, e3_1, e4_1)  # 128 * 160 * 160
            d5_2 = self._encoder(e0_2, e1_2, e2_2, e3_2, e4_2)  # 128 * 160 * 160

            ######################################################################################################
            cam_large_1 = self._up_to_target(cam_norm_1, d5_1)
            cam_large_2 = self._up_to_target(cam_norm_2, d5_2)
            neg_cam_large_1 = self._up_to_target(neg_cam_norm_1, d5_1)
            neg_cam_large_2 = self._up_to_target(neg_cam_norm_2, d5_2)

            # CAM 正掩码 Mask
            cam_mask_large_1 = torch.zeros_like(cam_large_1)
            cam_mask_large_2 = torch.zeros_like(cam_large_2)
            cam_mask_large_1[cam_large_1 > 0.5] = 1.0
            cam_mask_large_2[cam_large_2 > 0.5] = 1.0

            # CAM 负掩码 Mask
            cam_mask_min_large_1 = torch.zeros_like(neg_cam_large_1)
            cam_mask_min_large_2 = torch.zeros_like(neg_cam_large_2)
            cam_mask_min_large_1[neg_cam_large_1 > 0.5] = 1.0
            cam_mask_min_large_2[neg_cam_large_2 > 0.5] = 1.0

            # Our
            d5_mask_1 = torch.sum(torch.sum(cam_mask_large_1 * d5_1, dim=2), dim=2) / (torch.sum(cam_mask_large_1) + 1e-6)
            d5_mask_2 = torch.sum(torch.sum(cam_mask_large_2 * d5_2, dim=2), dim=2) / (torch.sum(cam_mask_large_2) + 1e-6)
            d5_mask_4d_1 = torch.unsqueeze(torch.unsqueeze(d5_mask_1, dim=-1), dim=-1).expand_as(d5_2)
            d5_mask_4d_2 = torch.unsqueeze(torch.unsqueeze(d5_mask_2, dim=-1), dim=-1).expand_as(d5_1)
            d5_mask_2_to_1 = torch.cosine_similarity(d5_mask_4d_2, d5_1, dim=1)
            d5_mask_1_to_2 = torch.cosine_similarity(d5_mask_4d_1, d5_2, dim=1)

            d5_mask_neg_1 = torch.sum(torch.sum(cam_mask_min_large_1 * d5_1, dim=2), dim=2) / (torch.sum(cam_mask_min_large_1) + 1e-6)
            d5_mask_neg_2 = torch.sum(torch.sum(cam_mask_min_large_2 * d5_2, dim=2), dim=2) / (torch.sum(cam_mask_min_large_2) + 1e-6)
            d5_mask_4d_neg_1 = torch.unsqueeze(torch.unsqueeze(d5_mask_neg_1, dim=-1), dim=-1).expand_as(d5_2)
            d5_mask_4d_neg_2 = torch.unsqueeze(torch.unsqueeze(d5_mask_neg_2, dim=-1), dim=-1).expand_as(d5_1)
            d5_mask_neg_2_to_1 = torch.cosine_similarity(d5_mask_4d_neg_2, d5_1, dim=1)
            d5_mask_neg_1_to_2 = torch.cosine_similarity(d5_mask_4d_neg_1, d5_2, dim=1)

            result["our"] = {"cam_mask_large_1": cam_mask_large_1, "cam_mask_large_2": cam_mask_large_2,
                             "cam_mask_min_large_1": cam_mask_min_large_1, "cam_mask_min_large_2": cam_mask_min_large_2,
                             "d5_mask_1_to_2": d5_mask_1_to_2, "d5_mask_2_to_1": d5_mask_2_to_1,
                             "d5_mask_neg_2_to_1": d5_mask_neg_2_to_1, "d5_mask_neg_1_to_2": d5_mask_neg_1_to_2}
            ######################################################################################################

            d5_out_1 = self.decoder_5_out(d5_1)  # 21 * 160 * 160
            d5_out_softmax_1 = torch.softmax(d5_out_1, dim=0)  # 21 * 160 * 160  # 小输出
            d5_out_up_1 = self._up_to_target(d5_out_1, x1)  # 21 * 320 * 320
            d5_out_up_softmax_1 = torch.softmax(d5_out_up_1, dim=0)  # 21 * 320 * 320  # 大输出

            d5_out_2 = self.decoder_5_out(d5_2)  # 21 * 160 * 160
            d5_out_softmax_2 = torch.softmax(d5_out_2, dim=0)  # 21 * 160 * 160  # 小输出
            d5_out_up_2 = self._up_to_target(d5_out_2, x2)  # 21 * 320 * 320
            d5_out_up_softmax_2 = torch.softmax(d5_out_up_2, dim=0)  # 21 * 320 * 320  # 大输出

            result["ss"] = {"out_1": d5_out_1, "out_2": d5_out_2,
                             "out_softmax_1": d5_out_softmax_1,"out_softmax_2": d5_out_softmax_2,
                             "out_up_1": d5_out_up_1, "out_up_2": d5_out_up_2,
                             "out_up_softmax_1": d5_out_up_softmax_1, "out_up_softmax_2": d5_out_up_softmax_2}
            pass

        return result

    def forward_inference(self, x, has_class=False, has_cam=False, has_ss=False):
        result = {}

        # -------------Encoder-------------
        feature = self.backbone(x)

        # -------------Class-------------
        if has_class:
            class_feature, class_logits = self._class_feature(feature=feature)
            result["class_logits"] = class_logits

            if has_cam:
                _, top_k_index = torch.topk(class_logits, 1, 1)
                pred_labels = [int(one[0]) for one in top_k_index]
                cam = self._cluster_activation_map(pred_labels, class_feature, self.class_l1.weight)  # 簇激活图

                cam_norm = self._feature_norm(cam)
                cam_norm_large = self._up_to_target(cam_norm, x)

                result["cam"] = {"cam": cam, "cam_norm": cam_norm, "cam_norm_large": cam_norm_large}
                pass

            pass

        # -------------Decoder-------------
        if has_ss:
            # -------------Convert-------------
            e0, e1, e2, e3, e4 = self._convert_feature(feature=feature)

            # -------------Decoder-------------
            d5 = self._encoder(e0, e1, e2, e3, e4)  # 128 * 160 * 160

            d5_out = self.decoder_5_out(d5)  # 21 * 160 * 160
            d5_out_softmax = torch.softmax(d5_out, dim=0)  # 21 * 160 * 160  # 小输出
            d5_out_up = self._up_to_target(d5_out, x)  # 21 * 320 * 320
            d5_out_up_softmax = torch.softmax(d5_out_up, dim=0)  # 21 * 320 * 320  # 大输出

            return_result = {"out": d5_out, "out_softmax": d5_out_softmax,
                             "out_up": d5_out_up, "out_up_softmax": d5_out_up_softmax}
            result["ss"] = return_result
            pass

        return result

    def _class_feature(self, feature):
        e4 = feature["e4"]  # (2048, 10)

        class_feature = self.class_c2(self.class_c1(e4))  # (512, 10)
        class_1x1 = F.adaptive_avg_pool2d(class_feature, output_size=(1, 1)).view((class_feature.size()[0], -1))
        class_logits = self.class_l1(class_1x1)
        return class_feature, class_logits

    def _convert_feature(self, feature):
        e0 = self.convert_1(feature["e0"])  # 128
        e1 = self.convert_2(feature["e1"])  # 256
        e2 = self.convert_3(feature["e2"])  # 256
        e3 = self.convert_4(feature["e3"])  # 512
        e4 = self.convert_5(feature["e4"])  # 512
        return e0, e1, e2, e3, e4

    def _encoder(self, e0, e1, e2, e3, e4):
        d1 = self.decoder_1_b2(self.decoder_1_b1(e4))  # 512 * 40 * 40
        d1_d2 = self._up_to_target(self.decoder_1_c(d1), e3) + e3  # 512 * 40 * 40

        d2 = self.decoder_2_b2(self.decoder_2_b1(d1_d2))  # 512 * 21 * 21
        d2_d3 = self._up_to_target(self.decoder_2_c(d2), e2) + e2  # 512 * 40 * 40

        d3 = self.decoder_3_b2(self.decoder_3_b1(d2_d3))  # 256 * 40 * 40
        d3_d4 = self._up_to_target(self.decoder_3_c(d3), e1) + e1  # 256 * 80 * 80

        d4 = self.decoder_4_b2(self.decoder_4_b1(d3_d4))  # 256 * 80 * 80
        d4_d5 = self._up_to_target(self.decoder_4_c(d4), e0) + e0  # 128 * 160 * 160

        d5 = self.decoder_5_b2(self.decoder_5_b1(d4_d5))  # 128 * 160 * 160
        return d5

    @staticmethod
    def _cluster_activation_map(pair_labels, class_feature, weight_softmax, labels=None):
        bz, nc, h, w = class_feature.shape

        cam_list, neg_cam_list = [], []
        for i in range(bz):
            cam_weight = weight_softmax[pair_labels[i]]
            cam_weight = cam_weight.view(nc, 1, 1).expand_as(class_feature[i])
            cam = torch.sum(torch.mul(cam_weight, class_feature[i]), dim=0, keepdim=True)
            cam[cam < 0] = 0
            cam_list.append(torch.unsqueeze(cam, 0))

            # TODO: ...
            if labels is not None:
                now_cam_sum = 0
                now_cam_where = torch.zeros_like(cam_list[i][0])
                now_label = torch.where(labels[i])[0]
                for class_one in now_label:
                    cam_weight = weight_softmax[class_one]
                    cam_weight = cam_weight.view(nc, 1, 1).expand_as(class_feature[i])
                    cam = torch.sum(torch.mul(cam_weight, class_feature[i]), dim=0, keepdim=True)
                    now_cam_where[cam > 0] = 1
                    now_cam_sum -= torch.unsqueeze(cam, 0)
                    pass
                now_cam_sum = now_cam_sum / (len(now_label) + 1e-6)
                now_cam_where = torch.unsqueeze(now_cam_where, 0)
                now_cam_sum[now_cam_where.bool()] = 0
                neg_cam_list.append(now_cam_sum)
                pass

            pass

        return torch.cat(cam_list) if labels is None else (torch.cat(cam_list), torch.cat(neg_cam_list))

    @staticmethod
    def _cluster_activation_map_old(pair_labels, class_feature, weight_softmax):
        bz, nc, h, w = class_feature.shape

        cam_list = []
        for i in range(bz):
            cam_weight = weight_softmax[pair_labels[i]]
            cam_weight = cam_weight.view(nc, 1, 1).expand_as(class_feature[i])
            cam = torch.sum(torch.mul(cam_weight, class_feature[i]), dim=0, keepdim=True)
            cam_list.append(torch.unsqueeze(cam, 0))
            pass
        return torch.cat(cam_list)

    @staticmethod
    def _up_to_target(source, target, mode="bilinear"):
        if source.size()[2] != target.size()[2] or source.size()[3] != target.size()[3]:
            align_corners = True if mode == "nearest" else False
            source = torch.nn.functional.interpolate(
                source, size=[target.size()[2], target.size()[3]], mode=mode, align_corners=align_corners)
            pass
        return source

    @staticmethod
    def _feature_norm(feature_map):
        feature_shape = feature_map.size()
        batch_min, _ = torch.min(feature_map.view((feature_shape[0], -1)), dim=-1, keepdim=True)
        batch_max, _ = torch.max(feature_map.view((feature_shape[0], -1)), dim=-1, keepdim=True)
        norm = torch.div(feature_map.view((feature_shape[0], -1)) - batch_min, batch_max - batch_min)
        return norm.view(feature_shape)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class DualNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # -------------Encoder--------------
        backbone = resnet.__dict__["resnet50"](pretrained=True, replace_stride_with_dilation=[False, False, False])
        return_layers = {'relu': 'e0', 'layer1': 'e1', 'layer2': 'e2', 'layer3': 'e3', 'layer4': 'e4'}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        self.l2norm = Normalize(2)

        # -------------Convert-------------
        self.convert_5 = ConvBlock(2048, 512)
        self.convert_4 = ConvBlock(1024, 512)
        self.convert_3 = ConvBlock(512, 256)
        self.convert_2 = ConvBlock(256, 256)
        self.convert_1 = ConvBlock(64, 128)

        # -------------MIC-------------
        convert_dim = 2048
        self.class_c1 = ConvBlock(convert_dim, convert_dim, has_relu=True)  # 28 32 40
        self.class_c2 = ConvBlock(convert_dim, convert_dim, has_relu=True)
        self.class_l1 = nn.Linear(convert_dim, self.num_classes)

        # -------------Decoder-------------
        self.decoder_1_b1 = resnet.BasicBlock(512, 512)  # 40
        self.decoder_1_b2 = resnet.BasicBlock(512, 512)  # 40
        self.decoder_1_c = ConvBlock(512, 512, has_relu=True)  # 40

        self.decoder_2_b1 = resnet.BasicBlock(512, 512)  # 40
        self.decoder_2_b2 = resnet.BasicBlock(512, 512)  # 40
        self.decoder_2_c = ConvBlock(512, 256, has_relu=True)  # 40

        self.decoder_3_b1 = resnet.BasicBlock(256, 256)  # 80
        self.decoder_3_b2 = resnet.BasicBlock(256, 256)  # 80
        self.decoder_3_c = ConvBlock(256, 256, has_relu=True)  # 80

        self.decoder_4_b1 = resnet.BasicBlock(256, 256)  # 160
        self.decoder_4_b2 = resnet.BasicBlock(256, 256)  # 160
        self.decoder_4_c = ConvBlock(256, 128, has_relu=True)  # 160

        self.decoder_5_b1 = resnet.BasicBlock(128, 128)  # 160
        self.decoder_5_b2 = resnet.BasicBlock(128, 128)  # 160
        self.decoder_5_out = nn.Conv2d(128, self.num_classes + 1, 3, padding=1, bias=False)  # 160
        pass

    def forward(self, x1, x2, pair_labels, has_class=False, has_cam=False, has_ss=False):
        result = {}

        # -------------Encoder-------------
        feature_1 = self.backbone(x1)  # (64, 160), (256, 80), (512, 40), (1024, 20), (2048, 10)
        feature_2 = self.backbone(x2)  # (64, 160), (256, 80), (512, 40), (1024, 20), (2048, 10)

        # -------------Class-------------
        class_feature_1, class_logits_1 = self._class_feature(feature=feature_1)
        class_feature_2, class_logits_2 = self._class_feature(feature=feature_2)

        if has_class:
            result["class_logits"] = {"x1": class_logits_1, "x2": class_logits_2}
            pass

        cam_1 = self._cluster_activation_map(pair_labels, class_feature_1, self.class_l1.weight)  # 簇激活图
        cam_2 = self._cluster_activation_map(pair_labels, class_feature_2, self.class_l1.weight)  # 簇激活图
        cam_norm_1 = self._feature_norm(cam_1)
        cam_norm_2 = self._feature_norm(cam_2)

        if has_cam:
            cam_norm_large_1 = self._up_to_target(cam_norm_1, x1)
            cam_norm_large_2 = self._up_to_target(cam_norm_2, x2)

            result["cam"] = {"cam_1": cam_1, "cam_2": cam_2,
                             "cam_norm_1": cam_norm_1, "cam_norm_2": cam_norm_2,
                             "cam_norm_large_1": cam_norm_large_1, "cam_norm_large_2": cam_norm_large_2}
            pass

        # -------------Decoder-------------
        if has_ss:
            # -------------Convert-------------
            e0_1, e1_1, e2_1, e3_1, e4_1 = self._convert_feature(feature=feature_1)
            e0_2, e1_2, e2_2, e3_2, e4_2 = self._convert_feature(feature=feature_2)

            # -------------Decoder-------------
            d5_1 = self._encoder(e0_1, e1_1, e2_1, e3_1, e4_1)  # 128 * 160 * 160
            d5_2 = self._encoder(e0_2, e1_2, e2_2, e3_2, e4_2)  # 128 * 160 * 160

            ######################################################################################################
            # CAM 正掩码 Mask
            cam_large_1 = self._up_to_target(cam_norm_1, x1)
            cam_large_2 = self._up_to_target(cam_norm_2, x2)
            d5_large_1 = self._up_to_target(d5_1, x1)
            d5_large_2 = self._up_to_target(d5_2, x2)

            # Our
            d5_mask_1 = torch.sum(torch.sum(cam_large_1 * d5_large_1, dim=2), dim=2) / (torch.sum(cam_large_1) + 1e-6)
            d5_mask_2 = torch.sum(torch.sum(cam_large_2 * d5_large_2, dim=2), dim=2) / (torch.sum(cam_large_2) + 1e-6)
            d5_mask_4d_1 = torch.unsqueeze(torch.unsqueeze(d5_mask_1, dim=-1), dim=-1).expand_as(d5_large_2)
            d5_mask_4d_2 = torch.unsqueeze(torch.unsqueeze(d5_mask_2, dim=-1), dim=-1).expand_as(d5_large_1)
            d5_mask_2_to_1 = torch.cosine_similarity(d5_mask_4d_2, d5_large_1, dim=1)
            d5_mask_1_to_2 = torch.cosine_similarity(d5_mask_4d_1, d5_large_2, dim=1)

            result["our"] = {"cam_large_1": cam_large_1, "cam_large_2": cam_large_2,
                             "d5_mask_1_to_2": d5_mask_1_to_2, "d5_mask_2_to_1": d5_mask_2_to_1}
            ######################################################################################################

            d5_out_1 = self.decoder_5_out(d5_1)  # 21 * 160 * 160
            d5_out_softmax_1 = torch.softmax(d5_out_1, dim=0)  # 21 * 160 * 160  # 小输出
            d5_out_up_1 = self._up_to_target(d5_out_1, x1)  # 21 * 320 * 320
            d5_out_up_softmax_1 = torch.softmax(d5_out_up_1, dim=0)  # 21 * 320 * 320  # 大输出

            d5_out_2 = self.decoder_5_out(d5_2)  # 21 * 160 * 160
            d5_out_softmax_2 = torch.softmax(d5_out_2, dim=0)  # 21 * 160 * 160  # 小输出
            d5_out_up_2 = self._up_to_target(d5_out_2, x2)  # 21 * 320 * 320
            d5_out_up_softmax_2 = torch.softmax(d5_out_up_2, dim=0)  # 21 * 320 * 320  # 大输出

            result["ss"] = {"out_1": d5_out_1, "out_2": d5_out_2,
                             "out_softmax_1": d5_out_softmax_1,"out_softmax_2": d5_out_softmax_2,
                             "out_up_1": d5_out_up_1, "out_up_2": d5_out_up_2,
                             "out_up_softmax_1": d5_out_up_softmax_1, "out_up_softmax_2": d5_out_up_softmax_2}
            pass

        return result

    def forward_inference(self, x, has_class=False, has_cam=False, has_ss=False):
        result = {}

        # -------------Encoder-------------
        feature = self.backbone(x)

        # -------------Class-------------
        if has_class:
            class_feature, class_logits = self._class_feature(feature=feature)
            result["class_logits"] = class_logits

            if has_cam:
                _, top_k_index = torch.topk(class_logits, 1, 1)
                pred_labels = [int(one[0]) for one in top_k_index]
                cam = self._cluster_activation_map(pred_labels, class_feature, self.class_l1.weight)  # 簇激活图

                cam_norm = self._feature_norm(cam)
                cam_norm_large = self._up_to_target(cam_norm, x)

                result["cam"] = {"cam": cam, "cam_norm": cam_norm, "cam_norm_large": cam_norm_large}
                pass

            pass

        # -------------Decoder-------------
        if has_ss:
            # -------------Convert-------------
            e0, e1, e2, e3, e4 = self._convert_feature(feature=feature)

            # -------------Decoder-------------
            d5 = self._encoder(e0, e1, e2, e3, e4)  # 128 * 160 * 160

            d5_out = self.decoder_5_out(d5)  # 21 * 160 * 160
            d5_out_softmax = torch.softmax(d5_out, dim=0)  # 21 * 160 * 160  # 小输出
            d5_out_up = self._up_to_target(d5_out, x)  # 21 * 320 * 320
            d5_out_up_softmax = torch.softmax(d5_out_up, dim=0)  # 21 * 320 * 320  # 大输出

            return_result = {"out": d5_out, "out_softmax": d5_out_softmax,
                             "out_up": d5_out_up, "out_up_softmax": d5_out_up_softmax}
            result["ss"] = return_result
            pass

        return result

    def _class_feature(self, feature):
        e4 = feature["e4"]  # (2048, 10)

        class_feature = self.class_c2(self.class_c1(e4))  # (512, 10)
        class_1x1 = F.adaptive_avg_pool2d(class_feature, output_size=(1, 1)).view((class_feature.size()[0], -1))
        class_logits = self.class_l1(class_1x1)
        return class_feature, class_logits

    def _convert_feature(self, feature):
        e0 = self.convert_1(feature["e0"])  # 128
        e1 = self.convert_2(feature["e1"])  # 256
        e2 = self.convert_3(feature["e2"])  # 256
        e3 = self.convert_4(feature["e3"])  # 512
        e4 = self.convert_5(feature["e4"])  # 512
        return e0, e1, e2, e3, e4

    def _encoder(self, e0, e1, e2, e3, e4):
        d1 = self.decoder_1_b2(self.decoder_1_b1(e4))  # 512 * 40 * 40
        d1_d2 = self._up_to_target(self.decoder_1_c(d1), e3) + e3  # 512 * 40 * 40

        d2 = self.decoder_2_b2(self.decoder_2_b1(d1_d2))  # 512 * 21 * 21
        d2_d3 = self._up_to_target(self.decoder_2_c(d2), e2) + e2  # 512 * 40 * 40

        d3 = self.decoder_3_b2(self.decoder_3_b1(d2_d3))  # 256 * 40 * 40
        d3_d4 = self._up_to_target(self.decoder_3_c(d3), e1) + e1  # 256 * 80 * 80

        d4 = self.decoder_4_b2(self.decoder_4_b1(d3_d4))  # 256 * 80 * 80
        d4_d5 = self._up_to_target(self.decoder_4_c(d4), e0) + e0  # 128 * 160 * 160

        d5 = self.decoder_5_b2(self.decoder_5_b1(d4_d5))  # 128 * 160 * 160
        return d5

    @staticmethod
    def _cluster_activation_map(pair_labels, class_feature, weight_softmax):
        bz, nc, h, w = class_feature.shape

        cam_list = []
        for i in range(bz):
            cam_weight = weight_softmax[pair_labels[i]]
            cam_weight = cam_weight.view(nc, 1, 1).expand_as(class_feature[i])
            cam = torch.sum(torch.mul(cam_weight, class_feature[i]), dim=0, keepdim=True)
            cam[cam < 0] = 0
            cam_list.append(torch.unsqueeze(cam, 0))
            pass

        return torch.cat(cam_list)

    @staticmethod
    def _cluster_activation_map_old(pair_labels, class_feature, weight_softmax):
        bz, nc, h, w = class_feature.shape

        cam_list = []
        for i in range(bz):
            cam_weight = weight_softmax[pair_labels[i]]
            cam_weight = cam_weight.view(nc, 1, 1).expand_as(class_feature[i])
            cam = torch.sum(torch.mul(cam_weight, class_feature[i]), dim=0, keepdim=True)
            cam_list.append(torch.unsqueeze(cam, 0))
            pass
        return torch.cat(cam_list)

    @staticmethod
    def _up_to_target(source, target, mode="bilinear"):
        if source.size()[2] != target.size()[2] or source.size()[3] != target.size()[3]:
            align_corners = True if mode == "nearest" else False
            source = torch.nn.functional.interpolate(
                source, size=[target.size()[2], target.size()[3]], mode=mode, align_corners=align_corners)
            pass
        return source

    @staticmethod
    def _feature_norm(feature_map):
        feature_shape = feature_map.size()
        batch_min, _ = torch.min(feature_map.view((feature_shape[0], -1)), dim=-1, keepdim=True)
        batch_max, _ = torch.max(feature_map.view((feature_shape[0], -1)), dim=-1, keepdim=True)
        norm = torch.div(feature_map.view((feature_shape[0], -1)) - batch_min, batch_max - batch_min + 1e-6)
        return norm.view(feature_shape)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class DualNetDeepLabHeadV3Plus(nn.Module):

    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(low_level_channels, 48, 1, bias=False),
                                     nn.BatchNorm2d(48), nn.ReLU(inplace=True))

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(nn.Conv2d(304, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True), nn.Conv2d(256, num_classes, 1))
        self._init_weight()
        pass

    def forward(self, feature_out_1, feature_out_2, feature_low_level_1, feature_low_level_2, cam_1=None, cam_2=None):
        low_level_feature_1 = self.project(feature_low_level_1)
        low_level_feature_2 = self.project(feature_low_level_2)

        output_feature_1 = self.aspp(feature_out_1)
        output_feature_2 = self.aspp(feature_out_2)
        output_feature_large_1 = self._up_to_target(output_feature_1, low_level_feature_1)
        output_feature_large_2 = self._up_to_target(output_feature_2, feature_low_level_2)

        output_feature_large_fusion_1 = torch.cat([low_level_feature_1, output_feature_large_1], dim=1)
        output_feature_large_fusion_2 = torch.cat([low_level_feature_2, output_feature_large_2], dim=1)

        out_1 = self.classifier(output_feature_large_fusion_1)
        out_2 = self.classifier(output_feature_large_fusion_2)

        #############################################################################################################
        if cam_1 is not None and cam_2 is not None:
            cam_large_1 = self._up_to_target(cam_1, output_feature_large_fusion_1)
            cam_large_2 = self._up_to_target(cam_2, output_feature_large_fusion_2)

            out_mask_1 = torch.sum(torch.sum(
                cam_large_1 * output_feature_large_fusion_1, dim=2), dim=2) / (torch.sum(cam_large_1) + 1e-6)
            out_mask_2 = torch.sum(torch.sum(
                cam_large_2 * output_feature_large_fusion_2, dim=2), dim=2) / (torch.sum(cam_large_2) + 1e-6)
            out_mask_4d_1 = torch.unsqueeze(torch.unsqueeze(
                out_mask_1, dim=-1), dim=-1).expand_as(output_feature_large_fusion_2)
            out_mask_4d_2 = torch.unsqueeze(torch.unsqueeze(
                out_mask_2, dim=-1), dim=-1).expand_as(output_feature_large_fusion_1)
            out_mask_2_to_1 = torch.cosine_similarity(out_mask_4d_2, output_feature_large_fusion_1, dim=1)
            out_mask_1_to_2 = torch.cosine_similarity(out_mask_4d_1, output_feature_large_fusion_2, dim=1)

            result_our = {"cam_large_1": cam_large_1, "cam_large_2": cam_large_2,
                          "d5_mask_1_to_2": out_mask_1_to_2, "d5_mask_2_to_1": out_mask_2_to_1}
            return out_1, out_2, result_our
        #############################################################################################################
        return out_1, out_2, None

    def forward_inference(self, feature_out, feature_low_level):
        low_level_feature = self.project(feature_low_level)
        output_feature = self.aspp(feature_out)
        output_feature_large = self._up_to_target(output_feature, low_level_feature)
        output_feature_large_fusion = torch.cat([low_level_feature, output_feature_large], dim=1)
        out = self.classifier(output_feature_large_fusion)
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                pass
            pass
        pass

    @staticmethod
    def _up_to_target(source, target, mode="bilinear"):
        if source.size()[2] != target.size()[2] or source.size()[3] != target.size()[3]:
            align_corners = True if mode == "nearest" else False
            source = torch.nn.functional.interpolate(
                source, size=[target.size()[2], target.size()[3]], mode=mode, align_corners=align_corners)
            pass
        return source

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class DualNetCAMClassifier(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()
        # self.class_c1 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 2, 1, bias=False),
        #                               nn.BatchNorm2d(in_channels // 2), nn.ReLU(inplace=True))
        # self.class_c2 = nn.Sequential(nn.Conv2d(in_channels // 2, in_channels // 4, 1, bias=False),
        #                               nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True))

        self.class_c1 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
                                      nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True))

        self.class_l1 = nn.Linear(in_channels // 4, num_classes)
        self._init_weight()
        pass

    def forward(self, feature):
        # class_feature = self.class_c2(self.class_c1(feature))  # (512, 10)
        class_feature = self.class_c1(feature)  # (512, 10)
        class_1x1 = F.adaptive_avg_pool2d(class_feature, output_size=(1, 1)).view((class_feature.size()[0], -1))
        class_logits = self.class_l1(class_1x1)
        return class_feature, class_logits

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                pass
            pass
        pass

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class DualNetDeepLabV3PlusResNet50(nn.Module):

    def __init__(self, num_classes, output_stride):
        super().__init__()
        self.num_classes = num_classes

        backbone = resnet.__dict__["resnet50"](pretrained=True, replace_stride_with_dilation=[
            False, True, True] if output_stride == 8 else [False, False, True])

        self.backbone = IntermediateLayerGetter(backbone, return_layers={'layer4': 'out', 'layer1': 'low_level'})
        self.cam_classifier = DualNetCAMClassifier(2048, num_classes=self.num_classes)
        self.classifier = DualNetDeepLabHeadV3Plus(
            2048, 256, num_classes + 1, aspp_dilate=[12, 24, 36] if output_stride == 8 else [6, 12, 18])

        self.aff = PAMR(10, [1, 2, 4, 8, 12, 24])
        pass

    def forward(self, x1, x2, pair_labels, has_class=False, has_cam=False, has_ss=False):
        result = {}

        ###########################################################################################################
        # -------------Encoder-------------
        feature_1 = self.backbone(x1)  # (64, 160), (256, 80), (512, 40), (1024, 20), (2048, 10)
        feature_2 = self.backbone(x2)  # (64, 160), (256, 80), (512, 40), (1024, 20), (2048, 10)
        # ###########################################################################################################

        cam_norm_1, cam_norm_2 = None, None

        ###########################################################################################################
        if has_class:
            class_feature_1, class_logits_1 = self.cam_classifier(feature=feature_1['out'])
            class_feature_2, class_logits_2 = self.cam_classifier(feature=feature_2['out'])
            result["class_logits"] = {"x1": class_logits_1, "x2": class_logits_2}

            if has_cam:
                cam_1 = self._cluster_activation_map(pair_labels, class_feature_1, self.cam_classifier.class_l1.weight)  # 簇激活图
                cam_2 = self._cluster_activation_map(pair_labels, class_feature_2, self.cam_classifier.class_l1.weight)  # 簇激活图
                cam_norm_1 = self._feature_norm(cam_1)
                cam_norm_2 = self._feature_norm(cam_2)
                cam_norm_1 = self._up_to_target(cam_norm_1, x1)
                cam_norm_2 = self._up_to_target(cam_norm_2, x2)
                cam_norm_aff_1 = self.aff(x1, cam_norm_1)
                cam_norm_aff_2 = self.aff(x2, cam_norm_2)

                result["cam"] = {"cam_1": cam_1, "cam_2": cam_2,
                                 "cam_norm_1": cam_norm_1, "cam_norm_2": cam_norm_2,
                                 "cam_norm_aff_1": cam_norm_aff_1, "cam_norm_aff_2": cam_norm_aff_2}
                pass

            pass
        ###########################################################################################################

        ###########################################################################################################
        if has_ss:
            out_1, out_2, our = self.classifier(feature_1['out'], feature_2['out'], feature_1['low_level'],
                                                feature_2['low_level'], cam_norm_1, cam_norm_2)
            if our is not None:
                for key in our:
                    if len(our[key].shape) == 4:
                        our[key] = self._up_to_target(our[key], x1)
                    else:
                        our[key] = torch.squeeze(self._up_to_target(torch.unsqueeze(our[key], dim=1), x1), dim=1)
                    pass
                result["our"] = our
                pass

            out_softmax_1, out_softmax_2 = torch.softmax(out_1, dim=0), torch.softmax(out_2, dim=0)
            out_up_1, out_up_2 = self._up_to_target(out_1, x1), self._up_to_target(out_2, x2)
            out_up_softmax_1, out_up_softmax_2 = torch.softmax(out_up_1, dim=0), torch.softmax(out_up_2, dim=0)
            result["ss"] = {"out_1": out_1, "out_2": out_2,
                            "out_softmax_1": out_softmax_1, "out_softmax_2": out_softmax_2,
                            "out_up_1": out_up_1, "out_up_2": out_up_2,
                            "out_up_softmax_1": out_up_softmax_1, "out_up_softmax_2": out_up_softmax_2}
            pass
        # ###########################################################################################################

        return result

    def forward_inference(self, x, has_class=False, has_cam=False, has_ss=False):
        result = {}

        # -------------Encoder-------------
        feature = self.backbone(x)

        # -------------Class-------------
        if has_class:
            class_feature, class_logits = self.cam_classifier(feature=feature['out'])
            result["class_logits"] = class_logits

            if has_cam:
                _, top_k_index = torch.topk(class_logits, 1, 1)
                pred_labels = [int(one[0]) for one in top_k_index]
                cam = self._cluster_activation_map(pred_labels, class_feature, self.cam_classifier.class_l1.weight)  # 簇激活图

                cam_norm = self._feature_norm(cam)
                cam_norm_large = self._up_to_target(cam_norm, x)

                result["cam"] = {"cam": cam, "cam_norm": cam_norm, "cam_norm_large": cam_norm_large}
                pass

            pass

        # -------------Decoder-------------
        if has_ss:
            out = self.classifier.forward_inference(feature['out'], feature['low_level'])
            out_softmax = torch.softmax(out, dim=0)
            out_up = self._up_to_target(out, x)
            out_up_softmax = torch.softmax(out_up, dim=0)

            return_result = {"out": out, "out_softmax": out_softmax,
                             "out_up": out_up, "out_up_softmax": out_up_softmax}
            result["ss"] = return_result
            pass

        return result

    @staticmethod
    def _cluster_activation_map(pair_labels, class_feature, weight_softmax):
        bz, nc, h, w = class_feature.shape

        cam_list = []
        for i in range(bz):
            cam_weight = weight_softmax[pair_labels[i]]
            cam_weight = cam_weight.view(nc, 1, 1).expand_as(class_feature[i])
            cam = torch.sum(torch.mul(cam_weight, class_feature[i]), dim=0, keepdim=True)
            cam[cam < 0] = 0
            cam_list.append(torch.unsqueeze(cam, 0))
            pass

        return torch.cat(cam_list)

    @staticmethod
    def _cluster_activation_map_old(pair_labels, class_feature, weight_softmax):
        bz, nc, h, w = class_feature.shape

        cam_list = []
        for i in range(bz):
            cam_weight = weight_softmax[pair_labels[i]]
            cam_weight = cam_weight.view(nc, 1, 1).expand_as(class_feature[i])
            cam = torch.sum(torch.mul(cam_weight, class_feature[i]), dim=0, keepdim=True)
            cam_list.append(torch.unsqueeze(cam, 0))
            pass
        return torch.cat(cam_list)

    @staticmethod
    def _up_to_target(source, target, mode="bilinear"):
        if source.size()[2] != target.size()[2] or source.size()[3] != target.size()[3]:
            align_corners = True if mode == "nearest" else False
            source = torch.nn.functional.interpolate(
                source, size=[target.size()[2], target.size()[3]], mode=mode, align_corners=align_corners)
            pass
        return source

    @staticmethod
    def _feature_norm(feature_map):
        feature_shape = feature_map.size()
        batch_min, _ = torch.min(feature_map.view((feature_shape[0], -1)), dim=-1, keepdim=True)
        batch_max, _ = torch.max(feature_map.view((feature_shape[0], -1)), dim=-1, keepdim=True)
        norm = torch.div(feature_map.view((feature_shape[0], -1)) - batch_min, batch_max - batch_min + 1e-6)
        return norm.view(feature_shape)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class DualNetDeepLabV3Plus(nn.Module):

    def __init__(self, num_classes, output_stride=8):
        super().__init__()
        self.model = DualNetDeepLabV3PlusResNet50(num_classes=num_classes, output_stride=output_stride)
        pass

    def forward(self, x1, x2, pair_labels, has_class=False, has_cam=False, has_ss=False):
        out = self.model(x1, x2, pair_labels, has_class=has_class, has_cam=has_cam, has_ss=has_ss)
        return out

    def forward_inference(self, x, has_class=False, has_cam=False, has_ss=False):
        out = self.model.forward_inference(x, has_class=has_class, has_cam=has_cam, has_ss=has_ss)
        return out

    def get_params_groups(self):
        return list(self.model.backbone.parameters()), list(
            self.model.cam_classifier.parameters()), list(self.model.classifier.parameters())

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass

