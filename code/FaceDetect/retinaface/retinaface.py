import time
from itertools import product as product
from math import ceil
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import nms
from .nets.retinaface import RetinaFace


cfg_mnet = {
    'name'              : 'mobilenet0.25',
    'min_sizes'         : [[16, 32], [64, 128], [256, 512]],
    'steps'             : [8, 16, 32],
    'variance'          : [0.1, 0.2],
    'clip'              : False,
    'loc_weight'        : 2.0,
    'train_image_size'  : 840,
    'return_layers'     : {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel'        : 32,
    'out_channel'       : 64
}

cfg_re50 = {
    'name'              : 'Resnet50',
    'min_sizes'         : [[16, 32], [64, 128], [256, 512]],
    'steps'             : [8, 16, 32],
    'variance'          : [0.1, 0.2],
    'clip'              : False,
    'loc_weight'        : 2.0,
    'train_image_size'  : 840,
    'return_layers'     : {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel'        : 256,
    'out_channel'       : 256
}


def decode(loc, priors, variances):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, :, 2:] * variances[1])), -1)
    boxes[:, :, :2] -= boxes[:, :, 2:] / 2
    boxes[:, :, 2:] += boxes[:, :, :2]
    return boxes


class Anchors(object):
    def __init__(self, cfg, image_size=None):
        super(Anchors, self).__init__()
        self.min_sizes  = cfg['min_sizes']
        self.steps      = cfg['steps']
        self.clip       = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def get_anchors(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class Retinaface(object):
    _defaults = {
        "model_path": '../checkpoints/retinaface/Retinaface_mobilenet0.25.pth',
        "backbone": 'mobilenet',
        "confidence": 0.75,
        "nms_iou": 0.45,
        "input_shape": [1280, 1280, 3],
        "input_size": (1280, 1280),
        "letterbox_image": True,
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, device,model_path, **kwargs):
        self.__dict__.update(self._defaults)
        self.device = device
        self.model_path=model_path
        for name, value in kwargs.items():
            setattr(self, name, value)
        if self.backbone == "mobilenet":
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50
        if self.letterbox_image:
            self.anchors = Anchors(self.cfg, image_size=[self.input_shape[0], self.input_shape[1]]).get_anchors()
        self.generate()

    def generate(self):
        self.net = RetinaFace(cfg=self.cfg, mode='eval').eval()
        self.net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.net = self.net.eval()
        self.net = nn.DataParallel(self.net, device_ids=[self.device.index])
        self.net = self.net.to(self.device)


    def predict_on_batch(self, image: np.ndarray or torch.Tensor):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        assert image.shape[3] == 3
        assert image.shape[1] == self.input_shape[0]
        assert image.shape[2] == self.input_shape[1]
        image = image.float() - torch.Tensor((104, 117, 123))
        image = image.permute((0, 3, 1, 2))
        self.anchors = self.anchors.to(self.device)
        image = image.to(self.device)
        with torch.no_grad():
            loc, conf, landms = self.net(image)
        boxes = decode(loc, self.anchors, self.cfg['variance'])
        mask = conf[:, :, 1] >= self.confidence
        detections = []
        for i, _mask in enumerate(mask):
            _conf = conf.data[i][_mask][:, 1: 2]
            _boxes = boxes.data[i][_mask]
            keep = nms(_boxes, _conf[:, 0], 0.3)
            _detections = torch.cat([_boxes, _conf], -1)
            _detections = _detections[keep]
            _detections = _detections[:, [1, 0, 3, 2, 4]]
            detections.append(_detections)
        return detections

