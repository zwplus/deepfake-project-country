import json
import os
import random
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
from torchvision import transforms


class Net(torch.nn.Module):
    def __init__(self, size, checkpoint, net, device, is_norm=True):
        super(Net, self).__init__()
        self.net = net().to(device)
        self.size = size
        self.is_norm = is_norm
        if checkpoint:
            model_dict = self.net.state_dict()
            weight = torch.load(checkpoint, map_location=device)
            if 'state_dict' in weight.keys():
                weight = weight['state_dict']
                new_weight = {}
                for k in model_dict:
                    if 'backbone.' + k in weight.keys():
                        new_weight[k] = weight['backbone.' + k]
                weight = new_weight
            self.net.load_state_dict(weight, strict=False)

    def forward(self, x):
        x = F.interpolate(x, (self.size, self.size), mode='nearest')
        feature = self.net(x)
        if self.is_norm:
            feature = F.normalize(feature, p=2, dim=-1)
        return feature


transformer_val = transforms.Compose([
    transforms.Resize((224, 224), Image.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


class Model:
    def __init__(self, checkpoints, device):
        self.device = device
        nets = []
        for _cp in checkpoints:
            size, checkpoint, _net = _cp
            net = Net(size, checkpoint, _net, device).to(device)
            net.eval()
            net.net.eval()
            for param in net.net.parameters():
                param.requires_grad = False
            nets.append(net)
        self.nets = nets

    def predict(self, images):
        images = [transformer_val(image.convert('RGB')) for image in images]
        inputs = torch.stack(images, dim=0)
        inputs = inputs.to(self.device)
        outputs = self.nets[0](inputs)
        return outputs

