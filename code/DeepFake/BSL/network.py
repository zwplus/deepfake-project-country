import torch
import torch.nn as nn
from torch.nn import functional as F

from .architectures import fornet


class BSXception(fornet.Xception):
    def __init__(self, num_class=1, is_train=False, is_bs_adv=False, is_rs_adv=False):
        super(BSXception, self).__init__(num_class)
        # self.up = nn.PixelShuffle(2)
        self.is_train = is_train
        self.is_bs_adv = is_bs_adv
        self.is_rs_adv = is_rs_adv
        if self.is_train:
            self.bs_conv14 = nn.Conv2d(728, 1, (1, 1), bias=False)
            self.bs_conv7 = nn.Conv2d(2048, 1, (1, 1), bias=False)
            self.rs_linear = nn.Linear(2048, 1)
            self.rs_conv7 = nn.Conv2d(2048, 2, (1, 1), bias=False)
            self.rs_relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor):
        out = {}
        # 224x224
        x = self.xception.conv1(x)
        x = self.xception.bn1(x)
        x = self.xception.relu1(x)
        # 32x112x112
        x = self.xception.conv2(x)
        x = self.xception.bn2(x)
        x = self.xception.relu2(x)
        # 64x112x112
        x = self.xception.block1(x)
        # 128x56x56
        x = self.xception.block2(x)
        # 256x28x28
        x = self.xception.block3(x)
        # 728x14x14
        x = self.xception.block4(x)
        x = self.xception.block5(x)
        x = self.xception.block6(x)
        x = self.xception.block7(x)
        x = self.xception.block8(x)
        x = self.xception.block9(x)
        x = self.xception.block10(x)
        x = self.xception.block11(x)

        if self.is_train:
            if self.is_bs_adv:
                bs1 = self.bs_conv14(x)
                bs1 = torch.flatten(bs1, 1)
                out['bs1'] = bs1

        x = self.xception.block12(x)
        # 1024x7x7
        x = self.xception.conv3(x)
        x = self.xception.bn3(x)
        x = self.xception.relu3(x)
        # 1536x7x7
        x = self.xception.conv4(x)
        x = self.xception.bn4(x)
        x = nn.ReLU(inplace=True)(x)
        # 2048x7x7

        if self.is_train:
            if self.is_bs_adv:
                bs2 = self.bs_conv7(x)
                bs2 = torch.flatten(bs2, 1)
                out['bs2'] = bs2

            if self.is_rs_adv:
                rsi = self.rs_conv7(x)
                rsi = self.rs_relu(rsi)
                rsi = torch.flatten(rsi, 1)
                out['rsi'] = rsi

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        if self.is_train and self.is_rs_adv:
            rs = self.rs_linear(x)
            out['rs'] = rs

        _out = self.xception.last_linear(x)
        out['out'] = _out
        return out

