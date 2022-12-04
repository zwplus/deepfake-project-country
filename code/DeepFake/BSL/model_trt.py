import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize
from torch2trt import TRTModule
from .transformer import trans_test, letterbox

from loguru import logger


class Model:
    def __init__(self, batchsize, load_path):
        self.transformer = trans_test
        logger.info('Loading network ...')
        self.batchsize = batchsize
        self.net = TRTModule()
        self.net.load_state_dict(torch.load(load_path))
        self.inputs = torch.ones((self.batchsize, 3, 224, 224)).cuda()
        self.normalize = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True).cuda()
        logger.info('Loaded network.')

    def predict(self, imgs):
        assert len(imgs) <= self.batchsize
        elen = len(imgs)
        imgs = [torch.from_numpy(letterbox(np.array(img), (224, 224))[0]) for img in imgs]
        imgs = torch.stack(imgs, dim=0).permute((0, 3, 1, 2))
        self.inputs[: len(imgs)] = imgs
        self.inputs = self.normalize(self.inputs / 255)
        outputs = self.net(self.inputs)
        outputs = torch.sigmoid(outputs[: elen]).cpu().flatten().numpy()
        return outputs