import torch

from .network import BSXception
from .transformer import trans_test

from loguru import logger

class Model:
    def __init__(self, load_path, device='cuda:0'):
        self.device = torch.device(device)
        self.transformer = trans_test

        logger.info('Loading network ...')
        net = BSXception(num_class=1, is_train=False, is_bs_adv=False, is_rs_adv=False)
        self.net = net.eval().to(self.device)
        checkpoint = torch.load(load_path, map_location=self.device)
        model_dict = checkpoint['net']
        self.net.load_state_dict(model_dict, strict=False)
        logger.info('Loaded network ...')
    def predict(self, imgs):
        imgs = [self.transformer(img) for img in imgs]
        imgs = torch.stack(imgs, dim=0)
        imgs = imgs.to(self.device)
        with torch.no_grad():
            outputs = self.net(imgs)
        outputs = torch.sigmoid(outputs['out']).cpu().flatten().numpy()
        return outputs









