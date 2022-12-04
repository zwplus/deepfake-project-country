import sys
sys.path.append('./DeepFake/capsule')
import os
from PIL import Image
sys.setrecursionlimit(15000)
import torch
from torch.autograd import Variable
import torch.utils.data
import torchvision.transforms as transforms
import model22
from efficientnet_pytorch import EfficientNet

class Test:
    def __init__(self,path='../checkpoints/capsule/capsule_11.pt',imageSize=300,
                random=False,batchsize=16,
                device = torch.device('cuda:0')):
        self.imageSize=imageSize
        self.device=device
        self.random=random
        self.batchsize=batchsize
        self.vgg_ext = model22.VggExtractor()

        self.effi = EfficientNet.from_pretrained('efficientnet-b6')
        self.effi.to(self.device)
        self.capnet = model22.CapsuleNet(2)
        self.capnet = torch.nn.DataParallel(self.capnet, device_ids=[0])
        self.capnet.load_state_dict(torch.load(path))
        self.capnet.eval()
        self.capnet.to(self.device)
        self.transformer=transforms.Compose([
            transforms.Resize((self.imageSize, self.imageSize)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def detect(self, images):
        #传入的是一个list，list里是一张张图片
        # image = Image.open(path)
        # images=[]
        images=[self.transformer(img) for img in images]
        # resize = transforms.Resize((self.imageSize, self.imageSize))
        # image = resize(image)
        # trans = transforms.ToTensor()
        # image = trans(image)
        # nor = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # image = nor(image)
        images=torch.stack(images, dim=0)
        # img_data = torch.empty(self.batchsize, 3, self.imageSize, self.imageSize)
        # img_data[:images.shape[0]] = images
        images = images.to(self.device)
        input_v = Variable(images)

        classes, class_ = self.capnet(input_v, random=self.random)
        output_dis = class_.data.cpu()
        label_list=[]
        prob_list=[]
        for i in range(output_dis.shape[0]):
            if output_dis[i, 1] >= output_dis[i, 0]:
                label = 0
                prob = output_dis[i, 1].item()
                #label为0，置信度为越接近于0
                prob = 1-prob
            else:
                label = 1
                prob = output_dis[i, 0].item()
            label_list.append(label)
            prob_list.append(prob)
        return label_list, prob_list
