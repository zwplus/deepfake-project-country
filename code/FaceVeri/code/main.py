import os
from PIL import Image
import torch
# from models.facenet import  InceptionResnetV1
# from  model import  Model
import numpy as np
from .models import InceptionResnetV1
from .model import Model
import traceback
# import sys
# sys.path.append('/root/workspace/engine_stable/outline/bsl/bsl/code/FaceVeri')

class faceveri:
    def __init__(self,load_path='./FaceVeri/checkpoints/facenet1.pt',
                dataset='./FaceVeri/dataset',
                device_id=0):
        self.td_device=torch.device('cuda:{}'.format(device_id))
        self.td_checkpoints = [
            [160, load_path, InceptionResnetV1],
        ]
        self.td_model=Model(self.td_checkpoints,self.td_device)
        self.td_features_dir={}    #空字典
        self.target_id=['0','1','2','3','4','8','9','10']
        self.gen_features(dataset)
        
    
    def gen_features(self,dataset='./bsl/code/FaceVeri/dataset'):
        #构建当前特定人识别的特定人字典
        with torch.no_grad():
            for i in os.listdir(dataset):
                if i in self.target_id:   #只生成在待测特定人列表里的特定人特征
                    td_image_dir=os.path.join(dataset,i)
                    td_image=[Image.open(os.path.join(td_image_dir,j)) for j in os.listdir(td_image_dir)]
                    td_features=self.td_model.predict(td_image)
                    td_index=i.zfill(3)
                    self.td_features_dir[td_index]=td_features

    #注意传入的images一定是list,images_pos一定是list，不然会出bug
    def get_td_id(self,images,images_pos):
        td_id='000'
        try:
            with torch.no_grad():
                td_features = self.td_model.predict(images)
            frames_td_id=[]
            for i in np.unique(images_pos):
                frame_feature=td_features[images_pos==i]
                frame_td_conf=[]
                frame_td_id=[]
                for j in frame_feature:
                    temp_list={}
                    for td_id in self.td_features_dir.keys():
                        temp_list[td_id]=torch.max(torch.sum(j*self.td_features_dir[td_id],dim=-1))
                    predict_id=max(temp_list,key=lambda x:temp_list[x])
                    predict_conf=temp_list[predict_id]
                    frame_td_id.append(predict_id)
                    frame_td_conf.append(predict_conf)
                max_conf=max(frame_td_conf)
                if max_conf>=0.8:
                    frame_td=frame_td_id[frame_td_conf.index(max(frame_td_conf))]
                else:
                    frame_td='000'
                frames_td_id.append(frame_td)
            td_id=max(frames_td_id,key=frames_td_id.count)
        except Exception as e:
            print(traceback.print_exc())
        return td_id
    
    def update_fearures_dir(self,method=''):
        #增加特定，减少特定人，修改特定人,用来进行后续更新
        pass






