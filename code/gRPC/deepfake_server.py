import os
import time
import json
from loguru import logger
from urllib.request import urlopen
import traceback
import datetime
import hashlib
from PIL import Image
import tempfile
import numpy as np
from gRPC import xty_pb2
from gRPC import xty_pb2_grpc
import torch



#下载文件，返回二进制数据
def download_file(fileUri, fmd5, filepath=None):
    byte_data = None
    dmd5 = ''
    for i in range(3):
        try:
            byte_data = urlopen(fileUri).read()
        except:
            logger.error('download failed, try again...')
        else:
            dmd5 = hashlib.md5(byte_data).hexdigest()
            if dmd5.lower() != fmd5.lower():
                logger.error('file md5 error, try again...')
            else:
                break
    assert dmd5.lower() == fmd5.lower()
    # if filepath:
    if False:
        #需要设置保存的上限，不能超过保存
        logger.info('file save {}'.format(filepath))
        with open(filepath, 'wb') as file:
            file.write(byte_data)
    return byte_data


class Deepfake_Greeter(xty_pb2_grpc.EngineAPIServicer):
    # def __init__(self,face_extractor,faceveri, models, save_path, save_file=True, init_data=None,batch_size=32):
    def __init__(self,face_extractor,faceveri, models, save_path, save_file=False, init_data=None,batch_size=32):
        super(Deepfake_Greeter, self).__init__()
        self.save_path = save_path
        self.save_file = save_file
        self.face_extractor = face_extractor
        self.models = models
        self.batch_size = batch_size
        self.faceveri=faceveri

        if init_data:
            images, images_pos = self.face_detect(init_data['dataType'], init_data['filepath'])
            images=[Image.fromarray(image).convert('RGB') for image in images ]
            frame_result,frames_conf = self.deepfake_detect(images, images_pos,init_data['dataType'])
            fake_num = np.sum(frames_conf > 0.7)
            logger.info('fake frame num: {} all frame num: {}'.format(fake_num, len(frames_conf)))


    def engineApi(self, _request, context):
        logger.info('Get request: {}'.format(_request.message))
        save_path = os.path.join(self.save_path, datetime.datetime.now().strftime('%Y%m%d'))
        if not os.path.isdir(save_path):
            logger.info('create dir {}'.format(save_path))
            os.makedirs(save_path)

        request = json.loads(_request.message)
        fileUri = request['fileUri']
        filename = '{}_{}'.format(int(time.time() * 1000) % 100000, os.path.split(fileUri)[-1])
        filepath = os.path.join(save_path, filename) if self.save_file else False
        dataType = request['dataType']
        filemd5 = request['md5']

        #数据无法正常检测时，默认结果
        code = 1
        msg = 'false'
        engineResult = 0
        faceNum = 0
        engineConf = 0
        engineName = 'cap'
        targetPerson=''
        try:
            with torch.no_grad():
                logger.info('downloading file...')
                byte_data = download_file(fileUri, filemd5, filepath)

                #起始时间
                startTs = int(round(time.time()*1000))

                with tempfile.NamedTemporaryFile() as temp:
                    temp.write(byte_data)
                    images, images_pos = self.face_detect(dataType, temp.name)
                assert len(images) > 0 ,"no_face"

                images=[Image.fromarray(image).convert('RGB') for image in images ]

                targetPerson=self.faceveri.get_td_id(images,images_pos)

                faceNum = np.max(np.bincount(images_pos))

                # 提取的视频帧的置信度一维数组
                frame_result,frames_conf = self.deepfake_detect(images, images_pos,dataType)
                if frame_result.mean()>0.3:
                    engineResult=1
                else:
                    engineResult=0
                engineConf=float(frames_conf.mean())

                #记录结束时间
                endTs = int(round(time.time()*1000))
                logger.info(str(endTs-startTs),result='True')

                logger.info('engineResult: {} engineConf: {}'.format(engineResult, engineConf))
                code = 0
                msg = 'success'
        except AssertionError as a: #人脸数目为0时返回信息
            msg='no_face'
        except Exception as e:
            logger.error(traceback.format_exc())

        data = {
            'engineName': engineName,
            'engineResult': int(engineResult),
            'faceNum': int(faceNum),
            'targetPerson': '' if targetPerson =='000' else targetPerson,
            'engineConf': float(engineConf),
            'timestamp': time.time(),
        }
        logger.info('Reply: code:{} msg:{} data:{}'.format(code, msg, json.dumps(data)))

        return  xty_pb2.Reply(
            code=code, msg=msg, data=json.dumps(data)
        )

    def face_detect(self, dataType, filepath):
        images = []
        images_pos = []
        logger.info('face detecting...')
        if dataType == 'video':
            faces = self.face_extractor.process_video(filepath)
            for i, _face in enumerate(faces):
                for face in _face['faces']:
                    image = face
                    #判断人脸是否大于40
                    if image.shape[0]>40 or image.shape[1]>40:
                        images.append(image)
                        images_pos.append(i)   
        elif dataType == 'image':
            faces = self.face_extractor.process_image(filepath)
            for face in faces['faces']:
                image = face
                #判断人脸是否大于40
                if image.shape[0]>40 or image.shape[1]>40:
                    images.append(image)
                    images_pos.append(0)
                    
                
        images_pos = np.array(images_pos)
        logger.info('find face num: {}'.format(len(images)))
        return images, images_pos

    def deepfake_detect(self,images,images_pos,dataType):
        logger.info('deepfake detecting...')
        frame_result = []
        frame_conf = []
        if dataType=='video':
            for i in range(0,len(images),self.batch_size):
                _results,_confs=self.models[0].detect(images[i:i+self.batch_size])
                frame_result.append(_results)
                frame_conf.append(_confs)
        elif dataType=='image':
            for i in range(0,len(images),self.batch_size):
                _results,_confs=self.models[0].detect(images[i:i+self.batch_size])
                frame_result.append(_results)
                frame_conf.append(_confs)
        frame_result=np.concatenate(frame_result)
        frame_conf=np.concatenate(frame_conf)
        return frame_result,frame_conf






