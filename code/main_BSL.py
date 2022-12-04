from concurrent import futures
from multiprocessing import Process
import grpc
from gRPC import xty_pb2
from gRPC import xty_pb2_grpc

from gRPC import Manager_Greeter
from FaceVeri import FaceVeri
from utils import log
log.init_log()
from loguru import logger

from server_config import *


def start_server(
    device_id, port, init_data, 
    facedetect_batchsize, facedetect_model_path, frames_per_video,
    deepfake_batchsize, deepfake_model_path
    ):
    logger.info('FaceDetection initing...')
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    logger.info('GPU devices use {}'.format(device_id))
    import torch
    from gRPC import Deepfake_Greeter
    logger.info('FaceDetection initing...')
    from utils import FaceExtractor
    from utils import VideoReader
    from FaceDetect import Retinaface
    facedet = Retinaface(torch.device('cuda:'+str(0)), facedetect_model_path)
    videoreader = VideoReader(verbose=False)
    video_read_fn = lambda x: videoreader.read_frames(x, num_frames=-1,second_num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn=video_read_fn, facedet=facedet, batch_size=facedetect_batchsize)

    faceveri=FaceVeri(device_id=0)

    logger.info('Deepfake model initing...')
    from DeepFake import imagedetect
    models = [imagedetect(
            path=deepfake_model_path,
            batchsize=deepfake_batchsize,
            device=torch.device('cuda:'+str(0))
        )]
    logger.info('xty_deepfake_njust capsule {} grpc server start...'.format(init_data['dataType']))
    greeter = Deepfake_Greeter(face_extractor,faceveri, models, '../data', init_data=init_data, batch_size=deepfake_batchsize)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    xty_pb2_grpc.add_EngineAPIServicer_to_server(greeter, server)
    server.add_insecure_port('[::]:{}'.format(port))
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    processes = []
    for data in image_servers_data:
        process = Process(target=start_server, kwargs=data)
        process.start()
        processes.append(process)
    for data in video_servers_data:
        process = Process(target=start_server, kwargs=data)
        process.start()
        processes.append(process)
    logger.info('xty_deepfake_njust capsule grpc server start...')
    image_ports = [data['port'] for data in image_servers_data]
    video_ports = [data['port'] for data in video_servers_data]
    greeter = Manager_Greeter(image_ports, video_ports)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
    xty_pb2_grpc.add_EngineAPIServicer_to_server(greeter, server)
    #绑定指令服务
    xty_pb2_grpc.add_IssueOrderServicer_to_server(greeter,server)

    server.add_insecure_port('[::]:5000')
    server.start()
    server.wait_for_termination()




