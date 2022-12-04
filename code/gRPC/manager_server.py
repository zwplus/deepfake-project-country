import json
import traceback
import sys
from urllib import request
sys.path.append('./code/gRPC')
import os
import grpc
from gRPC import xty_pb2
from gRPC import xty_pb2_grpc
from loguru import logger




class Manager_Greeter(xty_pb2_grpc.EngineAPIServicer,xty_pb2_grpc.IssueOrderServicer):
    def __init__(self, image_ports, video_ports):
        xty_pb2_grpc.EngineAPI.__init__(self)
        xty_pb2_grpc.IssueOrderServicer.__init__(self)
        self.image_ports = image_ports
        self.video_ports = video_ports
        self.image_index = 0
        self.video_index = 0

    def engineApi(self, _request, context):

        #是否需要定义一个默认值即在except部分定义个reply？


        try:
            logger.info('Get request: {}'.format(_request.message))
            _str = '{}'.format(_request.message)
            _str = _str.replace('{', '{{')
            _str = _str.replace('}', '}}')
            #loguru中会将关键字参数自动加入record的extra字段中record{extra:{request:True}}
            logger.info(_str, request=True)
            request = json.loads(_request.message)
            dataType = request['dataType']
            if dataType == 'image':
                with grpc.insecure_channel('127.0.0.1:{}'.format(self.image_ports[self.image_index])) as channel:
                    self.image_index = (self.image_index + 1) % len(self.image_ports)
                    stub = xty_pb2_grpc.EngineAPIStub(channel)
                    response = stub.engineApi(xty_pb2.Request(message=_request.message))
            elif dataType == 'video':
                with grpc.insecure_channel('127.0.0.1:{}'.format(self.video_ports[self.video_index])) as channel:
                    self.video_index = (self.video_index + 1) % len(self.video_ports)
                    stub = xty_pb2_grpc.EngineAPIStub(channel)
                    response = stub.engineApi(xty_pb2.Request(message=_request.message))
            logger.info('Reply: code:{} msg:{} data:{}'.format(response.code, response.msg, response.data))
            reply = xty_pb2.Reply(
                code=response.code, msg=response.msg, data=response.data
            )
        except Exception as e:
            logger.error(traceback.format_exc())
        return reply
    
    def EngineIssueOrder(self, _request, context):
       try:
           logger.info('Get order: {}'.format(_request.message))
           _str = '{}'.format(_request.message)
           _str = _str.replace('{', '{{')
           _str = _str.replace('}', '}}')
           logger.info(_str, order=True)

           request = json.loads(_request.message)
           # 根据收到message信息做出相应响应
           orderNo = request['orderNo']
           if orderNo == '01':
               # 特定人识别注册信息指令
               print(orderNo)
           elif orderNo == '02':
               # 模型更新指令
               print(orderNo)

           # 这里用来测试
           code = 0
           msg = 'success'
           logger.info('Reply: code:{} msg:{}'.format(code, msg))
           reply = xty_pb2.OrderReply(code=code, msg=msg)
       except Exception as e:
           logger.error(traceback.format_exc())
       return reply

        


