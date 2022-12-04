import json
import time
from urllib.request import urlopen
import hashlib
from multiprocessing import Pool
import grpc
from gRPC import xty_pb2
from gRPC import xty_pb2_grpc
import os

def run(file,file_type):
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    fileUri = 'http://127.0.0.1:8001/'+file
    print(fileUri)
    fcont = urlopen(fileUri).read()
    md5 = hashlib.md5(fcont).hexdigest()

    request = {
        "fileUri": fileUri,
        "md5": md5,
        "dataType": file_type,
        "euAddr": "10.1.10.1:15001",
        "result": 0,
        "metaData": {},
        "timestamp": 1626334986.8592472,
        "imageStrategyOrder":"01",
        "videoStrategyOrder":"02",
    }

    try:
        start_time = time.time()
        with grpc.insecure_channel('localhost:5000') as channel:
            stub = xty_pb2_grpc.EngineAPIStub(channel)
            response = stub.engineApi(xty_pb2.Request(message=json.dumps(request)))
        print("Greeter client received: ", response.code, response.msg, response.data)
        end_time = time.time()
        print(end_time - start_time)
    except Exception as e:
        print(e)

def run_order():
    request={ 
        "province":"34", 
        "city":"26", 
        "operator":"M",
        "machineRoom":"01", 
        "taskId":"1324127394234123", 
        "orderNo":"01", 
        "order":"CREATE",
        "timestamp":1234567891234,
        "orderParam":{ 
            "org":"XF", 
            "ips":"127.0.0.1",
            "targets":[ 
                { "targetNo":"01", 
                    "featureNo":"01", 
                "url":"" 
                }
            ] 
        }
    }
    with grpc.insecure_channel('localhost:27005') as channel:
        stub = xty_pb2_grpc.IssueOrderStub(channel)

        r=xty_pb2.OrderRequest(message=json.dumps(request))
        response=stub.EngineIssueOrder(r)
    print("Greeter client received: ", response.code, response.msg)

if __name__ == '__main__':
    pool = Pool(1)

    start_time = time.time()
    # for i in range(1):
    #     pool.apply_async(run)
    # pool.close()
    # pool.join()
    for i in os.listdir('../data/test'):
        if os.path.splitext(i)[-1] == '.png':
            file_type='image'
        else:
            file_type='video'
        
        file=i

        run(file,file_type)


    end_time = time.time()
    print(end_time - start_time)


