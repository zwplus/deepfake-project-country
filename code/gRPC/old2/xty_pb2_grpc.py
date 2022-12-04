# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import xty_pb2 as xty__pb2


class EngineAPIStub(object):
    """引擎调度接口
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.EngineApi = channel.unary_unary(
                '/protos.EngineAPI/EngineApi',
                request_serializer=xty__pb2.Request.SerializeToString,
                response_deserializer=xty__pb2.Reply.FromString,
                )


class EngineAPIServicer(object):
    """引擎调度接口
    """

    def EngineApi(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_EngineAPIServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'EngineApi': grpc.unary_unary_rpc_method_handler(
                    servicer.EngineApi,
                    request_deserializer=xty__pb2.Request.FromString,
                    response_serializer=xty__pb2.Reply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'protos.EngineAPI', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class EngineAPI(object):
    """引擎调度接口
    """

    @staticmethod
    def EngineApi(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/protos.EngineAPI/EngineApi',
            xty__pb2.Request.SerializeToString,
            xty__pb2.Reply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class IssueOrderStub(object):
    """下发指令接口
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.EngineIssueOrder = channel.unary_unary(
                '/protos.IssueOrder/EngineIssueOrder',
                request_serializer=xty__pb2.OrderRequest.SerializeToString,
                response_deserializer=xty__pb2.OrderReply.FromString,
                )


class IssueOrderServicer(object):
    """下发指令接口
    """

    def EngineIssueOrder(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_IssueOrderServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'EngineIssueOrder': grpc.unary_unary_rpc_method_handler(
                    servicer.EngineIssueOrder,
                    request_deserializer=xty__pb2.OrderRequest.FromString,
                    response_serializer=xty__pb2.OrderReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'protos.IssueOrder', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class IssueOrder(object):
    """下发指令接口
    """

    @staticmethod
    def EngineIssueOrder(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/protos.IssueOrder/EngineIssueOrder',
            xty__pb2.OrderRequest.SerializeToString,
            xty__pb2.OrderReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
