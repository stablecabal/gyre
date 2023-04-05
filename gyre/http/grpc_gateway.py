import engines_pb2
import generation_pb2
import grpc
from google.protobuf import json_format as pb_json_format
from twisted.internet import reactor
from twisted.web import resource
from twisted.web.resource import NoResource
from twisted.web.server import NOT_DONE_YET

from gyre.http.grpc_gateway_controller import GRPCGatewayController


class GrpcGateway_EnginesController(GRPCGatewayController):
    input_class = engines_pb2.ListEnginesRequest

    def handle_BOTH(self, http_request, list_engines_request):
        return self.servicer.ListEngines(
            list_engines_request, http_request.grpc_context
        )


class GrpcGateway_GenerateController(GRPCGatewayController):
    input_class = generation_pb2.Request

    def handle_POST(self, http_request, generation_request):
        reactor.callInThread(self._generate, http_request, generation_request)
        return NOT_DONE_YET

    def _write_block(self, request, data: bytes):
        reactor.callFromThread(request.write, data)
        reactor.callFromThread(request.write, b"\n")

    def _error(self, request, code, message):
        # We can't set the http status code during a stream, so just write
        # the status message into the body
        status = grpc.Status(code=code, details=message)
        json_result = pb_json_format.MessageToJson(status).encode("utf-8")
        self._write_block(request, json_result)

    def _generate(self, http_request, generation_request):
        try:
            for result in self.servicer.Generate(
                generation_request, http_request.grpc_context
            ):
                json_result = pb_json_format.MessageToJson(result).encode("utf-8")
                self._write_block(http_request, json_result)
        except grpc.RpcError:
            self._error(
                http_request,
                http_request.grpc_context.code,
                http_request.grpc_context.message,
            )
        except BaseException:
            self._error(http_request, grpc.StatusCode.INTERNAL, "Internal Error")

        reactor.callFromThread(
            lambda: http_request.finish() if not http_request._disconnected else None
        )


class GrpcGateway_AsyncGenerateController(GRPCGatewayController):
    input_class = generation_pb2.Request

    def handle_POST(self, http_request, generation_request):
        return self.servicer.AsyncGenerate(
            generation_request, http_request.grpc_context
        )


class GrpcGateway_AsyncResultController(GRPCGatewayController):
    input_class = generation_pb2.AsyncHandle

    def handle_POST(self, http_request, async_handle):
        return self.servicer.AsyncResult(async_handle, http_request.grpc_context)


class GrpcGateway_AsyncCancelController(GRPCGatewayController):
    input_class = generation_pb2.AsyncHandle

    def handle_POST(self, http_request, async_handle):
        return self.servicer.AsyncCancel(async_handle, http_request.grpc_context)


class GrpcGatewayRouter(resource.Resource):
    def __init__(self):
        super().__init__()

        self.engines_bridge = GrpcGateway_EnginesController()
        self.generate_bridge = GrpcGateway_GenerateController()
        self.async_generate_bridge = GrpcGateway_AsyncGenerateController()
        self.async_result_bridge = GrpcGateway_AsyncResultController()
        self.async_cancel_bridge = GrpcGateway_AsyncCancelController()

    def getChild(self, path, request):
        if path == b"engines":
            return self.engines_bridge
        if path == b"generate":
            return self.generate_bridge
        if path == b"asyncGenerate":
            return self.async_generate_bridge
        if path == b"asyncResult":
            return self.async_result_bridge
        if path == b"asyncCancel":
            return self.async_cancel_bridge

        return NoResource()

    def render(self, request):
        return NoResource().render(request)

    def add_EnginesServiceServicer(self, engines_servicer):
        self.engines_bridge.add_servicer(engines_servicer)

    def add_GenerationServiceServicer(self, generation_servicer):
        self.generate_bridge.add_servicer(generation_servicer)
        self.async_generate_bridge.add_servicer(generation_servicer)
        self.async_result_bridge.add_servicer(generation_servicer)
        self.async_cancel_bridge.add_servicer(generation_servicer)
