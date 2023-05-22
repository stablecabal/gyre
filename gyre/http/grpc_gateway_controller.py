from typing import Callable, cast

import grpc
from google.protobuf import json_format as pb_json_format
from twisted.web.server import Request

from gyre.http.json_api_controller import JSONAPIController, JSONError

GRPC_HTTP_CODES = {
    grpc.StatusCode.OK: 200,
    grpc.StatusCode.CANCELLED: 499,
    grpc.StatusCode.UNKNOWN: 500,
    grpc.StatusCode.INVALID_ARGUMENT: 400,
    grpc.StatusCode.DEADLINE_EXCEEDED: 504,
    grpc.StatusCode.NOT_FOUND: 404,
    grpc.StatusCode.ALREADY_EXISTS: 409,
    grpc.StatusCode.PERMISSION_DENIED: 403,
    grpc.StatusCode.UNAUTHENTICATED: 401,
    grpc.StatusCode.RESOURCE_EXHAUSTED: 429,
    grpc.StatusCode.FAILED_PRECONDITION: 400,
    grpc.StatusCode.ABORTED: 409,
    grpc.StatusCode.OUT_OF_RANGE: 400,
    grpc.StatusCode.UNIMPLEMENTED: 501,
    grpc.StatusCode.INTERNAL: 500,
    grpc.StatusCode.UNAVAILABLE: 503,
    grpc.StatusCode.DATA_LOSS: 500,
}


class GRPCContext:
    def __init__(self, request):
        self.request = request
        self.code = grpc.StatusCode.OK
        self.message = "OK"
        self.cancel_callback: Callable | None = None

        self.request.notifyFinish().addErrback(self._finishError)

    def _finishError(self, *args):
        print(*args)

        if self.cancel_callback:
            self.cancel_callback()

    def add_callback(self, callback):
        self.cancel_callback = callback

    def set_code(self, code):
        self.code = code

    @property
    def http_code(self):
        return GRPC_HTTP_CODES[self.code]

    def set_details(self, message):
        self.message = message

    @property
    def http_message(self):
        return self.message

    def abort(self, code, message):
        if code == grpc.StatusCode.OK:
            raise ValueError("Abort called with OK as status code")

        self.set_code(code)
        self.set_details(message)
        raise grpc.RpcError()

    def invocation_metadata(self):
        return []


class RequestWithContext(Request):
    grpc_context: GRPCContext


class GRPCServiceBridgeController(JSONAPIController):
    def __init__(self):
        super().__init__()
        self._servicer = None

    def add_servicer(self, servicer):
        self._servicer = servicer

    @property
    def servicer(self):
        if not self._servicer:
            raise JSONError(503, "Not ready yet")

        return self._servicer

    def encode_application_json(self, request: RequestWithContext, result):
        if request.grpc_context.code != grpc.StatusCode.OK:
            raise grpc.RpcError()

        return super().encode_application_json(request, result)

    def convert_exception(self, request: RequestWithContext, exception):
        if isinstance(exception, grpc.RpcError):
            return JSONError(
                request.grpc_context.http_code, request.grpc_context.http_message
            )

        return super().convert_exception(request, exception)

    def _render_common(self, request, decoder, handler):
        request.grpc_context = GRPCContext(request)
        return super()._render_common(request, decoder, handler)


class GRPCGatewayController(GRPCServiceBridgeController):
    input_class = None

    def _create_input_class(self, data):
        if not self.input_class:
            return data

        param = self.input_class()
        if data:
            pb_json_format.ParseDict(data, param, ignore_unknown_fields=True)
        return param

    def decode_GET(self, request):
        data = super().decode_GET(request)
        return self._create_input_class(data)

    def decode_POST(self, request):
        data = super().decode_POST(request)
        return self._create_input_class(data)

    def encode_application_json(self, request, result):
        if request.grpc_context.code != grpc.StatusCode.OK:
            raise grpc.RpcError()

        return pb_json_format.MessageToJson(result)
