import json
import traceback
import uuid

from accept_types import get_best_match
from twisted.internet import reactor
from twisted.web.resource import Resource
from twisted.web.server import NOT_DONE_YET, Request

# Where there is a commented version, that's the HTTP version. We
# use the "Stability.ai" version.
ERROR_CODES = {
    400: "bad_request",
    401: "unauthorized",
    402: "payment_required",
    403: "permission_denied",  # "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    406: "not_acceptable",
    407: "proxy_authentication_required",
    408: "request_timeout",
    409: "conflict",
    410: "gone",
    411: "length_required",
    412: "precondition_failed",
    413: "payload_too_large",
    414: "uri_too_long",
    415: "unsupported_media_type",
    416: "range_not_satisfiable",
    417: "expectation_failed",
    418: "im_a_teapot",
    421: "misdirected_request",
    422: "unprocessable_content",
    423: "locked",
    424: "failed_dependancy",
    425: "too_early",
    426: "upgrade_required",
    428: "precondition_required",
    429: "too_many_requests",
    431: "request_header_fields_too_large",
    451: "unavailable_for_legal_reasons",
    500: "server_error",  # "internal_server_error"
    503: "service_unavailable",
}


class JSONError(Resource, Exception):
    def __init__(
        self, status: int, detail: str = "No details", brief: str | None = None
    ):
        super(Resource).__init__()
        super(Exception).__init__()

        if brief is None:
            brief = ERROR_CODES.get(status, "unknown_error")

        self.code = status
        self.brief = brief
        self.detail = detail

    def render(self, request: Request) -> bytes:
        request.setResponseCode(self.code)
        request.setHeader(b"content-type", b"application/json")
        request.write(
            json.dumps(
                {"id": str(uuid.uuid4()), "name": self.brief, "messsage": self.detail}
            ).encode("utf-8")
        )
        return b""

    def getChild(self, chnam, request):
        return self


class NotAcceptable(JSONError):
    def __init__(
        self, message="Sorry, Accept header does not match a type we can serve"
    ):
        super().__init__(416, message)


class UnsupportedMediaType(JSONError):
    def __init__(
        self, message="Sorry, Content-Type header does not match a type we can process"
    ):
        super().__init__(415, message)


class JSONAPIController(Resource):
    preferred_return_type = "application/json"
    return_types = {"application/json"}
    cors = True

    def _cors_headers(self, request, is_preflight=False):
        if not self.cors:
            return

        request.setHeader(
            "Access-Control-Allow-Origin", "*" if self.cors is True else self.cors
        )

        if is_preflight:
            request.setHeader(
                "Access-Control-Allow-Methods",
                "GET, POST, OPTIONS",
            )
            request.setHeader(
                "Access-Control-Allow-Headers",
                request.getHeader("Access-Control-Request-Headers") or "*",
            )

    def decode_GET(self, request: Request):
        return None

    def decode_POST(self, request: Request):
        content_type_header = request.getHeader("content-type")
        if not content_type_header or content_type_header != "application/json":
            raise UnsupportedMediaType()

        return json.load(request.content)

    def encode_application_json(self, request: Request, result) -> bytes:
        request.setHeader("content-type", "application/json")
        return json.dumps(result).encode("utf-8")

    def convert_exception(self, request: Request, exception) -> JSONError:
        if isinstance(exception, ValueError):
            return JSONError(400, str(exception))

        # Log the exception locally
        details = [f"Exception in JSON controller {self.__class__.__name__}. "]
        for block in traceback.format_exception(exception):
            details.append(block)
        print("".join(details))

        return JSONError(500, str(exception))

    def _get_encoder(self, request):
        # Calculate what (if any) return type matches the accept header
        if accept_header := request.getHeader("accept"):
            return_type = get_best_match(
                accept_header,
                [self.preferred_return_type] + list(self.return_types),
            )
        else:
            return_type = self.preferred_return_type

        # Find the encoder for the return type
        if return_type:
            encoder = getattr(
                self, "encode_" + return_type.lower().replace("/", "_"), None
            )
        else:
            encoder = None

        # If no encoder, raise error
        if not encoder or not callable(encoder):
            raise NotAcceptable()

        # Otherwise reset the request accept header to just the return type
        request.requestHeaders.setRawHeaders("accept", [return_type])

        return encoder, return_type

    def _write_result(self, request, encoder, return_type, result, error):
        try:
            if error:
                raise error

            encoded = encoder(request, result)
            # If encoded is a str, turn it into utf-8-encoded bytes
            if isinstance(encoded, str):
                encoded = encoded.encode("utf-8")
                return_type = return_type + "; charset=utf-8"
            request.setHeader("content-type", return_type)
            request.write(encoded)

        except JSONError as e:
            e.render(request)
        except Exception as e:
            self.convert_exception(request, e).render(request)
        finally:
            request.finish()

    def _render_common(self, request, decoder, handler):
        result = error = encoder = return_type = None

        try:
            # Set CORS headers. Do first, so client can read any error
            self._cors_headers(request)

            # If no handler for this method, return method_not_allowed
            if handler is None:
                raise JSONError(405, "This method is not allowed")

            # Get the encoder. Do this before processing the request, to
            # avoid doing work if Accept type is unacceptable
            encoder, return_type = self._get_encoder(request)

            # Decode the request data
            data = decoder(request)

            # Calculate the actual response
            result = handler(request, data)

            # Handle having a callback returned which should be called in a thread
            if callable(result):

                def next(result=None, error=None):
                    reactor.callFromThread(
                        self._write_result, request, encoder, return_type, result, error
                    )

                reactor.callInThread(result, next)
                result = NOT_DONE_YET

        except Exception as e:
            error = e

        if result is not NOT_DONE_YET:
            self._write_result(request, encoder, return_type, result, error)

        return NOT_DONE_YET

    def render_GET(self, request):
        handler = getattr(self, "handle_GET", getattr(self, "handle_BOTH", None))
        return self._render_common(request, self.decode_GET, handler)

    def render_POST(self, request):
        handler = getattr(self, "handle_POST", getattr(self, "handle_BOTH", None))
        return self._render_common(request, self.decode_POST, handler)

    def render_OPTIONS(self, request):
        self._cors_headers(request, is_preflight=True)
        request.setResponseCode(204)
        return b""

    def getChild(self, path, request):
        return self
