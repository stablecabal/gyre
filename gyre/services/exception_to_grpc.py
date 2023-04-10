import inspect
import logging
import os
import re
import traceback

import grpc

from gyre.constants import IS_DEV

return_traceback = IS_DEV

logger = logging.getLogger(__name__)


def _handle_exception(func, e, context, mappings):
    details = [f"Exception in handler {func.__name__}. "]

    for block in traceback.format_exception(e):
        details.append(block)

    details = "".join(details)

    code, message = grpc.StatusCode.INTERNAL, "Internal Error"
    for exception_class, grpc_code in mappings.items():
        if isinstance(e, exception_class):
            if callable(grpc_code):
                code, message, details = grpc_code(e, details)
            else:
                code, message = grpc_code, str(e)
            break

    logger.error(details)
    context.abort(code, details if return_traceback else message)


def _exception_to_grpc_generator(func, mappings):
    def wrapper(*args, **kwargs):
        if "context" in kwargs:
            context = kwargs["context"]
        else:
            context = args[-1]

        try:
            yield from func(*args, **kwargs)
        except grpc.RpcError as e:
            # Allow grpc / whatever-called-Servicer to receive RpcError
            raise e
        except Exception as e:
            # Pass through any errors raised by context.abort (why doesn't it use RpcError? Good question!)
            if type(e) is Exception and context.code() is not None:
                raise e

            _handle_exception(func, e, context, mappings)

    return wrapper


def _exception_to_grpc_unary(func, mappings):
    def wrapper(*args, **kwargs):
        if "context" in kwargs:
            context = kwargs["context"]
        else:
            context = args[-1]

        try:
            return func(*args, **kwargs)
        except grpc.RpcError as e:
            # Allow grpc / whatever-called-Servicer to receive RpcError
            raise e
        except Exception as e:
            # Pass through any errors raised by context.abort (same passive-agressive moan as above :p)
            if type(e) is Exception and context.code() is not None:
                raise e

            _handle_exception(func, e, context, mappings)

    return wrapper


def exception_to_grpc(mapping):
    def decorator(func):
        if inspect.isgeneratorfunction(func):
            return _exception_to_grpc_generator(func, mapping)
        else:
            return _exception_to_grpc_unary(func, mapping)

    if callable(mapping):
        func, mapping = mapping, {}
        return decorator(func)
    else:
        return decorator
