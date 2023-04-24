import base64
import functools
import inspect
import io
import logging
import os
import string
import sys
import traceback
import uuid
from dataclasses import dataclass
from typing import ClassVar, Literal

import torch
import tqdm
from colorama import Back, Fore, Style, just_fix_windows_console
from diffusers import logging as diffusers_logging
from transformers import logging as transformers_logging
from twisted.web import resource
from twisted.web.resource import NoResource

from gyre import images
from gyre.cache import (
    CacheLookupError,
    TensorLRUCache_LockBase,
    TensorLRUCache_Spillover,
)
from gyre.constants import GB, MB, sd_cache_home

just_fix_windows_console()

# Capture stdout and stderr and turn anything printed to them
# into logger messages instead

original_stdout = sys.stdout
original_stderr = sys.stderr

tqdm_offsets = {}


class StdCapture:
    def __init__(self, std, level):
        self.std = std
        self.level = level

        self.line_buffer = ""

    def __getattr__(self, name):
        return getattr(self.std, name)

    def __enter__(self, *args, **kwargs):
        return self.std.__enter__(*args, **kwargs)

    def write(self, text):
        try:
            caller_module_name = sys._getframe(1).f_globals["__name__"]
        except Exception as e:
            caller_module_name = "gyre"

        self.line_buffer += text

        if "\n" in self.line_buffer:
            text = self.line_buffer.splitlines()

            if self.line_buffer.endswith("\n"):
                self.line_buffer = ""
            else:
                self.line_buffer = text.pop()

            logger = logging.getLogger(caller_module_name)
            for line in text:
                logger.log(self.level, line)


# Fix tqdm to print to original_stderr, so that any progress bars
# don't get converted into log messages

tqdm.tqdm.__init__ = functools.partialmethod(
    tqdm.tqdm.__init__, file=original_stderr, dynamic_ncols=True
)  # type: ignore


# And while we're at it, patch tqdm to track all in progress bars, so that
# we can manage interlacing tqdm and logging output

tqdm_in_progress = set()


def tqdm_monitor_refresh(self, *args, **kwargs):
    tqdm_in_progress.add(self)
    return orig_tqdm_refresh(self, *args, **kwargs)


def tqdm_monitor_close(self, *args, **kwargs):
    try:
        tqdm_in_progress.remove(self)
    except KeyError:
        pass

    return orig_tqdm_close(self, *args, **kwargs)


orig_tqdm_close, tqdm.tqdm.close = tqdm.tqdm.close, tqdm_monitor_close
orig_tqdm_refresh, tqdm.tqdm.refresh = tqdm.tqdm.refresh, tqdm_monitor_refresh


# Handle images either as the message or as some of the arguments via VisualRecord

log_cache = TensorLRUCache_Spillover(
    os.path.join(sd_cache_home, "log_cache"),
    memlimit=32 * MB,
    disklimit=1 * GB,
)


@dataclass
class CachedImage:
    image_key: str
    thumbnail: str  # base64-encoded png

    @property
    def url(self):
        return f"/log/{self.image_key}"

    def __str__(self):
        url = LogImagesController.get_url(self.image_key)
        if url:
            return f"[Image at {url}]"
        else:
            return "[Image]"


class VisualRecordFormatter(string.Formatter):
    def __init__(self, visual_record, mode: Literal["string", "fragments"] = "string"):
        self._visual_record = visual_record
        self._mode = mode

    # Taken from https://github.com/python/cpython/blob/main/Lib/string.py
    # Changes to return a list[string | CachedImage], rather than a single string
    def vformat(self, format_string, args, kwargs):
        result = []

        auto_arg_index = 0
        auto_arg_error_msg = (
            "cannot switch from manual field specification to automatic field numbering"
        )

        for literal_text, field_name, format_spec, conversion in self.parse(
            format_string
        ):
            if literal_text:
                result.append(literal_text)

            if field_name is not None:
                if field_name == "":
                    if auto_arg_index is False:
                        raise ValueError(auto_arg_error_msg)
                    field_name = str(auto_arg_index)
                    auto_arg_index += 1
                elif field_name.isdigit():
                    if auto_arg_index:
                        raise ValueError(auto_arg_error_msg)
                    auto_arg_index = False

                obj, _ = self.get_field(field_name, args, kwargs)
                obj = self._visual_record._translate_image(obj)

                if isinstance(obj, CachedImage):
                    result.append(obj)
                else:
                    obj = self.convert_field(obj, conversion)

                    format_spec, auto_arg_index = self._vformat(
                        format_spec, args, kwargs, set(), 1, auto_arg_index
                    )

                    result.append(self.format_field(obj, format_spec))

        if self._mode == "fragments":
            return result
        else:
            return "".join((str(x) for x in result))


class VisualRecord:
    def __init__(self, message, *args, **kwargs):
        self.message = self._translate_image(message)
        self.args = args
        self.kwargs = kwargs

    def _translate_image(self, arg):
        try:
            tensor = images.fromAuto(arg)
        except RuntimeError:
            return arg
        else:
            tensor = tensor.contiguous().to("cpu", torch.float32)

            key = str(uuid.uuid4())
            log_cache.set(key, {"image": tensor}, max_age=60 * 15)
            thumbnail = base64.b64encode(
                images.toPngBytes(images.rescale(tensor, 64, 64, "contain"))[0]
            ).decode("utf-8")

            return CachedImage(image_key=key, thumbnail=thumbnail)

    def __str__(self):
        if isinstance(self.message, CachedImage):
            return str(self.message)

        else:
            return VisualRecordFormatter(self, "string").format(
                self.message, *self.args, **self.kwargs
            )

    def as_fragments(self):
        if isinstance(self.message, CachedImage):
            return [{"thumbnail": self.message.thumbnail, "url": self.message.url}]

        formatted = VisualRecordFormatter(self, "fragments").format(
            self.message, *self.args, **self.kwargs
        )

        return [
            {"thumbnail": part.thumbnail, "url": part.url}
            if isinstance(part, CachedImage)
            else part
            for part in formatted
        ]


class LogImagesController(resource.Resource):
    log_host: ClassVar[str | None] = None

    def __init__(self, tensor: torch.Tensor | None = None):
        super().__init__()
        self.tensor = tensor

    def render_GET(self, request):
        if self.tensor is None:
            return NoResource().render(request)

        request.setHeader(b"Content-type", b"image/png")
        return bytes(images.toPngBytes(self.tensor)[0])

    def getChild(self, path, request):
        try:
            tensors, _ = log_cache.get(path.decode("utf-8"))
            return LogImagesController(tensors["image"])
        except CacheLookupError:
            pass

        return NoResource()

    def set_host_and_path(self, host_and_path):
        if host_and_path and host_and_path.endswith("/"):
            host_and_path = host_and_path[:-1]

        LogImagesController.host_and_path = host_and_path

    @classmethod
    def get_url(cls, key):
        if hasattr(cls, "host_and_path") and cls.host_and_path:
            return cls.host_and_path + "/" + key


class ColorFormatter(logging.Formatter):

    FORMATS = {
        logging.DEBUG: Fore.GREEN,
        logging.INFO: Fore.WHITE,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        record.color = self.FORMATS[record.levelno]
        record.reset = Style.RESET_ALL
        return super().format(record)


class StoreHandler(logging.Handler):
    def __init__(self):
        self.logs = []
        self.max_len = 1000

        super().__init__()

    def emit(self, record):
        if self.filter(record):
            base = dict(
                created=record.created, name=record.name, level=record.levelname
            )

            if isinstance(record.msg, VisualRecord):
                self.logs.append({**base, "fragments": record.msg.as_fragments()})
            else:
                self.logs.append({**base, "message": record.getMessage()})

            self.logs = self.logs[-self.max_len :]


class TqdmInterlacedStreamHandler(logging.StreamHandler):
    def emit(self, record):

        with tqdm.tqdm.get_lock():
            # Before we write anything, clear all current tqdms - this'll mean they
            # move below the log lines on next update.

            # Copy so changes don't affect us
            bars = set(tqdm_in_progress)

            for bar in bars:
                bar.clear(nolock=True)

            # Emit the line
            super().emit(record)

            # Redisplay the bars
            for bar in bars:
                bar.refresh(nolock=True)


stream_handler = TqdmInterlacedStreamHandler(original_stderr)
stream_handler.setFormatter(
    ColorFormatter(fmt="%(color)s%(name)-18.18s%(reset)s | %(message)s")
)

store_handler = StoreHandler()


def configure_logging():
    # Capture stdout and stderr
    sys.stdout = StdCapture(original_stdout, logging.INFO)
    sys.stderr = StdCapture(original_stderr, logging.ERROR)

    # Capture warnings
    logging.captureWarnings(True)

    # Default config is to not log anything except errors
    logging.basicConfig(level=logging.ERROR, handlers=[stream_handler])

    # Gyre config
    gyre_logger = logging.getLogger("gyre")

    gyre_logger.setLevel(logging.INFO)
    gyre_logger.propagate = False
    gyre_logger.addHandler(stream_handler)
    gyre_logger.addHandler(store_handler)

    # Override Transformers & Diffusers defaults
    diffusers_logging.disable_default_handler()
    diffusers_logging.enable_propagation()

    transformers_logging.disable_default_handler()
    transformers_logging.enable_propagation()


LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
