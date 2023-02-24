import inspect
import io
import logging
import sys
import traceback

from colorama import Back, Fore, Style, just_fix_windows_console

just_fix_windows_console()

original_stdout = sys.stdout
original_stderr = sys.stderr


class StdCapture:
    def __init__(self, std, level):
        self.std = std
        self.level = level

    def __getattr__(self, name):
        return getattr(self.std, name)

    def __enter__(self, *args, **kwargs):
        return self.std.__enter__(*args, **kwargs)

    def write(self, text):
        caller_module_name = sys._getframe(1).f_globals["__name__"]

        if caller_module_name.startswith("tqdm."):
            self.std.write(text)

        else:
            text = text.splitlines()
            logger = logging.getLogger(caller_module_name)

            for line in text:
                if line:
                    logger.log(self.level, line)


sys.stdout = StdCapture(original_stdout, logging.INFO)
sys.stderr = StdCapture(original_stderr, logging.ERROR)


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
            self.logs.append(
                dict(created=record.created, name=record.name, message=record.message)
            )

            self.logs = self.logs[-self.max_len :]


root_handler = logging.StreamHandler(original_stderr)
root_handler.setFormatter(
    ColorFormatter(fmt="%(color)s%(name)-18.18s%(reset)s | %(message)s")
)

stream_handler = logging.StreamHandler(original_stderr)
stream_handler.setFormatter(
    ColorFormatter(fmt="%(color)s%(name)-18.18s%(reset)s | %(message)s")
)

store_handler = StoreHandler()


def configure_logging():

    # Capture warnings
    logging.captureWarnings(True)

    # Default config is to not log anything except errors
    logging.basicConfig(level=logging.ERROR, handlers=[root_handler])

    # Gyre config
    gyre_logger = logging.getLogger("gyre")

    gyre_logger.setLevel(logging.INFO)
    gyre_logger.propagate = False
    gyre_logger.addHandler(stream_handler)
    gyre_logger.addHandler(store_handler)


LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
