"""Application logging format and dependency routing."""

import inspect
import json
import logging
import logging.config
import os
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

from huggingface_hub.utils import logging as hf_logging


class DebugLogFormatter(logging.Formatter):
    """Format source-aware logs in UTC+8, with colors on terminals."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return datetime.fromtimestamp(
            record.created, timezone(timedelta(hours=8))
        ).strftime(datefmt or "%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        first_line, separator, remaining = message.partition("\n")
        method_name = record.funcName
        frame = inspect.currentframe()
        try:
            while frame is not None:
                if (
                    frame.f_code.co_filename == record.pathname
                    and frame.f_code.co_name == record.funcName
                ):
                    if sys.version_info >= (3, 11):
                        method_name = frame.f_code.co_qualname
                    elif "self" in frame.f_locals:
                        method_name = f"{type(frame.f_locals['self']).__qualname__}.{record.funcName}"
                    else:
                        pass
                    break
                else:
                    frame = frame.f_back
        finally:
            del frame
        level_color = {
            logging.INFO: "\033[36m",
            logging.WARNING: "\033[33m",
            logging.ERROR: "\033[31m",
            logging.CRITICAL: "\033[31m",
        }.get(record.levelno, "")
        lines = [
            (level_color, f"[{self.formatTime(record)}] {record.levelname}"),
            ("\033[34m", f"  {record.filename}:{record.lineno}"),
            ("\033[35m", f"    方法：{method_name}()"),
            ("\033[32m", f"      内容：{first_line}"),
        ]
        if separator:
            lines.extend(
                ("\033[32m", line)
                for line in textwrap.indent(
                    textwrap.dedent(remaining), "        "
                ).splitlines()
            )
        else:
            pass
        if sys.stderr.isatty() and "NO_COLOR" not in os.environ:
            output = "\n".join(f"{color}{line}\033[0m" for color, line in lines)
        else:
            output = "\n".join(line for _, line in lines)
        return output + "\n"


def configure_logging(log_level: int | str) -> None:
    """Apply the shared configuration in each serving process."""
    config_path = Path(
        os.environ.get(
            "SGLANG_LOGGING_CONFIG_PATH",
            str(Path(__file__).with_name("log_config_debug.json")),
        )
    )
    logging.config.dictConfig(json.loads(config_path.read_text(encoding="utf-8")))
    logging.getLogger().setLevel(log_level)


def configure_dependency_loggers() -> None:
    hub_logger = hf_logging.get_logger()
    for console_handler in hub_logger.handlers[:]:
        if type(console_handler) is logging.StreamHandler:
            hub_logger.removeHandler(console_handler)
        else:
            pass
    hf_logging.enable_propagation()
    for logger_name in ("httpx", "httpcore"):
        http_logger = logging.getLogger(logger_name)
        if http_logger.level == logging.NOTSET:
            http_logger.setLevel(logging.WARNING)
        else:
            pass
