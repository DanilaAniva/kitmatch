"""
Flexible Loguru logging configuration using a single flag to select format.

Environment variables:
- LOG_LEVEL: default INFO (applies to JSON sink)
- LOG_JSON: when true → JSON sink only; when false → pretty sink only (default: false)
- LOG_JSON_SINK: stdout | path | none (default: stdout)
- LOG_BACKTRACE: default false (prod), true (dev)
- LOG_DIAGNOSE: default false (prod), true (dev)
- LOG_ENQUEUE: default true (when JSON sink not stdout), false otherwise
- LOG_ROTATION, LOG_RETENTION, LOG_COMPRESSION: file sink options (JSON file sink)

Stdlib logging is intercepted via InterceptHandler. Records are enriched with app/env/service.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

from loguru import logger


YES = {"1", "true", "yes", "y", "on", "t"}


def _flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in YES


def _str(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name, "").strip()
    return val if val else default


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller depth outside logging module
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging(app_name: str, *, service: Optional[str] = None, env: Optional[str] = None) -> None:
    # Always assume production defaults; ignore APP_ENV
    env = "prod"
    level = os.getenv("LOG_LEVEL", "INFO").strip().upper()

    # Decide single format strictly by LOG_JSON
    enable_json = _flag("LOG_JSON", "false")
    json_sink_cfg = os.getenv("LOG_JSON_SINK", "stdout").strip().lower()
    if json_sink_cfg in {"none", "off", "false"}:
        enable_json = False
    json_sink_obj = sys.stdout if json_sink_cfg in {"", "stdout"} else json_sink_cfg

    backtrace = _flag("LOG_BACKTRACE", "false")
    diagnose = _flag("LOG_DIAGNOSE", "false")
    enqueue = _flag("LOG_ENQUEUE", "false" if json_sink_obj is sys.stdout else "true")

    rotation = _str("LOG_ROTATION", "500 MB" if json_sink_obj is not sys.stdout else None)
    retention = _str("LOG_RETENTION", "14 days" if json_sink_obj is not sys.stdout else None)
    compression = _str("LOG_COMPRESSION", "zip" if json_sink_obj is not sys.stdout else None)
    pretty_level = "INFO"

    # Intercept stdlib logging
    root_logger = logging.getLogger()
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    root_logger.setLevel(logging.INFO)

    # Configure loguru
    logger.remove()

    handlers = []
    if enable_json:
        json_handler = {
            "sink": json_sink_obj,
            "serialize": True,
            "level": level,
            "backtrace": backtrace,
            "diagnose": diagnose,
            "enqueue": enqueue,
        }
        if json_sink_obj is not sys.stdout:
            if rotation:
                json_handler["rotation"] = rotation
            if retention:
                json_handler["retention"] = retention
            if compression:
                json_handler["compression"] = compression
        handlers.append(json_handler)
    else:
        handlers.append(
            {
                "sink": sys.stderr,
                "level": pretty_level,
                "colorize": True,
                "format": (
                    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                    "<level>{level: <8}</level> | "
                    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                    "<level>{message}</level>"
                ),
                "backtrace": backtrace,
                "diagnose": diagnose,
            }
        )

    def _patch(record: dict) -> None:
        extra = record["extra"]
        extra.setdefault("app", app_name)
        extra.setdefault("env", env)
        if service:
            extra.setdefault("service", service)
        extra["level_name"] = record["level"].name
        extra["severity"] = record["level"].no

    logger.configure(handlers=handlers, extra={"app": app_name, "env": env, **({"service": service} if service else {})}, patcher=_patch)

    # Reduce noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    logger.info("Logging configured")


