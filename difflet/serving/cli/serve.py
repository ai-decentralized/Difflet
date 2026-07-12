"""`difflet serve` command implementation."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
from typing import Any

from difflet.serving.factory import build_serving_stack
from difflet.serving.openai.api_server import create_app
from difflet.serving.options import CompilePolicy, DownloadPolicy, ServeOptions


class _ServingLogFileHandler(RotatingFileHandler):
    """Rotate logs by size and by daily time boundary."""

    def __init__(
        self,
        project_name: str,
        max_bytes: int,
        backupCount: int = 1024,
        encoding: str = "utf-8",
    ) -> None:
        self._project_name = project_name
        self._max_bytes = max_bytes
        self._active_day = datetime.now().date()
        filename = _build_log_path(project_name, datetime.now())
        super().__init__(filename, mode="a", maxBytes=0, backupCount=backupCount, encoding=encoding)

    def _set_daily_log_file(self, now: datetime) -> None:
        day = now.date()
        if day == self._active_day:
            return
        if self.stream:
            self.stream.close()
            self.stream = None
        self.baseFilename = _build_log_path(self._project_name, now)
        self._active_day = day
        self.stream = self._open()

    def shouldRollover(self, record: logging.LogRecord) -> bool:
        if self._max_bytes > 0 and self.stream is not None:
            current_position = self.stream.tell()
            formatted = self.format(record)
            msg = f"{formatted}\n"
            if self.encoding:
                msg_length = len(msg.encode(self.encoding))
            else:
                msg_length = len(msg)
            if current_position + msg_length >= self._max_bytes:
                return True
        return False

    def emit(self, record: logging.LogRecord) -> None:
        self._set_daily_log_file(datetime.now())
        super().emit(record)


def _resolve_project_name() -> str:
    cwd = Path.cwd()
    project = cwd.name
    if not project:
        return "difflet"
    return project.replace(" ", "_")


def _resolve_log_dir() -> Path:
    log_dir = Path.cwd() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _build_log_path(project_name: str, now: datetime) -> str:
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"{project_name}-{timestamp}.log"
    return str(_resolve_log_dir() / filename)


def _build_serving_logging_config(project_name: str) -> dict[str, Any]:
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(message)s",
                "use_colors": None,
            },
            "file": {
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "console",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "console",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "()": __name__ + "._ServingLogFileHandler",
                "project_name": project_name,
                "max_bytes": 5 * 1024 * 1024,
                "backupCount": 1024,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["access", "file"],
                "level": "INFO",
                "propagate": False,
            },
        },
        "root": {
            "level": "INFO",
            "handlers": ["console", "file"],
        },
    }


def options_from_args(args: argparse.Namespace) -> ServeOptions:
    validate_serve_args(args)
    compile_policy = CompilePolicy.FORCE if getattr(args, "force", False) else CompilePolicy.AUTO
    download_policy = DownloadPolicy.AUTO
    return ServeOptions(
        model_id=args.model_id,
        revision=args.revision,
        host=args.host,
        port=args.port,
        tp_degree=args.tp_degree,
        cp_degree=args.cp_degree,
        cp_mode=args.cp_mode,
        cfg_parallel=getattr(args, "cfg_parallel", None),
        sp_enabled=getattr(args, "sp_enabled", None),
        height=args.height,
        width=args.width,
        num_frames=getattr(args, "num_frames", None),
        cache_dir=args.cache_dir,
        host_vae=getattr(args, "host_vae", False),
        teacache_cadence=getattr(args, "teacache_cadence", None),
        teacache_online_delta=getattr(args, "teacache_online_delta", None),
        teacache_speedup=getattr(args, "teacache_speedup", None),
        teacache_calibration=getattr(args, "teacache_calibration", None),
        download_policy=download_policy,
        compile_policy=compile_policy,
        worker_heartbeat_interval=getattr(args, "worker_heartbeat_interval", 30.0),
    )


def validate_serve_args(args: argparse.Namespace) -> None:
    """Validate universal serve-process settings before adapter selection."""

    if getattr(args, "worker_heartbeat_interval", 30.0) <= 0:
        print(
            "Error: --worker-heartbeat-interval must be greater than 0.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def run(args: argparse.Namespace) -> None:
    _load_serving_environment()
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("uvicorn is required for `difflet serve`") from exc

    options = options_from_args(args)
    project_name = _resolve_project_name()
    log_config = _build_serving_logging_config(project_name)
    stack = build_serving_stack(options)
    app = create_app(
        options=options,
        resolved_model=stack.resolved_model,
        engine=stack.engine,
        request_validator=stack.request_validator,
    )
    uvicorn.run(
        app,
        host=options.host,
        port=options.port,
        workers=1,
        log_config=log_config,
        log_level="info",
    )


def _load_serving_environment() -> None:
    dotenv_path = Path.cwd() / ".env"
    if not dotenv_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("python-dotenv is required to load .env for `difflet serve`") from exc
    load_dotenv(dotenv_path=dotenv_path, override=False)
