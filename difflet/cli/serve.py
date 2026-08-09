"""`difflet serve` command implementation."""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
from typing import Any

from difflet.serving.artifact_store import S3ArtifactStore
from difflet.serving.factory import build_serving_stack
from difflet.serving.openai.api_server import create_app
from difflet.serving.options import (
    CompilePolicy,
    DownloadPolicy,
    ServeOptions,
    validate_worker_heartbeat_interval,
)


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
                "fmt": "%(asctime)s.%(msecs)03d %(levelprefix)s %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "use_colors": None,
            },
            "file": {
                "format": "%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
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
                "formatter": "file",
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
    api_key = getattr(args, "api_key", None)
    if api_key is None:
        api_key = os.environ.get("DIFFLET_API_KEY") or None
    return ServeOptions(
        model_id=args.model_id,
        revision=args.revision,
        host=args.host,
        port=args.port,
        api_key=api_key,
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
        clip_placement=getattr(args, "clip_placement", None),
        teacache_cadence=getattr(args, "teacache_cadence", None),
        teacache_online_delta=getattr(args, "teacache_online_delta", None),
        teacache_speedup=getattr(args, "teacache_speedup", None),
        teacache_calibration=getattr(args, "teacache_calibration", None),
        cache_profile_file=getattr(args, "cache_profile_file", None),
        cache_profile_qualification_file=getattr(args, "cache_profile_qualification_file", None),
        download_policy=download_policy,
        compile_policy=compile_policy,
        max_queued_requests=getattr(args, "max_queued_requests", 8),
        queue_timeout=getattr(args, "queue_timeout", None),
        request_timeout=getattr(args, "request_timeout", 300.0),
        artifact_store_timeout=getattr(args, "artifact_store_timeout", 60.0),
        worker_cancel_timeout=getattr(args, "worker_cancel_timeout", 10.0),
        worker_restart_timeout=getattr(args, "worker_restart_timeout", 900.0),
        worker_heartbeat_interval=getattr(args, "worker_heartbeat_interval", 30.0),
        validation_workers=getattr(args, "validation_workers", 4),
        validation_max_waiting=getattr(args, "validation_max_waiting", 32),
        validation_timeout=getattr(args, "validation_timeout", 30.0),
        video_retention_seconds=getattr(args, "video_retention_seconds", 25 * 60 * 60),
        video_max_jobs=getattr(args, "video_max_jobs", 4096),
        video_sweep_interval_seconds=getattr(args, "video_sweep_interval", 5 * 60.0),
    )


def validate_serve_args(args: argparse.Namespace) -> None:
    """Validate universal serve-process settings before adapter selection."""

    try:
        validate_worker_heartbeat_interval(getattr(args, "worker_heartbeat_interval", 30.0))
    except (TypeError, ValueError):
        print(
            "Error: --worker-heartbeat-interval must be a finite value "
            "between 5 and 120 seconds inclusive.",
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
    artifact_store = S3ArtifactStore.from_env_if_configured(
        client_timeout=options.artifact_store_timeout
    )
    stack = build_serving_stack(options)
    app = create_app(
        options=options,
        resolved_model=stack.resolved_model,
        engine=stack.engine,
        request_validator=stack.request_validator,
        artifact_store=artifact_store,
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
