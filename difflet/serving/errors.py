"""Serving-layer exceptions and OpenAI-style error payload helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DiffletServingError(Exception):
    """HTTP-visible serving error with a stable machine-readable code."""

    status_code: int
    code: str
    message: str
    error_type: str = "invalid_request_error"

    def __str__(self) -> str:
        return f"{self.status_code} {self.code}: {self.message}"

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.code,
            }
        }


def invalid_extra_body(message: str) -> DiffletServingError:
    return DiffletServingError(400, "invalid_extra_body", message)


def invalid_prompt(message: str) -> DiffletServingError:
    return DiffletServingError(400, "invalid_prompt", message)


def profile_mismatch(message: str) -> DiffletServingError:
    return DiffletServingError(400, "profile_mismatch", message)


def feature_not_supported(message: str) -> DiffletServingError:
    return DiffletServingError(400, "feature_not_supported", message)


def unsupported_input_modality(message: str) -> DiffletServingError:
    return DiffletServingError(400, "unsupported_input_modality", message)


def unsupported_modality(message: str) -> DiffletServingError:
    return DiffletServingError(400, "unsupported_modality", message)


def prompt_too_long(message: str) -> DiffletServingError:
    return DiffletServingError(400, "prompt_too_long", message)


def engine_unavailable(message: str) -> DiffletServingError:
    return DiffletServingError(503, "engine_unavailable", message, error_type="server_error")


def request_cancelled(message: str) -> DiffletServingError:
    return DiffletServingError(499, "request_cancelled", message, error_type="server_error")
