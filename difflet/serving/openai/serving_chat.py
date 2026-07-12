"""OpenAI chat-completions compatibility wrapper for image generation."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from difflet.serving.artifact_store import ArtifactStore, get_url_with_timeout, put_with_timeout
from difflet.serving.errors import (
    DiffletServingError,
    feature_not_supported,
    invalid_extra_body,
    invalid_prompt,
    profile_mismatch,
    unsupported_modality,
    unsupported_input_modality,
)
from difflet.serving.model_registry import ResolvedServingModel
from difflet.serving.types import DiffletGenerateRequest, DiffletGenerateOutput, ServingProfile

logger = logging.getLogger(__name__)

_IGNORED_RESPONSE_POLICY_FIELDS = {"response_format", "artifact_ttl_seconds"}
_ALLOWED_TOP_LEVEL_FIELDS = {
    "model",
    "messages",
    "modalities",
    "extra_body",
    "response_format",
    "artifact_ttl_seconds",
}
_TOP_LEVEL_UNSUPPORTED_FEATURES = {
    "stream",
    "tools",
    "tool_choice",
    "functions",
    "function_call",
    "temperature",
    "top_p",
    "n",
    "stop",
    "max_tokens",
    "metadata",
}
_KNOWN_DIFFLET_FIELDS = {
    "height",
    "width",
    "num_frames",
    "num_inference_steps",
    "steps",
    "guidance_scale",
    "true_cfg_scale",
    "seed",
    "negative_prompt",
    "output_format",
    "tp_degree",
    "cp_degree",
    "cp_mode",
    "cfg_parallel",
    "sp_enabled",
    "teacache_cadence",
    "teacache_online_delta",
    "teacache_speedup",
    "teacache_calibration",
}
_STARTUP_ONLY_FIELDS = {"tp_degree", "cp_degree", "cp_mode", "cfg_parallel", "sp_enabled"}
_TEACACHE_FIELDS = {
    "teacache_cadence",
    "teacache_online_delta",
    "teacache_speedup",
    "teacache_calibration",
}
_ALLOWED_EXTRA_FIELDS = {
    "height",
    "width",
    "num_frames",
    "num_inference_steps",
    "steps",
    "guidance_scale",
    "true_cfg_scale",
    "seed",
    "negative_prompt",
    "output_format",
    "response_format",
    "artifact_ttl_seconds",
}


def normalize_chat_request(
    body: Any,
    *,
    resolved_model: ResolvedServingModel,
) -> DiffletGenerateRequest:
    _validate_top_level(body)
    requested_model = body.get("model")
    if requested_model is not None and requested_model != resolved_model.model_id:
        raise DiffletServingError(
            400, "model_not_served", "request model does not match server model"
        )
    _validate_modalities(body.get("modalities"), resolved_model.metadata.output_modality)

    prompt = _extract_prompt(body.get("messages"))
    extra_raw = body.get("extra_body", {})
    if extra_raw is None:
        extra_raw = {}
    if not isinstance(extra_raw, dict):
        raise invalid_extra_body("extra_body must be an object")
    extra = dict(extra_raw)
    _validate_extra_keys(extra)

    profile = resolved_model.profile
    metadata = resolved_model.metadata
    height = _int_field(extra.get("height", profile.height), "height")
    width = _int_field(extra.get("width", profile.width), "width")
    if height != profile.height or width != profile.width:
        raise profile_mismatch("request shape does not match serving profile")

    if extra.get("num_frames") is not None:
        raise invalid_extra_body("num_frames is reserved for future video serving")

    output_format = str(extra.get("output_format", "png"))
    if output_format != "png":
        raise invalid_extra_body("P0 image serving supports output_format='png' only")

    if extra.get("true_cfg_scale") is not None:
        raise invalid_extra_body("true_cfg_scale is not supported by Qwen/Flux P0 serving")
    if extra.get("negative_prompt") is not None:
        raise invalid_extra_body("negative_prompt is not supported by Qwen/Flux P0 serving")

    steps = _resolve_steps(extra, metadata.default_steps)
    guidance = _float_field(
        extra.get("guidance_scale", metadata.default_guidance_scale),
        "guidance_scale",
    )
    seed = _int_field(extra.get("seed", 42), "seed")
    if seed < 0 or seed > 2**63 - 1:
        raise invalid_extra_body("seed must satisfy 0 <= seed <= 2**63 - 1")
    if steps < 1 or steps > 50:
        raise invalid_extra_body("num_inference_steps must satisfy 1 <= value <= 50")
    if guidance < 0:
        raise invalid_extra_body("guidance_scale must be non-negative")

    return DiffletGenerateRequest(
        request_id=str(body.get("id") or uuid.uuid4()),
        model=resolved_model.model_id,
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance,
        seed=seed,
        output_format=output_format,
    )


async def generate_chat_completion(
    body: Any,
    *,
    resolved_model: ResolvedServingModel,
    engine,
    request_validator=None,
    artifact_store: ArtifactStore,
    artifact_ttl_seconds: int,
    artifact_store_timeout: float,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    request = normalize_chat_request(body, resolved_model=resolved_model)
    logger.info(
        "chat request normalized model=%s request_id=%s prompt_len=%s steps=%s guidance=%s seed=%s",
        request.model,
        request.request_id,
        len(request.prompt),
        request.num_inference_steps,
        request.guidance_scale,
        request.seed,
    )
    try:
        if request_validator is not None:
            request_validator.validate(request)
    except DiffletServingError:
        logger.warning(
            "chat request validation_rejected request_id=%s model=%s", request.request_id, request.model
        )
        raise
    generate_started_at = time.perf_counter()
    try:
        output: DiffletGenerateOutput = await engine.generate(request)
    except DiffletServingError:
        logger.warning(
            "chat request engine_error request_id=%s model=%s", request.request_id, request.model
        )
        raise
    except Exception:
        logger.exception(
            "chat request engine_unhandled request_id=%s model=%s",
            request.request_id,
            request.model,
        )
        raise
    logger.info(
        "chat request engine_ok request_id=%s model=%s latency_ms=%.2f",
        request.request_id,
        request.model,
        (time.perf_counter() - generate_started_at) * 1000.0,
    )
    store_started_at = time.perf_counter()
    ref = await put_with_timeout(
        artifact_store,
        data=output.data,
        mime_type=output.mime_type,
        suffix=".png",
        ttl_seconds=artifact_ttl_seconds,
        timeout_s=artifact_store_timeout,
    )
    logger.info(
        "chat request artifact_uploaded request_id=%s file_id=%s",
        request.request_id,
        ref.file_id,
    )
    url = await get_url_with_timeout(
        artifact_store,
        ref,
        ttl_seconds=artifact_ttl_seconds,
        timeout_s=artifact_store_timeout,
    )
    logger.info(
        "chat request artifact_url_ready request_id=%s duration_ms=%.2f",
        request.request_id,
        (time.perf_counter() - store_started_at) * 1000.0,
    )
    total_ms = (time.perf_counter() - started_at) * 1000.0
    logger.info(
        "chat request completed request_id=%s model=%s total_ms=%.2f",
        request.request_id,
        request.model,
        total_ms,
    )
    now = int(time.time())
    return {
        "id": f"chatcmpl-{request.request_id}",
        "object": "chat.completion",
        "created": now,
        "model": resolved_model.model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": url},
                        }
                    ],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _validate_top_level(body: Any) -> None:
    if not isinstance(body, dict):
        raise DiffletServingError(400, "invalid_request", "request body must be an object")
    for field in _TOP_LEVEL_UNSUPPORTED_FEATURES:
        if field in body and body[field] is not None:
            raise feature_not_supported(f"top-level field {field!r} is not supported")
    for field in _KNOWN_DIFFLET_FIELDS:
        if field in body and field not in _IGNORED_RESPONSE_POLICY_FIELDS:
            raise invalid_extra_body(
                f"Difflet generation field {field!r} must be supplied inside extra_body"
            )
    for field in body:
        if field not in _ALLOWED_TOP_LEVEL_FIELDS and field not in _KNOWN_DIFFLET_FIELDS:
            raise feature_not_supported(f"top-level field {field!r} is not supported")


def _validate_modalities(modalities: Any, output_modality: str) -> None:
    if modalities is None:
        return
    if not isinstance(modalities, list) or not modalities:
        raise unsupported_modality("modalities must be a non-empty array")
    if any(not isinstance(item, str) for item in modalities):
        raise unsupported_modality("modalities must contain strings")
    if modalities != [output_modality]:
        raise unsupported_modality(
            f"P0 {output_modality} serving only supports modalities={[output_modality]!r}"
        )


def _validate_extra_keys(extra: dict[str, Any]) -> None:
    for field in extra:
        if field in _IGNORED_RESPONSE_POLICY_FIELDS:
            continue
        if field in _STARTUP_ONLY_FIELDS:
            raise invalid_extra_body(f"{field} is a startup-only serving field")
        if field in _TEACACHE_FIELDS:
            raise invalid_extra_body(f"{field} is not supported in P0 serving requests")
        if field not in _ALLOWED_EXTRA_FIELDS:
            raise invalid_extra_body(f"unsupported extra_body field {field!r}")


def _extract_prompt(messages: Any) -> str:
    if not isinstance(messages, list) or not messages:
        raise invalid_prompt("messages must contain a user prompt")
    last_user = None
    for message in messages:
        if not isinstance(message, dict):
            raise invalid_prompt("messages must be objects")
        if message.get("role") == "user":
            last_user = message
    if last_user is None:
        raise invalid_prompt("messages must contain a user prompt")
    text = _content_to_text(last_user.get("content"))
    if not text.strip():
        raise invalid_prompt("no non-empty user prompt found")
    return text


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                raise unsupported_input_modality("message content items must be objects")
            if item.get("type") != "text":
                raise unsupported_input_modality("P0 only accepts text input content")
            text = item.get("text")
            if not isinstance(text, str):
                raise unsupported_input_modality("text content item must contain string text")
            unknown = set(item) - {"type", "text"}
            if unknown:
                raise unsupported_input_modality("text content item has unsupported keys")
            parts.append(text)
        return "\n".join(parts)
    raise unsupported_input_modality("P0 user content must be text or text content items")


def _resolve_steps(extra: dict[str, Any], default: int) -> int:
    has_steps = "steps" in extra
    has_num_steps = "num_inference_steps" in extra
    if has_steps and has_num_steps and extra["steps"] != extra["num_inference_steps"]:
        raise invalid_extra_body("steps and num_inference_steps disagree")
    value = extra.get("num_inference_steps", extra.get("steps", default))
    return _int_field(value, "num_inference_steps")


def _int_field(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise invalid_extra_body(f"{field} must be an integer")
    return int(value)


def _float_field(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise invalid_extra_body(f"{field} must be a finite number")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise invalid_extra_body(f"{field} must be a finite number")
    return result
