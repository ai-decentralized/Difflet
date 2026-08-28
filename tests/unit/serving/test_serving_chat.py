from __future__ import annotations

import asyncio
import base64
import logging
import sys
import types

import pytest

from difflet.serving.artifact_store import ArtifactRef, MemoryArtifactStore, S3ArtifactStore
from difflet.serving.errors import DiffletServingError, prompt_too_long
from difflet.serving.model_registry import resolve_serving_model
from difflet.serving.openai.serving_chat import generate_chat_completion, normalize_chat_request
from difflet.serving.options import ServeOptions
from difflet.serving.types import DiffletGenerateOutput

_S3_SELECTOR_ENV = (
    "DIFFLET_S3_BUCKET",
    "DIFFLET_S3_ENDPOINT_URL",
    "DIFFLET_S3_REGION",
    "DIFFLET_S3_ACCESS_KEY_ID",
    "DIFFLET_S3_SECRET_ACCESS_KEY",
    "DIFFLET_S3_SESSION_TOKEN",
)


def _resolved():
    return resolve_serving_model(ServeOptions(model_id="black-forest-labs/FLUX.1-dev"))


def _body(**extra):
    body = {
        "model": "black-forest-labs/FLUX.1-dev",
        "messages": [{"role": "user", "content": "a cat"}],
    }
    body.update(extra)
    return body


def test_normalize_chat_request_defaults_from_model_metadata():
    req = normalize_chat_request(_body(), resolved_model=_resolved())

    assert req.prompt == "a cat"
    assert req.height == 1024
    assert req.width == 1024
    assert req.num_inference_steps == 28
    assert req.guidance_scale == 3.5
    assert req.output_format == "png"


def test_normalize_chat_request_uses_server_request_id():
    req = normalize_chat_request(
        _body(),
        resolved_model=_resolved(),
        request_id="server-request-id",
    )

    assert req.request_id == "server-request-id"


@pytest.mark.parametrize("steps", [1, 50])
def test_normalize_chat_request_accepts_bounded_steps(steps):
    req = normalize_chat_request(
        _body(extra_body={"num_inference_steps": steps}),
        resolved_model=_resolved(),
    )

    assert req.num_inference_steps == steps


@pytest.mark.parametrize("steps", [0, 51, 1_000_000])
def test_normalize_chat_request_rejects_steps_outside_serving_bound(steps):
    with pytest.raises(DiffletServingError) as exc:
        normalize_chat_request(
            _body(extra_body={"num_inference_steps": steps}),
            resolved_model=_resolved(),
        )

    assert exc.value.code == "invalid_extra_body"
    assert "1 <= value <= 50" in exc.value.message


@pytest.mark.parametrize("guidance", [0, 1, 3.5, 7.5, 20, 1e308])
def test_normalize_chat_request_preserves_finite_guidance_for_model_validation(guidance):
    request = normalize_chat_request(
        _body(extra_body={"guidance_scale": guidance}),
        resolved_model=_resolved(),
    )

    assert request.guidance_scale == float(guidance)


def test_normalize_chat_request_allows_omitted_model_for_single_model_server():
    body = _body()
    body.pop("model")

    req = normalize_chat_request(body, resolved_model=_resolved())

    assert req.model == "black-forest-labs/FLUX.1-dev"


def test_normalize_chat_request_accepts_image_modality():
    req = normalize_chat_request(_body(modalities=["image"]), resolved_model=_resolved())

    assert req.prompt == "a cat"


def test_text_content_parts_are_joined():
    req = normalize_chat_request(
        _body(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "a cat"},
                        {"type": "text", "text": "in snow"},
                    ],
                }
            ]
        ),
        resolved_model=_resolved(),
    )

    assert req.prompt == "a cat\nin snow"


def test_prompt_character_limit_accepts_boundary():
    prompt = "x" * 16_384

    request = normalize_chat_request(
        _body(messages=[{"role": "user", "content": prompt}]),
        resolved_model=_resolved(),
    )

    assert request.prompt == prompt


@pytest.mark.parametrize(
    "content",
    [
        "x" * 16_385,
        [
            {"type": "text", "text": "x" * 8192},
            {"type": "text", "text": "y" * 8192},
        ],
    ],
)
def test_prompt_character_limit_rejects_before_model_validation(content):
    with pytest.raises(DiffletServingError) as exc:
        normalize_chat_request(
            _body(messages=[{"role": "user", "content": content}]),
            resolved_model=_resolved(),
        )

    assert exc.value.code == "prompt_too_long"
    assert "16384 characters" in exc.value.message


def test_prompt_uses_last_user_message_only():
    with pytest.raises(DiffletServingError) as exc:
        normalize_chat_request(
            _body(
                messages=[
                    {"role": "user", "content": "older prompt"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "  "},
                ]
            ),
            resolved_model=_resolved(),
        )

    assert exc.value.code == "invalid_prompt"


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (_body(height=512), "invalid_extra_body"),
        (_body(extra_body={"tp_degree": 4}), "invalid_extra_body"),
        (_body(extra_body="not an object"), "invalid_extra_body"),
        (_body(extra_body=False), "invalid_extra_body"),
        (_body(extra_body=0), "invalid_extra_body"),
        (_body(extra_body=""), "invalid_extra_body"),
        (_body(extra_body=[]), "invalid_extra_body"),
        (_body(extra_body={"num_frames": 1}), "invalid_extra_body"),
        (_body(extra_body={"height": 512}), "profile_mismatch"),
        (_body(modalities=["text"]), "unsupported_modality"),
        (_body(modalities=["image", "text"]), "unsupported_modality"),
        (_body(temperature=0.1), "feature_not_supported"),
        (_body(max_tokens=16), "feature_not_supported"),
        (_body(id="client-request-id"), "feature_not_supported"),
        (_body(unknown_top_level=True), "feature_not_supported"),
        (_body(extra_body={"steps": 4, "num_inference_steps": 5}), "invalid_extra_body"),
        (_body(messages=[]), "invalid_prompt"),
        (_body(messages=[{"role": "user", "content": ""}]), "invalid_prompt"),
        (
            _body(messages=[{"role": "user", "content": [{"type": "image_url", "url": "x"}]}]),
            "unsupported_input_modality",
        ),
    ],
)
def test_normalize_chat_request_errors(body, code):
    with pytest.raises(DiffletServingError) as exc:
        normalize_chat_request(body, resolved_model=_resolved())

    assert exc.value.code == code


def test_response_policy_fields_are_ignored():
    req = normalize_chat_request(
        _body(
            response_format={"type": "json_object"},
            extra_body={"response_format": "data_url", "artifact_ttl_seconds": 1},
        ),
        resolved_model=_resolved(),
    )

    assert req.prompt == "a cat"


def test_extra_body_null_is_treated_as_absent():
    req = normalize_chat_request(_body(extra_body=None), resolved_model=_resolved())

    assert req.prompt == "a cat"


class _FakeEngine:
    called = False

    async def generate(self, request):
        self.called = True
        return DiffletGenerateOutput(data=b"png", mime_type="image/png", output_format="png")


class _RejectingValidator:
    def validate(self, request):
        raise prompt_too_long("too long")


class _TrackingValidator:
    called = False

    def validate(self, request):
        self.called = True


def test_generate_chat_completion_returns_artifact_url():
    store = MemoryArtifactStore()
    response = asyncio.run(
        generate_chat_completion(
            _body(),
            resolved_model=_resolved(),
            engine=_FakeEngine(),
            artifact_store=store,
            artifact_ttl_seconds=60,
            artifact_store_timeout=1,
        )
    )

    content = response["choices"][0]["message"]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("memory://difflet/")


def test_generate_chat_completion_returns_inline_data_url_without_store():
    response = asyncio.run(
        generate_chat_completion(
            _body(),
            resolved_model=_resolved(),
            engine=_FakeEngine(),
            artifact_store=None,
            artifact_ttl_seconds=60,
            artifact_store_timeout=1,
        )
    )

    url = response["choices"][0]["message"]["content"][0]["image_url"]["url"]
    prefix, encoded = url.split(",", 1)
    assert prefix == "data:image/png;base64"
    assert base64.b64decode(encoded) == b"png"


def test_s3_store_is_optional_when_all_required_environment_is_absent(monkeypatch):
    for name in _S3_SELECTOR_ENV:
        monkeypatch.delenv(name, raising=False)

    assert S3ArtifactStore.from_env_if_configured() is None


def test_s3_store_is_selected_for_complete_required_environment(monkeypatch):
    values = {
        "DIFFLET_S3_BUCKET": "bucket",
        "DIFFLET_S3_REGION": "ap-southeast-4",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    store = S3ArtifactStore.from_env_if_configured()

    assert store is not None
    assert store.bucket == "bucket"
    assert store.endpoint_url is None
    assert store.region_name == "ap-southeast-4"


def test_s3_store_reads_complete_explicit_credentials(monkeypatch):
    for name in _S3_SELECTOR_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DIFFLET_S3_BUCKET", "bucket")
    monkeypatch.setenv("DIFFLET_S3_ENDPOINT_URL", "https://s3-compatible.example")
    monkeypatch.setenv("DIFFLET_S3_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("DIFFLET_S3_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("DIFFLET_S3_SESSION_TOKEN", "token")

    store = S3ArtifactStore.from_env_if_configured()

    assert store is not None
    assert store.endpoint_url == "https://s3-compatible.example"
    assert store.access_key_id == "key"
    assert store.secret_access_key == "secret"
    assert store.session_token == "token"


@pytest.mark.parametrize(
    ("configured", "missing_name"),
    [
        ({"DIFFLET_S3_ACCESS_KEY_ID": "key"}, "DIFFLET_S3_SECRET_ACCESS_KEY"),
        ({"DIFFLET_S3_SECRET_ACCESS_KEY": "secret"}, "DIFFLET_S3_ACCESS_KEY_ID"),
        ({"DIFFLET_S3_SESSION_TOKEN": "token"}, "DIFFLET_S3_ACCESS_KEY_ID"),
    ],
)
def test_s3_store_rejects_partial_explicit_credentials(monkeypatch, configured, missing_name):
    for name in _S3_SELECTOR_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DIFFLET_S3_BUCKET", "bucket")
    for name, value in configured.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(DiffletServingError) as exc:
        S3ArtifactStore.from_env_if_configured()

    assert exc.value.code == "artifact_store_unavailable"
    assert missing_name in exc.value.message


@pytest.mark.parametrize(
    "configured_name",
    [
        "DIFFLET_S3_ENDPOINT_URL",
        "DIFFLET_S3_REGION",
        "DIFFLET_S3_ACCESS_KEY_ID",
        "DIFFLET_S3_SECRET_ACCESS_KEY",
        "DIFFLET_S3_SESSION_TOKEN",
    ],
)
def test_s3_store_rejects_partial_required_environment(monkeypatch, configured_name):
    for name in _S3_SELECTOR_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(configured_name, "configured")

    with pytest.raises(DiffletServingError) as exc:
        S3ArtifactStore.from_env_if_configured()

    assert exc.value.code == "artifact_store_unavailable"
    assert configured_name not in exc.value.message
    assert "missing" in exc.value.message


def test_s3_optional_environment_alone_does_not_enable_store(monkeypatch):
    for name in _S3_SELECTOR_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DIFFLET_S3_PREFIX", "generated")
    monkeypatch.setenv("DIFFLET_S3_CLIENT_TIMEOUT", "30")

    assert S3ArtifactStore.from_env_if_configured() is None


@pytest.mark.parametrize("value", ["", "   "])
def test_s3_empty_required_value_is_rejected_as_partial_configuration(monkeypatch, value):
    for name in _S3_SELECTOR_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DIFFLET_S3_BUCKET", value)

    with pytest.raises(DiffletServingError) as exc:
        S3ArtifactStore.from_env_if_configured()

    assert exc.value.code == "artifact_store_unavailable"
    assert "DIFFLET_S3_BUCKET" in exc.value.message


def test_generate_chat_completion_validates_before_engine():
    engine = _FakeEngine()
    with pytest.raises(DiffletServingError) as exc:
        asyncio.run(
            generate_chat_completion(
                _body(),
                resolved_model=_resolved(),
                engine=engine,
                request_validator=_RejectingValidator(),
                artifact_store=MemoryArtifactStore(),
                artifact_ttl_seconds=60,
                artifact_store_timeout=1,
            )
        )

    assert exc.value.code == "prompt_too_long"
    assert engine.called is False


def test_generate_chat_completion_rejects_raw_prompt_before_provider_validation():
    engine = _FakeEngine()
    validator = _TrackingValidator()
    with pytest.raises(DiffletServingError) as exc:
        asyncio.run(
            generate_chat_completion(
                _body(messages=[{"role": "user", "content": "x" * 16_385}]),
                resolved_model=_resolved(),
                engine=engine,
                request_validator=validator,
                artifact_store=MemoryArtifactStore(),
                artifact_ttl_seconds=60,
                artifact_store_timeout=1,
            )
        )

    assert exc.value.code == "prompt_too_long"
    assert validator.called is False
    assert engine.called is False


class _FailingS3Client:
    def put_object(self, **kwargs):
        raise RuntimeError("secret bucket and endpoint")

    def generate_presigned_url(self, *args, **kwargs):
        raise RuntimeError("secret request id")


def _failing_s3_store(monkeypatch) -> S3ArtifactStore:
    store = S3ArtifactStore(
        bucket="private-bucket",
        endpoint_url="https://private.invalid",
    )
    monkeypatch.setattr(store, "_client", lambda: _FailingS3Client())
    return store


def test_configured_s3_failure_does_not_fallback_to_inline_data(monkeypatch):
    with pytest.raises(DiffletServingError) as exc:
        asyncio.run(
            generate_chat_completion(
                _body(),
                resolved_model=_resolved(),
                engine=_FakeEngine(),
                artifact_store=_failing_s3_store(monkeypatch),
                artifact_ttl_seconds=60,
                artifact_store_timeout=1,
            )
        )

    assert exc.value.code == "internal_error"


def test_s3_store_reuses_client(monkeypatch):
    calls = []

    class _Client:
        pass

    def _client(*args, **kwargs):
        client = _Client()
        calls.append((args, kwargs, client))
        return client

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=_client))
    monkeypatch.setitem(
        sys.modules,
        "botocore.config",
        types.SimpleNamespace(Config=lambda **kwargs: kwargs),
    )
    store = S3ArtifactStore(
        bucket="bucket",
        region_name="ap-southeast-4",
    )

    assert store._client() is store._client()
    assert len(calls) == 1
    assert calls[0][0] == ("s3",)
    assert calls[0][1]["region_name"] == "ap-southeast-4"
    assert "endpoint_url" not in calls[0][1]
    assert "aws_access_key_id" not in calls[0][1]
    assert calls[0][1]["config"]["signature_version"] == "s3v4"
    assert calls[0][1]["config"]["s3"] == {"addressing_style": "virtual"}


def test_s3_store_passes_explicit_compatible_provider_credentials(monkeypatch):
    calls = []

    def _client(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=_client))
    monkeypatch.setitem(
        sys.modules,
        "botocore.config",
        types.SimpleNamespace(Config=lambda **kwargs: kwargs),
    )
    store = S3ArtifactStore(
        bucket="bucket",
        endpoint_url="https://s3-compatible.example",
        region_name="auto",
        access_key_id="key",
        secret_access_key="secret",
        session_token="token",
    )

    store._client()

    options = calls[0][1]
    assert options["endpoint_url"] == "https://s3-compatible.example"
    assert options["aws_access_key_id"] == "key"
    assert options["aws_secret_access_key"] == "secret"
    assert options["aws_session_token"] == "token"
    assert options["config"]["signature_version"] == "s3v4"
    assert options["config"]["s3"] == {"addressing_style": "auto"}


def test_s3_store_accepts_path_addressing_override(monkeypatch):
    calls = []

    def _client(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=_client))
    monkeypatch.setitem(
        sys.modules,
        "botocore.config",
        types.SimpleNamespace(Config=lambda **kwargs: kwargs),
    )
    store = S3ArtifactStore(
        bucket="bucket",
        endpoint_url="https://s3-compatible.example",
        addressing_style="path",
    )

    store._client()

    assert calls[0][1]["config"]["s3"] == {"addressing_style": "path"}


def test_s3_store_rejects_invalid_addressing_style():
    with pytest.raises(ValueError, match="addressing style"):
        S3ArtifactStore(bucket="bucket", addressing_style="invalid")


def test_s3_url_passes_ttl_to_presigner(monkeypatch):
    calls = []

    class _Client:
        def generate_presigned_url(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "https://signed.example/image.png"

    store = S3ArtifactStore(
        bucket="bucket",
        region_name="ap-southeast-4",
        prefix="generated",
    )
    monkeypatch.setattr(store, "_client", lambda: _Client())

    url = asyncio.run(
        store.get_url(
            ArtifactRef("image.png", "s3://bucket/generated/image.png", "image/png"),
            ttl_seconds=123,
        )
    )

    assert url == "https://signed.example/image.png"
    assert calls[0][1]["ExpiresIn"] == 123


@pytest.mark.parametrize("operation", ["upload", "presign"])
def test_s3_backend_errors_are_logged_and_sanitized(monkeypatch, caplog, operation):
    async def _run():
        store = _failing_s3_store(monkeypatch)
        if operation == "upload":
            return await store.put_bytes(
                data=b"png",
                mime_type="image/png",
                suffix=".png",
                ttl_seconds=60,
            )
        return await store.get_url(
            ArtifactRef("image.png", "s3://private-bucket/image.png", "image/png"),
            ttl_seconds=60,
        )

    caplog.set_level(logging.ERROR, logger="difflet.serving.artifact_store")
    with pytest.raises(DiffletServingError) as exc:
        asyncio.run(_run())

    assert exc.value.code == "internal_error"
    assert exc.value.message == "Internal artifact storage error"
    assert "secret" not in exc.value.message
    assert "secret" in caplog.text


def test_normalize_chat_request_enforces_shape_set_membership():
    resolved = resolve_serving_model(
        ServeOptions(model_id="black-forest-labs/FLUX.1-dev", shapes="1024x1024,512x512")
    )

    defaulted = normalize_chat_request(_body(), resolved_model=resolved)
    assert (defaulted.height, defaulted.width) == (1024, 1024)

    member = normalize_chat_request(
        _body(extra_body={"height": 512, "width": 512}), resolved_model=resolved
    )
    assert (member.height, member.width) == (512, 512)

    with pytest.raises(DiffletServingError) as exc:
        normalize_chat_request(
            _body(extra_body={"height": 768, "width": 768}), resolved_model=resolved
        )
    assert exc.value.code == "profile_mismatch"
    assert "768x768" in exc.value.message
    assert "1024x1024" in exc.value.message and "512x512" in exc.value.message
