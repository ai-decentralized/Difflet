from __future__ import annotations

import asyncio
import logging
import sys
import types

import pytest

from difflet.serving.artifact_store import ArtifactRef, MemoryArtifactStore, R2ArtifactStore
from difflet.serving.errors import DiffletServingError, prompt_too_long
from difflet.serving.model_registry import resolve_serving_model
from difflet.serving.openai.serving_chat import generate_chat_completion, normalize_chat_request
from difflet.serving.options import ServeOptions
from difflet.serving.types import DiffletGenerateOutput


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


class _FailingR2Client:
    def put_object(self, **kwargs):
        raise RuntimeError("secret bucket and endpoint")

    def generate_presigned_url(self, *args, **kwargs):
        raise RuntimeError("secret request id")


def _failing_r2_store(monkeypatch) -> R2ArtifactStore:
    store = R2ArtifactStore(
        bucket="private-bucket",
        endpoint_url="https://private.invalid",
        access_key_id="key",
        secret_access_key="secret",
    )
    monkeypatch.setattr(store, "_client", lambda: _FailingR2Client())
    return store


def test_r2_store_reuses_client(monkeypatch):
    clients = []

    class _Client:
        pass

    def _client(*args, **kwargs):
        client = _Client()
        clients.append(client)
        return client

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=_client))
    monkeypatch.setitem(
        sys.modules,
        "botocore.config",
        types.SimpleNamespace(Config=lambda **kwargs: kwargs),
    )
    store = R2ArtifactStore(
        bucket="bucket",
        endpoint_url="https://example.invalid",
        access_key_id="key",
        secret_access_key="secret",
    )

    assert store._client() is store._client()
    assert len(clients) == 1


def test_r2_public_url_mode_uses_custom_domain_without_presigning(monkeypatch):
    store = R2ArtifactStore(
        bucket="bucket",
        endpoint_url="https://example.invalid",
        access_key_id="key",
        secret_access_key="secret",
        prefix="generated",
        public_base_url="https://images.example.com/",
    )
    monkeypatch.setattr(
        store,
        "_client",
        lambda: (_ for _ in ()).throw(AssertionError("public mode must not presign")),
    )

    url = asyncio.run(
        store.get_url(
            ArtifactRef("image.png", "s3://bucket/generated/image.png", "image/png"),
            ttl_seconds=60,
        )
    )

    assert url == "https://images.example.com/generated/image.png"


def test_r2_private_url_mode_passes_ttl_to_presigner(monkeypatch):
    calls = []

    class _Client:
        def generate_presigned_url(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "https://signed.example/image.png"

    store = R2ArtifactStore(
        bucket="bucket",
        endpoint_url="https://example.invalid",
        access_key_id="key",
        secret_access_key="secret",
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
def test_r2_backend_errors_are_logged_and_sanitized(monkeypatch, caplog, operation):
    async def _run():
        store = _failing_r2_store(monkeypatch)
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
