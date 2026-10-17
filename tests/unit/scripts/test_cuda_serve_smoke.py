"""Pure parts of scripts/cuda_serve_smoke.py (the server run is the script itself, on a GPU)."""

import base64
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))

import cuda_serve_smoke as smoke  # noqa: E402


def test_chat_body_carries_steps_seed_and_shape():
    body = smoke.build_chat_body("a cat", steps=50, seed=0, height=1024, width=1024)
    assert body["model"] == "Qwen/Qwen-Image"
    assert body["messages"] == [{"role": "user", "content": "a cat"}]
    assert body["extra_body"] == {
        "num_inference_steps": 50,
        "seed": 0,
        "height": 1024,
        "width": 1024,
    }
    assert "guidance_scale" not in body["extra_body"]


def test_image_data_url_round_trip():
    url = "data:image/png;base64," + base64.b64encode(b"\x89PNG...").decode()
    assert smoke.decode_image_data_url(url) == b"\x89PNG..."
    with pytest.raises(ValueError):
        smoke.decode_image_data_url("https://example.com/x.png")


def test_image_url_is_taken_from_the_first_choice():
    body = {
        "choices": [
            {"message": {"content": [{"type": "image_url", "image_url": {"url": "data:x"}}]}}
        ]
    }
    assert smoke.image_url_from_response(body) == "data:x"
    with pytest.raises(ValueError):
        smoke.image_url_from_response({"choices": [{"message": {"content": []}}]})


def test_serve_command_runs_the_cli_on_cuda_with_the_port_and_cache_dir():
    cmd, env = smoke.serve_command(port=8099, cache_dir="/c", python="py")
    assert cmd[:3] == ["py", "-m", "difflet.cli.main"]
    assert cmd[3:] == [
        "serve",
        "--model-id",
        "Qwen/Qwen-Image",
        "--port",
        "8099",
        "--cache-dir",
        "/c",
    ]
    assert env["DIFFLET_BACKEND"] == "cuda"


def test_image_bytes_come_from_a_data_url_or_an_http_fetch():
    # Review finding: with an S3 .env in the repo root the server returns https URLs, and the
    # script raised after a successful generation.
    fetched = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"PNG-from-s3"

    def opener(url, timeout):
        fetched.append(url)
        return _Resp()

    data_url = "data:image/png;base64," + base64.b64encode(b"inline").decode()
    assert smoke.fetch_image_bytes(data_url, opener=opener) == b"inline" and fetched == []
    assert smoke.fetch_image_bytes("https://s3/x.png", opener=opener) == b"PNG-from-s3"
    assert fetched == ["https://s3/x.png"]
    with pytest.raises(ValueError):
        smoke.fetch_image_bytes("ftp://nope", opener=opener)


def test_request_headers_carry_the_api_key_when_the_server_requires_one():
    assert smoke.request_headers(None) == {"Content-Type": "application/json"}
    assert smoke.request_headers("k1") == {
        "Content-Type": "application/json",
        "Authorization": "Bearer k1",
    }
