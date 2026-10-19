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


# --- replicas and concurrency (F2b task 2) ----------------------------------------------------


def test_serve_command_carries_the_replica_count_and_queue_timeout():
    cmd, _ = smoke.serve_command(port=8099, cache_dir=None, python="py", dp=2, queue_timeout=120)
    assert cmd[3:] == [
        "serve",
        "--model-id",
        "Qwen/Qwen-Image",
        "--port",
        "8099",
        "--dp-degree",
        "2",
        "--queue-timeout",
        "120",
    ]
    cmd, _ = smoke.serve_command(port=8099, cache_dir=None, python="py")
    assert "--dp-degree" not in cmd and "--queue-timeout" not in cmd


def test_fire_concurrently_runs_the_posts_side_by_side_and_keeps_their_order():
    import time

    def poster(index: int):
        time.sleep(0.2)
        return {"index": index}, 0.2

    started = time.monotonic()
    results, wall = smoke.fire_concurrently(poster, 3)
    assert [payload["index"] for payload, _ in results] == [0, 1, 2]
    assert wall < 0.45 and time.monotonic() - started < 0.6
    results, wall = smoke.fire_concurrently(poster, 1)
    assert len(results) == 1


def test_fire_concurrently_raises_the_first_failure():
    def poster(index: int):
        if index == 1:
            raise RuntimeError("boom")
        return {"index": index}, 0.0

    with pytest.raises(RuntimeError, match="boom"):
        smoke.fire_concurrently(poster, 2)


def test_identical_to_reference_compares_bytes(tmp_path):
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"\x89PNGabc")
    assert smoke.identical_to_reference(b"\x89PNGabc", ref) is True
    assert smoke.identical_to_reference(b"\x89PNGabd", ref) is False
    assert smoke.identical_to_reference(b"\x89PNGabc", None) is None


# --- F3c: Flux and Wan ---------------------------------------------------------------------

_FLUX = "black-forest-labs/FLUX.1-dev"
_WAN = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"


def test_each_model_has_the_cuda_cli_s_defaults():
    qwen, flux, wan = (smoke.MODEL_DEFAULTS[m] for m in (smoke.MODEL_ID, _FLUX, _WAN))
    assert (qwen["steps"], qwen["height"], qwen["width"], qwen["kind"]) == (50, 1024, 1024, "image")
    assert (flux["steps"], flux["height"], flux["width"], flux["kind"]) == (28, 1024, 1024, "image")
    assert (wan["steps"], wan["height"], wan["width"], wan["num_frames"]) == (40, 480, 832, 9)
    assert wan["kind"] == "video" and wan["startup_timeout"] > qwen["startup_timeout"]


def test_flags_left_out_take_the_model_s_defaults():
    args = smoke.parse_args(["--model-id", _WAN])
    assert (args.steps, args.height, args.width, args.num_frames) == (40, 480, 832, 9)
    assert args.output.endswith(".mp4") and args.startup_timeout == 1800.0
    args = smoke.parse_args(["--model-id", _FLUX, "--steps", "4"])
    assert (args.steps, args.height) == (4, 1024) and args.output.endswith(".png")
    assert smoke.parse_args([]).model_id == smoke.MODEL_ID == "Qwen/Qwen-Image"
    with pytest.raises(SystemExit):
        smoke.parse_args(["--model-id", "Wan-AI/Wan2.1-T2V-14B-Diffusers"])


def test_flux_chat_body_names_the_model_and_carries_guidance_only_when_given():
    body = smoke.build_chat_body(
        "a cat", steps=28, seed=1, height=1024, width=1024, model_id=_FLUX, guidance_scale=3.5
    )
    assert body["model"] == _FLUX and body["extra_body"]["guidance_scale"] == 3.5
    body = smoke.build_chat_body("a cat", steps=28, seed=1, height=1024, width=1024, model_id=_FLUX)
    assert "guidance_scale" not in body["extra_body"]


def test_wan_video_form_carries_the_profile_shape_and_the_request_knobs():
    fields = dict(
        smoke.build_video_form(
            "a cat walking",
            steps=40,
            seed=2,
            height=480,
            width=832,
            num_frames=9,
            model_id=_WAN,
            guidance_scale=4.0,
            guidance_scale_2=3.0,
        )
    )
    assert fields == {
        "model": _WAN,
        "prompt": "a cat walking",
        "size": "832x480",
        "num_frames": "9",
        "num_inference_steps": "40",
        "seed": "2",
        "guidance_scale": "4",
        "guidance_scale_2": "3",
    }
    fields = dict(
        smoke.build_video_form(
            "x", steps=4, seed=0, height=480, width=832, num_frames=9, model_id=_WAN
        )
    )
    assert "guidance_scale" not in fields and "guidance_scale_2" not in fields


def test_multipart_encoding_is_what_the_videos_endpoint_parses():
    from email.parser import BytesParser
    from email.policy import default

    body, content_type = smoke.encode_multipart([("prompt", "a cat"), ("num_frames", "9")])
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\n\r\n".encode() + body
    )
    parts = {
        p.get_param("name", header="content-disposition"): p.get_content()
        for p in message.iter_parts()
    }
    assert parts == {"prompt": "a cat", "num_frames": "9"}


def test_serve_command_names_the_model():
    cmd, _ = smoke.serve_command(port=8099, cache_dir=None, python="py", model_id=_WAN)
    assert cmd[3:7] == ["serve", "--model-id", _WAN, "--port"]
