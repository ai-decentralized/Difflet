"""scripts/yottalabs_pod.py against a fake transport: request bodies per region, pool selection,
the SSH endpoint, polling decisions, the capacity watcher, the lifecycle subcommands, and that
the API key never reaches stdout, stderr, an exception or ``--dry-run`` output. No network."""

import io
import json
import pathlib
import sys
import urllib.error

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))

import yottalabs_pod as yp  # noqa: E402

PRO6000 = "NVIDIA_RTX_PRO_6000_96G"
VOLUME_ID = 469225541779435520
POD_ID = "502630011196547564"
SECRET = "yk-SENTINEL-0123456789abcdef"


def ok(data):
    return {"message": "success", "code": 10000, "data": data}


def err(code, message):
    return {"message": message, "code": code, "data": None}


def pool(region, status="AVAILABLE", gpu_type=PRO6000, counts=(1, 2)):
    return {
        "gpuType": gpu_type,
        "displayName": "RTX PRO 6000",
        "memoryInGb": 96,
        "maxGpuCount": 2,
        "gpuCountOptions": list(counts),
        "pricePerGpuPerHour": "0.45",
        "containerDiskLimitInGb": 178,
        "region": region,
        "status": status,
        "storageType": "SSD",
    }


def pod(status, host="", port=None, region="us-east-3"):
    return {
        "id": POD_ID,
        "status": status,
        "region": region,
        "gpuCount": 1,
        "containerVolumeInGb": 150,
        "volumes": [],
        "singleCardPrice": "0.45",
        "sshCmd": "ssh user@203.0.113.10 -p 30000 -i <private key file>",
        "expose": [
            {"port": 8888, "proxyPort": 30001, "protocol": "HTTP", "host": "other.example"},
            {
                "port": 22,
                "proxyPort": port,
                "protocol": "SSH",
                "host": host,
                "healthy": bool(host),
            },
        ],
    }


class FakeTransport:
    """Scripted responses per (method, path); a list is consumed one per call, the last
    element repeating. Every call is recorded."""

    def __init__(self, routes):
        self.routes = {k: list(v) if isinstance(v, list) else [v] for k, v in routes.items()}
        self.calls = []

    def __call__(self, method, path, body=None):
        self.calls.append((method, path, body))
        queue = self.routes.get((method, path))
        if queue is None:
            return {"message": "parameter error", "code": 10001, "data": None}
        resp = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(resp, Exception):
            raise resp
        return resp

    def count(self, method, path):
        return sum(1 for m, p, _ in self.calls if (m, p) == (method, path))

    def bodies(self, method, path):
        return [b for m, p, b in self.calls if (m, p) == (method, path)]


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def run(argv, transport, clock=None):
    clock = clock or FakeClock()
    out, errs = io.StringIO(), io.StringIO()
    rc = yp.main(
        argv, transport=transport, sleep=clock.sleep, clock=clock.clock, stdout=out, stderr=errs
    )
    return rc, out.getvalue(), errs.getvalue()


# --- pure functions -------------------------------------------------------------------------


def test_create_body_per_region():
    # disk_gb None: --disk-gb not given -> no container disk size beside the volume.
    east = yp.build_create_body(
        "difflet-f8", 1, "us-east-3", None, VOLUME_ID, "/data", yp.DEFAULT_IMAGE
    )
    # The volume region with a volume: the v1 create, the only one that attaches a network
    # volume (v2 takes R2 volumes only, 12016; an unknown key is dropped silently).
    assert yp.create_path("us-east-3", VOLUME_ID) == "/openapi/v1/pods/create"
    assert yp.create_path("us-east-3", None) == yp.create_path("eu-3", VOLUME_ID) == "/v2/pods"
    assert east["regionList"] == ["us-east-3"]
    assert east["persistentVolumes"] == [
        {"id": VOLUME_ID, "volumeType": "Ceph", "mountMode": "Data", "mountPath": "/data"}
    ]
    assert "containerVolumeInGb" not in east and "volumes" not in east
    assert east["podName"] == "difflet-f8" and east["gpuType"] == PRO6000 and east["gpuCount"] == 1
    assert east["image"] == "yottalabsai/pytorch:2.9.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
    assert east["expose"] == [{"port": 22, "protocol": "SSH"}]

    eu = yp.build_create_body("difflet-f8", 2, "eu-3", None, VOLUME_ID, "/data", yp.DEFAULT_IMAGE)
    assert eu["regions"] == ["eu-3"]
    assert eu["containerVolumeInGb"] == yp.DEFAULT_DISK_GB == 150
    assert "volumes" not in eu and "persistentVolumes" not in eu
    assert eu["name"] == "difflet-f8" and eu["gpuCount"] == 2

    east_no_volume = yp.build_create_body("n", 1, "us-east-3", 120, None, "/data", yp.DEFAULT_IMAGE)
    assert east_no_volume["containerVolumeInGb"] == 120
    assert "volumes" not in east_no_volume and east_no_volume["regions"] == ["us-east-3"]
    default_no_volume = yp.build_create_body(
        "n", 1, "us-east-3", None, None, "/data", yp.DEFAULT_IMAGE
    )
    assert default_no_volume["containerVolumeInGb"] == 150
    # the volume id is sent as an exact JSON integer (int64 on the v1 schema; it exceeds 2**53)
    assert f'"id": {VOLUME_ID}' in json.dumps(east)


def test_create_body_per_region_through_the_cli():
    t = FakeTransport(
        {
            ("GET", "/v2/gpus"): ok([pool("eu-3"), pool("us-east-3")]),
            ("POST", yp.V1_CREATE_PATH): ok(POD_ID),  # v1 answers the bare id
        }
    )
    rc, out, _ = run(["create", "--name", "n", "--gpus", "1", "--volume-id", str(VOLUME_ID)], t)
    assert rc == 0 and out.strip() == POD_ID and t.count("POST", "/v2/pods") == 0
    (body,) = t.bodies("POST", yp.V1_CREATE_PATH)
    assert body["regionList"] == ["us-east-3"]
    assert body["persistentVolumes"][0]["id"] == VOLUME_ID

    t = FakeTransport(
        {
            ("GET", "/v2/gpus"): ok([pool("us-east-3", "UNAVAILABLE"), pool("eu-3")]),
            ("POST", "/v2/pods"): ok(pod("INITIALIZING", region="eu-3")),
        }
    )
    rc, out, _ = run(["create", "--name", "n", "--gpus", "2", "--volume-id", str(VOLUME_ID)], t)
    assert rc == 0
    (body,) = t.bodies("POST", "/v2/pods")
    assert body["regions"] == ["eu-3"] and body["containerVolumeInGb"] == 150
    assert "volumes" not in body and "persistentVolumes" not in body


def test_create_body_keeps_an_explicit_disk_with_a_volume():
    # An explicit --disk-gb is sent with the volume too (the API took both on difflet-cuda-f5b):
    # the F9 pod needs a 170 GB container disk for the Wan download beside the cache volume.
    east = yp.build_create_body("n", 1, "us-east-3", 170, VOLUME_ID, "/data", yp.DEFAULT_IMAGE)
    assert east["containerVolumeInGb"] == 170
    assert east["persistentVolumes"] == [
        {"id": VOLUME_ID, "volumeType": "Ceph", "mountMode": "Data", "mountPath": "/data"}
    ]
    # An explicit 150 is still explicit: sent with the volume.
    east = yp.build_create_body("n", 1, "us-east-3", 150, VOLUME_ID, "/data", yp.DEFAULT_IMAGE)
    assert east["containerVolumeInGb"] == 150 and "persistentVolumes" in east
    # Outside the volume's region the explicit size is the container disk, as before.
    eu = yp.build_create_body("n", 2, "eu-3", 170, VOLUME_ID, "/data", yp.DEFAULT_IMAGE)
    assert eu["containerVolumeInGb"] == 170 and "persistentVolumes" not in eu


def test_disk_gb_not_given_is_none_and_given_is_kept_through_the_cli():
    parser = yp.build_parser()
    assert parser.parse_args(["create", "--name", "n", "--gpus", "1"]).disk_gb is None
    assert (
        parser.parse_args(["create", "--name", "n", "--gpus", "1", "--disk-gb", "170"]).disk_gb
        == 170
    )

    def volume_pod(extra):
        t = FakeTransport(
            {
                ("GET", "/v2/gpus"): ok([pool("us-east-3")]),
                ("POST", yp.V1_CREATE_PATH): ok(POD_ID),
            }
        )
        rc, out, _ = run(
            ["create", "--name", "n", "--gpus", "1", "--volume-id", str(VOLUME_ID), *extra], t
        )
        assert rc == 0 and out.strip() == POD_ID
        (body,) = t.bodies("POST", yp.V1_CREATE_PATH)
        return body

    body = volume_pod(["--disk-gb", "170"])
    assert body["containerVolumeInGb"] == 170
    assert body["persistentVolumes"][0]["id"] == VOLUME_ID
    body = volume_pod([])
    assert "containerVolumeInGb" not in body and "persistentVolumes" in body


def test_dry_run_shows_the_explicit_disk_beside_the_volume():
    def no_network(*a, **k):
        raise AssertionError("dry-run must not send")

    rc, out, _ = run(
        [
            "create",
            "--name",
            "n",
            "--gpus",
            "1",
            "--regions",
            "us-east-3",
            "--volume-id",
            str(VOLUME_ID),
            "--disk-gb",
            "170",
            "--dry-run",
        ],
        no_network,
    )
    assert rc == 0
    body = json.loads(out.split("# region us-east-3\n", 1)[1])
    assert body["containerVolumeInGb"] == 170 and body["persistentVolumes"]
    assert f"POST {yp.API_BASE}{yp.V1_CREATE_PATH}  # region us-east-3" in out


def test_create_help_says_an_explicit_disk_goes_with_the_volume():
    help_text = " ".join(
        yp.build_parser()._subparsers._group_actions[0].choices["create"].format_help().split()
    )
    assert "--disk-gb" in help_text and "even with a volume" in help_text, help_text


def test_pick_region_follows_preference_not_pool_order():
    pools = [pool("eu-2"), pool("eu-3"), pool("us-east-3", "UNAVAILABLE")]
    assert yp.pick_region(pools, ["us-east-3", "eu-3", "eu-2"]) == "eu-3"
    pools.append(pool("us-east-3"))  # a second listing row for the same region, AVAILABLE
    assert yp.pick_region(pools, ["us-east-3", "eu-3", "eu-2"]) == "us-east-3"
    assert yp.pick_region([pool("eu-3", "UNAVAILABLE")], ["us-east-3", "eu-3"]) is None
    assert yp.pick_region([pool("eu-2")], ["us-east-3", "eu-3"]) is None
    assert yp.pick_region([], ["us-east-3"]) is None


def test_pick_region_skips_pools_that_do_not_sell_the_count():
    pools = [pool("eu-3", counts=(2,)), pool("eu-2", counts=(1, 2))]
    assert yp.pick_region(pools, ["eu-3", "eu-2"], gpu_count=1) == "eu-2"
    assert yp.pick_region(pools, ["eu-3", "eu-2"], gpu_count=2) == "eu-3"
    assert yp.pick_region(pools, ["eu-3", "eu-2"]) == "eu-3"


def test_ssh_endpoint():
    assert yp.ssh_endpoint(pod("RUNNING", "c3th3gvgo406k-m.proxy.yottalabs.ai", 30000)) == (
        "c3th3gvgo406k-m.proxy.yottalabs.ai",
        30000,
    )
    assert yp.ssh_endpoint(pod("INITIALIZING")) is None
    assert yp.ssh_endpoint(pod("RUNNING", "h.proxy.yottalabs.ai", None)) is None
    assert yp.ssh_endpoint(pod("RUNNING", "", 30000)) is None
    assert yp.ssh_endpoint(None) is None
    assert yp.ssh_endpoint({"id": POD_ID, "status": "RUNNING"}) is None


@pytest.mark.parametrize(
    "status, target, previous, decision",
    [
        ("RUNNING", "RUNNING", None, "done"),
        ("INITIALIZING", "RUNNING", None, "wait"),
        ("PAUSED", "RUNNING", None, "wait"),  # resume not taken yet
        ("PAUSED", "RUNNING", "PAUSED", "wait"),
        ("PAUSED", "RUNNING", "INITIALIZING", "no_capacity"),
        ("FAILED", "RUNNING", None, "failed"),
        ("TERMINATING", "RUNNING", None, "failed"),
        ("GONE", "RUNNING", None, "failed"),
        ("RUNNING", "PAUSED", None, "wait"),
        ("PAUSING", "PAUSED", "RUNNING", "wait"),
        ("PAUSED", "PAUSED", "PAUSING", "done"),
        ("FAILED", "PAUSED", None, "failed"),
        ("TERMINATING", "GONE", None, "wait"),
        ("RUNNING", "GONE", None, "wait"),
        ("GONE", "GONE", "TERMINATING", "done"),
    ],
)
def test_next_poll_decision(status, target, previous, decision):
    assert yp.next_poll_decision(status, target, previous) == decision


# --- the capacity watcher -------------------------------------------------------------------


def test_wait_for_capacity_creates_once():
    t = FakeTransport(
        {
            ("GET", "/v2/gpus"): [
                ok([pool("us-east-3", "UNAVAILABLE"), pool("eu-3", "UNAVAILABLE")]),
                ok([pool("us-east-3", "UNAVAILABLE"), pool("eu-3", "UNAVAILABLE")]),
                ok([pool("us-east-3", "UNAVAILABLE"), pool("eu-3", "AVAILABLE")]),
            ],
            ("POST", "/v2/pods"): ok(pod("INITIALIZING", region="eu-3")),
        }
    )
    clock = FakeClock()
    rc, out, errs = run(
        ["create", "--name", "n", "--gpus", "2", "--wait-for-capacity", "--interval", "300"],
        t,
        clock,
    )
    assert rc == 0, errs
    assert out.strip() == POD_ID
    assert t.count("POST", "/v2/pods") == 1
    assert t.count("GET", "/v2/gpus") == 3
    assert clock.sleeps == [300, 300]


def test_wait_for_capacity_keeps_waiting_through_a_12014_and_creates_one_pod():
    t = FakeTransport(
        {
            ("GET", "/v2/gpus"): ok([pool("us-east-3")]),
            ("POST", "/v2/pods"): [
                err(12014, "Pod insufficient resources"),
                ok(pod("INITIALIZING")),
            ],
        }
    )
    clock = FakeClock()
    rc, out, errs = run(
        ["create", "--name", "n", "--gpus", "1", "--wait-for-capacity", "--interval", "60"],
        t,
        clock,
    )
    assert rc == 0 and out.strip() == POD_ID
    assert t.count("POST", "/v2/pods") == 2  # the refused one created nothing
    assert clock.sleeps == [60]
    assert "insufficient" in errs.lower() or "no capacity" in errs.lower()


def test_wait_for_capacity_gives_up_after_max_wait_without_posting():
    t = FakeTransport({("GET", "/v2/gpus"): ok([pool("us-east-3", "UNAVAILABLE")])})
    clock = FakeClock()
    rc, out, errs = run(
        [
            "create",
            "--name",
            "n",
            "--gpus",
            "1",
            "--wait-for-capacity",
            "--interval",
            "300",
            "--max-wait",
            "900",
        ],
        t,
        clock,
    )
    assert rc == 2 and out == ""
    assert t.count("POST", "/v2/pods") == 0
    assert clock.now <= 900 and "no capacity" in errs.lower()


def test_wait_for_capacity_survives_a_network_blip_on_the_pool_listing():
    t = FakeTransport(
        {
            ("GET", "/v2/gpus"): [
                yp.ApiError("network", "GET /v2/gpus: timed out"),
                ok([pool("eu-3")]),
            ],
            ("POST", "/v2/pods"): ok(pod("INITIALIZING", region="eu-3")),
        }
    )
    rc, out, errs = run(["create", "--name", "n", "--gpus", "2", "--wait-for-capacity"], t)
    assert rc == 0 and out.strip() == POD_ID and t.count("POST", "/v2/pods") == 1
    assert "timed out" in errs

    # without the watcher the same blip is an API error
    t = FakeTransport({("GET", "/v2/gpus"): yp.ApiError("network", "timed out")})
    rc, _, _ = run(["create", "--name", "n", "--gpus", "2"], t)
    assert rc == 1


def test_wait_for_capacity_does_not_retry_a_post_that_may_have_landed():
    t = FakeTransport(
        {
            ("GET", "/v2/gpus"): ok([pool("us-east-3")]),
            ("POST", "/v2/pods"): yp.ApiError("network", "POST /v2/pods: timed out"),
        }
    )
    rc, out, _ = run(["create", "--name", "n", "--gpus", "1", "--wait-for-capacity"], t)
    assert rc == 1 and out == "" and t.count("POST", "/v2/pods") == 1


def test_wait_for_capacity_stops_on_a_non_capacity_api_error():
    t = FakeTransport(
        {
            ("GET", "/v2/gpus"): ok([pool("us-east-3")]),
            ("POST", "/v2/pods"): err(10001, "parameter error"),
        }
    )
    rc, out, errs = run(["create", "--name", "n", "--gpus", "1", "--wait-for-capacity"], t)
    assert rc == 1 and out == ""
    assert t.count("POST", "/v2/pods") == 1
    assert "10001" in errs and "parameter error" in errs


def test_create_without_wait_reports_no_capacity():
    t = FakeTransport({("GET", "/v2/gpus"): ok([pool("us-east-3", "UNAVAILABLE")])})
    rc, out, _ = run(["create", "--name", "n", "--gpus", "1"], t)
    assert rc == 2 and out == "" and t.count("POST", "/v2/pods") == 0

    t = FakeTransport(
        {
            ("GET", "/v2/gpus"): ok([pool("us-east-3")]),
            ("POST", "/v2/pods"): err(12014, "Pod insufficient resources"),
        }
    )
    rc, out, _ = run(["create", "--name", "n", "--gpus", "1"], t)
    assert rc == 2 and out == "" and t.count("POST", "/v2/pods") == 1


def test_create_filters_pools_by_gpu_type():
    t = FakeTransport(
        {
            ("GET", "/v2/gpus"): ok(
                [pool("us-east-3", gpu_type="NVIDIA_H100_80G"), pool("eu-3", "UNAVAILABLE")]
            ),
        }
    )
    rc, _, _ = run(["create", "--name", "n", "--gpus", "1", "--regions", "us-east-3,eu-3"], t)
    assert rc == 2 and t.count("POST", "/v2/pods") == 0


# --- the key never leaks --------------------------------------------------------------------


class FakeResponse:
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def key_file(tmp_path):
    p = tmp_path / "api_key"
    p.write_text(SECRET + "\n")
    p.chmod(0o600)
    return p


def test_dry_run_and_errors_never_print_the_key(tmp_path, monkeypatch):
    kf = key_file(tmp_path)
    monkeypatch.setenv("YOTTALABS_API_KEY_FILE", str(kf))

    # --dry-run: prints the request(s), sends nothing, shows no key
    def no_network(*a, **k):
        raise AssertionError("dry-run must not send")

    rc, out, errs = run(
        ["create", "--name", "n", "--gpus", "1", "--volume-id", str(VOLUME_ID), "--dry-run"],
        no_network,
    )
    assert rc == 0
    assert "POST" in out and "/v2/pods" in out and yp.V1_CREATE_PATH in out
    assert "persistentVolumes" in out
    assert SECRET not in out + errs

    # the real transport: the key goes in the header and nowhere else
    seen = []

    def echoing_404(req, timeout):
        seen.append(req)
        body = io.BytesIO(json.dumps({"message": f"bad key {SECRET}", "code": 10001}).encode())
        raise urllib.error.HTTPError(req.full_url, 404, f"Not Found {SECRET}", {}, body)

    transport = yp.UrllibTransport(key_file=kf, opener=echoing_404)
    rc, out, errs = run(["status", POD_ID], transport)
    assert rc == 1
    assert seen and seen[0].get_header("X-api-key") == SECRET
    assert seen[0].full_url == f"https://api.yottalabs.ai/v2/pods/{POD_ID}"
    assert SECRET not in out + errs
    envelope = transport("GET", f"/v2/pods/{POD_ID}", None)
    assert envelope["code"] == 10001 and SECRET not in json.dumps(envelope)
    with pytest.raises(yp.ApiError) as exc:
        yp.api(transport, "GET", f"/v2/pods/{POD_ID}")
    assert SECRET not in str(exc.value) and SECRET not in repr(exc.value)

    def unparseable_500(req, timeout):
        raise urllib.error.HTTPError(
            req.full_url, 500, "x", {}, io.BytesIO(f"<html>{SECRET}</html>".encode())
        )

    with pytest.raises(yp.ApiError) as exc:
        yp.api(yp.UrllibTransport(key_file=kf, opener=unparseable_500), "GET", "/v2/gpus")
    assert "HTTP 500" in str(exc.value) and SECRET not in str(exc.value)

    def unreachable(req, timeout):
        raise urllib.error.URLError(f"connection refused {SECRET}")

    rc, out, errs = run(["gpus"], yp.UrllibTransport(key_file=kf, opener=unreachable))
    assert rc == 1 and SECRET not in out + errs

    def capacity(req, timeout):
        return FakeResponse(err(12014, f"Pod insufficient resources {SECRET}"))

    def pools(req, timeout):
        if req.get_method() == "GET":
            return FakeResponse(ok([pool("us-east-3")]))
        return capacity(req, timeout)

    rc, out, errs = run(
        ["create", "--name", "n", "--gpus", "1"], yp.UrllibTransport(key_file=kf, opener=pools)
    )
    assert rc == 2 and SECRET not in out + errs


def test_missing_key_file_names_the_path(tmp_path):
    missing = tmp_path / "nope"
    rc, out, errs = run(["gpus"], yp.UrllibTransport(key_file=missing))
    assert rc == 1 and str(missing) in errs


def test_urllib_transport_sends_json_and_parses_the_envelope(tmp_path):
    kf = key_file(tmp_path)
    seen = []

    def opener(req, timeout):
        seen.append((req, timeout))
        return FakeResponse(ok(pod("INITIALIZING")))

    t = yp.UrllibTransport(key_file=kf, opener=opener)
    resp = t("POST", "/v2/pods", {"name": "n"})
    assert resp["data"]["id"] == POD_ID
    req, timeout = seen[0]
    assert req.get_method() == "POST" and json.loads(req.data) == {"name": "n"}
    assert req.get_header("Content-type") == "application/json" and timeout == 30


# --- lifecycle subcommands ------------------------------------------------------------------


def test_wait_running_prints_the_ssh_endpoint():
    host = "c3th3gvgo406k-m.proxy.yottalabs.ai"
    t = FakeTransport(
        {
            ("GET", f"/v2/pods/{POD_ID}"): [
                ok(pod("INITIALIZING")),
                ok(pod("RUNNING")),  # RUNNING before the proxy is wired: keep waiting
                ok(pod("RUNNING", host, 30000)),
            ]
        }
    )
    rc, out, _ = run(["wait-running", POD_ID], t)
    assert rc == 0 and out.strip() == f"{host} 30000"


def test_wait_running_times_out():
    t = FakeTransport({("GET", f"/v2/pods/{POD_ID}"): ok(pod("INITIALIZING"))})
    clock = FakeClock()
    rc, out, errs = run(["wait-running", POD_ID, "--timeout", "60"], t, clock)
    assert rc == 2 and out == "" and "timeout" in errs.lower()
    assert clock.now <= 60 + max(clock.sleeps)


def test_wait_running_reports_a_failed_pod():
    t = FakeTransport({("GET", f"/v2/pods/{POD_ID}"): ok(pod("FAILED"))})
    rc, _, errs = run(["wait-running", POD_ID], t)
    assert rc == 1 and "FAILED" in errs


def test_pause_polls_until_paused():
    t = FakeTransport(
        {
            ("POST", f"/v2/pods/{POD_ID}/pause"): ok(None),
            ("GET", f"/v2/pods/{POD_ID}"): [
                ok(pod("RUNNING")),
                ok(pod("PAUSING")),
                ok(pod("PAUSED")),
            ],
        }
    )
    rc, out, _ = run(["pause", POD_ID], t)
    assert rc == 0 and "PAUSED" in out
    assert t.count("POST", f"/v2/pods/{POD_ID}/pause") == 1


def test_pause_without_a_volume_says_delete_instead():
    t = FakeTransport(
        {("POST", f"/v2/pods/{POD_ID}/pause"): err(12005, "The current pod cannot be paused")}
    )
    rc, _, errs = run(["pause", POD_ID], t)
    assert rc == 1 and "12005" in errs and "delete" in errs


def test_resume_falling_back_to_paused_is_no_capacity():
    t = FakeTransport(
        {
            ("POST", f"/v2/pods/{POD_ID}/resume"): ok(None),
            ("GET", f"/v2/pods/{POD_ID}"): [
                ok(pod("PAUSED")),
                ok(pod("INITIALIZING")),
                ok(pod("PAUSED")),
            ],
        }
    )
    rc, out, errs = run(["resume", POD_ID], t)
    assert rc == 2 and "no capacity" in errs.lower()


def test_resume_reaches_running():
    host = "h.proxy.yottalabs.ai"
    t = FakeTransport(
        {
            ("POST", f"/v2/pods/{POD_ID}/resume"): ok(None),
            ("GET", f"/v2/pods/{POD_ID}"): [
                ok(pod("INITIALIZING")),
                ok(pod("RUNNING", host, 30002)),
            ],
        }
    )
    rc, out, _ = run(["resume", POD_ID], t)
    assert rc == 0 and f"{host} 30002" in out


def test_resume_that_never_leaves_paused_is_no_capacity():
    t = FakeTransport(
        {
            ("POST", f"/v2/pods/{POD_ID}/resume"): ok(None),
            ("GET", f"/v2/pods/{POD_ID}"): ok(pod("PAUSED")),
        }
    )
    clock = FakeClock()
    rc, _, errs = run(["resume", POD_ID], t, clock)
    assert rc == 2 and "no capacity" in errs.lower()
    assert clock.now < 900  # well before the resume timeout


def test_delete_polls_until_gone():
    t = FakeTransport(
        {
            ("DELETE", f"/v2/pods/{POD_ID}"): ok(None),
            ("GET", f"/v2/pods/{POD_ID}"): [ok(pod("TERMINATING")), ok(None)],
        }
    )
    rc, out, _ = run(["delete", POD_ID], t)
    assert rc == 0 and "gone" in out.lower()
    assert t.count("DELETE", f"/v2/pods/{POD_ID}") == 1


def test_status_prints_the_summary():
    t = FakeTransport(
        {("GET", f"/v2/pods/{POD_ID}"): ok(pod("RUNNING", "h.proxy.yottalabs.ai", 30000))}
    )
    rc, out, _ = run(["status", POD_ID], t)
    assert rc == 0
    assert POD_ID in out and "RUNNING" in out and "us-east-3" in out
    assert "h.proxy.yottalabs.ai 30000" in out

    t = FakeTransport({("GET", f"/v2/pods/{POD_ID}"): ok(None)})
    rc, out, _ = run(["status", POD_ID], t)
    assert rc == 0 and "GONE" in out


def test_ssh_config_prints_the_host_block():
    t = FakeTransport(
        {
            ("GET", f"/v2/pods/{POD_ID}"): ok(
                pod("RUNNING", "c3th3gvgo406k-m.proxy.yottalabs.ai", 30000)
            )
        }
    )
    rc, out, _ = run(["ssh-config", POD_ID, "--alias", "difflet-gpu"], t)
    assert rc == 0
    assert out == (
        "Host difflet-gpu\n"
        "  HostName c3th3gvgo406k-m.proxy.yottalabs.ai\n"
        "  Port 30000\n"
        "  User user\n"
        "  IdentityFile ~/.ssh/yottalabs_difflet_cuda_qwen.pem\n"
        "  StrictHostKeyChecking accept-new\n"
        "  IdentitiesOnly yes\n"
        "  ServerAliveInterval 30\n"
    )


def test_ssh_config_before_running_fails():
    t = FakeTransport({("GET", f"/v2/pods/{POD_ID}"): ok(pod("INITIALIZING"))})
    rc, out, errs = run(["ssh-config", POD_ID, "--alias", "a"], t)
    assert rc == 2 and out == "" and "INITIALIZING" in errs


def test_gpus_table_filters_by_type():
    t = FakeTransport(
        {
            ("GET", "/v2/gpus"): ok(
                [
                    pool("us-east-3", "UNAVAILABLE"),
                    pool("eu-3", counts=(2,)),
                    pool("us-central-7", gpu_type="NVIDIA_H100_80G"),
                ]
            )
        }
    )
    rc, out, _ = run(["gpus", "--type", PRO6000], t)
    assert rc == 0
    lines = out.strip().splitlines()
    assert len(lines) == 3  # header + two PRO 6000 pools
    assert "us-east-3" in lines[1] and "UNAVAILABLE" in lines[1] and "0.45" in lines[1]
    assert "eu-3" in lines[2] and "AVAILABLE" in lines[2] and "178" in lines[2]
    assert "us-central-7" not in out

    rc, out, _ = run(["gpus"], t)
    assert rc == 0 and "us-central-7" in out


def test_pods_listing_shape_is_accepted_both_ways():
    assert yp.pod_items({"items": [{"id": "1"}]}) == [{"id": "1"}]
    assert yp.pod_items([{"id": "2"}]) == [{"id": "2"}]
    assert yp.pod_items(None) == []


def test_list_prints_one_line_per_pod():
    record = {**pod("PAUSED"), "name": "difflet-f2"}
    t = FakeTransport({("GET", "/v2/pods"): ok({"items": [record]})})
    rc, out, _ = run(["list"], t)
    assert rc == 0
    (line,) = out.strip().splitlines()[1:]
    assert POD_ID in line and "difflet-f2" in line and "PAUSED" in line and "us-east-3" in line


# --- --prefer-first-for: wait for the first region, then fall back (F8b) ------------------

_FALLBACK = ["--regions", "us-east-3,eu-2", "--volume-id", str(VOLUME_ID), "--disk-gb", "250"]


def _prefer(window, interval="60"):
    return [
        "create",
        "--name",
        "n",
        "--gpus",
        "1",
        *_FALLBACK,
        "--prefer-first-for",
        window,
        "--interval",
        interval,
    ]


def test_prefer_first_for_creates_in_the_first_region_when_it_comes_back_in_time():
    t = FakeTransport(
        {
            ("GET", "/v2/gpus"): [
                ok([pool("us-east-3", "UNAVAILABLE"), pool("eu-2")]),
                ok([pool("us-east-3", "UNAVAILABLE"), pool("eu-2")]),
                ok([pool("us-east-3"), pool("eu-2")]),
            ],
            ("POST", yp.V1_CREATE_PATH): ok(POD_ID),
        }
    )
    clock = FakeClock()
    rc, out, errs = run(_prefer("1800"), t, clock)
    assert rc == 0 and out.strip() == POD_ID, errs
    # The volume pod in us-east-3 (v1), never the eu fallback.
    (body,) = t.bodies("POST", yp.V1_CREATE_PATH)
    assert body["regionList"] == ["us-east-3"] and body["persistentVolumes"][0]["id"] == VOLUME_ID
    assert t.count("POST", "/v2/pods") == 0 and clock.sleeps == [60, 60]
    assert "waiting for us-east-3" in errs


def test_prefer_first_for_falls_back_after_the_window_with_a_container_disk():
    t = FakeTransport(
        {
            ("GET", "/v2/gpus"): ok([pool("us-east-3", "UNAVAILABLE"), pool("eu-2")]),
            ("POST", "/v2/pods"): ok(pod("INITIALIZING", region="eu-2")),
        }
    )
    clock = FakeClock()
    rc, out, errs = run(_prefer("150"), t, clock)
    assert rc == 0 and out.strip() == POD_ID, errs
    (body,) = t.bodies("POST", "/v2/pods")
    assert body["regions"] == ["eu-2"] and body["containerVolumeInGb"] == 250
    assert "volumes" not in body and t.count("POST", yp.V1_CREATE_PATH) == 0
    # The sleeps stop at the window: 60 + 60 + the 30 s left, not a third full interval.
    assert clock.sleeps == [60, 60, 30] and clock.now == 150


def test_prefer_first_for_zero_is_today_s_immediate_fallback():
    t = FakeTransport(
        {
            ("GET", "/v2/gpus"): ok([pool("us-east-3", "UNAVAILABLE"), pool("eu-2")]),
            ("POST", "/v2/pods"): ok(pod("INITIALIZING", region="eu-2")),
        }
    )
    clock = FakeClock()
    rc, out, _ = run(["create", "--name", "n", "--gpus", "1", *_FALLBACK], t, clock)
    assert rc == 0 and out.strip() == POD_ID and clock.sleeps == []
    assert (
        yp.build_parser().parse_args(["create", "--name", "n", "--gpus", "1"]).prefer_first_for == 0
    )


def test_prefer_first_for_with_nothing_free_anywhere_waits_only_with_the_watcher():
    listing = ok([pool("us-east-3", "UNAVAILABLE"), pool("eu-2", "UNAVAILABLE")])
    t = FakeTransport({("GET", "/v2/gpus"): listing})
    clock = FakeClock()
    rc, out, _ = run(_prefer("120"), t, clock)
    # Nothing free in any region: the window still waits for the first region, then gives up
    # (no --wait-for-capacity), as a plain create does.
    assert rc == 2 and out == "" and clock.now == 120
