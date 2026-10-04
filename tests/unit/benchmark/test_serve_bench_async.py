"""benchmark.serve_bench's async Videos-job client against a stub of the six
/v1/videos routes: create -> poll -> content -> DELETE, latency accounting,
the closed-loop level summary and the open-loop burst (admission cap -> 429)."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from benchmark import serve_bench as sb
from benchmark.models import resolve


class _Stub:
    """Jobs complete `service_s` after they reach the head of a FIFO with one
    worker; capacity = 1 + max_queued; content = `size` bytes."""

    def __init__(self, service_s=0.3, max_queued=2, size=1000):
        self.service_s, self.capacity, self.size = service_s, 1 + max_queued, size
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.n = 0
        self.deleted: list[str] = []
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def active(self):
        return [j for j in self.jobs.values() if j["status"] in ("queued", "in_progress")]

    def create(self):
        with self.lock:
            if len(self.active()) >= self.capacity:
                return 429, {"error": {"code": "queue_full"}}
            self.n += 1
            vid = f"video_gen_{self.n:032d}"
            self.jobs[vid] = {"id": vid, "object": "video", "status": "queued", "progress": 0,
                              "created_at": int(time.time()), "completed_at": None, "error": None}
            return 200, self.jobs[vid]

    def _run(self):
        while True:
            with self.lock:
                q = [j for j in self.jobs.values() if j["status"] == "queued"]
                job = q[0] if q else None
                if job:
                    job["status"] = "in_progress"
            if job is None:
                time.sleep(0.01)
                continue
            time.sleep(self.service_s)
            with self.lock:
                job["status"], job["progress"] = "completed", 100
                job["completed_at"] = int(time.time())


def _serve(stub: _Stub):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path == "/v1/videos":
                self._json(*stub.create())
            elif self.path == "/v1/videos/sync":
                time.sleep(stub.service_s)
                self.send_response(200)
                self.send_header("Content-Length", str(stub.size))
                self.end_headers()
                self.wfile.write(b"m" * stub.size)
            else:
                self._json(404, {"error": "no route"})

        def do_GET(self):
            parts = self.path.strip("/").split("/")
            if len(parts) == 4 and parts[3] == "content":
                job = stub.jobs.get(parts[2])
                if not job or job["status"] != "completed":
                    return self._json(409, {"error": {"code": "not_completed"}})
                self.send_response(200)
                self.send_header("Content-Length", str(stub.size))
                self.end_headers()
                self.wfile.write(b"m" * stub.size)
            elif len(parts) == 3:
                job = stub.jobs.get(parts[2])
                self._json(200, job) if job else self._json(404, {"error": "unknown"})
            else:
                self._json(404, {"error": "no route"})

        def do_DELETE(self):
            vid = self.path.strip("/").split("/")[-1]
            if vid in stub.jobs:
                stub.deleted.append(vid)
                stub.jobs.pop(vid)
                self._json(200, {"id": vid, "object": "video", "deleted": True})
            else:
                self._json(404, {"error": "unknown"})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


@pytest.fixture
def stub():
    s = _Stub()
    srv, base = _serve(s)
    yield s, base
    srv.shutdown()


CFG = resolve("wan_2_1", "tp4")


def test_async_request_creates_polls_downloads_and_deletes(stub):
    s, base = stub
    code, nbytes, err, extra = sb._video_async_request(base, CFG, timeout=10, poll_s=0.05)
    assert (code, nbytes, err) == (200, 1000, "")
    assert extra["create_http"] == 200 and extra["create_s"] < 0.2
    assert extra["job_status"] == "completed" and extra["polls"] >= 1
    assert extra["terminal_s"] >= s.service_s
    assert extra["latency_s"] >= extra["terminal_s"] and "download_s" in extra
    assert extra["delete_http"] == 200 and s.deleted == [extra["job_id"]]
    assert s.jobs == {}                                   # nothing retained


def test_sync_request_is_unchanged_but_returns_the_4_tuple(stub):
    s, base = stub
    assert sb._video_request(base, CFG, timeout=10) == (200, 1000, "", {})


def test_closed_loop_level_reports_async_summary_and_queueing(stub):
    s, base = stub
    send = lambda: sb._video_async_request(base, CFG, timeout=10, poll_s=0.05)  # noqa: E731
    lv = sb.run_level(send, concurrency=2, n_requests=4)
    assert lv["successes"] == 4 and lv["http_codes"] == {"200": 4}
    a = lv["async"]
    assert a["create_s"]["p50"] < 0.2 and a["polls_mean"] >= 1
    assert a["overhead_s"]["p50"] >= 0
    # one worker: with c=2 the second request of each pair waits ~one service time
    assert lv["latency_s"]["max"] >= 2 * s.service_s - 0.1
    assert all("delete_s" in r for r in lv["records"])
    assert s.jobs == {}


def test_burst_measures_admission_cap_and_completion_profile(stub):
    s, base = stub                                        # capacity 3 -> 5 creates = 3 admitted + 2x429
    b = sb.run_burst(base, CFG, n=5, timeout=10, poll_s=0.05)
    assert b["jobs"] == 5 and b["admitted"] == 3 and b["completed"] == 3
    assert b["http_codes"] == {"200": 3, "429": 2}
    assert b["all_acked_s"] < 1.0 and b["create_s"]["max"] < 0.5
    assert len(b["completion_s"]) == 3 and b["completion_s"] == sorted(b["completion_s"])
    assert all(x >= s.service_s * 0.9 for x in b["inter_completion_s"])
    assert b["throughput_per_hour"] > 0
    assert all(r.get("bytes") == 1000 for r in b["records"] if r.get("job_status") == "completed")
    assert s.jobs == {} and len(s.deleted) == 3


def test_image_models_refuse_async_mode(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("sys.argv", ["serve_bench", "--model", "flux_1_dev", "--api-mode", "async",
                                     "--out", str(tmp_path / "x.json")])
    assert sb.main() == 2
    assert "no async endpoint" in capsys.readouterr().err
