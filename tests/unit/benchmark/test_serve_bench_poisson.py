"""benchmark.serve_bench.run_poisson: open-loop arrivals, routing, SLO accounting."""
import threading
import time

from benchmark.serve_bench import run_poisson


def _server(service_s: float, log: list, lock: threading.Lock):
    busy = threading.Lock()                  # one resident worker: FIFO by lock

    def send():
        with busy:
            time.sleep(service_s)
        with lock:
            log.append(time.perf_counter())
        return 200, 1, "", {}
    return send


def test_same_seed_same_arrivals_and_all_complete():
    lock = threading.Lock()
    a = run_poisson([_server(0.01, [], lock)], rate_per_s=50.0, n_arrivals=10, seed=7, slo_s=1.0)
    b = run_poisson([_server(0.01, [], lock)], rate_per_s=50.0, n_arrivals=10, seed=7, slo_s=1.0)
    assert a["successes"] == 10 and a["slo_met_frac"] == 1.0
    assert a["expected_window_s"] == b["expected_window_s"]
    assert abs(a["arrival_window_s"] - b["arrival_window_s"]) < 0.05
    assert [r["arrival"] for r in a["records"]] == list(range(10))
    assert a["records"][0]["arrival_s"] < 0.01   # first arrival at t=0


def test_open_loop_queues_and_routes_to_least_loaded():
    lock = threading.Lock()
    # arrivals far faster than service: one server queues, latency grows
    one = run_poisson([_server(0.05, [], lock)], 1000.0, 6, seed=1, slo_s=0.12)
    assert one["latency_s"]["max"] >= 0.25
    assert one["slo_met_frac"] < 1.0
    two = run_poisson([_server(0.05, [], lock), _server(0.05, [], lock)], 1000.0, 6,
                      seed=1, slo_s=0.12)
    assert two["per_server"] == {"0": 3, "1": 3}
    assert two["latency_s"]["max"] < one["latency_s"]["max"]


def test_failed_requests_do_not_meet_slo():
    def bad():
        return 500, 0, "boom", {}
    r = run_poisson([bad], 100.0, 4, seed=0, slo_s=10.0)
    assert r["successes"] == 0 and r["slo_met_frac"] == 0.0 and r["latency_s"] is None
    assert r["http_codes"] == {"500": 4}
