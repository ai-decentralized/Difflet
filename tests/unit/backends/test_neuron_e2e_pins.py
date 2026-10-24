"""Pinned end-to-end outputs of the three composed neuron applications (CPU, gloo).

Why: the Wan, Qwen-Image and LTX-2 applications share their host-side plumbing (the DiT
adapter, the rank-0 encoders and their broadcast, the initial-latents broadcast, the call
dispatch). Every other test compares a run with a reference within a tolerance, or the ranks
with each other; none would notice a refactor of that plumbing that moved a cast, a seed or a
broadcast and changed the bits. These pin the sha256 of each tiny run's output (the final fp32
latents, or the decoded frames), recorded before that refactor, so the refactor has to be bit
for bit a no-op on every path it touches: TP, CFG parallelism, CP (gather_kv, ring, ulysses),
SP, the per-block compile, ``seed=``, a prompt through the host encoder, and the device VAE.

The bits depend on the CPU kernels, so the pins are keyed by the torch build that recorded
them (``PINS``); under any other build the tests skip with the hashes they measured, which is
what a new key needs.
"""

import hashlib

import pytest

torch = pytest.importorskip("torch")

from types import SimpleNamespace  # noqa: E402

from difflet.backends.neuron.ops_impl import parallel_mesh  # noqa: E402
from tests.unit.backends._neuron_gloo import run_ranks  # noqa: E402
from tests.unit.backends._neuron_ltx2_toy import (  # noqa: E402
    l2_app_worker,
    write_tiny_ltx2_snapshot,
)
from tests.unit.backends._neuron_qwen_app_toy import (  # noqa: E402
    q2_pipeline_worker,
    q5_true_cfg_worker,
    tokenizer_available,
    write_tiny_qwen_snapshot,
)
from tests.unit.backends._neuron_wan_toy import write_tiny_wan_model  # noqa: E402
from tests.unit.backends._neuron_workers import (  # noqa: E402
    w1_wan_cfg_parallel_worker,
    w1_wan_cp_worker,
    w1_wan_device_vae_worker,
    w1_wan_initial_noise_worker,
    w1_wan_pipeline_worker,
    w1_wan_sp_worker,
)

OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

#: torch build -> case -> sha256 of the output every rank returned.
PINS: dict[str, dict[str, str]] = {
    "2.14.0+cpu": {
        "wan_tp4": "7fadce8629f756a9d5e8d66793ac963486a2eab9b00cad87d9a4852ef533c76a",
        "wan_tp2_cfg2": "79c801d78bd12e70d81596546b610b44ad0980204c863ddc6d77ed303fd59c86",
        "wan_tp2_cp2_gather_kv": "0f45c29849fbdb89b15bd8131e65755791735266576fcab0871d7dfcd62b3fa5",
        "wan_tp2_cp2_ring": "f1d06fa3089da89d2be5e23158060054f8488edb65ee5fe52102f4abf73c5822",
        "wan_tp2_cp2_ulysses": "0f45c29849fbdb89b15bd8131e65755791735266576fcab0871d7dfcd62b3fa5",
        "wan_tp4_sp": "6a06f50aad1a8d9d591b596bba6f2ba119dc4381682322794a4ba0d59201c484",
        "wan_seed_noise": "5f919088c66d46496b72530bff8cea6502960d783862c6612f63e61ec9b7a8dd",
        "wan_tp2_device_vae": "a937d321e808b437860480895f18471ee55ff2f65870093031ad6290fa77bcb3",
        "qwen_tp2_cp2_gather_kv": "b6e93895734eba1f7597816a3cbb83bf0e12b2b9be2e5f14b7f84f494c7b68d2",
        "qwen_tp2_cp2_ring": "56b2365137798affac1be0f14e334f46b095d59ff249f2a12b11639518f24a52",
        "qwen_tp2_cp2_ulysses": "eead04f340dc03af6cea185b02de5a198fd6f1cecc61cb53025c2558a8f7c8c2",
        "qwen_tp2_sp_compile": "8fe459b4cbdce94704fd37a640a1e002ed8b54877ded11eee182db1eac7e9ac1",
        "qwen_tp2_prompt": "60a665d3d6fde1f560bae39432a2fb656c2d2764e470a08ef6677341575c7a10",
        "qwen_tp2_device_vae": "1b7aa96c609c05bf89db6b4eb366897a3c2c8ea411e681af076c4f761cb34732",
        "qwen_tp2_cfg2_true_cfg": "e0806dcca3a207c21a7c3067bfafb70623e18f5ed09cdc8648468475943118e1",
        "ltx2_tp1": "8de1f6beb034d18d32c9991ca9265f5b4a609471e65b0ca3cb50d8f6f045101e",
        "ltx2_tp2_cfg2": "8762f76bd8bca6e896206b1a98d6f13a873d299a21e829331ac470d79742af9a",
        "ltx2_tp2_compile": "8762f76bd8bca6e896206b1a98d6f13a873d299a21e829331ac470d79742af9a",
        "ltx2_tp2_cp2_gather_kv": "8762f76bd8bca6e896206b1a98d6f13a873d299a21e829331ac470d79742af9a",
        "ltx2_tp2_cp2_ulysses": "8762f76bd8bca6e896206b1a98d6f13a873d299a21e829331ac470d79742af9a",
    },
}


@pytest.fixture(autouse=True)
def _clean_mesh(monkeypatch):
    monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
    parallel_mesh.destroy_parallel_mesh()
    yield
    parallel_mesh.destroy_parallel_mesh()


@pytest.fixture(scope="module")
def snapshots(tmp_path_factory):
    for module in ("safetensors.torch", "accelerate", "diffusers", "transformers"):
        pytest.importorskip(module)
    wan = tmp_path_factory.mktemp("pin_wan")
    write_tiny_wan_model(wan)
    qwen = tmp_path_factory.mktemp("pin_qwen")
    write_tiny_qwen_snapshot(qwen)
    ltx = tmp_path_factory.mktemp("pin_ltx2")
    write_tiny_ltx2_snapshot(ltx)
    return SimpleNamespace(wan=str(wan), qwen=str(qwen), ltx=str(ltx))


def _sha(*tensors) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(torch.tensor(tensor, dtype=torch.float32).contiguous().numpy().tobytes())
    return digest.hexdigest()


def _check(case: str, hashes: list[str]) -> None:
    assert len(set(hashes)) == 1, f"{case}: the ranks disagree: {hashes}"
    pins = PINS.get(torch.__version__)
    if pins is None or case not in pins:
        pytest.skip(f"no pin for {case} under torch {torch.__version__}: measured {hashes[0]}")
    assert hashes[0] == pins[case], f"{case}: {hashes[0]} != pinned {pins[case]}"


# ----------------------------------------------------------------------------------- Wan


def test_wan_tp4(snapshots):
    results = run_ranks(w1_wan_pipeline_worker, snapshots.wan, world_size=4, env=OFFLINE)
    _check("wan_tp4", [r["sha256"] for r in results])


def test_wan_tp2_cfg2(snapshots):
    results = run_ranks(w1_wan_cfg_parallel_worker, snapshots.wan, world_size=4, env=OFFLINE)
    _check("wan_tp2_cfg2", [r["sha256"] for r in results])


@pytest.mark.parametrize("cp_mode", ["gather_kv", "ring", "ulysses"])
def test_wan_tp2_cp2(snapshots, cp_mode):
    results = run_ranks(w1_wan_cp_worker, snapshots.wan, cp_mode, world_size=4, env=OFFLINE)
    _check(f"wan_tp2_cp2_{cp_mode}", [r["sha256"] for r in results])


def test_wan_tp4_sp(snapshots):
    results = run_ranks(w1_wan_sp_worker, snapshots.wan, False, world_size=4, env=OFFLINE)
    _check("wan_tp4_sp", [r["sha256"] for r in results])


def test_wan_initial_noise_from_seed(snapshots):
    results = run_ranks(w1_wan_initial_noise_worker, snapshots.wan, world_size=4, env=OFFLINE)
    _check("wan_seed_noise", [_sha(r["seed"], r["global_rng"]) for r in results])


def test_wan_device_vae_frames(snapshots):
    results = run_ranks(w1_wan_device_vae_worker, snapshots.wan, world_size=2, env=OFFLINE)
    _check("wan_tp2_device_vae", [r["sha256"] for r in results])


# ---------------------------------------------------------------------------- Qwen-Image


@pytest.mark.parametrize("cp_mode", ["gather_kv", "ring", "ulysses"])
def test_qwen_tp2_cp2(snapshots, cp_mode):
    results = run_ranks(q2_pipeline_worker, snapshots.qwen, 2, cp_mode, "eager", False,
                        world_size=4, env=OFFLINE)
    _check(f"qwen_tp2_cp2_{cp_mode}", [r["sha256"] for r in results])


def test_qwen_tp2_sp_compiled(snapshots):
    results = run_ranks(q2_pipeline_worker, snapshots.qwen, 2, "gather_kv", "compile", False,
                        "latent", True, world_size=2, env=OFFLINE)
    _check("qwen_tp2_sp_compile", [r["sha256"] for r in results])


def test_qwen_prompt_through_the_host_encoder(snapshots):
    if not tokenizer_available():
        pytest.skip("the real Qwen-Image tokenizer is not on this machine")
    results = run_ranks(q2_pipeline_worker, snapshots.qwen, 2, "gather_kv", "eager", True,
                        world_size=2, timeout=300.0, env=OFFLINE)
    _check("qwen_tp2_prompt", [_sha(r["latents"], r["embeds"]) for r in results])


def test_qwen_device_vae_images(snapshots):
    results = run_ranks(q2_pipeline_worker, snapshots.qwen, 2, "gather_kv", "eager", False,
                        "pt", world_size=2, env=OFFLINE)
    _check("qwen_tp2_device_vae", [_sha(r["latents"], r["images"]) for r in results])


def test_qwen_true_cfg_parallel(snapshots):
    results = run_ranks(q5_true_cfg_worker, snapshots.qwen, 2, True, "eager", 4.0,
                        world_size=4, env=OFFLINE)
    _check("qwen_tp2_cfg2_true_cfg", [r["sha256"] for r in results])


# --------------------------------------------------------------------------------- LTX-2


def test_ltx2_tp1(snapshots):
    results = run_ranks(l2_app_worker, snapshots.ltx, 1, False, 1.0, 3, world_size=1,
                        env=OFFLINE)
    _check("ltx2_tp1", [_sha(r["video"], r["audio"]) for r in results])


def test_ltx2_tp2_cfg2(snapshots):
    results = run_ranks(l2_app_worker, snapshots.ltx, 2, True, 3.0, 3, world_size=4,
                        env=OFFLINE)
    _check("ltx2_tp2_cfg2", [_sha(r["video"], r["audio"]) for r in results])


def test_ltx2_tp2_compiled(snapshots):
    results = run_ranks(l2_app_worker, snapshots.ltx, 2, False, 3.0, 3, "compile",
                        world_size=2, env=OFFLINE)
    _check("ltx2_tp2_compile", [_sha(r["video"], r["audio"]) for r in results])


@pytest.mark.parametrize("cp_mode", ["gather_kv", "ulysses"])
def test_ltx2_tp2_cp2(snapshots, cp_mode):
    results = run_ranks(l2_app_worker, snapshots.ltx, 2, False, 3.0, 3, "eager", 2, cp_mode,
                        world_size=4, env=OFFLINE)
    _check(f"ltx2_tp2_cp2_{cp_mode}", [_sha(r["video"], r["audio"]) for r in results])


# ------------------------------------------------------------------------- the weight store
# The same pins with the pre-sharded weight store on (``core/weight_store.py``; the unit-test
# conftest turns it off everywhere else): a first run writes every device component's entry, a
# second run reads them, and both must give the pinned bits.


def _out_sha(result) -> str:
    return result["sha256"]


def _qwen_images_sha(result) -> str:
    return _sha(result["latents"], result["images"])


def _ltx2_sha(result) -> str:
    return _sha(result["video"], result["audio"])


_STORE_CASES = {
    "wan_tp4": (w1_wan_pipeline_worker, "wan", (), 4, _out_sha),
    "wan_tp2_device_vae": (w1_wan_device_vae_worker, "wan", (), 2, _out_sha),
    "qwen_tp2_cp2_gather_kv": (q2_pipeline_worker, "qwen", (2, "gather_kv", "eager", False), 4,
                               _out_sha),
    "qwen_tp2_device_vae": (q2_pipeline_worker, "qwen", (2, "gather_kv", "eager", False, "pt"),
                            2, _qwen_images_sha),
    "ltx2_tp2_cfg2": (l2_app_worker, "ltx", (2, True, 3.0, 3), 4, _ltx2_sha),
}


def _manifests(root):
    return {str(p): p.stat().st_mtime_ns for p in root.rglob("*.manifest.json")}


@pytest.mark.parametrize("case", sorted(_STORE_CASES))
def test_e2e_pins_with_the_store(snapshots, tmp_path, case):
    worker, model, args, world, digest = _STORE_CASES[case]
    # The tiny snapshots are fp32 and load in fp32: written only with ..._STORE_ALL.
    env = {**OFFLINE, "DIFFLET_NEURON_WEIGHT_STORE": str(tmp_path / "store"),
           "DIFFLET_NEURON_WEIGHT_STORE_ALL": "1"}
    first = run_ranks(worker, getattr(snapshots, model), *args, world_size=world, env=env)
    written = _manifests(tmp_path / "store")
    assert written, "the first run wrote no store entry"
    second = run_ranks(worker, getattr(snapshots, model), *args, world_size=world, env=env)
    assert _manifests(tmp_path / "store") == written  # read, not rewritten
    _check(case, [digest(r) for r in first])
    _check(case, [digest(r) for r in second])
