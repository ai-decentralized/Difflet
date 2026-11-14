"""Compare the per-rank compiled programs of a torch_tpu run (structure and collective buffers).

    python -m benchmark.tpu_profile.hlo_rank_diff <dump_base>

``<dump_base>/rank<r>/xla`` are XLA dumps (``--xla_dump_to``, one directory per
rank, e.g. ``cm_padded_repro.py`` with ``REPRO_DUMP_BASE``; pass ``--salt`` so a
cached program does not skip the compile). For the compiled graph's module it
prints, per rank: the ENTRY op count and a fingerprint with numbers stripped,
the first differing regions against rank 0, and the offsets of the
``collective-permute`` receive buffers from the buffer assignment.

On torch_tpu every rank compiles its own program and the SP rings are inlined
into it; a collective-permute writes into its peer at the receive buffer's
offset, so the programs -- and those offsets -- must agree across ranks
(2026-11-10: LTX-2's in-graph padding moved rank 3's by +18.9 MB).
"""

import difflib
import glob
import hashlib
import re
import sys

MODULE = "*fx_to_mlir*"


def _read(base, rank, suffix):
    files = sorted(glob.glob(f"{base}/rank{rank}/xla/{MODULE}{suffix}"))
    return open(files[0]).read() if files else None


def _entry(text):
    m = re.search(r"\nENTRY [^\n]*\{\n(.*?)\n\}", text, re.S)
    return m.group(1).splitlines() if m else []


def _norm(line):
    line = re.sub(r", metadata=\{[^}]*\}", "", line)
    line = re.sub(r", backend_config=.*", "", line)
    return re.sub(r"\d+", "#", line).strip()


def main() -> int:
    base = sys.argv[1]
    entries = {}
    print("== ENTRY computation per rank (numbers stripped for the fingerprint)")
    for r in range(4):
        text = _read(base, r, ".after_optimizations.txt")
        if text is None:
            print(f"rank{r}: no dump")
            continue
        entries[r] = [_norm(line) for line in _entry(text)]
        digest = hashlib.md5("\n".join(entries[r]).encode()).hexdigest()[:10]
        print(f"rank{r}: {len(entries[r])} ops, fingerprint {digest}")
    for r in (1, 2, 3):
        if 0 not in entries or r not in entries:
            continue
        sm = difflib.SequenceMatcher(None, entries[0], entries[r], autojunk=False)
        diffs = [op for op in sm.get_opcodes() if op[0] != "equal"]
        print(f"rank0 vs rank{r}: {len(diffs)} differing regions")
        for tag, i1, i2, j1, j2 in diffs[:3]:
            print(f"  {tag} rank0[{i1}:{i2}] {entries[0][i1:i2][:1]}")
            print(f"  {' ' * len(tag)} rank{r}[{j1}:{j2}] {entries[r][j1:j2][:1]}")
    print("== collective-permute receive buffers (buffer assignment)")
    for r in range(4):
        text = _read(base, r, ".after_optimizations-buffer-assignment.txt")
        if text is None:
            continue
        found = re.findall(r"<\d+ (collective-permute-start[.\d]*)\{1\} @1> \(size=(\d+),offset=(\d+)\)", text)
        print(f"rank{r}: " + " ".join(f"{n}@{o}" for n, _s, o in found[:4]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
