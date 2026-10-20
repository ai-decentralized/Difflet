#!/usr/bin/env python3
"""scan_public.py — stop internal details from reaching a public repository.

Scans exactly what a push or a pull request would publish:

  --range A..B       every commit in A..B: added lines, added paths and binaries, commit messages
  --commits SHA ...  the same for specific commits
  --text FILE        a PR title, description or comment draft (repeatable)
  --tree REF         every text file in a tree (an audit of what is already published)

Rules come in two sets:
  * generic rules (below) are safe to publish, so the same script can run in a public CI;
  * a private denylist adds regexes that name internal things (company, provider, hosts, pods,
    volumes, accounts). It is read from --denylist, $PUBLIC_HYGIENE_DENYLIST, or
    ~/.config/public-hygiene/denylist.txt, and is never committed anywhere public.

Suppress a reviewed false positive with an allow file (default: <repo>/.public-hygiene-allow,
lines `<path glob><TAB><regex>`) or an inline `public-hygiene: ok` marker on the same line.

Exit 0: nothing blocking. Exit 1: blocking findings. Exit 2: usage error.
Warnings (lifecycle or cost narration) are printed but only fail with --strict.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

BLOCK, WARN = "BLOCK", "WARN"

MEDIA_EXT = {
    "png", "jpg", "jpeg", "gif", "webp", "bmp", "tif", "tiff", "mp4", "mov", "webm", "avi", "mkv",
    "wav", "mp3", "flac", "npy", "npz", "pt", "pth", "ckpt", "safetensors", "bin", "onnx", "pkl",
    "pickle", "h5", "zip", "tar", "gz", "tgz", "7z", "parquet", "nsys-rep", "ncu-rep", "qdrep",
}
RAW_EXT = {"log", "out", "err", "trace", "prof", "pstats", "sqlite", "qdstrm", "nvprof", "etl"}
RAW_NAME = re.compile(r"(?:^|/)logs?/|trace\.json$|(?:^|/)(?:results?|metrics|timings?)[\w.-]*\.(?:json|jsonl|csv)$", re.I)
MAX_BYTES = 512 * 1024
PROSE_EXT = {"md", "txt", "rst", "adoc"}

ALLOWED_EMAIL = re.compile(
    r"(?:noreply@anthropic\.com|noreply@github\.com|[\w.+-]+@users\.noreply\.github\.com|git@github\.com"
    r"|[\w.+-]+@example\.(?:com|org|net)|[\w.+-]+@localhost)$",
    re.I,
)


@dataclass(frozen=True)
class Rule:
    id: str
    level: str
    pattern: re.Pattern
    why: str
    prose_only: bool = False
    skip_after: re.Pattern | None = None  # a match right after this context is not a finding


def _r(p: str, flags: int = 0) -> re.Pattern:
    return re.compile(p, flags)


GENERIC = [
    Rule("cloud-id", BLOCK, _r(r"(?<![\w.])\d{17,20}(?![\w.])"), "17-20 digit number: looks like a pod/instance/volume ID"),
    Rule("instance-id", BLOCK, _r(r"\bi-[0-9a-f]{8}(?:[0-9a-f]{9})?\b"), "cloud instance ID"),
    Rule("gpu-uuid", BLOCK, _r(r"\bGPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){0,3}"), "GPU UUID identifies a physical card"),
    Rule("price", BLOCK, _r(r"~\$\d+(?:\.\d+)?|\$\d+\.\d+|\$\d+(?:\.\d+)?\s*/\s*(?:h|hr|hour|mo|month|GB|GPU|day)\b"
                            r"|\b\d+(?:\.\d+)?\s*(?:USD|EUR)\s*/\s*(?:h|hr|hour|mo|month)\b", re.I), "price or cost"),
    Rule("region", BLOCK, _r(r"\b(?:us|eu|ap|sa|ca|me|af|il|mx)-(?:east|west|north|south|central|northeast|southeast"
                             r"|northwest|southwest)-\d+\b|\b(?:us|europe|asia|australia|northamerica|southamerica)"
                             r"-(?:east|west|north|south|central|northeast|southeast)\d+\b"), "cloud region"),
    Rule("secret", BLOCK, _r(r"\bAKIA[0-9A-Z]{16}\b|\bhf_[A-Za-z0-9]{30,}\b|\bgh[pousr]_[A-Za-z0-9]{30,}\b"
                             r"|\bgithub_pat_[A-Za-z0-9_]{20,}\b|\bsk-[A-Za-z0-9_-]{20,}\b|\bxox[abprs]-[A-Za-z0-9-]{10,}"
                             r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"), "credential or token"),
    Rule("key-path", BLOCK, _r(r"~?/?\.ssh/[\w.-]+\.(?:pem|key)\b|~?/\.ssh/id_[\w]+\b|~?/?\.config/[\w.-]+/"
                               r"(?:api_?key|token|credentials?|secret)s?\b|\.aws/credentials\b|\.netrc\b"), "credential file path"),  # public-hygiene: ok
    # (?<![\w.=v~^<>!]): a dotted quad after ==, ~=, >= or v is a version pin, not a host (user@IP is a host)
    Rule("ipv4", BLOCK, _r(r"(?<![\w.=v~^<>!])(?!127\.0\.0\.1\b|0\.0\.0\.0\b|192\.0\.2\.|198\.51\.100\.|203\.0\.113\.|255\.)"
                           r"(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\w.])"), "IP address of a host",
         # "Driver: 2.28.0.0", "aws-neuronx-dkms 2.30.2.0", "runtime-lib 2.34.10.0" are versions
         skip_after=_r(r"(?:version|driver|runtime|sdk|dkms|lib|toolkit|release|firmware|neuron[\w-]*|cuda|nccl|torch"
                       r"|python|pip)\W{0,3}$", re.I)),
    Rule("lifecycle", WARN, _r(r"\b(?:pod|instance|vm|node|volume|box|cluster)s?\b[^.\n]{0,60}?\b(?:terminated|paused|resumed"
                               r"|deleted|stopped|spun up|torn down|read back)\b|\bowner's (?:word|yes|rule)\b"
                               r"|\bno capacity\b|\bINITIALIZING\b", re.I), "infrastructure lifecycle narration", True),
    Rule("cost-talk", WARN, _r(r"\b(?:billed|billing|invoice|per hour|hourly rate|pod cost|spend)\b", re.I),  # public-hygiene: ok
         "billing narration", True),
]
EMAIL = _r(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b")


def load_denylist(path: str | None) -> list[Rule]:
    cand = [path, os.environ.get("PUBLIC_HYGIENE_DENYLIST"), str(Path.home() / ".config/public-hygiene/denylist.txt")]
    for c in cand:
        if c and Path(c).expanduser().is_file():
            rules = []
            for i, line in enumerate(Path(c).expanduser().read_text().splitlines(), 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    rules.append(Rule(f"private:{i}", BLOCK, re.compile(line, re.I), "matches the private denylist"))
                except re.error as e:
                    sys.exit(f"denylist {c}:{i}: bad regex {line!r}: {e}")
            return rules
    return []


def load_allow(path: str | None, repo: Path | None) -> list[tuple[str, re.Pattern]]:
    p = Path(path) if path else (repo / ".public-hygiene-allow" if repo else None)
    out = []
    if p and p.is_file():
        for line in p.read_text().splitlines():
            if line.strip() and not line.startswith("#") and "\t" in line:
                glob, rx = line.split("\t", 1)
                out.append((glob.strip(), re.compile(rx.strip())))
    return out


@dataclass(frozen=True)
class Finding:
    level: str
    where: str
    rule: str
    snippet: str
    why: str


class Scanner:
    def __init__(self, rules: list[Rule], allow: list[tuple[str, re.Pattern]], repo: Path | None):
        self.rules, self.allow, self.repo = rules, allow, repo
        self.findings: dict[tuple, Finding] = {}

    def _allowed(self, path: str, line: str, match: str) -> bool:
        if "public-hygiene: ok" in line:
            return True
        return any(fnmatch.fnmatch(path, g) and (rx.search(match) or rx.search(line)) for g, rx in self.allow)

    def _add(self, level, where, rule, snippet, why, key):
        if key not in self.findings:
            self.findings[key] = Finding(level, where, rule, snippet.strip()[:160], why)

    def text(self, path: str, lineno: int | str, line: str, prose: bool, origin: str = "") -> None:
        loc = f"{origin}{path}:{lineno}"
        for r in self.rules:
            if r.prose_only and not prose:
                continue
            for m in r.pattern.finditer(line):
                if r.skip_after and r.skip_after.search(line[max(0, m.start() - 40):m.start()]):
                    continue
                if not self._allowed(path, line, m.group(0)):
                    self._add(r.level, loc, r.id, line, r.why, (r.id, path, m.group(0).lower()))
        for m in EMAIL.finditer(line):
            e = m.group(0)
            if not ALLOWED_EMAIL.search(e) and not self._allowed(path, line, e):
                self._add(BLOCK, loc, "email", line, "an email address identifies a person or account", ("email", path, e.lower()))

    def path(self, path: str, origin: str, size: int | None, binary: bool, added: bool) -> None:
        ext = path.rsplit(".", 1)[-1].lower() if "." in path.rsplit("/", 1)[-1] else ""
        loc = f"{origin}{path}"
        checks = []
        if added and path.startswith(".claude/"):
            checks.append(("claude-dir", "agent skills/settings are internal runbooks: keep them out of public repos"))
        if ext in MEDIA_EXT or binary:
            checks.append(("artifact", "binary or media artifact: history keeps it forever; use release assets or storage"))
        if size is not None and size > MAX_BYTES:
            checks.append(("large-file", f"{size // 1024} KB file: artifacts belong outside the repo"))
        for rid, why in checks:
            if not self._allowed(path, path, path):
                self._add(BLOCK, loc, rid, path, why, (rid, path))
        # A scrubbed log is still a raw log: warn (not block), since some projects keep logs on purpose.
        if added and (ext in RAW_EXT or RAW_NAME.search(path)) and not self._allowed(path, path, path):
            self._add(WARN, loc, "raw-evidence", path,
                      "raw log/trace/result dump: transcribe the numbers into the doc, keep the file internal", ("raw-evidence", path))


def git(repo: Path, *a: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, errors="replace")
    if r.returncode:
        sys.exit(f"git {' '.join(a)} failed: {r.stderr.strip()}")
    return r.stdout


def scan_commit(s: Scanner, repo: Path, c: str) -> None:
    short = c[:8]
    msg = git(repo, "log", "-1", "--format=%B", c)
    for i, line in enumerate(msg.splitlines(), 1):
        s.text("(commit message)", i, line, True, origin=f"{short} ")
    numstat = git(repo, "show", "--format=", "--numstat", "--no-renames", c)
    status = dict(
        (p, st) for st, p in (l.split("\t", 1) for l in git(repo, "show", "--format=", "--name-status", "--no-renames", c).splitlines() if "\t" in l)
    )
    for line in numstat.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        a, _d, p = parts[0], parts[1], parts[2]
        if status.get(p) == "D":
            continue
        size = None
        try:
            size = int(git(repo, "cat-file", "-s", f"{c}:{p}").strip())
        except SystemExit:
            pass
        s.path(p, f"{short} ", size, a == "-", status.get(p) == "A")
    diff = git(repo, "show", "--format=", "--unified=0", "--no-color", "--no-ext-diff", "--no-renames", c)
    path, ln = None, 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            ln = int(m.group(1)) if m else 0
        elif line.startswith("+") and path:
            ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
            s.text(path, ln, line[1:], ext in PROSE_EXT, origin=f"{short} ")
            ln += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--range", dest="rng")
    ap.add_argument("--commits", nargs="*", default=[])
    ap.add_argument("--text", nargs="+", action="extend", default=[], metavar="FILE")
    ap.add_argument("--tree")
    ap.add_argument("--denylist")
    ap.add_argument("--no-private", action="store_true", help="generic rules only (what a public CI can run)")
    ap.add_argument("--allow")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="summary line only")
    a = ap.parse_args()
    if not (a.rng or a.commits or a.text or a.tree):
        ap.print_usage(sys.stderr)
        return 2
    repo = Path(a.repo).resolve()
    in_git = (repo / ".git").exists() or subprocess.run(["git", "-C", str(repo), "rev-parse"], capture_output=True).returncode == 0
    private = [] if a.no_private else load_denylist(a.denylist)
    s = Scanner(GENERIC + private, load_allow(a.allow, repo if in_git else None), repo)
    commits = list(a.commits)
    if a.rng:
        commits += git(repo, "rev-list", "--reverse", a.rng).split()
    for c in commits:
        scan_commit(s, repo, c)
    for f in a.text:
        for i, line in enumerate(Path(f).read_text(errors="replace").splitlines(), 1):
            s.text(Path(f).name, i, line, True)
    if a.tree:
        for rec in git(repo, "ls-tree", "-r", "--long", a.tree).splitlines():
            meta, p = rec.split("\t", 1)
            size = int(meta.split()[3]) if meta.split()[3].isdigit() else None
            blob = subprocess.run(["git", "-C", str(repo), "cat-file", "blob", f"{a.tree}:{p}"], capture_output=True).stdout
            binary = b"\0" in blob[:8000]
            s.path(p, "", size, binary, False)
            if not binary:
                ext = p.rsplit(".", 1)[-1].lower() if "." in p else ""
                for i, line in enumerate(blob.decode(errors="replace").splitlines(), 1):
                    s.text(p, i, line, ext in PROSE_EXT)
    blocks = [f for f in s.findings.values() if f.level == BLOCK]
    warns = [f for f in s.findings.values() if f.level == WARN]
    if not a.quiet:
        for f in sorted(s.findings.values(), key=lambda f: (f.level != BLOCK, f.where)):
            print(f"{f.level:5} {f.where}  [{f.rule}] {f.snippet}")
            print(f"      ↳ {f.why}")
    src = f"{len(commits)} commit(s)" + (f", {len(a.text)} text file(s)" if a.text else "") + (f", tree {a.tree}" if a.tree else "")
    priv = "no private denylist" if not private else f"{len(private)} private rule(s)"
    print(f"public-hygiene: {len(blocks)} blocking, {len(warns)} warning(s) — scanned {src}; {priv}")
    return 1 if blocks or (a.strict and warns) else 0


if __name__ == "__main__":
    sys.exit(main())
