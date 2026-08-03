#!/usr/bin/env python3
"""Prepare a browser-based, identity-hidden review of FLUX image pairs.

The generated reviewer page contains no candidate identifiers and randomizes
both pair order and left/right placement.  A separate key file is written for
the study owner and must not be given to reviewers before their answers are
frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_cache_quality import _load_json, _validate_quality_input  # noqa: E402
from scripts.flux_cache_protocol import canonical_sha256  # noqa: E402

REVIEW_SCHEMA = "difflet-cache-blind-review"
REVIEW_SCHEMA_REVISION = 1
SEVERITY_OPTIONS = (
    ("equivalent", "No meaningful difference"),
    ("minor-difference", "Visible difference, but both are usable"),
    ("damaged", "One image is clearly worse or misses important prompt content"),
    ("unusable", "One image is unusable"),
)
PREFERENCE_OPTIONS = (
    ("left", "Left"),
    ("tie", "Tie"),
    ("right", "Right"),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_review_assignments(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Return a deterministic shuffle with independently randomized sides."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("review seed must be a nonnegative integer")
    if not comparisons:
        raise ValueError("blind review requires at least one comparison")
    generator = random.Random(seed)
    ordered = list(comparisons)
    generator.shuffle(ordered)
    assignments = []
    for index, comparison in enumerate(ordered, start=1):
        candidate_side = "left" if generator.randrange(2) == 0 else "right"
        assignments.append(
            {
                "pair_id": f"pair-{index:03d}",
                "candidate_id": comparison["candidate_id"],
                "sample_id": comparison["sample_id"],
                "prompt_index": int(comparison["prompt_index"]),
                "seed": int(comparison["seed"]),
                "prompt": comparison["prompt"],
                "candidate_side": candidate_side,
                "baseline": comparison["baseline"],
                "candidate": comparison["candidate"],
            }
        )
    return assignments


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _radio_group(pair_id: str, name: str, options: Sequence[tuple[str, str]]) -> str:
    return "".join(
        (
            "<label class=\"choice\">"
            f"<input type=\"radio\" name=\"{html.escape(pair_id)}-{html.escape(name)}\" "
            f"value=\"{html.escape(value)}\">{html.escape(label)}</label>"
        )
        for value, label in options
    )


def _review_html(rows: Sequence[Mapping[str, Any]], packet_sha256: str) -> str:
    cards = []
    for row in rows:
        pair_id = row["pair_id"]
        cards.append(
            f"""
<article class="pair" data-pair-id="{html.escape(pair_id)}">
  <div class="pair-head"><span>{html.escape(pair_id)}</span><p>{html.escape(row['prompt'])}</p></div>
  <div class="images">
    <figure><img src="images/{pair_id}-left.png" alt="Left image"><figcaption>Left</figcaption></figure>
    <figure><img src="images/{pair_id}-right.png" alt="Right image"><figcaption>Right</figcaption></figure>
  </div>
  <fieldset><legend>How large is the meaningful difference?</legend>
    {_radio_group(pair_id, 'severity', SEVERITY_OPTIONS)}
  </fieldset>
  <fieldset><legend>Which image better matches the prompt and is more usable?</legend>
    {_radio_group(pair_id, 'preference', PREFERENCE_OPTIONS)}
  </fieldset>
  <label class="notes">Optional note <input type="text" data-notes maxlength="300"></label>
</article>"""
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DiffCache blind image review</title>
<style>
:root {{ color-scheme: light; font-family: Inter, Arial, sans-serif; color: #17202a; background: #fff; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #fff; }}
header {{ position: sticky; top: 0; z-index: 2; padding: 18px 5vw; background: #fff; border-bottom: 1px solid #ccd6df; }}
header h1 {{ margin: 0 0 6px; font-size: 24px; }}
header p {{ margin: 4px 0; max-width: 1000px; line-height: 1.45; }}
main {{ width: min(1220px, 94vw); margin: 24px auto 100px; }}
.reviewer {{ display: flex; gap: 10px; align-items: center; margin: 16px 0 24px; }}
.reviewer input {{ width: 260px; padding: 9px; }}
.pair {{ border: 1px solid #ccd6df; border-radius: 10px; padding: 18px; margin: 0 0 26px; break-inside: avoid; }}
.pair-head {{ display: grid; grid-template-columns: 90px 1fr; gap: 12px; align-items: start; }}
.pair-head span {{ font: 700 14px ui-monospace, monospace; color: #1d5b79; }}
.pair-head p {{ margin: 0 0 14px; font-size: 17px; line-height: 1.45; }}
.images {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
figure {{ margin: 0; }}
img {{ display: block; width: 100%; height: auto; border: 1px solid #dce3e8; }}
figcaption {{ text-align: center; font-weight: 700; padding: 7px; }}
fieldset {{ border: 0; padding: 10px 0 0; margin: 0; }}
legend {{ font-weight: 700; padding: 0 0 4px; }}
.choice {{ display: inline-flex; gap: 5px; align-items: center; margin: 5px 18px 5px 0; }}
.notes {{ display: block; margin-top: 10px; }}
.notes input {{ width: min(700px, 80vw); padding: 7px; margin-left: 8px; }}
button {{ position: fixed; right: 24px; bottom: 22px; border: 0; border-radius: 7px; padding: 13px 20px; color: #fff; background: #145a72; font-weight: 700; cursor: pointer; }}
@media (max-width: 760px) {{ .images {{ grid-template-columns: 1fr; }} .pair-head {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>DiffCache blind image review</h1>
  <p>The two images use the same prompt and seed. Their identities are hidden and their positions are randomized.</p>
  <p>Judge visible usefulness and prompt match. Do not inspect filenames, the study key, or automatic scores before exporting your answers.</p>
</header>
<main>
  <label class="reviewer">Reviewer ID <input id="reviewer" autocomplete="off" placeholder="required before export"></label>
  {''.join(cards)}
</main>
<button id="export">Export answers</button>
<script>
const packetSha256 = {json.dumps(packet_sha256)};
document.getElementById('export').addEventListener('click', () => {{
  const reviewer = document.getElementById('reviewer').value.trim();
  if (!reviewer) {{ alert('Enter a reviewer ID first.'); return; }}
  const answers = [...document.querySelectorAll('.pair')].map(card => {{
    const pairId = card.dataset.pairId;
    const checked = suffix => card.querySelector(`input[name="${{pairId}}-${{suffix}}"]:checked`)?.value ?? null;
    return {{ pair_id: pairId, severity: checked('severity'), preference: checked('preference'), notes: card.querySelector('[data-notes]').value.trim() }};
  }});
  if (answers.some(row => row.severity === null || row.preference === null)) {{ alert('Every pair needs both answers.'); return; }}
  const report = {{ schema: 'difflet-cache-blind-review-response', schema_revision: 1, packet_sha256: packetSha256, reviewer_id: reviewer, answers }};
  const blob = new Blob([JSON.stringify(report, null, 2) + '\\n'], {{type: 'application/json'}});
  const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = `diffcache-review-${{reviewer}}.json`; link.click(); URL.revokeObjectURL(link.href);
}});
</script>
</body>
</html>
"""


def prepare_review(
    quality_input_path: Path,
    output_dir: Path,
    *,
    seed: int,
    candidate_ids: Sequence[str] | None = None,
) -> tuple[Path, Path]:
    """Create the reviewer page and the separately held identity key."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    quality_input = _load_json(quality_input_path, "quality input")
    _, definitions, comparisons = _validate_quality_input(quality_input)
    selected_ids = tuple(candidate_ids or definitions)
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("candidate selection must be non-empty and unique")
    unknown = set(selected_ids) - set(definitions)
    if unknown:
        raise ValueError(f"unknown candidate identifiers: {sorted(unknown)}")
    selected = [row for row in comparisons if row["candidate_id"] in selected_ids]
    assignments = build_review_assignments(selected, seed=seed)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    reviewer_rows = []
    key_rows = []
    for row in assignments:
        sources = {
            "baseline": (quality_input_path.parent / row["baseline"]["image"]).resolve(),
            "candidate": (quality_input_path.parent / row["candidate"]["image"]).resolve(),
        }
        if any(not path.is_file() for path in sources.values()):
            raise ValueError(f"review pair {row['pair_id']} references a missing image")
        side_role = (
            {"left": "candidate", "right": "baseline"}
            if row["candidate_side"] == "left"
            else {"left": "baseline", "right": "candidate"}
        )
        side_hashes = {}
        for side, role in side_role.items():
            destination = images_dir / f"{row['pair_id']}-{side}.png"
            _link_or_copy(sources[role], destination)
            side_hashes[side] = _sha256_file(destination)
        reviewer_rows.append(
            {"pair_id": row["pair_id"], "prompt": row["prompt"], "image_sha256": side_hashes}
        )
        key_rows.append(
            {
                "pair_id": row["pair_id"],
                "candidate_id": row["candidate_id"],
                "sample_id": row["sample_id"],
                "prompt_index": row["prompt_index"],
                "seed": row["seed"],
                "candidate_side": row["candidate_side"],
                "image_sha256": side_hashes,
            }
        )

    packet_payload = {
        "schema": REVIEW_SCHEMA,
        "schema_revision": REVIEW_SCHEMA_REVISION,
        "source_manifest_sha256": _sha256_file(quality_input_path),
        "randomization_seed": seed,
        "candidate_count": len(selected_ids),
        "pair_count": len(reviewer_rows),
        "severity_options": [value for value, _ in SEVERITY_OPTIONS],
        "preference_options": [value for value, _ in PREFERENCE_OPTIONS],
        "pairs": reviewer_rows,
    }
    packet = {**packet_payload, "sha256": canonical_sha256(packet_payload)}
    key_payload = {
        "schema": "difflet-cache-blind-review-key",
        "schema_revision": 1,
        "packet_sha256": packet["sha256"],
        "pairs": key_rows,
    }
    key = {**key_payload, "sha256": canonical_sha256(key_payload)}
    packet_path = output_dir / "review-packet.json"
    key_path = output_dir / "study-owner-key.json"
    page_path = output_dir / "review.html"
    packet_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    key_path.write_text(json.dumps(key, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    page_path.write_text(_review_html(reviewer_rows, packet["sha256"]), encoding="utf-8")
    return page_path, key_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--candidate-id", action="append", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        page, key = prepare_review(
            Path(args.quality_input).expanduser().resolve(),
            Path(args.out_dir).expanduser().resolve(),
            seed=args.seed,
            candidate_ids=args.candidate_id,
        )
    except (FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", flush=True)
        return 2
    print(f"[blind-review] reviewer page: {page}", flush=True)
    print(f"[blind-review] keep this key from reviewers: {key}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
