"""The gateway's page at ``/``: every model, its shapes, its status, an example request.

Server-rendered on each load from the fleet file, no JavaScript (design §6.5).
"""

from __future__ import annotations

import json
from html import escape
from typing import Mapping

from difflet.serving.gateway.fleet import FleetConfig, FleetEntry

QUEUE_LINE = ("Each model runs one request at a time; others wait in its queue (HTTP 429 when the "
              "queue is full or times out).")
_STATUS_HELP = {
    "ready": "serving requests",
    "loading": "the backend process is up and still loading its model",
    "recovering": "the backend is restarting its worker after a failure",
    "down": "no backend process, or it serves another model",
}
_AUTH = '-H "Authorization: Bearer $DIFFLET_API_KEY"'
_CSS = """
:root { color-scheme: light dark; --bg: #ffffff; --fg: #1f2328; --muted: #59636e;
  --line: #d1d9e0; --code: #f6f8fa; --ready: #1a7f37; --loading: #9a6700;
  --recovering: #bc4c00; --down: #cf222e; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --line: #3d444d; --code: #151b23;
    --ready: #3fb950; --loading: #d29922; --recovering: #db6d28; --down: #f85149; } }
* { box-sizing: border-box; }
body { margin: 0 auto; max-width: 72rem; padding: 1.5rem 1rem 3rem; background: var(--bg);
  color: var(--fg); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
h2 { font-size: 1.15rem; margin: 2rem 0 .5rem; }
h3 { font-size: .95rem; margin: 1.25rem 0 .4rem; }
p { color: var(--muted); margin: .25rem 0 1rem; }
a { color: inherit; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: .55rem .75rem; border-bottom: 1px solid var(--line);
  white-space: nowrap; vertical-align: top; }
th { color: var(--muted); font-size: .8rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: .03em; }
code, pre { font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre { margin: 0; padding: .75rem 1rem; background: var(--code); border: 1px solid var(--line);
  border-radius: 6px; overflow-x: auto; }
.badge { display: inline-block; padding: 0 .55rem; border: 1px solid currentColor;
  border-radius: 1rem; font-size: .8rem; font-weight: 600; }
.ready { color: var(--ready); } .loading { color: var(--loading); }
.recovering { color: var(--recovering); } .down { color: var(--down); }
"""


def render_page(config: FleetConfig, statuses: Mapping[str, str], *, base_url: str) -> str:
    rows, examples = [], []
    for entry in config.models:
        status = statuses.get(entry.model_id, "down")
        shapes = ", ".join("×".join(str(n) for n in shape if n is not None)
                           for shape in entry.shapes)
        rows.append(
            f"<tr><td><code>{escape(entry.model_id)}</code></td><td>{escape(entry.kind)}</td>"
            f"<td>{entry.device}</td><td>{escape(shapes)}</td>"
            f'<td><span class="badge {escape(status)}" title="{escape(_STATUS_HELP.get(status, ""))}">'
            f"{escape(status)}</span></td></tr>")
        examples.append(f"<h3><code>{escape(entry.model_id)}</code></h3>\n"
                        f"<pre><code>{escape(_example(entry, base_url))}</code></pre>")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>Difflet models</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Difflet models</h1>
<p>{escape(QUEUE_LINE)} A request must use one of its model's shapes: H×W for images,
H×W×F (frames) for videos. This page refreshes every 30 s.</p>
<div class="scroll"><table>
<thead><tr><th>Model</th><th>Kind</th><th>Device</th><th>Shapes</th><th>Status</th></tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody>
</table></div>
<h2>Example requests</h2>
<p>Each uses the model's first shape. Send the gateway's API key as a Bearer token when it has
one; the header is ignored when it has none. JSON: <a href="/v1/models">/v1/models</a>,
<a href="/ready">/ready</a>, <a href="/health">/health</a>.</p>
{chr(10).join(examples)}
</body>
</html>
"""


def _example(entry: FleetEntry, base_url: str) -> str:
    height, width, frames = entry.shapes[0]
    base = base_url.rstrip("/")
    if entry.kind == "image":
        body = json.dumps(entry.model_id)
        return (
            f"curl -sS -X POST {base}/v1/chat/completions \\\n"
            f"  {_AUTH} \\\n"
            '  -H "Content-Type: application/json" \\\n'
            f"  -d '{{\"model\": {body},\n"
            '       "messages": [{"role": "user", "content": "a small red sailboat on a calm blue lake"}],\n'
            f'       "extra_body": {{"height": {height}, "width": {width}, "seed": 42}}}}\' \\\n'
            "  | jq -r '.choices[0].message.content[0].image_url.url' | cut -d, -f2- "
            "| base64 -d > image.png")
    return (
        f"video_id=$(curl -sS -X POST {base}/v1/videos \\\n"
        f"  {_AUTH} \\\n"
        f"  -F 'model={entry.model_id}' \\\n"
        "  -F 'prompt=a cinematic mountain landscape at sunrise' \\\n"
        f"  -F 'size={width}x{height}' -F 'num_frames={frames}' | jq -r .id)\n"
        "# repeat until the status is \"completed\"\n"
        f'curl -sS {_AUTH} "{base}/v1/videos/$video_id" | jq .status\n'
        f'curl -fL {_AUTH} "{base}/v1/videos/$video_id/content" -o video.mp4')
