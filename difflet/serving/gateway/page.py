"""The gateway's page at ``/``: generate an image or a video with any ready model, plus every
model's shapes, status and an example API request.

Server-rendered from the fleet file; a small inline script calls the gateway's ``/v1`` routes
(the page sends no API key: it works when the gateway runs without one) and polls ``/ready`` and
``/stats`` instead of reloading the page, so a running generation is never lost. A static copy
(``difflet gateway page``), hosted elsewhere, calls the gateway at its API base and loads the
models from ``/ui/models``. Every generation request carries the browser's ``X-Difflet-Client``
id for the gateway's per-client limits.
"""

from __future__ import annotations

import functools
import json
from dataclasses import asdict
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
EXAMPLE_PROMPTS = {
    "image": (
        "a small red sailboat on a calm blue lake at sunrise",
        "a cat sitting on a wooden bench in a sunlit park, shallow depth of field",
        "a cozy coffee shop interior on a rainy evening, warm light, film photo",
        "an astronaut riding a horse on the moon, highly detailed illustration",
    ),
    "video": (
        "a red fox walking through fresh snow in a pine forest",
        "a golden retriever running on a beach at sunset",
        "a cat walking across a sunlit garden",
        "waves crashing against a rocky coast, slow motion, cinematic",
    ),
}
_CSS = """
:root { color-scheme: light dark; --bg: #ffffff; --panel: #f6f8fa; --fg: #1f2328;
  --muted: #59636e; --line: #d1d9e0; --accent: #0969da; --accent-fg: #ffffff;
  --ready: #1a7f37; --loading: #9a6700; --recovering: #bc4c00; --down: #cf222e; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #0d1117; --panel: #151b23; --fg: #e6edf3; --muted: #9198a1; --line: #3d444d;
    --accent: #4493f8; --accent-fg: #0d1117;
    --ready: #3fb950; --loading: #d29922; --recovering: #db6d28; --down: #f85149; } }
* { box-sizing: border-box; }
body { margin: 0 auto; max-width: 80rem; padding: 1.25rem 1rem 3rem; background: var(--bg);
  color: var(--fg); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between;
  gap: .5rem 1.5rem; margin-bottom: 1rem; }
h1 { font-size: 1.6rem; margin: 0; }
h2 { font-size: 1.1rem; margin: 0 0 .5rem; }
h3 { font-size: .95rem; margin: 1.25rem 0 .4rem; }
p { color: var(--muted); margin: .25rem 0 1rem; }
a { color: var(--accent); }
.counter { color: var(--muted); } .counter b { color: var(--fg); font-size: 1.15rem; }
.layout { display: grid; grid-template-columns: minmax(18rem, 26rem) 1fr; gap: 1.5rem;
  align-items: start; }
@media (max-width: 860px) { .layout { grid-template-columns: 1fr; } }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 1rem; }
.tabs { display: flex; gap: .25rem; margin-bottom: 1rem; }
.tabs button { flex: 1; padding: .5rem; border: 1px solid var(--line); border-radius: 8px;
  background: var(--bg); color: var(--fg); font: inherit; cursor: pointer; }
.tabs button[aria-selected="true"] { background: var(--accent); color: var(--accent-fg);
  border-color: var(--accent); font-weight: 600; }
label { display: block; font-size: .85rem; font-weight: 600; margin: .75rem 0 .3rem; }
select, textarea, input { width: 100%; padding: .5rem .6rem; border: 1px solid var(--line);
  border-radius: 8px; background: var(--bg); color: var(--fg); font: inherit; }
textarea { min-height: 6rem; resize: vertical; }
.hint { color: var(--muted); font-size: .8rem; margin-top: .3rem; }
.examples { display: flex; flex-wrap: wrap; gap: .35rem; }
.examples button { border: 1px solid var(--line); border-radius: 1rem; background: var(--bg);
  color: var(--muted); padding: .2rem .65rem; font: inherit; font-size: .8rem; cursor: pointer;
  text-align: left; }
.examples button:hover { color: var(--fg); border-color: var(--muted); }
.warn { color: var(--recovering); } .warn:empty { display: none; }
.warn button { border: 1px solid currentColor; border-radius: 1rem; background: var(--bg);
  color: var(--accent); padding: 0 .55rem; font: inherit; cursor: pointer; }
details.advanced { margin-top: .75rem; } details summary { cursor: pointer; color: var(--muted); }
.row3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: .5rem; }
#run { width: 100%; margin-top: 1rem; padding: .65rem; border: 0; border-radius: 8px;
  background: var(--accent); color: var(--accent-fg); font: inherit; font-weight: 600;
  cursor: pointer; }
#run:disabled { opacity: .55; cursor: progress; }
#state { min-height: 1.5rem; margin-top: .6rem; font-size: .9rem; color: var(--muted); }
#progress { width: 100%; height: .5rem; margin-top: .75rem; accent-color: var(--accent); }
.estimate { color: var(--fg); font-weight: 600; }
#state.error { color: var(--down); }
.results { display: grid; gap: 1rem; }
.empty { color: var(--muted); border: 1px dashed var(--line); border-radius: 10px; padding: 3rem 1rem;
  text-align: center; }
.card { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; background: var(--panel); }
.card img, .card video { display: block; width: 100%; height: auto; background: #000; }
.card .meta { padding: .6rem .8rem; font-size: .85rem; color: var(--muted); }
.card .meta b { color: var(--fg); }
.notice { font-size: .8rem; color: var(--muted); margin-top: 1rem; }
section.api { margin-top: 2.5rem; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: .55rem .75rem; border-bottom: 1px solid var(--line);
  white-space: nowrap; vertical-align: top; }
th { color: var(--muted); font-size: .8rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: .03em; }
code, pre { font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre { margin: 0; padding: .75rem 1rem; background: var(--panel); border: 1px solid var(--line);
  border-radius: 6px; overflow-x: auto; }
.badge { display: inline-block; padding: 0 .55rem; border: 1px solid currentColor;
  border-radius: 1rem; font-size: .8rem; font-weight: 600; }
.ready { color: var(--ready); } .loading { color: var(--loading); }
.recovering { color: var(--recovering); } .down { color: var(--down); }
"""
_SCRIPT = r"""
const $ = (id) => document.getElementById(id);
const API = window.DIFFLET_API || '';  // the gateway; '' when it served this page itself
const UI = JSON.parse($('ui').textContent);
const EXAMPLES = JSON.parse($('example-prompts').textContent);
// A static copy of the page embeds no models: it loads them, with the wait thresholds and the
// limits, from GET /ui/models.
const STATIC = UI.models.length === 0;
let MODELS = UI.models;
let WAIT = UI.wait;
let LIMITS = UI.limits;
let kind = 'image';
let running = false;
let QUEUE = {};  // model id -> requests running or waiting (GET /queue)
// This browser's id for the gateway's per-client limits; in memory when storage is blocked.
const CLIENT = (() => {
  const fresh = () => Array.from(crypto.getRandomValues(new Uint8Array(16)),
    (b) => b.toString(16).padStart(2, '0')).join('');
  try {
    let id = localStorage.getItem('difflet-client');
    if (!/^[A-Za-z0-9_-]{8,64}$/.test(id || '')) {
      id = fresh();
      localStorage.setItem('difflet-client', id);
    }
    return id;
  } catch (e) { return fresh(); }
})();

function shapeText(s) { return s[2] ? s[0] + '×' + s[1] + '×' + s[2] : s[0] + '×' + s[1]; }
function duration(seconds) {
  if (seconds < 90) return Math.round(seconds) + ' s';
  const [value, unit] = seconds < 5400 ? [seconds / 60, ' min'] : [seconds / 3600, ' h'];
  return value.toFixed(1).replace(/\.0$/, '') + unit;  // '2 h', '1.5 h'
}
// Seconds a request should take (measured: base + per step), null without a measurement.
function estimate(m, steps) {
  const n = steps === null ? m.steps : steps;
  return m && m.estimate && n ? m.estimate.base + m.estimate.per_step * n : null;
}
function current() { return MODELS.find((m) => m.id === $('model').value); }
function setState(text, error) { $('state').textContent = text; $('state').className = error ? 'error' : ''; }

function renderModels() {
  const select = $('model');
  const keep = select.value;
  select.innerHTML = '';
  for (const m of MODELS.filter((m) => m.kind === kind)) {
    const option = document.createElement('option');
    option.value = m.id;
    const waiting = QUEUE[m.id] || 0;
    option.textContent = m.label + (m.status === 'ready' ? '' : '  (' + m.status + ')')
      + (waiting ? '  · ' + waiting + ' in queue' : '');
    option.disabled = m.status !== 'ready';
    select.appendChild(option);
  }
  const ready = [...select.options].filter((o) => !o.disabled);
  if (ready.some((o) => o.value === keep)) select.value = keep;
  else if (ready.length) select.value = ready[0].value;
  renderModelInfo();
}

function renderModelInfo() {
  const m = current();
  const info = $('model-info');
  info.textContent = m ? 'Shape ' + shapeText(m.shapes[0]) + ' · device ' + m.device
    + (m.steps ? ' · default ' + m.steps + ' steps' : '')
    : (STATIC && !MODELS.length ? 'Loading models…' : 'No ' + kind + ' model is ready yet.');
  const steps = readNumber('steps');
  const seconds = m ? estimate(m, steps) : null;
  if (seconds !== null) {
    const est = document.createElement('span');
    est.className = 'estimate';
    est.textContent = 'Estimated ~' + duration(seconds) + ' at ' + (steps === null ? m.steps : steps) + ' steps';
    info.append(document.createElement('br'), est);
  }
  if (m) {
    // The requests ahead are assumed to run at their model's default steps.
    const waiting = QUEUE[m.id] || 0;
    const each = estimate(m, null);
    const queue = document.createElement('span');
    queue.className = 'estimate';
    queue.textContent = waiting === 0 ? 'Queue: empty, starts right away'
      : 'Queue: ' + waiting + ' running or waiting'
        + (each !== null && seconds !== null ? ' · total wait ~' + duration(waiting * each + seconds) : '');
    info.append(document.createElement('br'), queue);
  }
  const tooBusy = renderWait(m);
  $('guidance-box').hidden = !m || m.guidance === null;
  $('steps').placeholder = m && m.steps ? String(m.steps) : 'default';
  $('guidance').placeholder = m && m.guidance !== null ? String(m.guidance) : '';
  $('run').disabled = running || !m || m.status !== 'ready' || tooBusy;
}

// Seconds before a new request on m starts, as the gateway estimates it: the requests ahead at
// the model's default steps. null without an estimate.
function waitFor(m) {
  const each = estimate(m, null);
  return each === null ? null : (QUEUE[m.id] || 0) * each;
}

// The long-wait warning under the queue line; true when the gateway would refuse the request.
function renderWait(m) {
  const box = $('wait');
  box.textContent = '';
  const wait = m && WAIT ? waitFor(m) : null;
  if (wait === null || wait < WAIT.notice_s) return false;
  const tooBusy = wait >= WAIT.reject_s;
  box.append(tooBusy ? 'This model is too busy right now (over ' + duration(WAIT.reject_s)
    + '); please choose another model.' : 'Long wait: about ' + duration(wait) + ' before it starts.');
  if (wait >= WAIT.suggest_s) {
    const better = MODELS.filter((o) => o.id !== m.id && o.kind === m.kind && o.status === 'ready'
      && waitFor(o) !== null && waitFor(o) < wait).sort((a, b) => waitFor(a) - waitFor(b)).slice(0, 2);
    if (better.length) box.append(' Consider another model: ');
    for (const o of better) {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = o.label + (waitFor(o) ? ' (~' + duration(waitFor(o)) + ' wait)' : ' (no wait)');
      b.onclick = () => { $('model').value = o.id; renderModelInfo(); };
      box.append(b, ' ');
    }
  }
  return tooBusy;
}

function renderLimits() {
  $('limits').textContent = LIMITS ? 'Each browser can have ' + LIMITS.client_active
    + (LIMITS.client_active === 1 ? ' request' : ' requests') + ' running or waiting, and run up to '
    + LIMITS.client_per_hour + ' per hour.' : '';
}

function renderExamples() {
  const box = $('examples');
  box.innerHTML = '';
  for (const text of EXAMPLES[kind]) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = text;
    b.onclick = () => { $('prompt').value = text; };
    box.appendChild(b);
  }
}

function selectTab(next) {
  kind = next;
  for (const b of document.querySelectorAll('.tabs button')) {
    b.setAttribute('aria-selected', String(b.dataset.kind === kind));
  }
  renderModels();
  renderExamples();
}

async function api(url, options) {
  const response = await fetch(url, options);
  const text = await response.text();
  let body = null;
  try { body = JSON.parse(text); } catch (e) { body = null; }
  if (!response.ok) {
    const message = body && body.error && body.error.message;
    throw new Error(message || ('HTTP ' + response.status));
  }
  return body;
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function runImage(m, prompt, seed, steps, guidance) {
  const extra = { height: m.shapes[0][0], width: m.shapes[0][1], seed: seed };
  if (steps !== null) extra.num_inference_steps = steps;
  if (guidance !== null) extra.guidance_scale = guidance;
  const body = await api(API + '/v1/chat/completions', {
    method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Difflet-Client': CLIENT },
    body: JSON.stringify({ model: m.id, messages: [{ role: 'user', content: prompt }], extra_body: extra }),
  });
  return { type: 'image', url: body.choices[0].message.content[0].image_url.url };
}

async function runVideo(m, prompt, seed, steps, guidance, onStatus) {
  const [height, width, frames] = m.shapes[0];
  const form = new FormData();
  form.append('model', m.id);
  form.append('prompt', prompt);
  form.append('size', width + 'x' + height);
  form.append('num_frames', String(frames));
  form.append('seed', String(seed));
  if (steps !== null) form.append('num_inference_steps', String(steps));
  if (guidance !== null) form.append('guidance_scale', String(guidance));
  const job = await api(API + '/v1/videos', {
    method: 'POST', headers: { 'X-Difflet-Client': CLIENT }, body: form });
  const path = API + '/v1/videos/' + encodeURIComponent(job.id);
  for (;;) {
    await sleep(3000);
    const state = await api(path);
    if (state.status === 'completed') break;
    if (state.status === 'failed' || state.status === 'cancelled') {
      throw new Error((state.error && state.error.message) || ('video ' + state.status));
    }
    onStatus(state.status);
  }
  return { type: 'video', url: path + '/content' };
}

function addResult(m, prompt, seed, steps, seconds, media) {
  const card = document.createElement('div');
  card.className = 'card';
  const el = document.createElement(media.type === 'image' ? 'img' : 'video');
  el.src = media.url;
  if (media.type === 'video') { el.controls = true; el.autoplay = true; el.loop = true; el.muted = true; el.playsInline = true; }
  else el.alt = prompt;
  const meta = document.createElement('div');
  meta.className = 'meta';
  const head = document.createElement('b');
  head.textContent = m.label;
  const link = document.createElement('a');
  link.href = media.url;
  link.download = media.type === 'image' ? 'difflet.png' : 'difflet.mp4';
  link.textContent = 'download';
  meta.append(head, ' · ' + seconds.toFixed(1) + ' s · seed ' + seed
    + ' · ' + (steps === null ? 'default' : steps) + ' steps · ', link,
    document.createElement('br'), prompt);
  card.append(el, meta);
  const results = $('results');
  const empty = results.querySelector('.empty');
  if (empty) empty.remove();
  results.prepend(card);
}

function readNumber(id) {
  const raw = $(id).value.trim();
  return raw === '' ? null : Number(raw);
}

async function run() {
  const m = current();
  const prompt = $('prompt').value.trim();
  if (!m) return;
  if (!prompt) { setState('Enter a prompt first.', true); return; }
  const seedValue = readNumber('seed');
  const seed = seedValue === null ? Math.floor(Math.random() * 2147483647) : seedValue;
  const steps = readNumber('steps');
  const guidance = m.guidance === null ? null : readNumber('guidance');
  running = true;
  renderModelInfo();
  const started = performance.now();
  const expected = estimate(m, steps);
  const bar = $('progress');
  bar.hidden = expected === null;
  bar.value = 0;
  let phase = 'running';
  const tick = () => {
    const elapsed = (performance.now() - started) / 1000;
    if (expected !== null) bar.value = Math.min(elapsed / expected, 0.99);
    setState(m.label + ': ' + phase + ' · ' + duration(elapsed)
      + (expected === null ? '' : ' of ~' + duration(expected)));
  };
  tick();
  const timer = setInterval(tick, 500);
  try {
    const media = m.kind === 'image'
      ? await runImage(m, prompt, seed, steps, guidance)
      : await runVideo(m, prompt, seed, steps, guidance, (s) => { phase = s.replace('_', ' '); });
    const seconds = (performance.now() - started) / 1000;
    addResult(m, prompt, seed, steps, seconds, media);
    setState('Done in ' + seconds.toFixed(1) + ' s.');
    refresh();
  } catch (e) {
    setState(e.message, true);
  } finally {
    clearInterval(timer);
    bar.hidden = true;
    running = false;
    renderModelInfo();
  }
}

async function refreshQueue() {
  try {
    QUEUE = (await fetch(API + '/queue').then((r) => r.json())).models || {};
    renderModels();
  } catch (e) { /* keep the last known queue */ }
}

async function refresh() {
  try {
    if (STATIC) {
      const ui = await fetch(API + '/ui/models').then((r) => r.json());
      MODELS = ui.models;
      WAIT = ui.wait;
      LIMITS = ui.limits;
      renderLimits();
    } else {
      const ready = await fetch(API + '/ready').then((r) => r.json());
      for (const m of MODELS) m.status = (ready.models || {})[m.id] || 'down';
    }
    renderModels();
    const stats = await fetch(API + '/stats').then((r) => r.json());
    $('runs').textContent = String(stats.total_runs);
  } catch (e) { /* keep the last known state */ }
}

for (const b of document.querySelectorAll('.tabs button')) b.onclick = () => selectTab(b.dataset.kind);
$('model').onchange = renderModelInfo;
$('steps').addEventListener('input', renderModelInfo);
$('run').onclick = run;
$('prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) run(); });
selectTab(MODELS.length && !MODELS.some((m) => m.kind === 'image') ? 'video' : 'image');
renderLimits();
if (STATIC) refresh();
setInterval(refresh, 15000);
refreshQueue();
setInterval(refreshQueue, 5000);
"""


def render_page(config: FleetConfig | None, statuses: Mapping[str, str], *, base_url: str = "",
                total_runs: int = 0, api_base: str = "") -> str:
    """The page. Its script calls the gateway at ``api_base`` ('': the page's own origin). Without a
    ``config`` it is the static page (``difflet gateway page``): no models embedded, so the
    script loads them from ``{api_base}/ui/models``, and no model table or API examples."""
    ui = ({"models": [], "wait": None, "limits": None} if config is None
          else ui_models(config, statuses))
    api = "" if config is None else _api_section(config, statuses, base_url)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Difflet models</title>
<style>{_CSS}</style>
</head>
<body>
<header>
<div><h1>Difflet</h1><p>Text-to-image and text-to-video on AWS Trainium.</p></div>
<div class="counter">Total inference runs <b id="runs">{"–" if config is None else int(total_runs)}</b></div>
</header>
<div class="layout">
<div class="panel">
<div class="tabs" role="tablist">
<button type="button" role="tab" data-kind="image" aria-selected="true">Text-to-Image</button>
<button type="button" role="tab" data-kind="video" aria-selected="false">Text-to-Video</button>
</div>
<label for="model">Model</label>
<select id="model"></select>
<div class="hint" id="model-info"></div>
<div class="hint warn" id="wait" aria-live="polite"></div>
<label for="prompt">Prompt</label>
<textarea id="prompt" placeholder="Describe what to generate"></textarea>
<label>Examples</label>
<div class="examples" id="examples"></div>
<details class="advanced"><summary>Advanced</summary>
<div class="row3">
<div><label for="steps">Steps</label><input id="steps" type="number" min="1" max="50"></div>
<div id="guidance-box"><label for="guidance">Guidance</label>
<input id="guidance" type="number" min="0" max="20" step="0.5"></div>
<div><label for="seed">Seed</label><input id="seed" type="number" min="0" placeholder="random"></div>
</div>
<div class="hint">Empty fields use the model's defaults; an empty seed is random.</div>
</details>
<button id="run" type="button">Run</button>
<progress id="progress" max="1" value="0" hidden></progress>
<div id="state" aria-live="polite"></div>
<p class="notice">{escape(QUEUE_LINE)} Estimated times were measured on this server with one
model busy at a time; the wait counts the requests ahead at their model's default steps. A request
must use one of its model's shapes: H×W for images, H×W×F (frames) for videos.
<span id="limits"></span></p>
</div>
<div class="results" id="results"><div class="empty">Results appear here.</div></div>
</div>
{api}
<script type="application/json" id="ui">{_json(ui)}</script>
<script type="application/json" id="example-prompts">{_json(EXAMPLE_PROMPTS)}</script>
<script>window.DIFFLET_API = {_json(api_base.rstrip("/"))};</script>
<script>{_SCRIPT}</script>
</body>
</html>
"""


def _api_section(config: FleetConfig, statuses: Mapping[str, str], base_url: str) -> str:
    """The server-rendered model table and example requests, folded under the form."""
    rows, examples = [], []
    for entry in config.models:
        status = statuses.get(entry.model_id, "down")
        shapes = ", ".join("×".join(str(n) for n in shape if n is not None)
                           for shape in entry.shapes)
        rows.append(
            f"<tr><td><code>{escape(entry.model_id)}</code></td><td>{escape(entry.kind)}</td>"
            f"<td>{entry.device}</td><td>{escape(shapes)}</td>"
            f"<td>{escape(estimate_text(entry, model_defaults(entry.model_id)[0]))}</td>"
            f'<td><span class="badge {escape(status)}" title="{escape(_STATUS_HELP.get(status, ""))}">'
            f"{escape(status)}</span></td></tr>")
        examples.append(f"<h3><code>{escape(entry.model_id)}</code></h3>\n"
                        f"<pre><code>{escape(_example(entry, base_url))}</code></pre>")
    return f"""<section class="api">
<details><summary>Models, status and API examples</summary>
<div class="scroll"><table>
<thead><tr><th>Model</th><th>Kind</th><th>Device</th><th>Shapes</th><th>Est. time</th><th>Status</th></tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody>
</table></div>
<h2 style="margin-top:1.5rem">Example requests</h2>
<p>Each uses the model's first shape. Send the gateway's API key as a Bearer token when it has
one; the header is ignored when it has none. JSON: <a href="/v1/models">/v1/models</a>,
<a href="/ready">/ready</a>, <a href="/health">/health</a>.</p>
{chr(10).join(examples)}
</details>
</section>"""


def _json(value) -> str:
    """JSON safe inside a ``<script>`` element: no ``</`` can end it early."""
    return json.dumps(value).replace("</", "<\\/")


def ui_models(config: FleetConfig, statuses: Mapping[str, str]) -> dict:
    """What the page's script needs (``GET /ui/models``, embedded in the server-rendered page):
    every model with its live status, the wait thresholds and the limits."""
    return {"models": [_ui_model(e, statuses.get(e.model_id, "down")) for e in config.models],
            "wait": asdict(config.wait), "limits": asdict(config.limits)}


def _ui_model(entry: FleetEntry, status: str) -> dict:
    steps, guidance = model_defaults(entry.model_id)
    return {
        "id": entry.model_id,
        "label": entry.model_id.rsplit("/", 1)[-1],
        "kind": entry.kind,
        "device": entry.device,
        "shapes": [list(shape) for shape in entry.shapes],
        "status": status,
        "steps": steps,
        "guidance": guidance,
        "estimate": (None if entry.estimate is None
                     else {"base": entry.estimate[0], "per_step": entry.estimate[1]}),
    }


def estimate_text(entry: FleetEntry, steps: int | None) -> str:
    """"~9 s (28 steps)" from the entry's measured estimate, "" without one."""
    if entry.estimate is None or not steps:
        return ""
    return f"~{_duration(entry.estimate[0] + entry.estimate[1] * steps)} ({steps} steps)"


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{round(seconds)} s"
    value, unit = (seconds / 60, "min") if seconds < 5400 else (seconds / 3600, "h")
    return f"{value:.1f}".removesuffix(".0") + f" {unit}"  # "2 h", "1.5 h"


@functools.cache
def model_defaults(model_id: str) -> tuple[int | None, float | None]:
    """The steps and guidance scale a backend uses when a request omits them (the model's
    ``ModelSpec``; guidance None when the model takes none). A model without a spec (MiniMax-H3)
    takes its serving metadata's steps and no guidance scale."""
    from difflet.models.spec import gpu_model_spec
    from difflet.registry import resolve_model
    from difflet.serving.model_registry import serving_default_steps

    try:
        spec = gpu_model_spec(resolve_model(model_id).name)
    except (KeyError, ValueError):
        spec = None
    if spec is None:
        return serving_default_steps(model_id), None
    guidance = None if spec.guidance == "none" else spec.default_guidance_scale
    return spec.default_steps, guidance


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
