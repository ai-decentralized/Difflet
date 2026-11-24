"""The gateway's page at ``/compare``: one Trainium3 chip against one NVIDIA RTX PRO 6000, per-step
DiT latency and output quality, as paired-bar charts with the raw numbers below.

Static. The trn3 figures come from the November 2026 single-chip performance and quality reports;
the RTX PRO 6000 figures are the ones already in docs/verification (gpu-backend-results.md,
hunyuanvideo-cuda.md, hunyuanvideo15-cuda.md), not re-measured. Design:
docs/plans/2026-11-22-trn3-vs-rtx-pro-6000-page-design.md.
"""

from __future__ import annotations

from difflet.serving.gateway.page import _CSS, _json

# Per-step DiT latency in ms. An RTX value's ``mark``: "" measured DiT forward; "≈" two forwards
# (CFG) from one measured forward; "≤" a whole-request time divided by its steps, an upper bound.
SPEED = [
    {"model": "FLUX.1-dev", "config": "1024×1024", "trn3_tp4": 221.4, "trn3": 213.9,
     "layout": "TP4 + SP", "rtx": {"compiled": [241, ""], "eager": [303, ""]},
     "source": "DiT forward, gpu-backend-results.md",
     "warm": 50.7, "cold": 103.6, "compile": 150, "rtx_request": "9.4 / 7.1 (28 steps)"},
    {"model": "Qwen-Image", "config": "1024×1024", "trn3_tp4": 302.5, "trn3": 280.4,
     "layout": "TP4 + SP", "rtx": {"compiled": [243, ""], "eager": [323, ""]},
     "source": "DiT forward, gpu-backend-results.md",
     "warm": 78.0, "cold": 103.7, "compile": 107, "rtx_request": "16.5 / 12.4"},
    {"model": "Wan 2.2 A14B", "config": "480×832×9, both experts", "trn3_tp4": 872.2,
     "trn3": 754.7, "layout": "TP2 × CFG2 + SP",
     "rtx": {"compiled": [820, "≤"], "eager": [1072, "≈"]},
     "source": "eager 2 × 536 ms forward; compiled 32.8 s ÷ 40 steps, gpu-backend-results.md",
     "warm": 118.8, "cold": 154.7, "compile": 278, "rtx_request": "44.5 / 32.8 (40 steps)"},
    {"model": "Wan 2.1 14B", "config": "480×832×9, CFG 5.0", "trn3_tp4": 872.8, "trn3": 754.1,
     "layout": "TP2 × CFG2 + SP", "rtx": None, "source": "never run on a GPU",
     "warm": 115.1, "cold": 142.3, "compile": 235, "rtx_request": "—"},
    {"model": "HunyuanVideo", "config": "320×512×61", "trn3_tp4": 724, "trn3": 664,
     "layout": "TP4 + SP", "rtx": {"compiled": [1228.8, "≤"], "eager": [1312.5, "≤"]},
     "source": "61.4 / 65.6 s request ÷ 50 steps, hunyuanvideo-cuda.md",
     "warm": 103.4, "cold": 143.4, "compile": 182, "rtx_request": "65.6 / 61.4"},
    {"model": "HunyuanVideo 1.5", "config": "480×848×121", "trn3_tp4": 9914, "trn3": 9914,
     "layout": "TP2 × CFG2", "rtx": {"compiled": [8159, "≤"], "eager": [8246, "≤"]},
     "source": "408.0 / 412.3 s request ÷ 50 steps, hunyuanvideo15-cuda.md",
     "warm": 775.6, "cold": 805.4, "compile": 1239, "rtx_request": "412.3 / 408.0"},
    {"model": "LTX-2", "config": "480×704×49, CFG 3.5", "trn3_tp4": 557.9, "trn3": 495.9,
     "layout": "TP2 × CFG2", "rtx": None, "source": "not measured on an RTX PRO 6000",
     "warm": 96.7, "cold": 127.1, "compile": 103, "rtx_request": "—"},
]

# PSNR dB and SSIM against diffusers fp32 on the RTX PRO 6000; ``rtx`` is diffusers bf16 on the
# same card. ``summary``: the row that stands for its model in the chart's count.
QUALITY = [
    {"model": "FLUX.1-dev", "layout": "TP4", "trn3": [16.04, 0.773], "rtx": [15.96, 0.772],
     "summary": True},
    {"model": "Qwen-Image", "layout": "TP4", "trn3": [28.39, 0.976], "rtx": [34.51, 0.993],
     "summary": True},
    {"model": "Wan 2.2, both experts", "layout": "TP4", "trn3": [24.95, 0.920],
     "rtx": [24.05, 0.854], "summary": True},
    {"model": "Wan 2.2, high-noise expert", "layout": "TP4", "trn3": [23.88, 0.858],
     "rtx": [23.22, 0.848], "summary": False},
    {"model": "Wan 2.1", "layout": "TP4", "trn3": [24.34, 0.913], "rtx": [18.92, 0.823],
     "summary": True},
    {"model": "HunyuanVideo", "layout": "TP4", "trn3": [21.81, 0.913], "rtx": [19.51, 0.865],
     "summary": True},
    {"model": "HunyuanVideo 1.5", "layout": "TP2 × CFG2", "trn3": [30.25, 0.958],
     "rtx": [32.58, 0.970], "summary": True},
    {"model": "LTX-2", "layout": "TP4", "trn3": [26.84, 0.928], "rtx": [34.12, 0.964],
     "summary": True},
]

#: Differences this small read as level: speed ratio, PSNR dB, SSIM.
LEVEL = {"speed": 0.02, "psnr": 0.1, "ssim": 0.005}

_COMPARE_CSS = """
:root { --trn3: #2a78d6; --rtx: #eb6834; }
@media (prefers-color-scheme: dark) { :root { --trn3: #3987e5; --rtx: #d95926; } }
.nav { color: var(--muted); font-size: .9rem; }
.chart-card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 1rem 1.25rem; margin-top: 1.25rem; }
.card-head { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline;
  gap: .5rem 1rem; }
.summary { font-size: 1.05rem; font-weight: 600; margin: .25rem 0 0; color: var(--fg); }
.legend { display: flex; flex-wrap: wrap; gap: 1rem; font-size: .85rem; color: var(--muted);
  margin: .75rem 0 .25rem; }
.swatch { display: inline-block; width: .8rem; height: .8rem; border-radius: 3px;
  vertical-align: -1px; margin-right: .35rem; }
.toggle { display: inline-flex; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
.toggle button { border: 0; background: var(--bg); color: var(--fg); font: inherit;
  font-size: .85rem; padding: .35rem .8rem; cursor: pointer; }
.toggle button[aria-pressed="true"] { background: var(--accent); color: var(--accent-fg);
  font-weight: 600; }
.rows { margin-top: .5rem; }
.pair { display: grid; grid-template-columns: 12rem 1fr 11rem; gap: .25rem 1rem;
  align-items: center; padding: .6rem 0; border-top: 1px solid var(--line); }
@media (max-width: 760px) { .pair { grid-template-columns: 1fr; } }
.name b { display: block; } .name small { color: var(--muted); }
.bar-row { display: grid; grid-template-columns: 3.2rem 1fr 6.5rem; gap: .5rem;
  align-items: center; font-size: .85rem; }
.bar-row + .bar-row { margin-top: 2px; }
.who { color: var(--muted); }
.track { height: 14px; }
.bar { height: 14px; border-radius: 0 4px 4px 0; min-width: 2px; }
.bar.trn3 { background: var(--trn3); } .bar.rtx { background: var(--rtx); }
.bar.bound, .swatch.bound { background: repeating-linear-gradient(45deg, var(--rtx) 0 4px,
  color-mix(in srgb, var(--rtx) 45%, transparent) 4px 7px); }
.val { text-align: right; font-variant-numeric: tabular-nums; }
.none { color: var(--muted); font-size: .85rem; font-style: italic; }
.verdict { font-weight: 600; font-variant-numeric: tabular-nums; }
.verdict small { display: block; font-weight: 400; color: var(--muted); }
.axis { display: grid; grid-template-columns: 12rem 1fr 11rem; gap: 1rem; font-size: .75rem;
  color: var(--muted); padding-top: .35rem; }
.axis .ticks { display: flex; justify-content: space-between; margin: 0 7rem 0 3.7rem; }
@media (max-width: 760px) { .axis { display: none; } }
.note { font-size: .85rem; color: var(--muted); margin: .75rem 0 0; }
#tip { position: fixed; pointer-events: none; z-index: 10; max-width: 22rem; padding: .45rem .6rem;
  border: 1px solid var(--line); border-radius: 6px; background: var(--bg); color: var(--fg);
  font-size: .8rem; box-shadow: 0 4px 14px rgba(0,0,0,.18); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
"""

_COMPARE_SCRIPT = r"""
const D = JSON.parse(document.getElementById('compare-data').textContent);
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const ms = (v) => (v >= 1000 ? Math.round(v).toLocaleString('en-US') : String(+v.toFixed(1))) + ' ms';
let rtxMode = 'compiled';
let qMetric = 'psnr';

function bar(who, cls, value, width, tip) {
  return `<div class="bar-row" data-tip="${esc(tip)}"><span class="who">${who}</span>` +
    `<div class="track"><div class="bar ${cls}" style="width:${width.toFixed(1)}%"></div></div>` +
    `<span class="val">${value}</span></div>`;
}

function drawSpeed() {
  let wins = 0, have = 0;
  const rows = D.speed.map((m) => {
    const r = m.rtx && m.rtx[rtxMode];
    const name = `<div class="name"><b>${esc(m.model)}</b><small>${esc(m.config)}</small></div>`;
    const trnTip = `trn3 ${ms(m.trn3)} per step, ${m.layout} (TP4: ${ms(m.trn3_tp4)})`;
    if (!r) {
      return `<div class="pair">${name}<div>${bar('trn3', 'trn3', ms(m.trn3), 100, trnTip)}` +
        `<div class="bar-row"><span class="who">RTX</span><span class="none">no RTX PRO 6000 data</span><span></span></div></div>` +
        `<div class="verdict">—<small>${esc(m.source)}</small></div></div>`;
    }
    const [g, mark] = r;
    const top = Math.max(m.trn3, g);
    const x = g / m.trn3;
    have += 1;
    if (x >= 1 + D.level.speed) wins += 1;
    const lead = mark === '≤' ? (x >= 1 ? '≤ ' : '≥ ') : (mark === '≈' ? '≈ ' : '');
    const verdict = Math.abs(x - 1) < D.level.speed ? 'level'
      : x > 1 ? `trn3 ${lead}${x.toFixed(2)}× faster` : `RTX ${lead}${(1 / x).toFixed(2)}× faster`;
    const rtxTip = `RTX PRO 6000 ${rtxMode === 'compiled' ? 'torch.compile' : 'eager'}: ${mark}${ms(g)} per step — ${m.source}`;
    return `<div class="pair">${name}<div>` +
      bar('trn3', 'trn3', ms(m.trn3), m.trn3 / top * 100, trnTip) +
      bar('RTX', mark === '≤' ? 'rtx bound' : 'rtx', mark + ms(g), g / top * 100, rtxTip) +
      `</div><div class="verdict">${verdict}<small>${esc(m.layout)} vs ${rtxMode === 'compiled' ? 'torch.compile' : 'eager'}</small></div></div>`;
  });
  $('speed-rows').innerHTML = rows.join('');
  const missing = D.speed.length - have;
  $('speed-summary').textContent = `trn3 is faster per step on ${wins} of the ${have} models with RTX data` +
    (missing ? ` (${missing} more have none)` : '');
}

function drawQuality() {
  const i = qMetric === 'psnr' ? 0 : 1;
  const top = qMetric === 'psnr' ? 40 : 1;
  const fmt = (v) => (qMetric === 'psnr' ? v.toFixed(2) + ' dB' : v.toFixed(3));
  let good = 0, counted = 0;
  const rows = D.quality.map((q) => {
    const t = q.trn3[i], r = q.rtx[i], d = t - r;
    const level = Math.abs(d) < D.level[qMetric];
    if (q.summary) { counted += 1; if (level || d > 0) good += 1; }
    const delta = level ? 'level' : (d > 0 ? '+' : '−') + (qMetric === 'psnr' ? Math.abs(d).toFixed(1) + ' dB' : Math.abs(d).toFixed(3));
    const who = level ? 'level with RTX bf16' : d > 0 ? 'trn3 closer to fp32' : 'RTX bf16 closer to fp32';
    return `<div class="pair"><div class="name"><b>${esc(q.model)}</b><small>trn3 ${esc(q.layout)}</small></div><div>` +
      bar('trn3', 'trn3', fmt(t), t / top * 100, `trn3 bf16 vs fp32: ${fmt(t)}`) +
      bar('RTX', 'rtx', fmt(r), r / top * 100, `RTX PRO 6000 diffusers bf16 vs fp32: ${fmt(r)}`) +
      `</div><div class="verdict">${delta}<small>${who}</small></div></div>`;
  });
  $('quality-rows').innerHTML = rows.join('');
  $('quality-ticks').innerHTML = (qMetric === 'psnr' ? [0, 10, 20, 30, 40].map((v) => v + ' dB')
    : ['0', '0.25', '0.5', '0.75', '1']).map((v) => `<span>${v}</span>`).join('');
  $('quality-summary').textContent = `trn3 is level with or closer to fp32 than RTX bf16 on ${good} of ${counted} models (${qMetric.toUpperCase()})`;
}

function toggle(group, value, draw) {
  for (const b of document.querySelectorAll(`[data-group="${group}"]`)) {
    b.setAttribute('aria-pressed', String(b.dataset.value === value));
  }
  draw();
}
for (const b of document.querySelectorAll('[data-group="rtx"]')) {
  b.onclick = () => { rtxMode = b.dataset.value; toggle('rtx', rtxMode, drawSpeed); };
}
for (const b of document.querySelectorAll('[data-group="q"]')) {
  b.onclick = () => { qMetric = b.dataset.value; toggle('q', qMetric, drawQuality); };
}

const tip = $('tip');
document.addEventListener('mousemove', (e) => {
  const row = e.target.closest && e.target.closest('[data-tip]');
  if (!row) { tip.hidden = true; return; }
  tip.textContent = row.dataset.tip;
  tip.hidden = false;
  tip.style.left = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8) + 'px';
  tip.style.top = (e.clientY + 14) + 'px';
});

drawSpeed();
drawQuality();
"""


def render_compare() -> str:
    def cell(v, spec=","):
        return f'<td class="num">{v if isinstance(v, str) else format(v, spec)}</td>'

    def rtx(m, mode):
        return "—" if m["rtx"] is None else f'{m["rtx"][mode][1]}{m["rtx"][mode][0]:,}'

    speed_rows = "\n".join(
        f'<tr><td>{m["model"]}<br><small>{m["config"]}</small></td>{cell(m["trn3_tp4"])}'
        f'{cell(m["trn3"])}<td>{m["layout"]}</td>{cell(rtx(m, "compiled"))}{cell(rtx(m, "eager"))}'
        f'{cell(m["warm"])}{cell(m["cold"])}{cell(m["compile"])}{cell(m["rtx_request"])}'
        f'<td>{m["source"]}</td></tr>' for m in SPEED)
    quality_rows = "\n".join(
        f'<tr><td>{q["model"]}</td><td>{q["layout"]}</td>{cell(q["trn3"][0], ".2f")}'
        f'{cell(q["trn3"][1], ".3f")}{cell(q["rtx"][0], ".2f")}{cell(q["rtx"][1], ".3f")}</tr>' for q in QUALITY)
    data = {"speed": SPEED, "quality": QUALITY, "level": LEVEL}
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trainium3 vs RTX PRO 6000</title>
<style>{_CSS}{_COMPARE_CSS}</style>
</head>
<body>
<header>
<div><div class="nav"><a href="./">← Difflet models</a></div>
<h1>Trainium3 vs NVIDIA RTX PRO 6000</h1>
<p>One Trainium3 chip (trn3.48xlarge, 4 logical cores) against one RTX PRO 6000 (96 GB): bf16,
50 steps, seed 42. RTX numbers come from Difflet's existing GPU verification records; they were
not re-measured for this page.</p></div>
</header>

<section class="chart-card" aria-labelledby="speed-title">
<div class="card-head">
<div><h2 id="speed-title">Time per denoising step (DiT)</h2>
<p class="summary" id="speed-summary" aria-live="polite"></p></div>
<div class="toggle" role="group" aria-label="RTX PRO 6000 mode">
<button type="button" data-group="rtx" data-value="compiled" aria-pressed="true">RTX torch.compile</button>
<button type="button" data-group="rtx" data-value="eager" aria-pressed="false">RTX eager</button>
</div>
</div>
<div class="legend"><span><span class="swatch" style="background:var(--trn3)"></span>Trainium3, best
single-chip layout</span><span><span class="swatch" style="background:var(--rtx)"></span>RTX PRO
6000, Difflet CUDA backend</span><span><span class="swatch bound"></span>RTX
upper bound (whole request ÷ steps)</span></div>
<div class="rows" id="speed-rows"></div>
<p class="note">Shorter is faster. Each model's two bars share one scale (step times range from
0.2 to 10 s across models). trn3: median of the 49 step intervals of a warm 50-step run. ≤ / ≥:
the RTX step is a whole-request time divided by its steps, so it overstates the RTX step; ≈: two
CFG passes from one measured DiT forward.</p>
</section>

<section class="chart-card" aria-labelledby="quality-title">
<div class="card-head">
<div><h2 id="quality-title">Output quality against an fp32 reference</h2>
<p class="summary" id="quality-summary" aria-live="polite"></p></div>
<div class="toggle" role="group" aria-label="Quality metric">
<button type="button" data-group="q" data-value="psnr" aria-pressed="true">PSNR</button>
<button type="button" data-group="q" data-value="ssim" aria-pressed="false">SSIM</button>
</div>
</div>
<div class="legend"><span><span class="swatch" style="background:var(--trn3)"></span>Trainium3 bf16
vs fp32</span><span><span class="swatch" style="background:var(--rtx)"></span>RTX PRO 6000
diffusers bf16 vs fp32</span></div>
<div class="rows" id="quality-rows"></div>
<div class="axis"><span></span><div class="ticks" id="quality-ticks"></div><span></span></div>
<p class="note">Longer is closer to the reference. Reference: diffusers 0.38.0 in fp32 on the
RTX PRO 6000 with TF32 off; the RTX bar is the same pipeline in bf16 on the same card, the error
bf16 alone introduces. Both sides start from identical initial latents. One prompt and one seed
per model: this detects regressions but does not characterise quality (Qwen-Image moves from
−6.1 dB to +1.5 dB against the RTX bar with a second prompt). LTX-2 with an fp32 timestep:
27.62 dB / 0.939.</p>
</section>

<section class="chart-card" aria-labelledby="raw-title">
<h2 id="raw-title">Raw numbers</h2>
<div class="scroll"><table>
<thead><tr><th>Model</th><th class="num">trn3 step TP4 (ms)</th><th class="num">trn3 step best (ms)</th>
<th>trn3 best layout</th><th class="num">RTX step compiled (ms)</th><th class="num">RTX step eager (ms)</th>
<th class="num">trn3 warm e2e (s)</th><th class="num">trn3 cold e2e (s)</th><th class="num">trn3 compile (s)</th>
<th class="num">RTX request eager / compiled (s)</th><th>RTX source</th></tr></thead>
<tbody>
{speed_rows}
</tbody></table></div>
<p class="note">trn3 warm e2e is one <code>generate</code> process (launch, load, warm-up, encode,
50 steps, decode, save), cold e2e the same after dropping the page cache, compile a build into an
empty cache. RTX request times exclude loading and use each verification run's step count, so the
two end-to-end columns are not comparable and the charts compare per-step time only.</p>
<div class="scroll"><table>
<thead><tr><th>Model</th><th>trn3 layout</th><th class="num">trn3 PSNR (dB)</th><th class="num">trn3 SSIM</th>
<th class="num">RTX bf16 PSNR (dB)</th><th class="num">RTX bf16 SSIM</th></tr></thead>
<tbody>
{quality_rows}
</tbody></table></div>
</section>
<div id="tip" hidden></div>
<script type="application/json" id="compare-data">{_json(data)}</script>
<script>{_COMPARE_SCRIPT}</script>
</body>
</html>
"""
