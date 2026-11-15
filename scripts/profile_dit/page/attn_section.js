  // ---- MXFP8 attention: bf16 vs both (Task 6) ----
  const MX_KEYS = ['qwen_image', 'flux', 'ltx_2', 'wan21_official', 'wan22_official', 'hunyuan_video_L', 'hv15_official'];
  const MX_PHASES = ['qk_matmul', 'mask_softmax', 'exp', 'pv_matmul', 'qk_quantize', 'v_quantize', 'p_quantize', 'other'];
  const MX_PHASE_LABEL = {qk_matmul: 'MM1 (QK^T)', mask_softmax: 'mask / softmax', exp: 'exp', pv_matmul: 'MM2 (PV)', qk_quantize: 'Q+K quantize', v_quantize: 'V quantize', p_quantize: 'P quantize', other: 'other'};
  const mxColor = (p) => (p !== 'other' && COLORS[MX_PHASES.indexOf(p)]) || 'var(--cds-chart-muted)';
  document.querySelectorAll('#dbp .sw[data-phase]').forEach((s) => { s.style.background = mxColor(s.dataset.phase); });
  document.querySelectorAll('#dbp .sw[data-color]').forEach((s) => { s.style.background = COLORS[+s.dataset.color] || 'var(--cds-chart-muted)'; });
  const mxRows = () => [...rows('attn_runs')].sort((a, b) => MX_KEYS.indexOf(a.key) - MX_KEYS.indexOf(b.key));
  const n3 = (v) => v == null ? '—' : d3.format('.3f')(v);
  const r3 = (v) => v == null ? '—' : `${n3(v)}x`;
  const qcell = (psnr, worst, ssim, lpips) => psnr == null ? '—' : `${f2(psnr)}${worst == null ? '' : ` (${f2(worst)})`} / ${n3(ssim)} / ${n3(lpips)}`;

  dash.draw('mx-sum', (el) => {
    if (el.dataset.part === 'm') { el.replaceChildren(String(MX_KEYS.length)); return; }
    if (waiting(el, 'attn_runs')) return;
    const all = rows('attn_runs');
    const span = (vals) => {
      const v = vals.filter((x) => x != null);
      return !v.length ? '—' : d3.min(v) === d3.max(v) ? r3(v[0]) : `${r3(d3.min(v))}–${r3(d3.max(v))}`;
    };
    const v = {
      n: all.length, arm: span(all.map((r) => r.arm_over_bf16)), linear: span(all.map((r) => r.linear_over_bf16)),
      g3: all.filter((r) => r.g3_faster).length,
    }[el.dataset.part];
    el.replaceChildren(String(v ?? '—'));
  });

  dash.draw('mx-table', (tb) => {
    if (waiting(tb, 'attn_runs')) return;
    const sp = d3.format('.2%'), p1 = d3.format('.1%');
    tb.replaceChildren(...mxRows().map((r) => {
      const tr = document.createElement('tr');
      const dim = r.below_floor ? 'muted' : '';  // not a quality reading (the plan's caveat)
      [[r.model], [r.shape], [r.arm],
        [`${f1(r.bf16_ms)} (${sp(r.bf16_spread)})`, 'num'], [`${f1(r.arm_ms)} (${sp(r.arm_spread)})`, 'num'], [r3(r.arm_over_bf16), 'num'],
        [r.linear_over_bf16 == null ? 'none (no linear default)' : `${r3(r.linear_over_bf16)} (${r.linear_source === 'this session' ? '' : '§'}${r.linear_source})`, 'num'],
        [`${r.g3_faster ? 'faster' : 'not faster'}: ${n3(r.arm_over_bf16)} ${r.g3_faster ? '<' : '≥'} ${n3(r.g3_threshold)} (J ${p1(r.J)})`],
        [qcell(r.psnr, r.psnr_worst, r.ssim, r.lpips), `num ${dim}`], [r.latents_cos == null ? '—' : d3.format('.4f')(r.latents_cos), `num ${dim}`],
        [r.lin_psnr == null ? '—' : `${qcell(r.lin_psnr, r.lin_psnr_worst, r.lin_ssim, r.lin_lpips)} (§${r.lin_source})`, 'num'], [r.floor || '—'],
        [qcell(r.fp32_psnr, r.fp32_psnr_worst, r.fp32_ssim, r.fp32_lpips), 'num'],
        [r.bf16_fp32_psnr == null ? '—' : `${f2(r.bf16_fp32_psnr)} · ${r.ev_bf16_fp32_psnr == null ? '—' : f2(r.ev_bf16_fp32_psnr)}`, 'num'],
        [r.ev_linear_fp32_psnr == null ? '—' : `${f2(r.ev_linear_fp32_psnr)} / ${n3(r.ev_linear_fp32_ssim)}`, 'num'],
        [[r.caveat, r.k_smoothing].filter(Boolean).join('; ') || '—', 'wrap']]
        .forEach(([text, cls]) => { const td = document.createElement('td'); td.textContent = text; if (cls) td.className = cls.trim(); tr.append(td); });
      return tr;
    }));
  });

  /** horizontal bars: the quantized arm's DiT step over bf16, and linear alone's, per model, against bf16 = 1 */
  dash.draw('mx-ratio', (el) => {
    if (waiting(el, 'attn_runs')) return;
    const data = mxRows();
    if (!data.length) { el.replaceChildren(); return; }
    const items = data.flatMap((r) => [{header: r.model},
      {r, label: `${r.arm} / bf16`, v: r.arm_over_bf16, fill: COLORS[0] || 'var(--cds-chart-muted)', note: `G3 ${r.g3_faster ? 'faster' : 'not faster'}`},
      ...(r.linear_over_bf16 == null ? [] : [{r, label: 'linear / bf16', v: r.linear_over_bf16, fill: 'var(--cds-chart-muted)', note: `linear alone, ${r.linear_source}`}])]);
    const W = Math.max(320, el.clientWidth), rowH = 22, barH = 12, top = 18, axisH = 34;
    const H = top + items.length * rowH + axisH;
    const svg = svgIn(el, W, H, `DiT step over bf16 of the quantized arm and of linear MXFP8 alone, ${data.length} models`, 'group');
    const tip = tooltip(el);
    const left = widest(svg, items.filter((i) => !i.header).map((i) => i.label)) + 12;
    const right = widest(svg, ['0.000x']) + 12;
    const x = d3.scaleLinear().domain([0, Math.max(1, d3.max(items, (i) => i.v || 0))]).nice().range([left, Math.max(left + 40, W - right)]);
    const ticks = Math.max(2, Math.floor((W - left - right) / 80));
    svg.append('g').attr('class', 'grid').attr('transform', `translate(0,${H - axisH})`).call(d3.axisBottom(x).ticks(ticks).tickSize(-(H - axisH - top)).tickFormat(''));
    items.forEach((it, i) => {
      const y = top + i * rowH;
      if (it.header) { svg.append('text').attr('class', 'head').attr('x', 0).attr('y', y + rowH / 2).attr('dy', '0.35em').text(it.header); return; }
      svg.append('text').attr('class', 'lbl').attr('x', left - 8).attr('y', y + rowH / 2).attr('dy', '0.35em').attr('text-anchor', 'end').text(it.label);
      svg.append('rect').attr('class', 'mark').attr('tabindex', 0).attr('x', x(0)).attr('y', y + (rowH - barH) / 2)
        .attr('width', Math.max(1, x(it.v) - x(0))).attr('height', barH).attr('rx', 2).style('fill', it.fill)
        .attr('aria-label', `${it.r.model}, ${it.label}: ${r3(it.v)}`)
        .on('pointermove focus', (ev) => tip.show(ev, [[r3(it.v), it.label], [it.r.model, it.note]]))
        .on('pointerleave blur', tip.hide);
      svg.append('text').attr('class', 'lbl').attr('x', x(it.v) + 6).attr('y', y + rowH / 2).attr('dy', '0.35em').text(r3(it.v));
    });
    svg.append('line').attr('class', 'ref').attr('x1', x(1)).attr('x2', x(1)).attr('y1', top - 4).attr('y2', H - axisH);
    svg.append('text').attr('class', 'lbl').attr('x', x(1)).attr('y', top - 8).attr('text-anchor', 'middle').text('bf16');
    svg.append('g').attr('transform', `translate(0,${H - axisH})`).call(d3.axisBottom(x).ticks(ticks).tickFormat((v) => `${d3.format('~g')(v)}x`));
  });

  dash.draw('mx-phases', (el) => {
    if (waiting(el, 'attn_runs')) return;
    const data = mxRows();
    if (!data.length) { el.replaceChildren(); return; }
    const items = data.flatMap((r) => [{header: r.model}, ...[['bf16', 'bf16'], ['arm', r.arm]].map(([pre, label]) => {
      const parts = MX_PHASES.map((p) => ({name: MX_PHASE_LABEL[p], ms: r[`${pre}_${p}_ms`] || 0, fill: mxColor(p)}));
      return {label, parts, end: `${fmtMs(d3.sum(parts, (p) => p.ms))} ms`};
    })]);
    stackRows(el, items, `Routed attention per step by phase, bf16 against the quantized arm, ${data.length} models`, 'ms per step of busy time, engines added');
  });
