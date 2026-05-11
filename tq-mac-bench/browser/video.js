import * as ort from '/ort/dist/ort.webgpu.bundle.min.mjs';
window.ort = ort;

const MODELS = {
  lfm: {
    url: '/models/lfm/model_q4f16_lasttok.onnx',
    meta: '/models/lfm/model_q4f16_lasttok.meta.json',
    label: 'LFM2.5-1.2B-Instruct',
    slot_bytes: 36,
    eos: [7],
  },
  qwen3: {
    url: '/models/qwen3/model_q4f16_lasttok.onnx',
    meta: '/models/qwen3/model_q4f16_lasttok.meta.json',
    label: 'Qwen3-0.6B',
    slot_bytes: 68,
    eos: [151645, 151643],
  },
};

const $ = (id) => document.getElementById(id);
const log = (s, cls) => {
  const el = $('log');
  const span = document.createElement('span');
  if (cls) span.className = cls;
  span.textContent = s + '\n';
  el.appendChild(span);
  el.scrollTop = el.scrollHeight;
};
const setStatus = (s) => { $('status').textContent = s; };

async function tokenize(model, text) {
  const r = await fetch('/tokenize', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model, chat: [{ role: 'user', content: text }] }),
  });
  if (!r.ok) throw new Error('tokenize failed: ' + (await r.text()));
  return (await r.json()).ids;
}

async function detokenize(model, ids) {
  if (!ids.length) return { pieces: [] };
  const r = await fetch('/detokenize', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model, ids }),
  });
  return await r.json();
}

async function makeSession(modelUrl, useTq) {
  const opts = {
    executionProviders: ['webgpu'],
    graphOptimizationLevel: 'all',
    logSeverityLevel: 2,
  };
  if (useTq) {
    opts.extra = {
      'optimization.turboquant_kv_method': 'turboquant_4bit_nc',
      'optimization.turboquant_kv_boundary': '0',
    };
  }
  const baseName = modelUrl.split('/').pop() + '_data';
  opts.externalData = [{ data: modelUrl + '_data', path: baseName }];
  return await ort.InferenceSession.create(modelUrl, opts);
}

function emptyPastKV(meta, useTq, slot_bytes, pastSeq) {
  const out = {};
  for (const spec of meta.inputs) {
    if (!spec.name.startsWith('past_key_values') && !spec.name.startsWith('past_conv')) continue;
    const shape = spec.shape.map((d) => {
      if (typeof d === 'number') return d;
      if (d === 'batch_size') return 1;
      if (d === 'past_sequence_length') return pastSeq;
      return 0;
    });
    let dtype = spec.type;
    let s = [...shape];
    if (useTq && spec.name.startsWith('past_key_values') && shape.length === 4) {
      s[s.length - 1] = slot_bytes;
      dtype = 'uint8';
    }
    const size = s.reduce((a, b) => a * b, 1) || 0;
    let data;
    if (dtype === 'uint8') data = new Uint8Array(size);
    else if (dtype === 'int64') data = new BigInt64Array(size);
    else if (dtype === 'float16') data = new Uint16Array(size);
    else if (dtype === 'float32') data = new Float32Array(size);
    else throw new Error('unsupported type: ' + dtype);
    out[spec.name] = new ort.Tensor(dtype, data, s);
  }
  return out;
}

function argmaxLastRow(logitsTensor) {
  // logits: [1, seq, vocab] (lasttok export → seq=1)
  const dims = logitsTensor.dims;
  const vocab = dims[dims.length - 1];
  const seq = dims[dims.length - 2];
  const data = logitsTensor.data; // Uint16Array (float16) or Float32Array
  const offset = (seq - 1) * vocab;
  let best = -Infinity, bestIdx = 0;
  if (data instanceof Uint16Array) {
    // float16 → float32 manually
    for (let i = 0; i < vocab; i++) {
      const h = data[offset + i];
      const f = f16ToF32(h);
      if (f > best) { best = f; bestIdx = i; }
    }
  } else {
    for (let i = 0; i < vocab; i++) {
      const f = data[offset + i];
      if (f > best) { best = f; bestIdx = i; }
    }
  }
  return bestIdx;
}

function f16ToF32(h) {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

function pickPresent(outs, meta) {
  const fed = {};
  let pastSeq = 0;
  for (const name of meta.outputs) {
    if (name.startsWith('present.') && (name.includes('.key') || name.includes('.value'))) {
      fed['past_key_values.' + name.slice('present.'.length)] = outs[name];
      pastSeq = outs[name].dims[2];
    } else if (name.startsWith('present_conv.')) {
      fed['past_conv.' + name.slice('present_conv.'.length)] = outs[name];
    } else if (name.startsWith('present_recurrent.')) {
      fed['past_recurrent.' + name.slice('present_recurrent.'.length)] = outs[name];
    }
  }
  return { fed, pastSeq };
}

async function generate({ modelKey, useTq, promptIds, padIds, maxNew, paneEl, statsEl }) {
  const cfg = MODELS[modelKey];
  const meta = await (await fetch(cfg.meta)).json();
  const sess = await makeSession(cfg.url, useTq);

  const events = [];
  const generatedIds = [];

  // Step 1: pad context (synthetic ids) to inflate KV cache, if requested.
  let past;
  let totalContext = 0;
  if (padIds.length > 0) {
    const ids = new BigInt64Array(padIds.length);
    for (let i = 0; i < padIds.length; i++) ids[i] = BigInt(padIds[i]);
    const empty = emptyPastKV(meta, useTq, cfg.slot_bytes, 0);
    const inputs = {
      ...empty,
      input_ids: new ort.Tensor('int64', ids, [1, padIds.length]),
      attention_mask: new ort.Tensor('int64', new BigInt64Array(padIds.length).fill(1n), [1, padIds.length]),
      position_ids: new ort.Tensor('int64', BigInt64Array.from(padIds.map((_, i) => BigInt(i))), [1, padIds.length]),
    };
    const t0 = performance.now();
    const outs = await sess.run(inputs);
    log(`  pad-context (${padIds.length} tok): ${(performance.now() - t0).toFixed(0)} ms`);
    past = pickPresent(outs, meta);
    totalContext = padIds.length;
  } else {
    past = { fed: emptyPastKV(meta, useTq, cfg.slot_bytes, 0), pastSeq: 0 };
  }

  // Step 2: real prompt forward pass on top of (possibly empty) padded cache.
  const promptArr = new BigInt64Array(promptIds.length);
  for (let i = 0; i < promptIds.length; i++) promptArr[i] = BigInt(promptIds[i]);
  const promptTotal = totalContext + promptIds.length;
  const promptInputs = {
    ...past.fed,
    input_ids: new ort.Tensor('int64', promptArr, [1, promptIds.length]),
    attention_mask: new ort.Tensor('int64', new BigInt64Array(promptTotal).fill(1n), [1, promptTotal]),
    position_ids: new ort.Tensor('int64', BigInt64Array.from(promptIds.map((_, i) => BigInt(totalContext + i))), [1, promptIds.length]),
  };
  const tPrompt0 = performance.now();
  let outs = await sess.run(promptInputs);
  const promptMs = performance.now() - tPrompt0;
  log(`  prompt (${promptIds.length} tok): ${promptMs.toFixed(0)} ms`);
  past = pickPresent(outs, meta);
  totalContext += promptIds.length;

  let nextId = argmaxLastRow(outs.logits);

  // Step 3: greedy decode loop with per-step timings.
  let tCum = 0;
  const eosSet = new Set(cfg.eos);
  const decodeStart = performance.now();
  for (let s = 0; s < maxNew; s++) {
    if (eosSet.has(nextId)) {
      log(`  eos at step ${s}`);
      break;
    }
    generatedIds.push(nextId);
    const stepInputs = {
      ...past.fed,
      input_ids: new ort.Tensor('int64', new BigInt64Array([BigInt(nextId)]), [1, 1]),
      attention_mask: new ort.Tensor('int64', new BigInt64Array(totalContext + 1).fill(1n), [1, totalContext + 1]),
      position_ids: new ort.Tensor('int64', new BigInt64Array([BigInt(totalContext)]), [1, 1]),
    };
    const t0 = performance.now();
    outs = await sess.run(stepInputs);
    const dt = performance.now() - t0;
    tCum += dt;
    past = pickPresent(outs, meta);
    totalContext += 1;
    const sampled = argmaxLastRow(outs.logits);
    events.push({ step: s, t_ms: +tCum.toFixed(2), dt_ms: +dt.toFixed(2), token_id: nextId });
    // live preview using just-generated id (we'll fill `token` text after a batch detok)
    if (s % 4 === 3 || s === maxNew - 1) {
      const { pieces } = await detokenize(modelKey, generatedIds);
      paneEl.textContent = pieces.join('');
      const tokps = (s + 1) / ((performance.now() - decodeStart) / 1000);
      statsEl.textContent = `${s + 1} tok · ${tokps.toFixed(1)} tok/s · ${tCum.toFixed(0)} ms`;
    }
    nextId = sampled;
  }

  // Final detokenize: attach piece string to each event.
  const { pieces } = await detokenize(modelKey, generatedIds);
  for (let i = 0; i < events.length; i++) events[i].token = pieces[i] ?? '';
  paneEl.textContent = pieces.join('');

  sess.release();
  return { promptMs, totalDecodeMs: tCum, events, generatedIds, finalText: pieces.join('') };
}

$('run').onclick = async () => {
  const btn = $('run');
  btn.disabled = true;
  $('log').textContent = '';
  $('fp16-out').textContent = '';
  $('tq-out').textContent = '';
  $('fp16-stats').textContent = '';
  $('tq-stats').textContent = '';
  try {
    const modelKey = $('model').value;
    const cfg = MODELS[modelKey];
    const promptText = $('prompt').value;
    const maxNew = parseInt($('maxtok').value, 10);
    const padCtx = parseInt($('padctx').value, 10);

    log(`# Model: ${cfg.label}`, 'head');
    log(`# Prompt: ${promptText}`);

    setStatus('tokenizing prompt…');
    const promptIds = await tokenize(modelKey, promptText);
    log(`  prompt tokens: ${promptIds.length}`);

    // Synthetic padding ids (just to inflate KV cache realistically).
    const padIds = [];
    for (let i = 0; i < padCtx; i++) padIds.push((i * 1103515245 + 12345) & 0x7fff);
    log(`  pad-context tokens: ${padIds.length} (KV cache will hold ${padIds.length + promptIds.length} tok before decode)`);

    const adapter = await navigator.gpu?.requestAdapter();
    log(`# WebGPU adapter: ${adapter?.info?.vendor || 'unknown'}`);

    const results = {};
    for (const useTq of [false, true]) {
      const variant = useTq ? 'tq' : 'fp16';
      log('');
      log(`== ${variant} ==`, 'head');
      setStatus(`generating ${variant}…`);
      results[variant] = await generate({
        modelKey, useTq, promptIds, padIds, maxNew,
        paneEl: $(`${variant}-out`),
        statsEl: $(`${variant}-stats`),
      });
    }

    const speedup = results.fp16.totalDecodeMs / results.tq.totalDecodeMs;
    log('');
    log(`# fp16 decode total: ${results.fp16.totalDecodeMs.toFixed(0)} ms (${results.fp16.events.length} tok)`);
    log(`# TQ   decode total: ${results.tq.totalDecodeMs.toFixed(0)} ms (${results.tq.events.length} tok)`);
    log(`# decode speedup: ${speedup.toFixed(2)}x`, 'good');

    // Persist JSONL + meta
    const stamp = `${modelKey}-pad${padCtx}-n${maxNew}`;
    for (const variant of ['fp16', 'tq']) {
      const r = results[variant];
      const jsonl = r.events.map((e) => JSON.stringify(e)).join('\n') + '\n';
      const fname = `real-${variant}-${stamp}.jsonl`;
      await fetch(`/save?name=${encodeURIComponent(fname)}`, { method: 'POST', body: jsonl });
      log(`  wrote recordings/${fname}`, 'good');
    }
    const metaOut = {
      model: cfg.label,
      prompt: promptText,
      pad_context: padCtx,
      max_new_tokens: maxNew,
      fp16: { promptMs: results.fp16.promptMs, totalDecodeMs: results.fp16.totalDecodeMs, text: results.fp16.finalText, n: results.fp16.events.length },
      tq:   { promptMs: results.tq.promptMs,   totalDecodeMs: results.tq.totalDecodeMs,   text: results.tq.finalText,   n: results.tq.events.length },
      decode_speedup: +speedup.toFixed(2),
      recorded_at: new Date().toISOString(),
    };
    const metaName = `real-meta-${stamp}.json`;
    await fetch(`/save?name=${encodeURIComponent(metaName)}`, {
      method: 'POST', body: JSON.stringify(metaOut, null, 2),
    });
    log(`  wrote recordings/${metaName}`, 'good');
    setStatus(`done — ${speedup.toFixed(2)}x speedup`);
  } catch (e) {
    log('!! ERROR: ' + (e?.stack || e?.message || e), 'bad');
    console.error(e);
  } finally {
    btn.disabled = false;
  }
};
