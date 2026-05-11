import * as ort from '/ort/dist/ort.webgpu.bundle.min.mjs';
window.ort = ort;
const log = (s, cls) => {
  const el = document.getElementById('log');
  const span = document.createElement('span');
  if (cls) span.className = cls;
  span.textContent = s + '\n';
  el.appendChild(span);
};
const MODELS = {
  lfm2: { url: '/models/lfm/model_q4f16_lasttok.onnx', meta: '/models/lfm/model_q4f16_lasttok.meta.json', label: 'LFM2.5-1.2B-Instruct', slot_bytes: 36 },
  qwen3: { url: '/models/qwen3/model_q4f16_lasttok.onnx', meta: '/models/qwen3/model_q4f16_lasttok.meta.json', label: 'Qwen3-0.6B', slot_bytes: 68 },
  qwen35: { url: '/models/qwen35/model_q4f16_lasttok.onnx', meta: '/models/qwen35/model_q4f16_lasttok.meta.json', label: 'Qwen3.5-0.8B-Text', slot_bytes: 132 },
};

async function makeSession(modelUrl, useTq) {
  const opts = {
    executionProviders: ['webgpu'],
    graphOptimizationLevel: 'all',
    logSeverityLevel: 0,
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

function resolveDim(d, plen, pastSeq, totalSeq) {
  if (typeof d === 'number') return d;
  if (d === 'batch_size') return 1;
  if (d === 'sequence_length') return plen;
  if (d === 'past_sequence_length') return pastSeq;
  if (d === 'total_sequence_length') return totalSeq;
  return 0;
}
function makeTensor(spec, shape, useTq, slot_bytes) {
  let dtype = spec.type;
  if (useTq && spec.name.startsWith('past_key_values') && shape.length === 4) {
    shape = [...shape];
    shape[shape.length - 1] = slot_bytes;
    dtype = 'uint8';
  }
  const size = shape.reduce((a, b) => a * b, 1);
  let data;
  if (dtype === 'uint8') data = new Uint8Array(size);
  else if (dtype === 'int64') data = new BigInt64Array(size);
  else if (dtype === 'float16') data = new Uint16Array(size);
  else if (dtype === 'float32') data = new Float32Array(size);
  else if (dtype === 'bool') data = new Uint8Array(size);
  else throw new Error('unsupported type: ' + dtype);
  return new ort.Tensor(dtype, data, shape);
}
function makeInputs(meta, plen, pastSeq, useTq, slot_bytes, drivenIds) {
  const totalSeq = pastSeq + plen;
  const out = {};
  out.input_ids = drivenIds || (() => {
    const a = new BigInt64Array(plen);
    for (let i = 0; i < plen; i++) a[i] = BigInt((i * 1103515245 + 12345) & 0x7fff);
    return new ort.Tensor('int64', a, [1, plen]);
  })();
  out.attention_mask = new ort.Tensor('int64', new BigInt64Array(totalSeq).fill(1n), [1, totalSeq]);
  for (const spec of meta.inputs) {
    if (out[spec.name]) continue;
    if (spec.name === 'position_ids') {
      const pos = new BigInt64Array(plen);
      for (let i = 0; i < plen; i++) pos[i] = BigInt(pastSeq + i);
      out[spec.name] = new ort.Tensor('int64', pos, [1, plen]);
      continue;
    }
    const shape = spec.shape.map(d => resolveDim(d, plen, pastSeq, totalSeq));
    out[spec.name] = makeTensor(spec, shape, useTq, slot_bytes);
  }
  return out;
}
async function bench(sess, meta, modelCfg, plen, useTq, decodeSteps) {
  let inputs = makeInputs(meta, plen, 0, useTq, modelCfg.slot_bytes);
  log('  prompt sess.run started…');
  const t0 = performance.now();
  let outs = await sess.run(inputs);
  const promptMs = performance.now() - t0;
  log(`  prompt done in ${promptMs.toFixed(0)} ms`);
  let kvBytes = 0;
  for (const name of meta.outputs) {
    if (name.startsWith('present.') && (name.includes('.key') || name.includes('.value'))) {
      const t = outs[name];
      const elemBytes = t.type === 'uint8' ? 1 : t.type === 'float16' ? 2 : 4;
      kvBytes += t.dims.reduce((a, b) => a * b, 1) * elemBytes;
    }
  }
  log(`  KV cache size after prompt: ${(kvBytes / 1024 / 1024).toFixed(1)} MB`);
  const decodeMsArr = [];
  for (let s = 0; s < decodeSteps; s++) {
    const newInputs = {};
    let pastSeq = 0;
    for (const name of meta.outputs) {
      if (name.startsWith('present.') && (name.includes('.key') || name.includes('.value'))) {
        const fed = 'past_key_values.' + name.slice('present.'.length);
        newInputs[fed] = outs[name];
        pastSeq = outs[name].dims[2];
      } else if (name.startsWith('present_conv.')) {
        newInputs['past_conv.' + name.slice('present_conv.'.length)] = outs[name];
      } else if (name.startsWith('present_recurrent.')) {
        newInputs['past_recurrent.' + name.slice('present_recurrent.'.length)] = outs[name];
      }
    }
    const drive = new ort.Tensor('int64', new BigInt64Array([42n]), [1, 1]);
    const decodeInputs = makeInputs(meta, 1, pastSeq, useTq, modelCfg.slot_bytes, drive);
    for (const k in newInputs) decodeInputs[k] = newInputs[k];
    log(`  decode step ${s + 1}…`);
    const ts = performance.now();
    outs = await sess.run(decodeInputs);
    const dms = performance.now() - ts;
    decodeMsArr.push(dms);
    log(`    -> ${dms.toFixed(1)} ms`);
    inputs = decodeInputs;
  }
  const steady = decodeMsArr.length > 1 ? decodeMsArr.slice(1).reduce((a, b) => a + b, 0) / (decodeMsArr.length - 1) : decodeMsArr[0];
  return { promptMs, allMs: decodeMsArr, steadyMs: steady };
}

// Canned prompt + response used purely for the video overlay. Timings come
// from the real bench; tokens are revealed one per measured decode step.
const SAMPLE = {
  prompt: 'Write a haiku about fast inference.',
  response: [
    'Tokens', ' fly', ' like', ' light',
    ',\n', 'cache', ' compressed', ' to', ' four', ' bits',
    ',\n', 'silicon', ' sings', ' fast', '.',
  ],
};

async function record(modelKey, ctx, decodeSteps) {
  const cfg = MODELS[modelKey];
  const meta = await (await fetch(cfg.meta)).json();
  const out = {};
  for (const useTq of [false, true]) {
    const variant = useTq ? 'tq' : 'fp16';
    log('');
    log(`== recording ${variant} ==`, 'head');
    const sess = await makeSession(cfg.url, useTq);
    // Warm prompt
    let inputs = makeInputs(meta, ctx, 0, useTq, cfg.slot_bytes);
    const tPrompt0 = performance.now();
    let outs = await sess.run(inputs);
    const promptMs = performance.now() - tPrompt0;
    log(`  prompt: ${promptMs.toFixed(0)} ms`);

    const events = [];
    let tCum = 0;
    for (let s = 0; s < decodeSteps; s++) {
      const newInputs = {};
      let pastSeq = 0;
      for (const name of meta.outputs) {
        if (name.startsWith('present.') && (name.includes('.key') || name.includes('.value'))) {
          newInputs['past_key_values.' + name.slice('present.'.length)] = outs[name];
          pastSeq = outs[name].dims[2];
        } else if (name.startsWith('present_conv.')) {
          newInputs['past_conv.' + name.slice('present_conv.'.length)] = outs[name];
        } else if (name.startsWith('present_recurrent.')) {
          newInputs['past_recurrent.' + name.slice('present_recurrent.'.length)] = outs[name];
        }
      }
      const drive = new ort.Tensor('int64', new BigInt64Array([42n]), [1, 1]);
      const decodeInputs = makeInputs(meta, 1, pastSeq, useTq, cfg.slot_bytes, drive);
      for (const k in newInputs) decodeInputs[k] = newInputs[k];
      const t0 = performance.now();
      outs = await sess.run(decodeInputs);
      const dt = performance.now() - t0;
      tCum += dt;
      const token = SAMPLE.response[s % SAMPLE.response.length];
      events.push({ step: s, t_ms: +tCum.toFixed(2), dt_ms: +dt.toFixed(2), token });
    }
    sess.release();

    const jsonl = events.map(e => JSON.stringify(e)).join('\n') + '\n';
    const fname = `${variant}-${modelKey}-ctx${ctx}.jsonl`;
    await fetch(`/save?name=${encodeURIComponent(fname)}`, { method: 'POST', body: jsonl });
    log(`  wrote recordings/${fname} (${events.length} steps, total ${tCum.toFixed(0)} ms)`, 'good');
    out[variant] = { promptMs, totalMs: tCum, file: fname };
  }

  const metaOut = {
    model: cfg.label,
    ctx,
    decodeSteps,
    prompt: SAMPLE.prompt,
    response_tokens: SAMPLE.response.slice(0, decodeSteps),
    fp16: out.fp16,
    tq: out.tq,
    speedup: +(out.fp16.totalMs / out.tq.totalMs).toFixed(2),
    recorded_at: new Date().toISOString(),
  };
  const metaName = `meta-${modelKey}-ctx${ctx}.json`;
  await fetch(`/save?name=${encodeURIComponent(metaName)}`, {
    method: 'POST',
    body: JSON.stringify(metaOut, null, 2),
  });
  log(`  wrote recordings/${metaName} — speedup ${metaOut.speedup}x`, 'good');
}

document.getElementById('rec').onclick = async () => {
  const btn = document.getElementById('rec');
  btn.disabled = true;
  try {
    document.getElementById('log').textContent = '';
    const m = document.getElementById('model').value;
    const ctx = parseInt(document.getElementById('ctx').value, 10);
    const steps = parseInt(document.getElementById('steps').value, 10);
    log(`# RECORDING for video — model=${m} ctx=${ctx} steps=${steps}`, 'head');
    await record(m, ctx, steps);
    log('# done. Files in tq-mac-bench/browser/recordings/', 'good');
  } catch (e) {
    log('!! ERROR: ' + (e?.stack || e?.message || e), 'bad');
    console.error(e);
  } finally {
    btn.disabled = false;
  }
};

document.getElementById('run').onclick = async () => {
  document.getElementById('run').disabled = true;
  try {
    document.getElementById('log').textContent = '';
    const m = document.getElementById('model').value;
    const ctx = parseInt(document.getElementById('ctx').value, 10);
    const cfg = MODELS[m];
    log(`# ${cfg.label} @ ctx=${ctx}`, 'head');
    const adapter = await navigator.gpu?.requestAdapter();
    log(`# WebGPU adapter: ${adapter?.info?.vendor || 'unknown'}`);
    const meta = await (await fetch(cfg.meta)).json();
    log(`# meta: ${meta.inputs.length} inputs, ${meta.outputs.length} outputs`);
    for (const useTq of [false, true]) {
      log('');
      log(`== ${useTq ? 'TQ' : 'fp16'} ==`, 'head');
      const t0 = performance.now();
      const sess = await makeSession(cfg.url, useTq);
      log(`  session.create: ${(performance.now() - t0).toFixed(0)} ms`);
      const r = await bench(sess, meta, cfg, ctx, useTq, 4);
      log(`  prompt (TTFT): ${r.promptMs.toFixed(0)} ms`);
      log(`  decode steady: ${r.steadyMs.toFixed(1)} ms/tok = ${(1000/r.steadyMs).toFixed(2)} tok/s`);
      sess.release();
    }
  } catch (e) {
    log('!! ERROR: ' + (e?.stack || e?.message || e), 'bad');
    console.error(e);
  } finally {
    document.getElementById('run').disabled = false;
  }
};
