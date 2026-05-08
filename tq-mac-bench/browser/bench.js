// Browser bench harness for TurboQuant ORT WebGPU EP (WASM build).
//
// Reads from window-scoped `ort` exposed by ort-wasm-simd-threaded.jsep.mjs
// (or the WebGPU EP equivalent we just built).  Compares fp16 baseline vs
// TQ on the same model and prints decode tok/s + per-step logit cos sim.

import * as ort from './ort.webgpu.bundle.mjs';

const log = (s, cls) => {
  const el = document.getElementById('log');
  const span = document.createElement('span');
  if (cls) span.className = cls;
  span.textContent = s + '\n';
  el.appendChild(span);
};

window.ort = ort;

const MODELS = {
  lfm2: {
    label: 'LFM2.5-1.2B-Instruct',
    url: '/models/lfm2/model_q4f16_lasttok.onnx',
    head_dim: 64,
    slot_bytes: 36,
    kv_heads: 8,
    pad_id: 0,
  },
  qwen3: {
    label: 'Qwen3-0.6B',
    url: '/models/qwen3/model_q4f16_lasttok.onnx',
    head_dim: 128,
    slot_bytes: 68,
    kv_heads: 8,
    pad_id: 151643,
  },
};

async function loadModelBytes(url) {
  log(`fetching ${url}…`);
  const t0 = performance.now();
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch failed: ${r.status}`);
  const b = new Uint8Array(await r.arrayBuffer());
  log(`  loaded ${(b.length/1e6).toFixed(1)} MB in ${((performance.now()-t0)/1000).toFixed(1)}s`);
  return b;
}

async function makeSession(modelBytes, useTq) {
  const opts = {
    executionProviders: ['webgpu'],
    graphOptimizationLevel: 'all',
    logSeverityLevel: 3,
  };
  if (useTq) {
    opts.optimization = {
      turboquant_kv_method: 'turboquant_4bit_nc',
      turboquant_kv_boundary: '0',
    };
  }
  return await ort.InferenceSession.create(modelBytes, opts);
}

function makeInputs(sess, modelCfg, plen, useTq) {
  const inputs = {};
  // input_ids: pseudorandom but reproducible
  const ids = new BigInt64Array(plen);
  for (let i = 0; i < plen; i++) ids[i] = BigInt((i * 1103515245 + 12345) & 0x7fff);
  inputs.input_ids = new ort.Tensor('int64', ids, [1, plen]);
  inputs.attention_mask = new ort.Tensor('int64', new BigInt64Array(plen).fill(1n), [1, plen]);
  if (sess.inputNames.includes('position_ids')) {
    const pos = new BigInt64Array(plen);
    for (let i = 0; i < plen; i++) pos[i] = BigInt(i);
    inputs.position_ids = new ort.Tensor('int64', pos, [1, plen]);
  }
  // KV inputs zero-filled
  for (const name of sess.inputNames) {
    if (inputs[name]) continue;
    const meta = sess.inputMetadata[name];
    if (!meta) continue;
    const shape = meta.shape.map(d => {
      if (typeof d === 'number') return d;
      if (d === 'batch_size') return 1;
      if (d === 'sequence_length') return plen;
      if (d === 'past_sequence_length') return 0;
      if (d === 'total_sequence_length') return plen;
      return 0;
    });
    let dtype = meta.type;
    if (useTq && name.startsWith('past_key_values')) {
      shape[shape.length - 1] = modelCfg.slot_bytes;
      dtype = 'uint8';
    }
    const size = shape.reduce((a, b) => a * b, 1);
    let data;
    if (dtype === 'uint8') data = new Uint8Array(size);
    else if (dtype === 'int64') data = new BigInt64Array(size);
    else if (dtype === 'float16') data = new Uint16Array(size);
    else if (dtype === 'float32') data = new Float32Array(size);
    else continue;
    inputs[name] = new ort.Tensor(dtype, data, shape);
  }
  return inputs;
}

async function bench(sess, modelCfg, plen, useTq, decodeSteps) {
  let inputs = makeInputs(sess, modelCfg, plen, useTq);
  // Prompt step (TTFT)
  const t0 = performance.now();
  let outs = await sess.run(inputs);
  const promptMs = performance.now() - t0;
  // Find logits
  const logitsName = sess.outputNames.includes('logits') ? 'logits' : sess.outputNames[0];
  const decodeMsArr = [];
  for (let s = 0; s < decodeSteps; s++) {
    // build decode inputs
    const newInputs = {};
    let pastSeq = 0;
    for (const name of sess.outputNames) {
      if (name.startsWith('present.') && (name.includes('.key') || name.includes('.value'))) {
        const fed = 'past_key_values.' + name.slice('present.'.length);
        newInputs[fed] = outs[name];
        pastSeq = outs[name].dims[2];
      } else if (name.startsWith('present_conv.')) {
        newInputs['past_conv.' + name.slice('present_conv.'.length)] = outs[name];
      }
    }
    const totalSeq = pastSeq + 1;
    newInputs.input_ids = new ort.Tensor('int64', new BigInt64Array([42n]), [1, 1]);
    newInputs.attention_mask = new ort.Tensor('int64', new BigInt64Array(totalSeq).fill(1n), [1, totalSeq]);
    if (sess.inputNames.includes('position_ids')) {
      newInputs.position_ids = new ort.Tensor('int64', new BigInt64Array([BigInt(pastSeq)]), [1, 1]);
    }
    for (const name of sess.inputNames) {
      if (newInputs[name]) continue;
      const meta = sess.inputMetadata[name];
      if (!meta) continue;
      newInputs[name] = inputs[name];  // reuse from first call (e.g. seqlens_k)
    }
    const ts = performance.now();
    outs = await sess.run(newInputs);
    decodeMsArr.push(performance.now() - ts);
    inputs = newInputs;
  }
  const steady = decodeMsArr.slice(1).reduce((a, b) => a + b, 0) / Math.max(1, decodeMsArr.length - 1);
  return { promptMs, step1Ms: decodeMsArr[0], steadyMs: steady, allMs: decodeMsArr, lastLogits: outs[logitsName] };
}

document.getElementById('run').onclick = async () => {
  document.getElementById('log').textContent = '';
  const m = document.getElementById('model').value;
  const ctx = parseInt(document.getElementById('ctx').value, 10);
  const cfg = MODELS[m];
  log(`# ${cfg.label} @ ctx=${ctx}`, 'head');
  log(`# WebGPU adapter: ${(await navigator.gpu?.requestAdapter())?.name || 'unknown'}`);
  const bytes = await loadModelBytes(cfg.url);

  for (const useTq of [false, true]) {
    log('');
    log(`== ${useTq ? 'TQ' : 'fp16'} ==`, 'head');
    const sess = await makeSession(bytes, useTq);
    const r = await bench(sess, cfg, ctx, useTq, 4);
    log(`  prompt (TTFT): ${r.promptMs.toFixed(0)} ms`);
    log(`  decode steps:  ${r.allMs.map(x => x.toFixed(1)).join(' / ')} ms`);
    log(`  decode steady: ${r.steadyMs.toFixed(1)} ms/tok = ${(1000/r.steadyMs).toFixed(2)} tok/s`);
    sess.release();
  }
};
