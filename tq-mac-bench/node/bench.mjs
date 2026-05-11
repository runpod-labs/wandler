// Node.js TurboQuant bench using onnxruntime-node WebGPU EP.
// Same shape as the browser bench but no DOM, no WASM — native NAPI bindings.
//
// Usage:
//   node bench.mjs lfm2 4096
//   node bench.mjs qwen35 16384
//   node bench.mjs qwen35 16384 8         # 8 decode steps instead of 4
import * as ort from 'onnxruntime-node';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

const MODELS = {
  lfm2: {
    label: 'LFM2.5-1.2B-Instruct',
    onnx: path.join(ROOT, 'model/onnx/model_q4f16_lasttok.onnx'),
    meta: path.join(ROOT, 'model/onnx/model_q4f16_lasttok.meta.json'),
    slot_bytes: 36,
  },
  qwen3: {
    label: 'Qwen3-0.6B',
    onnx: path.join(ROOT, 'qwen3/onnx/model_q4f16_lasttok.onnx'),
    meta: path.join(ROOT, 'qwen3/onnx/model_q4f16_lasttok.meta.json'),
    slot_bytes: 68,
  },
  qwen35: {
    label: 'Qwen3.5-0.8B-Text',
    onnx: path.join(ROOT, 'qwen35/onnx/model_q4f16_lasttok.onnx'),
    meta: path.join(ROOT, 'qwen35/onnx/model_q4f16_lasttok.meta.json'),
    slot_bytes: 132,
  },
};

const [, , modelKey = 'qwen35', ctxStr = '4096', stepsStr = '4'] = process.argv;
const cfg = MODELS[modelKey];
if (!cfg) { console.error('unknown model:', modelKey); process.exit(1); }
const ctx = parseInt(ctxStr, 10);
const decodeSteps = parseInt(stepsStr, 10);
const meta = JSON.parse(fs.readFileSync(cfg.meta, 'utf8'));

console.log(`# ${cfg.label} @ ctx=${ctx} (Node.js ORT-node, WebGPU EP via Dawn → Metal)`);

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
function makeInputs(plen, pastSeq, useTq, drivenIds) {
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
    out[spec.name] = makeTensor(spec, shape, useTq, cfg.slot_bytes);
  }
  return out;
}

async function makeSession(useTq) {
  const opts = {
    executionProviders: ['webgpu'],
    graphOptimizationLevel: 'all',
    logSeverityLevel: 3,
  };
  if (useTq) {
    opts.extra = {
      'optimization.turboquant_kv_method': 'turboquant_4bit_nc',
      'optimization.turboquant_kv_boundary': '0',
    };
  }
  return await ort.InferenceSession.create(cfg.onnx, opts);
}

async function bench(useTq) {
  const sess = await makeSession(useTq);
  let inputs = makeInputs(ctx, 0, useTq);
  const t0 = performance.now();
  let outs = await sess.run(inputs);
  const promptMs = performance.now() - t0;
  let kvBytes = 0;
  for (const name of meta.outputs) {
    if (name.startsWith('present.') && (name.includes('.key') || name.includes('.value'))) {
      const t = outs[name];
      const elemBytes = t.type === 'uint8' ? 1 : t.type === 'float16' ? 2 : 4;
      kvBytes += t.dims.reduce((a, b) => a * b, 1) * elemBytes;
    }
  }
  console.log(`  prompt (TTFT): ${promptMs.toFixed(0)} ms`);
  console.log(`  KV cache size: ${(kvBytes / 1024 / 1024).toFixed(1)} MB`);
  const decodeMs = [];
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
    const decodeInputs = makeInputs(1, pastSeq, useTq, drive);
    for (const k in newInputs) decodeInputs[k] = newInputs[k];
    const ts = performance.now();
    outs = await sess.run(decodeInputs);
    decodeMs.push(performance.now() - ts);
    inputs = decodeInputs;
  }
  const steady = decodeMs.length > 1
    ? decodeMs.slice(1).reduce((a, b) => a + b, 0) / (decodeMs.length - 1)
    : decodeMs[0];
  console.log(`  decode steps:  ${decodeMs.map(x => x.toFixed(1)).join(' / ')} ms`);
  console.log(`  decode steady: ${steady.toFixed(1)} ms/tok = ${(1000 / steady).toFixed(2)} tok/s`);
  await sess.release();
}

console.log('\n== fp16 ==');
await bench(false);
console.log('\n== TQ ==');
await bench(true);
