// Compare TQ vs fp16 logits in Node — feed SAME tokens to both sessions.
import * as ort from 'onnxruntime-node';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = '/Users/timpietrusky/.t3/worktrees/wandler/t3code-eb53f7f5/tq-mac-bench/qwen35';
const ONNX = `${MODEL_DIR}/onnx/model_q4f16_lasttok.onnx`;
const META = JSON.parse(fs.readFileSync(`${MODEL_DIR}/onnx/model_q4f16_lasttok.meta.json`, 'utf8'));
const SLOT_BYTES = 132;
const PROMPT_TOKEN_IDS = [785n, 6722n, 315, 9625n, 374n].map(BigInt);  // "The capital of France is"
const N_STEPS = 8;

function resolveDim(d, plen, pastSeq, totalSeq) {
  if (typeof d === 'number') return d;
  if (d === 'batch_size') return 1;
  if (d === 'sequence_length') return plen;
  if (d === 'past_sequence_length') return pastSeq;
  if (d === 'total_sequence_length') return totalSeq;
  return 0;
}
function makeTensor(spec, shape, useTq) {
  let dtype = spec.type;
  if (useTq && spec.name.startsWith('past_key_values') && shape.length === 4) {
    shape = [...shape];
    shape[shape.length - 1] = SLOT_BYTES;
    dtype = 'uint8';
  }
  const size = shape.reduce((a, b) => a * b, 1);
  if (dtype === 'uint8') return new ort.Tensor('uint8', new Uint8Array(size), shape);
  if (dtype === 'int64') return new ort.Tensor('int64', new BigInt64Array(size), shape);
  if (dtype === 'float16') return new ort.Tensor('float16', new Uint16Array(size), shape);
  if (dtype === 'float32') return new ort.Tensor('float32', new Float32Array(size), shape);
  if (dtype === 'bool') return new ort.Tensor('bool', new Uint8Array(size), shape);
  throw new Error('bad dtype ' + dtype);
}
function makeInputs(plen, pastSeq, useTq, drivenIds) {
  const totalSeq = pastSeq + plen;
  const out = {};
  out.input_ids = drivenIds || new ort.Tensor('int64', new BigInt64Array(PROMPT_TOKEN_IDS), [1, plen]);
  out.attention_mask = new ort.Tensor('int64', new BigInt64Array(totalSeq).fill(1n), [1, totalSeq]);
  for (const spec of META.inputs) {
    if (out[spec.name]) continue;
    if (spec.name === 'position_ids') {
      const pos = new BigInt64Array(plen);
      for (let i = 0; i < plen; i++) pos[i] = BigInt(pastSeq + i);
      out.position_ids = new ort.Tensor('int64', pos, [1, plen]);
      continue;
    }
    const shape = spec.shape.map(d => resolveDim(d, plen, pastSeq, totalSeq));
    out[spec.name] = makeTensor(spec, shape, useTq);
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
  return ort.InferenceSession.create(ONNX, opts);
}

function getLogitsLastTok(outs) {
  const t = outs.logits ?? outs[Object.keys(outs)[0]];
  const last = t.data;  // shape [1, S, vocab] but with last_token slice should be [1,1,V]
  return Array.from(last);
}

function cos(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
  return dot / (Math.sqrt(na)*Math.sqrt(nb) + 1e-12);
}

function argmax(a) { let m = -Infinity, idx = 0; for (let i = 0; i < a.length; i++) if (a[i] > m) { m = a[i]; idx = i; } return idx; }

async function runChain(useTq, driveTokens) {
  const sess = await makeSession(useTq);
  const plen = PROMPT_TOKEN_IDS.length;
  let inputs = makeInputs(plen, 0, useTq);
  let outs = await sess.run(inputs);
  const logits = [getLogitsLastTok(outs)];
  const tokens = [argmax(logits[0])];
  for (let s = 0; s < N_STEPS; s++) {
    const newInputs = {};
    let pastSeq = 0;
    for (const name of META.outputs) {
      if (name.startsWith('present.') && (name.includes('.key') || name.includes('.value'))) {
        newInputs['past_key_values.' + name.slice('present.'.length)] = outs[name];
        pastSeq = outs[name].dims[2];
      } else if (name.startsWith('present_conv.')) {
        newInputs['past_conv.' + name.slice('present_conv.'.length)] = outs[name];
      }
    }
    const driveTok = useTq ? driveTokens[s] : tokens[s];
    const decodeInputs = makeInputs(1, pastSeq, useTq, new ort.Tensor('int64', new BigInt64Array([BigInt(driveTok)]), [1, 1]));
    for (const k in newInputs) decodeInputs[k] = newInputs[k];
    outs = await sess.run(decodeInputs);
    logits.push(getLogitsLastTok(outs));
    tokens.push(argmax(logits[logits.length - 1]));
    inputs = decodeInputs;
  }
  await sess.release();
  return { tokens, logits };
}

console.log('== fp16 self-driven ==');
const fp = await runChain(false, []);
console.log('  tokens:', fp.tokens);
console.log('== TQ driven by fp16 tokens ==');
const tq = await runChain(true, fp.tokens);
console.log('  tokens:', tq.tokens);
console.log('\n== per-step cos sim ==');
let allOk = true;
for (let i = 0; i < fp.logits.length; i++) {
  const c = cos(fp.logits[i], tq.logits[i]);
  const match = fp.tokens[i] === tq.tokens[i] ? '✓' : '✗';
  console.log(`  step${i}: cos=${c.toFixed(5)} fp16=${fp.tokens[i]} tq=${tq.tokens[i]} ${match}`);
  if (c < 0.95) allOk = false;
}
const matches = fp.tokens.filter((t, i) => t === tq.tokens[i]).length;
console.log(`\n  cos sim ≥ 0.95 on all steps: ${allOk}`);
console.log(`  top-1 matches: ${matches}/${fp.tokens.length}`);
