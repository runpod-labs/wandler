// Real-generation recorder for the TurboQuant comparison video.
// Runs sampled generation twice (fp16, then TQ) on the same prompt with
// per-token wallclock timings, then writes JSONL + meta.json into
// ../browser/recordings/ for downstream Remotion rendering.
//
// Usage:
//   node record-video.mjs <model> <doc> <maxNew> [tag] [TEMP] [TOP_P] [TOP_K] [REP_PEN] [DOC_BYTES]
//   - model: lfm | qwen3 | qwen35
//   - doc: path to source document (.md/.txt)
//   - maxNew: max tokens to generate
//   - tag: filename suffix (e.g. "run3")
//   - TEMP/TOP_P/TOP_K/REP_PEN: sampling (defaults below)
//   - DOC_BYTES: truncate doc to first N bytes (0 = no truncation)
// Same generation seed (47) is reused across fp16 and TQ so token streams match.
import * as ort from 'onnxruntime-node';
import { AutoTokenizer, env as hfEnv } from '@huggingface/transformers';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

hfEnv.allowRemoteModels = false;
hfEnv.localModelPath = ROOT + '/';

const MODELS = {
  lfm: { dir: 'model', label: 'LFM2.5-1.2B-Instruct', slot_bytes: 36, eos: [7] },
  qwen3: { dir: 'qwen3', label: 'Qwen3-0.6B', slot_bytes: 68, eos: [151645, 151643] },
  qwen35: { dir: 'qwen35', label: 'Qwen3.5-0.8B-Text', slot_bytes: 132, eos: [248044] },
};

const DEFAULT_DOC = path.resolve(ROOT, '..', 'TURBOQUANT-KV-CACHE-PLAN.md');
const RECORDINGS = path.resolve(ROOT, 'browser', 'recordings');

const [, , modelKey = 'qwen35', docPath = DEFAULT_DOC, maxNewStr = '300',
  tag = '', tempStr = '0.6', topPStr = '0.95', topKStr = '20',
  repPenStr = '1.15', docBytesStr = '0'] = process.argv;
const cfg = MODELS[modelKey];
if (!cfg) { console.error('unknown model:', modelKey); process.exit(1); }
const maxNew = parseInt(maxNewStr, 10);
const TEMP = parseFloat(tempStr);
const TOP_P = parseFloat(topPStr);
const TOP_K = parseInt(topKStr, 10);
const REP_PEN = parseFloat(repPenStr);
const DOC_BYTES = parseInt(docBytesStr, 10);

const onnxPath = path.join(ROOT, cfg.dir === 'model' ? 'model' : cfg.dir, 'onnx', 'model_q4f16_lasttok.onnx');
const metaPath = path.join(ROOT, cfg.dir === 'model' ? 'model' : cfg.dir, 'onnx', 'model_q4f16_lasttok.meta.json');
const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));

console.log(`# ${cfg.label}`);
console.log(`# doc: ${docPath}`);

let docText = fs.readFileSync(docPath, 'utf8');
if (DOC_BYTES > 0 && docText.length > DOC_BYTES) docText = docText.slice(0, DOC_BYTES);
console.log(`# doc bytes: ${docText.length}`);
console.log(`# sampling: T=${TEMP} top_p=${TOP_P} top_k=${TOP_K} rep_pen=${REP_PEN}`);
const userMessage =
  'Below is a technical document. Read it carefully, then summarize it in 200–300 words for an engineering audience.\n\n' +
  '---\n' + docText + '\n---\n\nSummary:';

console.log('# loading tokenizer…');
const tok = await AutoTokenizer.from_pretrained(cfg.dir);
const t = tok.apply_chat_template(
  [{ role: 'user', content: userMessage }],
  { add_generation_prompt: true, tokenize: true },
);
const promptIds = Array.from(t.input_ids.data, (x) => Number(x));
console.log(`# prompt tokens: ${promptIds.length}`);

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
  let s = [...shape];
  if (useTq && spec.name.startsWith('past_key_values') && shape.length === 4) {
    s[s.length - 1] = cfg.slot_bytes;
    dtype = 'uint8';
  }
  const size = s.reduce((a, b) => a * b, 1);
  let data;
  if (dtype === 'uint8') data = new Uint8Array(size);
  else if (dtype === 'int64') data = new BigInt64Array(size);
  else if (dtype === 'float16') data = new Uint16Array(size);
  else if (dtype === 'float32') data = new Float32Array(size);
  else throw new Error('unsupported type: ' + dtype);
  return new ort.Tensor(dtype, data, s);
}

function buildInputs(plen, pastSeq, useTq, drivenIds, posStart) {
  const totalSeq = pastSeq + plen;
  const out = {};
  out.input_ids = drivenIds;
  out.attention_mask = new ort.Tensor('int64', new BigInt64Array(totalSeq).fill(1n), [1, totalSeq]);
  for (const spec of meta.inputs) {
    if (out[spec.name]) continue;
    if (spec.name === 'position_ids') {
      const pos = new BigInt64Array(plen);
      for (let i = 0; i < plen; i++) pos[i] = BigInt(posStart + i);
      out[spec.name] = new ort.Tensor('int64', pos, [1, plen]);
      continue;
    }
    const shape = spec.shape.map((d) => resolveDim(d, plen, pastSeq, totalSeq));
    out[spec.name] = makeTensor(spec, shape, useTq);
  }
  return out;
}

function pickPresent(outs) {
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

function f16ToF32(h) {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

function readLastLogits(logitsTensor) {
  const dims = logitsTensor.dims;
  const vocab = dims[dims.length - 1];
  const seq = dims[dims.length - 2];
  const data = logitsTensor.data;
  const offset = (seq - 1) * vocab;
  const out = new Float32Array(vocab);
  if (data instanceof Uint16Array) {
    for (let i = 0; i < vocab; i++) out[i] = f16ToF32(data[offset + i]);
  } else {
    for (let i = 0; i < vocab; i++) out[i] = data[offset + i];
  }
  return out;
}

// Mulberry32 PRNG — same seed reproduces fp16 and TQ token streams identically
// when the same logits arrive (different logits → different streams, naturally).
function makeRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sampleNext(logits, rng, generated) {
  const vocab = logits.length;
  // Repetition penalty: down-weight previously-generated tokens
  if (REP_PEN !== 1.0 && generated.length > 0) {
    const seen = new Set(generated);
    for (const id of seen) {
      const v = logits[id];
      logits[id] = v > 0 ? v / REP_PEN : v * REP_PEN;
    }
  }
  // Temperature
  if (TEMP > 0) {
    for (let i = 0; i < vocab; i++) logits[i] /= TEMP;
  } else {
    // T=0 → argmax shortcut
    let best = -Infinity, bestIdx = 0;
    for (let i = 0; i < vocab; i++) if (logits[i] > best) { best = logits[i]; bestIdx = i; }
    return bestIdx;
  }
  // Top-K: keep K largest, mask the rest
  let kept;
  if (TOP_K > 0 && TOP_K < vocab) {
    const idx = new Int32Array(vocab);
    for (let i = 0; i < vocab; i++) idx[i] = i;
    // partial sort by logit desc — full sort is fine for small vocab
    const arr = Array.from(idx).sort((a, b) => logits[b] - logits[a]);
    kept = arr.slice(0, TOP_K);
  } else {
    kept = [];
    for (let i = 0; i < vocab; i++) kept.push(i);
  }
  // Softmax over kept
  let maxL = -Infinity;
  for (const i of kept) if (logits[i] > maxL) maxL = logits[i];
  const exps = new Float64Array(kept.length);
  let sum = 0;
  for (let j = 0; j < kept.length; j++) { exps[j] = Math.exp(logits[kept[j]] - maxL); sum += exps[j]; }
  for (let j = 0; j < kept.length; j++) exps[j] /= sum;
  // Top-P: sort kept by prob desc, keep until cumulative >= TOP_P
  const order = Array.from(exps.keys()).sort((a, b) => exps[b] - exps[a]);
  let cum = 0;
  const finalIdxs = [];
  const finalProbs = [];
  for (const j of order) {
    finalIdxs.push(kept[j]);
    finalProbs.push(exps[j]);
    cum += exps[j];
    if (cum >= TOP_P) break;
  }
  // Renormalize and sample
  let s = 0; for (const p of finalProbs) s += p;
  const r = rng() * s;
  let acc = 0;
  for (let j = 0; j < finalProbs.length; j++) {
    acc += finalProbs[j];
    if (r <= acc) return finalIdxs[j];
  }
  return finalIdxs[finalIdxs.length - 1];
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
      'optimization.turboquant_kv_boundary': process.env.TQ_BOUNDARY ?? '0',
    };
  }
  return await ort.InferenceSession.create(onnxPath, opts);
}

async function generate(useTq) {
  const variant = useTq ? 'tq' : 'fp16';
  console.log(`\n== ${variant} ==`);
  const sess = await makeSession(useTq);

  // Prompt forward pass
  const promptArr = new BigInt64Array(promptIds.length);
  for (let i = 0; i < promptIds.length; i++) promptArr[i] = BigInt(promptIds[i]);
  const promptTensor = new ort.Tensor('int64', promptArr, [1, promptIds.length]);
  const promptInputs = buildInputs(promptIds.length, 0, useTq, promptTensor, 0);
  const tPrompt0 = performance.now();
  let outs = await sess.run(promptInputs);
  const promptMs = performance.now() - tPrompt0;
  console.log(`  prompt (TTFT): ${promptMs.toFixed(0)} ms`);

  let kvBytes = 0;
  for (const name of meta.outputs) {
    if (name.startsWith('present.') && (name.includes('.key') || name.includes('.value'))) {
      const t = outs[name];
      const elemBytes = t.type === 'uint8' ? 1 : t.type === 'float16' ? 2 : 4;
      kvBytes += t.dims.reduce((a, b) => a * b, 1) * elemBytes;
    }
  }
  console.log(`  KV cache after prompt: ${(kvBytes / 1024 / 1024).toFixed(1)} MB`);

  let past = pickPresent(outs);
  const rng = makeRng(47);
  let nextId = sampleNext(readLastLogits(outs.logits), rng, []);
  let totalCtx = promptIds.length;
  const eosSet = new Set(process.env.NO_EOS === '1' ? [] : cfg.eos);
  const events = [];
  const generatedIds = [];
  let tCum = 0;
  const tDecode0 = performance.now();
  for (let s = 0; s < maxNew; s++) {
    if (eosSet.has(nextId)) { console.log(`  eos at step ${s}`); break; }
    generatedIds.push(nextId);
    const drive = new ort.Tensor('int64', new BigInt64Array([BigInt(nextId)]), [1, 1]);
    const stepInputs = buildInputs(1, totalCtx, useTq, drive, totalCtx);
    for (const k in past.fed) stepInputs[k] = past.fed[k];
    const ts = performance.now();
    outs = await sess.run(stepInputs);
    const dt = performance.now() - ts;
    tCum += dt;
    past = pickPresent(outs);
    totalCtx += 1;
    events.push({ step: s, t_ms: +tCum.toFixed(2), dt_ms: +dt.toFixed(2), token_id: nextId });
    nextId = sampleNext(readLastLogits(outs.logits), rng, generatedIds);
    if ((s + 1) % 25 === 0 || s === maxNew - 1) {
      const wall = (performance.now() - tDecode0) / 1000;
      console.log(`  step ${s + 1}/${maxNew}: ${(s + 1 / wall).toFixed(1)} tok, ${((s + 1) / wall).toFixed(1)} tok/s`);
    }
  }
  await sess.release();

  // Detokenize each id individually so each event has its own piece of text.
  const pieces = generatedIds.map((id) => tok.decode([id], { skip_special_tokens: false }));
  for (let i = 0; i < events.length; i++) events[i].token = pieces[i] ?? '';
  const finalText = pieces.join('');
  return { variant, promptMs, totalDecodeMs: tCum, events, finalText, kvBytes };
}

fs.mkdirSync(RECORDINGS, { recursive: true });
const stamp = `${modelKey}-real-n${maxNew}${tag ? '-' + tag : ''}`;

const fp16 = await generate(false);
const tq = await generate(true);

for (const r of [fp16, tq]) {
  const fname = `real-${r.variant}-${stamp}.jsonl`;
  const body = r.events.map((e) => JSON.stringify(e)).join('\n') + '\n';
  fs.writeFileSync(path.join(RECORDINGS, fname), body);
  console.log(`wrote ${fname}`);
}

const speedup = fp16.totalDecodeMs / tq.totalDecodeMs;
const metaOut = {
  model: cfg.label,
  doc: docPath,
  prompt_tokens: promptIds.length,
  max_new_tokens: maxNew,
  fp16: { promptMs: fp16.promptMs, totalDecodeMs: fp16.totalDecodeMs, n: fp16.events.length, kvBytes: fp16.kvBytes, text: fp16.finalText },
  tq:   { promptMs: tq.promptMs,   totalDecodeMs: tq.totalDecodeMs,   n: tq.events.length,   kvBytes: tq.kvBytes,   text: tq.finalText },
  decode_speedup: +speedup.toFixed(2),
  recorded_at: new Date().toISOString(),
};
fs.writeFileSync(path.join(RECORDINGS, `real-meta-${stamp}.json`), JSON.stringify(metaOut, null, 2));
console.log(`\n# decode speedup: ${speedup.toFixed(2)}x`);
console.log(`# wrote real-meta-${stamp}.json`);
