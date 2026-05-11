// Real-generation recorder for the Qwen3.5-4B-ONNX-OPT comparison video.
// Differs from record-video.mjs:
//   - Two ONNX sessions: embed_tokens + decoder_model_merged
//   - decoder takes inputs_embeds (not input_ids)
//   - No lasttok variant: pass num_logits_to_keep=1 instead
//   - 32 layers, every 4th is full_attention (8 KV layers, 24 linear layers)
//
// Usage:
//   node record-video-4b.mjs <doc> <maxNew> [tag] [TEMP] [TOP_P] [TOP_K] [REP_PEN] [DOC_BYTES]
import * as ort from 'onnxruntime-node';
import { AutoTokenizer, env as hfEnv } from '@huggingface/transformers';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

function detectChip() {
  try {
    if (process.platform === 'darwin') {
      const out = execSync('sysctl -n machdep.cpu.brand_string', { encoding: 'utf8' }).trim();
      if (out) return out;
    }
  } catch (_) { /* ignore */ }
  return os.cpus()[0]?.model ?? 'unknown';
}

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

hfEnv.allowRemoteModels = false;
hfEnv.localModelPath = ROOT + '/';

const cfg = {
  dir: 'qwen35-4b',
  label: 'Qwen3.5-4B-ONNX-OPT',
  num_layers: 32,
  num_kv_heads: 4,
  head_dim: 256,
  conv_dim: 8192, // Q+K+V projections: 2048+2048+4096 (4B has 16 query heads × 128)
  conv_kernel_minus_1: 3,
  recurrent_shape: [1, 32, 128, 128], // [batch, linear_num_value_heads, key_head_dim, value_head_dim]
  slot_bytes_q4: 132,
  eos: [248044],
};

// layer_types from config: every 4th (3,7,11,...) is full_attention
const LAYER_FULL = new Set();
for (let i = 3; i < cfg.num_layers; i += 4) LAYER_FULL.add(i);

const DEFAULT_DOC = path.resolve(ROOT, '..', 'TURBOQUANT-KV-CACHE-PLAN.md');
const RECORDINGS = path.resolve(ROOT, 'browser', 'recordings');

const [, , docPath = DEFAULT_DOC, maxNewStr = '200',
  tag = '', tempStr = '0.6', topPStr = '0.95', topKStr = '20',
  repPenStr = '1.05', docBytesStr = '0'] = process.argv;
const maxNew = parseInt(maxNewStr, 10);
const TEMP = parseFloat(tempStr);
const TOP_P = parseFloat(topPStr);
const TOP_K = parseInt(topKStr, 10);
const REP_PEN = parseFloat(repPenStr);
const DOC_BYTES = parseInt(docBytesStr, 10);
const NO_EOS = process.env.NO_EOS === '1';

const embedPath = path.join(ROOT, cfg.dir, 'onnx', 'embed_tokens_q4f16.onnx');
const decoderPath = path.join(ROOT, cfg.dir, 'onnx', 'decoder_model_merged_q4f16.onnx');

console.log(`# ${cfg.label}`);
console.log(`# doc: ${docPath}`);
console.log(`# sampling: T=${TEMP} top_p=${TOP_P} top_k=${TOP_K} rep_pen=${REP_PEN} no_eos=${NO_EOS}`);

let docText = fs.readFileSync(docPath, 'utf8');
if (DOC_BYTES > 0 && docText.length > DOC_BYTES) docText = docText.slice(0, DOC_BYTES);
console.log(`# doc bytes: ${docText.length}`);

const userMessage =
  'Below is a technical document. Read it carefully, then summarize it in 200–300 words for an engineering audience.\n\n' +
  '---\n' + docText + '\n---\n\nSummary:';

console.log('# loading tokenizer…');
const tok = await AutoTokenizer.from_pretrained(cfg.dir);
const tApplied = tok.apply_chat_template(
  [{ role: 'user', content: userMessage }],
  { add_generation_prompt: true, tokenize: true },
);
const promptIds = Array.from(tApplied.input_ids.data, (x) => Number(x));
console.log(`# prompt tokens: ${promptIds.length}`);

// ----- math helpers (sampling) -----
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
  if (REP_PEN !== 1.0 && generated.length > 0) {
    const seen = new Set(generated);
    for (const id of seen) {
      const v = logits[id];
      logits[id] = v > 0 ? v / REP_PEN : v * REP_PEN;
    }
  }
  if (TEMP <= 0) {
    let best = -Infinity, bestIdx = 0;
    for (let i = 0; i < vocab; i++) if (logits[i] > best) { best = logits[i]; bestIdx = i; }
    return bestIdx;
  }
  for (let i = 0; i < vocab; i++) logits[i] /= TEMP;
  let kept;
  if (TOP_K > 0 && TOP_K < vocab) {
    const idx = Array.from({ length: vocab }, (_, i) => i);
    idx.sort((a, b) => logits[b] - logits[a]);
    kept = idx.slice(0, TOP_K);
  } else {
    kept = Array.from({ length: vocab }, (_, i) => i);
  }
  let maxL = -Infinity;
  for (const i of kept) if (logits[i] > maxL) maxL = logits[i];
  const exps = new Float64Array(kept.length);
  let sum = 0;
  for (let j = 0; j < kept.length; j++) { exps[j] = Math.exp(logits[kept[j]] - maxL); sum += exps[j]; }
  for (let j = 0; j < kept.length; j++) exps[j] /= sum;
  const order = Array.from(exps.keys()).sort((a, b) => exps[b] - exps[a]);
  let cum = 0;
  const finalIdxs = [], finalProbs = [];
  for (const j of order) {
    finalIdxs.push(kept[j]);
    finalProbs.push(exps[j]);
    cum += exps[j];
    if (cum >= TOP_P) break;
  }
  let s = 0; for (const p of finalProbs) s += p;
  const r = rng() * s;
  let acc = 0;
  for (let j = 0; j < finalProbs.length; j++) {
    acc += finalProbs[j];
    if (r <= acc) return finalIdxs[j];
  }
  return finalIdxs[finalIdxs.length - 1];
}

// ----- session helpers -----
async function makeSession(modelPath, useTq) {
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
  return await ort.InferenceSession.create(modelPath, opts);
}

console.log('# loading embed_tokens session…');
const embedSess = await makeSession(embedPath, false);

async function embedTokens(ids) {
  const arr = new BigInt64Array(ids.length);
  for (let i = 0; i < ids.length; i++) arr[i] = BigInt(ids[i]);
  const inName = embedSess.inputNames[0];
  const outName = embedSess.outputNames[0];
  const out = await embedSess.run({
    [inName]: new ort.Tensor('int64', arr, [1, ids.length]),
  });
  return out[outName];
}

// Build a fresh empty state for all 32 layers (used at step 0 / first prompt pass).
function emptyState(useTq, pastSeq) {
  const out = {};
  for (let layer = 0; layer < cfg.num_layers; layer++) {
    if (LAYER_FULL.has(layer)) {
      const lastDim = useTq ? cfg.slot_bytes_q4 : cfg.head_dim;
      const dtype = useTq ? 'uint8' : 'float16';
      const shape = [1, cfg.num_kv_heads, pastSeq, lastDim];
      const size = shape.reduce((a, b) => a * b, 1);
      const make = () => dtype === 'uint8' ? new Uint8Array(size) : new Uint16Array(size);
      out[`past_key_values.${layer}.key`] = new ort.Tensor(dtype, make(), shape);
      out[`past_key_values.${layer}.value`] = new ort.Tensor(dtype, make(), shape);
    } else {
      const convShape = [1, cfg.conv_dim, cfg.conv_kernel_minus_1];
      const convSize = convShape.reduce((a, b) => a * b, 1);
      out[`past_conv.${layer}`] = new ort.Tensor('float16', new Uint16Array(convSize), convShape);
      const recShape = cfg.recurrent_shape;
      const recSize = recShape.reduce((a, b) => a * b, 1);
      out[`past_recurrent.${layer}`] = new ort.Tensor('float16', new Uint16Array(recSize), recShape);
    }
  }
  return out;
}

function pickPresent(outs) {
  const fed = {};
  let pastSeq = 0;
  for (const name of Object.keys(outs)) {
    if (name.startsWith('present.') && (name.endsWith('.key') || name.endsWith('.value'))) {
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

function totalKvBytes(state) {
  let kv = 0;
  for (const [name, t] of Object.entries(state)) {
    if (name.startsWith('past_key_values.')) {
      const elem = t.type === 'uint8' ? 1 : t.type === 'float16' ? 2 : 4;
      kv += t.dims.reduce((a, b) => a * b, 1) * elem;
    }
  }
  return kv;
}

async function decoderRun(decoderSess, embeds, attnLen, posStart, plen, pastState) {
  // mRoPE: position_ids shape is [3, batch, seq_len]. For text-only generation
  // all three temporal/height/width sections share the same sequence positions.
  const oneSection = BigInt64Array.from({ length: plen }, (_, i) => BigInt(posStart + i));
  const stacked = new BigInt64Array(3 * plen);
  stacked.set(oneSection, 0);
  stacked.set(oneSection, plen);
  stacked.set(oneSection, 2 * plen);
  const inputs = {
    inputs_embeds: embeds,
    attention_mask: new ort.Tensor('int64', new BigInt64Array(attnLen).fill(1n), [1, attnLen]),
    position_ids: new ort.Tensor('int64', stacked, [3, 1, plen]),
    num_logits_to_keep: new ort.Tensor('int64', new BigInt64Array([1n]), []),
    ...pastState,
  };
  return await decoderSess.run(inputs);
}

// ----- main generate -----
async function generate(useTq) {
  const variant = useTq ? 'tq' : 'fp16';
  console.log(`\n== ${variant} ==`);
  const sess = await makeSession(decoderPath, useTq);

  // Prompt step
  console.log('  embedding prompt…');
  const promptEmbeds = await embedTokens(promptIds);

  let past = { fed: emptyState(useTq, 0), pastSeq: 0 };
  const tPrompt0 = performance.now();
  let outs = await decoderRun(sess, promptEmbeds, promptIds.length, 0, promptIds.length, past.fed);
  const promptMs = performance.now() - tPrompt0;
  console.log(`  prompt (TTFT): ${promptMs.toFixed(0)} ms`);
  past = pickPresent(outs);
  const kvBytes = totalKvBytes(past.fed);
  console.log(`  KV cache after prompt: ${(kvBytes / 1024 / 1024).toFixed(1)} MB`);

  let nextId = sampleNext(readLastLogits(outs.logits), makeRng(47), []);
  const rng = makeRng(47);
  // Re-seed and skip first call so subsequent calls match seeded sequence.
  // (Not strictly needed; same seed used identically in both fp16 and tq branches.)

  let totalCtx = promptIds.length;
  const eosSet = new Set(NO_EOS ? [] : cfg.eos);
  const events = [];
  const generatedIds = [];
  let tCum = 0;
  const tDecode0 = performance.now();
  for (let s = 0; s < maxNew; s++) {
    if (eosSet.has(nextId)) { console.log(`  eos at step ${s}`); break; }
    generatedIds.push(nextId);
    const stepEmbeds = await embedTokens([nextId]);
    const t0 = performance.now();
    outs = await decoderRun(sess, stepEmbeds, totalCtx + 1, totalCtx, 1, past.fed);
    const dt = performance.now() - t0;
    tCum += dt;
    past = pickPresent(outs);
    totalCtx += 1;
    events.push({ step: s, t_ms: +tCum.toFixed(2), dt_ms: +dt.toFixed(2), token_id: nextId });
    nextId = sampleNext(readLastLogits(outs.logits), rng, generatedIds);
    if ((s + 1) % 25 === 0 || s === maxNew - 1) {
      const wall = (performance.now() - tDecode0) / 1000;
      console.log(`  step ${s + 1}/${maxNew}: ${(s + 1).toFixed(0)} tok, ${((s + 1) / wall).toFixed(2)} tok/s`);
    }
  }
  await sess.release();

  const pieces = generatedIds.map((id) => tok.decode([id], { skip_special_tokens: false }));
  for (let i = 0; i < events.length; i++) events[i].token = pieces[i] ?? '';
  const finalText = pieces.join('');
  return { variant, promptMs, totalDecodeMs: tCum, events, finalText, kvBytes };
}

fs.mkdirSync(RECORDINGS, { recursive: true });
const stamp = `qwen35-4b-real-n${maxNew}${tag ? '-' + tag : ''}`;

const fp16 = await generate(false);
const tq = await generate(true);

for (const r of [fp16, tq]) {
  const fname = `real-${r.variant}-${stamp}.jsonl`;
  fs.writeFileSync(path.join(RECORDINGS, fname), r.events.map((e) => JSON.stringify(e)).join('\n') + '\n');
  console.log(`wrote ${fname}`);
}
const speedup = fp16.totalDecodeMs / tq.totalDecodeMs;
const metaOut = {
  model: cfg.label,
  hardware: detectChip(),
  runtime: 'ONNX Runtime · WebGPU · Node.js',
  doc: docPath,
  prompt_tokens: promptIds.length,
  max_new_tokens: maxNew,
  fp16: { promptMs: fp16.promptMs, totalDecodeMs: fp16.totalDecodeMs, n: fp16.events.length, kvBytes: fp16.kvBytes, text: fp16.finalText },
  tq: { promptMs: tq.promptMs, totalDecodeMs: tq.totalDecodeMs, n: tq.events.length, kvBytes: tq.kvBytes, text: tq.finalText },
  decode_speedup: +speedup.toFixed(2),
  recorded_at: new Date().toISOString(),
};
fs.writeFileSync(path.join(RECORDINGS, `real-meta-${stamp}.json`), JSON.stringify(metaOut, null, 2));
console.log(`\n# decode speedup: ${speedup.toFixed(2)}x`);
console.log(`# wrote real-meta-${stamp}.json`);
