// Tiny bun static file server with CORS + COOP/COEP for SharedArrayBuffer
// (required by ort-web's threaded WASM build).
import { AutoTokenizer, env as hfEnv } from '@huggingface/transformers';
hfEnv.allowRemoteModels = false;
hfEnv.localModelPath = new URL('../', import.meta.url).pathname;

const ROOT = new URL('.', import.meta.url).pathname;
const MODEL_LFM = new URL('../model/onnx', import.meta.url).pathname;
const MODEL_QWN = new URL('../qwen3/onnx', import.meta.url).pathname;
const MODEL_QWN35 = new URL('../qwen35/onnx', import.meta.url).pathname;

const TOK_DIR: Record<string, string> = { lfm: 'model', qwen3: 'qwen3' };
const tokCache: Record<string, any> = {};
async function getTokenizer(model: string) {
  if (tokCache[model]) return tokCache[model];
  const dir = TOK_DIR[model];
  if (!dir) throw new Error('unknown model ' + model);
  tokCache[model] = await AutoTokenizer.from_pretrained(dir);
  return tokCache[model];
}
// ort-web JS lib (pre-built source we'll point at via direct file paths)
const ORT_LIB = new URL('./node_modules/onnxruntime-web', import.meta.url).pathname;
// our WASM artifacts
const WASM_BUILD = new URL('../../external/onnxruntime/build/MacOS/Release', import.meta.url).pathname;

const MIME: Record<string, string> = {
  html: 'text/html; charset=utf-8',
  js: 'application/javascript; charset=utf-8',
  mjs: 'application/javascript; charset=utf-8',
  ts: 'application/javascript; charset=utf-8',
  wasm: 'application/wasm',
  onnx: 'application/octet-stream',
  onnx_data: 'application/octet-stream',
};

const headers = (path: string): HeadersInit => {
  const ext = path.split('.').pop() ?? '';
  const m = MIME[ext];
  return {
    'content-type': m ?? 'application/octet-stream',
    'cross-origin-opener-policy': 'same-origin',
    'cross-origin-embedder-policy': 'require-corp',
    'access-control-allow-origin': '*',
  };
};

const RECORDINGS = new URL('./recordings', import.meta.url).pathname;

Bun.serve({
  port: 8787,
  async fetch(req) {
    console.log(req.method, req.url);
    const u = new URL(req.url);
    let p = decodeURIComponent(u.pathname);

    if (req.method === 'POST' && p === '/tokenize') {
      const { model, text, chat } = await req.json();
      const tok = await getTokenizer(model);
      let ids: any;
      if (chat) {
        const t = tok.apply_chat_template(chat, { add_generation_prompt: true, tokenize: true });
        // returns { input_ids: Tensor[1,N] (BigInt64Array), attention_mask: ... }
        ids = Array.from(t.input_ids.data as BigInt64Array);
      } else {
        ids = tok.encode(text);
      }
      return Response.json({ ids: ids.map((x: any) => Number(x)) }, {
        headers: { 'access-control-allow-origin': '*' },
      });
    }

    if (req.method === 'POST' && p === '/detokenize') {
      const { model, ids } = await req.json();
      const tok = await getTokenizer(model);
      const pieces = ids.map((id: number) => tok.decode([id], { skip_special_tokens: false }));
      return Response.json({ pieces }, {
        headers: { 'access-control-allow-origin': '*' },
      });
    }

    if (req.method === 'POST' && p === '/save') {
      const name = u.searchParams.get('name');
      if (!name || !/^[\w.\-]+$/.test(name)) {
        return new Response('bad name', { status: 400 });
      }
      await Bun.write(`${RECORDINGS}/${name}`, await req.text());
      return new Response('ok', { headers: { 'access-control-allow-origin': '*' } });
    }

    if (p === '/') p = '/index.html';
    let file: string | null = null;
    if (p.startsWith('/models/lfm/')) file = MODEL_LFM + p.slice('/models/lfm'.length);
    else if (p.startsWith('/models/qwen35/')) file = MODEL_QWN35 + p.slice('/models/qwen35'.length);
    else if (p.startsWith('/models/qwen3/')) file = MODEL_QWN + p.slice('/models/qwen3'.length);
    else if (p.startsWith('/wasm/')) file = WASM_BUILD + p.slice('/wasm'.length);
    else if (p.startsWith('/ort/')) file = ORT_LIB + p.slice('/ort'.length);
    else file = ROOT + p;
    try {
      const f = Bun.file(file);
      if (!(await f.exists())) return new Response('not found: ' + p, { status: 404 });
      return new Response(f, { headers: headers(file) });
    } catch (e) {
      return new Response('error: ' + e, { status: 500 });
    }
  },
});
console.log('http://localhost:8787/');
