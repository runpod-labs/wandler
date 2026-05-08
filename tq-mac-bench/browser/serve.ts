// Tiny bun static file server with CORS + COOP/COEP for SharedArrayBuffer
// (required by ort-web's threaded WASM build).
const ROOT = new URL('.', import.meta.url).pathname;
const MODEL_LFM = new URL('../model/onnx', import.meta.url).pathname;
const MODEL_QWN = new URL('../qwen3/onnx', import.meta.url).pathname;
// ort-web JS lib (pre-built source we'll point at via direct file paths)
const ORT_LIB = new URL('../../external/onnxruntime/js/web/lib', import.meta.url).pathname;
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

Bun.serve({
  port: 8787,
  async fetch(req) {
    const u = new URL(req.url);
    let p = decodeURIComponent(u.pathname);
    if (p === '/') p = '/index.html';
    let file: string | null = null;
    if (p.startsWith('/models/lfm/')) file = MODEL_LFM + p.slice('/models/lfm'.length);
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
