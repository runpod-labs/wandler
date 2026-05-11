// npm pack strips symlinks. After install, recreate them so
// onnxruntime-node's binding can resolve @rpath/libonnxruntime.1.dylib.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const base = path.join(here, '..', 'node_modules', 'onnxruntime-node', 'bin', 'napi-v6');

// transformers.js bundles its own pinned `onnxruntime-node@1.24.x` under
// node_modules/@huggingface/transformers/node_modules/. That nested copy
// wins over the workspace-root patched version we just installed. Wipe it
// so module resolution falls back to the hoisted patched build.
const trxNested = path.join(here, '..', '..', 'node_modules', '@huggingface', 'transformers', 'node_modules', 'onnxruntime-node');
const trxCommonNested = path.join(here, '..', '..', 'node_modules', '@huggingface', 'transformers', 'node_modules', 'onnxruntime-common');
for (const target of [trxNested, trxCommonNested]) {
  if (fs.existsSync(target)) {
    fs.rmSync(target, { recursive: true, force: true });
    console.log('[wandler] removed nested', target);
  }
}

if (!fs.existsSync(base)) { process.exit(0); }
const platform = process.platform === 'darwin' ? 'darwin' : process.platform === 'linux' ? 'linux' : 'win32';
const arch = process.arch;
const dir = path.join(base, platform, arch);
if (!fs.existsSync(dir)) { process.exit(0); }
const ext = platform === 'darwin' ? 'dylib' : platform === 'linux' ? 'so' : 'dll';

// Find versioned dylib (e.g. libonnxruntime.1.27.0.dylib) and recreate the
// short symlinks the binding's @rpath lookup needs.
const files = fs.readdirSync(dir);
const versioned = files.find(f => /^libonnxruntime\.\d+\.\d+\.\d+\.(dylib|so|dll)$/.test(f));
if (!versioned) { process.exit(0); }
const major = versioned.split('.')[1];
const short = `libonnxruntime.${major}.${ext}`;
const plain = `libonnxruntime.${ext}`;
for (const [src, target] of [[short, versioned], [plain, short]]) {
  const targetPath = path.join(dir, src);
  try { fs.unlinkSync(targetPath); } catch {}
  fs.symlinkSync(target, targetPath);
  console.log('[wandler] symlink', src, '->', target);
}
