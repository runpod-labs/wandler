import * as ort from 'onnxruntime-node';
const path = '/Users/timpietrusky/.t3/worktrees/wandler/t3code-eb53f7f5/tq-mac-bench/qwen35/onnx/model_q4f16_lasttok.onnx';
console.log('available providers:', ort.listSupportedBackends?.() || 'N/A');
try {
  const opts = { executionProviders: ['webgpu', 'cpu'], logSeverityLevel: 0 };
  const sess = await ort.InferenceSession.create(path, opts);
  console.log('OK, inputs:', sess.inputNames.slice(0, 5));
  await sess.release();
} catch (e) {
  console.error('FAIL:', e.message);
}
