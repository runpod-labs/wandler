#!/usr/bin/env tsx
import { execFileSync } from "node:child_process";
import { existsSync, readdirSync } from "node:fs";
import { AutoModelForCausalLM } from "@huggingface/transformers";

function run(command: string, args: string[]): { ok: boolean; output: string } {
  try {
    return {
      ok: true,
      output: execFileSync(command, args, {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "pipe"],
      }),
    };
  } catch (error) {
    const e = error as { stdout?: string; stderr?: string; message?: string };
    return {
      ok: false,
      output: `${e.stdout ?? ""}${e.stderr ?? ""}${e.message ?? ""}`,
    };
  }
}

function listDir(path: string): string[] {
  if (!existsSync(path)) return [];
  return readdirSync(path);
}

function requireCheck(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(message);
  }
}

async function main(): Promise<void> {
  const modelId = process.env.WANDLER_VERIFY_WEBGPU_MODEL ?? "onnx-community/gemma-4-E4B-it-ONNX";
  const dtype = process.env.WANDLER_VERIFY_WEBGPU_DTYPE ?? "q4";

  console.log("[webgpu] node", process.version);
  console.log("[webgpu] NVIDIA_DRIVER_CAPABILITIES", process.env.NVIDIA_DRIVER_CAPABILITIES ?? "<unset>");
  console.log("[webgpu] NVIDIA_VISIBLE_DEVICES", process.env.NVIDIA_VISIBLE_DEVICES ?? "<unset>");
  console.log("[webgpu] navigator.gpu", globalThis.navigator?.gpu ? "present" : "missing");
  console.log("[webgpu] /dev/nvidia*", listDir("/dev").filter((name) => name.startsWith("nvidia")).join(",") || "<none>");
  console.log("[webgpu] /dev/dri", listDir("/dev/dri").join(",") || "<none>");

  const vulkan = run("vulkaninfo", ["--summary"]);
  requireCheck(vulkan.ok, `vulkaninfo failed:\n${vulkan.output}`);
  console.log(vulkan.output);
  requireCheck(
    /NVIDIA|RTX|GeForce/i.test(vulkan.output) && !/deviceName\s*=\s*llvmpipe/i.test(vulkan.output),
    "Vulkan did not expose a physical NVIDIA adapter. WebGPU would be testing CPU llvmpipe or fail.",
  );

  console.log(`[webgpu] loading ${modelId} (${dtype}) on device=webgpu`);
  const model = await AutoModelForCausalLM.from_pretrained(modelId, {
    dtype,
    device: "webgpu",
  });
  await model.dispose?.();
  console.log("[webgpu] ok");
}

main().catch((error: unknown) => {
  console.error("[webgpu] failed", error instanceof Error ? error.stack : error);
  process.exit(1);
});
