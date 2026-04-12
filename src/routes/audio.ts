import type http from "node:http";
import type { LoadedModels } from "../models/manager.js";
import { errorJson, json, readBody } from "../utils/http.js";
import { parseMultipart } from "../utils/multipart.js";
import type { MultipartFile } from "../utils/multipart.js";

export async function handleAudioTranscriptions(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  models: LoadedModels,
): Promise<void> {
  if (!models.transcriber) {
    errorJson(res, 503, "STT model not loaded", "server_error");
    return;
  }

  try {
    const body = await readBody(req);
    const contentType = req.headers["content-type"] || "";

    let audioBuffer: Buffer | undefined;
    if (contentType.includes("multipart/form-data")) {
      const boundary = contentType.split("boundary=")[1];
      if (!boundary) {
        errorJson(res, 400, "Missing boundary in multipart content-type");
        return;
      }
      const parts = parseMultipart(body, boundary);
      const filePart = parts.file;
      if (filePart && typeof filePart !== "string") {
        audioBuffer = (filePart as MultipartFile).data;
      }
    } else {
      audioBuffer = body;
    }

    if (!audioBuffer?.length) {
      errorJson(res, 400, "No audio data provided");
      return;
    }

    const aligned = new ArrayBuffer(audioBuffer.byteLength);
    new Uint8Array(aligned).set(audioBuffer);
    const float32 = new Float32Array(aligned);

    const result = await models.transcriber(float32);
    json(res, 200, { text: result.text.trim() });
  } catch (e) {
    console.error("[wandler] STT error:", e);
    errorJson(res, 500, (e as Error).message, "server_error");
  }
}
