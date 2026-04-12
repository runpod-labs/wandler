import type { Context } from "hono";
import type { AppEnv } from "../server.js";
import { parseMultipart } from "../utils/multipart.js";
import type { MultipartFile } from "../utils/multipart.js";

export async function audioTranscriptions(c: Context<AppEnv>) {
  const models = c.get("models");

  if (!models.transcriber) {
    return c.json(
      { error: { message: "STT model not loaded", type: "server_error", param: null, code: null } },
      503,
    );
  }

  const contentType = c.req.header("content-type") || "";
  const body = Buffer.from(await c.req.arrayBuffer());

  let audioBuffer: Buffer | undefined;
  if (contentType.includes("multipart/form-data")) {
    const boundary = contentType.split("boundary=")[1];
    if (!boundary) {
      return c.json(
        { error: { message: "Missing boundary in multipart content-type", type: "invalid_request_error", param: null, code: null } },
        400,
      );
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
    return c.json(
      { error: { message: "No audio data provided", type: "invalid_request_error", param: null, code: null } },
      400,
    );
  }

  const aligned = new ArrayBuffer(audioBuffer.byteLength);
  new Uint8Array(aligned).set(audioBuffer);
  const float32 = new Float32Array(aligned);

  const result = await models.transcriber(float32);
  return c.json({ text: result.text.trim() });
}
