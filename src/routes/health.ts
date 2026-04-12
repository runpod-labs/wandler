import type http from "node:http";
import type { ServerConfig } from "../config.js";
import type { HealthResponse } from "../types/openai.js";
import { json } from "../utils/http.js";

export function handleHealth(
  _req: http.IncomingMessage,
  res: http.ServerResponse,
  config: ServerConfig,
): void {
  const response: HealthResponse = {
    status: "ok",
    engine: "transformers.js",
    device: config.device,
    models: {
      llm: config.modelId,
      ...(config.sttModelId ? { stt: config.sttModelId } : {}),
    },
  };
  json(res, 200, response);
}
