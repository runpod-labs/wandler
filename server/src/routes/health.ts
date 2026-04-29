import type { Context } from "hono";
import type { AppEnv } from "../server.js";

export function health(c: Context<AppEnv>) {
  const config = c.get("config");
  const models = c.get("models");
  return c.json({
    status: "ok",
    engine: "transformers.js",
    device: models.device ?? config.device,
    models: {
      llm: config.modelId,
      ...(config.sttModelId ? { stt: config.sttModelId } : {}),
      ...(config.embeddingModelId ? { embedding: config.embeddingModelId } : {}),
    },
  });
}
