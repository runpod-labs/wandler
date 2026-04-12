import type { Context } from "hono";
import type { AppEnv } from "../server.js";
import type { ModelObject } from "../types/openai.js";

function buildModelList(config: { modelId: string; sttModelId: string; embeddingModelId: string }): ModelObject[] {
  const created = Math.floor(Date.now() / 1000);
  const models: ModelObject[] = [
    { id: config.modelId, object: "model", created, owned_by: "wandler" },
  ];
  if (config.sttModelId) {
    models.push({ id: config.sttModelId, object: "model", created, owned_by: "wandler" });
  }
  if (config.embeddingModelId) {
    models.push({ id: config.embeddingModelId, object: "model", created, owned_by: "wandler" });
  }
  return models;
}

export function listModels(c: Context<AppEnv>) {
  const config = c.get("config");
  return c.json({ object: "list", data: buildModelList(config) });
}

export function getModel(c: Context<AppEnv>) {
  const config = c.get("config");
  const modelId = c.req.param("id");
  const models = buildModelList(config);
  const found = models.find((m) => m.id === modelId);
  if (!found) {
    return c.json(
      { error: { message: `Model '${modelId}' not found`, type: "invalid_request_error", param: "model", code: "model_not_found" } },
      404,
    );
  }
  return c.json(found);
}
