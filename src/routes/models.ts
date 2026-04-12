import type http from "node:http";
import type { ServerConfig } from "../config.js";
import type { ModelListResponse, ModelObject } from "../types/openai.js";
import { errorJson, json } from "../utils/http.js";

function buildModelList(config: ServerConfig): ModelObject[] {
  const created = Math.floor(Date.now() / 1000);
  const models: ModelObject[] = [
    { id: config.modelId, object: "model", created, owned_by: "wandler" },
  ];
  if (config.sttModelId) {
    models.push({ id: config.sttModelId, object: "model", created, owned_by: "wandler" });
  }
  return models;
}

export function handleListModels(
  _req: http.IncomingMessage,
  res: http.ServerResponse,
  config: ServerConfig,
): void {
  const response: ModelListResponse = {
    object: "list",
    data: buildModelList(config),
  };
  json(res, 200, response);
}

export function handleGetModel(
  _req: http.IncomingMessage,
  res: http.ServerResponse,
  config: ServerConfig,
  modelId: string,
): void {
  const models = buildModelList(config);
  const found = models.find((m) => m.id === modelId);
  if (!found) {
    errorJson(res, 404, `Model '${modelId}' not found`, "invalid_request_error", "model", "model_not_found");
    return;
  }
  json(res, 200, found);
}
