import http from "node:http";
import type { ServerConfig } from "./config.js";
import type { LoadedModels } from "./models/manager.js";
import { handleChatCompletions } from "./routes/chat.js";
import { handleListModels, handleGetModel } from "./routes/models.js";
import { handleAudioTranscriptions } from "./routes/audio.js";
import { handleHealth } from "./routes/health.js";
import { errorJson, setCorsHeaders } from "./utils/http.js";

export function createServer(config: ServerConfig, models: LoadedModels): http.Server {
  const server = http.createServer(async (req, res) => {
    // CORS preflight
    if (req.method === "OPTIONS") {
      setCorsHeaders(res);
      res.writeHead(204);
      res.end();
      return;
    }

    const url = req.url ?? "/";

    // GET /v1/models
    if (req.method === "GET" && url === "/v1/models") {
      handleListModels(req, res, config);
      return;
    }

    // GET /v1/models/{model} — model ID may contain slashes
    if (req.method === "GET" && url.startsWith("/v1/models/")) {
      const modelId = decodeURIComponent(url.slice("/v1/models/".length));
      handleGetModel(req, res, config, modelId);
      return;
    }

    // POST /v1/chat/completions
    if (req.method === "POST" && url === "/v1/chat/completions") {
      await handleChatCompletions(req, res, models, config.modelId);
      return;
    }

    // POST /v1/audio/transcriptions
    if (req.method === "POST" && url === "/v1/audio/transcriptions") {
      await handleAudioTranscriptions(req, res, models);
      return;
    }

    // GET /health or GET /
    if (req.method === "GET" && (url === "/health" || url === "/")) {
      handleHealth(req, res, config);
      return;
    }

    errorJson(res, 404, "Not found");
  });

  return server;
}
