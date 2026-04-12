import http from "node:http";
import type { ServerConfig } from "./config.js";
import type { LoadedModels } from "./models/manager.js";
import { handleChatCompletions } from "./routes/chat.js";
import { handleCompletions } from "./routes/completions.js";
import { handleEmbeddings } from "./routes/embeddings.js";
import { handleListModels, handleGetModel } from "./routes/models.js";
import { handleAudioTranscriptions } from "./routes/audio.js";
import { handleTokenize, handleDetokenize } from "./routes/tokenize.js";
import { handleHealth } from "./routes/health.js";
import { handleAdminMetrics, trackRequest } from "./routes/admin.js";
import { errorJson, setCorsHeaders } from "./utils/http.js";

function checkAuth(req: http.IncomingMessage, apiKey: string): boolean {
  if (!apiKey) return true; // no key configured = open access
  const header = req.headers.authorization;
  if (!header) return false;
  // Accept "Bearer <key>" or just "<key>"
  const token = header.startsWith("Bearer ") ? header.slice(7) : header;
  return token === apiKey;
}

export function createServer(config: ServerConfig, models: LoadedModels): http.Server {
  const server = http.createServer(async (req, res) => {
    // CORS preflight — always allowed
    if (req.method === "OPTIONS") {
      setCorsHeaders(res);
      res.writeHead(204);
      res.end();
      return;
    }

    const url = req.url ?? "/";

    // Health — always allowed (load balancers need it)
    if (req.method === "GET" && (url === "/health" || url === "/")) {
      handleHealth(req, res, config);
      return;
    }

    // Auth check for all other endpoints
    if (!checkAuth(req, config.apiKey)) {
      errorJson(res, 401, "Invalid API key", "authentication_error", null, "invalid_api_key");
      return;
    }

    trackRequest();

    // GET /v1/models
    if (req.method === "GET" && url === "/v1/models") {
      handleListModels(req, res, config);
      return;
    }

    // GET /v1/models/{model}
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

    // POST /v1/completions
    if (req.method === "POST" && url === "/v1/completions") {
      await handleCompletions(req, res, models, config.modelId);
      return;
    }

    // POST /v1/embeddings
    if (req.method === "POST" && url === "/v1/embeddings") {
      await handleEmbeddings(req, res, models, config.embeddingModelId || config.modelId);
      return;
    }

    // POST /v1/audio/transcriptions
    if (req.method === "POST" && url === "/v1/audio/transcriptions") {
      await handleAudioTranscriptions(req, res, models);
      return;
    }

    // POST /tokenize
    if (req.method === "POST" && url === "/tokenize") {
      await handleTokenize(req, res, models);
      return;
    }

    // POST /detokenize
    if (req.method === "POST" && url === "/detokenize") {
      await handleDetokenize(req, res, models);
      return;
    }

    // GET /admin/metrics
    if (req.method === "GET" && url === "/admin/metrics") {
      handleAdminMetrics(req, res, config);
      return;
    }

    errorJson(res, 404, "Not found");
  });

  return server;
}
