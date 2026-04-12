import { Hono } from "hono";
import { cors } from "hono/cors";
import { bearerAuth } from "hono/bearer-auth";
import { serve } from "@hono/node-server";
import type { ServerConfig } from "./config.js";
import type { LoadedModels } from "./models/manager.js";
import { chatCompletions } from "./routes/chat.js";
import { completions } from "./routes/completions.js";
import { embeddings } from "./routes/embeddings.js";
import { listModels, getModel } from "./routes/models.js";
import { audioTranscriptions } from "./routes/audio.js";
import { tokenize, detokenize } from "./routes/tokenize.js";
import { health } from "./routes/health.js";
import { adminMetrics, trackRequest } from "./routes/admin.js";

export type AppEnv = {
  Variables: {
    config: ServerConfig;
    models: LoadedModels;
  };
};

export function createApp(config: ServerConfig, models: LoadedModels) {
  const app = new Hono<AppEnv>();

  // Inject config + models into every request context
  app.use("*", async (c, next) => {
    c.set("config", config);
    c.set("models", models);
    await next();
  });

  // CORS on everything
  app.use("*", cors());

  // Health — always open (load balancers need it)
  app.get("/health", health);
  app.get("/", health);

  // Auth middleware — only if API key is configured
  if (config.apiKey) {
    app.use("/v1/*", bearerAuth({ token: config.apiKey }));
    app.use("/admin/*", bearerAuth({ token: config.apiKey }));
    app.use("/tokenize", bearerAuth({ token: config.apiKey }));
    app.use("/detokenize", bearerAuth({ token: config.apiKey }));
  }

  // Track requests
  app.use("/v1/*", async (_c, next) => {
    await next();
    trackRequest();
  });

  // OpenAI-compatible endpoints
  app.get("/v1/models", listModels);
  app.get("/v1/models/:id{.+}", getModel);
  app.post("/v1/chat/completions", chatCompletions);
  app.post("/v1/completions", completions);
  app.post("/v1/embeddings", embeddings);
  app.post("/v1/audio/transcriptions", audioTranscriptions);

  // Utility endpoints
  app.post("/tokenize", tokenize);
  app.post("/detokenize", detokenize);

  // Admin
  app.get("/admin/metrics", adminMetrics);

  // Global error handler
  app.onError((err, c) => {
    // Hono's bearerAuth throws HTTPException — re-throw to let Hono handle it
    if ("status" in err && "res" in err) {
      const httpErr = err as { status: number; res: Response };
      return httpErr.res;
    }
    if (err instanceof SyntaxError) {
      return c.json(
        { error: { message: "Invalid JSON", type: "invalid_request_error", param: null, code: null } },
        400,
      );
    }
    console.error("[wandler] Error:", err);
    return c.json(
      { error: { message: err.message, type: "server_error", param: null, code: null } },
      500,
    );
  });

  // 404 handler
  app.notFound((c) => {
    return c.json(
      { error: { message: "Not found", type: "invalid_request_error", param: null, code: null } },
      404,
    );
  });

  return app;
}

export function startServer(config: ServerConfig, models: LoadedModels) {
  const app = createApp(config, models);
  return serve({ fetch: app.fetch, port: config.port });
}
