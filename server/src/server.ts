import { Hono } from "hono";
import type { MiddlewareHandler } from "hono";
import { cors } from "hono/cors";
import { bearerAuth } from "hono/bearer-auth";
import { serve } from "@hono/node-server";
import type { ServerConfig } from "./config.js";
import type { LoadedModels } from "./models/manager.js";
import { chatCompletions } from "./routes/chat.js";
import { responses } from "./routes/responses.js";
import { completions } from "./routes/completions.js";
import { embeddings } from "./routes/embeddings.js";
import { listModels, getModel } from "./routes/models.js";
import { audioTranscriptions } from "./routes/audio.js";
import { tokenize, detokenize } from "./routes/tokenize.js";
import { health } from "./routes/health.js";
import { adminMetrics } from "./routes/admin.js";

export type AppEnv = {
  Variables: {
    config: ServerConfig;
    models: LoadedModels;
  };
};

class QueueTimeoutError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "QueueTimeoutError";
  }
}

class RequestAbortedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RequestAbortedError";
  }
}

type QueuedRequest = {
  enter: () => void;
  reject: (error: Error) => void;
  cleanup: () => void;
};

class RequestLimiter {
  private active = 0;
  private readonly queue: QueuedRequest[] = [];

  constructor(
    private readonly maxConcurrent: number,
    private readonly timeoutMs: number,
  ) {}

  acquire(signal?: AbortSignal): Promise<() => void> {
    if (this.maxConcurrent < 1) return Promise.resolve(() => {});
    if (signal?.aborted) {
      return Promise.reject(new RequestAbortedError("Request aborted before entering generation queue"));
    }

    if (this.active < this.maxConcurrent) {
      this.active++;
      return Promise.resolve(() => this.release());
    }

    return new Promise((resolve, reject) => {
      let settled = false;
      let timeout: NodeJS.Timeout | undefined;

      const queued: QueuedRequest = {
        enter: () => {
          if (settled) return;
          settled = true;
          queued.cleanup();
          this.active++;
          resolve(() => this.release());
        },
        reject: (error: Error) => {
          if (settled) return;
          settled = true;
          queued.cleanup();
          const index = this.queue.indexOf(queued);
          if (index >= 0) this.queue.splice(index, 1);
          reject(error);
        },
        cleanup: () => {
          if (timeout) clearTimeout(timeout);
          signal?.removeEventListener("abort", abort);
        },
      };

      const abort = () => queued.reject(new RequestAbortedError("Request aborted while waiting for generation"));
      signal?.addEventListener("abort", abort, { once: true });

      if (this.timeoutMs > 0) {
        timeout = setTimeout(
          () => queued.reject(new QueueTimeoutError("Timed out waiting for an available generation slot")),
          this.timeoutMs,
        );
      }

      this.queue.push(queued);
    });
  }

  private release(): void {
    this.active = Math.max(0, this.active - 1);
    while (this.active < this.maxConcurrent) {
      const next = this.queue.shift();
      if (!next) return;
      next.enter();
      if (this.active >= this.maxConcurrent) return;
    }
  }
}

export function createApp(config: ServerConfig, models: LoadedModels) {
  const app = new Hono<AppEnv>();
  const generationLimiter = new RequestLimiter(config.maxConcurrent, config.timeout);

  // Inject config + models into every request context
  app.use("*", async (c, next) => {
    c.set("config", config);
    c.set("models", models);
    await next();
  });

  const limitGeneration: MiddlewareHandler<AppEnv> = async (c, next) => {
    let release: (() => void) | null = null;
    try {
      release = await generationLimiter.acquire(c.req.raw.signal);
      await next();
    } catch (err) {
      if (err instanceof QueueTimeoutError) {
        return c.json(
          { error: { message: err.message, type: "server_overloaded", param: null, code: "generation_queue_timeout" } },
          503,
        );
      }
      if (err instanceof RequestAbortedError) {
        return c.json(
          { error: { message: err.message, type: "client_closed_request", param: null, code: "request_aborted" } },
          408,
        );
      }
      throw err;
    } finally {
      release?.();
    }
  };

  // CORS — configurable origin
  app.use("*", cors({ origin: config.corsOrigin }));

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

  // OpenAI-compatible endpoints
  app.get("/v1/models", listModels);
  app.get("/v1/models/:id{.+}", getModel);
  app.use("/v1/responses", limitGeneration);
  app.use("/v1/chat/completions", limitGeneration);
  app.use("/v1/completions", limitGeneration);
  app.post("/v1/responses", responses);
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
  return serve({ fetch: app.fetch, port: config.port, hostname: config.host });
}
