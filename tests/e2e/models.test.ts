import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { TestServer } from "./helpers.js";
import { startTestServer } from "./helpers.js";

describe("GET /v1/models", () => {
  let ts: TestServer;

  beforeAll(async () => {
    ts = await startTestServer();
  });

  afterAll(async () => {
    await ts.close();
  });

  it("returns list of available models", async () => {
    const res = await fetch(`${ts.baseUrl}/v1/models`);
    expect(res.status).toBe(200);

    const body = (await res.json()) as {
      object: string;
      data: Array<{
        id: string;
        object: string;
        created: number;
        owned_by: string;
      }>;
    };
    expect(body.object).toBe("list");
    expect(body.data).toHaveLength(2); // LLM + STT
    expect(body.data[0]!.id).toBe("mock-model/test");
    expect(body.data[0]!.object).toBe("model");
    expect(body.data[0]!.owned_by).toBe("wandler");
    expect(body.data[1]!.id).toBe("mock-stt/test");
  });
});

describe("GET /v1/models/{model}", () => {
  let ts: TestServer;

  beforeAll(async () => {
    ts = await startTestServer();
  });

  afterAll(async () => {
    await ts.close();
  });

  it("returns model details for existing model", async () => {
    const res = await fetch(
      `${ts.baseUrl}/v1/models/${encodeURIComponent("mock-model/test")}`,
    );
    expect(res.status).toBe(200);

    const body = (await res.json()) as {
      id: string;
      object: string;
      owned_by: string;
    };
    expect(body.id).toBe("mock-model/test");
    expect(body.object).toBe("model");
  });

  it("returns 404 for non-existent model", async () => {
    const res = await fetch(
      `${ts.baseUrl}/v1/models/${encodeURIComponent("does-not/exist")}`,
    );
    expect(res.status).toBe(404);

    const body = (await res.json()) as {
      error: { message: string; code: string };
    };
    expect(body.error.code).toBe("model_not_found");
  });
});
