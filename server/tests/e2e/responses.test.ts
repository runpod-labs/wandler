import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { TestServer, NamedSSEEvent } from "./helpers.js";
import { startTestServer, postJson, collectNamedSSE } from "./helpers.js";

describe("Responses API (/v1/responses)", () => {
  let ts: TestServer;

  beforeAll(async () => {
    ts = await startTestServer();
  });

  afterAll(async () => {
    await ts.close();
  });

  // ── Non-streaming ──────────────────────────────────────────────────────

  it("returns a response with string input", async () => {
    const { status, body } = await postJson(`${ts.baseUrl}/v1/responses`, {
      model: "mock-model/test",
      input: "Hello",
    });

    expect(status).toBe(200);
    const resp = body as Record<string, unknown>;
    expect(resp.id).toMatch(/^resp-/);
    expect(resp.object).toBe("response");
    expect(resp.status).toBe("completed");
    expect(resp.model).toBe("mock-model/test");

    const output = resp.output as Array<Record<string, unknown>>;
    expect(output).toHaveLength(1);
    expect(output[0]!.type).toBe("message");
    expect(output[0]!.role).toBe("assistant");

    const content = (output[0] as Record<string, unknown>).content as Array<Record<string, unknown>>;
    expect(content[0]!.type).toBe("output_text");
    expect(content[0]!.text).toBeTruthy();

    const usage = resp.usage as Record<string, number>;
    expect(usage.input_tokens).toBeGreaterThan(0);
    expect(usage.output_tokens).toBeGreaterThan(0);
    expect(usage.total_tokens).toBe(usage.input_tokens + usage.output_tokens);
  });

  it("returns a response with array input", async () => {
    const { status, body } = await postJson(`${ts.baseUrl}/v1/responses`, {
      model: "mock-model/test",
      input: [{ role: "user", content: "Hello" }],
    });

    expect(status).toBe(200);
    const resp = body as Record<string, unknown>;
    expect(resp.status).toBe("completed");
    const output = resp.output as Array<Record<string, unknown>>;
    expect(output).toHaveLength(1);
    expect(output[0]!.type).toBe("message");
  });

  it("supports instructions as system message", async () => {
    const { status, body } = await postJson(`${ts.baseUrl}/v1/responses`, {
      model: "mock-model/test",
      input: "Hi",
      instructions: "You are a helpful assistant.",
    });

    expect(status).toBe(200);
    const resp = body as Record<string, unknown>;
    expect(resp.status).toBe("completed");
    expect(resp.instructions).toBe("You are a helpful assistant.");
  });

  it("maps developer role to system", async () => {
    const { status, body } = await postJson(`${ts.baseUrl}/v1/responses`, {
      model: "mock-model/test",
      input: [
        { role: "developer", content: "Be concise." },
        { role: "user", content: "Hello" },
      ],
    });

    expect(status).toBe(200);
    const resp = body as Record<string, unknown>;
    expect(resp.status).toBe("completed");
  });

  it("returns 400 when input is missing", async () => {
    const { status, body } = await postJson(`${ts.baseUrl}/v1/responses`, {
      model: "mock-model/test",
    });

    expect(status).toBe(400);
    const resp = body as Record<string, unknown>;
    expect(resp.error).toBeDefined();
  });

  // ── Streaming ──────────────────────────────────────────────────────────

  it("streams named SSE events for text generation", async () => {
    const { status, events } = await collectNamedSSE(`${ts.baseUrl}/v1/responses`, {
      model: "mock-model/test",
      input: "Hello",
      stream: true,
    });

    expect(status).toBe(200);

    // Verify event sequence
    const eventTypes = events.map((e) => e.event);
    expect(eventTypes[0]).toBe("response.created");
    expect(eventTypes[1]).toBe("response.output_item.added");
    expect(eventTypes[2]).toBe("response.content_part.added");

    // Text deltas should be in the middle
    const deltas = events.filter((e) => e.event === "response.output_text.delta");
    expect(deltas.length).toBeGreaterThan(0);

    // Check delta shape
    const firstDelta = deltas[0]!.data as Record<string, unknown>;
    expect(firstDelta.type).toBe("response.output_text.delta");
    expect(firstDelta.delta).toBeTruthy();
    expect(firstDelta.output_index).toBe(0);
    expect(firstDelta.content_index).toBe(0);

    // Verify closing events
    expect(eventTypes).toContain("response.output_text.done");
    expect(eventTypes).toContain("response.output_item.done");
    expect(eventTypes[eventTypes.length - 1]).toBe("response.completed");

    // Verify completed response has usage
    const completed = events.find((e) => e.event === "response.completed")!.data as Record<string, unknown>;
    const resp = completed.response as Record<string, unknown>;
    expect(resp.status).toBe("completed");
    const usage = resp.usage as Record<string, number>;
    expect(usage.input_tokens).toBeGreaterThan(0);
    expect(usage.output_tokens).toBeGreaterThan(0);
  });

  it("streams response.created with in_progress status", async () => {
    const { events } = await collectNamedSSE(`${ts.baseUrl}/v1/responses`, {
      model: "mock-model/test",
      input: "Hello",
      stream: true,
    });

    const created = events.find((e) => e.event === "response.created")!.data as Record<string, unknown>;
    const resp = created.response as Record<string, unknown>;
    expect(resp.id).toMatch(/^resp-/);
    expect(resp.status).toBe("in_progress");
    expect(resp.object).toBe("response");
  });

  // ── Function call output in input (round-trip) ───────────────────────

  it("accepts function_call and function_call_output in input", async () => {
    const { status, body } = await postJson(`${ts.baseUrl}/v1/responses`, {
      model: "mock-model/test",
      input: [
        { role: "user", content: "What is the weather?" },
        { type: "function_call", call_id: "call_123", name: "get_weather", arguments: '{"city":"Berlin"}' },
        { type: "function_call_output", call_id: "call_123", output: '{"temp":20}' },
      ],
    });

    expect(status).toBe(200);
    const resp = body as Record<string, unknown>;
    expect(resp.status).toBe("completed");
  });
});
