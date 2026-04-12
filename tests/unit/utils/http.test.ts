import { describe, expect, it } from "vitest";
import { makeId } from "../../../src/utils/http.js";

describe("makeId", () => {
  it("generates IDs with default prefix", () => {
    const id = makeId();
    expect(id).toMatch(/^chatcmpl-[a-z0-9]+$/);
  });

  it("generates IDs with custom prefix", () => {
    const id = makeId("test");
    expect(id).toMatch(/^test-[a-z0-9]+$/);
  });

  it("generates unique IDs", () => {
    const ids = new Set(Array.from({ length: 100 }, () => makeId()));
    expect(ids.size).toBe(100);
  });
});
