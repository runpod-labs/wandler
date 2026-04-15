import { describe, expect, it } from "vitest";
import { execFileSync } from "node:child_process";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const cli = resolve(__dirname, "..", "..", "..", "dist", "cli.js");

function run(...args: string[]): string {
  return execFileSync("node", [cli, ...args], { encoding: "utf-8", timeout: 10_000 });
}

function runFail(...args: string[]): { stderr: string; code: number | null } {
  try {
    execFileSync("node", [cli, ...args], { encoding: "utf-8", timeout: 10_000, stdio: "pipe" });
    return { stderr: "", code: 0 };
  } catch (e: unknown) {
    const err = e as { stderr: string; status: number | null };
    return { stderr: err.stderr ?? "", code: err.status };
  }
}

describe("CLI integration", () => {
  describe("wandler model ls", () => {
    it("lists all models", () => {
      const out = run("model", "ls");
      expect(out).toContain("llm");
      expect(out).toContain("embedding");
      expect(out).toContain("stt");
      expect(out).toContain("model(s) found");
    });

    it("filters by --type llm", () => {
      const out = run("model", "ls", "--type", "llm");
      expect(out).toContain("llm");
      expect(out).not.toMatch(/^embedding\s/m);
      expect(out).not.toMatch(/^stt\s/m);
    });

    it("filters by --type embedding", () => {
      const out = run("model", "ls", "--type", "embedding");
      expect(out).toContain("embedding");
      expect(out).not.toMatch(/^llm\s/m);
    });

    it("filters by --type stt", () => {
      const out = run("model", "ls", "--type", "stt");
      expect(out).toContain("stt");
      expect(out).not.toMatch(/^llm\s/m);
    });

    it("shows help with --help", () => {
      const out = run("model", "ls", "--help");
      expect(out).toContain("--type");
    });
  });

  describe("wandler --version", () => {
    it("prints version", () => {
      const out = run("--version");
      expect(out.trim()).toMatch(/^\d+\.\d+\.\d+$/);
    });
  });

  describe("wandler --help", () => {
    it("shows model subcommand", () => {
      const out = run("--help");
      expect(out).toContain("model");
    });
  });

  describe("error handling", () => {
    it("exits 1 with no model flags", () => {
      const { code } = runFail();
      expect(code).toBe(1);
    });

    it("rejects unknown flags", () => {
      const { code } = runFail("--bogus");
      expect(code).toBe(1);
    });
  });
});
