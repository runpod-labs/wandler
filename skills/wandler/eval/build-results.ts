#!/usr/bin/env tsx
/**
 * Reads responses.txt (separated by <<<SEP>>>) and writes results.json.
 * Response order: for each scenario, withSkill first, then withoutSkill.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { scenarios } from "./scenarios.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const raw = readFileSync(resolve(__dirname, "responses.txt"), "utf-8");
const parts = raw.split("<<<SEP>>>").map((s) => s.trim());

const expected = scenarios.length * 2;
if (parts.length !== expected) {
  console.error(`Expected ${expected} responses, got ${parts.length}`);
  process.exit(1);
}

const results: Record<string, { withSkill: string; withoutSkill: string }> = {};

for (let i = 0; i < scenarios.length; i++) {
  results[scenarios[i].name] = {
    withSkill: parts[i * 2],
    withoutSkill: parts[i * 2 + 1],
  };
}

const outPath = resolve(__dirname, "results.json");
writeFileSync(outPath, JSON.stringify(results, null, 2));
console.log(`Wrote ${outPath} (${scenarios.length} scenarios)`);
