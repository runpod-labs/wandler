#!/usr/bin/env tsx
/**
 * Merges individual model JSON files from registry/models/ into a single catalog.json.
 * Output is written to dist/catalog.json (bundled with the npm package).
 *
 * Usage: npx tsx scripts/build-catalog.ts
 */

import { readdirSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const modelsDir = resolve(__dirname, "..", "registry", "models");
const outFile = resolve(__dirname, "..", "dist", "catalog.json");

const files = readdirSync(modelsDir).filter((f) => f.endsWith(".json")).sort();
const models: unknown[] = [];
const seenIds = new Set<string>();

for (const file of files) {
  const raw = readFileSync(resolve(modelsDir, file), "utf-8");
  const model = JSON.parse(raw) as { id: string };

  if (!model.id) {
    console.error(`[build-catalog] Skipping ${file}: missing "id"`);
    continue;
  }
  if (seenIds.has(model.id)) {
    console.error(`[build-catalog] ERROR: duplicate id "${model.id}" in ${file}`);
    process.exit(1);
  }
  seenIds.add(model.id);
  models.push(model);
}

const catalog = { version: "1", models };

mkdirSync(dirname(outFile), { recursive: true });
writeFileSync(outFile, JSON.stringify(catalog, null, 2) + "\n");
console.log(`[build-catalog] Wrote ${models.length} models to ${outFile}`);
