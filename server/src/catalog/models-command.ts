/**
 * `wandler model ls` — lists available models from the catalog.
 */

import { loadCatalog } from "./catalog.js";
import type { CatalogModel } from "./types.js";

function formatRow(m: CatalogModel): string {
  const prec = m.defaultPrecision;
  const caps = m.capabilities.join(", ");
  const repo = `${m.id}:${prec}`;
  return `${m.type.padEnd(9)} | ${m.size.padEnd(5)} | ${prec.padEnd(4)} | ${caps.padEnd(24)} | ${repo.padEnd(48)} | ${m.name}`;
}

export async function runModelLsCommand(opts: { type?: string }): Promise<void> {
  const catalog = await loadCatalog();
  const filterType = opts.type?.toLowerCase();

  if (filterType && !["llm", "embedding", "stt"].includes(filterType)) {
    console.error(`[wandler] Error: unknown type "${filterType}". Use: llm, embedding, stt`);
    process.exit(1);
  }

  const filtered = filterType
    ? catalog.models.filter((m) => m.type === filterType)
    : catalog.models;

  if (filtered.length === 0) {
    console.log("[wandler] No models found.");
    return;
  }

  console.log(
    `${"type".padEnd(9)} | ${"size".padEnd(5)} | ${"prec".padEnd(4)} | ${"capabilities".padEnd(24)} | ${"repo:precision".padEnd(48)} | name`,
  );
  console.log("-".repeat(120));

  for (const m of filtered) {
    console.log(formatRow(m));
  }

  console.log(`\n${filtered.length} model(s) found. Use the repo:precision value with --llm, --embedding, or --stt.`);
}
