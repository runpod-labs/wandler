/**
 * `wandler models` CLI command — lists available models from the catalog.
 *
 * Usage:
 *   wandler models                    # all models
 *   wandler models --type llm         # LLMs only
 *   wandler models --type embedding   # embeddings only
 *   wandler models --type stt         # STT only
 *
 * Output: type | size | default precision | capabilities | repo | name
 */

import { parseArgs } from "node:util";
import { loadCatalog } from "./catalog.js";
import type { CatalogModel } from "./types.js";

function formatRow(m: CatalogModel): string {
  const prec = m.defaultPrecision;
  const caps = m.capabilities.join(", ");
  const repo = `${m.id}:${prec}`;
  return `${m.type.padEnd(9)} | ${m.size.padEnd(5)} | ${prec.padEnd(4)} | ${caps.padEnd(24)} | ${repo.padEnd(48)} | ${m.name}`;
}

export async function runModelsCommand(args: string[]): Promise<void> {
  const { values } = parseArgs({
    args,
    options: {
      type: { type: "string", short: "t" },
      help: { type: "boolean", short: "h" },
    },
    strict: true,
    allowPositionals: false,
  });

  if (values.help) {
    console.log(`
wandler models — list available models

Usage:
  wandler models [options]

Options:
  -t, --type <type>   Filter by type: llm, embedding, stt
  -h, --help          Show this help

Output format: type | size | precision | capabilities | repo:precision | name
Use the repo:precision value with --llm, --embedding, or --stt flags.

Examples:
  wandler models
  wandler models --type llm
  wandler models --type embedding
`);
    return;
  }

  const catalog = await loadCatalog();
  const filterType = values.type?.toLowerCase();

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

  // Header
  console.log(
    `${"type".padEnd(9)} | ${"size".padEnd(5)} | ${"prec".padEnd(4)} | ${"capabilities".padEnd(24)} | ${"repo:precision".padEnd(48)} | name`,
  );
  console.log("-".repeat(120));

  for (const m of filtered) {
    console.log(formatRow(m));
  }

  console.log(`\n${filtered.length} model(s) found. Use the repo:precision value with --llm, --embedding, or --stt.`);
}
