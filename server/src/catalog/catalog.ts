/**
 * Catalog loader with TTL-based caching.
 *
 * Resolution order:
 * 1. Local cache (if fresh, < 1 hour old)
 * 2. Remote fetch from GitHub (writes to local cache on success)
 * 3. Stale local cache (if remote fails)
 * 4. Bundled catalog shipped with the npm package (always available)
 */

import { readFileSync, writeFileSync, mkdirSync, statSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { homedir } from "node:os";
import type { Catalog } from "./types.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

const CATALOG_URL =
  "https://raw.githubusercontent.com/runpod-labs/wandler/main/server/dist/catalog.json";
const CACHE_TTL_MS = 60 * 60 * 1000; // 1 hour

function cacheDir(): string {
  return resolve(homedir(), ".wandler", "cache");
}

function cachePath(): string {
  return resolve(cacheDir(), "catalog.json");
}

function bundledPath(): string {
  // dist/catalog/catalog.js → dist/catalog.json
  return resolve(__dirname, "..", "catalog.json");
}

function isCacheFresh(): boolean {
  try {
    const stat = statSync(cachePath());
    return Date.now() - stat.mtimeMs < CACHE_TTL_MS;
  } catch {
    return false;
  }
}

function readCache(): Catalog | null {
  try {
    return JSON.parse(readFileSync(cachePath(), "utf-8")) as Catalog;
  } catch {
    return null;
  }
}

function writeCache(catalog: Catalog): void {
  try {
    mkdirSync(cacheDir(), { recursive: true });
    writeFileSync(cachePath(), JSON.stringify(catalog, null, 2));
  } catch {
    // Cache write failure is non-fatal
  }
}

function readBundled(): Catalog {
  return JSON.parse(readFileSync(bundledPath(), "utf-8")) as Catalog;
}

async function fetchRemote(): Promise<Catalog | null> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10_000);
    const res = await fetch(CATALOG_URL, { signal: controller.signal });
    clearTimeout(timeout);
    if (!res.ok) return null;
    return (await res.json()) as Catalog;
  } catch {
    return null;
  }
}

/**
 * Load the model catalog. Tries local cache first, then remote, then bundled.
 */
export async function loadCatalog(): Promise<Catalog> {
  // 1. Fresh cache
  if (isCacheFresh()) {
    const cached = readCache();
    if (cached) return cached;
  }

  // 2. Remote fetch
  const remote = await fetchRemote();
  if (remote) {
    writeCache(remote);
    return remote;
  }

  // 3. Stale cache
  const stale = readCache();
  if (stale) {
    process.stderr.write("[wandler] Warning: using stale catalog cache (remote fetch failed)\n");
    return stale;
  }

  // 4. Bundled fallback
  return readBundled();
}
