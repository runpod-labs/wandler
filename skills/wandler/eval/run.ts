#!/usr/bin/env tsx
/**
 * wandler skill eval — validation harness
 *
 * This script defines scenarios and validates LLM responses against them.
 * It does NOT call any LLM API itself. Instead, the calling agent (Claude Code,
 * or any other system) runs each scenario's prompt through an LLM with and
 * without the SKILL.md context, collects the responses, and feeds them back
 * here for scoring.
 *
 * Usage:
 *   tsx run.ts list                          # List scenario names + prompts
 *   tsx run.ts prompt <name> [--with-skill]  # Print full prompt for a scenario
 *   tsx run.ts validate <results.json>       # Score responses and print table
 *
 * results.json format:
 *   {
 *     "<scenario-name>": {
 *       "withSkill": "LLM response text...",
 *       "withoutSkill": "LLM response text..."
 *     },
 *     ...
 *   }
 */

import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { scenarios } from "./scenarios.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const skillPath = resolve(__dirname, "..", "SKILL.md");
const skillContent = readFileSync(skillPath, "utf-8");

// --- Types ---

interface ResponsePair {
  withSkill: string;
  withoutSkill: string;
}

interface CheckResult {
  name: string;
  passed: boolean;
}

interface ScenarioScore {
  name: string;
  description: string;
  withSkill: { checks: CheckResult[]; passed: number; total: number };
  withoutSkill: { checks: CheckResult[]; passed: number; total: number };
  delta: number;
}

// --- Commands ---

const [command, ...args] = process.argv.slice(2);

switch (command) {
  case "list":
    cmdList();
    break;
  case "prompt":
    cmdPrompt(args);
    break;
  case "validate":
    cmdValidate(args);
    break;
  default:
    console.log("Usage:");
    console.log("  tsx run.ts list                          # List scenarios");
    console.log(
      "  tsx run.ts prompt <name> [--with-skill]  # Print prompt for a scenario",
    );
    console.log(
      "  tsx run.ts validate <results.json>        # Validate and score",
    );
    process.exit(command ? 1 : 0);
}

// --- list ---

function cmdList() {
  console.log(JSON.stringify(
    scenarios.map((s) => ({
      name: s.name,
      description: s.description,
      prompt: s.prompt,
      checks: s.checks.map((c) => c.name),
    })),
    null,
    2,
  ));
}

// --- prompt ---

function cmdPrompt(args: string[]) {
  const name = args[0];
  const withSkill = args.includes("--with-skill");

  if (!name) {
    console.error("Error: provide a scenario name");
    process.exit(1);
  }

  const scenario = scenarios.find((s) => s.name === name);
  if (!scenario) {
    console.error(`Error: unknown scenario "${name}"`);
    console.error(`Available: ${scenarios.map((s) => s.name).join(", ")}`);
    process.exit(1);
  }

  const systemPrompt = withSkill
    ? `You are a helpful technical assistant.\nYou have access to the following tool documentation:\n\n${skillContent}`
    : "You are a helpful technical assistant.";

  console.log(
    JSON.stringify({ system: systemPrompt, user: scenario.prompt }, null, 2),
  );
}

// --- validate ---

function cmdValidate(args: string[]) {
  const filePath = args[0];
  if (!filePath) {
    console.error("Error: provide a results.json path");
    process.exit(1);
  }

  const raw = readFileSync(resolve(filePath), "utf-8");
  const results: Record<string, ResponsePair> = JSON.parse(raw);

  const scores: ScenarioScore[] = [];
  let totalWith = 0;
  let totalWithout = 0;
  let totalChecks = 0;

  for (const scenario of scenarios) {
    const pair = results[scenario.name];
    if (!pair) {
      console.error(`Warning: no results for scenario "${scenario.name}"`);
      continue;
    }

    const withChecks = scenario.checks.map((c) => ({
      name: c.name,
      passed: c.test(pair.withSkill),
    }));
    const withoutChecks = scenario.checks.map((c) => ({
      name: c.name,
      passed: c.test(pair.withoutSkill),
    }));

    const withPassed = withChecks.filter((c) => c.passed).length;
    const withoutPassed = withoutChecks.filter((c) => c.passed).length;

    scores.push({
      name: scenario.name,
      description: scenario.description,
      withSkill: { checks: withChecks, passed: withPassed, total: scenario.checks.length },
      withoutSkill: { checks: withoutChecks, passed: withoutPassed, total: scenario.checks.length },
      delta: withPassed - withoutPassed,
    });

    totalWith += withPassed;
    totalWithout += withoutPassed;
    totalChecks += scenario.checks.length;
  }

  // --- Detailed output ---

  console.log();
  for (const s of scores) {
    console.log(`--- ${s.name} ---`);
    console.log(`${s.description}\n`);

    console.log(`  With skill (${s.withSkill.passed}/${s.withSkill.total}):`);
    for (const c of s.withSkill.checks) {
      console.log(`    ${c.passed ? "PASS" : "FAIL"}  ${c.name}`);
    }

    console.log(`  Without skill (${s.withoutSkill.passed}/${s.withoutSkill.total}):`);
    for (const c of s.withoutSkill.checks) {
      console.log(`    ${c.passed ? "PASS" : "FAIL"}  ${c.name}`);
    }
    console.log();
  }

  // --- Summary table ---

  console.log("=".repeat(64));
  console.log("Summary");
  console.log("=".repeat(64));
  console.log();
  console.log(`${pad("Scenario", 32)}${pad("With", 10)}${pad("Without", 10)}Delta`);
  console.log("-".repeat(64));

  for (const s of scores) {
    const sign = s.delta > 0 ? "+" : s.delta === 0 ? " " : "";
    console.log(
      `${pad(s.name, 32)}${pad(`${s.withSkill.passed}/${s.withSkill.total}`, 10)}${pad(`${s.withoutSkill.passed}/${s.withoutSkill.total}`, 10)}${sign}${s.delta}`,
    );
  }

  console.log("-".repeat(64));
  const totalDelta = totalWith - totalWithout;
  const sign = totalDelta > 0 ? "+" : totalDelta === 0 ? " " : "";
  console.log(
    `${pad("TOTAL", 32)}${pad(`${totalWith}/${totalChecks}`, 10)}${pad(`${totalWithout}/${totalChecks}`, 10)}${sign}${totalDelta}`,
  );

  const pctWith = ((totalWith / totalChecks) * 100).toFixed(1);
  const pctWithout = ((totalWithout / totalChecks) * 100).toFixed(1);
  const pctDelta = ((totalDelta / totalChecks) * 100).toFixed(1);

  console.log();
  console.log(`  With skill:    ${pctWith}%`);
  console.log(`  Without skill: ${pctWithout}%`);
  console.log(`  Improvement:   ${sign}${pctDelta} percentage points`);

  if (totalDelta <= 0) {
    console.log("\nRESULT: FAIL -- skill did not improve responses\n");
    process.exit(1);
  }

  console.log("\nRESULT: PASS -- skill improved responses\n");
}

// --- Helpers ---

function pad(s: string, len: number): string {
  return s.length >= len ? s + " " : s + " ".repeat(len - s.length);
}
