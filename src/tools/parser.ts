import type { ToolCall } from "../types/openai.js";

// ── Tool call detection — parse model output into OpenAI format ─────────────

function makeCallId(): string {
  return `call_${Math.random().toString(36).slice(2, 10)}`;
}

function toToolCall(name: string, args: Record<string, unknown> | string): ToolCall {
  return {
    id: makeCallId(),
    type: "function",
    function: {
      name,
      arguments: typeof args === "string" ? args : JSON.stringify(args),
    },
  };
}

/** Strategy 1a: LFM Pythonic format — [func_name(arg="val", arg2="val2")] */
function parseLfmPythonic(text: string): ToolCall[] | null {
  const match = text.match(/\[(\w+)\(([^)]*)\)\]/);
  if (!match) return null;

  const name = match[1]!;
  const argsStr = match[2]!;
  const args: Record<string, string> = {};
  const kwargPattern = /(\w+)\s*=\s*"([^"]*)"/g;
  let m;
  while ((m = kwargPattern.exec(argsStr)) !== null) {
    args[m[1]!] = m[2]!;
  }
  return [toToolCall(name, args)];
}

/** Strategy 1b: LFM JSON format — [tool_calls [{...}]] */
function parseLfmJson(text: string): ToolCall[] | null {
  const match = text.match(/\[tool_calls\s*([\s\S]*?)(?:\]\s*$|\<\|tool_call_end\|>)/);
  if (!match) return null;

  const inner = match[1]!.trim();
  const arrStart = inner.indexOf("[");
  if (arrStart < 0) return null;

  let depth = 0;
  let arrEnd = -1;
  for (let i = arrStart; i < inner.length; i++) {
    if (inner[i] === "[" || inner[i] === "{") depth++;
    if (inner[i] === "]" || inner[i] === "}") depth--;
    if (depth === 0) {
      arrEnd = i;
      break;
    }
  }

  if (arrEnd <= arrStart) return null;

  try {
    const calls = JSON.parse(inner.substring(arrStart, arrEnd + 1)) as unknown;
    if (!Array.isArray(calls)) return null;

    return calls.map((tc: { name: string; arguments?: unknown }) =>
      toToolCall(
        tc.name,
        typeof tc.arguments === "string"
          ? tc.arguments
          : JSON.stringify(tc.arguments ?? {}),
      ),
    );
  } catch {
    return null;
  }
}

/** Strategy 2: Qwen format — <tool_call>{"name": "...", "arguments": {...}}</tool_call> */
function parseQwen(text: string): ToolCall[] | null {
  const match = text.match(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/);
  if (!match) return null;

  try {
    const call = JSON.parse(match[1]!) as { name: string; arguments?: unknown };
    return [
      toToolCall(
        call.name,
        typeof call.arguments === "string"
          ? call.arguments
          : JSON.stringify(call.arguments ?? {}),
      ),
    ];
  } catch {
    return null;
  }
}

/** Strategy 3: OpenAI JSON format — {"tool_calls": [...]} */
function parseOpenAiJson(text: string): ToolCall[] | null {
  const match = text.match(/\{[\s\S]*"tool_calls"[\s\S]*\}/);
  if (!match) return null;

  try {
    const parsed = JSON.parse(match[0]) as {
      tool_calls?: Array<{
        function?: { name?: string; arguments?: unknown };
        name?: string;
        arguments?: unknown;
      }>;
    };
    if (!Array.isArray(parsed.tool_calls)) return null;

    return parsed.tool_calls.map((tc) =>
      toToolCall(
        tc.function?.name ?? tc.name ?? "unknown",
        typeof tc.function?.arguments === "string"
          ? tc.function.arguments
          : JSON.stringify(tc.function?.arguments ?? tc.arguments ?? {}),
      ),
    );
  } catch {
    return null;
  }
}

/**
 * Parse model output text for tool calls in any supported format.
 * Returns null if no tool calls are detected.
 */
export function parseToolCalls(text: string): ToolCall[] | null {
  // Strip thinking blocks (e.g. Qwen outputs <think>...</think> before tool calls)
  const cleaned = text.replace(/<think>[\s\S]*?<\/think>/g, "").trim();

  return (
    parseLfmPythonic(cleaned) ??
    parseLfmJson(cleaned) ??
    parseQwen(cleaned) ??
    parseOpenAiJson(cleaned)
  );
}
