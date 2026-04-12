import type http from "node:http";
import type { ErrorResponse } from "../types/openai.js";

export function makeId(prefix = "chatcmpl"): string {
  return `${prefix}-${Math.random().toString(36).slice(2, 14)}`;
}

export function json(res: http.ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
  });
  res.end(JSON.stringify(body));
}

export function errorJson(
  res: http.ServerResponse,
  status: number,
  message: string,
  type = "invalid_request_error",
  param: string | null = null,
  code: string | null = null,
): void {
  const body: ErrorResponse = {
    error: { message, type, param, code },
  };
  json(res, status, body);
}

export async function readBody(req: http.IncomingMessage): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(chunk as Buffer);
  }
  return Buffer.concat(chunks);
}

export function setCorsHeaders(res: http.ServerResponse): void {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
}
