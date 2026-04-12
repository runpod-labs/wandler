import { describe, expect, it } from "vitest";
import { parseMultipart } from "../../../src/utils/multipart.js";
import type { MultipartFile } from "../../../src/utils/multipart.js";

function buildMultipart(
  boundary: string,
  parts: Array<{ name: string; filename?: string; content: string | Buffer }>,
): Buffer {
  const chunks: Buffer[] = [];
  for (const part of parts) {
    chunks.push(Buffer.from(`--${boundary}\r\n`));
    if (part.filename) {
      chunks.push(
        Buffer.from(
          `Content-Disposition: form-data; name="${part.name}"; filename="${part.filename}"\r\n` +
            "Content-Type: application/octet-stream\r\n\r\n",
        ),
      );
    } else {
      chunks.push(
        Buffer.from(
          `Content-Disposition: form-data; name="${part.name}"\r\n\r\n`,
        ),
      );
    }
    chunks.push(Buffer.isBuffer(part.content) ? part.content : Buffer.from(part.content));
    chunks.push(Buffer.from("\r\n"));
  }
  chunks.push(Buffer.from(`--${boundary}--\r\n`));
  return Buffer.concat(chunks);
}

describe("parseMultipart", () => {
  const boundary = "----TestBoundary123";

  it("parses a text field", () => {
    const buffer = buildMultipart(boundary, [
      { name: "model", content: "whisper-1" },
    ]);
    const parts = parseMultipart(buffer, boundary);
    expect(parts.model).toBe("whisper-1");
  });

  it("parses a file field", () => {
    const fileData = Buffer.from([0x52, 0x49, 0x46, 0x46]); // RIFF header
    const buffer = buildMultipart(boundary, [
      { name: "file", filename: "audio.wav", content: fileData },
    ]);
    const parts = parseMultipart(buffer, boundary);
    const file = parts.file as MultipartFile;
    expect(file.filename).toBe("audio.wav");
    expect(file.data).toBeInstanceOf(Buffer);
  });

  it("parses multiple fields", () => {
    const buffer = buildMultipart(boundary, [
      { name: "model", content: "whisper-1" },
      { name: "language", content: "en" },
      { name: "file", filename: "test.wav", content: Buffer.from("audio") },
    ]);
    const parts = parseMultipart(buffer, boundary);
    expect(parts.model).toBe("whisper-1");
    expect(parts.language).toBe("en");
    expect((parts.file as MultipartFile).filename).toBe("test.wav");
  });

  it("returns empty object for empty body", () => {
    const buffer = Buffer.from(`--${boundary}--\r\n`);
    const parts = parseMultipart(buffer, boundary);
    expect(Object.keys(parts)).toHaveLength(0);
  });
});
