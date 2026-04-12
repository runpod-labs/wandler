// ── Minimal multipart/form-data parser ──────────────────────────────────────

export interface MultipartFile {
  filename: string;
  data: Buffer;
}

export type MultipartParts = Record<string, string | MultipartFile>;

export function parseMultipart(buffer: Buffer, boundary: string): MultipartParts {
  const parts: MultipartParts = {};
  const boundaryBuf = Buffer.from(`--${boundary}`);
  let start = buffer.indexOf(boundaryBuf) + boundaryBuf.length;

  while (start < buffer.length) {
    const nextBoundary = buffer.indexOf(boundaryBuf, start);
    if (nextBoundary === -1) break;

    const part = buffer.subarray(start, nextBoundary);
    const headerEnd = part.indexOf("\r\n\r\n");
    if (headerEnd === -1) {
      start = nextBoundary + boundaryBuf.length;
      continue;
    }

    const headers = part.subarray(0, headerEnd).toString();
    const body = part.subarray(headerEnd + 4, part.length - 2);

    const nameMatch = headers.match(/name="([^"]+)"/);
    if (nameMatch) {
      const name = nameMatch[1]!;
      const filenameMatch = headers.match(/filename="([^"]+)"/);
      parts[name] = filenameMatch
        ? { filename: filenameMatch[1]!, data: body }
        : body.toString();
    }

    start = nextBoundary + boundaryBuf.length;
  }
  return parts;
}
