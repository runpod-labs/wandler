function envQuiet(): boolean {
  return ["1", "true", "yes", "on"].includes((process.env.WANDLER_QUIET ?? "").toLowerCase());
}

let quiet = envQuiet();

export function configureLogging(opts: { quiet?: boolean }): void {
  quiet = opts.quiet ?? envQuiet();
}

export function isQuiet(): boolean {
  return quiet;
}

export function logInfo(...args: unknown[]): void {
  if (!quiet) console.log(...args);
}

export function logWarn(...args: unknown[]): void {
  if (!quiet) console.warn(...args);
}
