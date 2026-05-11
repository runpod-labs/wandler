import React from "react";
import {
  AbsoluteFill,
  CalculateMetadataFunction,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { z } from "zod";
import { loadFont as loadGeistMono } from "@remotion/google-fonts/GeistMono";
import { loadFont as loadGeist } from "@remotion/google-fonts/Geist";

const { fontFamily: GEIST_MONO } = loadGeistMono("normal", {
  weights: ["400", "500", "600", "700"],
});
const { fontFamily: GEIST } = loadGeist("normal", {
  weights: ["500", "600", "700"],
});

export const comparisonSchema = z.object({
  fp16Path: z.string(),
  tqPath: z.string(),
  metaPath: z.string(),
  holdFrames: z.number(),
  speed: z.number(), // playback speed multiplier (2 = 2x faster)
});

type Event = { step: number; t_ms: number; dt_ms: number; token: string };
type Meta = {
  model: string;
  hardware?: string;
  runtime?: string;
  prompt_tokens: number;
  fp16: { totalDecodeMs: number; n: number; kvBytes: number };
  tq: { totalDecodeMs: number; n: number; kvBytes: number };
  decode_speedup: number;
};

type Props = z.infer<typeof comparisonSchema> & {
  fp16Events: Event[];
  tqEvents: Event[];
  meta: Meta;
};

const parseJsonl = (text: string): Event[] =>
  text
    .split("\n")
    .filter((l) => l.trim().length > 0)
    .map((l) => JSON.parse(l) as Event);

export const calculateComparisonMetadata: CalculateMetadataFunction<Props> = async ({
  props,
}) => {
  const [fp16Text, tqText, metaText] = await Promise.all([
    fetch(staticFile(props.fp16Path)).then((r) => r.text()),
    fetch(staticFile(props.tqPath)).then((r) => r.text()),
    fetch(staticFile(props.metaPath)).then((r) => r.text()),
  ]);
  const fp16Events = parseJsonl(fp16Text);
  const tqEvents = parseJsonl(tqText);
  const meta: Meta = JSON.parse(metaText);
  const fps = 60;
  const maxMs = Math.max(
    fp16Events[fp16Events.length - 1]?.t_ms ?? 0,
    tqEvents[tqEvents.length - 1]?.t_ms ?? 0,
  );
  const durationInFrames =
    Math.ceil((maxMs / 1000) * fps / props.speed) + props.holdFrames;
  return {
    durationInFrames,
    props: {
      ...props,
      fp16Events,
      tqEvents,
      meta,
    },
  };
};

const visibleTokensAt = (events: Event[], elapsedMs: number): Event[] => {
  // Linear scan is fine — at most 200 events.
  const out: Event[] = [];
  for (const e of events) {
    if (e.t_ms <= elapsedMs) out.push(e);
    else break;
  }
  return out;
};

const COLOR_BG = "#000000";
const COLOR_TEXT = "#e6e6e6";
const COLOR_DIM = "#6a6a6a";
const COLOR_FP16 = "#ff8a5c"; // warm = slow
const COLOR_TQ = "#5cb8ff"; // cool = fast
const COLOR_FAST = "#62e58a";
const FONT_MONO = `${GEIST_MONO}, ui-monospace, monospace`;
const FONT_SANS = `${GEIST}, ui-sans-serif, system-ui, sans-serif`;

const Pane: React.FC<{
  title: string;
  accent: string;
  events: Event[];
  visible: number;
  totalMs: number;
  elapsedMs: number;
  finished: boolean;
  finishedAtFrame: number | null;
  frame: number;
  fps: number;
}> = ({
  title,
  accent,
  events,
  visible,
  totalMs,
  elapsedMs,
  finished,
  frame,
  fps,
}) => {
  const visibleEvents = events.slice(0, visible);
  const text = visibleEvents.map((e) => e.token).join("");
  const cleanText = text
    .replace(/<\|im_end\|>|<\|im_start\|>/g, "")
    .replace(/�/g, "")
    .trimEnd();
  // Always show the final (steady-state) tok/s — no live updates so the
  // numbers don't flicker.
  const tokps =
    events.length > 0 && totalMs > 0 ? events.length / (totalMs / 1000) : 0;

  return (
    <div
      style={{
        flex: "1 1 0",
        minWidth: 0,
        maxWidth: "100%",
        display: "flex",
        flexDirection: "column",
        fontFamily: FONT_MONO,
      }}
    >
      {/* header: BIG variant label + BIG tok/s */}
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          gap: 28,
          marginBottom: 24,
          flexWrap: "nowrap",
        }}
      >
        <span
          style={{
            color: accent,
            fontSize: 64,
            fontWeight: 700,
            letterSpacing: -1.5,
            lineHeight: 1,
            fontFamily: FONT_SANS,
          }}
        >
          {title}
        </span>
        <span
          style={{
            color: COLOR_TEXT,
            fontVariantNumeric: "tabular-nums",
            fontSize: 64,
            fontWeight: 700,
            lineHeight: 1,
            fontFamily: FONT_SANS,
            letterSpacing: -1,
          }}
        >
          {tokps.toFixed(1)}
          <span
            style={{
              color: COLOR_DIM,
              fontSize: 28,
              fontWeight: 500,
              marginLeft: 10,
              letterSpacing: 0,
            }}
          >
            tok/s
          </span>
        </span>
      </div>

      {/* terminal-style box: full size, doesn't grow with text */}
      <div
        style={{
          flex: 1,
          minHeight: 0,
          background: "#0d0d0d",
          border: `1px solid #1a1a1a`,
          borderRadius: 10,
          boxShadow: "inset 0 0 0 1px #000",
          padding: "20px 24px",
          color: COLOR_TEXT,
          fontFamily: FONT_MONO,
          fontSize: 18,
          fontWeight: 400,
          lineHeight: 1.55,
          letterSpacing: 0,
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
          overflow: "hidden",
          position: "relative",
        }}
      >
        {/* neutral terminal dots */}
        <div
          style={{
            position: "absolute",
            top: 14,
            left: 16,
            display: "flex",
            gap: 8,
          }}
        >
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              style={{
                width: 10,
                height: 10,
                borderRadius: "50%",
                border: "1.5px solid #3a3a3a",
              }}
            />
          ))}
        </div>
        <div style={{ paddingTop: 18 }}>
          {cleanText}
          {!finished && (
            <span
              style={{
                display: "inline-block",
                width: 11,
                height: 22,
                marginLeft: 3,
                verticalAlign: "-3px",
                background: accent,
                opacity: Math.floor(frame / (fps / 4)) % 2 === 0 ? 1 : 0,
              }}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export const ComparisonComposition: React.FC<Props> = ({
  fp16Events,
  tqEvents,
  meta,
  speed,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const elapsedMs = (frame / fps) * 1000 * speed;

  const fp16Visible = visibleTokensAt(fp16Events, elapsedMs).length;
  const tqVisible = visibleTokensAt(tqEvents, elapsedMs).length;
  const fp16Done = fp16Visible >= fp16Events.length;
  const tqDone = tqVisible >= tqEvents.length;

  const tqDoneFrame = tqDone
    ? Math.ceil((tqEvents[tqEvents.length - 1].t_ms / 1000) * fps / speed)
    : null;
  const fp16DoneFrame = fp16Done
    ? Math.ceil((fp16Events[fp16Events.length - 1].t_ms / 1000) * fps / speed)
    : null;

  return (
    <AbsoluteFill
      style={{
        background: COLOR_BG,
        color: COLOR_TEXT,
        fontFamily: FONT_MONO,
        padding: "56px 72px",
      }}
    >
      {/* Top: model name + runtime */}
      <div style={{ marginBottom: 48, fontFamily: FONT_SANS }}>
        <div
          style={{
            color: COLOR_TEXT,
            fontSize: 88,
            fontWeight: 700,
            letterSpacing: -2,
            lineHeight: 1.02,
          }}
        >
          {meta.model}
        </div>
        <div
          style={{
            color: COLOR_DIM,
            fontSize: 24,
            marginTop: 14,
            letterSpacing: 0,
            fontFamily: FONT_MONO,
          }}
        >
          {meta.runtime ?? "ONNX Runtime · WebGPU · Node.js"} ·{" "}
          {meta.hardware ?? "Apple Silicon"}
        </div>
      </div>

      {/* Two columns — no borders, equal width, terminal-style */}
      <div
        style={{
          flex: 1,
          display: "flex",
          gap: 72,
          minHeight: 0,
        }}
      >
        <Pane
          title="fp16"
          accent={COLOR_FP16}
          events={fp16Events}
          visible={fp16Visible}
          totalMs={meta.fp16.totalDecodeMs}
          elapsedMs={elapsedMs}
          finished={fp16Done}
          finishedAtFrame={fp16DoneFrame}
          frame={frame}
          fps={fps}
        />
        <Pane
          title="TurboQuant"
          accent={COLOR_TQ}
          events={tqEvents}
          visible={tqVisible}
          totalMs={meta.tq.totalDecodeMs}
          elapsedMs={elapsedMs}
          finished={tqDone}
          finishedAtFrame={tqDoneFrame}
          frame={frame}
          fps={fps}
        />
      </div>

      {/* Persistent speedup label — visible from frame 0, floating over right column */}
      <div
        style={{
          position: "absolute",
          top: "50%",
          left: "75%",
          transform: "translate(-50%, -50%)",
          background: "#0e2a16",
          border: `2px solid ${COLOR_FAST}`,
          padding: "28px 56px",
          fontFamily: FONT_SANS,
          color: COLOR_FAST,
          fontWeight: 700,
          fontSize: 96,
          letterSpacing: -2,
          lineHeight: 1,
          whiteSpace: "nowrap",
          fontVariantNumeric: "tabular-nums",
          zIndex: 20,
          pointerEvents: "none",
        }}
      >
        {meta.decode_speedup.toFixed(2)}× faster
      </div>
    </AbsoluteFill>
  );
};
