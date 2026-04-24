"use client";

import { useEffect, useRef } from "react";
import Image from "next/image";

// ── CRT overlay (vignette + chromatic aberration only) ──
function CRTOverlay() {
  return (
    <div className="absolute inset-0 z-20 pointer-events-none">
      {/* Vignette */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.7) 100%)",
        }}
      />
      {/* Chromatic aberration edges */}
      <div
        className="absolute inset-0 opacity-[0.03]"
        style={{
          boxShadow:
            "inset 3px 0 8px rgba(255,0,0,0.3), inset -3px 0 8px rgba(0,0,255,0.3)",
        }}
      />
    </div>
  );
}

// ── Main Hero Component ──
export function CyberpunkHero({ children }: { children?: React.ReactNode }) {
  const videoRef = useRef<HTMLVideoElement>(null);

  // iOS Safari autoplay is finicky:
  //   - React's `muted` JSX prop doesn't reliably set the DOM `muted` property
  //     before iOS evaluates autoplay, so Safari blocks playback and overlays
  //     its native "tap to play" button.
  //   - Even when muted is set right, the first .play() call can reject if
  //     the media isn't fully ready yet.
  // Strategy: force the property, set the legacy `webkit-playsinline`
  // attribute, and retry .play() at three points: immediately on mount,
  // when the media signals `canplay`, and on the first user interaction
  // anywhere on the page as a last resort.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    video.muted = true;
    video.defaultMuted = true;
    video.playsInline = true;
    video.setAttribute("webkit-playsinline", "true");

    const tryPlay = () => {
      const attempt = video.play();
      if (attempt && typeof attempt.catch === "function") {
        attempt.catch(() => {
          // Will be retried on `canplay` and / or first user interaction.
        });
      }
    };

    tryPlay();
    video.addEventListener("canplay", tryPlay);

    const onInteraction = () => {
      tryPlay();
      window.removeEventListener("touchstart", onInteraction);
      window.removeEventListener("pointerdown", onInteraction);
    };
    window.addEventListener("touchstart", onInteraction, { once: true, passive: true });
    window.addEventListener("pointerdown", onInteraction, { once: true });

    return () => {
      video.removeEventListener("canplay", tryPlay);
      window.removeEventListener("touchstart", onInteraction);
      window.removeEventListener("pointerdown", onInteraction);
    };
  }, []);

  return (
    <section className="relative w-full h-screen overflow-hidden bg-black">
      {/* Video background */}
      <video
        ref={videoRef}
        autoPlay
        loop
        muted
        playsInline
        preload="auto"
        disableRemotePlayback
        // `webkit-playsinline` (legacy iOS Safari) is set via the ref effect
        // below so TS/React don't complain about unknown attributes.
        className="absolute inset-0 z-[1] w-full h-full object-cover"
        style={{ filter: "brightness(0.5) saturate(1.3)" }}
      >
        <source src="/assets/wandler_hero.mp4" type="video/mp4" />
      </video>

      {/* Dark overlay for readability */}
      <div className="absolute inset-0 bg-black/40 z-[2]" />

      {/* CRT effect overlay */}
      <CRTOverlay />

      {/* Bottom fade into next section */}
      <div className="absolute bottom-0 left-0 right-0 h-40 sm:h-56 md:h-72 bg-gradient-to-b from-transparent to-black z-[3] pointer-events-none" />

      {/* Centered content */}
      <div className="absolute inset-0 z-10 flex flex-col items-center justify-center">
        <div className="relative flex flex-col items-center w-full max-w-3xl mx-auto px-4">
          {/* Frame: backdrop filter on the video behind it (sits behind content, under brackets) */}
          <div className="absolute -top-8 -left-2 -right-2 -bottom-8 sm:-top-10 sm:-left-8 sm:-right-8 sm:-bottom-10 md:-top-12 md:-left-12 md:-right-12 md:-bottom-12 lg:-left-[141px] lg:-right-[141px] backdrop-blur-sm backdrop-brightness-75 backdrop-saturate-150 pointer-events-none z-0" />

          {/* Viewfinder brackets */}
          <div className="absolute -top-8 -left-2 sm:-top-10 sm:-left-8 md:-top-12 md:-left-12 lg:-left-[141px] w-6 h-6 border-t border-l border-white/20 pointer-events-none z-20" />
          <div className="absolute -top-8 -right-2 sm:-top-10 sm:-right-8 md:-top-12 md:-right-12 lg:-right-[141px] w-6 h-6 border-t border-r border-white/20 pointer-events-none z-20" />
          <div className="absolute -bottom-8 -left-2 sm:-bottom-10 sm:-left-8 md:-bottom-12 md:-left-12 lg:-left-[141px] w-6 h-6 border-b border-l border-white/20 pointer-events-none z-20" />
          <div className="absolute -bottom-8 -right-2 sm:-bottom-10 sm:-right-8 md:-bottom-12 md:-right-12 lg:-right-[141px] w-6 h-6 border-b border-r border-white/20 pointer-events-none z-20" />

          {/* Logo */}
          <Image
            src="/assets/wandler_logo_v5.svg"
            alt="wandler"
            width={700}
            height={180}
            className="relative z-10 w-[320px] sm:w-[400px] md:w-[600px] lg:w-[950px] max-w-none h-auto"
            priority
          />

          {/* Identity text */}
          <div className="relative z-10 flex flex-col items-center space-y-10 mt-6">
            <p className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl tracking-tight text-center">
              <a
                href="https://github.com/huggingface/transformers.js"
                target="_blank"
                rel="noopener noreferrer"
                className="font-bold text-primary hover:underline underline-offset-4 decoration-2"
              >
                transformers.js
              </a>{" "}
              <span className="text-white/80 font-light">inference server</span>
            </p>
            <div className="px-6 py-3 rounded-full border border-white/15 bg-black/40 backdrop-blur-sm">
              <p className="text-sm sm:text-base text-white/70 leading-relaxed text-left max-w-[340px]">
                run open-weight models on mac, linux &amp; win via an OpenAI-compatible api, built in ts
              </p>
            </div>
          </div>

          {/* Quickstart commands */}
          {children && (
            <div className="relative z-10 w-full sm:max-w-xl mx-auto space-y-10 mt-14">
              {children}
            </div>
          )}
        </div>
      </div>

    </section>
  );
}
