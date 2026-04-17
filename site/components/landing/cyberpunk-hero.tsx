"use client";

import { useEffect, useState } from "react";
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
  const [loaded, setLoaded] = useState(false);

  // Fade in
  useEffect(() => {
    const timer = setTimeout(() => setLoaded(true), 100);
    return () => clearTimeout(timer);
  }, []);

  return (
    <section
      className={`relative w-full h-screen overflow-hidden bg-black transition-opacity duration-1000 ${loaded ? "opacity-100" : "opacity-0"}`}
    >
      {/* Video background */}
      <video
        autoPlay
        loop
        muted
        playsInline
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
            className="w-[260px] sm:w-[400px] md:w-[600px] lg:w-[950px] max-w-none h-auto"
            priority
          />

          {/* Identity text */}
          <div className="text-center space-y-6 mt-6">
            <p className="text-xl sm:text-2xl md:text-3xl lg:text-4xl tracking-tight">
              <a
                href="https://github.com/huggingface/transformers.js"
                target="_blank"
                rel="noopener noreferrer"
                className="font-bold text-primary hover:underline underline-offset-4 decoration-2"
                style={{
                  textShadow: "0 0 20px rgba(255,236,25,0.55)",
                }}
              >
                transformers.js
              </a>{" "}
              <span className="text-white/80 font-light">inference server</span>
            </p>
            <p className="text-xs md:text-sm text-white/30 tracking-[0.2em] uppercase font-mono">
              OpenAI-compatible API{" \u00b7 "}Mac, Linux & Windows
            </p>
          </div>

          {/* Quickstart commands */}
          {children && (
            <div className="w-full space-y-10 mt-14">
              {children}
            </div>
          )}
        </div>
      </div>

    </section>
  );
}
