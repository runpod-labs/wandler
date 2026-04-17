"use client";

import { toPng } from "html-to-image";
import Image from "next/image";
import { useSearchParams } from "next/navigation";
import { Suspense, useRef, useState } from "react";

function BannerCanvas() {
	return (
		<>
			{/* Background still (first frame of hero video) */}
			<Image
				src="/assets/final_videos/01_street_source.png"
				alt=""
				fill
				priority
				sizes="1200px"
				style={{
					objectFit: "cover",
					filter: "brightness(0.5) saturate(1.3)",
				}}
			/>

			{/* Dark readability overlay */}
			<div className="absolute inset-0 bg-black/40" />

			{/* CRT-style vignette */}
			<div
				className="absolute inset-0"
				style={{
					background:
						"radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.7) 100%)",
				}}
			/>

			{/* Chromatic aberration */}
			<div
				className="absolute inset-0"
				style={{
					opacity: 0.03,
					boxShadow:
						"inset 3px 0 8px rgba(255,0,0,0.3), inset -3px 0 8px rgba(0,0,255,0.3)",
				}}
			/>

			{/* Bottom fade */}
			<div
				className="absolute bottom-0 left-0 right-0"
				style={{
					height: 200,
					background:
						"linear-gradient(to bottom, rgba(0,0,0,0) 0%, rgba(0,0,0,1) 100%)",
				}}
			/>

			{/* Viewfinder brackets */}
			<div className="absolute top-12 left-12 w-8 h-8 border-t border-l border-white/30" />
			<div className="absolute top-12 right-12 w-8 h-8 border-t border-r border-white/30" />
			<div className="absolute bottom-12 left-12 w-8 h-8 border-b border-l border-white/30" />
			<div className="absolute bottom-12 right-12 w-8 h-8 border-b border-r border-white/30" />

			{/* Centered content */}
			<div
				className="absolute inset-0 flex flex-col items-center justify-center"
				style={{ paddingLeft: 80, paddingRight: 80 }}
			>
				<Image
					src="/assets/wandler_logo_v5.svg"
					alt="wandler"
					width={700}
					height={180}
					priority
					style={{ width: 760, height: "auto" }}
				/>

				<div className="text-center" style={{ marginTop: 16 }}>
					<p
						style={{
							fontSize: 44,
							lineHeight: 1.15,
							letterSpacing: "-0.02em",
						}}
					>
						<span style={{ color: "#ffec19", fontWeight: 700 }}>transformers.js</span>
						<span style={{ color: "rgba(255,255,255,0.85)", fontWeight: 300 }}>
							{" "}
							inference server
						</span>
					</p>
					<p
						style={{
							color: "rgba(255,255,255,0.4)",
							fontSize: 16,
							letterSpacing: "0.25em",
							textTransform: "uppercase",
							fontFamily:
								"ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
							marginTop: 20,
						}}
					>
						OpenAI-compatible API · Mac, Linux & Windows
					</p>
				</div>
			</div>
		</>
	);
}

function BannerContent() {
	const searchParams = useSearchParams();
	const raw = searchParams.get("raw") === "1";
	const bannerRef = useRef<HTMLDivElement>(null);
	const [downloading, setDownloading] = useState(false);

	if (raw) {
		return (
			<div
				style={{ width: 1200, height: 630 }}
				className="relative overflow-hidden bg-black"
			>
				<BannerCanvas />
			</div>
		);
	}

	const handleDownload = async () => {
		if (!bannerRef.current) return;
		setDownloading(true);
		try {
			const dataUrl = await toPng(bannerRef.current, {
				width: 1200,
				height: 630,
				pixelRatio: 2,
				cacheBust: true,
			});
			const link = document.createElement("a");
			link.download = "wandler-og.png";
			link.href = dataUrl;
			link.click();
		} catch (err) {
			console.error("Banner export failed:", err);
		} finally {
			setDownloading(false);
		}
	};

	return (
		<div className="min-h-screen bg-[#0a0a0a] flex flex-col items-center py-16 gap-6">
			<div className="text-center">
				<h1 className="text-white text-2xl font-bold tracking-tight mb-1">banner preview</h1>
				<p className="text-white/40 font-mono text-xs">1200 × 630 · og image</p>
			</div>

			<div
				ref={bannerRef}
				style={{ width: 1200, height: 630 }}
				className="relative overflow-hidden shrink-0 bg-black"
			>
				<BannerCanvas />
			</div>

			<div className="flex items-center gap-3 mt-2">
				<button
					type="button"
					onClick={handleDownload}
					disabled={downloading}
					className="px-5 py-2.5 bg-primary text-black font-bold text-sm tracking-tight hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
				>
					{downloading ? "generating…" : "download PNG (1200×630 @ 2x)"}
				</button>
			</div>

			<p className="text-white/30 text-xs max-w-xl text-center mt-4 leading-relaxed">
				edit <code className="font-mono text-white/50">site/app/banner/page.tsx</code> and
				refresh. once it looks right, click download and save the file as{" "}
				<code className="font-mono text-white/50">site/public/og.png</code>.
			</p>
		</div>
	);
}

export default function BannerPage() {
	return (
		<Suspense fallback={null}>
			<BannerContent />
		</Suspense>
	);
}
