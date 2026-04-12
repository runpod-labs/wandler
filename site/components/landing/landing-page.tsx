"use client";

import { Check, Copy, Terminal } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { useState } from "react";

import { Header } from "@/components/header";

export function LandingPage() {
	const [copied, setCopied] = useState<string | null>(null);

	const handleCopy = async (text: string, id: string) => {
		await navigator.clipboard.writeText(text);
		setCopied(id);
		setTimeout(() => setCopied(null), 2000);
	};

	const quickstart = "npx wandler --llm LiquidAI/LFM2.5-1.2B-Instruct-ONNX";

	const sdkExample = `import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8000/v1",
  apiKey: "-",
});

const res = await client.chat.completions.create({
  model: "LiquidAI/LFM2.5-1.2B-Instruct-ONNX",
  messages: [{ role: "user", content: "Hello!" }],
  stream: true,
});

for await (const chunk of res) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
}`;

	return (
		<div className="flex flex-col min-h-screen bg-black text-white">
			<Header />

			<main className="flex-grow">
				{/* ── Hero ── */}
				<section className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden px-4">
					{/* Robot head as atmospheric background */}
					<div className="absolute inset-0 flex items-center justify-center pointer-events-none select-none">
						<div className="relative w-[600px] h-[600px] md:w-[900px] md:h-[900px] opacity-[0.08]">
							<Image
								src="https://5xvkmufwzznj1ey2.public.blob.vercel-storage.com/20250202_wandler_head_v2-Ma4f25yqpXRSf32ZnmivUnnV1LGQ69.jpg"
								alt=""
								fill
								className="object-contain"
								priority
							/>
						</div>
					</div>

					{/* Subtle scan line overlay */}
					<div className="absolute inset-0 pointer-events-none opacity-[0.03]" style={{
						backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.03) 2px, rgba(255,255,255,0.03) 4px)",
					}} />

					<div className="relative z-10 flex flex-col items-center text-center max-w-4xl mx-auto space-y-10">
						{/* Large centered logo */}
						<div className="w-[320px] md:w-[560px] lg:w-[700px]">
							<Image
								src="https://5xvkmufwzznj1ey2.public.blob.vercel-storage.com/wandler_logo_v5-vJ2L3NmauebkFJs9fOcFe7bPVM14To.svg"
								alt="wandler"
								width={700}
								height={140}
								className="w-full h-auto"
								priority
							/>
						</div>

						{/* Core proposition — no borders, just typography */}
						<p className="text-xl md:text-2xl lg:text-3xl font-light tracking-tight text-white/80">
							openai-compatible inference server
						</p>

						{/* Key facts as clean inline items */}
						<div className="flex flex-wrap justify-center gap-x-8 gap-y-3 text-sm md:text-base text-white/40 font-mono">
							<span>powered by <span className="text-primary/70">transformers.js</span></span>
							<span>local inference</span>
							<span>webgpu accelerated</span>
							<span>no python</span>
						</div>

						{/* Quickstart command */}
						<div className="w-full max-w-2xl mt-4">
							<button
								onClick={() => handleCopy(quickstart, "qs")}
								className="group w-full flex items-center gap-3 bg-white/[0.04] hover:bg-white/[0.07] border border-white/[0.06] hover:border-primary/20 transition-all px-4 py-3 text-left cursor-pointer"
							>
								<Terminal className="w-4 h-4 text-primary/60 shrink-0" />
								<code className="font-mono text-sm md:text-base text-white/60 group-hover:text-white/80 transition-colors truncate">
									{quickstart}
								</code>
								<span className="ml-auto shrink-0">
									{copied === "qs" ? (
										<Check className="w-4 h-4 text-primary" />
									) : (
										<Copy className="w-4 h-4 text-white/20 group-hover:text-white/40 transition-colors" />
									)}
								</span>
							</button>
						</div>
					</div>
				</section>

				{/* ── What it does ── */}
				<section className="py-24 md:py-32 border-t border-white/[0.06]">
					<div className="container mx-auto px-4 max-w-5xl">
						<div className="grid md:grid-cols-3 gap-16 md:gap-12">
							<div>
								<div className="text-primary/50 font-mono text-xs tracking-widest uppercase mb-4">speed</div>
								<div className="text-4xl md:text-5xl font-bold tracking-tighter">248</div>
								<div className="text-white/40 mt-1">tokens per second</div>
								<div className="text-white/20 text-sm mt-2 font-mono">LFM2.5-350M · WebGPU · q4</div>
							</div>
							<div>
								<div className="text-primary/50 font-mono text-xs tracking-widest uppercase mb-4">latency</div>
								<div className="text-4xl md:text-5xl font-bold tracking-tighter">16<span className="text-lg text-white/40 ml-1">ms</span></div>
								<div className="text-white/40 mt-1">time to first token</div>
								<div className="text-white/20 text-sm mt-2 font-mono">streaming · SSE</div>
							</div>
							<div>
								<div className="text-primary/50 font-mono text-xs tracking-widest uppercase mb-4">models</div>
								<div className="text-4xl md:text-5xl font-bold tracking-tighter">2900+</div>
								<div className="text-white/40 mt-1">ONNX models on HuggingFace</div>
								<div className="text-white/20 text-sm mt-2 font-mono">transformers.js compatible</div>
							</div>
						</div>
					</div>
				</section>

				{/* ── Endpoints ── */}
				<section className="py-24 md:py-32 border-t border-white/[0.06]">
					<div className="container mx-auto px-4 max-w-5xl">
						<h2 className="text-3xl md:text-4xl font-bold tracking-tighter mb-16">
							endpoints
						</h2>
						<div className="grid md:grid-cols-2 gap-x-16 gap-y-10">
							{[
								{ path: "POST /v1/chat/completions", desc: "Streaming & non-streaming chat with tool calling" },
								{ path: "POST /v1/completions", desc: "Legacy text completions with echo & suffix" },
								{ path: "POST /v1/embeddings", desc: "Text embeddings for RAG and semantic search" },
								{ path: "GET  /v1/models", desc: "List and inspect loaded models" },
								{ path: "POST /v1/audio/transcriptions", desc: "Speech-to-text via Whisper" },
								{ path: "POST /tokenize", desc: "Convert between text and token IDs" },
							].map((ep) => (
								<div key={ep.path} className="group">
									<code className="text-primary/70 text-sm font-mono">{ep.path}</code>
									<p className="text-white/40 text-sm mt-1">{ep.desc}</p>
								</div>
							))}
						</div>
					</div>
				</section>

				{/* ── Code ── */}
				<section className="py-24 md:py-32 border-t border-white/[0.06]">
					<div className="container mx-auto px-4 max-w-5xl">
						<h2 className="text-3xl md:text-4xl font-bold tracking-tighter mb-3">
							works with any openai sdk
						</h2>
						<p className="text-white/40 mb-10">
							drop-in replacement — change the base URL and go
						</p>
						<div className="relative bg-white/[0.02] border border-white/[0.06] p-5 md:p-8">
							<button
								onClick={() => handleCopy(sdkExample, "sdk")}
								className="absolute top-4 right-4 p-2 text-white/20 hover:text-primary transition-colors"
								title="Copy code"
							>
								{copied === "sdk" ? (
									<Check className="w-4 h-4" />
								) : (
									<Copy className="w-4 h-4" />
								)}
							</button>
							<pre className="font-mono text-sm leading-relaxed overflow-x-auto text-white/50">
								<code>{sdkExample}</code>
							</pre>
						</div>
					</div>
				</section>

				{/* ── Benchmarks ── */}
				<section className="py-24 md:py-32 border-t border-white/[0.06]">
					<div className="container mx-auto px-4 max-w-5xl">
						<h2 className="text-3xl md:text-4xl font-bold tracking-tighter mb-3">
							benchmarks
						</h2>
						<p className="text-white/40 mb-10 font-mono text-sm">
							WebGPU · q4 · 10 runs per scenario
						</p>
						<div className="overflow-x-auto">
							<table className="w-full text-left">
								<thead>
									<tr className="border-b border-white/[0.08]">
										<th className="py-3 pr-6 text-white/30 font-mono text-xs tracking-widest uppercase">Model</th>
										<th className="py-3 pr-6 text-white/30 font-mono text-xs tracking-widest uppercase">tok/s</th>
										<th className="py-3 pr-6 text-white/30 font-mono text-xs tracking-widest uppercase">TTFT</th>
										<th className="py-3 pr-6 text-white/30 font-mono text-xs tracking-widest uppercase">Load</th>
										<th className="py-3 text-white/30 font-mono text-xs tracking-widest uppercase">Tools</th>
									</tr>
								</thead>
								<tbody className="font-mono text-sm">
									{[
										{ model: "LFM2.5-350M", tps: "248", ttft: "16ms", load: "0.5s", tools: "—" },
										{ model: "LFM2.5-1.2B", tps: "118", ttft: "34ms", load: "1.7s", tools: "yes", highlight: true },
										{ model: "Qwen3.5-0.8B", tps: "37", ttft: "276ms", load: "1.8s", tools: "partial" },
										{ model: "Gemma 4 E4B", tps: "20", ttft: "636ms", load: "13.4s", tools: "—" },
									].map((row) => (
										<tr key={row.model} className="border-b border-white/[0.04]">
											<td className={`py-3 pr-6 ${row.highlight ? "text-white" : "text-white/60"}`}>{row.model}</td>
											<td className={`py-3 pr-6 ${row.highlight ? "text-primary" : "text-primary/60"}`}>{row.tps}</td>
											<td className="py-3 pr-6 text-white/40">{row.ttft}</td>
											<td className="py-3 pr-6 text-white/40">{row.load}</td>
											<td className="py-3 text-white/40">{row.tools}</td>
										</tr>
									))}
								</tbody>
							</table>
						</div>
					</div>
				</section>

				{/* ── Features ── */}
				<section className="py-24 md:py-32 border-t border-white/[0.06]">
					<div className="container mx-auto px-4 max-w-5xl">
						<div className="grid md:grid-cols-2 gap-20">
							<div>
								<h2 className="text-3xl md:text-4xl font-bold tracking-tighter mb-10">
									features
								</h2>
								<ul className="space-y-4">
									{[
										"SSE streaming with real-time token generation",
										"Multi-format tool calling (LFM, Qwen, OpenAI)",
										"Quantized inference: q4, q8, fp16, fp32",
										"WebGPU acceleration with CPU fallback",
										"Text embeddings for RAG workflows",
										"Speech-to-text via Whisper",
										"API key authentication",
										"Admin metrics endpoint",
									].map((f) => (
										<li key={f} className="flex items-start gap-3 text-white/40 text-sm">
											<span className="text-primary/40 mt-0.5">—</span>
											{f}
										</li>
									))}
								</ul>
							</div>
							<div>
								<h2 className="text-3xl md:text-4xl font-bold tracking-tighter mb-10">
									compatible with
								</h2>
								<ul className="space-y-4">
									{[
										"OpenAI SDK (Python & TypeScript)",
										"Vercel AI SDK",
										"LangChain",
										"LlamaIndex",
										"Any OpenAI-compatible client",
									].map((c) => (
										<li key={c} className="flex items-start gap-3 text-white/40 text-sm">
											<span className="text-primary/40 mt-0.5">—</span>
											{c}
										</li>
									))}
								</ul>
							</div>
						</div>
					</div>
				</section>
			</main>

			<footer className="py-8 border-t border-white/[0.06]">
				<div className="container mx-auto px-4 max-w-5xl flex justify-between items-center text-white/20 text-sm font-mono">
					<span>wandler</span>
					<Link
						href="https://github.com/runpod-labs/wandler"
						className="hover:text-primary/60 transition-colors"
						target="_blank"
					>
						github
					</Link>
				</div>
			</footer>
		</div>
	);
}
