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

			<main className="flex-grow pt-16">
				{/* ── Hero ── */}
				<section className="min-h-screen relative overflow-hidden flex flex-col justify-center px-4 md:px-0 py-12">
					<div className="container mx-auto">
						<div className="grid md:grid-cols-[3fr_2fr] gap-8 md:gap-12 items-center">
							{/* Left: Logo + Info */}
							<div className="space-y-8">
								<Image
									src="https://5xvkmufwzznj1ey2.public.blob.vercel-storage.com/wandler_logo_v5-vJ2L3NmauebkFJs9fOcFe7bPVM14To.svg"
									alt="wandler"
									width={500}
									height={100}
									className="w-[320px] md:w-[500px] h-auto"
									priority
								/>

								<h1 className="text-3xl md:text-5xl font-bold tracking-tighter">
									openai-compatible{" "}
									<span className="text-primary">inference server</span>
								</h1>

								<p className="text-lg text-muted-foreground">
									powered by 🤗{" "}
									<a
										href="https://huggingface.co/docs/transformers.js/en/index"
										className="text-primary hover:underline"
									>
										transformers.js
									</a>
									{" "}— run ONNX models locally with WebGPU. no python, no CUDA.
								</p>

								{/* Key facts in cyberpunk corner cards */}
								<div className="grid grid-cols-2 md:grid-cols-4 gap-4">
									{[
										{ label: "up to", value: "248 tok/s" },
										{ label: "TTFT", value: "16ms" },
										{ label: "models", value: "2900+" },
										{ label: "quantization", value: "q4 / q8" },
									].map((stat) => (
										<div key={stat.value} className="cyberpunk-corners bg-secondary p-4">
											<div className="text-xs text-muted-foreground uppercase tracking-wider">{stat.label}</div>
											<div className="text-xl font-bold text-primary mt-1">{stat.value}</div>
										</div>
									))}
								</div>

								{/* Quickstart */}
								<div className="cyberpunk-corners bg-secondary p-4">
									<button
										onClick={() => handleCopy(quickstart, "qs")}
										className="w-full flex items-center gap-3 text-left cursor-pointer group"
									>
										<Terminal className="w-4 h-4 text-primary shrink-0" />
										<code className="font-mono text-sm md:text-base text-white truncate">
											{quickstart}
										</code>
										<span className="ml-auto shrink-0">
											{copied === "qs" ? (
												<Check className="w-4 h-4 text-primary" />
											) : (
												<Copy className="w-4 h-4 text-muted-foreground group-hover:text-primary transition-colors" />
											)}
										</span>
									</button>
								</div>
							</div>

							{/* Right: Robot head */}
							<div className="relative flex justify-center mt-8 md:mt-0">
								<div className="w-[280px] h-[280px] md:w-[500px] md:h-[500px] relative">
									<Image
										src="https://5xvkmufwzznj1ey2.public.blob.vercel-storage.com/20250202_wandler_head_v2-Ma4f25yqpXRSf32ZnmivUnnV1LGQ69.jpg"
										alt="wandler"
										fill
										className="object-contain"
										priority
									/>
									<div className="absolute inset-0 bg-gradient-to-r from-black via-transparent to-transparent" />
								</div>
							</div>
						</div>
					</div>
				</section>

				{/* ── Endpoints ── */}
				<section className="py-20 md:py-28 border-t border-primary/20">
					<div className="container mx-auto px-4">
						<h2 className="text-3xl md:text-5xl font-bold tracking-tighter mb-12">
							endpoints
						</h2>
						<div className="grid md:grid-cols-3 gap-6">
							{[
								{ path: "/v1/chat/completions", title: "Chat Completions", desc: "Streaming & non-streaming chat with tool calling" },
								{ path: "/v1/completions", title: "Text Completions", desc: "Legacy completions with echo & suffix" },
								{ path: "/v1/embeddings", title: "Embeddings", desc: "Text embeddings for RAG and semantic search" },
								{ path: "/v1/models", title: "Models", desc: "List and inspect loaded models" },
								{ path: "/v1/audio/transcriptions", title: "Audio Transcriptions", desc: "Speech-to-text via Whisper" },
								{ path: "/tokenize", title: "Tokenize / Detokenize", desc: "Convert between text and token IDs" },
							].map((ep) => (
								<div key={ep.path} className="cyberpunk-corners bg-secondary p-6">
									<code className="text-primary text-sm font-mono">{ep.path}</code>
									<h3 className="text-lg font-bold mt-2">{ep.title}</h3>
									<p className="text-muted-foreground text-sm mt-1">{ep.desc}</p>
								</div>
							))}
						</div>
					</div>
				</section>

				{/* ── Code Example ── */}
				<section className="py-20 md:py-28 border-t border-primary/20">
					<div className="container mx-auto px-4">
						<h2 className="text-3xl md:text-5xl font-bold tracking-tighter mb-4">
							works with any openai sdk
						</h2>
						<p className="text-muted-foreground mb-8">
							drop-in replacement — change the base URL and go
						</p>
						<div className="cyberpunk-corners bg-secondary p-5 md:p-8 relative">
							<button
								onClick={() => handleCopy(sdkExample, "sdk")}
								className="absolute top-4 right-4 p-2 text-muted-foreground hover:text-primary transition-colors"
								title="Copy code"
							>
								{copied === "sdk" ? (
									<Check className="w-4 h-4" />
								) : (
									<Copy className="w-4 h-4" />
								)}
							</button>
							<pre className="font-mono text-sm leading-relaxed overflow-x-auto text-white">
								<code>{sdkExample}</code>
							</pre>
						</div>
					</div>
				</section>

				{/* ── Benchmarks ── */}
				<section className="py-20 md:py-28 border-t border-primary/20">
					<div className="container mx-auto px-4">
						<h2 className="text-3xl md:text-5xl font-bold tracking-tighter mb-4">
							benchmarks
						</h2>
						<p className="text-muted-foreground mb-8 font-mono text-sm">
							WebGPU · q4 quantization · 10 runs per scenario
						</p>
						<div className="overflow-x-auto">
							<table className="w-full text-left">
								<thead>
									<tr className="border-b border-primary/30">
										<th className="py-3 pr-6 text-primary font-mono text-sm">Model</th>
										<th className="py-3 pr-6 text-primary font-mono text-sm">tok/s</th>
										<th className="py-3 pr-6 text-primary font-mono text-sm">TTFT</th>
										<th className="py-3 pr-6 text-primary font-mono text-sm">Load</th>
										<th className="py-3 text-primary font-mono text-sm">Tools</th>
									</tr>
								</thead>
								<tbody className="font-mono text-sm">
									{[
										{ model: "LFM2.5-350M", tps: "248", ttft: "16ms", load: "0.5s", tools: "—" },
										{ model: "LFM2.5-1.2B", tps: "118", ttft: "34ms", load: "1.7s", tools: "yes", highlight: true },
										{ model: "Qwen3.5-0.8B", tps: "37", ttft: "276ms", load: "1.8s", tools: "partial" },
										{ model: "Gemma 4 E4B", tps: "20", ttft: "636ms", load: "13.4s", tools: "—" },
									].map((row) => (
										<tr key={row.model} className={`border-b border-primary/10 ${row.highlight ? "bg-primary/5" : ""}`}>
											<td className="py-3 pr-6 text-white">{row.model}</td>
											<td className={`py-3 pr-6 font-bold ${row.highlight ? "text-primary" : "text-white"}`}>{row.tps}</td>
											<td className="py-3 pr-6 text-white">{row.ttft}</td>
											<td className="py-3 pr-6 text-white">{row.load}</td>
											<td className="py-3 text-white">{row.tools}</td>
										</tr>
									))}
								</tbody>
							</table>
						</div>
					</div>
				</section>

				{/* ── Features ── */}
				<section className="py-20 md:py-28 border-t border-primary/20">
					<div className="container mx-auto px-4">
						<div className="grid md:grid-cols-2 gap-12">
							<div>
								<h2 className="text-3xl md:text-5xl font-bold tracking-tighter mb-10">
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
										<li key={f} className="flex items-start gap-3 text-white text-sm">
											<span className="text-primary mt-0.5">▸</span>
											{f}
										</li>
									))}
								</ul>
							</div>
							<div>
								<h2 className="text-3xl md:text-5xl font-bold tracking-tighter mb-10">
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
										<li key={c} className="flex items-start gap-3 text-white text-sm">
											<span className="text-primary mt-0.5">▸</span>
											{c}
										</li>
									))}
								</ul>
							</div>
						</div>
					</div>
				</section>
			</main>

			<footer className="py-8 border-t border-primary/20">
				<div className="container mx-auto px-4 flex justify-between items-center text-muted-foreground text-sm">
					<span>wandler — inference for the typescript ecosystem</span>
					<Link
						href="https://github.com/runpod-labs/wandler"
						className="text-primary hover:underline"
						target="_blank"
					>
						github
					</Link>
				</div>
			</footer>
		</div>
	);
}
