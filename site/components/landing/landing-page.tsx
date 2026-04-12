"use client";

import { Check, Code, Copy, Server, Zap, Shield } from "lucide-react";
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
				{/* Hero */}
				<section className="min-h-screen relative overflow-hidden flex flex-col justify-center space-y-12 px-4 md:px-0 my-12 md:my-4">
					<div className="relative z-10">
						<div className="container mx-auto">
							<div className="grid md:grid-cols-[3fr_1fr] gap-8 md:gap-12 items-center w-full">
								<div className="space-y-10 md:space-y-6">
									<h1 className="text-5xl md:text-8xl font-bold tracking-tighter mb-4">
										openai-compatible
										<br />
										<span className="text-primary">inference server</span>
									</h1>
									<p className="text-lg md:text-xl text-muted-foreground mb-4">
										powered by 🤗{" "}
										<a
											href="https://huggingface.co/docs/transformers.js/en/index"
											className="text-primary hover:underline"
										>
											transformers.js
										</a>
										{" "}— run ONNX models locally with WebGPU. no python, no CUDA.
									</p>

									<div className="flex flex-wrap items-center gap-3 md:gap-4 text-sm text-muted-foreground">
										<div className="flex items-center gap-2">
											<Zap className="w-4 h-4 text-primary" />
											up to 248 tok/s
										</div>
										<div className="flex items-center gap-2">
											<Server className="w-4 h-4 text-primary" />
											openai api compatible
										</div>
										<div className="flex items-center gap-2">
											<Shield className="w-4 h-4 text-primary" />
											local & private
										</div>
										<div className="flex items-center gap-2">
											<Code className="w-4 h-4 text-primary" />
											open source
										</div>
									</div>

									{/* Quickstart */}
									<div className="cyberpunk-corners bg-secondary p-3 md:p-4 w-full">
										<div className="flex items-center justify-between">
											<div className="flex items-center gap-2 overflow-x-auto">
												<button
													onClick={() => handleCopy(quickstart, "quickstart")}
													className="p-1 md:p-2 hover:text-primary transition-colors shrink-0"
													title="Copy to clipboard"
												>
													{copied === "quickstart" ? (
														<Check className="w-4 h-4" />
													) : (
														<Copy className="w-4 h-4" />
													)}
												</button>
												<code className="font-mono text-sm md:text-base whitespace-nowrap">
													{quickstart}
												</code>
											</div>
										</div>
									</div>
								</div>

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
					</div>
				</section>

				{/* Endpoints */}
				<section className="py-16 md:py-24 border-t border-primary/20">
					<div className="container mx-auto px-4">
						<h2 className="text-3xl md:text-5xl font-bold tracking-tighter mb-12">
							endpoints
						</h2>
						<div className="grid md:grid-cols-3 gap-6">
							{[
								{
									title: "Chat Completions",
									path: "/v1/chat/completions",
									desc: "Streaming & non-streaming chat with tool calling",
								},
								{
									title: "Text Completions",
									path: "/v1/completions",
									desc: "Legacy completions with echo & suffix",
								},
								{
									title: "Embeddings",
									path: "/v1/embeddings",
									desc: "Text embeddings for RAG and search",
								},
								{
									title: "Models",
									path: "/v1/models",
									desc: "List and inspect loaded models",
								},
								{
									title: "Audio Transcriptions",
									path: "/v1/audio/transcriptions",
									desc: "Speech-to-text via Whisper",
								},
								{
									title: "Tokenize / Detokenize",
									path: "/tokenize",
									desc: "Convert text to tokens and back",
								},
							].map((ep) => (
								<div
									key={ep.path}
									className="cyberpunk-corners bg-secondary/50 p-6 hover:bg-secondary/80 transition-colors"
								>
									<code className="text-primary text-sm font-mono">{ep.path}</code>
									<h3 className="text-lg font-bold mt-2">{ep.title}</h3>
									<p className="text-muted-foreground text-sm mt-1">{ep.desc}</p>
								</div>
							))}
						</div>
					</div>
				</section>

				{/* Code Example */}
				<section className="py-16 md:py-24 border-t border-primary/20">
					<div className="container mx-auto px-4">
						<h2 className="text-3xl md:text-5xl font-bold tracking-tighter mb-4">
							works with any openai sdk
						</h2>
						<p className="text-muted-foreground mb-8">
							drop-in replacement — just change the base URL
						</p>
						<div className="cyberpunk-corners bg-secondary p-4 md:p-6 relative">
							<button
								onClick={() => handleCopy(sdkExample, "sdk")}
								className="absolute top-4 right-4 p-2 hover:text-primary transition-colors"
								title="Copy code"
							>
								{copied === "sdk" ? (
									<Check className="w-4 h-4" />
								) : (
									<Copy className="w-4 h-4" />
								)}
							</button>
							<pre className="font-mono text-sm overflow-x-auto text-muted-foreground">
								<code>{sdkExample}</code>
							</pre>
						</div>
					</div>
				</section>

				{/* Benchmarks */}
				<section className="py-16 md:py-24 border-t border-primary/20">
					<div className="container mx-auto px-4">
						<h2 className="text-3xl md:text-5xl font-bold tracking-tighter mb-4">
							benchmarks
						</h2>
						<p className="text-muted-foreground mb-8">
							WebGPU, q4 quantization, 10 runs per scenario
						</p>
						<div className="overflow-x-auto">
							<table className="w-full text-left">
								<thead>
									<tr className="border-b border-primary/20">
										<th className="py-3 pr-4 text-primary font-mono text-sm">Model</th>
										<th className="py-3 pr-4 text-primary font-mono text-sm">tok/s</th>
										<th className="py-3 pr-4 text-primary font-mono text-sm">TTFT</th>
										<th className="py-3 pr-4 text-primary font-mono text-sm">Load</th>
										<th className="py-3 text-primary font-mono text-sm">Tools</th>
									</tr>
								</thead>
								<tbody className="font-mono text-sm">
									{[
										{ model: "LFM2.5-350M", tps: "248", ttft: "16ms", load: "0.5s", tools: "-" },
										{ model: "LFM2.5-1.2B", tps: "118", ttft: "34ms", load: "1.7s", tools: "yes" },
										{ model: "Qwen3.5-0.8B", tps: "37", ttft: "276ms", load: "1.8s", tools: "partial" },
										{ model: "Gemma 4 E4B", tps: "20", ttft: "636ms", load: "13.4s", tools: "-" },
									].map((row) => (
										<tr key={row.model} className="border-b border-primary/10">
											<td className="py-3 pr-4">{row.model}</td>
											<td className="py-3 pr-4 text-primary">{row.tps}</td>
											<td className="py-3 pr-4">{row.ttft}</td>
											<td className="py-3 pr-4">{row.load}</td>
											<td className="py-3">{row.tools}</td>
										</tr>
									))}
								</tbody>
							</table>
						</div>
					</div>
				</section>

				{/* Features */}
				<section className="py-16 md:py-24 border-t border-primary/20">
					<div className="container mx-auto px-4">
						<div className="grid md:grid-cols-2 gap-8">
							<div>
								<h2 className="text-3xl md:text-5xl font-bold tracking-tighter mb-8">
									features
								</h2>
								<ul className="space-y-4 text-muted-foreground">
									{[
										"SSE streaming with real-time token generation",
										"Multi-format tool calling (LFM, Qwen, OpenAI JSON)",
										"Quantized inference: q4, q8, fp16, fp32",
										"WebGPU acceleration + CPU fallback",
										"Text embeddings for RAG workflows",
										"Speech-to-text via Whisper",
										"API key authentication",
										"Admin metrics endpoint",
										"Built on Hono — fast, lightweight, composable",
									].map((f) => (
										<li key={f} className="flex items-start gap-3">
											<Check className="w-4 h-4 text-primary mt-1 shrink-0" />
											{f}
										</li>
									))}
								</ul>
							</div>
							<div>
								<h2 className="text-3xl md:text-5xl font-bold tracking-tighter mb-8">
									compatible with
								</h2>
								<ul className="space-y-4 text-muted-foreground">
									{[
										"OpenAI SDK (Python & TypeScript)",
										"Vercel AI SDK",
										"LangChain",
										"LlamaIndex",
										"Any OpenAI-compatible client",
									].map((c) => (
										<li key={c} className="flex items-start gap-3">
											<Check className="w-4 h-4 text-primary mt-1 shrink-0" />
											{c}
										</li>
									))}
								</ul>
							</div>
						</div>
					</div>
				</section>
			</main>

			<footer className="py-6 md:py-8 border-t border-primary/20">
				<div className="container mx-auto px-4 flex justify-between items-center text-muted-foreground text-sm">
					<p>wandler — inference for the typescript ecosystem</p>
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
