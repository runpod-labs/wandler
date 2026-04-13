"use client";

import { Check, Copy, Terminal } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { useEffect, useState } from "react";
import { createHighlighter, type Highlighter } from "shiki";

import { Header } from "@/components/header";

let highlighterPromise: Promise<Highlighter> | null = null;
function getHighlighter() {
	if (!highlighterPromise) {
		highlighterPromise = createHighlighter({
			themes: ["github-dark"],
			langs: ["typescript", "python", "bash"],
		});
	}
	return highlighterPromise;
}

function useHighlightedCode(code: string, lang: string) {
	const [html, setHtml] = useState<string | null>(null);
	useEffect(() => {
		getHighlighter().then((h) => {
			setHtml(h.codeToHtml(code.replace(/^\n+|\n+$/g, ""), { lang, theme: "github-dark" }));
		});
	}, [code, lang]);
	return html;
}

function SdkCodeBlock({ code, lang }: { code: string; lang: string }) {
	const html = useHighlightedCode(code, lang);
	if (html) {
		return (
			<div
				className="[&_pre]:!bg-transparent [&_pre]:!p-0 [&_pre]:overflow-x-auto font-mono text-sm leading-relaxed"
				dangerouslySetInnerHTML={{ __html: html }}
			/>
		);
	}
	// Fallback while loading
	return (
		<pre className="font-mono text-sm leading-relaxed overflow-x-auto text-white">
			<code>{code}</code>
		</pre>
	);
}

export function LandingPage() {
	const [copied, setCopied] = useState<string | null>(null);

	const handleCopy = async (text: string, id: string) => {
		await navigator.clipboard.writeText(text);
		setCopied(id);
		setTimeout(() => setCopied(null), 2000);
	};

	const quickstart = "npx wandler --llm LiquidAI/LFM2.5-1.2B-Instruct-ONNX";

	type SdkTab = "openai" | "ai-sdk" | "langchain" | "llamaindex" | "curl";
	const [activeTab, setActiveTab] = useState<SdkTab>("openai");

	const sdkExamples: Record<SdkTab, { label: string; lang: string; code: string }> = {
		openai: {
			label: "OpenAI SDK",
			lang: "typescript",
			code: `import OpenAI from "openai";

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
}`,
		},
		"ai-sdk": {
			label: "Vercel AI SDK",
			lang: "typescript",
			code: `import { generateText, streamText } from "ai";
import { createOpenAI } from "@ai-sdk/openai";

const provider = createOpenAI({
  baseURL: "http://localhost:8000/v1",
  apiKey: "-",
  compatibility: "compatible",
});

const { text } = await generateText({
  model: provider.chat("LiquidAI/LFM2.5-1.2B-Instruct-ONNX"),
  prompt: "Hello!",
});

console.log(text);`,
		},
		langchain: {
			label: "LangChain",
			lang: "typescript",
			code: `import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({
  modelName: "LiquidAI/LFM2.5-1.2B-Instruct-ONNX",
  configuration: {
    baseURL: "http://localhost:8000/v1",
    apiKey: "-",
  },
});

const response = await model.invoke("Hello!");
console.log(response.content);`,
		},
		llamaindex: {
			label: "LlamaIndex",
			lang: "python",
			code: `from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    model="LiquidAI/LFM2.5-1.2B-Instruct-ONNX",
    api_base="http://localhost:8000/v1",
    api_key="-",
)

response = llm.complete("Hello!")
print(response)`,
		},
		curl: {
			label: "curl",
			lang: "bash",
			code: `curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "LiquidAI/LFM2.5-1.2B-Instruct-ONNX",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'`,
		},
	};

	return (
		<div className="flex flex-col min-h-screen bg-black text-white">
			<Header />

			<main className="flex-grow">
				{/* ── Hero ── */}
				<section className="relative overflow-hidden flex flex-col justify-center px-4 md:px-0 pt-20 pb-16 md:pt-24 md:pb-20">
					<div className="container mx-auto">
						<div className="grid md:grid-cols-[1fr_1fr] gap-8 md:gap-12 items-center">
							{/* Left: Logo centered + cards */}
							<div className="flex flex-col items-center space-y-10">
								<Image
									src="https://5xvkmufwzznj1ey2.public.blob.vercel-storage.com/wandler_logo_v5-vJ2L3NmauebkFJs9fOcFe7bPVM14To.svg"
									alt="wandler"
									width={600}
									height={120}
									className="w-[360px] md:w-[520px] lg:w-[600px] h-auto"
									priority
								/>

								<h1 className="text-2xl md:text-4xl font-bold tracking-tighter text-center">
									<span className="text-primary">transformers.js</span>{" "}
									inference server
								</h1>

								{/* Identity cards */}
								<div className="grid grid-cols-2 gap-4 w-full max-w-md">
									<div className="cyberpunk-corners bg-secondary p-4 text-center">
										<div className="text-primary font-bold">OpenAI API</div>
										<div className="text-xs text-muted-foreground mt-1">compatible</div>
									</div>
									<div className="cyberpunk-corners bg-secondary p-4 text-center">
										<div className="text-primary font-bold">WebGPU</div>
										<div className="text-xs text-muted-foreground mt-1">accelerated</div>
									</div>
								</div>

								{/* Quickstart */}
								<div className="cyberpunk-corners bg-secondary p-4 w-full max-w-md">
									<button
										onClick={() => handleCopy(quickstart, "qs")}
										className="w-full flex items-center gap-3 text-left cursor-pointer group"
									>
										<Terminal className="w-4 h-4 text-primary shrink-0" />
										<code className="font-mono text-xs md:text-sm text-white truncate">
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
								<div className="w-[300px] h-[300px] md:w-[520px] md:h-[520px] relative">
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

				{/* ── Hazard divider ── */}
				<div className="w-full h-4 bg-[repeating-linear-gradient(45deg,#000,#000_10px,hsl(58_96%_51%)_10px,hsl(58_96%_51%)_20px)] animate-experimental-bg"></div>

				{/* ── Endpoints ── */}
				<section className="py-20 md:py-28">
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
								{ path: "/v1/audio/transcriptions", title: "Audio", desc: "Speech-to-text via Whisper" },
								{ path: "/tokenize", title: "Tokenize", desc: "Convert between text and token IDs" },
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

				{/* ── Code Examples (tabbed) ── */}
				<section className="py-20 md:py-28 border-t border-primary/20">
					<div className="container mx-auto px-4">
						<h2 className="text-3xl md:text-5xl font-bold tracking-tighter mb-4">
							works with any openai sdk
						</h2>
						<p className="text-muted-foreground mb-8">
							drop-in replacement — change the base URL and go
						</p>

						<div className="cyberpunk-corners bg-secondary p-3 md:p-4">
							{/* Tabs */}
							<div className="flex gap-1 md:gap-2 mb-3">
								{(Object.keys(sdkExamples) as SdkTab[]).map((tab) => (
									<div
										key={tab}
										className={`cursor-pointer px-2 md:px-3 py-1 rounded text-sm ${
											activeTab === tab
												? "bg-primary/20 text-primary"
												: "text-muted-foreground hover:bg-primary/10"
										}`}
										onClick={() => setActiveTab(tab)}
									>
										{sdkExamples[tab].label}
									</div>
								))}
							</div>

							{/* Code with syntax highlighting + copy */}
							<div className="relative">
								<button
									onClick={() => handleCopy(sdkExamples[activeTab].code, "sdk")}
									className="absolute top-2 right-2 p-2 text-muted-foreground hover:text-primary transition-colors z-10"
									title="Copy to clipboard"
								>
									{copied === "sdk" ? (
										<Check className="w-4 h-4" />
									) : (
										<Copy className="w-4 h-4" />
									)}
								</button>
								<SdkCodeBlock code={sdkExamples[activeTab].code} lang={sdkExamples[activeTab].lang} />
							</div>
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
						<div className="cyberpunk-corners bg-secondary p-4 md:p-6 overflow-x-auto">
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
						<h2 className="text-3xl md:text-5xl font-bold tracking-tighter mb-12">
							features
						</h2>
						<div className="grid grid-cols-2 md:grid-cols-4 gap-4">
							{[
								{ title: "Streaming", desc: "SSE with real-time token generation" },
								{ title: "Tool Calling", desc: "LFM, Qwen, and OpenAI formats" },
								{ title: "Quantized", desc: "q4, q8, fp16, fp32 inference" },
								{ title: "WebGPU", desc: "GPU acceleration with CPU fallback" },
								{ title: "Embeddings", desc: "Text embeddings for RAG" },
								{ title: "Speech-to-Text", desc: "Whisper transcription" },
								{ title: "Auth", desc: "API key authentication" },
								{ title: "Metrics", desc: "Admin monitoring endpoint" },
							].map((f) => (
								<div key={f.title} className="border border-primary/10 p-4">
									<div className="text-primary font-bold text-sm">{f.title}</div>
									<div className="text-muted-foreground text-xs mt-1">{f.desc}</div>
								</div>
							))}
						</div>
					</div>
				</section>
			</main>

			<footer className="py-8 border-t border-primary/20">
				<div className="container mx-auto px-4 flex justify-between items-center text-muted-foreground text-sm">
					<span>wandler — transformers.js inference server</span>
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
