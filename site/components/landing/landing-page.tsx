"use client";

import { Check, Copy } from "lucide-react";
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
	return (
		<pre className="font-mono text-sm leading-relaxed overflow-x-auto text-white">
			<code>{code}</code>
		</pre>
	);
}

export function LandingPage() {
	const [copied, setCopied] = useState<string | null>(null);
	const [expandedEndpoint, setExpandedEndpoint] = useState<string | null>("/v1/chat/completions");

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
				<section className="relative overflow-hidden flex flex-col justify-center px-4 pt-24 pb-20 md:pt-32 md:pb-28 min-h-[85vh]">
					{/* Background grid */}
					<div className="absolute inset-0 opacity-[0.04]" style={{
						backgroundImage: "linear-gradient(hsl(58 96% 51%) 1px, transparent 1px), linear-gradient(90deg, hsl(58 96% 51%) 1px, transparent 1px)",
						backgroundSize: "48px 48px",
					}} />
					{/* Radial fade */}
					<div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_20%,black_80%)]" />

					<div className="container mx-auto relative z-10">
						<div className="flex flex-col items-center text-center space-y-8 max-w-4xl mx-auto">
							{/* Logo */}
							<Image
								src="https://5xvkmufwzznj1ey2.public.blob.vercel-storage.com/wandler_logo_v5-vJ2L3NmauebkFJs9fOcFe7bPVM14To.svg"
								alt="wandler"
								width={700}
								height={140}
								className="w-[340px] md:w-[520px] lg:w-[640px] h-auto"
								priority
							/>

							{/* Identity */}
							<div className="space-y-3">
								<p className="text-xl md:text-2xl lg:text-3xl tracking-tight">
									inference server for{" "}
									<span className="text-primary font-bold">transformers.js</span>
								</p>
								<p className="text-sm md:text-base text-muted-foreground tracking-wide">
									OpenAI-compatible{" · "}WebGPU accelerated{" · "}zero config
								</p>
							</div>

							{/* Quickstart terminal */}
							<div className="w-full max-w-2xl mt-4">
								<div className="bg-[#0a0a0a] border border-primary/20 overflow-hidden">
									<div className="flex items-center gap-2 px-4 py-2 bg-[#111] border-b border-primary/10">
										<div className="flex gap-1.5">
											<div className="w-2.5 h-2.5 rounded-full bg-[#e41832]/60" />
											<div className="w-2.5 h-2.5 rounded-full bg-primary/40" />
											<div className="w-2.5 h-2.5 rounded-full bg-[#00ff00]/40" />
										</div>
									</div>
									<button
										onClick={() => handleCopy(quickstart, "qs")}
										className="w-full flex items-center gap-3 px-4 py-3 text-left cursor-pointer group"
									>
										<span className="text-primary/60 font-mono text-sm select-none">$</span>
										<code className="font-mono text-sm md:text-base text-white/80">
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
						</div>
					</div>
				</section>

				{/* ── Hazard divider ── */}
				<div className="w-full h-3 bg-[repeating-linear-gradient(45deg,#000,#000_10px,hsl(58_96%_51%)_10px,hsl(58_96%_51%)_20px)] animate-experimental-bg" />

				{/* ── Get Started ── */}
				<section className="py-20 md:py-28 relative overflow-hidden">
					<div className="container mx-auto px-4 relative z-10">
						<div className="max-w-4xl">
							<h2 className="text-3xl md:text-5xl font-bold tracking-tighter mb-10">
								get started
							</h2>

							{/* 1. Start the server */}
							<p className="text-lg text-white mb-4">start the server</p>

							<div className="bg-[#0a0a0a] border border-white/[0.06] p-4 mb-3">
								<button
									onClick={() => handleCopy("npx wandler --llm LiquidAI/LFM2.5-1.2B-Instruct-ONNX", "qs1")}
									className="w-full flex items-center gap-3 text-left cursor-pointer group"
								>
									<span className="text-primary/50 font-mono text-sm select-none">$</span>
									<code className="font-mono text-sm text-white/80">npx wandler --llm LiquidAI/LFM2.5-1.2B-Instruct-ONNX</code>
									<span className="ml-auto shrink-0">
										{copied === "qs1" ? <Check className="w-3.5 h-3.5 text-primary" /> : <Copy className="w-3.5 h-3.5 text-white/20 group-hover:text-white/50 transition-colors" />}
									</span>
								</button>
							</div>

							<p className="text-muted-foreground text-sm mb-16">
								or let your agent handle it:{" "}
								<button
									onClick={() => handleCopy("npx add runpod-labs/wandler --skill wandler", "qs-skill")}
									className="inline-flex items-center gap-1.5 cursor-pointer group"
								>
									<code className="font-mono text-white/60 group-hover:text-white/80 transition-colors">npx add runpod-labs/wandler --skill wandler</code>
									<span className="shrink-0">
										{copied === "qs-skill" ? <Check className="w-3 h-3 text-primary" /> : <Copy className="w-3 h-3 text-white/20 group-hover:text-white/40 transition-colors" />}
									</span>
								</button>
							</p>

							{/* 2. Use it — drop-in replacement */}
							<p className="text-lg text-white mb-4">use it with any OpenAI-compatible SDK</p>

							<div className="bg-[#0a0a0a] border border-white/[0.06] overflow-hidden">
								{/* Tab bar */}
								<div className="flex border-b border-white/[0.06] bg-[#080808] overflow-x-auto">
									{(Object.keys(sdkExamples) as SdkTab[]).map((tab) => (
										<button
											key={tab}
											className={`px-4 py-2.5 text-sm font-mono whitespace-nowrap transition-colors ${
												activeTab === tab
													? "text-primary bg-primary/5 border-b-2 border-primary"
													: "text-muted-foreground hover:text-white hover:bg-white/[0.02]"
											}`}
											onClick={() => setActiveTab(tab)}
										>
											{sdkExamples[tab].label}
										</button>
									))}
								</div>

								{/* Code */}
								<div className="relative p-4 md:p-6">
									<button
										onClick={() => handleCopy(sdkExamples[activeTab].code, "sdk")}
										className="absolute top-3 right-3 p-2 text-muted-foreground hover:text-primary transition-colors z-10"
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
					</div>
				</section>

				{/* ── Thin accent line ── */}
				<div className="w-full h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />

				{/* ── API Reference ── */}
				<section className="py-20 md:py-28 relative overflow-hidden">
					<div className="container mx-auto px-4 relative z-10">
						<div className="max-w-4xl">
							<h2 className="text-3xl md:text-5xl font-bold tracking-tighter mb-12">
								API reference
							</h2>

							<div className="space-y-0">
								{([
									{
										method: "POST", path: "/v1/chat/completions", desc: "Chat completion with streaming and tool calling",
										params: [
											{ name: "messages", type: "array", required: true, desc: "Input messages with role and content" },
											{ name: "temperature", type: "float", required: false, desc: "Sampling temperature, 0-2. Default 0.7" },
											{ name: "top_p", type: "float", required: false, desc: "Nucleus sampling threshold. Default 0.95" },
											{ name: "max_tokens", type: "int", required: false, desc: "Maximum tokens to generate" },
											{ name: "stream", type: "boolean", required: false, desc: "Enable SSE streaming. Default false" },
											{ name: "stop", type: "string | string[]", required: false, desc: "Stop sequences" },
											{ name: "tools", type: "array", required: false, desc: "Function calling tool definitions" },
											{ name: "response_format", type: "object", required: false, desc: '{"type": "json_object"} for JSON mode' },
											{ name: "top_k", type: "int", required: false, desc: "Top-k sampling" },
											{ name: "min_p", type: "float", required: false, desc: "Minimum probability threshold" },
											{ name: "repetition_penalty", type: "float", required: false, desc: "Repetition penalty, > 1.0 to penalize" },
											{ name: "stream_options", type: "object", required: false, desc: '{"include_usage": true} for usage stats' },
										],
									},
									{
										method: "POST", path: "/v1/completions", desc: "Text completion (legacy) with echo and suffix",
										params: [
											{ name: "prompt", type: "string", required: true, desc: "Input text prompt" },
											{ name: "temperature", type: "float", required: false, desc: "Sampling temperature, 0-2. Default 0.7" },
											{ name: "max_tokens", type: "int", required: false, desc: "Maximum tokens to generate" },
											{ name: "stream", type: "boolean", required: false, desc: "Enable SSE streaming. Default false" },
											{ name: "stop", type: "string | string[]", required: false, desc: "Stop sequences" },
											{ name: "echo", type: "boolean", required: false, desc: "Echo the prompt in the response" },
											{ name: "suffix", type: "string", required: false, desc: "Text to append after completion" },
										],
									},
									{
										method: "POST", path: "/v1/embeddings", desc: "Text embeddings for RAG and semantic search",
										params: [
											{ name: "input", type: "string | string[]", required: true, desc: "Text to embed" },
											{ name: "encoding_format", type: "string", required: false, desc: '"float" or "base64". Default "float"' },
										],
									},
									{
										method: "GET", path: "/v1/models", desc: "List and inspect loaded models",
										params: [],
									},
									{
										method: "POST", path: "/v1/audio/transcriptions", desc: "Speech-to-text via Whisper",
										params: [
											{ name: "file", type: "binary", required: true, desc: "Audio file to transcribe" },
											{ name: "language", type: "string", required: false, desc: "Language code (e.g. en, de)" },
										],
									},
									{
										method: "POST", path: "/tokenize", desc: "Convert between text and token IDs",
										params: [
											{ name: "text", type: "string", required: true, desc: "Text to tokenize" },
										],
									},
								] as const).map((ep) => {
									const isExpanded = expandedEndpoint === ep.path;
									return (
										<div key={ep.path} className="border-b border-white/[0.06]">
											<button
												onClick={() => setExpandedEndpoint(isExpanded ? null : ep.path)}
												className="w-full flex items-center gap-4 py-4 text-left cursor-pointer group"
											>
												<span className="font-mono text-[11px] text-white/70 bg-white/[0.08] px-1.5 py-0.5 rounded-sm shrink-0">{ep.method}</span>
												<code className="font-mono text-sm text-white group-hover:text-primary transition-colors">{ep.path}</code>
												<span className="text-muted-foreground text-sm ml-auto hidden md:block">{ep.desc}</span>
												<span className="text-muted-foreground text-xs shrink-0 ml-2">{isExpanded ? "−" : "+"}</span>
											</button>

											{isExpanded && ep.params.length > 0 && (
												<div className="pb-6 pl-4 md:pl-[88px]">
													<p className="text-muted-foreground text-sm mb-4 md:hidden">{ep.desc}</p>
													<div className="text-xs text-muted-foreground uppercase tracking-wider mb-3">Body</div>
													{ep.params.map((p) => (
														<div key={p.name} className="py-3 border-t border-white/[0.04]">
															<div className="flex items-center gap-2 flex-wrap">
																<code className="font-mono text-[13px] text-white font-medium">{p.name}</code>
																<span className="font-mono text-[11px] text-muted-foreground bg-white/[0.06] px-1.5 py-0.5 rounded-sm">{p.type}</span>
																{p.required && <span className="text-[11px] text-primary/70">Required</span>}
															</div>
															<p className="text-muted-foreground text-sm mt-1">{p.desc}</p>
														</div>
													))}
												</div>
											)}
										</div>
									);
								})}
							</div>
						</div>
					</div>
				</section>

				{/* ── Thin accent line ── */}
				<div className="w-full h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />

				{/* ── Benchmarks ── */}
				<section className="py-20 md:py-28 relative">
					<div className="container mx-auto px-4 relative z-10">
						<h2 className="text-3xl md:text-5xl font-bold tracking-tighter mb-4">
							benchmarks
						</h2>
						<p className="text-muted-foreground mb-8 font-mono text-sm">
							WebGPU · q4 quantization · 10 runs per scenario
						</p>

						<div className="border border-white/[0.06] bg-[#0a0a0a] overflow-hidden">
							<div className="overflow-x-auto">
								<table className="w-full text-left">
									<thead>
										<tr className="border-b border-white/[0.08] bg-[#080808]">
											<th className="py-3 px-4 text-white/40 font-mono text-xs tracking-wider uppercase">Model</th>
											<th className="py-3 px-4 text-white/40 font-mono text-xs tracking-wider uppercase">Params</th>
											<th className="py-3 px-4 text-white/40 font-mono text-xs tracking-wider uppercase">Weights</th>
											<th className="py-3 px-4 text-white/40 font-mono text-xs tracking-wider uppercase">Context</th>
											<th className="py-3 px-4 text-white/40 font-mono text-xs tracking-wider uppercase">tok/s</th>
											<th className="py-3 px-4 text-white/40 font-mono text-xs tracking-wider uppercase">TTFT</th>
											<th className="py-3 px-4 text-white/40 font-mono text-xs tracking-wider uppercase">Load</th>
											<th className="py-3 px-4 text-white/40 font-mono text-xs tracking-wider uppercase">Capabilities</th>
										</tr>
									</thead>
									<tbody className="font-mono text-sm">
										{[
											{ org: "LiquidAI", model: "LFM2.5-350M-ONNX", repo: "LiquidAI/LFM2.5-350M-ONNX", params: "350M", weights: "~200 MB", context: "32K", tps: "248", ttft: "16ms", load: "0.5s", caps: "text" },
											{ org: "LiquidAI", model: "LFM2.5-1.2B-Instruct-ONNX", repo: "LiquidAI/LFM2.5-1.2B-Instruct-ONNX", params: "1.2B", weights: "~700 MB", context: "32K", tps: "118", ttft: "34ms", load: "1.7s", caps: "text, tools" },
											{ org: "onnx-community", model: "Qwen3.5-0.8B-Text-ONNX", repo: "onnx-community/Qwen3.5-0.8B-Text-ONNX", params: "0.8B", weights: "~500 MB", context: "32K", tps: "37", ttft: "276ms", load: "1.8s", caps: "text, tools" },
											{ org: "onnx-community", model: "gemma-4-E4B-it-ONNX", repo: "onnx-community/gemma-4-E4B-it-ONNX", params: "4B", weights: "~2.5 GB", context: "32K", tps: "20", ttft: "636ms", load: "13.4s", caps: "text, tools, vision" },
											{ org: "onnx-community", model: "gemma-4-E2B-it-ONNX", repo: "onnx-community/gemma-4-E2B-it-ONNX", params: "2B", weights: "~1.2 GB", context: "32K", tps: "12", ttft: "890ms", load: "7.0s", caps: "text, tools, vision" },
										].map((row) => (
											<tr key={row.model} className="border-b border-white/[0.04]">
												<td className="py-3 px-4">
													<div className="flex items-center gap-2">
														<a
															href={`https://huggingface.co/${row.repo}`}
															target="_blank"
															rel="noopener noreferrer"
															className="hover:text-primary transition-colors"
														>
															<span className="text-white/40">{row.org}/</span><span className="text-white font-medium">{row.model}</span>
														</a>
														<button
															onClick={() => handleCopy(row.repo, `bench-${row.model}`)}
															className="shrink-0 text-white/20 hover:text-white/50 transition-colors cursor-pointer"
															title={`Copy ${row.repo}`}
														>
															{copied === `bench-${row.model}` ? <Check className="w-3 h-3 text-primary" /> : <Copy className="w-3 h-3" />}
														</button>
													</div>
												</td>
												<td className="py-3 px-4 text-white/50">{row.params}</td>
												<td className="py-3 px-4 text-white/50">{row.weights}</td>
												<td className="py-3 px-4 text-white/50">{row.context}</td>
												<td className="py-3 px-4 text-white font-bold">{row.tps}</td>
												<td className="py-3 px-4 text-white/70">{row.ttft}</td>
												<td className="py-3 px-4 text-white/70">{row.load}</td>
												<td className="py-3 px-4 text-white/70">{row.caps}</td>
											</tr>
										))}
									</tbody>
								</table>
							</div>
						</div>
					</div>
				</section>

				{/* ── Thin accent line ── */}
				<div className="w-full h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />

				{/* ── Features ── */}
				<section className="py-20 md:py-28">
					<div className="container mx-auto px-4">
						<h2 className="text-3xl md:text-5xl font-bold tracking-tighter mb-12">
							features
						</h2>
						<div className="grid grid-cols-2 md:grid-cols-4 gap-3">
							{[
								{ title: "Streaming", desc: "SSE with real-time token generation" },
								{ title: "Tool Calling", desc: "LFM, Qwen, Gemma, and OpenAI formats" },
								{ title: "Quantized", desc: "q4, q8, fp16, fp32 inference" },
								{ title: "WebGPU", desc: "GPU acceleration with CPU fallback" },
								{ title: "Embeddings", desc: "Text embeddings for RAG" },
								{ title: "Speech-to-Text", desc: "Whisper transcription" },
								{ title: "Auth", desc: "API key authentication" },
								{ title: "Metrics", desc: "Admin monitoring endpoint" },
							].map((f) => (
								<div key={f.title} className="group border border-white/[0.04] bg-[#0a0a0a] p-4 transition-all hover:border-primary/20 hover:bg-primary/[0.02]">
									<div className="text-primary font-bold text-sm mb-1">{f.title}</div>
									<div className="text-muted-foreground text-xs leading-relaxed">{f.desc}</div>
								</div>
							))}
						</div>
					</div>
				</section>

				{/* ── Bottom hazard stripe ── */}
				<div className="w-full h-3 bg-[repeating-linear-gradient(45deg,#000,#000_10px,hsl(58_96%_51%)_10px,hsl(58_96%_51%)_20px)] animate-experimental-bg" />
			</main>

			<footer className="py-8 bg-[#050505]">
				<div className="container mx-auto px-4 flex justify-between items-center text-muted-foreground text-sm">
					<span className="font-mono text-xs">wandler — transformers.js inference server</span>
					<Link
						href="https://github.com/runpod-labs/wandler"
						className="text-primary hover:underline font-mono text-xs"
						target="_blank"
					>
						github
					</Link>
				</div>
			</footer>
		</div>
	);
}
