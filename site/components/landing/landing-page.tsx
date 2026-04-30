"use client";

import { ArrowUpRight, Check, Copy } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { type ReactNode, useEffect, useState } from "react";
import { createHighlighter, type Highlighter } from "shiki";

import { Header } from "@/components/header";
import { CyberpunkHero } from "@/components/landing/cyberpunk-hero";

let highlighterPromise: Promise<Highlighter> | null = null;
function getHighlighter() {
	if (!highlighterPromise) {
		highlighterPromise = createHighlighter({
			themes: ["github-dark"],
			langs: ["typescript", "python", "bash", "json", "yaml", "markdown", "tsx", "jsx"],
		});
	}
	return highlighterPromise;
}

function useHighlightedCode(code: string, lang: string) {
	const [html, setHtml] = useState<string | null>(null);
	useEffect(() => {
		getHighlighter().then((h) => {
			try {
				setHtml(h.codeToHtml(code.replace(/^\n+|\n+$/g, ""), { lang, theme: "github-dark" }));
			} catch {
				setHtml(null);
			}
		});
	}, [code, lang]);
	return html;
}

// Custom bash renderer — shiki grammars don't reliably distinguish
// "first word as the command keyword" vs "arbitrary identifier", so we
// tokenize ourselves:
//   npx / npm runners                       → magenta (hero #ff00ff)
//   first keyword after any runner          → yellow (matches wandler logo)
//   any flag token (-g, --llm, ...)         → cyan (hero #00ffff)
//   everything else                         → white
function BashHighlight({ code }: { code: string }) {
	const cleaned = code.replace(/^\n+|\n+$/g, "");
	const lines = cleaned.split("\n");
	return (
		<>
			{lines.map((line, i) => {
				if (line.trimStart().startsWith("#")) {
					return (
						<span key={i} className="text-white/45">
							{line}
							{i < lines.length - 1 && "\n"}
						</span>
					);
				}

				const trimmed = line.trimStart();
				let sawKeyword = trimmed.startsWith("-");
				const parts = line.split(/(\s+)/);
				return (
					<span key={i}>
						{parts.map((tok, j) => {
							const key = `${i}-${j}`;
							if (tok === "") return null;
							if (/^\s+$/.test(tok)) return <span key={key}>{tok}</span>;
							if (/^-\S/.test(tok)) {
								return <span key={key} className="text-[#00ffff]">{tok}</span>;
							}
							if (!sawKeyword && (tok === "npx" || tok === "npm")) {
								return <span key={key} className="text-[#ff00ff]">{tok}</span>;
							}
							if (!sawKeyword && /^[A-Za-z_]/.test(tok)) {
								sawKeyword = true;
								return <span key={key} className="text-primary">{tok}</span>;
							}
							return <span key={key} className="text-white/85">{tok}</span>;
						})}
						{i < lines.length - 1 && "\n"}
					</span>
				);
			})}
		</>
	);
}

function SdkCodeBlock({ code, lang }: { code: string; lang: string }) {
	const html = useHighlightedCode(lang === "bash" ? "" : code, lang);
	if (lang === "bash") {
		return (
			<pre className="font-mono text-sm leading-relaxed whitespace-pre-wrap break-all">
				<BashHighlight code={code} />
			</pre>
		);
	}
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

function InlineCode({ children }: { children: ReactNode }) {
	return (
		<code className="font-mono text-[13px] bg-white/[0.08] text-white px-1.5 py-0.5 rounded">
			{children}
		</code>
	);
}

function CodeLine({
	cmd,
	id,
	handleCopy,
	copied,
}: {
	cmd: string;
	id: string;
	handleCopy: (text: string, id: string) => void;
	copied: string | null;
}) {
	return (
		<div className="bg-[#0a0a0a] border border-white/[0.08] rounded-md p-4">
			<button
				onClick={() => handleCopy(cmd, id)}
				className="w-full flex items-start gap-3 text-left cursor-pointer group"
			>
				<pre className="flex-1 min-w-0 font-mono text-sm whitespace-pre-wrap break-all">
					<BashHighlight code={cmd} />
				</pre>
				<span className="shrink-0">
					{copied === id ? (
						<Check className="w-4 h-4 text-primary" />
					) : (
						<Copy className="w-4 h-4 text-white/20 group-hover:text-white/60 transition-colors" />
					)}
				</span>
			</button>
		</div>
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

	type AgentTab = "hermes";
	const [activeAgentTab, setActiveAgentTab] = useState<AgentTab>("hermes");

	type SdkTab = "openai" | "ai-sdk" | "langchain" | "llamaindex" | "curl";
	const [activeTab, setActiveTab] = useState<SdkTab>("openai");

	type SetupTab = "text" | "embedding" | "stt";
	const [activeSetupTab, setActiveSetupTab] = useState<SetupTab>("text");

	const setupExamples: Record<SetupTab, { label: string; code: string }> = {
		text: {
			label: "text",
			code: "wandler --llm LiquidAI/LFM2.5-1.2B-Instruct-ONNX",
		},
		embedding: {
			label: "embedding",
			code: `wandler --llm LiquidAI/LFM2.5-1.2B-Instruct-ONNX \\
  --embedding Xenova/all-MiniLM-L6-v2:q8`,
		},
		stt: {
			label: "speech-to-text",
			code: `wandler --llm LiquidAI/LFM2.5-1.2B-Instruct-ONNX \\
  --stt onnx-community/whisper-tiny:q4`,
		},
	};

	const sdkExamples: Record<SdkTab, { label: string; lang: string; code: string }> = {
		openai: {
			label: "OpenAI SDK",
			lang: "typescript",
			code: `import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8000/v1",
  // replace with your --api-key value if you started wandler with one
  apiKey: "changeme",
});

const res = await client.chat.completions.create({
  model: "LiquidAI/LFM2.5-1.2B-Instruct-ONNX",
  messages: [{ role: "user", content: "What is the capital of Germany" }],
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

const wandler = createOpenAI({
  baseURL: "http://localhost:8000/v1",
  // replace with your --api-key value if you started wandler with one
  apiKey: "changeme",
});

const { text } = await generateText({
  model: wandler("LiquidAI/LFM2.5-1.2B-Instruct-ONNX"),
  prompt: "What is the capital of Germany",
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
    // replace with your --api-key value if you started wandler with one
    apiKey: "changeme",
  },
});

const response = await model.invoke("What is the capital of Germany");
console.log(response.content);`,
		},
		llamaindex: {
			label: "LlamaIndex",
			lang: "python",
			code: `from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    model="LiquidAI/LFM2.5-1.2B-Instruct-ONNX",
    api_base="http://localhost:8000/v1",
    # replace with your --api-key value if you started wandler with one
    api_key="changeme",
)

response = llm.complete("What is the capital of Germany")
print(response)`,
		},
		curl: {
			label: "curl",
			lang: "bash",
			code: `curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "LiquidAI/LFM2.5-1.2B-Instruct-ONNX",
    "messages": [{"role": "user", "content": "What is the capital of Germany"}]
  }'`,
		},
	};

	return (
		<div className="flex flex-col min-h-screen bg-black text-white">
			<Header />

			<main className="flex-grow">
				{/* ── 3D Cyberpunk Hero ── */}
				<CyberpunkHero>
					{/* Run the server — cyan neon with L-bracket corners */}
					<div className="w-full pt-5">
						<div className="relative">
							<div className="absolute bottom-full translate-y-px left-0 z-10 bg-black/90 border border-[#00ffff]/30 px-2 py-0.5">
								<span className="font-mono text-[11px] text-[#00ffff] uppercase tracking-[0.2em] drop-shadow-[0_0_8px_rgba(0,255,255,0.6)]">run the server</span>
							</div>
							{/* L-bracket top-right */}
							<div className="absolute -top-[2px] -right-[2px] w-4 h-4 z-10 border-t-2 border-r-2 border-[#00ffff]" />
							{/* L-bracket bottom-left */}
							<div className="absolute -bottom-[2px] -left-[2px] w-4 h-4 z-10 border-b-2 border-l-2 border-[#00ffff]" />
							{/* L-bracket bottom-right */}
							<div className="absolute -bottom-[2px] -right-[2px] w-4 h-4 z-10 border-b-2 border-r-2 border-[#00ffff]" />
							<button
								onClick={() => handleCopy(quickstart, "qs")}
								className="w-full bg-black/90 border border-[#00ffff]/30 px-5 py-5 flex items-center gap-3 text-left cursor-pointer group shadow-[0_0_20px_rgba(0,255,255,0.06),inset_0_0_30px_rgba(0,255,255,0.02)]"
							>
								<code className="font-mono text-sm text-white/80">{quickstart}</code>
								<span className="ml-auto shrink-0">
									{copied === "qs" ? <Check className="w-4 h-4 text-[#00ffff]" /> : <Copy className="w-4 h-4 text-white/20 group-hover:text-[#00ffff] transition-colors" />}
								</span>
							</button>
						</div>
					</div>

					{/* Let your agent do it — magenta neon with crosshairs */}
					<div className="w-full pt-5">
						<div className="relative">
							<div className="absolute bottom-full translate-y-px left-0 z-10 bg-black/90 border border-[#ff00ff]/30 px-2 py-0.5">
								<span className="font-mono text-[11px] text-[#ff00ff] uppercase tracking-[0.2em] drop-shadow-[0_0_8px_rgba(255,0,255,0.6)]">let your agent run the server</span>
							</div>
							{/* Crosshair top-right */}
							<div className="absolute -top-[6px] -right-[6px] w-[12px] h-[12px] z-10">
								<div className="absolute top-1/2 left-0 w-full h-px bg-[#ff00ff] shadow-[0_0_4px_rgba(255,0,255,0.8)]" />
								<div className="absolute left-1/2 top-0 h-full w-px bg-[#ff00ff] shadow-[0_0_4px_rgba(255,0,255,0.8)]" />
							</div>
							{/* Crosshair bottom-left */}
							<div className="absolute -bottom-[6px] -left-[6px] w-[12px] h-[12px] z-10">
								<div className="absolute top-1/2 left-0 w-full h-px bg-[#ff00ff] shadow-[0_0_4px_rgba(255,0,255,0.8)]" />
								<div className="absolute left-1/2 top-0 h-full w-px bg-[#ff00ff] shadow-[0_0_4px_rgba(255,0,255,0.8)]" />
							</div>
							{/* Crosshair bottom-right */}
							<div className="absolute -bottom-[6px] -right-[6px] w-[12px] h-[12px] z-10">
								<div className="absolute top-1/2 left-0 w-full h-px bg-[#ff00ff] shadow-[0_0_4px_rgba(255,0,255,0.8)]" />
								<div className="absolute left-1/2 top-0 h-full w-px bg-[#ff00ff] shadow-[0_0_4px_rgba(255,0,255,0.8)]" />
							</div>
							<button
								onClick={() => handleCopy("npx skills add runpod-labs/wandler --skill wandler", "qs-hero-skill")}
								className="w-full bg-black/90 border border-[#ff00ff]/30 px-5 py-5 flex items-center gap-3 text-left cursor-pointer group shadow-[0_0_20px_rgba(255,0,255,0.06),inset_0_0_30px_rgba(255,0,255,0.02)]"
							>
								<code className="font-mono text-sm text-white/80">npx skills add runpod-labs/wandler --skill wandler</code>
								<span className="ml-auto shrink-0">
									{copied === "qs-hero-skill" ? <Check className="w-4 h-4 text-[#ff00ff]" /> : <Copy className="w-4 h-4 text-white/20 group-hover:text-[#ff00ff] transition-colors" />}
								</span>
							</button>
						</div>
					</div>
				</CyberpunkHero>

				{/* ── Setup ── */}
				<section className="py-20 md:py-28 relative overflow-hidden">
					<div className="container mx-auto px-4 relative z-10">
						<h2 id="setup" className="text-3xl md:text-5xl font-bold tracking-tighter mb-4 scroll-mt-20">setup</h2>
						<div className="max-w-3xl">
							<p className="text-white/90 text-base leading-relaxed mb-3">
								install it globally
							</p>
							<div className="mb-8">
								<CodeLine cmd="npm install -g wandler" id="qs-install" handleCopy={handleCopy} copied={copied} />
							</div>

							<p className="text-white/90 text-base leading-relaxed mb-3">
								or use <InlineCode>npx</InlineCode>
							</p>
							<div>
								<CodeLine cmd="npx wandler --llm <org/repo:precision>" id="qs-npx" handleCopy={handleCopy} copied={copied} />
							</div>
						</div>
					</div>
				</section>

				{/* ── Thin accent line ── */}
				<div className="w-full h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />

				{/* ── Run the server ── */}
				<section className="py-20 md:py-28 relative overflow-hidden">
					<div className="container mx-auto px-4 relative z-10">
						<h2 id="run-the-server" className="text-3xl md:text-5xl font-bold tracking-tighter mb-4 scroll-mt-20">run the server</h2>
						<p className="text-muted-foreground mb-8">
							the server runs at <InlineCode>http://127.0.0.1:8000</InlineCode> and exposes an{" "}
							<a href="#api-reference" className="text-white hover:text-primary transition-colors underline underline-offset-4 decoration-white/30 hover:decoration-primary">OpenAI-compatible API</a>
							, so any OpenAI client works out of the box.
						</p>

						<div className="max-w-3xl">
							<div className="bg-[#0a0a0a] border border-white/[0.06] overflow-hidden mb-6">
								<div className="flex border-b border-white/[0.06] bg-[#080808] overflow-x-auto">
									{(Object.keys(setupExamples) as SetupTab[]).map((tab) => (
										<button
											key={tab}
											className={`px-4 py-2.5 text-sm font-mono whitespace-nowrap transition-colors ${
												activeSetupTab === tab
													? "text-white bg-white/[0.06] border-b-2 border-white"
													: "text-muted-foreground hover:text-white hover:bg-white/[0.02]"
											}`}
											onClick={() => setActiveSetupTab(tab)}
										>
											{setupExamples[tab].label}
										</button>
									))}
								</div>
								<div className="relative p-4 md:p-6 min-h-[8.5rem]">
									<button
										onClick={() => handleCopy(setupExamples[activeSetupTab].code, "setup")}
										className="absolute top-3 right-3 p-2 text-white/20 hover:text-white/60 transition-colors z-10"
										title="Copy to clipboard"
									>
										{copied === "setup" ? <Check className="w-4 h-4 text-primary" /> : <Copy className="w-4 h-4" />}
									</button>
									<div className="[&_pre]:whitespace-pre-wrap [&_pre]:break-all pr-8">
										<SdkCodeBlock code={setupExamples[activeSetupTab].code} lang="bash" />
									</div>
								</div>
							</div>

							<p className="text-white/90 text-base leading-relaxed mb-6">
								here is every flag <InlineCode>wandler</InlineCode> accepts:
							</p>

							<div className="grid grid-cols-1 md:grid-cols-[max-content_1fr] gap-x-10 gap-y-6 mb-8">
								{([
									{ flag: "--llm", arg: "<id>", desc: "LLM model.", meta: <>format: <InlineCode>org/repo[:precision]</InlineCode></> },
									{ flag: "--backend", arg: "<name>", desc: "LLM backend.", meta: <>default: <InlineCode>wandler</InlineCode> · baseline: <InlineCode>transformersjs</InlineCode></> },
									{ flag: "--embedding", arg: "<id>", desc: "Embedding model." },
									{ flag: "--stt", arg: "<id>", desc: "Speech-to-text model." },
									{ flag: "--device", arg: "<type>", desc: "Inference device.", meta: <>default: <InlineCode>auto</InlineCode> · options: <InlineCode>auto</InlineCode>, <InlineCode>webgpu</InlineCode>, <InlineCode>cpu</InlineCode>, <InlineCode>wasm</InlineCode></> },
									{ flag: "--port", arg: "<n>", desc: "Server port.", meta: <>default: <InlineCode>8000</InlineCode></> },
									{ flag: "--host", arg: "<addr>", desc: "Bind address.", meta: <>default: <InlineCode>127.0.0.1</InlineCode></> },
									{ flag: "--api-key", arg: "<key>", desc: "Bearer auth token.", meta: <>reads env <InlineCode>WANDLER_API_KEY</InlineCode></> },
									{ flag: "--hf-token", arg: "<token>", desc: "HuggingFace token for gated models." },
									{ flag: "--cors-origin", arg: "<origin>", desc: "Allowed CORS origin.", meta: <>default: <InlineCode>*</InlineCode></> },
									{ flag: "--max-tokens", arg: "<n>", desc: "Max tokens per request.", meta: <>default: the loaded model&apos;s max context length</> },
									{ flag: "--max-concurrent", arg: "<n>", desc: "Concurrent requests.", meta: <>default: <InlineCode>1</InlineCode></> },
									{ flag: "--timeout", arg: "<ms>", desc: "Request timeout in milliseconds.", meta: <>default: <InlineCode>120000</InlineCode></> },
									{ flag: "--log-level", arg: "<level>", desc: "Log verbosity.", meta: <>default: <InlineCode>info</InlineCode> · options: <InlineCode>debug</InlineCode>, <InlineCode>info</InlineCode>, <InlineCode>warn</InlineCode>, <InlineCode>error</InlineCode></> },
									{ flag: "--quiet", arg: "", desc: "Suppress non-error startup and profile logs." },
									{ flag: "--cache-dir", arg: "<path>", desc: "Model cache directory.", meta: <>default: <InlineCode>~/.cache/huggingface</InlineCode> (standard HuggingFace cache, also respects <InlineCode>HF_HOME</InlineCode>)</> },
									{ flag: "--prefill-chunk-size", arg: "<n>", desc: "Chunk size for long-prompt prefill.", meta: <>default: <InlineCode>auto</InlineCode> (640MB WebGPU attention budget) · use <InlineCode>auto:&lt;mb&gt;</InlineCode> to tune or <InlineCode>0</InlineCode>/<InlineCode>off</InlineCode> to disable</> },
									{ flag: "--decode-loop", arg: "<mode>", desc: "Wandler-owned decode loop.", meta: <>default: <InlineCode>auto</InlineCode> uses transformers.js <InlineCode>generate()</InlineCode> · use <InlineCode>on</InlineCode> to opt into the experimental loop</> },
									{ flag: "--prefix-cache", arg: "<mode>", desc: "Enable prefix KV cache.", meta: <>default: <InlineCode>true</InlineCode></> },
									{ flag: "--prefix-cache-entries", arg: "<n>", desc: "Prefix KV cache entries.", meta: <>default: <InlineCode>2</InlineCode></> },
									{ flag: "--prefix-cache-min-tokens", arg: "<n>", desc: "Minimum prefix tokens to cache.", meta: <>default: <InlineCode>512</InlineCode></> },
									{ flag: "--warmup-tokens", arg: "<n>", desc: "Approximate prompt tokens to run once before serving.", meta: <>default: <InlineCode>0</InlineCode></> },
									{ flag: "--warmup-max-new-tokens", arg: "<n>", desc: "Max new tokens for startup warmup.", meta: <>default: <InlineCode>8</InlineCode></> },
								] as const).flatMap((o) => [
									<div key={`${o.flag}-l`} className="font-mono text-[13px] whitespace-nowrap pt-0.5">
										<span className="text-[#00ffff]">{o.flag}</span>
										{" "}
										<span className="text-white/50">{o.arg}</span>
									</div>,
									<div key={`${o.flag}-r`} className="text-white/90 text-base leading-relaxed">
										{o.desc}
										{"meta" in o && o.meta && (
											<div className="text-white/50 text-sm mt-1">{o.meta}</div>
										)}
									</div>,
								])}
							</div>

							<p className="text-white/90 text-base leading-relaxed">
								precision suffixes: <InlineCode>q1</InlineCode>,{" "}
								<InlineCode>q2</InlineCode>, <InlineCode>q4</InlineCode> (default),{" "}
								<InlineCode>q8</InlineCode>, <InlineCode>fp16</InlineCode>, <InlineCode>fp32</InlineCode>.
							</p>
						</div>
					</div>
				</section>

				{/* ── Thin accent line ── */}
				<div className="w-full h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />

				{/* ── Discover models ── */}
				<section className="py-20 md:py-28 relative overflow-hidden">
					<div className="container mx-auto px-4 relative z-10">
						<h2 id="discover-models" className="text-3xl md:text-5xl font-bold tracking-tighter mb-4 scroll-mt-20">discover models</h2>
						<p className="text-muted-foreground mb-8">
							list every model in the wandler registry with type, size, precision and capabilities
						</p>

						<div className="max-w-3xl mb-20">
							<div className="mb-6">
								<CodeLine cmd="wandler model ls" id="cli-model-ls" handleCopy={handleCopy} copied={copied} />
							</div>
							<p className="text-white/90 text-base leading-relaxed">
								filter by type with <InlineCode>--type llm</InlineCode>, <InlineCode>--type embedding</InlineCode>, or <InlineCode>--type stt</InlineCode>.
							</p>
						</div>

						<h3 id="benchmarks" className="text-xl md:text-2xl font-bold tracking-tight mb-2 scroll-mt-20">benchmarks</h3>
						<p className="text-muted-foreground mb-6 font-mono text-sm">
							WebGPU · q4 quantization · 10 runs per scenario · tested on m3 pro 18gb
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
											{ org: "LiquidAI", model: "LFM2.5-350M-ONNX", repo: "LiquidAI/LFM2.5-350M-ONNX", params: "350M", weights: "~200 MB", context: "128K", tps: "248", ttft: "16ms", load: "0.5s", caps: "text" },
											{ org: "LiquidAI", model: "LFM2.5-1.2B-Instruct-ONNX", repo: "LiquidAI/LFM2.5-1.2B-Instruct-ONNX", params: "1.2B", weights: "~700 MB", context: "128K", tps: "118", ttft: "34ms", load: "1.7s", caps: "text, tools" },
											{ org: "onnx-community", model: "Qwen3.5-0.8B-Text-ONNX", repo: "onnx-community/Qwen3.5-0.8B-Text-ONNX", params: "0.8B", weights: "~500 MB", context: "256K", tps: "37", ttft: "276ms", load: "1.8s", caps: "text, tools" },
											{ org: "onnx-community", model: "gemma-4-E4B-it-ONNX", repo: "onnx-community/gemma-4-E4B-it-ONNX", params: "4B", weights: "~2.5 GB", context: "128K", tps: "20", ttft: "636ms", load: "13.4s", caps: "text, tools, vision" },
											{ org: "onnx-community", model: "gemma-4-E2B-it-ONNX", repo: "onnx-community/gemma-4-E2B-it-ONNX", params: "2B", weights: "~1.2 GB", context: "128K", tps: "12", ttft: "890ms", load: "7.0s", caps: "text, tools, vision" },
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

						<div className="mt-8 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
							<p className="text-muted-foreground text-sm">
								these are the ones we tested. any transformers.js-compatible model on Hugging Face works.
							</p>
							<a
								href="https://huggingface.co/models?library=transformers.js"
								target="_blank"
								rel="noopener noreferrer"
								className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-black font-bold text-sm tracking-tight hover:bg-primary/90 transition-colors self-start sm:self-auto whitespace-nowrap"
							>
								find more on Hugging Face
								<ArrowUpRight className="w-4 h-4" />
							</a>
						</div>
					</div>
				</section>

				{/* ── Thin accent line ── */}
				<div className="w-full h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />

				{/* ── Use it in your app ── */}
				<section className="py-20 md:py-28 relative overflow-hidden">
					<div className="container mx-auto px-4 relative z-10">
						<h2 id="use-it-in-your-app" className="text-3xl md:text-5xl font-bold tracking-tighter mb-4 scroll-mt-20">
							use it in your app
						</h2>
						<p className="text-muted-foreground mb-8">
							drop-in replacement for any OpenAI-compatible SDK
						</p>

						<div className="bg-[#0a0a0a] border border-white/[0.06] overflow-hidden max-w-4xl">
							{/* Tab bar */}
							<div className="flex border-b border-white/[0.06] bg-[#080808] overflow-x-auto">
								{(Object.keys(sdkExamples) as SdkTab[]).map((tab) => (
									<button
										key={tab}
										className={`px-4 py-2.5 text-sm font-mono whitespace-nowrap transition-colors ${
											activeTab === tab
												? "text-white bg-white/[0.06] border-b-2 border-white"
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
									className="absolute top-3 right-3 p-2 text-white/20 hover:text-white/60 transition-colors z-10"
									title="Copy to clipboard"
								>
									{copied === "sdk" ? (
										<Check className="w-4 h-4 text-primary" />
									) : (
										<Copy className="w-4 h-4" />
									)}
								</button>
								<SdkCodeBlock code={sdkExamples[activeTab].code} lang={sdkExamples[activeTab].lang} />
							</div>
						</div>
					</div>
				</section>

				{/* ── Thin accent line ── */}
				<div className="w-full h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />

				{/* ── Use it with your agent ── */}
				<section className="py-20 md:py-28 relative overflow-hidden">
					<div className="container mx-auto px-4 relative z-10">
						<h2 id="use-it-with-your-agent" className="text-3xl md:text-5xl font-bold tracking-tighter mb-4 scroll-mt-20">
							use it with your agent
						</h2>
						<p className="text-muted-foreground mb-8">
							point your agent to wandler. works with any agent that supports custom OpenAI endpoints
						</p>

						<div className="bg-[#0a0a0a] border border-white/[0.06] overflow-hidden max-w-4xl">
							{/* Agent tab bar */}
							<div className="flex border-b border-white/[0.06] bg-[#080808] overflow-x-auto">
								{([
									{ key: "hermes" as AgentTab, label: "⚕ Hermes" },
								]).map((tab) => (
									<button
										key={tab.key}
										className={`px-4 py-2.5 text-sm font-mono whitespace-nowrap transition-colors ${
											activeAgentTab === tab.key
												? "text-white bg-white/[0.06] border-b-2 border-white"
												: "text-muted-foreground hover:text-white hover:bg-white/[0.02]"
										}`}
										onClick={() => setActiveAgentTab(tab.key)}
									>
										{tab.label}
									</button>
								))}
							</div>

							{/* Tab content */}
							{activeAgentTab === "hermes" && (
								<div className="p-4 md:p-6 space-y-6">
									<div>
										<p className="text-white text-sm mb-3">
											assuming your <code className="font-mono text-primary">wandler</code> server is running
											{" "}
											<code className="font-mono text-primary">LiquidAI/LFM2.5-1.2B-Instruct-ONNX</code>,
											{" "}
											configure Hermes via the CLI like this
										</p>
										<p className="text-white/50 text-xs mb-3">
											replace <code className="font-mono">model.default</code> with the model slug you actually loaded in
											{" "}
											<code className="font-mono">wandler</code>.
										</p>
										<div className="bg-black border border-white/[0.04] overflow-hidden">
											<div className="relative p-4">
												<button
													onClick={() => handleCopy(`hermes config set model.default LiquidAI/LFM2.5-1.2B-Instruct-ONNX\nhermes config set model.provider custom\nhermes config set model.base_url http://localhost:8000/v1\n# if you started wandler with --api-key your-local-key\n# hermes config set model.api_key your-local-key`, "hermes-cli")}
													className="absolute top-3 right-3 p-2 text-white/20 hover:text-white/60 transition-colors z-10"
													title="Copy to clipboard"
												>
													{copied === "hermes-cli" ? <Check className="w-4 h-4 text-primary" /> : <Copy className="w-4 h-4" />}
												</button>
												<SdkCodeBlock code={`hermes config set model.default LiquidAI/LFM2.5-1.2B-Instruct-ONNX
hermes config set model.provider custom
hermes config set model.base_url http://localhost:8000/v1
# if you started wandler with --api-key your-local-key
# hermes config set model.api_key your-local-key`} lang="bash" />
											</div>
										</div>
									</div>

									<div>
										<p className="text-white text-sm mb-3">
											or put the same settings into <code className="font-mono text-primary">~/.hermes/config.yaml</code>
										</p>
										<p className="text-white/50 text-xs mb-3">
											again, replace <code className="font-mono">model.default</code> if your server is running a different model.
										</p>
										<div className="bg-black border border-white/[0.04] overflow-hidden">
											<div className="relative p-4">
												<button
													onClick={() => handleCopy(`model:\n  default: "LiquidAI/LFM2.5-1.2B-Instruct-ONNX"\n  provider: "custom"\n  base_url: "http://localhost:8000/v1"\n  # if you started wandler with --api-key your-local-key\n  # api_key: "your-local-key"`, "hermes")}
													className="absolute top-3 right-3 p-2 text-white/20 hover:text-white/60 transition-colors z-10"
													title="Copy to clipboard"
												>
													{copied === "hermes" ? <Check className="w-4 h-4 text-primary" /> : <Copy className="w-4 h-4" />}
												</button>
												<SdkCodeBlock code={`model:
  default: "LiquidAI/LFM2.5-1.2B-Instruct-ONNX"
  provider: "custom"
  base_url: "http://localhost:8000/v1"
  # if you started wandler with --api-key your-local-key
  # api_key: "your-local-key"`} lang="yaml" />
											</div>
										</div>
									</div>
								</div>
							)}
						</div>
					</div>
				</section>

				{/* ── Thin accent line ── */}
				<div className="w-full h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />

				{/* ── API Reference (always expanded, no collapse) ── */}
				<section className="py-20 md:py-28 relative overflow-hidden">
					<div className="container mx-auto px-4 relative z-10">
						<div className="max-w-4xl">
							<h2 id="api-reference" className="text-3xl md:text-5xl font-bold tracking-tighter mb-12 scroll-mt-20">
								API reference
							</h2>

							{([
								{
									method: "POST", path: "/v1/responses", desc: "Create a model response with streaming, tool calling, and multi-turn input",
									params: [
										{ name: "input", type: "string | array", required: true, desc: "Input text or array of message/function_call/function_call_output items" },
										{ name: "instructions", type: "string", required: false, desc: "System-level instructions (replaces system message)" },
										{ name: "temperature", type: "float", required: false, desc: "Sampling temperature, 0-2. Default 0.7" },
										{ name: "top_p", type: "float", required: false, desc: "Nucleus sampling threshold. Default 0.95" },
										{ name: "max_output_tokens", type: "int", required: false, desc: "Maximum tokens to generate" },
										{ name: "stream", type: "boolean", required: false, desc: "Enable named SSE event streaming. Default false" },
										{ name: "tools", type: "array", required: false, desc: 'Function tools: {type, name, description, parameters}', note: "Flat format — name and parameters are top-level, not nested under a function key." },
										{ name: "tool_choice", type: "string | object", required: false, desc: '"auto", "none", "required", or {type: "function", name: "..."}' },
										{ name: "text", type: "object", required: false, desc: '{"format": {"type": "json_object"}} for JSON mode' },
										{ name: "top_k", type: "int", required: false, desc: "Top-k sampling" },
										{ name: "min_p", type: "float", required: false, desc: "Minimum probability threshold" },
										{ name: "repetition_penalty", type: "float", required: false, desc: "Repetition penalty, > 1.0 to penalize" },
									],
								},
								{
									method: "POST", path: "/v1/chat/completions", desc: "Chat completion with streaming and tool calling",
									params: [
										{ name: "messages", type: "array", required: true, desc: "Input messages with role and content" },
										{ name: "temperature", type: "float", required: false, desc: "Sampling temperature, 0-2. Default 0.7" },
										{ name: "top_p", type: "float", required: false, desc: "Nucleus sampling threshold. Default 0.95" },
										{ name: "max_tokens", type: "int", required: false, desc: "Maximum tokens to generate" },
										{ name: "stream", type: "boolean", required: false, desc: "Enable SSE streaming. Default false" },
										{ name: "stop", type: "string | string[]", required: false, desc: "Stop sequences", note: "Only the final token of each stop string triggers stopping. Multi-token sequences are not matched exactly." },
										{ name: "tools", type: "array", required: false, desc: "Function calling tool definitions", note: "When set, streaming is emulated. The full response is generated first, then re-chunked as SSE." },
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
										{ name: "stop", type: "string | string[]", required: false, desc: "Stop sequences", note: "Only the final token of each stop string triggers stopping. Multi-token sequences are not matched exactly." },
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
									method: "POST", path: "/v1/audio/transcriptions", desc: "Speech-to-text",
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
							] as const).map((ep) => (
								<div key={ep.path} className="border-b border-white/[0.06] mb-8 pb-8 last:border-0 last:mb-0 last:pb-0">
									<div className="flex items-center gap-4 mb-4">
										<span className="font-mono text-[11px] text-white/70 bg-white/[0.08] px-1.5 py-0.5 rounded-sm shrink-0">{ep.method}</span>
										<code className="font-mono text-sm text-white">{ep.path}</code>
										<span className="text-muted-foreground text-sm hidden md:block">{ep.desc}</span>
									</div>
									<p className="text-muted-foreground text-sm mb-4 md:hidden">{ep.desc}</p>

									{ep.params.length > 0 && (
										<div className="pl-4 md:pl-[72px]">
											<div className="text-xs text-muted-foreground uppercase tracking-wider mb-3">Body</div>
											{ep.params.map((p) => (
												<div key={p.name} className="py-3 border-t border-white/[0.04]">
													<div className="flex items-center gap-2 flex-wrap">
														<code className="font-mono text-[13px] text-white font-medium">{p.name}</code>
														<span className="font-mono text-[11px] text-muted-foreground bg-white/[0.06] px-1.5 py-0.5 rounded-sm">{p.type}</span>
														{p.required && <span className="text-[11px] text-primary/70">Required</span>}
													</div>
													<p className="text-muted-foreground text-sm mt-1">{p.desc}</p>
													{"note" in p && p.note && (
														<p className="text-white/40 text-xs italic mt-1.5 border-l-2 border-white/10 pl-3">{p.note}</p>
													)}
												</div>
											))}
										</div>
									)}
								</div>
							))}
						</div>
					</div>
				</section>

				{/* ── Bottom hazard stripe ── */}
				<div className="w-full h-3 bg-[repeating-linear-gradient(45deg,#000,#000_10px,hsl(58_96%_51%)_10px,hsl(58_96%_51%)_20px)] animate-experimental-bg" />
			</main>

			<footer className="px-6 pb-16 pt-12 flex flex-col items-center gap-2 bg-[#050505]">
				<a
					href="https://github.com/runpod-labs/wandler"
					target="_blank"
					rel="noopener noreferrer"
					className="text-xs text-muted-foreground/60 hover:text-muted-foreground transition-colors"
				>
					provided by
				</a>
				<a
					href="https://labs.runpod.io"
					target="_blank"
					rel="noopener noreferrer"
					className="flex items-center gap-2.5 group"
				>
					{/* eslint-disable-next-line @next/next/no-img-element */}
					<img
						src="/assets/runpod-icon.svg"
						alt=""
						width={20}
						height={21}
						className="opacity-70 group-hover:opacity-100 transition-opacity"
					/>
					{/* eslint-disable-next-line @next/next/no-img-element */}
					<img
						src="/assets/runpod-wordmark-text.webp"
						alt="RunPod"
						width={90}
						height={20}
						className="opacity-70 group-hover:opacity-100 transition-opacity"
					/>
					<span
						className="italic text-base text-[#a78bfa] opacity-70 group-hover:opacity-100 transition-opacity"
						style={{ fontFamily: "var(--font-display)" }}
					>
						Labs
					</span>
				</a>
			</footer>
		</div>
	);
}
