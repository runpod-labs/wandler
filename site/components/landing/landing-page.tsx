"use client";

import { Check, Code, Copy, Lock, PiggyBank, Server } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { useState } from "react";

import { Header } from "@/components/header";
import { HubSection } from "@/components/landing/hub-section";

import { DemocratizeBanner } from "./democratize-banner";
import { ExperimentalBanner } from "./experimental-banner";
import { RoadmapSection } from "./roadmap-section";

export function LandingPage() {
	type PackageManager = "npm" | "yarn" | "pnpm";
	const [activeTab, setActiveTab] = useState<PackageManager>("npm");
	const [copied, setCopied] = useState(false);

	const installCommands: Record<PackageManager, string> = {
		npm: "npm i wandler",
		yarn: "yarn add wandler",
		pnpm: "pnpm add wandler",
	};

	const handleCopy = async () => {
		await navigator.clipboard.writeText(installCommands[activeTab]);
		setCopied(true);
		setTimeout(() => setCopied(false), 2000);
	};

	return (
		<div className="flex flex-col min-h-screen bg-black text-white">
			<Header />

			<main className="flex-grow pt-16">
				<section className="min-h-screen relative overflow-hidden flex flex-col justify-center space-y-12 px-4 md:px-0 my-12 md:my-4">
					<div className="relative z-10">
						<div className="container mx-auto">
							<div className="grid md:grid-cols-[3fr_1fr] gap-8 md:gap-12 items-center w-full">
								<div className="space-y-10 md:space-y-6">
									<h1 className="text-5xl md:text-8xl font-bold tracking-tighter mb-4">
										run ai in your browser
									</h1>
									<p className="text-lg md:text-xl text-muted-foreground mb-4">
										inspired by the ▲{" "}
										<a href="https://sdk.vercel.ai" className="text-primary hover:underline">
											AI SDK
										</a>{" "}
										& built on top of 🤗{" "}
										<a
											href="https://huggingface.co/docs/transformers.js/en/index"
											className="text-primary hover:underline"
										>
											transformers.js
										</a>
									</p>

									<div className="flex flex-wrap items-center gap-3 md:gap-4 text-sm text-muted-foreground">
										<div className="flex items-center gap-2">
											<PiggyBank className="w-4 h-4 text-primary" />
											free
										</div>
										<div className="flex items-center gap-2">
											<Lock className="w-4 h-4 text-primary" />
											private
										</div>
										<div className="flex items-center gap-2">
											<Server className="w-4 h-4 text-primary" />
											local
										</div>
										<div className="flex items-center gap-2">
											<Code className="w-4 h-4 text-primary" />
											open source
										</div>
									</div>

									<div className="cyberpunk-corners bg-secondary p-3 md:p-4 w-full md:w-fit">
										<div className="install-tabs flex gap-1 md:gap-2 mb-3">
											{Object.keys(installCommands).map(tab => (
												<div
													key={tab}
													className={`install-tab cursor-pointer px-2 md:px-3 py-1 rounded ${
														activeTab === tab ? "bg-primary/20 text-primary" : "hover:bg-primary/10"
													}`}
													onClick={() => setActiveTab(tab as PackageManager)}
												>
													{tab}
												</div>
											))}
										</div>
										<div className="install-content flex items-center justify-between">
											<div className="flex items-center gap-2">
												<button
													onClick={handleCopy}
													className="p-1 md:p-2 hover:text-primary transition-colors shrink-0"
													title="Copy to clipboard"
												>
													{copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
												</button>
												<code className="font-mono w-[180px]">{installCommands[activeTab]}</code>
											</div>
											<Link
												href="/llms.txt"
												className="group relative px-2 md:px-3 py-1 md:py-2 text-[#0ff] rounded shrink-0 ml-4 md:ml-8"
											>
												<span className="relative z-10">llms.txt</span>
												<div className="pointer-events-none absolute -inset-[1px] opacity-0 group-hover:opacity-100 animate-border"></div>
											</Link>
										</div>
									</div>
								</div>

								<div className="relative flex justify-center mt-8 md:mt-0">
									<div className="w-[280px] h-[280px] md:w-[500px] md:h-[500px] relative">
										<Image
											src="https://5xvkmufwzznj1ey2.public.blob.vercel-storage.com/20250202_wandler_head_v2-Ma4f25yqpXRSf32ZnmivUnnV1LGQ69.jpg"
											alt="AI and Human Integration"
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
					<ExperimentalBanner className="rotate-2" />
				</section>

				<HubSection />

				<RoadmapSection />

				<DemocratizeBanner />
			</main>

			<footer className="py-6 md:py-8 border-t border-primary/20">
				<div className="container mx-auto px-4 text-center text-muted-foreground text-sm md:text-base">
					<p>generated 2025 by wandler</p>
				</div>
			</footer>
		</div>
	);
}
