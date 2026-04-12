"use client";

import { ArrowRight, Cpu, MessageSquare } from "lucide-react";
import Link from "next/link";

export function HubSection() {
	return (
		<section
			id="hub"
			className="min-h-screen min-w-full relative overflow-hidden flex flex-col justify-center bg-black/90 border-t border-primary/20"
		>
			<div className="absolute inset-0 bg-grid-pattern opacity-10"></div>

			<div className="container mx-auto px-4 py-16 relative z-10">
				<div className="text-center mb-12">
					<h2 className="text-4xl md:text-6xl font-bold tracking-tighter mb-4">
						try <span className="text-primary">wandler</span> in your browser
					</h2>
					<p className="text-xl text-muted-foreground max-w-3xl mx-auto">
						Experience AI running directly in your browser with our interactive demos and compatible
						models.
					</p>
				</div>

				<div className="grid md:grid-cols-2 gap-8 max-w-6xl mx-auto">
					<div className="bg-black/60 backdrop-blur-sm border border-primary/20 rounded-lg p-6 flex flex-col h-full">
						<div className="mb-6">
							<div className="bg-primary/10 p-3 rounded-full w-fit mb-4">
								<MessageSquare className="h-8 w-8 text-primary" />
							</div>
							<h3 className="text-2xl font-bold mb-2">Chat Demos</h3>
							<p className="text-muted-foreground">
								Chat with powerful language models running entirely in your browser. No server
								required.
							</p>
						</div>
						<div className="mt-auto">
							<div className="rounded-md mb-4 w-full h-48 bg-gradient-to-r from-primary/20 to-primary/5 flex items-center justify-center">
								<MessageSquare className="h-12 w-12 text-primary/40" />
							</div>
							<Link
								href="/hub/chat"
								className="flex items-center text-primary hover:underline group"
							>
								Try the chat demo
								<ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
							</Link>
						</div>
					</div>

					<div className="bg-black/60 backdrop-blur-sm border border-primary/20 rounded-lg p-6 flex flex-col h-full">
						<div className="mb-6">
							<div className="bg-primary/10 p-3 rounded-full w-fit mb-4">
								<Cpu className="h-8 w-8 text-primary" />
							</div>
							<h3 className="text-2xl font-bold mb-2">Browser-Compatible Models</h3>
							<p className="text-muted-foreground">
								Explore our growing collection of AI models optimized to run efficiently in the
								browser.
							</p>
						</div>
						<div className="mt-auto">
							<div className="grid grid-cols-2 gap-4 mb-4">
								<div className="bg-black/80 border border-primary/10 rounded-md p-4">
									<h4 className="font-medium mb-1">DeepSeek-R1-Distill-Qwen-1.5B</h4>
									<p className="text-sm text-muted-foreground">
										Compact but powerful language model
									</p>
								</div>
								<div className="bg-black/80 border border-primary/10 rounded-md p-4">
									<h4 className="font-medium mb-1">Qwen2.5-Coder-0.5B</h4>
									<p className="text-sm text-muted-foreground">Specialized for coding assistance</p>
								</div>
							</div>
							<Link href="/hub" className="flex items-center text-primary hover:underline group">
								Explore the hub
								<ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
							</Link>
						</div>
					</div>
				</div>

				<div className="text-center mt-12">
					<Link
						href="/hub"
						className="inline-flex items-center justify-center px-6 py-3 bg-primary text-black font-medium rounded-md hover:bg-primary/90 transition-colors"
					>
						Visit the Wandler Hub
						<ArrowRight className="ml-2 h-4 w-4" />
					</Link>
				</div>
			</div>
		</section>
	);
}
