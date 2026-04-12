"use client";

import { Header } from "@/components/header";

export default function HubLayout({ children }: { children: React.ReactNode }) {
	return (
		<div className="flex flex-col min-h-screen bg-black text-white">
			<Header />

			<div className="flex flex-grow pt-16">
				{/* Main Content - Full width on all pages */}
				<main className="flex-grow ml-0">
					<div className="container mx-auto px-4 py-8 relative z-10">{children}</div>
				</main>
			</div>

			<footer className="py-6 md:py-8 border-t border-primary/20">
				<div className="container mx-auto px-4 text-center text-muted-foreground text-sm md:text-base">
					<p>generated 2025 by wandler</p>
				</div>
			</footer>
		</div>
	);
}
