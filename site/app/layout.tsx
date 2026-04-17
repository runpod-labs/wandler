import "./globals.css";

import type { Metadata } from "next";
import { Instrument_Serif, Space_Grotesk } from "next/font/google";
import type React from "react";

import { ThemeProvider } from "@/components/theme-provider";
import { Toaster } from "@/components/ui/toaster";

const spaceGrotesk = Space_Grotesk({ subsets: ["latin"] });
const instrumentSerif = Instrument_Serif({
	weight: "400",
	subsets: ["latin"],
	variable: "--font-display",
});

export const metadata: Metadata = {
	title: "wandler — transformers.js inference server",
	description:
		"Run ONNX models locally with WebGPU acceleration. Drop-in replacement for OpenAI API. No Python, no CUDA — just npx wandler.",
	metadataBase: new URL("https://wandler.ai"),
	openGraph: {
		title: "wandler — transformers.js inference server",
		description:
			"OpenAI-compatible. WebGPU accelerated. Zero config. Run ONNX models locally with npx wandler.",
		url: "https://wandler.ai",
		siteName: "wandler",
		type: "website",
	},
	twitter: {
		card: "summary_large_image",
		title: "wandler — transformers.js inference server",
		description:
			"OpenAI-compatible. WebGPU accelerated. Zero config. Run ONNX models locally with npx wandler.",
		creator: "@wandler________",
	},
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
	return (
		<html lang="en" suppressHydrationWarning>
			<head />
			<body className={`${spaceGrotesk.className} ${instrumentSerif.variable} bg-black text-white`}>
				<ThemeProvider
					attribute="class"
					defaultTheme="dark"
					enableSystem={false}
					disableTransitionOnChange
				>
					{children}
					<Toaster />
				</ThemeProvider>
			</body>
		</html>
	);
}
