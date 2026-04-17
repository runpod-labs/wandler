import "./globals.css";

import type { Metadata } from "next";
import { Space_Grotesk } from "next/font/google";
import type React from "react";

import { ThemeProvider } from "@/components/theme-provider";
import { Toaster } from "@/components/ui/toaster";

const spaceGrotesk = Space_Grotesk({ subsets: ["latin"] });

const TITLE = "wandler by Runpod Labs";
const DESCRIPTION =
	"transformers.js inference server, OpenAI-compatible API, works on Mac, Linux & Windows";

export const metadata: Metadata = {
	title: TITLE,
	description: DESCRIPTION,
	metadataBase: new URL("https://wandler.ai"),
	openGraph: {
		title: TITLE,
		description: DESCRIPTION,
		url: "https://wandler.ai",
		siteName: "wandler",
		type: "website",
		images: [
			{
				url: "/og.jpg",
				width: 1200,
				height: 630,
				alt: TITLE,
			},
		],
	},
	twitter: {
		card: "summary_large_image",
		title: TITLE,
		description: DESCRIPTION,
		images: ["/og.jpg"],
		creator: "@wandler________",
	},
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
	return (
		<html lang="en" suppressHydrationWarning>
			<head />
			<body className={`${spaceGrotesk.className} bg-black text-white`}>
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
