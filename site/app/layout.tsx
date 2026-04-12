import "./globals.css";

import type { Metadata } from "next";
import { Space_Grotesk } from "next/font/google";
import type React from "react";

import { ThemeProvider } from "@/components/theme-provider";
import { Toaster } from "@/components/ui/toaster";

const spaceGrotesk = Space_Grotesk({ subsets: ["latin"] });

export const metadata: Metadata = {
	title: "wandler",
	description:
		"run ai in your browser. inspired by the ▲ AI SDK and build on top of 🤗 transformers.js",

	openGraph: {
		images: [
			"https://5xvkmufwzznj1ey2.public.blob.vercel-storage.com/wandler_banner_v1-DmmG40A1TwkpZiQiBxqsmHyXwrmSSS.jpg",
		],
		url: "https://wandler.ai",
	},
	twitter: {
		card: "summary_large_image",
		title: "wandler",
		description: "run ai in your browser",
		siteId: "",
		creator: "@wandler________",
		creatorId: "",
		images: [
			"https://5xvkmufwzznj1ey2.public.blob.vercel-storage.com/wandler_banner_v1-DmmG40A1TwkpZiQiBxqsmHyXwrmSSS.jpg",
		],
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
