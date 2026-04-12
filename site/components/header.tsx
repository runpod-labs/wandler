"use client";

import Image from "next/image";
import Link from "next/link";
import { useEffect, useState } from "react";

export function Header() {
	const [scrolled, setScrolled] = useState(false);

	useEffect(() => {
		const onScroll = () => setScrolled(window.scrollY > 200);
		window.addEventListener("scroll", onScroll, { passive: true });
		return () => window.removeEventListener("scroll", onScroll);
	}, []);

	return (
		<header className="fixed w-full z-50 border-b border-primary/10 backdrop-filter backdrop-blur-[40px] bg-black/60">
			<div className="container mx-auto flex justify-between items-center h-14 px-4">
				<Link href="/" className={`transition-opacity duration-300 ${scrolled ? "opacity-100" : "opacity-0 pointer-events-none"}`}>
					<Image
						src="https://5xvkmufwzznj1ey2.public.blob.vercel-storage.com/wandler_logo_v5-vJ2L3NmauebkFJs9fOcFe7bPVM14To.svg"
						alt="wandler"
						width={120}
						height={40}
						className="brightness-100"
					/>
				</Link>

				<Link
					href="https://github.com/runpod-labs/wandler"
					className="flex items-center gap-2 text-sm text-muted-foreground hover:text-primary transition-colors"
					target="_blank"
					rel="noopener noreferrer"
				>
					<Image
						src="/github.svg"
						alt="GitHub"
						width={18}
						height={18}
						className="invert opacity-70"
					/>
					<span className="hidden sm:inline font-mono">github</span>
				</Link>
			</div>
		</header>
	);
}
