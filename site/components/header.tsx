"use client";

import { Menu, X } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

// Import version from utility file
import { PACKAGE_VERSION } from "@/lib/version";

// Use the version from the utility
const VERSION = PACKAGE_VERSION;

export function Header() {
	const [isMenuOpen, setIsMenuOpen] = useState(false);
	const pathname = usePathname();

	const isDocsActive = pathname?.startsWith("/docs");
	const isHubActive = pathname?.startsWith("/hub");

	return (
		<header
			className="fixed w-full z-50 border-b border-primary/20 backdrop-filter backdrop-blur-[40px]"
			style={{ backgroundColor: "#14171203" }}
		>
			<div className="container mx-auto flex justify-between items-center h-16 px-4">
				<div className="flex items-center gap-4">
					<Link href="/">
						<Image
							src="https://5xvkmufwzznj1ey2.public.blob.vercel-storage.com/wandler_logo_v5-vJ2L3NmauebkFJs9fOcFe7bPVM14To.svg"
							alt="Wandler Logo"
							width={150}
							height={50}
							className="brightness-100"
						/>
					</Link>
					<div className="hidden sm:block relative group">
						<Link
							href="https://www.npmjs.com/package/wandler"
							target="_blank"
							rel="noopener noreferrer"
							className="px-2 py-1 text-xs font-mono bg-primary/5 text-primary/50 border border-primary/20 rounded-md transition-all duration-300 group-hover:bg-primary/10 group-hover:text-primary group-hover:border-primary/50"
						>
							v{VERSION}
						</Link>
					</div>
				</div>

				{/* Mobile menu button */}
				<button
					className="md:hidden p-2 text-white hover:text-primary transition-colors"
					onClick={() => setIsMenuOpen(!isMenuOpen)}
					aria-label="Toggle menu"
				>
					{isMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
				</button>

				{/* Desktop navigation */}
				<nav className="hidden md:block">
					<div className="cyberpunk-nav">
						{/* Hub link - first in navigation */}
						<Link
							href="/hub"
							className={`cyberpunk-nav-item ${
								isHubActive ? "bg-primary/20 text-primary" : "text-white"
							}`}
						>
							Hub
						</Link>
						<Link
							href="/docs"
							className={`cyberpunk-nav-item ${
								isDocsActive ? "bg-primary/20 text-primary" : "text-white"
							}`}
						>
							Docs
						</Link>
						<Link
							href="https://github.com/wandlerai"
							className="cyberpunk-nav-item flex items-center"
							target="_blank"
							rel="noopener noreferrer"
						>
							<Image
								src="/github.svg"
								alt="GitHub"
								width={16}
								height={16}
								className="text-current invert"
							/>
						</Link>
						<Link
							href="https://bsky.app/profile/wandler.ai"
							className="cyberpunk-nav-item flex items-center"
							target="_blank"
							rel="noopener noreferrer"
						>
							<Image
								src="/bluesky.svg"
								alt="Bluesky"
								width={16}
								height={16}
								className="text-current invert"
							/>
						</Link>
						<Link
							href="https://x.com/wandler_______"
							className="cyberpunk-nav-item flex items-center"
							target="_blank"
							rel="noopener noreferrer"
						>
							<Image
								src="/x.svg"
								alt="X (Twitter)"
								width={16}
								height={16}
								className="text-current"
							/>
						</Link>
					</div>
				</nav>

				{/* Mobile navigation */}
				{isMenuOpen && (
					<div className="absolute top-full left-0 right-0 bg-black/95 border-b border-primary/20 backdrop-blur-md md:hidden">
						<nav className="container mx-auto py-4 px-4">
							{/* Social icons row */}
							<div className="flex justify-center gap-6">
								{/* Hub link - first in mobile navigation */}
								<Link
									href="/hub"
									className={`p-2 ${isHubActive ? "bg-primary/20 text-primary" : "text-white"}`}
								>
									Hub
								</Link>
								<Link
									href="/docs"
									className={`p-2 ${isDocsActive ? "bg-primary/20 text-primary" : "text-white"}`}
								>
									Docs
								</Link>
								<Link
									href="https://github.com/wandlerai"
									className="p-2 text-white hover:text-primary transition-colors"
									target="_blank"
									rel="noopener noreferrer"
								>
									<Image
										src="/github.svg"
										alt="GitHub"
										width={20}
										height={20}
										className="text-current invert"
									/>
								</Link>
								<Link
									href="https://bsky.app/profile/wandler.ai"
									className="p-2 text-white hover:text-primary transition-colors"
									target="_blank"
									rel="noopener noreferrer"
								>
									<Image
										src="/bluesky.svg"
										alt="Bluesky"
										width={20}
										height={20}
										className="text-current invert"
									/>
								</Link>
								<Link
									href="https://x.com/wandler_______"
									className="p-2 text-white hover:text-primary transition-colors"
									target="_blank"
									rel="noopener noreferrer"
								>
									<Image
										src="/x.svg"
										alt="X (Twitter)"
										width={20}
										height={20}
										className="text-current"
									/>
								</Link>
							</div>
						</nav>
					</div>
				)}
			</div>
		</header>
	);
}
