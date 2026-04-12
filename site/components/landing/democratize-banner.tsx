import { cn } from "@/lib/utils";

interface DemocratizeBannerProps {
	className?: string;
}

export function DemocratizeBanner({ className }: DemocratizeBannerProps) {
	return (
		<section className={cn("h-screen relative overflow-hidden", className)}>
			{/* Screen content with CRT effects */}
			<div className="absolute inset-0 bg-[#e41832] flex items-center justify-center">
				{/* Base scan lines pattern */}
				<div className="absolute inset-0 overflow-hidden">
					<div className="absolute inset-0 h-[200%] animate-[scanlines_4s_linear_infinite]">
						<div className="absolute inset-0 bg-[repeating-linear-gradient(0deg,rgba(0,0,0,0.15),rgba(0,0,0,0.15)_2px,transparent_2px,transparent_4px)]"></div>
						<div className="absolute inset-0 translate-y-[100%] bg-[repeating-linear-gradient(0deg,rgba(0,0,0,0.15),rgba(0,0,0,0.15)_2px,transparent_2px,transparent_4px)]"></div>
					</div>
				</div>

				{/* Screen glow */}
				<div className="absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_10%,rgba(0,0,0,0.4)_100%)]"></div>

				{/* Color bands */}
				<div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(255,0,0,0.1),rgba(0,255,0,0.1),rgba(0,0,255,0.1))] mix-blend-overlay"></div>

				{/* Moving scanlines - bright lines */}
				<div className="absolute inset-0">
					<div className="absolute inset-0 animate-[scanline_4s_linear_infinite]">
						<div className="w-full h-[2px] bg-[rgba(255,255,255,0.1)]"></div>
					</div>
				</div>

				{/* Flicker animation */}
				<div className="absolute inset-0 opacity-[0.03] animate-[flicker_0.15s_infinite_alternate]"></div>

				{/* Noise overlay */}
				<div className="absolute inset-0 bg-noise opacity-[0.15] mix-blend-overlay"></div>

				{/* Main content */}
				<h1
					data-text="DEMOCRATIZE AI"
					className="crt-text text-[8vw] md:text-[13vw] font-black text-white leading-none tracking-tighter text-center relative mix-blend-screen px-4"
				>
					DEMOCRATIZE AI
				</h1>
			</div>
		</section>
	);
}
