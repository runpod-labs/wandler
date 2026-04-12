import { cn } from "@/lib/utils";

interface ExperimentalBannerProps {
	className?: string;
}

export function ExperimentalBanner({ className }: ExperimentalBannerProps) {
	return (
		<div className={cn("w-full bg-black relative overflow-hidden flex flex-col py-2", className)}>
			{/* Top line pattern */}
			<div className="w-full h-4 bg-[repeating-linear-gradient(45deg,#000,#000_10px,#e41832_10px,#e41832_20px)] animate-experimental-bg"></div>
			<div className="py-4 text-center flex-1 flex flex-col justify-start">
				<h2 className="text-[#e41832] text-3xl md:text-5xl font-black mb-2 tracking-tight">
					EXPERIMENTAL
				</h2>
				<p className="text-lg">this is an alpha release, expect everything to be totally broken</p>
			</div>
			{/* Bottom line pattern */}
			<div className="w-full h-4 bg-[repeating-linear-gradient(45deg,#000,#000_10px,#e41832_10px,#e41832_20px)] animate-experimental-bg-reverse"></div>
		</div>
	);
}
