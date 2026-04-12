import Image from "next/image";

export default function BannerPage() {
	return (
		<div className="w-full min-h-screen bg-neutral-800 flex items-center justify-center p-8">
			{/* Fixed size banner container */}
			<div className="w-[1200px] h-[630px] relative bg-black">
				<div className="absolute inset-0 flex flex-col">
					{/* Content Grid */}
					<div className="flex-1 grid grid-cols-[2fr_1fr] gap-1 items-center px-12">
						<div className="space-y-6">
							{/* Logo */}
							<div className="flex justify-center mb-4">
								<Image
									src="https://5xvkmufwzznj1ey2.public.blob.vercel-storage.com/wandler_logo_v5-vJ2L3NmauebkFJs9fOcFe7bPVM14To.svg"
									alt="Wandler Logo"
									width={400}
									height={100}
									className="brightness-100"
									priority
								/>
							</div>

							<h1 className="text-7xl font-bold tracking-tighter mb-4">run ai in your browser</h1>
							<p className="text-xl text-muted-foreground">
								inspired by the ▲ <span className="text-primary">AI SDK</span> & built on top of 🤗{" "}
								<span className="text-primary">transformers.js</span>
							</p>
						</div>

						<div className="relative flex justify-center">
							<div className="w-[400px] h-[400px] relative">
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
		</div>
	);
}
