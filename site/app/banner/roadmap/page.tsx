import Image from "next/image";

export default function RoadmapBanner() {
	return (
		<div className="w-full min-h-screen bg-neutral-800 flex items-center justify-center p-8">
			{/* Fixed size banner container */}
			<div className="w-[1200px] h-[630px] relative bg-black border-[30px] border-[#ffff00]">
				{/* Background noise */}
				<div className="absolute inset-0 z-0">
					<div className="absolute inset-0 bg-noise opacity-20"></div>
				</div>

				{/* Content */}
				<div className="absolute inset-0 p-8 z-10">
					{/* Title */}
					<h1 className="text-3xl text-left font-black mb-1 text-[#ffff00] tracking-tight uppercase">
						roadmap
					</h1>
					<h2 className="text-8xl text-left font-black mb-8 text-[#00ffff] tracking-tight uppercase">
						e2e tests
					</h2>

					{/* Test Results Grouped */}
					<div className="grid grid-cols-3 gap-x-8">
						{/* loadModel Group */}
						<div className="space-y-4">
							<h2 className="text-3xl text-[#00ffff] font-bold">loadModel</h2>
							<div className="flex items-center gap-3 text-[#00ffff]">
								<div className="w-4 h-4 bg-[#00ffff] rotate-45" />
								<span className="text-xl">loads model and shows capabilities</span>
							</div>
							<div className="flex items-center gap-3 text-[#00ffff]">
								<div className="w-4 h-4 bg-[#00ffff] rotate-45" />
								<span className="text-xl">handles invalid model paths</span>
							</div>
							<div className="flex items-center gap-3 text-[#00ffff]">
								<div className="w-4 h-4 bg-[#00ffff] rotate-45" />
								<span className="text-xl">loads model in main thread</span>
							</div>
						</div>

						{/* generateText Group */}
						<div className="space-y-4">
							<h2 className="text-3xl text-[#00ffff] font-bold">generateText</h2>
							<div className="flex items-center gap-3 text-[#00ffff]">
								<div className="w-4 h-4 bg-[#00ffff] rotate-45" />
								<span className="text-xl">generates text from prompt</span>
							</div>
							<div className="flex items-center gap-3 text-[#00ffff]">
								<div className="w-4 h-4 bg-[#00ffff] rotate-45" />
								<span className="text-xl">can abort generation</span>
							</div>
							<div className="flex items-center gap-3 text-[#00ffff]">
								<div className="w-4 h-4 bg-[#00ffff] rotate-45" />
								<span className="text-xl">can generate text in main thread</span>
							</div>
							<div className="flex items-center gap-3 text-[#00ffff]">
								<div className="w-4 h-4 bg-[#00ffff] rotate-45" />
								<span className="text-xl">fails with incompatible dtype</span>
							</div>
						</div>

						{/* streamText Group */}
						<div className="space-y-4">
							<h2 className="text-3xl text-[#00ffff] font-bold">streamText</h2>
							<div className="flex items-center gap-3 text-[#00ffff]">
								<div className="w-4 h-4 bg-[#00ffff] rotate-45" />
								<span className="text-xl">streams text from prompt</span>
							</div>
							<div className="flex items-center gap-3 text-[#00ffff]">
								<div className="w-4 h-4 bg-[#00ffff] rotate-45" />
								<span className="text-xl">can abort streaming</span>
							</div>
							<div className="flex items-center gap-3 text-[#00ffff]">
								<div className="w-4 h-4 bg-[#00ffff] rotate-45" />
								<span className="text-xl">fails with incompatible dtype</span>
							</div>
						</div>
					</div>

					{/* Logo in bottom right */}
					<div className="absolute bottom-12 right-12">
						<Image
							src="https://5xvkmufwzznj1ey2.public.blob.vercel-storage.com/wandler_logo_v5-vJ2L3NmauebkFJs9fOcFe7bPVM14To.svg"
							alt="Wandler Logo"
							width={200}
							height={50}
							className="brightness-200"
							priority
						/>
					</div>
				</div>
			</div>
		</div>
	);
}
