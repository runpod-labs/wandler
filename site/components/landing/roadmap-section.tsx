import { Check } from "lucide-react";

import { cn } from "@/lib/utils";

interface RoadmapItem {
	title: string;
	status: "done" | "current" | "upcoming";
	description?: string;
}

interface RoadmapSectionProps {
	className?: string;
}

const roadmapItems: RoadmapItem[] = [
	{
		title: "loadModel",
		status: "done",
		description: "load any onnx model directly in your browser",
	},
	{
		title: "generateText",
		status: "done",
		description: "generate text with any llm",
	},
	{
		title: "streamText",
		status: "done",
		description: "stream text with any llm",
	},
	{
		title: "webworker",
		status: "done",
		description: "run models in a separate thread for better performance",
	},
	{
		title: "e2e tests",
		status: "done",
		description: "comprehensive testing suite for all features",
	},
	{
		title: "useChat",
		status: "done",
		description: "react hooks for chat interfaces",
	},
	{
		title: "chat demo",
		status: "done",
		description: "simple chat ui leveraging useChat",
	},
	{
		title: "auto-generated docs",
		status: "current",
		description: "workflow that generates the docs based on the lib",
	},
	{
		title: "hub",
		status: "current",
		description: "browse compatible models and try them out",
	},
	{
		title: "generateImage",
		status: "upcoming",
		description: "create an image using a prompt",
	},
	{
		title: "generateAudio",
		status: "upcoming",
		description: "text to speech and speech to text",
	},
];

export function RoadmapSection({ className }: RoadmapSectionProps) {
	return (
		<section className={cn("relative bg-black w-full", className)}>
			<style jsx>{`
				.line-pattern {
					position: absolute;
					top: 0;
					left: 0;
					right: 0;
					z-index: 10;
					height: 45px;
					opacity: 0.6;
					background:
						repeating-linear-gradient(
							45deg,
							#ffec1999,
							#ffec1999 15px,
							transparent 15px,
							transparent 30px
						),
						repeating-linear-gradient(
							0deg,
							#ffec1999,
							#ffec1999 15px,
							transparent 15px,
							transparent 30px
						);
				}
			`}</style>
			<div className="absolute inset-0 bg-black">
				<div className="absolute inset-0 bg-noise"></div>
			</div>
			{/* Top line pattern */}
			<div className="line-pattern"></div>

			<div className="h-[15vh] md:h-[50vh] w-full relative z-10 flex items-end border-l-[50px] md:border-l-[300px] border-primary/40 border-dash-dense">
				<div className="w-full">
					<h2 className="text-[12vw] md:text-[10vw] text-left font-black m-0 text-primary tracking-tight uppercase relative container leading-[0.75] bg-transparent pl-4 md:pl-0">
						roadmap
					</h2>
				</div>
			</div>

			<div className="min-h-[70vh] w-full relative z-10 border-l-[50px] md:border-l-[300px] border-secondary/40">
				<div className="py-4 md:py-8 px-2 md:px-4">
					<div className="grid grid-cols-1 md:grid-cols-[repeat(auto-fit,minmax(300px,1fr))] gap-4 md:gap-12">
						{roadmapItems.map((item, index) => (
							<div
								key={index}
								className={cn(
									"relative p-4 md:p-6 border-2",
									item.status === "done" && "border-[#0ff] text-[#0ff]/60",
									item.status === "current" &&
										"border-primary bg-primary/20 text-white animate-border-pulse",
									item.status === "upcoming" && "border-white/10 text-white/40"
								)}
							>
								<div className="flex items-center gap-3 md:gap-4">
									<div
										className={cn(
											"w-3 h-3 md:w-4 md:h-4 rounded-full flex items-center justify-center",
											item.status === "done" && "bg-[#0ff]",
											item.status === "current" && "bg-primary",
											item.status === "upcoming" && "bg-black border border-white/10"
										)}
									>
										{item.status === "done" && (
											<Check className="w-2 h-2 md:w-3 md:h-3 text-black" />
										)}
										{item.status === "current" && (
											<div className="w-1.5 h-1.5 md:w-2 md:h-2 rounded-full bg-white" />
										)}
									</div>
									<h3
										className={cn(
											"text-base md:text-lg font-medium",
											item.status === "done" && "text-[#0ff]",
											item.status === "current" && "text-white",
											item.status === "upcoming" && "text-white/40"
										)}
									>
										{item.title}
									</h3>
								</div>
								<p
									className={cn(
										"mt-2 text-sm md:text-base",
										item.status === "done" && "text-[#0ff]/60",
										item.status === "current" && "text-white/80",
										item.status === "upcoming" && "text-white/40"
									)}
								>
									{item.description}
								</p>
							</div>
						))}
					</div>
				</div>
			</div>
		</section>
	);
}
